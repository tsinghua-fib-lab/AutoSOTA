import torch, os, hydra, logging
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
from torchvision import transforms
from monai.metrics import PSNRMetric, SSIMMetric
from taming.modules.losses.lpips import LPIPS

from src.utils import set_all_seed
from src.tasks import get_operator, get_noise
from src.models import get_model
from src.samplers import get_sampler


@hydra.main(version_base=None, config_path="configs", config_name="default")
def posterior_sample(cfg):
    set_all_seed(cfg.seed)

    # load configurations
    data_config = cfg.data
    task_config = cfg.task
    model_config = cfg.model
    sampler_config = cfg.sampler

    # device setting
    device_str = f"cuda:{cfg.gpu}" if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)

    # prepare dataloader
    transform = transforms.Compose([
        transforms.Normalize((0.5), (0.5))
    ])
    inv_transform = transforms.Compose([
        transforms.Normalize((-1), (2)),
        transforms.Lambda(lambda x: x.clamp(0, 1).detach())
    ])
    dataset = np.load(data_config.root)
    dataset = torch.from_numpy(dataset).float().permute(0,3,1,2) / 255.0
    num_test_images = len(dataset)

    # load masks
    mask = np.load(f'./data/{data_config.mask_name}.npy')
    mask = torch.tensor(mask).float().permute(0, 3, 1, 2)
    
    # prepare task (forward model and noise)
    operator = get_operator(**task_config.operator, mask=None, device=device)
    noiser = get_noise(**task_config.noise)

    # load model
    model = get_model(**model_config)
    model = model.to(device)
    model.eval()

    # load sampler
    sampler = get_sampler(
        sampler_config,
        model=model, operator=operator, noiser=noiser, device=device)

    # working directory
    exp_name = '_'.join([
        f'{data_config.name}_{data_config.mask_name}',
        # operator.display_name,
        noiser.display_name,
        sampler.display_name
    ])
    exp_name += '' if len(cfg.add_exp_name) == 0 else '_' + cfg.add_exp_name
    logger = logging.getLogger(exp_name)
    out_path = os.path.join("results", exp_name)
    if os.path.exists(out_path):
        from datetime import datetime
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_name += f'_{current_time}'
        out_path = os.path.join("results", exp_name)
        logger = logging.getLogger(exp_name)
    os.makedirs(out_path, exist_ok=True)
    for img_dir in ['gt', 'meas', 'recon', 'progress']:
        os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)

    # inference
    meta_log_x = defaultdict(list)
    meta_log_x["statistics_based_on_one_sample"] = defaultdict(list)
    meta_log_x["statistics_based_on_mean"] = defaultdict(list)
    meta_log_z = defaultdict(list)
    meta_log_z["statistics_based_on_one_sample"] = defaultdict(list)
    meta_log_z["statistics_based_on_mean"] = defaultdict(list)
    meta_log = {'x': meta_log_x, 'z': meta_log_z}
    metrics = {
        'psnr': PSNRMetric(max_val=1),
        'ssim': SSIMMetric(spatial_dims=2),
        'lpips': LPIPS().to(device).eval(),
    }
    for i in range(num_test_images):
        logger.info(f"Inference for image {i} on device {device_str}")
        file_idx = f"{i:05d}"

        ref_img = dataset[i].to(device)
        ref_img = transform(dataset[i]).to(device).unsqueeze(0)
        ref_mask = mask[i].to(device)
        operator.update_mask(ref_mask)
        cmap = 'gray' if ref_img.shape[1] == 1 else None

        y_n = noiser(operator.forward(ref_img))

        # logging
        log_x = defaultdict(list)
        log_x["consistency_gt"] = torch.norm(operator.forward(ref_img) - y_n).item()
        log_x["gt"] = inv_transform(ref_img).permute(0, 2, 3, 1).squeeze().cpu().numpy()
        plt.imsave(os.path.join(out_path, 'gt', file_idx+'.png'), log_x["gt"], cmap=cmap)
        y_masked = operator.transpose(y_n)
        log_x["meas"] = inv_transform(y_masked).permute(0, 2, 3, 1).squeeze().cpu().numpy()
        plt.imsave(os.path.join(out_path, 'meas', file_idx+'.png'), log_x["meas"], cmap=cmap)

        log_z = defaultdict(list)
        log_z["consistency_gt"] = torch.norm(operator.forward(ref_img) - y_n).item()
        log_z["gt"] = inv_transform(ref_img).permute(0, 2, 3, 1).squeeze().cpu().numpy()
        log_z["meas"] = inv_transform(y_masked).permute(0, 2, 3, 1).squeeze().cpu().numpy()

        log = {'x': log_x, 'z': log_z}

        # sampling
        for j in tqdm(range(cfg.num_runs)):
            x_samples, z_samples = sampler(
                gt=ref_img, 
                y_n=y_n, 
                record=cfg.record, 
                fname=file_idx+f'_run_{j}', 
                save_root=out_path, 
                inv_transform=inv_transform, 
                metrics=metrics
            )
            all_samples = {'x': x_samples, 'z': z_samples}
            for xz in ['x', 'z']:
                samples = all_samples[xz]
                samples = inv_transform(samples)
                sample = samples[[-1]] # take the last sample as the single sample for calculating metrics
                if len(samples) > 1:
                    mean, std = torch.mean(samples, dim=0, keepdim=True), torch.std(samples, dim=0, keepdim=True)

                # logging
                log[xz]["samples"].append(sample.permute(0, 2, 3, 1).squeeze().cpu().numpy())
                for name, metric in metrics.items():
                    log[xz][name+"_sample"].append(metric(sample.cuda(), inv_transform(ref_img).cuda()).item())
                log[xz]["consistency_sample"].append(torch.norm(operator.forward(transform(sample.cuda())) - y_n.cuda()).item())
                plt.imsave(os.path.join(out_path, 'recon', file_idx+f'_run_{j}_sample_{xz}.png'), log[xz]["samples"][-1], cmap=cmap)

                if len(samples) > 1:
                    log[xz]["means"].append(mean.permute(0, 2, 3, 1).squeeze().cpu().numpy())
                    log[xz]["stds"].append(std.permute(0, 2, 3, 1).squeeze().cpu().numpy())
                    for name, metric in metrics.items():
                        log[xz][name+"_mean"].append(metric(mean.cuda(), inv_transform(ref_img).cuda()).item())
                    log[xz]["consistency_mean"].append(torch.norm(operator.forward(transform(mean.cuda())) - y_n.cuda()).item())
                    plt.imsave(os.path.join(out_path, 'recon', file_idx+f'_run_{j}_mean_{xz}.png'), log[xz]["means"][-1], cmap=cmap)
                    # plt.imsave(os.path.join(out_path, 'recon', file_idx+f'_run_{j}_std.png'), log["stds"][-1], cmap=cmap)

        np.save(os.path.join(out_path, 'recon', file_idx+'_log.npy'), log)

        for xz in ['x', 'z']:
            with open(os.path.join(out_path, 'recon', file_idx+f'_metrics_{xz}.txt'), "w") as f:
                f.write(f'Statistics based on ONE sample for each run ({cfg.num_runs} runs in total):\n')
                f.write('\n')
                for name, _ in metrics.items():
                    f.write(f'{name} (avg over {cfg.num_runs} runs): {np.mean(log[xz][name+"_sample"])}\n')
                f.write(f'consistency_sample (avg over {cfg.num_runs} runs): {np.mean(log[xz]["consistency_sample"])}\n')
                f.write('\n')
                for name, _ in metrics.items():
                    best_fn = np.amin if name == 'lpips' else np.amax
                    f.write(f'{name} (best among {cfg.num_runs} runs): {best_fn(log[xz][name+"_sample"])}\n')
                f.write(f'consistency_sample (best among {cfg.num_runs} runs): {np.amin(log[xz]["consistency_sample"])}\n')
                if len(samples) > 1:
                    f.write('\n')
                    f.write('='*70+'\n')
                    f.write('\n')
                    f.write(f'Statistics based on the mean over {len(samples)} samples for each run ({cfg.num_runs} runs in total):\n')
                    f.write('\n')
                    for name, _ in metrics.items():
                        f.write(f'{name} (avg over {cfg.num_runs} runs): {np.mean(log[xz][name+"_mean"])}\n')
                    f.write(f'consistency_mean (avg over {cfg.num_runs} runs): {np.mean(log[xz]["consistency_mean"])}\n')
                    f.write('\n')
                    for name, _ in metrics.items():
                        best_fn = np.amin if name == 'lpips' else np.amax
                        f.write(f'{name} (best among {cfg.num_runs} runs): {best_fn(log[xz][name+"_mean"])}\n')
                    f.write(f'consistency_mean (best among {cfg.num_runs} runs): {np.amin(log[xz]["consistency_mean"])}\n')
                f.write('\n')
                f.write('='*70+'\n')
                f.write('\n')
                f.write(f'consistency (gt): {log[xz]["consistency_gt"]}\n')
                f.close()

        # meta logging
        for xz in ['x', 'z']:
            meta_log[xz]["consistency_gt"].append(log[xz]["consistency_gt"])
            sample_recon_mean = torch.mean(torch.from_numpy(np.array(log[xz]["samples"])), dim=0)
            if len(sample_recon_mean.shape) == 2:
                sample_recon_mean = sample_recon_mean.unsqueeze(2) # add a channel dimension
            sample_recon_mean = sample_recon_mean.permute(2, 0, 1).unsqueeze(0).to(device)
            for name, metric in metrics.items():
                meta_log[xz]["statistics_based_on_one_sample"][name+"_mean_recon_of_all_runs"].append(metric(sample_recon_mean, inv_transform(ref_img)).item())
                meta_log[xz]["statistics_based_on_one_sample"][name+"_last_of_all_runs"].append(log[xz][name+"_sample"][-1])
                best_fn = np.amin if name == 'lpips' else np.amax
                meta_log[xz]["statistics_based_on_one_sample"][name+"_best_of_all_runs"].append(best_fn(log[xz][name+"_sample"]))
            meta_log[xz]["statistics_based_on_one_sample"]["consistency_mean_recon_of_all_runs"].append(torch.norm(operator.forward(transform(sample_recon_mean)) - y_n).item())
            meta_log[xz]["statistics_based_on_one_sample"]["consistency_last_of_all_runs"].append(log[xz]["consistency_sample"][-1])
            meta_log[xz]["statistics_based_on_one_sample"]["consistency_best_of_all_runs"].append(np.amin(log[xz]["consistency_sample"]))
            if len(samples) > 1:
                mean_recon_mean = torch.mean(torch.from_numpy(np.array(log[xz]["means"])), dim=0)
                if len(mean_recon_mean.shape) == 2:
                    mean_recon_mean = mean_recon_mean.unsqueeze(2) # add a channel dimension
                mean_recon_mean = mean_recon_mean.permute(2, 0, 1).unsqueeze(0).to(device)
                for name, metric in metrics.items():
                    meta_log[xz]["statistics_based_on_mean"][name+"_mean_recon_of_all_runs"].append(metric(mean_recon_mean, inv_transform(ref_img)).item())
                    meta_log[xz]["statistics_based_on_mean"][name+"_last_of_all_runs"].append(log[xz][name+"_mean"][-1])
                    best_fn = np.amin if name == 'lpips' else np.amax
                    meta_log[xz]["statistics_based_on_mean"][name+"_best_of_all_runs"].append(best_fn(log[xz][name+"_mean"]))
                meta_log[xz]["statistics_based_on_mean"]["consistency_mean_recon_of_all_runs"].append(torch.norm(operator.forward(transform(mean_recon_mean)) - y_n).item())
                meta_log[xz]["statistics_based_on_mean"]["consistency_last_of_all_runs"].append(log[xz]["consistency_mean"][-1])
                meta_log[xz]["statistics_based_on_mean"]["consistency_best_of_all_runs"].append(np.amin(log[xz]["consistency_mean"]))

    # meta logging
    np.save(os.path.join(out_path, 'meta_log.npy'), meta_log)
    for xz in ['x', 'z']:
        with open(os.path.join(out_path, f'meta_metrics_{xz}.txt'), "w") as f:
            f.write(f'Statistics based on ONE sample for each run ({cfg.num_runs} runs in total) of each test image:\n')
            f.write('\n')
            for name, _ in metrics.items():
                f.write(f'{name}_mean_recon_of_{cfg.num_runs}_runs (avg over {num_test_images} test images): {np.mean(meta_log[xz]["statistics_based_on_one_sample"][name+"_mean_recon_of_all_runs"])}\n')
            f.write(f'consistency_mean_recon_of_{cfg.num_runs}_runs (avg over {num_test_images} test images): {np.mean(meta_log[xz]["statistics_based_on_one_sample"]["consistency_mean_recon_of_all_runs"])}\n')
            f.write('\n')
            for name, _ in metrics.items():
                f.write(f'{name}_last_of_{cfg.num_runs}_runs (avg over {num_test_images} test images): {np.mean(meta_log[xz]["statistics_based_on_one_sample"][name+"_last_of_all_runs"])}\n')
            f.write(f'consistency_last_of_all_runs (avg over {num_test_images} test images): {np.mean(meta_log[xz]["statistics_based_on_one_sample"]["consistency_last_of_all_runs"])}\n')
            f.write('\n')
            for name, _ in metrics.items():
                f.write(f'{name}_best_of_{cfg.num_runs}_runs (avg over {num_test_images} test images): {np.mean(meta_log[xz]["statistics_based_on_one_sample"][name+"_best_of_all_runs"])}\n')
            f.write(f'consistency_best_of_all_runs (avg over {num_test_images} test images): {np.mean(meta_log[xz]["statistics_based_on_one_sample"]["consistency_best_of_all_runs"])}\n')
            if len(samples) > 1:
                f.write('\n')
                f.write('='*70+'\n')
                f.write('\n')
                f.write(f'Statistics based on the mean over {len(samples)} samples for each run ({cfg.num_runs} runs in total) of each test image:\n')
                f.write('\n')
                for name, _ in metrics.items():
                    f.write(f'{name}_mean_recon_of_{cfg.num_runs}_runs (avg over {num_test_images} test images): {np.mean(meta_log[xz]["statistics_based_on_mean"][name+"_mean_recon_of_all_runs"])}\n')
                f.write(f'consistency_mean_recon_of_{cfg.num_runs}_runs (avg over {num_test_images} test images): {np.mean(meta_log[xz]["statistics_based_on_mean"]["consistency_mean_recon_of_all_runs"])}\n')
                f.write('\n')
                for name, _ in metrics.items():
                    f.write(f'{name}_last_of_{cfg.num_runs}_runs (avg over {num_test_images} test images): {np.mean(meta_log[xz]["statistics_based_on_mean"][name+"_last_of_all_runs"])}\n')
                f.write(f'consistency_last_of_all_runs (avg over {num_test_images} test images): {np.mean(meta_log[xz]["statistics_based_on_mean"]["consistency_last_of_all_runs"])}\n')
                f.write('\n')
                for name, _ in metrics.items():
                    f.write(f'{name}_best_of_{cfg.num_runs}_runs (avg over {num_test_images} test images): {np.mean(meta_log[xz]["statistics_based_on_mean"][name+"_best_of_all_runs"])}\n')
                f.write(f'consistency_best_of_all_runs (avg over {num_test_images} test images): {np.mean(meta_log[xz]["statistics_based_on_mean"]["consistency_best_of_all_runs"])}\n')
            f.write('\n')
            f.write('='*70+'\n')
            f.write('\n')
            f.write(f'consistency (gt) (avg over {num_test_images} test images): {np.mean(meta_log[xz]["consistency_gt"])}\n')
            f.close()

    logger.info(f"Finished inference")

if __name__ == '__main__':
    posterior_sample()