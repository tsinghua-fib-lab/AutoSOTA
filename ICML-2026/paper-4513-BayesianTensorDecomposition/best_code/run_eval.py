"""Flexible evaluation script for DiffBCP SOTA optimization.
Usage: python run_eval.py [hydra overrides] +num_val=N +run_tag=NAME
Example: python run_eval.py gpu=0 +data=ffhq data.mask_name=random_mask_obs03 +task=completion task.noise.sigma=0.05 +model=edm_unet_adm_dps_ffhq +sampler=pnp_edm sampler.mode=vp_sde sampler.num_iters=100 sampler.anneal_const=100.0 +num_val=5 +run_tag=baseline
"""
import torch, os, hydra, json, sys, numpy as np, shutil, subprocess
# Clear stale bytecode cache to ensure code changes take effect
subprocess.run(['find', '/repo', '-name', '__pycache__', '-type', 'd', '-exec', 'rm', '-rf', '{}', '+'], capture_output=True)
# Prevent new bytecode writes to overlay
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['MPLCONFIGDIR'] = '/autosota_artifacts/paper-4513/sota/tmp/mpl'
os.makedirs('/autosota_artifacts/paper-4513/sota/tmp/mpl', exist_ok=True)
# Prevent writes to overlay filesystem
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['MPLCONFIGDIR'] = '/autosota_artifacts/paper-4513/sota/tmp/mpl'
os.makedirs('/autosota_artifacts/paper-4513/sota/tmp/mpl', exist_ok=True)
from torchvision import transforms
from monai.metrics import PSNRMetric, SSIMMetric
from taming.modules.losses.lpips import LPIPS
from src.utils import set_all_seed
from src.tasks import get_operator, get_noise
from src.models import get_model
from src.samplers import get_sampler
from omegaconf import OmegaConf, DictConfig


@hydra.main(version_base=None, config_path="configs", config_name="default")
def run_eval(cfg):
    set_all_seed(cfg.seed)
    OmegaConf.set_struct(cfg, False)

    device_str = f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"[run_eval] Using device: {device_str}", flush=True)

    transform = transforms.Compose([transforms.Normalize((0.5), (0.5))])
    inv_transform = transforms.Compose([
        transforms.Normalize((-1), (2)),
        transforms.Lambda(lambda x: x.clamp(0, 1).detach())
    ])
    dataset = np.load(cfg.data.root)
    dataset = torch.from_numpy(dataset).float().permute(0, 3, 1, 2) / 255.0
    num_test_images = min(len(dataset), cfg.get("num_val", len(dataset)))
    print(f"[run_eval] Evaluating on {num_test_images} images", flush=True)

    mask = np.load(f"./data/{cfg.data.mask_name}.npy")
    mask = torch.tensor(mask).float().permute(0, 3, 1, 2)

    model = get_model(**cfg.model)
    model = model.to(device)
    model.eval()

    all_sample_psnr = []
    all_sample_ssim = []
    all_sample_lpips = []
    all_mean_psnr = []
    all_mean_ssim = []
    all_mean_lpips = []

    for i in range(num_test_images):
        print(f"[run_eval] Image {i+1}/{num_test_images}", flush=True)

        metrics = {
            "psnr": PSNRMetric(max_val=1),
            "ssim": SSIMMetric(spatial_dims=2),
            "lpips": LPIPS().to(device).eval(),
        }

        ref_img = dataset[i].to(device)
        ref_img_t = transform(dataset[i]).to(device).unsqueeze(0)
        ref_mask = mask[i].to(device)

        operator = get_operator(**cfg.task.operator, mask=ref_mask, device=device)
        noiser = get_noise(**cfg.task.noise)

        sampler_cfg = OmegaConf.to_container(cfg.sampler, resolve=True)
        sampler_cfg["decomposition"]["shape"] = [3, 256, 256]
        sampler_cfg["decomposition"]["orig_shape"] = [3, 256, 256]

        sampler = get_sampler(DictConfig(sampler_cfg), model=model, operator=operator, noiser=noiser, device=device)

        y_n = noiser(operator.forward(ref_img_t))

        # Use temp directory for progress artifacts
        tmp_root = f"/autosota_artifacts/paper-4513/sota/tmp/progress_{os.getpid()}"
        os.makedirs(os.path.join(tmp_root, "progress"), exist_ok=True)

        x_samples, z_samples = sampler(
            gt=ref_img_t, y_n=y_n, record=False, fname=f"eval_{i:05d}",
            save_root=tmp_root, inv_transform=inv_transform, metrics=metrics
        )

        # Clean up temp artifacts
        shutil.rmtree(tmp_root, ignore_errors=True)

        ref = inv_transform(ref_img_t)

        # Single sample (last)
        last_z = inv_transform(z_samples[[-1]])
        s_psnr = metrics["psnr"](last_z.cuda(), ref.cuda()).item()
        s_ssim = metrics["ssim"](last_z.cuda(), ref.cuda()).item()
        s_lpips = metrics["lpips"](last_z.cuda(), ref.cuda()).item()
        all_sample_psnr.append(s_psnr)
        all_sample_ssim.append(s_ssim)
        all_sample_lpips.append(s_lpips)

        # Posterior mean
        mean_z = inv_transform(torch.mean(z_samples, dim=0, keepdim=True))
        m_psnr = metrics["psnr"](mean_z.cuda(), ref.cuda()).item()
        m_ssim = metrics["ssim"](mean_z.cuda(), ref.cuda()).item()
        m_lpips = metrics["lpips"](mean_z.cuda(), ref.cuda()).item()
        all_mean_psnr.append(m_psnr)
        all_mean_ssim.append(m_ssim)
        all_mean_lpips.append(m_lpips)

        print(f"  PSNR(sample)={s_psnr:.2f} PSNR(mean)={m_psnr:.2f} "
              f"SSIM(mean)={m_ssim*100:.2f} LPIPS(mean)={m_lpips*100:.2f}", flush=True)

    result = {
        "num_images": num_test_images,
        "sample": {
            "PSNR": float(np.mean(all_sample_psnr)),
            "SSIM": float(np.mean(all_sample_ssim) * 100),
            "LPIPS": float(np.mean(all_sample_lpips) * 100),
        },
        "posterior_mean": {
            "PSNR": float(np.mean(all_mean_psnr)),
            "SSIM": float(np.mean(all_mean_ssim) * 100),
            "LPIPS": float(np.mean(all_mean_lpips) * 100),
        },
    }

    print("\n[run_eval] === FINAL (posterior mean) ===", flush=True)
    print(f"PSNR={result['posterior_mean']['PSNR']:.2f} "
          f"SSIM={result['posterior_mean']['SSIM']:.2f} "
          f"LPIPS={result['posterior_mean']['LPIPS']:.2f}", flush=True)

    os.makedirs("/repo/eval_results", exist_ok=True)
    result_path = f"/repo/eval_results/result_{cfg.get('run_tag', 'latest')}.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[run_eval] Result saved to {result_path}", flush=True)

    return result


if __name__ == "__main__":
    run_eval()
