import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src_pr import denoising_utils as denoising_utils_module
from src_pr.data_utils import Dataset, DatasetTurbulentPickle, Dataset_Paths
from src_pr.denoising_utils import (
    DenoisingDiffusion,
    b_xy_c_to_image,
    image_to_b_xy_c,
    load_model,
)
from src_pr.metrics import calculate_psnr, calculate_ssim
from src_pr.residuals_darcy import ResidualsDarcy
from src_pr.residuals_mechanics_K import ResidualsMechanics
from src_pr.residuals_turbulent import ResidualsTurbulent
from src_pr.residuals_poisson import ResidualsPoisson
from src_pr.unet_new import Unet3D

script_dir = Path(__file__).parent.absolute()
os.chdir(script_dir)


def normalize_positions(value):
    if value is None:
        positions = []
    elif isinstance(value, str):
        positions = [item.strip() for item in value.split(',') if item.strip()]
    elif isinstance(value, (list, tuple)):
        positions = list(value)
    else:
        raise TypeError('projection_positions must be a string, list, tuple, or null.')

    valid_positions = {'encoder', 'bottleneck', 'decoder', 'output'}
    unknown = sorted(set(positions) - valid_positions)
    if unknown:
        raise ValueError(f'Unknown projection_positions: {unknown}. Valid values: {sorted(valid_positions)}')
    return positions


def validate_config(config):
    gov_eqs = config.get('gov_eqs')
    if gov_eqs not in {'darcy', 'mechanics', 'turbulent', 'poisson'}:
        raise ValueError("gov_eqs must be one of: 'darcy', 'mechanics', 'turbulent', 'poisson'.")
    if gov_eqs != 'darcy' and bool(config.get('residual_grad_guidance', False)):
        raise ValueError('Residual gradient guidance is only implemented for Darcy.')
    if bool(config.get('use_projection_heads', False)):
        positions = normalize_positions(config.get('projection_positions', ['encoder', 'bottleneck', 'decoder']))
        if not positions:
            raise ValueError('use_projection_heads=True requires at least one projection_positions entry.')


def resolve_checkpoint(load_path, step):
    model_dir = Path(load_path) / 'model'
    if step is not None:
        checkpoint = model_dir / f'checkpoint_{step}.pt'
        if not checkpoint.exists():
            raise FileNotFoundError(f'Checkpoint not found: {checkpoint}')
        return checkpoint, step

    checkpoints = sorted(model_dir.glob('checkpoint_*.pt'), key=lambda p: int(p.stem.split('_')[-1]))
    if not checkpoints:
        raise FileNotFoundError(f'No checkpoint_*.pt files found in {model_dir}')
    checkpoint = checkpoints[-1]
    return checkpoint, int(checkpoint.stem.split('_')[-1])


def safe_psnr(gt, pred):
    if float(np.max(gt) - np.min(gt)) <= 1e-12:
        mse = float(np.mean((gt - pred) ** 2))
        return float('inf') if mse <= 1e-12 else float('nan')
    return float(calculate_psnr(gt, pred))


def safe_ssim(gt, pred):
    if float(np.max(gt) - np.min(gt)) <= 1e-12:
        return 1.0
    return float(calculate_ssim(gt, pred))


def reconstruction_metrics(gt_tensor, pred_tensor):
    gt_np = gt_tensor.detach().cpu().numpy()
    pred_np = pred_tensor.detach().cpu().numpy()
    rows = []
    for idx in range(gt_np.shape[0]):
        gt = gt_np[idx]
        pred = pred_np[idx]
        rows.append({
            'sample': idx,
            'mse': float(np.mean((gt - pred) ** 2)),
            'psnr': safe_psnr(gt, pred),
            'ssim': safe_ssim(gt, pred),
        })
    rows.append({
        'sample': 'mean',
        'mse': float(np.nanmean([r['mse'] for r in rows])),
        'psnr': float(np.nanmean([r['psnr'] for r in rows])),
        'ssim': float(np.nanmean([r['ssim'] for r in rows])),
    })
    return rows


def setup_task(config, use_double):
    gov_eqs = config['gov_eqs']
    data = {'gov_eqs': gov_eqs}

    if gov_eqs == 'darcy':
        data.update(input_dim=2, output_dim=2, pixels_per_dim=64, pixels_at_boundary=True,
                    domain_length=1.0, reverse_d1=True, bcs='none', return_optimizer=False,
                    return_inequality=False, sigmoid_last_channel=False)
        valid = Dataset(('./data/darcy/valid/p_data.csv', './data/darcy/valid/K_data.csv'), use_double=use_double)

    elif gov_eqs == 'mechanics':
        data.update(input_dim=2, output_dim=3, pixels_per_dim=64, pixels_at_boundary=True,
                    reverse_d1=True, bcs='none', return_optimizer=True,
                    return_inequality=True, sigmoid_last_channel=True)
        valid = Dataset_Paths('./data/mechanics/test/valid/fields/', use_double=use_double)


    elif gov_eqs == 'turbulent':
        pixels = int(config.get('pixels_per_dim', 64))
        data.update(input_dim=2, output_dim=1, pixels_per_dim=pixels, pixels_at_boundary=True,
                    domain_length=1.0, reverse_d1=False, bcs='none', return_optimizer=False,
                    return_inequality=False, sigmoid_last_channel=False)
        valid = DatasetTurbulentPickle(config.get('turbulent_data_path', './data/ch_2Dxysec.pickle'),
                                       pixels_per_dim=pixels, split='valid',
                                       train_fraction=float(config.get('turbulent_train_fraction', 0.9)),
                                       use_double=use_double)
    elif gov_eqs == 'poisson':
        data.update(input_dim=2, output_dim=1, pixels_per_dim=64, pixels_at_boundary=True,
                    domain_length=1.0, reverse_d1=False, bcs='none', return_optimizer=False,
                    return_inequality=False, sigmoid_last_channel=False)
        valid = Dataset(('./data/poisson/valid/rho_data.csv', './data/poisson/valid/U_data.csv'),
                       use_double=use_double)
    else:
        raise ValueError(f'Unknown gov_eqs: {gov_eqs}')

    data['valid'] = valid
    return data


def build_model(config, task, device):
    use_projection_heads = bool(config.get('use_projection_heads', False))
    projection_positions = normalize_positions(config.get('projection_positions', ['encoder', 'bottleneck', 'decoder']))
    projection_hidden_dim = int(config.get("projection_hidden_dim", 0) or 0)
    use_projection_residual = bool(config.get("use_projection_residual", False))
    gov_eqs = task['gov_eqs']

    kwargs = dict(
        sigmoid_last_channel=task['sigmoid_last_channel'],
        use_projection_heads=use_projection_heads,
        projection_positions=projection_positions,
        projection_hidden_dim=projection_hidden_dim,
        use_projection_residual=use_projection_residual,
    )
    if gov_eqs == 'mechanics':
        return Unet3D(dim=128, channels=task['output_dim'] + 3 + 4, out_dim=task['output_dim'], **kwargs).to(device)
    if gov_eqs == 'poisson':
        return Unet3D(dim=64, channels=task['output_dim'] + 1, out_dim=task['output_dim'], **kwargs).to(device)
    return Unet3D(dim=32, channels=task['output_dim'], **kwargs).to(device)


def build_residuals(config, task, model, device, use_ddim_x0, ddim_steps, residual_grad_guidance):
    gov_eqs = task['gov_eqs']
    if gov_eqs == 'darcy':
        return ResidualsDarcy(model=model, fd_acc=config['fd_acc'], pixels_per_dim=task['pixels_per_dim'],
                              pixels_at_boundary=task['pixels_at_boundary'], reverse_d1=task['reverse_d1'],
                              device=device, bcs=task['bcs'], domain_length=task['domain_length'],
                              residual_grad_guidance=residual_grad_guidance, use_ddim_x0=use_ddim_x0,
                              ddim_steps=ddim_steps)
    if gov_eqs == 'mechanics':
        return ResidualsMechanics(model=model, pixels_per_dim=task['pixels_per_dim'],
                                  pixels_at_boundary=task['pixels_at_boundary'], device=device, bcs=task['bcs'],
                                  no_BC_folder='./data/mechanics/solidspy_k_no_BC/', topopt_eval=True,
                                  use_ddim_x0=use_ddim_x0, ddim_steps=ddim_steps)
    if gov_eqs == 'turbulent':
        return ResidualsTurbulent(model=model, pixels_per_dim=task['pixels_per_dim'], device=device,
                                  lambda_wall=float(config.get('lambda_wall', 0.1)),
                                  lambda_smooth=float(config.get('lambda_smooth', 0.01)),
                                  lambda_gradient=float(config.get('lambda_gradient', 0.0)),
                                  lambda_near_wall=float(config.get('lambda_near_wall', 0.0)),
                                  near_wall_rows=int(config.get('near_wall_rows', 3)),
                                  residual_grad_guidance=residual_grad_guidance,
                                  use_ddim_x0=use_ddim_x0, ddim_steps=ddim_steps)
    if gov_eqs == 'poisson':
        return ResidualsPoisson(model=model, pixels_per_dim=task['pixels_per_dim'],
                                pixels_at_boundary=task['pixels_at_boundary'], device=device,
                                domain_length=task['domain_length'], fd_acc=config['fd_acc'],
                                use_ddim_x0=use_ddim_x0, ddim_steps=ddim_steps)
    raise ValueError(f'Unknown gov_eqs: {gov_eqs}')


def reconstruct_batch(batch, task, residuals, diffusion_utils, device):
    batch = batch.to(device)
    t_eval = torch.zeros(batch.shape[0], dtype=torch.long, device=device)
    gov_eqs = task['gov_eqs']

    if gov_eqs in ('darcy', 'turbulent'):
        x0 = batch
        residual_input = ((image_to_b_xy_c(x0), t_eval),)
    elif gov_eqs == 'poisson':
        rho = batch[:, 0:1]
        x0 = batch[:, 1:2]
        x_cond = torch.cat([x0, rho], dim=1)
        residual_input = ((image_to_b_xy_c(x_cond), t_eval), rho)
    elif gov_eqs == 'mechanics':
        conditioning, x0, bcs = torch.tensor_split(batch, (3, 6), dim=1)
        vf = conditioning[:, 0, 0, 0]
        residual_input = ((image_to_b_xy_c(x0), t_eval), bcs, vf, x0)
    else:
        raise ValueError(f'Unknown gov_eqs: {gov_eqs}')

    out = residuals.compute_residual(residual_input, reduce='none', return_model_out=True,
                                     return_optimizer=False, return_inequality=False,
                                     ddim_func=diffusion_utils.ddim_sample_x0)
    pred = out['model_out']
    if isinstance(pred, tuple):
        pred = pred[0]
    if pred.ndim == 3:
        pred = b_xy_c_to_image(pred)
    return x0.detach(), pred.detach(), out['residual'].detach()


def save_batch_images(out_dir, gt, pred, max_samples=4):
    out_dir.mkdir(parents=True, exist_ok=True)
    gt_np = gt.detach().cpu().numpy()
    pred_np = pred.detach().cpu().numpy()
    n = min(max_samples, gt_np.shape[0])
    for i in range(n):
        sample_dir = out_dir / f'sample_{i}'
        sample_dir.mkdir(exist_ok=True)
        for c in range(gt_np.shape[1]):
            np.savetxt(sample_dir / f'gt_{c}.csv', gt_np[i, c], delimiter=',')
            np.savetxt(sample_dir / f'pred_{c}.csv', pred_np[i, c], delimiter=',')
            fig, axes = plt.subplots(1, 2, figsize=(6, 3))
            axes[0].imshow(gt_np[i, c], cmap='viridis')
            axes[0].set_title('ground truth')
            axes[1].imshow(pred_np[i, c], cmap='viridis')
            axes[1].set_title('model output')
            for ax in axes:
                ax.axis('off')
            fig.tight_layout()
            fig.savefig(sample_dir / f'channel_{c}.png', bbox_inches='tight', pad_inches=0)
            plt.close(fig)


def save_generated_images(out_dir, samples, max_samples=4):
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_np = samples.detach().cpu().numpy()
    n = min(max_samples, samples_np.shape[0])
    for i in range(n):
        sample_dir = out_dir / f'sample_{i}'
        sample_dir.mkdir(exist_ok=True)
        for c in range(samples_np.shape[1]):
            np.savetxt(sample_dir / f'generated_{c}.csv', samples_np[i, c], delimiter=',')
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.imshow(samples_np[i, c], cmap='viridis')
            ax.set_title('generated')
            ax.axis('off')
            fig.tight_layout()
            fig.savefig(sample_dir / f'generated_channel_{c}.png', bbox_inches='tight', pad_inches=0)
            plt.close(fig)


def build_mechanics_conditioning(batch, num_samples, device):
    batch = batch.to(device)
    n = min(num_samples, batch.shape[0])
    batch = batch[torch.randperm(batch.shape[0], device=device)[:n]]
    conditioning, x0, bcs = torch.tensor_split(batch, (3, 6), dim=1)
    return (conditioning, bcs, x0), n


def run_generative_eval(args, task, residuals, diffusion_utils, dl_valid, output_dir, device):
    gov_eqs = task['gov_eqs']
    num_samples = int(args.num_samples)
    if gov_eqs in ('darcy', 'turbulent'):
        conditioning_input = None
        sample_shape = (num_samples, task['output_dim'], task['pixels_per_dim'], task['pixels_per_dim'])
        n_samples = num_samples
    elif gov_eqs == 'poisson':
        batch = next(iter(dl_valid)).to(device)
        n = min(num_samples, batch.shape[0])
        batch = batch[torch.randperm(batch.shape[0], device=device)[:n]]
        conditioning_input = batch[:, 0:1]  # rho
        sample_shape = (n, task['output_dim'], task['pixels_per_dim'], task['pixels_per_dim'])
        n_samples = n
    elif gov_eqs == 'mechanics':
        conditioning_input, n_samples = build_mechanics_conditioning(next(iter(dl_valid)), num_samples, device)
        sample_shape = (n_samples, task['output_dim'], task['pixels_per_dim'] + 1, task['pixels_per_dim'] + 1)
    else:
        raise ValueError(f'Unknown gov_eqs: {gov_eqs}')

    output = diffusion_utils.p_sample_loop(
        conditioning_input,
        sample_shape,
        save_output=False,
        surpress_noise=True,
        residual_func=residuals,
        eval_residuals=True,
        return_optimizer=task['return_optimizer'],
        return_inequality=task['return_inequality'],
    )
    x_seq = output[0][0]
    aux = output[1]
    samples = torch.stack(x_seq, dim=0)[-1]
    residual = aux['residual'].abs().mean(dim=tuple(range(1, aux['residual'].ndim))).detach().cpu().numpy()

    rows = {
        'sample': list(range(n_samples)) + ['mean'],
        'mean_abs_residual': list(residual) + [float(np.nanmean(residual))],
    }
    if task['return_inequality'] and 'inequality_quant' in aux:
        ineq = aux['inequality_quant'].detach().cpu().numpy()
        rows['inequality'] = list(ineq) + [float(np.nanmean(ineq))]
    if task['return_optimizer'] and 'optimized_quant' in aux:
        opt = aux['optimized_quant'].detach().cpu().numpy()
        rows['optimized_quantity'] = list(opt) + [float(np.nanmean(opt))]

    pd.DataFrame(rows).to_csv(output_dir / 'generative_sample_statistics.csv', index=False)
    if args.save_images:
        save_generated_images(output_dir / 'generated_examples', samples)
    return float(np.nanmean(residual))


def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained U-Net PIDM/REPA-P checkpoint.')
    parser.add_argument('--name', type=str, required=True, help='Run name under ./trained_models/')
    parser.add_argument('--step', type=int, default=None, help='Checkpoint step. Defaults to latest.')
    parser.add_argument('--gpu', '-g', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--num-batches', type=int, default=1, help='Number of validation batches to evaluate.')
    parser.add_argument('--mode', choices=('reconstruction', 'generative'), default='reconstruction',
                        help='reconstruction evaluates t=0 denoising on validation data; generative samples from noise.')
    parser.add_argument('--num-samples', type=int, default=20, help='Number of generated samples for --mode generative.')
    parser.add_argument('--use-ema', action='store_true', help='Load EMA weights from the checkpoint when available.')
    parser.add_argument('--pag-scale', type=float, default=0.0, help='PAG scale. 0=disabled. Recommended: 3.0.')
    parser.add_argument('--save-images', action='store_true', help='Save PNG/CSV examples for the first batch.')
    args = parser.parse_args()

    if args.gpu is not None:
        device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    denoising_utils_module.device = device
    print(f'Using device: {device}')

    load_path = Path('./trained_models') / args.name
    config = yaml.safe_load((load_path / 'model' / 'model.yaml').read_text())
    validate_config(config)
    checkpoint_path, step = resolve_checkpoint(load_path, args.step)
    print(f'Run: {args.name}')
    print(f'gov_eqs: {config["gov_eqs"]}')
    print(f'use_projection_heads: {bool(config.get("use_projection_heads", False))}')
    print(f'checkpoint: {checkpoint_path}')

    use_double = False
    task = setup_task(config, use_double)
    batch_size = args.batch_size or int(config.get('eval_batch_size', config.get('train_batch_size', 16)))
    dl_valid = DataLoader(task['valid'], batch_size=batch_size, shuffle=False)

    residual_grad_guidance = bool(config.get('residual_grad_guidance', False))
    use_ddim_x0 = config.get('x0_estimation', 'mean') == 'sample'
    ddim_steps = int(config.get('ddim_steps', 0))
    diffusion_utils = DenoisingDiffusion(int(config['diff_steps']), device, residual_grad_guidance)
    diffusion_utils.pag_scale = args.pag_scale
    model = build_model(config, task, device)
    load_model(checkpoint_path, model, use_ema=args.use_ema)
    model.eval()

    residuals = build_residuals(config, task, model, device, use_ddim_x0, ddim_steps, residual_grad_guidance)
    output_dir = load_path / 'evaluation' / f'step_{step}'
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == 'reconstruction':
        all_rows = []
        residual_means = []
        grad_context = torch.enable_grad if residual_grad_guidance else torch.no_grad
        with grad_context():
            for batch_idx, batch in enumerate(dl_valid):
                if args.num_batches >= 0 and batch_idx >= args.num_batches:
                    break
                gt, pred, residual = reconstruct_batch(batch, task, residuals, diffusion_utils, device)
                rows = reconstruction_metrics(gt, pred)
                for row in rows:
                    row['batch'] = batch_idx
                all_rows.extend(rows)
                residual_means.append(float(residual.abs().mean().cpu()))
                if batch_idx == 0 and args.save_images:
                    save_batch_images(output_dir / 'reconstruction_examples', gt, pred)

        pd.DataFrame(all_rows).to_csv(output_dir / 'reconstruction_metrics.csv', index=False)
        mean_abs_residual = float(np.mean(residual_means)) if residual_means else float('nan')
    else:
        mean_abs_residual = run_generative_eval(args, task, residuals, diffusion_utils, dl_valid, output_dir, device)

    summary = pd.DataFrame([{
        'run_name': args.name,
        'step': step,
        'mode': args.mode,
        'gov_eqs': config['gov_eqs'],
        'use_projection_heads': bool(config.get('use_projection_heads', False)),
        'projection_positions': ','.join(normalize_positions(config.get('projection_positions', ['encoder', 'bottleneck', 'decoder']))),
        'weights': 'ema' if args.use_ema else 'model',
        'mean_abs_residual': mean_abs_residual,
    }])
    summary.to_csv(output_dir / 'summary.csv', index=False)
    print(summary.to_string(index=False))
    print(f'Wrote evaluation files to {output_dir}')


if __name__ == '__main__':
    main()
