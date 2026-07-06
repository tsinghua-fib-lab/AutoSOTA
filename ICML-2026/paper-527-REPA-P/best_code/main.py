import os, yaml
import csv
import argparse
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from tqdm import tqdm
from pathlib import Path
import numpy as np
import pandas as pd

# =============================================================================

# Use paths relative to this script.
script_dir = Path(__file__).parent.absolute()
os.chdir(script_dir)

from src_pr.data_utils import Dataset, Dataset_Paths, DatasetTurbulentPickle, cycle
from torch.utils.data import DataLoader
from src_pr import denoising_utils as denoising_utils_module
from src_pr.denoising_utils import (
    DenoisingDiffusion,
    EMA,
    b_xy_c_to_image,
    exists,
    image_array_to_gif,
    image_to_b_xy_c,
    load_model,
    noop,
    save_model,
)
from src_pr.metrics import calculate_psnr, calculate_ssim
from src_pr.unet_new import Unet3D
from src_pr.residuals_darcy import ResidualsDarcy
from src_pr.residuals_mechanics_K import ResidualsMechanics
from src_pr.residuals_turbulent import ResidualsTurbulent
from src_pr.residuals_poisson import ResidualsPoisson


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


def validate_config(config, projection_positions):
    gov_eqs = config.get('gov_eqs')
    if gov_eqs not in {'darcy', 'mechanics', 'turbulent', 'poisson'}:
        raise ValueError("gov_eqs must be one of: 'darcy', 'mechanics', 'turbulent'.")

    x0_estimation = config.get('x0_estimation')
    if x0_estimation not in {'mean', 'sample'}:
        raise ValueError("x0_estimation must be either 'mean' or 'sample'.")

    residual_grad_guidance = bool(config.get('residual_grad_guidance', False))
    n_correction = int(config.get('N_correction', 0))
    m_correction = int(config.get('M_correction', 0))
    if gov_eqs != 'darcy' and (residual_grad_guidance or n_correction > 0 or m_correction > 0):
        raise ValueError('Gradient guidance and CoCoGen corrections are only implemented for Darcy.')

    if bool(config.get('use_projection_heads', False)):
        c_projection = float(config.get('c_projection', 0.0))
        if not projection_positions:
            raise ValueError('use_projection_heads=True requires at least one projection_positions entry.')
        if c_projection <= 0.0:
            raise ValueError('use_projection_heads=True requires c_projection > 0 so projection loss participates.')

# Command-line arguments
parser = argparse.ArgumentParser(description='Train diffusion models.')
parser.add_argument('--gpu', '-g', type=int, default=None, 
                    help='GPU device ID. If omitted, use the default device or CUDA_VISIBLE_DEVICES.')
parser.add_argument('--config', type=str, default=str(script_dir / 'model.yaml'),
                    help='Path to the YAML config file.')
args = parser.parse_args()

# Device setup
if args.gpu is not None:
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Using specified device: {device}')
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using default device: {device}')
denoising_utils_module.device = device

config_path = Path(args.config)
if not config_path.exists():
    raise FileNotFoundError(f'Config file not found: {config_path}')
config = yaml.safe_load(config_path.read_text())
print(f'Using config: {config_path} (gov_eqs={config.get("gov_eqs")})')

name = config.get('run_name', f'{config.get("gov_eqs", "run")}.mean')
wandb_track = bool(config.get('wandb_track', False))
resume_step = int(config.get('resume_step', 0))
load_model_flag = bool(config.get('load_model_flag', False))
load_path = f'./trained_models/{name}'
load_model_step = resume_step
start_iteration = load_model_step + 1 if load_model_flag else 1

# Diffusion parameters
if config['x0_estimation'] == 'mean':
    use_ddim_x0 = False
elif config['x0_estimation'] == 'sample':
    use_ddim_x0 = True
else:
    raise ValueError("x0_estimation must be either 'mean' or 'sample'.")
ddim_steps = config['ddim_steps']
residual_grad_guidance = config['residual_grad_guidance'] # gradient guidance scale as in https://www.sciencedirect.com/science/article/pii/S0021999123000670
# residual corrections (can be changed after training since only affects inference) similar to https://arxiv.org/abs/2312.10527
correction_mode = config['correction_mode'] # 'x0', 'xt', CoCoGen use xt
M_correction = config['M_correction'] # correction steps after x0
N_correction = config['N_correction'] # correction steps before x0
gov_eqs = config['gov_eqs']
if gov_eqs != 'darcy' and (residual_grad_guidance or N_correction > 0 or M_correction > 0):
    raise ValueError('Gradient guidance and CoCoGen only implemented for Darcy flow study.')
fd_acc = config['fd_acc'] # finite difference accuracy
c_data = config['c_data']
c_residual = config['c_residual']
c_ineq = config['c_ineq']
lambda_opt = config['lambda_opt'] # (negative sign corresponds to max.)
diff_steps = config['diff_steps']
use_dynamic_threshold = False
self_condition = False

# Projection-head parameters
use_projection_heads = bool(config.get('use_projection_heads', False))
projection_positions = normalize_positions(config.get('projection_positions', ['encoder', 'bottleneck', 'decoder']))
validate_config(config, projection_positions)
projection_hidden_dim = int(config.get("projection_hidden_dim", 64) or 0)
use_projection_residual = bool(config.get("use_projection_residual", False))
c_projection = float(config.get('c_projection', 0.0))

# Evaluation parameters, optionally overridden by YAML.
test_eval_freq = int(config.get('test_eval_freq', 500))
sample_freq = int(config.get('sample_freq', 500))
# Save a model checkpoint every N iterations.
checkpoint_save_freq = int(config.get('checkpoint_save_freq', 500))
ema_start = int(config.get('ema_start', 1000))
use_ema_for_eval = bool(config.get('use_ema_for_eval', True))
ema = EMA(0.99)
topopt_eval = True # evaluate topopt metrics (only for mechanics as governing equations)
use_double = False
no_samples = int(config.get('no_samples', 20))
save_output = True
eval_residuals = True
create_gif = False
# Optional: record per-step physics residuals along the reverse process.
collect_physics_per_step = bool(config.get('collect_physics_per_step', False))

# training parameters and datasets
data_paths = None
lambda_wall = float(config.get('lambda_wall', 0.1))
lambda_smooth = float(config.get('lambda_smooth', 0.01))
lambda_gradient = float(config.get('lambda_gradient', 0.0))
lambda_near_wall = float(config.get('lambda_near_wall', 0.0))
near_wall_rows = int(config.get('near_wall_rows', 3))
if gov_eqs == 'darcy':
    # [xi_1,xi_2] -> [p,K]
    input_dim = 2
    output_dim = 2
    pixels_at_boundary = True
    domain_length = 1.
    reverse_d1 = True # this is to be consistent with ascending coordinates in the figures
    data_paths = ('./data/darcy/train/p_data.csv', './data/darcy/train/K_data.csv')
    data_paths_valid = ('./data/darcy/valid/p_data.csv', './data/darcy/valid/K_data.csv')
    bcs = 'none' # 'none', 'periodic'
    pixels_per_dim = 64
    return_optimizer = False
    return_inequality = False
    ds = Dataset(data_paths, use_double=use_double)
    ds_valid = Dataset(data_paths_valid, use_double=use_double)
    if use_ddim_x0:
        train_batch_size = 16
    else:
        train_batch_size = int(config.get('train_batch_size', 64))
    sigmoid_last_channel = False
    train_iterations = int(config.get('train_iterations', 150000))
elif gov_eqs == 'mechanics':
    input_dim = 2
    output_dim = 3
    # [xi_1,xi_2] -> [u_1,u_2,rho]
    pixels_at_boundary = True
    reverse_d1 = True
    data_paths = ('./data/mechanics/train/fields/')
    data_paths_valid = ('./data/mechanics/test/valid/fields/')
    data_paths_test_level_1 = ('./data/mechanics/test/test_level_1/fields/')
    data_paths_test_level_2 = ('./data/mechanics/test/test_level_2/fields/')
    bcs = 'none' # 'none', 'periodic'
    pixels_per_dim = 64
    return_optimizer = True
    return_inequality = True
    ds = Dataset_Paths(data_paths, use_double=use_double)
    ds_valid = Dataset_Paths(data_paths_valid, use_double=use_double)
    # ds_test_level_1 = Dataset_Paths(data_paths_test_level_1, use_double=use_double)
    # ds_test_level_2 = Dataset_Paths(data_paths_test_level_2, use_double=use_double)
    if use_ddim_x0:
        train_batch_size = 8
    else:
        train_batch_size = int(config.get('train_batch_size', 8))  # lower default to avoid OOM
    # dl_test_level_1 = DataLoader(ds_test_level_1, batch_size = train_batch_size, shuffle=True, generator=torch.Generator(device=device))
    # dl_test_level_2 = DataLoader(ds_test_level_2, batch_size = train_batch_size, shuffle=True, generator=torch.Generator(device=device))
    sigmoid_last_channel = True
    train_iterations = int(config.get('train_iterations', 300000))
elif gov_eqs == 'turbulent':
    # single-channel turbulent channel-flow slice, resized to Darcy-style 64x64
    input_dim = 2
    output_dim = 1
    pixels_at_boundary = True
    domain_length = 1.0
    reverse_d1 = False
    bcs = 'none'
    pixels_per_dim = int(config.get('pixels_per_dim', 64))
    return_optimizer = False
    return_inequality = False
    turbulent_data_path = config.get(
        'turbulent_data_path',
        str(script_dir / 'data' / 'ch_2Dxysec.pickle'),
    )
    turbulent_train_fraction = float(config.get('turbulent_train_fraction', 0.9))
    ds = DatasetTurbulentPickle(
        turbulent_data_path,
        pixels_per_dim=pixels_per_dim,
        split='train',
        train_fraction=turbulent_train_fraction,
        use_double=use_double,
    )
    ds_valid = DatasetTurbulentPickle(
        turbulent_data_path,
        pixels_per_dim=pixels_per_dim,
        split='valid',
        train_fraction=turbulent_train_fraction,
        use_double=use_double,
    )
    if use_ddim_x0:
        train_batch_size = int(config.get('train_batch_size', 16))
    else:
        train_batch_size = int(config.get('train_batch_size', 64))
    sigmoid_last_channel = False
    train_iterations = int(config.get('train_iterations', 150000))
elif gov_eqs == 'poisson':
    # Conditional generation: rho -> U
    input_dim = 2
    output_dim = 1
    pixels_at_boundary = True
    domain_length = 1.
    reverse_d1 = False
    data_paths = ('./data/poisson/train/rho_data.csv', './data/poisson/train/U_data.csv')
    data_paths_valid = ('./data/poisson/valid/rho_data.csv', './data/poisson/valid/U_data.csv')
    bcs = 'none'
    pixels_per_dim = 64
    return_optimizer = False
    return_inequality = False
    ds = Dataset(data_paths, use_double=use_double)
    ds_valid = Dataset(data_paths_valid, use_double=use_double)
    if use_ddim_x0:
        train_batch_size = 8
    else:
        train_batch_size = int(config.get('train_batch_size', 8))
    sigmoid_last_channel = False
    train_iterations = int(config.get('train_iterations', 20000))
else:
    raise ValueError('Unknown governing equations.')



if use_double:
    torch.set_default_dtype(torch.float64)

dl = cycle(DataLoader(ds, batch_size = train_batch_size, shuffle=True))
dl_valid = cycle(DataLoader(ds_valid, batch_size = train_batch_size, shuffle=False))

# diffusion utils
diffusion_utils = DenoisingDiffusion(diff_steps, device, residual_grad_guidance)

# model 
if gov_eqs == 'darcy':
    model = Unet3D(
        dim = 32, 
        channels = output_dim, 
        sigmoid_last_channel = sigmoid_last_channel,
        use_projection_heads = use_projection_heads,
        projection_positions = projection_positions,
        projection_hidden_dim = projection_hidden_dim,
        use_projection_residual = use_projection_residual
    ).to(device)
elif gov_eqs == 'mechanics':
    model = Unet3D(
        dim = 128, 
        channels = output_dim+3+4, 
        out_dim = output_dim, 
        sigmoid_last_channel = sigmoid_last_channel,
        use_projection_heads = use_projection_heads,
        projection_positions = projection_positions,
        projection_hidden_dim = projection_hidden_dim,
        use_projection_residual = use_projection_residual
    ).to(device)
elif gov_eqs == 'turbulent':
    model = Unet3D(
        dim = 32,
        channels = output_dim,
        sigmoid_last_channel = False,
        use_projection_heads = use_projection_heads,
        projection_positions = projection_positions,
        projection_hidden_dim = projection_hidden_dim,
        use_projection_residual = use_projection_residual
    ).to(device)
elif gov_eqs == 'poisson':
    model = Unet3D(
        dim = 64,
        channels = output_dim + 1,  # U (1ch) + rho condition (1ch)
        out_dim = output_dim,
        sigmoid_last_channel = False,
        use_projection_heads = use_projection_heads,
        projection_positions = projection_positions,
        projection_hidden_dim = projection_hidden_dim,
        use_projection_residual = use_projection_residual
    ).to(device)
else:
    raise ValueError('Unknown governing equations, cannot create model.')
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Number of trainable parameters: {num_params}')

# residual computation based on governing equations
if gov_eqs == 'darcy':
    residuals = ResidualsDarcy(model = model, fd_acc = fd_acc, pixels_per_dim = pixels_per_dim, pixels_at_boundary = pixels_at_boundary, reverse_d1 = reverse_d1, device = device, bcs = bcs, domain_length = domain_length, residual_grad_guidance= residual_grad_guidance, use_ddim_x0 = use_ddim_x0, ddim_steps = ddim_steps)
elif gov_eqs == 'mechanics':
    no_BC_folder = str(script_dir / 'data' / 'mechanics' / 'solidspy_k_no_BC')
    if not os.path.exists(no_BC_folder):
        raise FileNotFoundError(f'Missing directory: {no_BC_folder}')
    residuals = ResidualsMechanics(model = model, pixels_per_dim = pixels_per_dim, pixels_at_boundary = pixels_at_boundary, device = device, bcs = bcs, no_BC_folder = no_BC_folder + '/', topopt_eval = topopt_eval, use_ddim_x0 = use_ddim_x0, ddim_steps = ddim_steps)
elif gov_eqs == 'turbulent':
    residuals = ResidualsTurbulent(
        model = model,
        pixels_per_dim = pixels_per_dim,
        device = device,
        lambda_wall = lambda_wall,
        lambda_smooth = lambda_smooth,
        lambda_gradient = lambda_gradient,
        lambda_near_wall = lambda_near_wall,
        near_wall_rows = near_wall_rows,
        residual_grad_guidance = residual_grad_guidance,
        use_ddim_x0 = use_ddim_x0,
        ddim_steps = ddim_steps,
    )
elif gov_eqs == 'poisson':
    residuals = ResidualsPoisson(
        model = model,
        pixels_per_dim = pixels_per_dim,
        pixels_at_boundary = pixels_at_boundary,
        device = device,
        domain_length = domain_length,
        fd_acc = fd_acc,
        use_ddim_x0 = use_ddim_x0,
        ddim_steps = ddim_steps,
    )
else:
    raise ValueError('Unknown residuals mode.')

optimizer = optim.Adam(model.parameters(), lr=1.e-4)
ema.register(model)
if load_model_flag:
    load_model(
        Path(load_path, 'model', 'checkpoint_' + str(load_model_step) + '.pt'),
        model,
        optimizer=optimizer,
        ema=ema,
    )

if wandb_track:
    import wandb
    wandb.init(project='pi_diffusion', name=name)
    log_fn = wandb.log
else:
    log_fn = noop
log_freq = int(config.get('log_freq', 20))
    
output_save_dir = f'./trained_models/{name}'
os.makedirs(output_save_dir, exist_ok=True)
# Store training metrics by iteration for plotting.
_training_metrics_csv = Path(output_save_dir) / 'training_metrics.csv'


def _safe_psnr(gt, pred):
    data_range = float(np.max(gt) - np.min(gt))
    if data_range <= 1e-12:
        mse = float(np.mean((gt - pred) ** 2))
        return float('inf') if mse <= 1e-12 else float('nan')
    return float(calculate_psnr(gt, pred))


def _safe_ssim(gt, pred):
    data_range = float(np.max(gt) - np.min(gt))
    if data_range <= 1e-12:
        return 1.0
    return float(calculate_ssim(gt, pred))


def compute_reconstruction_metrics(gt_tensor, pred_tensor):
    gt_np = gt_tensor.detach().cpu().numpy()
    pred_np = pred_tensor.detach().cpu().numpy()
    rows = []
    for idx in range(gt_np.shape[0]):
        gt = gt_np[idx]
        pred = pred_np[idx]
        mse_val = float(np.mean((gt - pred) ** 2))
        psnr_val = _safe_psnr(gt, pred)
        ssim_val = _safe_ssim(gt, pred)
        rows.append({
            'Sample Index': idx,
            'MSE': mse_val,
            'PSNR': psnr_val,
            'SSIM': ssim_val,
        })

    return rows, {
        'MSE': float(np.mean([r['MSE'] for r in rows])),
        'PSNR': float(np.mean([r['PSNR'] for r in rows])),
        'SSIM': float(np.mean([r['SSIM'] for r in rows])),
    }


def reconstruct_validation_batch(batch):
    batch = batch.to(device)
    t_eval = torch.zeros(batch.shape[0], dtype=torch.long, device=batch.device)

    if gov_eqs in ('darcy', 'turbulent'):
        x0 = batch
        model_input = (image_to_b_xy_c(x0), t_eval)
        residual_input = (model_input, )
    elif gov_eqs == 'poisson':
        rho = batch[:, 0:1]  # [B, 1, H, W]
        x0 = batch[:, 1:2]   # [B, 1, H, W]
        x_cond = torch.cat([x0, rho], dim=1)
        model_input = (image_to_b_xy_c(x_cond), t_eval)
        residual_input = (model_input, rho)
    elif gov_eqs == 'mechanics':
        conditioning, x0, bcs = torch.tensor_split(batch, (3, 6), dim=1)
        vf = conditioning[:, 0, 0, 0]
        model_input = (image_to_b_xy_c(x0), t_eval)
        residual_input = (model_input, bcs, vf, x0)
    else:
        raise ValueError(f'Unsupported gov_eqs for validation reconstruction: {gov_eqs}')

    out_dict = residuals.compute_residual(
        residual_input,
        reduce='none',
        return_model_out=True,
        return_optimizer=False,
        return_inequality=False,
        ddim_func=diffusion_utils.ddim_sample_x0,
    )
    pred = out_dict['model_out']
    if len(pred.shape) == 3:
        pred = b_xy_c_to_image(pred)
    return x0.detach(), pred.detach()

pbar = tqdm(range(start_iteration, train_iterations + 1))
for iteration in pbar:
    model.train()
    cur_batch = next(dl).to(device)
    loss, data_loss, residual_loss, ineq_loss, opt_loss, projection_loss = diffusion_utils.model_estimation_loss(
                cur_batch, residual_func = residuals, c_data = c_data, c_residual = c_residual,
                c_ineq = c_ineq, lambda_opt = lambda_opt, use_projection_heads = use_projection_heads, 
                c_projection = c_projection)    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    optimizer.step()        
    # logging
    if iteration % log_freq == 0:
        pbar.set_description(f'training loss: {loss.item():.3e}')
        log_fn({'loss': loss.item()}, step=iteration)
        log_fn({'loss_data': data_loss}, step=iteration)
        log_fn({'residual_mean_abs': residual_loss}, step=iteration)
        if c_ineq > 0:
            log_fn({'loss_inequality': ineq_loss}, step=iteration)
        if lambda_opt > 0:
            log_fn({'loss_optimization': opt_loss}, step=iteration)
        if use_projection_heads and c_projection > 0:
            log_fn({'loss_projection': projection_loss}, step=iteration)

        _tm_row = {
            'iteration': iteration,
            'loss': float(loss.item()),
            'loss_data': float(data_loss),
            'residual_mean_abs': float(residual_loss),
            'loss_inequality': float(ineq_loss),
            'loss_optimization': float(opt_loss),
            'loss_projection': (
                float(projection_loss) if (use_projection_heads and c_projection > 0) else float('nan')
            ),
        }
        _write_header = not _training_metrics_csv.exists()
        with open(_training_metrics_csv, 'a', newline='') as _tm_f:
            _w = csv.DictWriter(_tm_f, fieldnames=list(_tm_row.keys()))
            if _write_header:
                _w.writeheader()
            _w.writerow(_tm_row)
    
    # ema update
    if iteration > ema_start:
        ema.update(model)

    # evaluation on validation set
    if iteration % test_eval_freq == 0 and exists(dl_valid):
        model.eval()
        using_ema = use_ema_for_eval and iteration > ema_start
        if using_ema:
            ema.ema(model)
        cur_test_batch = next(dl_valid).to(device)
        # NOTE: we do not use torch.no_grad() since we may require residual gradient for classifier-free guidance
        loss_test, data_loss_test, residual_loss_test, ineq_loss_test, opt_loss_test, projection_loss_test = diffusion_utils.model_estimation_loss(
                    cur_test_batch, residual_func = residuals, c_data = c_data, c_residual = c_residual,
                    c_ineq = c_ineq, lambda_opt = lambda_opt, use_projection_heads = use_projection_heads, 
                    c_projection = c_projection)

        grad_context = torch.enable_grad if residual_grad_guidance else torch.no_grad
        with grad_context():
            gt_eval, pred_eval = reconstruct_validation_batch(cur_test_batch)
        metric_rows, metric_means = compute_reconstruction_metrics(gt_eval, pred_eval)
        output_save_dir_step = output_save_dir + f'/training/step_{iteration}/'
        os.makedirs(output_save_dir_step, exist_ok=True)
        metric_rows.append({
            'Sample Index': 'Mean',
            'MSE': metric_means['MSE'],
            'PSNR': metric_means['PSNR'],
            'SSIM': metric_means['SSIM'],
        })
        pd.DataFrame(metric_rows).to_csv(
            os.path.join(output_save_dir_step, 'validation_metrics.csv'),
            index=False
        )

        print(f'test loss at iteration {iteration}: {loss_test:.3e}')
        print(
            f'validation metrics at iteration {iteration}: '
            f'MSE={metric_means["MSE"]:.6e}, '
            f'PSNR={metric_means["PSNR"]:.4f}, '
            f'SSIM={metric_means["SSIM"]:.4f}'
        )
        log_fn({'loss_test': loss_test.item()}, step=iteration)
        log_fn({'loss_data_test': data_loss_test}, step=iteration)
        log_fn({'residual_mean_abs_test': residual_loss_test}, step=iteration)
        log_fn({'mse_test': metric_means['MSE']}, step=iteration)
        log_fn({'psnr_test': metric_means['PSNR']}, step=iteration)
        log_fn({'ssim_test': metric_means['SSIM']}, step=iteration)
        if c_ineq > 0:
            log_fn({'loss_inequality_test': ineq_loss_test}, step=iteration)
        if lambda_opt > 0:
            log_fn({'loss_optimization_test': opt_loss_test}, step=iteration)
        if use_projection_heads and c_projection > 0:
            log_fn({'loss_projection_test': projection_loss_test}, step=iteration)
        if using_ema:
            ema.restore(model)
        model.train()

     # generate and evaluate samples
    if ((iteration % sample_freq == 0) or (iteration == train_iterations)):        
        using_ema = use_ema_for_eval and iteration > ema_start
        if using_ema:
            ema.ema(model)

        if gov_eqs == 'darcy':
            conditioning_input = None
            n_samples_this_step = no_samples
            sample_shape = (n_samples_this_step, output_dim, pixels_per_dim, pixels_per_dim)
        elif gov_eqs == 'poisson':
            cur_batch = next(dl_valid).to(device)
            n_samples_this_step = min(no_samples, cur_batch.shape[0])
            sample_shape = (n_samples_this_step, output_dim, pixels_per_dim, pixels_per_dim)
            cur_batch = cur_batch[torch.randperm(cur_batch.shape[0], device=device)[:n_samples_this_step]]
            rho_samples = cur_batch[:, 0:1]
            conditioning_input = rho_samples
        elif gov_eqs == 'turbulent':
            conditioning_input = None
            n_samples_this_step = no_samples
            sample_shape = (n_samples_this_step, output_dim, pixels_per_dim, pixels_per_dim)
        elif gov_eqs == 'mechanics':
            cur_batch = next(dl_valid).to(device)
            n_samples_this_step = min(no_samples, cur_batch.shape[0])
            sample_shape = (n_samples_this_step, output_dim, pixels_per_dim+1, pixels_per_dim+1)
            cur_batch = cur_batch[torch.randperm(cur_batch.shape[0], device = device)[:n_samples_this_step]]
            conditioning, x_0, bcs = torch.tensor_split(cur_batch, (3, 6), dim=1)
            conditioning_input = (conditioning, bcs, x_0)            
            # save conditioning data for later evaluation
            cond_data = torch.cat((conditioning, x_0, bcs), dim=1)
            for cur_sample in range(n_samples_this_step):
                for channel_idx in range(cond_data.shape[1]):
                    os.makedirs(output_save_dir + f'/training/step_{iteration}/sample_{cur_sample}', exist_ok=True)
                    np.savetxt(output_save_dir + f'/training/step_{iteration}/sample_{cur_sample}/cond_channel_{channel_idx}.csv', cond_data[cur_sample, channel_idx].detach().cpu().numpy(), delimiter=',')

        output = diffusion_utils.p_sample_loop(conditioning_input, sample_shape, 
                                save_output=save_output, surpress_noise=True, 
                                use_dynamic_threshold=use_dynamic_threshold, 
                                residual_func=residuals, eval_residuals = eval_residuals, 
                                return_optimizer = return_optimizer, return_inequality = return_inequality,
                                M_correction = M_correction, N_correction = N_correction, correction_mode = correction_mode,
                                collect_physics_per_step = collect_physics_per_step)
        
        if eval_residuals:
            seqs = output[0]
            residual = output[1]['residual']
            residual = residual.abs().mean(dim=tuple(range(1, residual.ndim))) # reduce to batch dim
            if return_optimizer:
                optimized_quant = output[1]['optimized_quant']
            if return_inequality:
                ineq = output[1]['inequality_quant']
        else:
            seqs = output
            
        output_save_dir_step = output_save_dir + f'/training/step_{iteration}/'
        os.makedirs(output_save_dir_step, exist_ok=True)
                
        labels = ['sample', 'model_output']
        for seq_idx, seq in enumerate(seqs):

            # NOTE: We here only evaluate the sample at the final timestep and skip model_output as this is identical (since no noise is applied in last step).
            if seq_idx == 1:
                continue

            seq = torch.stack(seq, dim=0)

            if len(seq.shape) == 6:
                seq = seq.squeeze(-3)
                
            last_preds = seq[-1].numpy()
            sel_samples = np.arange(len(last_preds))
            channels = np.arange(output_dim)

            for sel_sample in sel_samples:
                for sel_channel in channels:
                    last_pred = last_preds[sel_sample, sel_channel]
                    denom = last_pred.max() - last_pred.min()
                    if denom < 1e-12:
                        last_pred_normalized = np.zeros_like(last_pred)
                    else:
                        last_pred_normalized = (last_pred - last_pred.min()) / denom

                    image = np.uint8(last_pred_normalized * 255)
                    fig, ax = plt.subplots()
                    ax.imshow(image, cmap='gray', vmin=0, vmax=255)
                    ax.axis('off')
                    if eval_residuals:
                        title = f'eq: {residual[sel_sample]:.2e}'
                        if return_optimizer:
                            title += f'\nopt: {optimized_quant[sel_sample]:.2f}'
                        if return_inequality:
                            title += f'\nineq: {ineq[sel_sample]:.2e}'
                        plt.title(title, color='green')
                    filename = labels[seq_idx] + '_sample_' + str(sel_sample) + '_' + str(sel_channel) + '.png'
                    plt.savefig(output_save_dir_step + filename, bbox_inches='tight', pad_inches=0)
                    plt.close(fig)

                    os.makedirs(output_save_dir_step + f'/sample_{sel_sample}/', exist_ok=True)
                    np.savetxt(output_save_dir_step + f'/sample_{sel_sample}/' + labels[seq_idx] + '_' + str(sel_channel) + '.csv', last_pred, delimiter=',')

                    if create_gif:
                        sel_seq = seq[:, sel_sample, sel_channel].detach().cpu().numpy()
                        image_array_to_gif(sel_seq, output_save_dir_step + f'/sample_{sel_sample}/' + labels[seq_idx] + '_' + str(sel_channel) + '.gif')


        if eval_residuals:
            residuals_array = residual.detach().cpu().numpy()
            ineq_array = ineq.detach().cpu().numpy() if return_inequality else None
            optimized_quant_array = optimized_quant.detach().cpu().numpy() if return_optimizer else None

            # logging
            log_fn({'residual_mean_abs_samples': np.nanmean(residuals_array)}, step=iteration)
            log_fn({'residual_median_abs_samples': np.nanmedian(residuals_array)}, step=iteration)
            df_data = {'Sample Index': list(range(n_samples_this_step)) + ['Mean'],
                    'Residuals (abs)': list(residuals_array)}
            if return_inequality:
                df_data['Inequality'] = list(ineq_array)
            if return_optimizer:
                df_data['Optimized quantity'] = list(optimized_quant_array)
            df_data['Residuals (abs)'].append(np.nanmean(residuals_array))
            if return_optimizer:
                df_data['Optimized quantity'].append(np.nanmean(optimized_quant_array))
            if return_inequality:
                df_data['Inequality'].append(np.nanmean(ineq_array))
            df = pd.DataFrame(df_data)
            csv_path = os.path.join(output_save_dir_step, 'sample_statistics.csv')
            df.to_csv(csv_path, index=False)
            if collect_physics_per_step:
                traj = output[1].get('physics_trajectory')
                if traj:
                    pd.DataFrame(traj).to_csv(
                        os.path.join(output_save_dir_step, 'physics_trajectory_denoising.csv'),
                        index=False,
                    )

        if topopt_eval and gov_eqs == 'mechanics':
            log_fn({'rel_CE_error': np.nanmean(output[1]['rel_CE_error_full_batch'].detach().cpu().numpy())}, step=iteration)
            log_fn({'rel_vf_error': np.nanmean(output[1]['vf_error_full_batch'].detach().cpu().numpy())}, step=iteration)
            log_fn({'fm_error': np.nanmean(output[1]['fm_error_full_batch'].detach().cpu().numpy())}, step=iteration)

        if using_ema:
            ema.restore(model)

        if iteration > 0:
            save_model(config, model, iteration, output_save_dir, optimizer=optimizer, ema=ema)

    # ema.restore(residuals.model)

    # Save periodic checkpoints.
    if iteration > 0 and checkpoint_save_freq > 0 and iteration % checkpoint_save_freq == 0:
        save_model(config, model, iteration, output_save_dir, optimizer=optimizer, ema=ema)
        print(f'Model saved at iteration {iteration} to {output_save_dir} (every {checkpoint_save_freq} steps)')

# Save the final model.
save_model(config, model, train_iterations, output_save_dir, optimizer=optimizer, ema=ema)
print(f'Final model saved to {output_save_dir}')

if wandb_track:
    wandb.finish()