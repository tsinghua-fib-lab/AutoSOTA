from warnings import simplefilter

import matplotlib.pyplot as plt
import torch

from model import ExponentialMovingAverage, get_rope_vit_model
from utils import Dict2Obj, plot_bias_analysis_2d, plot_loss_ess
from utils_potts import potts2d_ham, potts2d_magnetization_all
from utils_train import save_checkpoint, train

simplefilter(action='ignore', category=FutureWarning)
import argparse
import json
import os
import sys
from pathlib import Path
from pprint import pformat

import numpy as np
import wandb

from bias import BiasPotential, BiasPotentialMultiDim

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default="cuda:0")
parser.add_argument('--L', type=int, default=16)
parser.add_argument('--q', type=int, default=3)
parser.add_argument('--beta', type=float, default=0.5)
parser.add_argument('--J', type=float, default=1)
parser.add_argument('--dir_name', type=str, default=None)
parser.add_argument('--num_epochs', type=int, default=50000)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--eval_every', type=int, default=20)
parser.add_argument('--eval_batch_size', type=int, default=32)
parser.add_argument('--use_anneal', action='store_true')
parser.add_argument('--anneal_beta', type=float, default=None)
parser.add_argument('--anneal_epochs', type=int, default=None)
parser.add_argument('--resume_from_ckpt', type=str, default=None)
parser.add_argument('--loss_fn', type=str, default='wdce', help='Loss function: wdce or mse')
parser.add_argument('--resample_every_n_step', type=int, default=10, help='Resample every n step')
parser.add_argument('--wdce_num_replicates', type=int, default=8, help='WDCE number of replicates')
# Metadynamics / Bias Potential
parser.add_argument('--use_bias', action='store_true', help='Enable biased sampling (WT-ASBS)')
parser.add_argument('--bias_sigma', type=str, default="0.05", help='Sigma for Gaussian bias kernel (can be list)')
parser.add_argument('--bias_height', type=float, default=0.1, help='Initial bias height')
parser.add_argument('--bias_factor', type=float, default=10.0, help='Bias factor (gamma) for Well-Tempered Metadynamics')
parser.add_argument('--bias_grid_size', type=str, default="100", help='Grid size (can be list)')
parser.add_argument('--kernel_type', type=str, default='gaussian', help='Kernel type: gaussian or delta')
parser.add_argument('--cv_min', type=str, default="-0.6,-1.0", help="Min CV bounds (comma-separated list of floats, supports N-dimensions)")
parser.add_argument('--cv_max', type=str, default="1.1,1.0", help="Max CV bounds (comma-separated list of floats, supports N-dimensions)")
parser.add_argument('--scale_bias_with_size', action='store_true', help='Scale bias Delta_T with system size')
parser.add_argument('--no_normalize_bias_by_batch', dest='normalize_bias_by_batch', action='store_false', default=True, help="Disable bias height norm by batch size")
parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
parser.add_argument('--save_every', type=int, default=10000, help='Save checkpoint every N steps')
parser.add_argument('--save_best_metric', type=str, default='none', choices=['loss', 'none'],
                    help='Metric to track for best checkpoint (loss or none)')
parser.add_argument('--adaptive_sigma', action='store_true',
                    help='Enable adaptive sigma scheduling for bias kernel width')
parser.add_argument('--sigma_start', type=str, default=None,
                    help='Initial sigma for adaptive scheduling (comma-separated, default: 2x initial sigma)')
parser.add_argument('--sigma_end', type=str, default=None,
                    help='Final sigma for adaptive scheduling (comma-separated, default: 0.4x initial sigma)')
parser.add_argument('--buffer_size', type=int, default=0, help='Replay buffer size')
parser.add_argument('--buffer_ratio', type=float, default=0.0, help='Ratio of buffer samples in batch')
parser.add_argument('--buffer_n_bins', type=int, default=1, help='Number of bins for CV-based Replay Buffer')
parser.add_argument('--buffer_strategy', type=str, default='fifo', choices=['fifo', 'balanced'], 
                    help='Buffer storage strategy: fifo or balanced')
parser.add_argument('--wandb', dest='use_wandb', action='store_true', help="Enable wandb logging")
parser.add_argument('--no-wandb', dest='use_wandb', action='store_false', help="Disable wandb logging")
parser.set_defaults(use_wandb=False)
parser.add_argument('--wandb_project', type=str, default='mdns-potts', help="wandb project name")
parser.add_argument('--wandb_run_name', type=str, default=None, help="wandb run name")
parser.add_argument('--wandb_mode', type=str, default='online', choices=['offline', 'online', 'disabled'],
                    help="wandb logging mode: 'online' (default), 'offline' (for HPC without internet), or 'disabled'")
# Workaround for argparse negative number issue with cv_min/cv_max
def preprocess_args(argv):
    new_argv = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == '--cv_min' or arg == '--cv_max':
            if i + 1 < len(argv):
                val = argv[i+1]
                if val.startswith('-'):
                    # Check if it looks like a number (or list of numbers)
                    try:
                        parts = val.split(',')
                        float(parts[0])
                        # It is a number, combine with = to avoid argparse flag confusion
                        new_argv.append(f'{arg}={val}')
                        i += 2
                        continue
                    except ValueError:
                        pass
        new_argv.append(arg)
        i += 1
    return new_argv

if len(sys.argv) > 1:
    sys.argv = [sys.argv[0]] + preprocess_args(sys.argv[1:])

args = parser.parse_args()

if args.use_anneal:
    assert args.anneal_beta is not None, "anneal_beta must be specified if anneal is True"
    assert args.anneal_epochs is not None, "anneal_epochs must be specified if anneal is True"
    
device = args.device
L = args.L
D = L**2
q = args.q
N = q + 1
beta = args.beta
J = args.J
h = 0
resume_path = args.resume_from_ckpt
anneal_beta = args.anneal_beta
# Handle None dir_name (when script is imported rather than run directly)
if args.dir_name is None:
    dir_name = Path('exp_local/default')
else:
    dir_name = Path(args.dir_name)
dir_name.mkdir(parents=True, exist_ok=True)

def reward_fn_potts(S, beta=0.5, J=1, q=3, **kwargs):
    """Reward function for Potts model. Accepts **kwargs for compatibility."""
    return -beta * potts2d_ham(S, J, q)

cfg = {'tokens': q,
       "anneal": args.use_anneal,
        "anneal_beta": anneal_beta if args.use_anneal else None,
        "anneal_epochs": args.anneal_epochs,
        "resume_from_ckpt": resume_path,
       "L": L, 
       "q": q, 
       "beta": beta, 
       "J": J, 
       "dir_name": args.dir_name,
       'model': {'hidden_size': 128, 'n_blocks': 4, 'n_heads': 4, 'length': D,
                'use_checkpoint': False, 'dtype': 'bfloat16'},
       'num_epochs': args.num_epochs, 
       'resample_every_n_step': args.resample_every_n_step,
       'batch_size': args.batch_size,
       'eval_every': args.eval_every, 'eval_batch_size': args.eval_batch_size,
       'grad_clip': False, 'gradnorm_clip': 1,
       'loss_fn': args.loss_fn,
       'wdce_num_replicates': args.wdce_num_replicates,
       'seed': args.seed,
       'use_bias': args.use_bias,
       'bias_sigma': args.bias_sigma,
       'bias_height': args.bias_height,
       'bias_factor': args.bias_factor,
       'bias_grid_size': args.bias_grid_size,
       'kernel_type': args.kernel_type,
       'scale_bias_with_size': args.scale_bias_with_size,
       'buffer_size': args.buffer_size,
       'buffer_ratio': args.buffer_ratio,
       'cv_min': args.cv_min,
       'cv_max': args.cv_max,
       'save_every': args.save_every,
       'save_best_metric': args.save_best_metric,
       'buffer_n_bins': args.buffer_n_bins,
       'buffer_strategy': args.buffer_strategy,
       'wandb': args.use_wandb,
       'wandb_project': args.wandb_project,
       'wandb_run_name': args.wandb_run_name,
       'wandb_mode': args.wandb_mode}

model = get_rope_vit_model(L, embed_dim=cfg['model']['hidden_size'],
                           depth=cfg['model']['n_blocks'],
                           num_heads=cfg['model']['n_heads'],
                           vocab_size=cfg['tokens'] + 1,
                           dtype=cfg['model']['dtype'],
                           device=device)
ema = ExponentialMovingAverage(model.parameters(), decay=0.9999)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.00)

print('Model: num of params: {}, size: {:.2f} MB'.format(
    sum(p.numel() for p in model.parameters()),
    sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)))

print(f"Training config:\n{pformat(cfg)}")
with open(dir_name / 'config.json', 'w') as f:
    json.dump(cfg, f, indent=4)

wandb_run = None
if args.use_wandb:
    # Check for wandb run ID in environment variable (set by resume_training.py)
    wandb_run_id = os.environ.get("WANDB_RUN_ID", None)
    wandb_resume = os.environ.get("WANDB_RESUME", "never")
    
    # Calculate start_epoch from checkpoint if resuming (needed for wandb step)
    wandb_start_step = None
    if wandb_run_id and resume_path is not None:
        try:
            checkpoint = torch.load(resume_path, map_location='cpu')
            losses = checkpoint.get('losses', [])
            wandb_start_step = len(losses) if losses else 0
        except Exception as e:
            print(f"Warning: Could not determine start step from checkpoint: {e}")
    
    wandb_mode = args.wandb_mode if args.wandb_mode != "disabled" else "disabled"
    wandb_init_kwargs = {
        "project": args.wandb_project,
        "name": args.wandb_run_name or args.dir_name,
        "dir": str(dir_name),
        "config": cfg,
        "mode": wandb_mode,
    }
    
    # If resuming, add id and resume parameters
    if wandb_run_id:
        wandb_init_kwargs["id"] = wandb_run_id
        wandb_init_kwargs["resume"] = wandb_resume
        print(f"Resuming wandb run with ID: {wandb_run_id}")
        if wandb_start_step is not None:
            print(f"Starting wandb logging from step: {wandb_start_step}")
    
    wandb_run = wandb.init(**wandb_init_kwargs)
    
    # Note: wandb automatically handles step tracking when resuming
    if wandb_start_step is not None:
        print(f"Note: All logging will use step numbers starting from {wandb_start_step}")
    
if not args.use_anneal:
    start_epoch = 0
    if resume_path is not None:
        print("Loading checkpoint from: ", resume_path)
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        ema.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        losses = checkpoint['losses']
        ess_train = checkpoint['ess_train']
        ess_eval = checkpoint['ess_eval']
        bias_state = checkpoint.get('bias_potential', None)
        # Calculate starting epoch from checkpoint
        start_epoch = len(losses) if losses else 0
        print(f"Resuming from epoch {start_epoch}")
    else:
        print("No checkpoint provided, starting from scratch")
        losses = []
        ess_train = []
        ess_eval = []
        bias_state = None

    # Initialize BiasPotential
    bias_pot = None
    if args.use_bias:
        T_val = 1.0 / args.beta # Usually single beta for bias run

        # Parse list arguments
        def parse_list(arg, dtype=float):
            if isinstance(arg, (int, float)): return [dtype(arg)]
            return [dtype(x) for x in arg.split(',')]
            
        cv_min = parse_list(args.cv_min, float)
        cv_max = parse_list(args.cv_max, float)
        grid_size = parse_list(args.bias_grid_size, int)
        sigma = parse_list(args.bias_sigma, float)
        
        # Energy scaling
        # For Potts, H max is approx 2 * L^2 (if J=1). 
        # Standardizing against L=4 (D=16) like Ising/CuAu
        energy_scaling_val = float(D) / 16.0 if args.scale_bias_with_size else 1.0
        
        # Bias height normalization for diffusion samplers
        effective_bias_height = args.bias_height
        if args.normalize_bias_by_batch:
            effective_bias_height = args.bias_height / args.batch_size
            total_bias_per_cycle = effective_bias_height * args.batch_size
            kBT_ratio = args.bias_height / T_val if T_val > 0 else 0
            print(f"Bias height normalization: {args.bias_height:.6f} -> {effective_bias_height:.8f} per hill")
            print(f"  (Total per cycle: {total_bias_per_cycle:.6f} = {kBT_ratio:.3f} kBT, batch_size={args.batch_size})")
        else:
            print(f"Bias height (no normalization): {effective_bias_height:.6f} per hill")
        
        print(f"Initializing BiasPotential: sigma={args.bias_sigma}, height={effective_bias_height:.8f}, gamma={args.bias_factor}, type={args.kernel_type}")
        print(f"Energy scaling factor: {energy_scaling_val} (D={D})")
        
        bias_pot = BiasPotentialMultiDim(
            cv_min=cv_min, cv_max=cv_max,
            grid_size=grid_size,
            sigma=sigma,
            initial_height=effective_bias_height,
            bias_factor=args.bias_factor,
            T=1.0/beta, # Temperature ~ 1/beta
            kernel_type=args.kernel_type,
            device=device,
            energy_scaling=energy_scaling_val,
            sigma_schedule='adaptive' if args.adaptive_sigma else 'fixed',
            sigma_start=args.sigma_start,
            sigma_end=args.sigma_end,
        )
        if resume_path is not None and bias_state is not None:
             print("Loading BiasPotential from checkpoint...")
             bias_pot.load_state_dict(bias_state)
             
    # Potts CV Computation
    def compute_cv_potts(x):
        # x: [B, D] tokens 0..q-1
        # Convert to one-hot or just count 
        # We need fractions of states 1..3 for q=3?
        # User requested 2D projection for q=3
        
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = x
            
        # Reshape to B, D
        if x_np.ndim == 1:
            x_np = x_np.reshape(1, -1)
        
        B, D_ = x_np.shape
        
        # Count frequencies for each state 0..q-1
        # counts: [B, q]
        counts = np.zeros((B, q))
        for i in range(q):
            counts[:, i] = np.sum(x_np == i, axis=1)
            
        concentrations = counts / D_ # [B, q]
        
        if q == 3:
            # Projection:
            # x = c1 - 0.5 * (c2 + c3)
            # y = (sqrt(3)/2) * (c2 - c3)
            # Assuming states are 0, 1, 2. Map to user's "State 1, 2, 3"?
            # User says: State 1 (c1) -> (1,0). 
            # If we map token 0->State 1, token 1->State 2, token 2->State 3.
            # Then c_tokens = concentrations
            
            c1 = concentrations[:, 0]
            c2 = concentrations[:, 1]
            c3 = concentrations[:, 2]
            
            proj_x = c1 - 0.5 * (c2 + c3)
            proj_y = (np.sqrt(3)/2) * (c2 - c3)
            
            # Stack [B, 2]
            cv = np.stack([proj_x, proj_y], axis=1)
            return torch.tensor(cv, device=device, dtype=torch.float32)
        else:
            # Fallback for q!=3: return first q-1 concentrations
            return torch.tensor(concentrations[:, :-1], device=device, dtype=torch.float32)

    # Reward Wrapper
    def reward_fn_potts_biased(x, beta=0.5, J=1, q=3, use_bias=True, bias_potential=None, cv_compute_fn=None):
        r = reward_fn_potts(x, beta, J, q)
        if use_bias and bias_potential is not None:
            s = cv_compute_fn(x)
            v = bias_potential.evaluate(s)
            
            # r = -beta * H. 
            # We want biased distribution P_bias ~ exp(-beta*H + beta*V) = exp(-beta(H-V))?
            # Standard metadynamics: H_bias = H + V(s).
            # Reweighted: H' = H + V. 
            # MDNS reward is log_prob ~ -beta * Energy.
            # So reward should be -beta * (H + V) = -beta*H - beta*V.
            # r is -beta*H. So subtract beta*V.
            
            if isinstance(beta, (int, float)):
                beta_val = beta
            else:
                beta_val = beta
                
            r = r - beta_val * v
        return r

    model.train()
    
    # Curry the reward function 
    default_beta_potts = beta
    def reward_fn_train(x, beta=None, h=None, J=J, use_bias=None, **kwargs):
        """Reward function that accepts use_bias for compatibility with _compute_log_stats."""
        beta_val = beta if beta is not None else default_beta_potts
        use_bias_val = use_bias if use_bias is not None else args.use_bias
        return reward_fn_potts_biased(
            x, beta=beta_val, J=J, q=q, 
            use_bias=use_bias_val, 
            bias_potential=bias_pot, 
            cv_compute_fn=compute_cv_potts
        )

    model, optimizer, ema, losses, ess_train, ess_eval = train(
        model, optimizer, reward_fn_train, 
        Dict2Obj(cfg), device, ema=ema, num_epochs=args.num_epochs,
        losses=losses, ess_train=ess_train, ess_eval=ess_eval,
        wandb_run=wandb_run, L=L, bias_potential=bias_pot, cv_compute_fn=compute_cv_potts,
        buffer_size=args.buffer_size, buffer_ratio=args.buffer_ratio,
        buffer_n_bins=args.buffer_n_bins, buffer_strategy=args.buffer_strategy,
        save_dir=dir_name, cfg_dict=cfg,
        plot_bias_fn=plot_bias_analysis_2d,
        save_best_metric=args.save_best_metric,
        start_epoch=start_epoch) # Pass start_epoch for proper progress bar
    
    fig, ax = plot_loss_ess(losses, ess_train, ess_eval=ess_eval)
    plt.savefig(f"{dir_name}/loss_ess.png")
    save_checkpoint(model, optimizer, ema, losses, ess_train, ess_eval, cfg, f'{dir_name}/weights.pth', bias_pot)
else:
    model.train()
    def reward_fn_anneal(x, beta=None, h=None, J=J, use_bias=None, **kwargs):
        """Reward function for annealing phase."""
        beta_val = beta if beta is not None else args.anneal_beta
        return reward_fn_potts(x, beta=beta_val, J=J, q=q, **kwargs)
    model, optimizer, ema, losses, ess_train, ess_eval = train(
            model, optimizer, reward_fn_anneal, 
            Dict2Obj(cfg), device, num_epochs=args.anneal_epochs, ema=ema,
            wandb_run=wandb_run, L=L, save_dir=dir_name, cfg_dict=cfg)
    fig, ax = plot_loss_ess(losses, ess_train, ess_eval=ess_eval)
    plt.savefig(f"{dir_name}/loss_ess_anneal.png")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'ema_state_dict': ema.state_dict(),
        'losses': losses, 'ess_train': ess_train, 
        'ess_eval': ess_eval,
        'cfg': cfg}, f'{dir_name}/weights_warmup.pth')
    
    default_beta_main = beta
    def reward_fn_main(x, beta=None, h=None, J=J, use_bias=None, **kwargs):
        """Reward function for main training phase."""
        beta_val = beta if beta is not None else default_beta_main
        return reward_fn_potts(x, beta=beta_val, J=J, q=q, **kwargs)
    model, optimizer, ema, losses, ess_train, ess_eval = train(
        model, optimizer, reward_fn_main, 
        Dict2Obj(cfg), device, num_epochs=args.num_epochs, 
        ema=ema, losses=losses, ess_train=ess_train, ess_eval=ess_eval,
        wandb_run=wandb_run, L=L, save_dir=dir_name, cfg_dict=cfg,
        buffer_size=args.buffer_size, buffer_ratio=args.buffer_ratio,
        buffer_n_bins=args.buffer_n_bins, buffer_strategy=args.buffer_strategy)
    fig, ax = plot_loss_ess(losses, ess_train, ess_eval=ess_eval)
    plt.savefig(f"{dir_name}/loss_ess.png")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'ema_state_dict': ema.state_dict(),
        'losses': losses, 'ess_train': ess_train, 
        'ess_eval': ess_eval,
        'cfg': cfg}, f'{dir_name}/weights_final.pth')

if wandb_run is not None:
    wandb_run.finish()
