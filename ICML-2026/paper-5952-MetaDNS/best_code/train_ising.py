from warnings import simplefilter

import matplotlib.pyplot as plt
import numpy as np
import torch

from model import ExponentialMovingAverage, get_rope_vit_model
from utils import Dict2Obj, plot_loss_ess
from utils_ising import ising2d_ham, ising2d_mag, reward_fn_ising
from utils_train import save_checkpoint, train

simplefilter(action='ignore', category=FutureWarning)
import argparse
import json
import os
from pathlib import Path
from pprint import pformat

import wandb

from bias import BiasPotential
from utils_ising import ising2d_mag

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default="cuda:0")
parser.add_argument('--L', type=int, default=24)
parser.add_argument('--beta', type=float, default=0.28)
parser.add_argument('--J', type=float, default=1)
parser.add_argument('--dir_name', type=str, default=None)
parser.add_argument('--num_epochs', type=int, default=100000)
parser.add_argument('--use_anneal', action='store_true')
parser.add_argument('--anneal_beta', type=float, default=None)
parser.add_argument('--anneal_epochs', type=int, default=None)
parser.add_argument('--resample_every_n_step', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--eval_batch_size', type=int, default=32)
parser.add_argument('--eval_every', type=int, default=20)
parser.add_argument('--loss_fn', type=str, default='wdce')
parser.add_argument('--wdce_num_replicates', type=int, default=8)
parser.add_argument('--resume_from_ckpt', type=str, default=None)
parser.add_argument('--wandb', dest='use_wandb', action='store_true', help="Enable wandb logging")
parser.add_argument('--no-wandb', dest='use_wandb', action='store_false', help="Disable wandb logging")
parser.set_defaults(use_wandb=False)
parser.add_argument('--wandb_project', type=str, default='mdns-ising', help="wandb project name")
parser.add_argument('--wandb_run_name', type=str, default=None, help="wandb run name")
parser.add_argument('--wandb_mode', type=str, default='online', choices=['offline', 'online', 'disabled'],
                    help="wandb logging mode: 'online' (default), 'offline' (for HPC without internet), or 'disabled'")
parser.add_argument('--temps', type=float, nargs='+', default=None,
                    help='List of temperatures (overrides --beta). If provided, batch_size must be divisible by num_temps * num_fields.')
parser.add_argument('--fields', type=float, nargs='+', default=None,
                    help='List of field values. If provided, batch_size must be divisible by num_temps * num_fields.')
parser.add_argument('--sample_delta_temp', action='store_true',
                    help='Sample temperatures dynamically around provided temps')
parser.add_argument('--sample_delta_field', action='store_true',
                    help='Sample fields dynamically around provided fields')
parser.add_argument('--delta_temp', type=float, default=0.2,
                    help='Temperature sampling range for dynamic sampling')
parser.add_argument('--delta_field', type=float, default=0.05,
                    help='Field sampling range for dynamic sampling')
parser.add_argument('--min_temp', type=float, default=1.667,
                    help='Minimum temperature for dynamic sampling')
parser.add_argument('--max_temp', type=float, default=3.5714,
                    help='Maximum temperature for dynamic sampling')
parser.add_argument('--use_bias', action='store_true', help='Enable biased sampling (WT-ASBS)')
parser.add_argument('--bias_sigma', type=float, default=0.05, help='Sigma for Gaussian bias kernel')
parser.add_argument('--bias_height', type=float, default=0.1, 
                    help='Initial bias height. For diffusion samplers with batch updates, this is normalized by batch_size by default (see --no_normalize_bias_by_batch)')
parser.add_argument('--bias_factor', type=float, default=10.0, help='Bias factor (gamma) for Well-Tempered Metadynamics')
parser.add_argument('--bias_grid_size', type=int, default=100, help='Grid size for CV (Magnetization)')
parser.add_argument('--kernel_type', type=str, default='gaussian', help='Kernel type: gaussian or delta')
parser.add_argument('--cv_min', type=float, default=-1.0, help='Minimum value for CV')
parser.add_argument('--cv_max', type=float, default=1.0, help='Maximum value for CV')
parser.add_argument('--no_normalize_bias_by_batch', dest='normalize_bias_by_batch', action='store_false', default=True,
                    help='Disable normalization of bias_height by batch_size (default: normalization enabled). Recommended for diffusion samplers that deposit bias more frequently than traditional MCMC.')
parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
parser.add_argument('--save_every', type=int, default=10000, help='Save checkpoint every N steps')
parser.add_argument('--hidden_size', type=int, default=64, help='Model hidden size')
parser.add_argument('--n_blocks', type=int, default=4, help='Number of transformer blocks')
parser.add_argument('--n_heads', type=int, default=4, help='Number of attention heads')
parser.add_argument('--dtype', type=str, default='bfloat16', help='Model data type')
parser.add_argument('--use_checkpoint', action='store_true', help='Use gradient checkpointing')
parser.add_argument('--scale_bias_with_size', action='store_true', help='Scale bias Delta_T with system size (essential for large L)')
parser.add_argument('--buffer_size', type=int, default=0, help='Size of experience replay buffer')
parser.add_argument('--buffer_ratio', type=float, default=0.0, help='Ratio of buffer samples in training batch')
parser.add_argument('--buffer_n_bins', type=int, default=1, help='Number of bins for CV-based Replay Buffer')
parser.add_argument('--buffer_strategy', type=str, default='fifo', choices=['fifo', 'balanced'], 
                    help='Buffer storage strategy: fifo (shared memory) or balanced (partitioned memory)')
args = parser.parse_args()

if args.use_anneal:
    assert args.anneal_beta is not None, "anneal_beta must be specified if anneal is True"
    assert args.anneal_epochs is not None, "anneal_epochs must be specified if anneal is True"
    
device = args.device
L = args.L
D = L**2
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

# Handle multiple temperatures/fields
# If temps provided, use them; otherwise use single beta converted to temperature
temps = args.temps if args.temps is not None else [1.0 / beta]  # Convert beta to temperature
fields = args.fields if args.fields is not None else [0.0]
sample_delta_temp = args.sample_delta_temp if args.sample_delta_temp is not None else False
sample_delta_field = args.sample_delta_field if args.sample_delta_field is not None else False
delta_temp = args.delta_temp if args.delta_temp is not None else 0.2
delta_field = args.delta_field if args.delta_field is not None else 0.05
min_temp = args.min_temp if args.min_temp is not None else 1.667
max_temp = args.max_temp if args.max_temp is not None else 3.5714

cfg = {'tokens': 2,
       "anneal": args.use_anneal,
       "anneal_beta": anneal_beta if args.use_anneal else None,
       "anneal_epochs": args.anneal_epochs,
       "resume_from_ckpt": resume_path,
       "L": L,
       "beta": beta,
       "J": J,
       "dir_name": args.dir_name,
       'model': {'hidden_size': args.hidden_size, 'n_blocks': args.n_blocks, 'n_heads': args.n_heads, 'length': D, 
                 'use_checkpoint': args.use_checkpoint, 'dtype': args.dtype},
       'num_epochs': args.num_epochs,
       'resample_every_n_step': args.resample_every_n_step,
       'batch_size': args.batch_size, 
       'eval_every': args.eval_every,
       'eval_batch_size': args.eval_batch_size,
       'grad_clip': False, 'gradnorm_clip': 1,
       'loss_fn': args.loss_fn,
       'wdce_num_replicates': args.wdce_num_replicates,
       'seed': args.seed,
       'temps': temps,
       'fields': fields,
       'sample_delta_temp': sample_delta_temp,
       'sample_delta_field': sample_delta_field,
       'delta_temp': delta_temp,
       'delta_field': delta_field,
       'min_temp': min_temp,
       'max_temp': max_temp,
       'use_bias': args.use_bias,
       'bias_sigma': args.bias_sigma,
       'bias_height': args.bias_height,
       'bias_factor': args.bias_factor,
       'bias_grid_size': args.bias_grid_size,
       'kernel_type': args.kernel_type,
       'cv_min': args.cv_min,
       'cv_max': args.cv_max,
       'save_every': args.save_every,
       'J': J,
       'scale_bias_with_size': args.scale_bias_with_size,
       'buffer_size': args.buffer_size,
       'buffer_ratio': args.buffer_ratio,
       'buffer_n_bins': args.buffer_n_bins,
       'buffer_strategy': args.buffer_strategy
       }

# Check batch size compatibility if using multiple temps/fields
if len(temps) > 1 or len(fields) > 1:
    num_conditions = len(temps) * len(fields)
    if cfg['batch_size'] % num_conditions != 0:
        bs = cfg['batch_size']
        nt = len(temps)
        nf = len(fields)
        nc = num_conditions
        msg = f"batch_size ({bs}) must be divisible by num_temps ({nt}) * num_fields ({nf}) = {nc}"
        raise ValueError(msg)
    if cfg['eval_batch_size'] % num_conditions != 0:
        raise ValueError(
            f"eval_batch_size ({cfg['eval_batch_size']}) must be divisible by "
            f"num_temps ({len(temps)}) * num_fields ({len(fields)}) = {num_conditions}"
        )

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
            print(f"Note: All logging will use step numbers starting from {wandb_start_step}")
    
    wandb_run = wandb.init(**wandb_init_kwargs)

model = get_rope_vit_model(L, embed_dim=cfg['model']['hidden_size'], 
                          depth=cfg['model']['n_blocks'], 
                          num_heads=cfg['model']['n_heads'], 
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
        current_fields = checkpoint.get('current_fields', None)
        ess_eval = checkpoint['ess_eval']
        current_fields = checkpoint.get('current_fields', None)
        rng = checkpoint.get('rng', None)
        bias_state = checkpoint.get('bias_potential', None)
        buffer_state = checkpoint.get('replay_buffer', None)
        # Calculate starting epoch from checkpoint
        start_epoch = len(losses) if losses else 0
        print(f"Resuming from epoch {start_epoch}")
        if buffer_state is not None:
            print("Found replay buffer state in checkpoint (will load after buffer initialization)")
    else:
        print("No checkpoint provided, starting from scratch")
        losses = []
        ess_train = []
        ess_eval = []
        current_fields = None
        rng = None
        bias_state = None
        buffer_state = None

    # Initialize BiasPotential if enabled
    bias_pot = None
    if args.use_bias:
        T_val = 1.0 / args.beta # Usually single beta for bias run
        # normalize energy by system size w.r.t. 4x4 Ising model
        energy_scaling_val = float(D) / 16 if args.scale_bias_with_size else 1.0
        
        # Normalize bias_height by batch_size for diffusion samplers
        # Traditional metadynamics deposits 1 hill per step with height 0.1-0.5 kBT
        # Diffusion samplers deposit batch_size hills per cycle, so normalize accordingly
        # (Nam et al. 2020: "the Gaussian height h must be reduced, since diffusion 
        #  samplers generate uncorrelated samples and thus deposit bias more frequently")
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
        
        bias_pot = BiasPotential(
            cv_min=args.cv_min, cv_max=args.cv_max, 
            grid_size=args.bias_grid_size,
            sigma=args.bias_sigma,
            initial_height=effective_bias_height,
            bias_factor=args.bias_factor,
            T=T_val,
            kernel_type=args.kernel_type,
            device=device,
            energy_scaling=energy_scaling_val
        )
        if resume_path is not None and bias_state is not None:
             print("Loading BiasPotential from checkpoint...")
             bias_pot.load_state_dict(bias_state)
    model.train()
    # Create reward function wrapper that accepts optional beta/h for per-sample values
    default_beta = beta
    default_h = h
    def reward_fn(x, beta=None, h=None, J=J, **kwargs):
        """Reward function wrapper that handles scalar or per-sample betas/fields."""
        beta_val = beta if beta is not None else default_beta
        h_val = h if h is not None else default_h
        return reward_fn_ising(x, beta=beta_val, J=J, h=h_val)
    
    # Wrap reward function with bias if enabled
    def biased_reward_fn(x, beta=None, h=None, J=J, use_bias=True):
        # 1. Standard reward
        r = reward_fn(x, beta=beta, h=h, J=J)
        
        # 2. Add Bias: R' = R - beta * V(s)
        if bias_pot is not None:
             # Convert x (0,1) to spins (-1,1) for CV calc
             s = ising2d_mag(2*x - 1)
             v = bias_pot.evaluate(s)
             
             # Get beta for scaling
             beta_val = beta if beta is not None else default_beta
             # Handle tensor/scalar beta
             if isinstance(beta_val, (int, float)):
                 beta_tensor = torch.tensor(beta_val, device=x.device)
             elif isinstance(beta_val, torch.Tensor):
                 beta_tensor = beta_val.to(x.device)
             else:
                 beta_tensor = torch.tensor(beta_val, device=x.device) # Fallback
             
             r = r - beta_tensor * v if use_bias else r
        return r

    actual_reward_fn = biased_reward_fn if args.use_bias else reward_fn
    
    # Define CV computation function for Ising
    def compute_cv_ising(x):
        """Compute magnetization CV for Ising model."""
        x_spins = 2 * x - 1  # Convert {0,1} -> {-1,1}
        return ising2d_mag(x_spins)
    
    model, optimizer, ema, losses, ess_train, ess_eval = train(
        model, optimizer, actual_reward_fn, 
        Dict2Obj(cfg), device, ema=ema, num_epochs=args.num_epochs,
        losses=losses, ess_train=ess_train, ess_eval=ess_eval,
        wandb_run=wandb_run, L=L, bias_potential=bias_pot,
        current_fields=current_fields, rng=rng,
        save_dir=dir_name, cfg_dict=cfg,
        cv_compute_fn=compute_cv_ising,
        buffer_size=args.buffer_size, buffer_ratio=args.buffer_ratio,
        buffer_n_bins=args.buffer_n_bins, buffer_strategy=args.buffer_strategy,
        buffer_state_dict=buffer_state,
        start_epoch=start_epoch)
    
    fig, ax = plot_loss_ess(losses, ess_train, ess_eval=ess_eval)
    plt.savefig(f"{dir_name}/loss_ess.png")
    save_checkpoint(model, optimizer, ema, losses, ess_train, ess_eval, cfg, f'{dir_name}/weights.pth', bias_pot)
else:
    bias_pot = None
    model.train()
    # Create reward function for annealing phase
    default_beta_anneal = args.anneal_beta
    default_h_anneal = h
    def reward_fn_anneal(x, beta=None, h=None, J=J, **kwargs):
        beta_val = beta if beta is not None else default_beta_anneal
        h_val = h if h is not None else default_h_anneal
        return reward_fn_ising(x, beta=beta_val, J=J, h=h_val)
    
    model, optimizer, ema, losses, ess_train, ess_eval = train(
            model, optimizer, reward_fn_anneal, 
            Dict2Obj(cfg), device, num_epochs=args.anneal_epochs, ema=ema,
            wandb_run=wandb_run, L=L, save_dir=dir_name, cfg_dict=cfg)
    fig, ax = plot_loss_ess(losses, ess_train, ess_eval=ess_eval)
    plt.savefig(f"{dir_name}/loss_ess_anneal.png")
    save_checkpoint(model, optimizer, ema, losses, ess_train, ess_eval, cfg, f'{dir_name}/weights_warmup.pth', bias_pot)
    
    # Create reward function for main training phase
    default_beta_main = beta
    default_h_main = h
    def reward_fn_main(x, beta=None, h=None, J=J, **kwargs):
        beta_val = beta if beta is not None else default_beta_main
        h_val = h if h is not None else default_h_main
        return reward_fn_ising(x, beta=beta_val, J=J, h=h_val)
    
    model, optimizer, ema, losses, ess_train, ess_eval = train(
        model, optimizer, reward_fn_main, 
        Dict2Obj(cfg), device, num_epochs=args.num_epochs, 
        ema=ema, losses=losses, ess_train=ess_train, ess_eval=ess_eval,
        wandb_run=wandb_run, L=L, save_dir=dir_name, cfg_dict=cfg,
        buffer_size=args.buffer_size,
        buffer_ratio=args.buffer_ratio,
        buffer_n_bins=args.buffer_n_bins,
        buffer_strategy=args.buffer_strategy)
    fig, ax = plot_loss_ess(losses, ess_train, ess_eval=ess_eval)
    plt.savefig(f"{dir_name}/loss_ess.png")
    save_checkpoint(model, optimizer, ema, losses, ess_train, ess_eval, cfg, f'{dir_name}/weights_final.pth', bias_pot)

if wandb_run is not None:
    wandb_run.finish()
