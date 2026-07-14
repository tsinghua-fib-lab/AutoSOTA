import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import wandb
from tqdm import tqdm

from bias import BiasPotentialMultiDim
from potts import LatticePottsModel
from utils import plot_bias_analysis_2d
from utils_potts import potts2d_ham


def parse_args():
    parser = argparse.ArgumentParser(description="MCMC Well-Tempered Metadynamics for Potts Model")
    
    # System Args
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--L', type=int, default=16, help="Lattice size L")
    parser.add_argument('--q', type=int, default=3, help="Number of Potts states")
    parser.add_argument('--beta', type=float, default=0.5, help="Inverse temperature")
    parser.add_argument('--J', type=float, default=1.0, help="Coupling constant")
    parser.add_argument('--dir_name', type=str, default="exp_local/mcmc_potts", help="Output directory")
    
    # MCMC Args
    parser.add_argument('--num_steps', type=int, default=100000, help="Total MCMC steps")
    parser.add_argument('--batch_size', type=int, default=256, help="Number of parallel chains")
    parser.add_argument('--log_every', type=int, default=100)
    parser.add_argument('--save_every', type=int, default=1000)
    
    # Bias Args
    parser.add_argument('--use_bias', action='store_true', help='Enable biased sampling (WT-MetaD)')
    parser.add_argument('--bias_sigma', type=str, default="0.05", help='Sigma for Gaussian bias kernel')
    parser.add_argument('--bias_height', type=float, default=0.1, help='Initial bias height')
    parser.add_argument('--bias_factor', type=float, default=10.0, help='Bias factor (gamma)')
    parser.add_argument('--bias_grid_size', type=str, default="100", help='Grid size')
    parser.add_argument('--kernel_type', type=str, default='gaussian', help='Kernel type: gaussian or delta')
    parser.add_argument('--cv_min', type=str, default="-0.6,-1.0", help="Min CV bounds")
    parser.add_argument('--cv_max', type=str, default="1.1,1.0", help="Max CV bounds")
    parser.add_argument('--scale_bias_with_size', action='store_true', help='Scale bias Delta_T with system size')
    parser.add_argument('--no_normalize_bias_by_batch', dest='normalize_bias_by_batch', action='store_false', default=True, help="Disable bias height norm by batch size")
    parser.add_argument('--update_bias_every', type=int, default=1, help='Update bias every N steps')
    
    # WandB
    parser.add_argument('--wandb', dest='use_wandb', action='store_true', help="Enable wandb")
    parser.add_argument('--no-wandb', dest='use_wandb', action='store_false', help="Disable wandb")
    parser.set_defaults(use_wandb=False)
    parser.add_argument('--wandb_project', type=str, default='mdns-potts-mcmc', help="wandb project name")
    parser.add_argument('--wandb_run_name', type=str, default=None, help="wandb run name")

    # Arguments preprocessing for negative numbers in lists
    def preprocess_args(argv):
        new_argv = []
        i = 0
        while i < len(argv):
            arg = argv[i]
            if arg == '--cv_min' or arg == '--cv_max':
                if i + 1 < len(argv):
                    val = argv[i+1]
                    if val.startswith('-'):
                        try:
                            parts = val.split(',')
                            float(parts[0])
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

    return parser.parse_args()

def compute_cv_potts(x, q, device):
    """
    Compute Potts CVs (Projected Magnetization).
    x: [B, D] or [B, L, L] tensor of spins {0..q-1}
    """
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy()
    else:
        x_np = x
        
    x_np = x_np.reshape(x_np.shape[0], -1) # Flatten to [B, D]
    B, D_ = x_np.shape
    
    # Count frequencies
    counts = np.zeros((B, q))
    for i in range(q):
        counts[:, i] = np.sum(x_np == i, axis=1)
        
    concentrations = counts / D_
    
    if q == 3:
        # Projection for q=3
        c1 = concentrations[:, 0]
        c2 = concentrations[:, 1]
        c3 = concentrations[:, 2]
        
        proj_x = c1 - 0.5 * (c2 + c3)
        proj_y = (np.sqrt(3)/2) * (c2 - c3)
        
        cv = np.stack([proj_x, proj_y], axis=1)
    else:
        # Fallback
        cv = concentrations[:, :-1]
        
    return torch.tensor(cv, device=device, dtype=torch.float32)

class BiasedLatticePottsModel(LatticePottsModel):
    """
    Wraps LatticePottsModel to add bias potential to energy.
    """
    def __init__(self, *args, bias_potential=None, cv_compute_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.bias_potential = bias_potential
        self.cv_compute_fn = cv_compute_fn
    
    def get_energy(self, x, temps=None, fields=None, time=None):
        # Physical energy: H(x)
        # Note: LatticePottsModel.get_energy returns -beta * H (approx, or just energies).
        # Let's check source: 
        #   return (interaction_energy + field_energy) / denominator
        #   interaction_energy is -J * sum(delta).
        #   So this returns Energy (scaled by denominator=Temp).
        #   Strictly speaking, it returns E / T.
        
        base_energy_term = super().get_energy(x, temps, fields, time)
        
        if self.bias_potential is not None and self.cv_compute_fn is not None:
             # Calculate V(s(x))
             s = self.cv_compute_fn(x)
             s = s.to(x.device)
             v = self.bias_potential.evaluate(s)
             
             # The sampler uses MH criterion:
             # prob ~ exp( - (E_new - E_old) / T )
             # If step() uses model(x), it expects "Energy/T".
             # We want effective energy E_eff = E + V(s).
             # So we return (E + V) / T.
             # base_energy_term is E/T.
             # We need to add V/T.
             
             # bias_potential.evaluate returns V in energy units? 
             # MDNS/bias.py: height = initial_height * ...
             # It seems V is in energy units.
             
             # If temps is a tensor [B], we divide by it.
             denominator = temps if temps is not None else torch.tensor([1.0], device=x.device)
             
             # Ensure v matches shape
             if v.ndim == 1 and base_energy_term.ndim == 1:
                 pass
             
             # Bias is added to energy.
             return base_energy_term + v / denominator
             
        return base_energy_term

    def forward(self, x, temps, fields, time=None):
        return self.get_energy(x, temps, fields, time)

    def step(self, x, temps, fields, time=None, criterion="glauber"):
        return self.sampler.step(x, self, temps, fields, time, criterion)

def main():
    args = parse_args()
    device = args.device
    
    # Setup directories
    dir_name = Path(args.dir_name)
    dir_name.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(dir_name / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=4)

    # WandB
    if args.use_wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=args, dir=str(dir_name))

    # Initialize Bias
    bias_pot = None
    if args.use_bias:
        # Helper to parse lists
        def parse_list(arg, dtype=float):
            if isinstance(arg, (int, float)): return [dtype(arg)]
            return [dtype(x) for x in arg.split(',')]

        cv_min = parse_list(args.cv_min, float)
        cv_max = parse_list(args.cv_max, float)
        grid_size = parse_list(args.bias_grid_size, int)
        sigma = parse_list(args.bias_sigma, float)
        
        D = args.L**2
        energy_scaling_val = float(D) / 16.0 if args.scale_bias_with_size else 1.0
        
        # Bias height normalization
        effective_bias_height = args.bias_height
        if args.normalize_bias_by_batch:
            effective_bias_height = args.bias_height / args.batch_size
            
        print(f"Initializing Bias: height={effective_bias_height}, scaling={energy_scaling_val}")
        
        bias_pot = BiasPotentialMultiDim(
            cv_min=cv_min, cv_max=cv_max, 
            grid_size=grid_size,
            sigma=sigma,
            initial_height=effective_bias_height,
            bias_factor=args.bias_factor,
            T=1.0/args.beta, 
            kernel_type=args.kernel_type,
            device=device,
            energy_scaling=energy_scaling_val
        )

    # Initialize Model
    # We use our Biased wrapper
    # Note: LatticePottsModel defaults to rand=True (Random site selection per step)
    model = BiasedLatticePottsModel(
        dim=args.L, 
        q=args.q, 
        init_sigma=1.0, # Not used for logic, strictly for internal J... wait. 
        # LatticePottsModel uses init_sigma for self.sigma which is used in interaction energy
        # But we pass J explicitly in some places? 
        # In LatticeIsingModel, J = G * sigma.
        # In LatticePottsModel, J = G * sigma.
        # We should set init_sigma = args.J.
        n_samples=args.batch_size,
        rand=True,
        lattice_dim=2
    )
    # Correctly set sigma/J
    model.sigma.data.fill_(args.J)
    
    # Move to device
    model.to(device)
    model.sampler.to(device) # Sampler is a buffer/module
    
    # Attach bias machinery
    model.bias_potential = bias_pot
    def cv_wrapper(x):
        return compute_cv_potts(x, args.q, device)
    model.cv_compute_fn = cv_wrapper
    
    # Initialize Samples
    # [B, L*L] or [B, L, L] flattened?
    # LatticePottsModel.init_sample returns [B, D]
    samples = model.init_sample(args.batch_size).to(device)
    
    # Temperature and Fields tensors
    temps = torch.full((args.batch_size,), 1.0/args.beta, device=device)
    fields = torch.zeros((args.batch_size,), device=device) # Assume h=0 for now, or match args
    
    print(f"Starting MCMC for {args.num_steps} steps...")
    
    # Dictionary to store bias potential states at different steps
    bias_states_dict = {}
    
    # Loop
    # We can't use model.generate_samples strictly because we want to update bias *intermittently*
    # AND we want to log things.
    # So we write our own loop calling model.step()
    
    pbar = tqdm(range(args.num_steps))
    for step in pbar:
        # MCMC Step
        # step() performs one update sweep or one single flip?
        # PottsGibbsSampler.step does ONE flip per call (rand=True selects 1 site).
        # Typically "one sweep" = N flips. 
        # But train_potts uses resample_every_n_step...
        # Let's check mcmc_block_sampling.py: it loops 'gt_steps' times calling sampler.step.
        
        # We probably want to perform multiple updates per "step" here?
        # Or just treat 'step' as one flip per walker?
        # Usually MCMC implies sweeps. 
        # But PottsGibbsSampler.step selects 1 site (if rand=True).
        # Efficiency warning: Calling python loop for every single site flip is slow. 
        # But reusing existing code.
        
        # Let's do 1 full sweep (L*L updates) per "step" of our loop? 
        # Or just 1 update?
        # If we update bias every step, it should be frequent?
        # Typically MetaD updates every few ps.
        # Let's do 1 sampler step (one site flip) per loop iteration for granularity, 
        # but maybe speed is an issue. 
        # Actually user said "perform well-tempered metadynamics using MCMC".
        # Let's assume 1 loop iteration = 1 sampler step (single site update).
        # We can increase frequency if needed.
        
        samples = model.step(samples, temps, fields, criterion='metropolis')
        
        # Bias Update
        if args.use_bias and (step % args.update_bias_every == 0):
            cv = cv_wrapper(samples)
            bias_pot.update(cv)
            
        # Logging
        if step % args.log_every == 0:
            energies = model.get_energy(samples, temps, fields)
            avg_energy = energies.mean().item()
            std_energy = energies.std().item()
            
            cv = cv_wrapper(samples)
            # cv is [B, 2]
            avg_cv = cv.mean(dim=0).cpu().tolist()
            std_cv = cv.std(dim=0).cpu().tolist()
            
            logs = {
                "step": step,
                "val/E_over_kT_mean": avg_energy,
                "val/E_over_kT_std": std_energy,
                "val/cv_x_mean": avg_cv[0],
                "val/cv_x_std": std_cv[0],
            }
            if len(avg_cv) > 1:
                logs["val/cv_y_mean"] = avg_cv[1]
                logs["val/cv_y_std"] = std_cv[1]

            # Bias Metrics
            if args.use_bias and bias_pot is not None:
                # Calculate metrics from bias grid
                # bias_grid is on device
                bias_vals = bias_pot.bias_grid
                logs["bias/mean_height"] = bias_vals.mean().item()
                logs["bias/max_height"] = bias_vals.max().item()
                
                # Coverage: Fraction of grid points with non-zero bias (or > threshold)
                # Using a small threshold to approximate "visited"
                threshold = 1e-6
                coverage = (bias_vals > threshold).float().mean().item()
                logs["bias/coverage"] = coverage
                
                # Log energies at 3 state basins (relative to global min F)
                # F(s) = - gamma/(gamma-1) * V(s)
                # Min F(s) corresponds to Max V(s)
                # Rel F(s) = F(s) - Min_s' F(s') = factor * (Max_s' V(s') - V(s))
                target_cvs = torch.tensor([
                    [1.0, 0.0], # min
                    [-0.5, np.sqrt(3)/2], # min
                    [-0.5, -np.sqrt(3)/2], # min
                    [0.0, 0.0], # random
                    [0.25, np.sqrt(3)/4], # boundary
                    [0.25, -np.sqrt(3)/4], # boundary
                    [-0.5, 0.0], # boundary
                ], device=device)
                
                v_targets = bias_pot.evaluate(target_cvs)
                max_bias = bias_vals.max()
                
                gamma = bias_pot.gamma
                factor = gamma / (gamma - 1) if gamma > 1.0 else 1.0
                
                rel_energies = factor * (max_bias - v_targets)
                
                logs["bias/min_state1"] = rel_energies[0].item()
                logs["bias/min_state2"] = rel_energies[1].item()
                logs["bias/min_state3"] = rel_energies[2].item()
                logs["bias/random_state"] = rel_energies[3].item()
                logs["bias/boundary_state1"] = rel_energies[4].item()
                logs["bias/boundary_state2"] = rel_energies[5].item()
                logs["bias/boundary_state3"] = rel_energies[6].item()

                # Save bias potential state (state_dict returns cloned tensors, so each step is a snapshot)
                bias_states_dict[step] = bias_pot.state_dict()
                
                # Save all bias states to a single file
                bias_save_path = dir_name / "bias_potential_states.pth"
                torch.save(bias_states_dict, bias_save_path)

                # Plot bias analysis
                # We do this on log step to avoid too frequent plotting
                # Or maybe on save_every? Plotting is somewhat slow. 
                # Let's do it if wandb is enabled or just save to disk
                
                # Check dimensionality of CV
                if bias_pot.ndim == 2:
                    fig = plot_bias_analysis_2d(
                        bias_pot, step, s_batch=cv, biased_reward=None, num_sites=args.L**2,
                        save_path=dir_name / f"bias_analysis_{step}.png"
                    )
                    if args.use_wandb and fig is not None:
                        wandb.log({"val/bias_analysis_plot": wandb.Image(str(dir_name / f"bias_analysis_{step}.png"))}, step=step)
                
            if args.use_wandb:
                wandb.log(logs)

        # Save
        if step % args.save_every == 0:
             # Save bias potential state and samples
             save_dict = {
                 "step": step,
                 "samples": samples.cpu(),
                 "args": vars(args)
             }
             if bias_pot is not None:
                 save_dict["bias_potential"] = bias_pot.state_dict()
                 
             torch.save(save_dict, dir_name / f"ckpt_{step}.pth")

    # Final save
    torch.save({
        "step": args.num_steps,
        "samples": samples.cpu(),
        "bias_potential": bias_pot.state_dict() if bias_pot else None
    }, dir_name / "final.pth")
    
    # Save final bias potential state to dictionary and save all states
    if bias_pot is not None:
        bias_states_dict[args.num_steps] = bias_pot.state_dict()
        torch.save(bias_states_dict, dir_name / "bias_potential_states.pth")
    
    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
