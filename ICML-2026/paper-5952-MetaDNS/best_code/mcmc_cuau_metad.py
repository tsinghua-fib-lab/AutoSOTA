import argparse
import json
import os
import sys
from pathlib import Path

import ase.build
import ase.io
import ase.units
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from tqdm import tqdm

from bias import BiasPotential, BiasPotentialMultiDim
from energy_cuau import AuCuAlloyModel, K_B
from utils import plot_bias_analysis, plot_bias_analysis_2d
from utils_cuau import compute_order_parameter, get_sublattice_map

# Try to import clease/icet for setup
try:
    from clease.settings import CEBulk, Concentration
    from icet import ClusterExpansion
except ImportError:
    import logging
    logging.warning("CLEASE/ICET not found. Energy model initialization may fail.")


def parse_args():
    parser = argparse.ArgumentParser(description="MCMC Well-Tempered Metadynamics for CuAu Alloy")
    
    # System Args
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--size', type=int, nargs=3, default=[4, 4, 4], help="Supercell size [Nx, Ny, Nz]")
    parser.add_argument('--eci_file', type=str, required=True, help="Path to ECI JSON file")
    parser.add_argument('--input_file', type=str, default=None, help="Path to input .vasp file")
    parser.add_argument('--dir_name', type=str, default="exp_local/mcmc_cuau", help="Output directory")
    
    # MCMC Args
    parser.add_argument('--num_steps', type=int, default=100000, help="Total MCMC steps")
    parser.add_argument('--batch_size', type=int, default=256, help="Number of parallel chains")
    parser.add_argument('--log_every', type=int, default=100)
    parser.add_argument('--save_every', type=int, default=1000)
    
    # Temperature and Field
    parser.add_argument('--temp', type=float, default=600.0, help="Temperature in Kelvin")
    parser.add_argument('--field', type=float, default=0.0, help="Chemical potential in eV")
    
    # Bias Args
    parser.add_argument('--use_bias', action='store_true', help='Enable biased sampling (WT-MetaD)')
    parser.add_argument('--cv_type', type=str, default="composition", choices=["composition", "composition_order"],
                        help="Type of Collective Variables to use")
    parser.add_argument('--bias_sigma', type=str, default="0.05", help='Sigma for Gaussian bias kernel')
    parser.add_argument('--bias_height', type=float, default=0.1, help='Initial bias height')
    parser.add_argument('--bias_factor', type=float, default=10.0, help='Bias factor (gamma)')
    parser.add_argument('--bias_grid_size', type=str, default="100", help='Grid size')
    parser.add_argument('--kernel_type', type=str, default='gaussian', help='Kernel type: gaussian or delta')
    parser.add_argument('--cv_min', type=str, default="0.0", help="Min CV bounds (comma-separated for 2D)")
    parser.add_argument('--cv_max', type=str, default="1.0", help="Max CV bounds (comma-separated for 2D)")
    parser.add_argument('--scale_bias_with_size', action='store_true', help='Scale bias Delta_T with system size')
    parser.add_argument('--no_normalize_bias_by_batch', dest='normalize_bias_by_batch', action='store_false', default=True,
                        help="Disable bias height norm by batch size")
    parser.add_argument('--update_bias_every', type=int, default=1, help='Update bias every N steps')
    
    # WandB
    parser.add_argument('--wandb', dest='use_wandb', action='store_true', help="Enable wandb")
    parser.add_argument('--no-wandb', dest='use_wandb', action='store_false', help="Disable wandb")
    parser.set_defaults(use_wandb=False)
    parser.add_argument('--wandb_project', type=str, default='mdns-cuau-mcmc', help="wandb project name")
    parser.add_argument('--wandb_run_name', type=str, default=None, help="wandb run name")
    parser.add_argument('--wandb_mode', type=str, default="online", choices=["offline", "online", "disabled"],
                        help="wandb logging mode: 'online' (default), 'offline' (for HPC without internet), or 'disabled'")

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


def compute_cv_cuau(x, energy_model, device):
    """
    Compute 1D CV: Au concentration.
    x: [B, L] tensor of binary occupations {0, 1}
    """
    return energy_model.get_concentrations(x)


def compute_cv_cuau_2d(x, energy_model, sublattice_map, num_sites, device):
    """
    Compute 2D CV: [Au concentration, Order parameter].
    x: [B, L] tensor of binary occupations {0, 1}
    """
    comp = energy_model.get_concentrations(x)
    order = compute_order_parameter(x, sublattice_map, num_sites)
    return torch.stack([comp, order], dim=1)


class BiasedAuCuAlloyModel:
    """
    Wraps AuCuAlloyModel to add bias potential to energy.
    """
    def __init__(self, energy_model, bias_potential=None, cv_compute_fn=None):
        self.energy_model = energy_model
        self.bias_potential = bias_potential
        self.cv_compute_fn = cv_compute_fn
        # Expose sampler for step() calls
        self.sampler = energy_model.sampler
    
    def get_free_energies(self, x, temps, fields, time=None):
        """
        Get free energies with bias potential added.
        
        Args:
            x: [B, L] tensor of binary occupations {0, 1}
            temps: [B] tensor of temperatures in Kelvin
            fields: [B] tensor of chemical potentials in eV
            time: Optional time tensor (for AIS compatibility)
        
        Returns:
            [B] tensor of free energies in units of (E - h*M + V(s))/(kB*T)
        """
        # Base free energy: (E - h*M)/(kB*T)
        base_energy = self.energy_model.get_free_energies(x, temps, fields)
        
        if self.bias_potential is not None and self.cv_compute_fn is not None:
            # Compute CV and bias
            s = self.cv_compute_fn(x)
            s = s.to(x.device)
            v = self.bias_potential.evaluate(s)
            
            # Convert T (Kelvin) to kB*T for scaling
            # base_energy is already in units of (E - h*M)/(kB*T)
            # Bias V(s) needs to be added as V(s)/(kB*T)
            # So: base_energy + v / (kB * T)
            kBT = temps * K_B  # [B] in eV
            bias_term = v / kBT  # [B]
            
            return base_energy + bias_term
        
        return base_energy
    
    def __call__(self, x, temps, fields, time=None):
        """Forward compatibility: model(x, temps, fields) should work"""
        return self.get_free_energies(x, temps, fields)
    
    def step(self, x, temps, fields, time=None, criterion="metropolis"):
        """
        Perform one MCMC step using the sampler.
        The sampler will call get_free_energies() which includes bias.
        """
        return self.sampler.step(x, self, temps, fields, time, criterion)


def setup_energy_model(args, device):
    """Setup the CuAu Energy Model."""
    # Load ECI file
    with open(args.eci_file, "r", encoding="utf-8") as f:
        eci = json.load(f)

    try:
        conc = Concentration(basis_elements=[["Au", "Cu"]])
        settings = CEBulk(
            crystalstructure="fcc",
            a=3.8, 
            size=args.size,
            concentration=conc,
            db_name="aucu_dft.db",
            max_cluster_dia=[6.0, 4.5, 4.5],
        )
    except Exception as e:
        print(f"Warning: Could not create CEBulk settings: {e}")
        settings = None

    atoms = None
    if args.input_file:
        if not os.path.exists(args.input_file):
            raise FileNotFoundError(f"Input file not found: {args.input_file}")
        print(f"Loading structure from {args.input_file}")
        atoms = ase.io.read(args.input_file)
    else:
        if settings is not None:
            atoms = settings.atoms.copy()
        else:
            atoms = ase.build.bulk("Cu", "fcc", a=3.8).repeat(tuple(args.size))

    model = AuCuAlloyModel(structure=atoms, settings=settings, eci=eci)
    return model


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
        wandb_mode = args.wandb_mode if args.wandb_mode != "disabled" else "disabled"
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=args, dir=str(dir_name), mode=wandb_mode)

    # Setup Energy Model
    energy_model = setup_energy_model(args, device)
    num_sites = energy_model.num_sites
    
    # Move sampler to device if it has parameters
    if hasattr(energy_model.sampler, 'to'):
        energy_model.sampler.to(device)

    # Initialize Bias
    bias_pot = None
    cv_compute_fn = None
    sublattice_map = None
    
    # Default CV function for logging (always compute composition)
    def default_cv_fn(x):
        return compute_cv_cuau(x, energy_model, device)
    
    if args.use_bias:
        # Helper to parse lists
        def parse_list(arg, dtype=float):
            if isinstance(arg, (int, float)): return [dtype(arg)]
            return [dtype(x) for x in arg.split(',')]

        cv_min = parse_list(args.cv_min, float)
        cv_max = parse_list(args.cv_max, float)
        grid_size = parse_list(args.bias_grid_size, int)
        sigma = parse_list(args.bias_sigma, float)
        
        D = num_sites
        energy_scaling_val = float(D) / 16.0 if args.scale_bias_with_size else 1.0
        
        # Bias height normalization
        effective_bias_height = args.bias_height
        if args.normalize_bias_by_batch:
            effective_bias_height = args.bias_height / args.batch_size
            
        print(f"Initializing Bias: height={effective_bias_height}, scaling={energy_scaling_val}")
        
        # Temperature in energy units (kB*T in eV)
        T_val = K_B * args.temp  # Convert Kelvin to energy (eV)
        
        if args.cv_type == "composition":
            # 1D CV: Au concentration
            cv_compute_fn = lambda x: compute_cv_cuau(x, energy_model, device)
            
            bias_pot = BiasPotential(
                cv_min=cv_min[0],
                cv_max=cv_max[0],
                grid_size=grid_size[0],
                sigma=sigma[0],
                initial_height=effective_bias_height,
                bias_factor=args.bias_factor,
                T=T_val,
                kernel_type=args.kernel_type,
                device=device,
                energy_scaling=energy_scaling_val
            )
        elif args.cv_type == "composition_order":
            # 2D CV: Composition + Order Parameter
            if len(cv_min) < 2: cv_min = [0.0, 0.0]
            if len(cv_max) < 2: cv_max = [1.0, 1.0]
            if len(grid_size) < 2: grid_size = [100, 100]
            if len(sigma) < 2: sigma = [0.05, 0.05]
            
            # Precompute sublattice map
            sublattice_map = get_sublattice_map(energy_model.atoms, tuple(args.size)).to(device)
            
            cv_compute_fn = lambda x: compute_cv_cuau_2d(x, energy_model, sublattice_map, num_sites, device)
            
            bias_pot = BiasPotentialMultiDim(
                cv_min=cv_min,
                cv_max=cv_max,
                grid_size=grid_size,
                sigma=sigma,
                initial_height=effective_bias_height,
                bias_factor=args.bias_factor,
                T=T_val,
                kernel_type=args.kernel_type,
                device=device,
                energy_scaling=energy_scaling_val
            )
    else:
        cv_compute_fn = default_cv_fn

    # Wrap energy model with bias
    model = BiasedAuCuAlloyModel(
        energy_model=energy_model,
        bias_potential=bias_pot,
        cv_compute_fn=cv_compute_fn
    )
    
    # Initialize Samples
    samples = model.energy_model.init_sample(args.batch_size).to(device)
    
    # Temperature and Fields tensors
    temps = torch.full((args.batch_size,), args.temp, device=device, dtype=torch.float32)
    fields = torch.full((args.batch_size,), args.field, device=device, dtype=torch.float32)
    
    print(f"Starting MCMC for {args.num_steps} steps...")
    print(f"Temperature: {args.temp} K, Field: {args.field} eV")
    print(f"Batch size: {args.batch_size}, Num sites: {num_sites}")
    
    # Dictionary to store bias potential states at different steps
    bias_states_dict = {}
    
    # Main MCMC Loop
    pbar = tqdm(range(args.num_steps))
    for step in pbar:
        # MCMC Step
        samples = model.step(samples, temps, fields, criterion='metropolis')
        
        # Bias Update
        if args.use_bias and (step % args.update_bias_every == 0):
            cv = cv_compute_fn(samples)
            bias_pot.update(cv)
            
        # Logging
        if step % args.log_every == 0:
            energies = model.get_free_energies(samples, temps, fields)
            avg_energy = energies.mean().item()
            std_energy = energies.std().item()
            
            # Compute physical energy (not free energy) for logging
            physical_energies = model.energy_model.get_energy(samples)
            avg_physical_energy = physical_energies.mean().item()
            std_physical_energy = physical_energies.std().item()
            
            cv = cv_compute_fn(samples)
            if cv.ndim == 1:
                avg_cv = cv.mean().item()
                std_cv = cv.std().item()
                logs = {
                    "step": step,
                    "val/E_over_kT_mean": avg_energy,
                    "val/E_over_kT_std": std_energy,
                    "val/E_mean": avg_physical_energy,
                    "val/E_std": std_physical_energy,
                    "val/cv_comp_mean": avg_cv,
                    "val/cv_comp_std": std_cv,
                }
            else:
                avg_cv = cv.mean(dim=0).cpu().tolist()
                std_cv = cv.std(dim=0).cpu().tolist()
                logs = {
                    "step": step,
                    "val/E_over_kT_mean": avg_energy,
                    "val/E_over_kT_std": std_energy,
                    "val/E_mean": avg_physical_energy,
                    "val/E_std": std_physical_energy,
                    "val/cv_comp_mean": avg_cv[0],
                    "val/cv_comp_std": std_cv[0],
                    "val/cv_order_mean": avg_cv[1],
                    "val/cv_order_std": std_cv[1],
                }

            # Bias Metrics
            if args.use_bias and bias_pot is not None:
                bias_vals = bias_pot.bias_grid
                logs["bias/mean_height"] = bias_vals.mean().item()
                logs["bias/max_height"] = bias_vals.max().item()
                
                # Coverage: Fraction of grid points with non-zero bias
                threshold = 1e-6
                coverage = (bias_vals > threshold).float().mean().item()
                logs["bias/coverage"] = coverage
                
                # Save bias potential state
                bias_states_dict[step] = bias_pot.state_dict()
                
                # Save all bias states to a single file
                bias_save_path = dir_name / "bias_potential_states.pth"
                torch.save(bias_states_dict, bias_save_path)

                # Plot bias analysis
                if args.cv_type == "composition":
                    # 1D CV: Use plot_bias_analysis
                    fig = plot_bias_analysis(
                        bias_pot, epoch=step, s_batch=cv, biased_reward=None, num_sites=num_sites
                    )
                    if fig is not None:
                        save_path = dir_name / f"bias_analysis_{step}.png"
                        fig.savefig(save_path, dpi=150, bbox_inches='tight')
                        plt.close(fig)  # Close figure to free memory
                        if args.use_wandb:
                            wandb.log({"val/bias_analysis_plot": wandb.Image(str(save_path))}, step=step)
                elif args.cv_type == "composition_order" and hasattr(bias_pot, 'ndim') and bias_pot.ndim == 2:
                    # 2D CV: Use plot_bias_analysis_2d
                    fig = plot_bias_analysis_2d(
                        bias_pot, epoch=step, s_batch=cv, biased_reward=None, num_sites=num_sites,
                        save_path=dir_name / f"bias_analysis_{step}.png"
                    )
                    if args.use_wandb and fig is not None:
                        wandb.log({"val/bias_analysis_plot": wandb.Image(str(dir_name / f"bias_analysis_{step}.png"))}, step=step)
                
            if args.use_wandb:
                wandb.log(logs)

        # Save
        if step % args.save_every == 0:
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
    
    # Save final bias potential state to dictionary
    if bias_pot is not None:
        bias_states_dict[args.num_steps] = bias_pot.state_dict()
        torch.save(bias_states_dict, dir_name / "bias_potential_states.pth")
    
    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
