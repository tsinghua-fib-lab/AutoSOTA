
import argparse
import copy
import json
import logging
import os
import sys
import time

import ase.build
import ase.io
import ase.units
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from ase import Atoms

import utils_train
from bias import BiasPotential, BiasPotentialMultiDim
from energy_cuau import AuCuAlloyModel
from model import ExponentialMovingAverage
from model.transformer import MultiOutputTransformer
from utils import plot_bias_analysis_2d, plot_energy_au_conc_distributions
from utils_cuau import compute_order_parameter, get_sublattice_map

K_B = ase.units.kB  # eV/K

# Try to import clease/icet for setup
try:
    from clease.settings import CEBulk, Concentration
    from icet import ClusterExpansion
except ImportError:
    logging.warning("CLEASE/ICET not found. Energy model initialization may fail.")


class TransformerWrapper(nn.Module):
    """
    Wrapper for MultiOutputTransformer to match the interface expected by utils_train.
    
    Exposes:
        - vocab_size: Number of token types (Cu, Au, Mask)
        - length: Sequence length (number of sites)
        - forward(x, beta=None, h=None): Returns logits [B, L, vocab_size]
    """
    def __init__(self, model, vocab_size, length, device):
        super().__init__()
        self.model = model
        self.vocab_size = vocab_size
        self.length = length
        self.vocab_size = vocab_size
        self.length = length
        self.device = device
        self.default_field = None

    def set_default_field(self, field_val):
        self.default_field = field_val

    def forward(self, x, beta=None, h=None):
        # x is [B, L] indices (0, 1, 2). 2 is Mask.
        # MultiOutputTransformer expects atomic numbers or mapped indices.
        # We configured MultiOutputTransformer with num_atom_types=vocab_size (3).
        # So passing x directly is fine.
        
        # Convert beta (inverse temp) to temp.
        # utils_train might pass None, scalars, or batches.
        # MultiOutputTransformer expects tensor inputs for temp/field.
        
        batch_size = x.shape[0]
        
        # Handle beta -> temp
        if beta is None:
            # Default temp? Or 0?
            # Ideally shouldn't happen during training if reward_fn uses correct beta.
            # Let's assume a default small T or 1.0 if not provided.
            temp = torch.zeros(batch_size, device=self.device)
        elif isinstance(beta, torch.Tensor):
            # Avoid division by zero
            temp = 1.0 / (beta + 1e-6)
        else:
            temp = torch.full((batch_size,), 1.0 / (beta + 1e-6), device=self.device)
            
        # Handle h -> field
        if h is None:
            if self.default_field is not None:
                field = torch.full((batch_size, 1), self.default_field, device=self.device)
            else:
                field = torch.zeros(batch_size, 1, device=self.device)
        elif isinstance(h, torch.Tensor):
            if h.dim() == 1:
                field = h.unsqueeze(-1)
            else:
                field = h
        else:
            field = torch.full((batch_size, 1), h, device=self.device)

        outputs = self.model(x, temp=temp, field=field)
        
        # MultiOutputTransformer now returns log probabilities [B, L, vocab_size] (like vit_rope)
        # utils_train expects log probabilities [B, L, vocab_size] and will strip the last dimension
        # (mask token) before sampling: logits = model(x)[:, :, :-1]
        # So we return full log probabilities [B, L, 3] (Cu=0, Au=1, Mask=2), and utils_train strips Mask.
        
        return outputs['scalars']

    @property
    def logits(self):
        # Some utils_train functions might access model.logits(x)
        return self.forward


class CuAuRewardWrapper:
    """
    Wrapper for AuCuAlloyModel to match utils_train reward_fn interface.
    
    Signature: reward_fn(x, beta=None, h=None, J=1, use_bias=True) -> log_reward [B]
    """
    def __init__(self, energy_model, vocab_map={0: 29, 1: 79}, default_temp_k=None):
        """
        Args:
            energy_model: AuCuAlloyModel instance
            vocab_map: Mapping from vocab indices to atomic numbers
            default_temp_k: Default temperature in Kelvin for single-temp case (when beta is None)
        """
        self.energy_model = energy_model
        self.vocab_map = vocab_map # Map 0->Cu(29), 1->Au(79)
        self.vocab_map = vocab_map # Map 0->Cu(29), 1->Au(79)
        self.default_temp_k = default_temp_k  # Store default temp for single-temp case
        self.default_field = None

    def set_default_field(self, field_val):
        self.default_field = field_val

    def __call__(self, x, beta=None, h=None, J=1, use_bias=False, bias_potential=None, cv_compute_fn=None):
        """
        Args:
            bias_potential: BiasPotential instance (optional, for metadynamics)
            cv_compute_fn: Function to compute CV from x (optional, for bias)
        """
        # x is [B, L] indices (0, 1). MASK (2) should not remain in final samples.
        # AuCuAlloyModel expects {0, 1} inputs and maps internally.
        
        # Convert beta to temperature in Kelvin
        # beta = 1/(kB*T), so T = 1/(kB*beta)
        
        if beta is None:
            # Single temp/field case: utils_train doesn't pass beta
            # Use stored default temperature
            if self.default_temp_k is not None:
                temps = torch.full((x.shape[0],), self.default_temp_k, device=x.device, dtype=torch.float32)
            else:
                # Fallback: assume T=1K (shouldn't happen if default_temp_k is set)
                temps = torch.ones(x.shape[0], device=x.device, dtype=torch.float32)
        else:
            # Convert beta to temperature: T = 1/(kB*beta)
            if isinstance(beta, torch.Tensor):
                beta = beta.to(x.device)
                temps = 1.0 / (K_B * beta)
            else:
                temps = torch.full((x.shape[0],), 1.0 / (K_B * beta), device=x.device, dtype=torch.float32)
        
        # Handle field: h is in energy units (eV), use directly as fields
        if h is None:
            if self.default_field is not None:
                fields = torch.full((x.shape[0],), float(self.default_field), device=x.device, dtype=torch.float32)
            else:
                fields = torch.zeros(x.shape[0], device=x.device, dtype=torch.float32)
        else:
            if isinstance(h, torch.Tensor):
                fields = h.to(x.device)
                if fields.dim() == 0:
                    fields = fields.expand(x.shape[0])
            else:
                fields = torch.full((x.shape[0],), float(h), device=x.device, dtype=torch.float32)
        # Get free energies in units of kB*T (dimensionless)
        # get_free_energies returns (E - fields * lattices_factor) / (kB*T)
        # where E is formation energy (relative to pure phases), lower is better
        # For Boltzmann: P(x) ∝ exp(-beta * H) where H is the effective energy
        # In Ising: H = -J*sum(s_i*s_j) - h*sum(s_i), reward = -beta*H
        # For CuAu: H_eff = E - h*M, so log_reward = -beta*(E - h*M) = -(E - h*M)/(kB*T)
        free_energies = self.energy_model(x, temps, fields)  # [B] = (E - h*M)/(kB*T)
        log_reward = -free_energies
        
        # Add Bias: R' = R - beta * V(s)
        if bias_potential is not None and use_bias and cv_compute_fn is not None:
            # Compute CV (Au concentration)
            s = cv_compute_fn(x)  # [B]
            v = bias_potential.evaluate(s)  # [B]
            
            # Get beta for scaling
            if beta is None:
                # Use default temp to compute beta
                if self.default_temp_k is not None:
                    beta_val = 1.0 / (K_B * self.default_temp_k)
                else:
                    beta_val = 1.0  # Fallback
            elif isinstance(beta, torch.Tensor):
                beta_val = beta.to(x.device)
            else:
                beta_val = torch.tensor(beta, device=x.device)
            
            # Apply bias: subtract beta * V(s)
            if isinstance(beta_val, torch.Tensor):
                if beta_val.dim() == 0:
                    beta_val = beta_val.expand(x.shape[0])
                log_reward = log_reward - beta_val * v
            else:
                log_reward = log_reward - beta_val * v

        return log_reward



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

def get_args():
    # Preprocess sys.argv to handle negative flags
    sys.argv = [sys.argv[0]] + preprocess_args(sys.argv[1:])
    parser = argparse.ArgumentParser(description="Train CuAu Alloy Model")
    
    # System Config
    parser.add_argument("--size", type=int, nargs=3, default=[4, 4, 4], help="Supercell size (NxNxN)")
    parser.add_argument("--input_file", type=str, default=None, help="Path to input .vasp file")
    parser.add_argument("--eci_file", type=str, default=None, help="Path to ECI JSON file")
    
    # Model Config
    parser.add_argument("--model_type", type=str, choices=["transformer", "mlp"], default="transformer")
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_embed", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    
    # Training Config (utils_train compatible)
    parser.add_argument("--loss_fn", type=str, default="wdce", choices=["ce", "lv", "wdce", "re_rf"])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=100000, help="Number of training steps")
    parser.add_argument("--resample_every_n_step", type=int, default=10)
    parser.add_argument("--wdce_num_replicates", type=int, default=8, help="Replicates for WDCE")
    parser.add_argument("--grad_clip", action="store_true")
    parser.add_argument("--gradnorm_clip", type=float, default=1.0)
    parser.add_argument("--save_every", type=int, default=10000)
    parser.add_argument("--eval_every", type=int, default=20, help="Evaluate every N steps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from_ckpt", type=str, default=None)
    parser.add_argument("--use_anneal", action="store_true",
                        help="Pre-train at a high temperature before main training")
    parser.add_argument("--anneal_temp", type=float, default=None,
                        help="Warm-up temperature in Kelvin (must be >= temp_max). Required when --use_anneal is set.")
    parser.add_argument("--anneal_epochs", type=int, default=None,
                        help="Number of warm-up steps. Required when --use_anneal is set.")
    
    # Temps
    parser.add_argument("--temp_min", type=float, default=300.0)
    parser.add_argument("--temp_max", type=float, default=1000.0)
    # utils_train expects 'temps' arg for multi-temp training
    parser.add_argument("--num_temps", type=int, default=16) # for creating temp grid
    parser.add_argument("--field", type=float, default=0.0, help="External field (chemical potential bias) in eV")
    
    # Metadynamics / Bias
    parser.add_argument("--use_bias", action="store_true", help="Use metadynamics bias")
    parser.add_argument("--kernel_type", type=str, default="gaussian", choices=["gaussian", "delta"],
                        help="Bias kernel type: 'gaussian' (Gaussian hill) or 'delta' (binned histogram)")
    parser.add_argument("--bias_sigma", type=str, default="0.05", help="Sigma for Gaussian bias kernel (can be list)")
    parser.add_argument("--bias_height", type=float, default=0.1, 
                        help="Initial bias height. For diffusion samplers with batch updates, this is normalized by batch_size by default (see --normalize_bias_by_batch)")
    parser.add_argument("--bias_factor", type=float, default=10.0)
    parser.add_argument("--bias_grid_size", type=str, default="100", help="Grid size (can be list)")
    parser.add_argument("--cv_type", type=str, default="composition", choices=["composition", "composition_order"],
                        help="Type of Collective Variables to use")
    parser.add_argument("--cv_min", type=str, default="0.0", help="Minimum CV value (comma-separated list for 2D)")
    parser.add_argument("--cv_max", type=str, default="1.0", help="Maximum CV value (comma-separated list for 2D)")
    parser.add_argument("--scale_bias_with_size", action="store_true")
    parser.add_argument("--no_normalize_bias_by_batch", dest="normalize_bias_by_batch", action="store_false", default=True,
                        help="Disable normalization of bias_height by batch_size (default: normalization enabled). Recommended for diffusion samplers that deposit bias more frequently than traditional MCMC.")
    
    # Replay Buffer
    parser.add_argument("--buffer_size", type=int, default=0, help="Size of experience replay buffer")
    parser.add_argument("--buffer_ratio", type=float, default=0.0, help="Ratio of buffer samples in training batch")
    parser.add_argument("--buffer_n_bins", type=int, default=1, help="Number of bins for CV-based Replay Buffer")
    parser.add_argument("--buffer_strategy", type=str, default="fifo", choices=["fifo", "balanced"], 
                        help="Buffer storage strategy: fifo or balanced")
    
    # Logging
    parser.add_argument("--dir_name", type=str, default="results/cuau",
                        help="Output directory for checkpoints and logs")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="mdns-cuau")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["offline", "online", "disabled"],
                        help="wandb logging mode: 'online' (default), 'offline' (for HPC without internet), or 'disabled'")
    
    return parser.parse_args()


def setup_energy_model(args, device):
    """Setup the CuAu Energy Model."""
    if args.eci_file:
         with open(args.eci_file, "r", encoding="utf-8") as f:
            eci = json.load(f)
    else:
         if args.model_type != "dummy": # Keep quiet for tests if needed
            logging.warning("No ECI file provided. Using dummy/default initialization if possible.")
         eci = None

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
        logging.info(f"Loading structure from {args.input_file}")
        atoms = ase.io.read(args.input_file)
    else:
        if settings is not None:
            atoms = settings.atoms.copy()
        else:
             atoms = ase.build.bulk("Cu", "fcc", a=3.8).repeat(tuple(args.size))

    model = AuCuAlloyModel(structure=atoms, settings=settings, eci=eci)
    return model


def main():
    args = get_args()

    if args.use_anneal:
        assert args.anneal_temp is not None, "--anneal_temp must be specified when --use_anneal is set"
        assert args.anneal_epochs is not None, "--anneal_epochs must be specified when --use_anneal is set"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.dir_name, exist_ok=True)
    
    # Save config
    with open(os.path.join(args.dir_name, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    wandb_run = None
    if args.wandb:
        wandb_mode = args.wandb_mode if args.wandb_mode != "disabled" else "disabled"
        wandb_run = wandb.init(
            project=args.wandb_project, 
            name=args.wandb_run_name,
            config=args,
            dir=args.dir_name,
            mode=wandb_mode
        )
    
    # 1. Setup Energy Model
    energy_model = setup_energy_model(args, device)
    num_sites = energy_model.num_sites
    
    # 2. Setup Reward Function Wrapper
    # For single temp case, store the temperature so we can compute beta when it's None
    # For single temp: temp_min == temp_max, so use temp_min
    default_temp = args.temp_min if args.num_temps == 1 else None
    
    # Create CV computation function for CuAu (Au concentration)
    def compute_cv_cuau(x):
        """Compute Au concentration CV for CuAu alloy."""
        return energy_model.get_concentrations(x)
    
    # Create base reward function (will be wrapped with bias if enabled)
    reward_fn_base = CuAuRewardWrapper(energy_model, default_temp_k=default_temp)

    # 3. Setup Neural Model
    # Determine reference positions for RoPE
    if hasattr(energy_model, "atoms"):
        ref_atoms = energy_model.atoms
    else:
        ref_atoms = getattr(energy_model, "atoms", None)
    
    fixed_positions = None
    if ref_atoms is not None:
        # Don't move to device here - let the model move it when .to(device) is called
        fixed_positions = torch.tensor(
            ref_atoms.get_scaled_positions(), dtype=torch.float32
        )
    # Use vocab_size = 3 (Cu=0, Au=1, Mask=2)
    vocab_size = 3
    num_scalars = vocab_size # Output logits for all 3 tokens

    if args.model_type == "transformer":
        base_net = MultiOutputTransformer(
            num_scalars=num_scalars,
            num_marginal=1, # Unused for WDCE training currently
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            n_embed=args.n_embed,
            max_src_len=num_sites,
            physical_dim=3,
            grid_shape=tuple(args.size),
            fixed_positions=fixed_positions,
            num_atom_types=vocab_size, # Embedding input size
        ).to(device)
        
        net = TransformerWrapper(base_net, vocab_size=vocab_size, length=num_sites, device=device)
        
        # Determine default field (if using single field and provided)
        # Even if multiple temps, if we have a single field we might want to default it if h is None?
        # But if multiple temps, h_batch should be passed.
        # This safeguard is mainly for the single-temp/field case where utils_train passes None.
        if args.field != 0.0:
             print(f"Setting default field for wrappers: {args.field}")
             reward_fn_base.set_default_field(args.field)
             net.set_default_field(args.field)
    else:
        # MLP not yet adapted for wrapper
        raise NotImplementedError("MLP not yet adapted for new training loop")

    ema = ExponentialMovingAverage(net.parameters(), decay=0.9999) # utils_train uses EMA
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=0.00)

    # 4. Setup Bias Potential
    bias_potential = None
    if args.use_bias:
        D = num_sites
        energy_scaling_val = float(D) / 16.0 if args.scale_bias_with_size else 1.0
        T_kelvin = args.temp_max  # Reference T in Kelvin
        
        # BiasPotential expects T in kB*T units (energy units)
        # For CuAu: T (in Kelvin) -> kB*T (in eV) = kB * T_Kelvin
        # K_B is already imported from ase.units (eV/K)
        T_val = K_B * T_kelvin  # Convert Kelvin to energy (eV)
        
        # Helper to parse lists
        def parse_list(arg, dtype=float):
            if isinstance(arg, (int, float)): return [dtype(arg)]
            return [dtype(x) for x in arg.split(',')]
            
        cv_min_list = parse_list(args.cv_min, float)
        cv_max_list = parse_list(args.cv_max, float)
        grid_size_list = parse_list(args.bias_grid_size, int)
        sigma_list = parse_list(args.bias_sigma, float)
        
        # If 2D but user provided scalar, expand?
        # BiasPotentialMultiDim handles scalar->list, but we generally want to be explicit?
        # The parser helper returns list.
        
        # Determine CV Function and CV Params
        if args.cv_type == "composition":
            cv_min = cv_min_list[0]
            cv_max = cv_max_list[0]
            bias_grid_size = grid_size_list[0]
            bias_sigma = sigma_list[0]
            
            # Use 1D Compute Fn
            cv_compute_fn_final = compute_cv_cuau
            
        elif args.cv_type == "composition_order":
            # 2D CV: Comp, Order
            # If user didn't provide enough args, default or error?
            if len(cv_min_list) < 2: cv_min_list = [0.0, 0.0] 
            if len(cv_max_list) < 2: cv_max_list = [1.0, 1.0] # Order param max 1.0
            if len(grid_size_list) < 2: grid_size_list = [100, 100]
            if len(sigma_list) < 2: sigma_list = [0.05, 0.05]
            
            # Precompute sublattice map
            # args.size is [Nx, Ny, Nz]
            sublattice_map = get_sublattice_map(energy_model.atoms, tuple(args.size)).to(device)
            
            def compute_cv_cuau_2d(x):
                # Comp: [B]
                comp = energy_model.get_concentrations(x)
                # Order: [B]
                order = compute_order_parameter(x, sublattice_map, num_sites)
                # Stack: [B, 2]
                return torch.stack([comp, order], dim=1)
                
            cv_compute_fn_final = compute_cv_cuau_2d
            
        # Normalize bias_height by batch_size for diffusion samplers
        # Traditional metadynamics deposits 1 hill per step with height 0.1-0.5 kBT
        # Diffusion samplers deposit batch_size hills per cycle, so normalize accordingly
        effective_bias_height = args.bias_height
        if args.normalize_bias_by_batch:
            effective_bias_height = args.bias_height / args.batch_size
            total_bias_per_cycle = effective_bias_height * args.batch_size
            kBT_ratio = args.bias_height / T_val if T_val > 0 else 0
            print(f"Bias height normalization: {args.bias_height:.6f} eV -> {effective_bias_height:.8f} eV per hill")
            print(f"  (Total per cycle: {total_bias_per_cycle:.6f} eV = {kBT_ratio:.3f} kBT, batch_size={args.batch_size})")
        else:
            print(f"Bias height (no normalization): {effective_bias_height:.6f} eV per hill")
        
        print(f"Initializing Bias: kernel_type={args.kernel_type}, CV={args.cv_type}")
        print(f"Temperature: {T_kelvin} K = {T_val:.6f} eV (kB*T)")

        kernel_type = args.kernel_type
        
        if args.cv_type == "composition":
             print(f"1D Bias: sigma={bias_sigma}, factor={args.bias_factor}, CV range=[{cv_min}, {cv_max}]")
             bias_potential = BiasPotential(
                cv_min=cv_min,
                cv_max=cv_max,
                grid_size=bias_grid_size,
                sigma=bias_sigma,
                initial_height=effective_bias_height,
                bias_factor=args.bias_factor,
                T=T_val, 
                kernel_type=kernel_type,
                device=device,
                energy_scaling=energy_scaling_val
            )
        else:
             print(f"2D Bias: sigma={sigma_list}, factor={args.bias_factor}, CV range=[{cv_min_list}, {cv_max_list}]")
             bias_potential = BiasPotentialMultiDim(
                cv_min=cv_min_list,
                cv_max=cv_max_list,
                grid_size=grid_size_list,
                sigma=sigma_list,
                initial_height=effective_bias_height,
                bias_factor=args.bias_factor,
                T=T_val, 
                kernel_type=kernel_type,
                device=device,
                energy_scaling=energy_scaling_val
             )
        
        # Create biased reward function now that bias_potential exists
        def create_biased_reward_fn(base_reward, bias_pot, cv_fn):
            def biased_reward(x, beta=None, h=None, J=1, use_bias=True):
                return base_reward(x, beta=beta, h=h, J=J, use_bias=use_bias,
                                  bias_potential=bias_pot, cv_compute_fn=cv_fn)
            return biased_reward
        reward_fn = create_biased_reward_fn(reward_fn_base, bias_potential, cv_compute_fn_final)
    else:
        reward_fn = reward_fn_base
        cv_compute_fn_final = compute_cv_cuau # Default for plotting etc

    # 5. Load Checkpoint
    checkpoint = (
        torch.load(args.resume_from_ckpt, map_location=device)
        if args.resume_from_ckpt
        else None
    )
    if checkpoint:
        print(f"Loading checkpoint from {args.resume_from_ckpt}")
        net.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint.get("ema_state_dict"):
            ema.load_state_dict(checkpoint["ema_state_dict"])
        if bias_potential and checkpoint.get("bias_potential"):
            bias_potential.load_state_dict(checkpoint["bias_potential"])

    # 6. Prepare Temps/Fields for Training
    # utils_train expects args.temps as list or array if multi-temp.
    # get_temp_field_batch treats temps as 1/T (beta-like) when passed through
    # TransformerWrapper, which converts beta -> temp = 1/beta (energy units).
    # We pass temperatures in Kelvin; the reward wrapper handles kB scaling internally.
    temps_k = np.linspace(args.temp_min, args.temp_max, args.num_temps)
    args.temps = temps_k  # temperatures in Kelvin
    args.fields = np.array([args.field])

    print("Starting training with utils_train.train...")

    # Validation plotting callback (shared across phases)
    def validation_plot_callback(x, temps, fields, wandb_run=None, step=None):
        plot_energy_au_conc_distributions(
            x=x,
            energy_model=energy_model,
            temps=temps,
            fields=fields,
            save_path=os.path.join(args.dir_name, f"distributions_step_{step}.png") if step is not None else None,
            wandb_run=wandb_run,
            step=step,
            title_suffix=" (MDNS)",
        )

    common_train_kwargs = dict(
        model=net,
        optimizer=optimizer,
        device=device,
        ema=ema,
        wandb_run=wandb_run,
        bias_potential=bias_potential,
        save_dir=args.dir_name,
        cfg_dict=vars(args),
        validation_plot_callback=validation_plot_callback,
        cv_compute_fn=cv_compute_fn_final,
        buffer_size=args.buffer_size,
        buffer_ratio=args.buffer_ratio,
        buffer_n_bins=args.buffer_n_bins,
        buffer_strategy=args.buffer_strategy,
        plot_bias_fn=plot_bias_analysis_2d if args.cv_type == "composition_order" else None,
    )

    if not args.use_anneal:
        utils_train.train(
            reward_fn=reward_fn,
            args=args,
            num_epochs=args.num_epochs,
            **common_train_kwargs,
        )
        # Save final checkpoint after training completes
        utils_train.save_checkpoint(
            net, optimizer, ema,
            [], [], [],
            vars(args),
            os.path.join(args.dir_name, "weights.pth"),
            bias_potential,
        )
        print(f"Final checkpoint saved to {args.dir_name}/weights.pth")
    else:
        # ── Phase 1: warm-up at anneal_temp (high temperature) ─────────────────
        print(f"\n=== Annealing warm-up: {args.anneal_temp} K for {args.anneal_epochs} steps ===")

        # Build an args copy that points to a single warm-up temperature
        args_anneal = copy.copy(args)
        args_anneal.temps = np.array([args.anneal_temp])
        args_anneal.num_temps = 1
        args_anneal.fields = np.array([args.field])

        # Build a warm-up reward wrapper at anneal_temp
        reward_fn_anneal = CuAuRewardWrapper(energy_model, default_temp_k=args.anneal_temp)
        if args.field != 0.0:
            reward_fn_anneal.set_default_field(args.field)

        if args.use_bias:
            def create_biased_anneal_reward(base_reward, bias_pot, cv_fn):
                def biased(x, beta=None, h=None, J=1, use_bias=True):
                    return base_reward(x, beta=beta, h=h, J=J, use_bias=use_bias,
                                      bias_potential=bias_pot, cv_compute_fn=cv_fn)
                return biased
            reward_fn_anneal_wrapped = create_biased_anneal_reward(
                reward_fn_anneal, bias_potential, cv_compute_fn_final
            )
        else:
            reward_fn_anneal_wrapped = reward_fn_anneal

        utils_train.train(
            reward_fn=reward_fn_anneal_wrapped,
            args=args_anneal,
            num_epochs=args.anneal_epochs,
            **common_train_kwargs,
        )

        # Save warm-up checkpoint
        utils_train.save_checkpoint(
            net, optimizer, ema,
            [], [], [],  # losses / ess lists reset between phases
            vars(args_anneal),
            os.path.join(args.dir_name, "weights_warmup.pth"),
            bias_potential,
        )
        print(f"Warm-up checkpoint saved to {args.dir_name}/weights_warmup.pth")

        # ── Phase 2: main training at target temperature range ──────────────────
        print(f"\n=== Main training: {args.temp_min}–{args.temp_max} K for {args.num_epochs} steps ===")

        utils_train.train(
            reward_fn=reward_fn,
            args=args,
            num_epochs=args.num_epochs,
            **common_train_kwargs,
        )

        utils_train.save_checkpoint(
            net, optimizer, ema,
            [], [], [],
            vars(args),
            os.path.join(args.dir_name, "weights_final.pth"),
            bias_potential,
        )

if __name__ == "__main__":
    main()
