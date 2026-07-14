#!/usr/bin/env python
"""
MCMC sampling with pre-trained biased energy landscape for Potts or CuAu model.
Loads a bias potential from a checkpoint and performs fixed-bias MCMC sampling.
Supports both Potts model (2D lattice) and CuAu alloy model (cluster expansion).
Outputs follow the same format as mdns_sampling.py for compatibility.
"""

import argparse
import json
import logging
import pickle as pkl
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from tqdm import tqdm

# Add parent directory to path to allow imports from MDNS
mdns_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(mdns_root))

from baselines.energy.ising import LatticePottsModel

from bias import BiasPotential, BiasPotentialMultiDim
from mcmc_potts_metad import BiasedLatticePottsModel, compute_cv_potts
from utils_potts import potts2d_ham

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# CuAu imports (conditional to avoid errors if not available)
try:
    from energy_cuau import K_B, AuCuAlloyModel
    from mcmc_cuau_metad import (BiasedAuCuAlloyModel, compute_cv_cuau,
                                 compute_cv_cuau_2d, setup_energy_model)
    from utils_cuau import get_sublattice_map
    CUAU_AVAILABLE = True
except ImportError as e:
    logger.warning(f"CuAu imports not available: {e}. CuAu model type will not work.")
    CUAU_AVAILABLE = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MCMC sampling with pre-trained biased energy landscape for Potts or CuAu model."
    )
    
    # Model type
    parser.add_argument(
        "--model-type",
        type=str,
        default="potts",
        choices=["potts", "cuau"],
        help="Model type: 'potts' for Potts model or 'cuau' for CuAu alloy model"
    )
    
    # Potts model arguments
    parser.add_argument("--L", type=int, default=16, help="Lattice size (Linear dimension) for Potts model")
    parser.add_argument("--q", type=int, default=3, help="Number of Potts states (q)")
    parser.add_argument("--J", type=float, default=1.0, help="Coupling constant for Potts model")
    parser.add_argument("--beta", type=float, default=0.5, help="Inverse temperature (for Potts model)")
    
    # CuAu model arguments
    parser.add_argument("--size", type=int, nargs=3, default=[4, 4, 4], help="Supercell size [Nx, Ny, Nz] for CuAu")
    parser.add_argument("--eci-file", type=str, default=None, help="Path to ECI JSON file (required for CuAu)")
    parser.add_argument("--input-file", type=str, default=None, help="Path to input .vasp file (optional for CuAu)")
    parser.add_argument(
        "--cv-type",
        type=str,
        default="composition",
        choices=["composition", "composition_order"],
        help="CV type for CuAu: 'composition' (1D) or 'composition_order' (2D)"
    )
    parser.add_argument("--field", type=float, default=0.0, help="Chemical potential in eV (for CuAu)")
    parser.add_argument("--temp", type=float, default=500.0, help="Temperature in Kelvin (for CuAu)")
    
    # Bias checkpoint (required)
    parser.add_argument(
        "--bias-checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file containing bias potential (ckpt_*.pth or final.pth)"
    )
    
    # Sampling configuration
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10000,
        help="Total number of samples to generate"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for parallel chains"
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=100,
        help="Number of MCMC steps between samples (burn-in + thinning)"
    )
    parser.add_argument(
        "--burn-in",
        type=int,
        default=50,
        help="Number of burn-in steps before collecting samples"
    )
    
    # Output
    parser.add_argument(
        "--output-folder",
        type=str,
        default="outputs/biased_mcmc",
        help="Directory to save results"
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="biased_samples.pkl",
        help="Output filename"
    )
    
    # System
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )
    
    # Seed structures
    parser.add_argument(
        "--seed-pkl",
        type=str,
        default=None,
        help="Path to .pkl file containing seed structures. File should contain 'results' dict with 'configs' key. If multiple keys exist in configs, the first one will be used."
    )
    
    return parser.parse_args()


def load_seed_structures(seed_pkl_path: str, batch_size: int, device: torch.device) -> Optional[torch.Tensor]:
    """
    Load seed structures from a .pkl file.
    
    Args:
        seed_pkl_path: Path to .pkl file containing results dict with configs
        batch_size: Number of samples needed
        device: Torch device
    
    Returns:
        Tensor of seed structures [batch_size, D] or None if loading fails
    """
    logger.info(f"Loading seed structures from {seed_pkl_path}")
    
    try:
        with open(seed_pkl_path, "rb") as f:
            data = pkl.load(f)
        
        # Handle both direct results dict and nested results structure
        if "results" in data:
            results = data["results"]
        elif "configs" in data:
            results = data
        else:
            raise ValueError(f"Expected 'results' or 'configs' key in pkl file. Found keys: {list(data.keys())}")
        
        if "configs" not in results:
            raise ValueError(f"'configs' key not found in results. Available keys: {list(results.keys())}")
        
        configs_dict = results["configs"]
        
        # If configs is a dict (with temperature/field keys), use the first key
        if isinstance(configs_dict, dict):
            if len(configs_dict) == 0:
                raise ValueError("configs dictionary is empty")
            first_key = list(configs_dict.keys())[0]
            configs = configs_dict[first_key]
            logger.info(f"Using configs from key: {first_key}")
        else:
            # If configs is directly an array
            configs = configs_dict
        
        # Convert to numpy array if needed
        if isinstance(configs, torch.Tensor):
            configs = configs.detach().cpu().numpy()
        
        configs = np.asarray(configs)
        
        # Check shape: should be [N, D] where D = L*L
        if len(configs.shape) != 2:
            raise ValueError(f"Expected configs to be 2D array [N, D], got shape {configs.shape}")
        
        num_available = configs.shape[0]
        logger.info(f"Loaded {num_available} seed structures from pkl file")
        
        # Select batch_size samples (with replacement if needed)
        if num_available >= batch_size:
            # Randomly select batch_size samples
            indices = np.random.choice(num_available, size=batch_size, replace=False)
            seed_configs = configs[indices]
        else:
            # Use all available and pad with random selection (with replacement)
            logger.warning(f"Only {num_available} seed structures available, but batch_size={batch_size}. "
                          f"Will use all available and randomly sample with replacement for the rest.")
            seed_configs = configs.copy()
            # Sample additional ones with replacement
            additional_indices = np.random.choice(num_available, size=batch_size - num_available, replace=True)
            seed_configs = np.vstack([seed_configs, configs[additional_indices]])
        
        # Convert to torch tensor and move to device
        seed_tensor = torch.from_numpy(seed_configs).to(device)
        logger.info(f"Successfully loaded seed structures: shape {seed_tensor.shape}")
        
        return seed_tensor
        
    except Exception as e:
        logger.error(f"Failed to load seed structures from {seed_pkl_path}: {e}")
        logger.error("Will use random initialization instead")
        return None


def load_bias_from_checkpoint(checkpoint_path: str, device: torch.device, args: argparse.Namespace):
    """
    Load bias potential from checkpoint file.
    
    Args:
        checkpoint_path: Path to checkpoint file (ckpt_*.pth or final.pth)
        device: Torch device
        args: Arguments namespace (for fallback values)
    
    Returns:
        Initialized BiasPotential or BiasPotentialMultiDim with loaded state
    """
    logger.info(f"Loading bias potential from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract bias potential state dict
    if "bias_potential" not in checkpoint:
        raise KeyError(f"bias_potential not found in checkpoint. Available keys: {list(checkpoint.keys())}")
    
    bias_state = checkpoint["bias_potential"]
    
    # Extract parameters from state dict
    if "params" in bias_state:
        params = bias_state["params"]
        logger.info(f"Using bias params from checkpoint: {params}")
        
        # Get parameters
        cv_min = params.get("cv_min", [-0.6, -1.0])
        cv_max = params.get("cv_max", [1.1, 1.0])
        grid_size = params.get("grid_size", 100)
        sigma = params.get("sigma", 0.05)
        initial_height = params.get("initial_height", 0.1)
        bias_factor = params.get("bias_factor", 10.0)
        # Get T from checkpoint or args
        if args.model_type == "cuau":
            T = params.get("T", K_B * args.temp if hasattr(args, 'temp') else K_B * 500.0)
        else:
            T = params.get("T", 1.0 / args.beta)
        kernel_type = params.get("kernel_type", "gaussian")
        
        # Determine CV type from checkpoint args or use args.cv_type
        cv_type_from_checkpoint = None
        if "args" in checkpoint:
            checkpoint_args = checkpoint["args"]
            cv_type_from_checkpoint = checkpoint_args.get("cv_type")
        
        # Use cv_type from args only for CuAu (where it's a meaningful choice).
        # For Potts, infer from the checkpoint's own cv_min dimensions to avoid
        # the default "composition" value incorrectly selecting a 1D BiasPotential
        # when the checkpoint contains a 2D BiasPotentialMultiDim (no grid_vals key).
        if args.model_type == "cuau" and hasattr(args, 'cv_type') and args.cv_type:
            cv_type = args.cv_type
        elif cv_type_from_checkpoint:
            cv_type = cv_type_from_checkpoint
        else:
            # Infer from cv_min/cv_max dimensions
            if isinstance(cv_min, (int, float)):
                cv_type = "composition"  # 1D
            elif len(cv_min) == 1:
                cv_type = "composition"  # 1D
            else:
                cv_type = "composition_order"  # 2D
        
        # Convert to lists if needed
        if isinstance(cv_min, (int, float)):
            cv_min = [float(cv_min)]
        if isinstance(cv_max, (int, float)):
            cv_max = [float(cv_max)]
        if isinstance(grid_size, (int, str)):
            if isinstance(grid_size, str) and ',' in grid_size:
                grid_size = [int(x.strip()) for x in grid_size.split(',')]
            else:
                grid_size = int(grid_size)
        if isinstance(grid_size, int):
            grid_size = [grid_size] * len(cv_min)
        if isinstance(sigma, (int, float, str)):
            if isinstance(sigma, str) and ',' in sigma:
                sigma = [float(x.strip()) for x in sigma.split(',')]
            else:
                sigma = float(sigma)
        if isinstance(sigma, (int, float)):
            sigma = [float(sigma)] * len(cv_min)
        
        # Get energy scaling from checkpoint args if available
        energy_scaling = 1.0
        if "args" in checkpoint:
            checkpoint_args = checkpoint["args"]
            if "scale_bias_with_size" in checkpoint_args and checkpoint_args["scale_bias_with_size"]:
                if args.model_type == "cuau":
                    # For CuAu, use num_sites from checkpoint or estimate from size
                    if "num_sites" in checkpoint_args:
                        num_sites = checkpoint_args["num_sites"]
                    elif hasattr(args, 'size'):
                        num_sites = args.size[0] * args.size[1] * args.size[2]
                    else:
                        num_sites = 64  # default 4x4x4
                    energy_scaling = float(num_sites) / 16.0
                else:
                    # For Potts, use L^2
                    L_val = checkpoint_args.get("L", args.L)
                    D = L_val ** 2
                    energy_scaling = float(D) / 16.0
        
        # Create bias potential (1D or 2D based on CV type)
        if cv_type == "composition" or len(cv_min) == 1:
            # 1D bias potential
            bias_pot = BiasPotential(
                cv_min=cv_min[0],
                cv_max=cv_max[0],
                grid_size=grid_size[0],
                sigma=sigma[0],
                initial_height=initial_height,
                bias_factor=bias_factor,
                T=T,
                kernel_type=kernel_type,
                device=device,
                energy_scaling=energy_scaling
            )
        else:
            # 2D bias potential
            bias_pot = BiasPotentialMultiDim(
                cv_min=cv_min,
                cv_max=cv_max,
                grid_size=grid_size,
                sigma=sigma,
                initial_height=initial_height,
                bias_factor=bias_factor,
                T=T,
                kernel_type=kernel_type,
                device=device,
                energy_scaling=energy_scaling
            )
        
        # Load state
        bias_pot.load_state_dict(bias_state)
        logger.info("Successfully loaded bias potential from checkpoint")
        
    else:
        raise ValueError("Bias potential state dict does not contain 'params' key. Cannot reconstruct bias potential.")
    
    return bias_pot


def run_biased_mcmc_sampling(
    model,
    bias_pot,
    args: argparse.Namespace,
    device: torch.device,
    seed_samples: Optional[torch.Tensor] = None,
    energy_model=None,
    cv_compute_fn=None
) -> Dict[str, any]:
    """
    Run MCMC sampling with fixed bias potential.
    
    Args:
        model: BiasedLatticePottsModel or BiasedAuCuAlloyModel instance
        bias_pot: BiasPotential or BiasPotentialMultiDim instance (fixed, not updated)
        args: Arguments namespace
        device: Torch device
        seed_samples: Optional tensor of seed structures [batch_size, D]. If None, uses random initialization.
        energy_model: Optional AuCuAlloyModel instance (for CuAu model type)
        cv_compute_fn: Optional CV computation function (for CuAu model type)
    
    Returns:
        Dictionary with results matching mdns_sampling.py format
    """
    logger.info(f"Starting biased MCMC sampling: {args.num_samples} samples, batch_size={args.batch_size}")
    
    # Initialize samples
    if seed_samples is not None:
        samples = seed_samples.to(device)
        logger.info(f"Using seed structures for initialization: shape {samples.shape}")
    else:
        if args.model_type == "cuau" and energy_model is not None:
            samples = energy_model.init_sample(args.batch_size).to(device)
        else:
            samples = model.init_sample(args.batch_size).to(device)
        logger.info("Using random initialization")
    
    # Temperature and fields tensors
    if args.model_type == "cuau":
        # For CuAu, use temperature in Kelvin
        temps = torch.full((args.batch_size,), args.temp, device=device, dtype=torch.float32)
        # Fields are chemical potentials in eV
        fields = torch.full((args.batch_size,), args.field, device=device, dtype=torch.float32)
        temp_k = args.temp
        field = args.field
    else:
        # For Potts, use inverse temperature
        temps = torch.full((args.batch_size,), 1.0 / args.beta, device=device)
        fields = torch.zeros((args.batch_size,), device=device)
        temp_k = 1.0 / args.beta
        field = 0.0
    
    # Storage for results
    all_configs = []
    all_energies = []
    all_x_up = []
    all_weights = []
    all_cv_values = []
    
    # Format key (matching mdns_sampling.py format)
    key = f"{temp_k:.4f}K_h{field:.4f}"
    
    samples_collected = 0
    pbar = tqdm(total=args.num_samples, desc="Sampling")
    
    # Burn-in phase
    if args.burn_in > 0:
        logger.info(f"Running burn-in: {args.burn_in} steps")
        for _ in range(args.burn_in):
            samples = model.step(samples, temps, fields, criterion="metropolis")
    
    # Sampling phase
    while samples_collected < args.num_samples:
        # Run MCMC steps
        for _ in range(args.num_steps):
            samples = model.step(samples, temps, fields, criterion="metropolis")
        
        # Collect samples from current batch
        current_batch_size = min(args.batch_size, args.num_samples - samples_collected)
        batch_samples = samples[:current_batch_size]
        
        # Compute energies and CV values based on model type
        if args.model_type == "cuau":
            # For CuAu: compute raw energy (not biased, in eV)
            if energy_model is not None:
                raw_energies_tensor = energy_model.get_energy(batch_samples)  # [B] in eV
                # Convert to numpy
                if isinstance(raw_energies_tensor, torch.Tensor):
                    raw_energies = raw_energies_tensor.detach().cpu().numpy()
                else:
                    raw_energies = np.asarray(raw_energies_tensor)
            else:
                raise ValueError("energy_model is required for CuAu model type")
            
            # Compute CV values using the provided function
            if cv_compute_fn is not None:
                cv_batch = cv_compute_fn(batch_samples)  # [B] or [B, 2]
            else:
                raise ValueError("cv_compute_fn is required for CuAu model type")
            
            # Compute x_up: Au concentration (same as CV for composition case)
            if cv_batch.ndim == 1:
                # 1D CV: composition
                x_up_batch = cv_batch.detach().cpu().numpy() if isinstance(cv_batch, torch.Tensor) else np.asarray(cv_batch)
            else:
                # 2D CV: use composition (first dimension)
                x_up_batch = cv_batch[:, 0].detach().cpu().numpy() if isinstance(cv_batch, torch.Tensor) else np.asarray(cv_batch[:, 0])
        else:
            # For Potts: compute raw energy
            raw_energies_tensor = potts2d_ham(batch_samples, J=args.J, q=args.q)  # [B]
            # Convert to numpy
            if isinstance(raw_energies_tensor, torch.Tensor):
                raw_energies = raw_energies_tensor.detach().cpu().numpy()
            else:
                raw_energies = np.asarray(raw_energies_tensor)
            
            # Compute CV values
            cv_batch = compute_cv_potts(batch_samples, args.q, device)  # [B, 2] for q=3
            
            # Compute x_up: fraction of sites in most frequent state
            batch_samples_np = batch_samples.detach().cpu().numpy()
            B, D = batch_samples_np.shape
            L = int(np.sqrt(D))
            batch_samples_2d = batch_samples_np.reshape(B, L, L)
            
            x_up_batch = np.zeros(B)
            for i in range(B):
                sample = batch_samples_2d[i]
                counts = np.bincount(sample.flatten(), minlength=args.q)
                max_count = counts.max()
                x_up_batch[i] = max_count / (L * L)
        
        # Compute unbiasing weights: w = exp(beta * V(s))
        # Ensure cv_batch is on the right device
        if isinstance(cv_batch, torch.Tensor):
            cv_batch = cv_batch.to(device)
        else:
            cv_batch = torch.tensor(cv_batch, device=device)
        
        v_s = bias_pot.evaluate(cv_batch)  # [B] bias potential values
        beta_bias = 1.0 / bias_pot.T  # Inverse temperature for bias
        log_w = beta_bias * v_s  # [B]
        
        # Clamp to prevent overflow
        log_w_max = 50.0
        log_w_clamped = torch.clamp(log_w, max=log_w_max)
        w = torch.exp(log_w_clamped).detach().cpu().numpy()
        
        # Store results
        all_configs.append(batch_samples.detach().cpu().numpy())
        all_energies.append(raw_energies)  # Already numpy
        all_x_up.append(x_up_batch)  # Already numpy
        all_weights.append(w)  # Already numpy
        # Convert cv_batch to numpy if needed
        if isinstance(cv_batch, torch.Tensor):
            all_cv_values.append(cv_batch.detach().cpu().numpy())
        else:
            all_cv_values.append(np.asarray(cv_batch))
        
        samples_collected += current_batch_size
        pbar.update(current_batch_size)
    
    pbar.close()
    
    # Concatenate all results
    configs = np.concatenate(all_configs, axis=0)
    energies = np.concatenate(all_energies, axis=0)
    x_up = np.concatenate(all_x_up, axis=0)
    weights = np.concatenate(all_weights, axis=0)
    cv_values = np.concatenate(all_cv_values, axis=0)
    
    # Format results matching mdns_sampling.py
    results = {
        "configs": {key: configs},
        "energies": {key: energies},
        "x_up": {key: x_up},
        "weights": {key: weights},
        "cv_values": {key: cv_values},
    }
    
    logger.info(f"Sampling complete: {samples_collected} samples collected")
    logger.info(f"Energy range: [{energies.min():.4f}, {energies.max():.4f}]")
    logger.info(f"x_up range: [{x_up.min():.4f}, {x_up.max():.4f}]")
    logger.info(f"Weight range: [{weights.min():.4f}, {weights.max():.4f}]")
    
    return results


def save_results(args: argparse.Namespace, results: Dict[str, any]) -> None:
    """Save results to pickle file."""
    output_dir = Path(args.output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output_name
    
    logger.info(f"Saving results to {output_path}")
    with open(output_path, "wb") as f:
        pkl.dump(results, f)
    
    # Also save a summary JSON
    summary_path = output_dir / (args.output_name.replace(".pkl", "_summary.json"))
    summary = {
        "model_type": args.model_type,
        "num_samples": args.num_samples,
        "batch_size": args.batch_size,
        "num_steps": args.num_steps,
        "burn_in": args.burn_in,
        "bias_checkpoint": args.bias_checkpoint,
        "seed_pkl": args.seed_pkl if args.seed_pkl else None,
    }
    
    # Add model-specific parameters
    if args.model_type == "potts":
        summary.update({
            "L": args.L,
            "q": args.q,
            "J": args.J,
            "beta": args.beta,
        })
    else:  # cuau
        summary.update({
            "size": args.size,
            "eci_file": args.eci_file,
            "input_file": args.input_file if args.input_file else None,
            "cv_type": args.cv_type,
            "field": args.field,
            "temp": args.temp,
        })
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to {summary_path}")


def main():
    args = parse_args()
    
    # Validate CuAu arguments if needed
    if args.model_type == "cuau":
        if not CUAU_AVAILABLE:
            raise ImportError("CuAu model components not available. Please ensure mcmc_cuau_metad, utils_cuau, and energy_cuau are importable.")
        if args.eci_file is None:
            raise ValueError("--eci-file is required when --model-type=cuau")
    
    # Set device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Load bias potential from checkpoint
    bias_pot = load_bias_from_checkpoint(args.bias_checkpoint, device, args)
    
    # Initialize model based on model type
    if args.model_type == "potts":
        # Potts model path
        logger.info(f"Initializing Potts model: L={args.L}, q={args.q}, J={args.J}")
        
        def cv_wrapper(x):
            return compute_cv_potts(x, args.q, device)
        
        model = BiasedLatticePottsModel(
            dim=args.L,
            q=args.q,
            init_sigma=args.J,  # Set sigma to J
            n_samples=args.batch_size,
            rand=True,
            lattice_dim=2,
            bias_potential=bias_pot,
            cv_compute_fn=cv_wrapper
        )
        # Ensure sigma is set correctly
        model.sigma.data.fill_(args.J)
        model.to(device)
        model.sampler.to(device)
        
        energy_model = None
        cv_compute_fn = cv_wrapper
        num_sites = args.L * args.L
        
    else:  # args.model_type == "cuau"
        # CuAu model path
        logger.info(f"Initializing CuAu model: size={args.size}, cv_type={args.cv_type}")
        
        # Setup energy model
        energy_model = setup_energy_model(args, device)
        num_sites = energy_model.num_sites
        
        # Move sampler to device if it has parameters
        if hasattr(energy_model.sampler, 'to'):
            energy_model.sampler.to(device)
        
        # Set up CV computation function
        sublattice_map = None
        if args.cv_type == "composition":
            cv_compute_fn = lambda x: compute_cv_cuau(x, energy_model, device)
        elif args.cv_type == "composition_order":
            # Precompute sublattice map for 2D CV
            sublattice_map = get_sublattice_map(energy_model.atoms, tuple(args.size)).to(device)
            cv_compute_fn = lambda x: compute_cv_cuau_2d(x, energy_model, sublattice_map, num_sites, device)
        else:
            raise ValueError(f"Unknown CV type: {args.cv_type}")
        
        # Create biased model wrapper
        model = BiasedAuCuAlloyModel(
            energy_model=energy_model,
            bias_potential=bias_pot,
            cv_compute_fn=cv_compute_fn
        )
    
    # Load seed structures if provided
    seed_samples = None
    if args.seed_pkl:
        seed_samples = load_seed_structures(args.seed_pkl, args.batch_size, device)
    
    # Run sampling
    results = run_biased_mcmc_sampling(
        model, bias_pot, args, device, 
        seed_samples=seed_samples,
        energy_model=energy_model,
        cv_compute_fn=cv_compute_fn
    )
    
    # Save results
    save_results(args, results)
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
