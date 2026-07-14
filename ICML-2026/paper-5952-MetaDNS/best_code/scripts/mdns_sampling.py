#!/usr/bin/env python
"""
Standalone script for running MDNS sampling and evaluation.
Replicates functionality of mam_arm_sampling.py for MDNS models.
Supports Ising, CuAu, and Potts models.
"""

import argparse
import json
import logging
import os
import pickle as pkl
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import ase.build
import ase.io
import ase.units
import numpy as np
import torch
from tqdm import tqdm

# Add parent directory to path to allow imports from MDNS
sys.path.append(str(Path(__file__).resolve().parent.parent))

from bias import BiasPotential, BiasPotentialMultiDim
from energy_cuau import K_B, AuCuAlloyModel
from model import ExponentialMovingAverage, get_rope_vit_model
from model.transformer import MultiOutputTransformer
from train_cuau import CuAuRewardWrapper, TransformerWrapper
from utils import ess
from utils_cuau import compute_order_parameter, get_sublattice_map
from utils_ising import ising2d_ham, ising2d_mag
from utils_potts import potts2d_ham, potts2d_magnetization_all
from utils_train import _compute_log_stats, rnd

# Try to import clease/icet for CuAu setup
try:
    from clease.settings import CEBulk, Concentration
    from icet import ClusterExpansion
except ImportError:
    logging.warning("CLEASE/ICET not found. CuAu energy model initialization may fail.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run sampling for MDNS model over temperatures and fields. Supports Ising, CuAu, and Potts models."
    )
    
    # Model type
    parser.add_argument("--model-type", type=str, default="ising", choices=["ising", "cuau", "potts"],
                       help="Model type: 'ising', 'cuau', or 'potts'")
    
    # Model arguments (common)
    parser.add_argument("--L", type=int, default=16, help="Lattice size (Linear dimension) for Ising/Potts")
    parser.add_argument("--size", type=int, nargs=3, default=[2, 2, 4], help="Supercell size [nx, ny, nz] for CuAu")
    parser.add_argument("--embed-dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--depth", type=int, default=4, help="Transformer depth")
    parser.add_argument("--num-heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--vocab-size", type=int, default=3, help="Vocab size (default 3 for MDNS)")
    
    # Potts-specific arguments
    parser.add_argument("--q", type=int, default=3, help="Number of states for Potts model (q)")
    
    # CuAu-specific arguments
    parser.add_argument("--eci-file", type=str, default=None, help="Path to ECI JSON file for CuAu")
    parser.add_argument("--input-file", type=str, default=None, help="Path to input .vasp file for CuAu")
    
    # Bias Potential (WT-ASBS)
    parser.add_argument('--use_bias', action='store_true', help='Enable biased sampling (WT-ASBS)')
    parser.add_argument('--bias_method', type=str, default='gaussian', choices=['binned', 'gaussian'], 
                       help='Bias method: binned (delta kernel) or gaussian (gaussian kernel)')
    parser.add_argument('--bias_sigma', type=str, default='0.05', help='Sigma for Gaussian bias kernel (can be comma-separated for 2D CVs, e.g., "0.05,0.05")')
    parser.add_argument('--bias_height', type=float, default=0.1, help='Initial height (W) for bias kernel')
    parser.add_argument('--bias_factor', type=float, default=10.0, help='Bias factor (gamma) for Well-Tempered Metadynamics')
    parser.add_argument('--bias_grid_size', type=str, default='100', help='Grid size for CV (can be comma-separated for 2D CVs, e.g., "65,65")')
    parser.add_argument('--kernel_type', type=str, default=None, help='Kernel type: gaussian or delta (deprecated, use --bias_method instead)')
    parser.add_argument('--cv_type', type=str, default='composition', choices=['composition', 'composition_order'],
                       help='CV type for CuAu: composition (1D) or composition_order (2D)')
    parser.add_argument('--cv_min', type=str, default=None, help='Minimum value for CV (default: -1.0 for Ising, 0.0 for CuAu). For 2D CVs (Potts/CuAu composition_order), use comma-separated values like "-0.6,-1.0" or "0,0"')
    parser.add_argument('--cv_max', type=str, default=None, help='Maximum value for CV (default: 1.0 for Ising, 1.0 for CuAu). For 2D CVs (Potts/CuAu composition_order), use comma-separated values like "1.1,1.0" or "1,1"')
    
    # Checkpoint
    parser.add_argument(
        "--ckpt", 
        type=str, 
        required=True, 
        help="Path to the model checkpoint."
    )
    
    # Sampling configuration
    parser.add_argument(
        "--temps",
        type=float,
        nargs="+",
        default=[2.269],
        help="Temperatures to evaluate (in Kelvin for CuAu).",
    )
    parser.add_argument(
        "--fields",
        type=float,
        nargs="+",
        default=[0.0],
        help="External fields (chemical potentials in eV) to evaluate.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Batch size for sampling.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=4096,
        help="Total number of samples per condition.",
    )
    parser.add_argument(
        "--J",
        type=float,
        default=1.0,
        help="Interaction strength J (for Ising only).",
    )
    
    # Output
    parser.add_argument(
        "--output-folder",
        type=str,
        default="outputs/mdns",
        help="Directory to save results.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="mdns_results.pkl",
        help="Output filename.",
    )
    
    # System
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device to use.",
    )
    
    args = parser.parse_args()
    
    # Map bias_method to kernel_type (for backward compatibility)
    if args.kernel_type is None:
        # Use bias_method to determine kernel_type
        args.kernel_type = "delta" if args.bias_method == "binned" else "gaussian"
    else:
        # If kernel_type is explicitly provided, use it (backward compatibility)
        logger.warning("--kernel_type is deprecated, use --bias_method instead")
        if args.kernel_type == "delta":
            args.bias_method = "binned"
        elif args.kernel_type == "gaussian":
            args.bias_method = "gaussian"
    
    # Parse cv_min and cv_max (can be comma-separated for 2D CVs)
    def parse_cv_value(cv_str, default):
        if cv_str is None:
            return default
        if ',' in str(cv_str):
            # Comma-separated values for 2D CV
            return tuple(float(x.strip()) for x in str(cv_str).split(','))
        else:
            # Single value for 1D CV
            return float(cv_str)
    
    # Set default CV ranges based on model type and cv_type
    if args.cv_min is None:
        if args.model_type == "cuau":
            if args.cv_type == "composition_order":
                # 2D CV: [composition, order_parameter]
                args.cv_min = (0.0, 0.0)
            else:
                # 1D CV: composition only
                args.cv_min = 0.0
        elif args.model_type == "potts":
            # For Potts with q=3, CV is 2D, default is (-0.6, -1.0)
            args.cv_min = (-0.6, -1.0)
        else:  # ising
            args.cv_min = -1.0
    else:
        args.cv_min = parse_cv_value(args.cv_min, args.cv_min)
    
    if args.cv_max is None:
        if args.model_type == "cuau":
            if args.cv_type == "composition_order":
                # 2D CV: [composition, order_parameter]
                args.cv_max = (1.0, 1.0)
            else:
                # 1D CV: composition only
                args.cv_max = 1.0
        elif args.model_type == "potts":
            # For Potts with q=3, CV is 2D, default is (1.1, 1.0)
            args.cv_max = (1.1, 1.0)
        else:
            args.cv_max = 1.0
    else:
        args.cv_max = parse_cv_value(args.cv_max, args.cv_max)
    
    return args


def setup_cuau_energy_model(args, device):
    """Setup the CuAu Energy Model."""
    if args.eci_file:
        with open(args.eci_file, "r", encoding="utf-8") as f:
            eci = json.load(f)
    else:
        logger.warning("No ECI file provided. Using dummy/default initialization if possible.")
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
        logger.warning(f"Could not create CEBulk settings: {e}")
        settings = None

    atoms = None
    if args.input_file:
        if not os.path.exists(args.input_file):
            raise FileNotFoundError(f"Input file not found: {args.input_file}")
        logger.info(f"Loading structure from {args.input_file}")
        atoms = ase.io.read(args.input_file)
    else:
        if settings is not None:
            atoms = settings.atoms.copy()
        else:
            atoms = ase.build.bulk("Cu", "fcc", a=3.8).repeat(tuple(args.size))

    model = AuCuAlloyModel(structure=atoms, settings=settings, eci=eci)
    return model


def load_model(args, device):
    logger.info(f"Loading {args.model_type} model from {args.ckpt}")
    
    checkpoint = torch.load(args.ckpt, map_location=device, weights_only=False)
    
    # Load state dicts
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    
    # Check if checkpoint has conditioning weights (for safe loading)
    # For CuAu models wrapped in TransformerWrapper, check for model.* keys
    ckpt_has_weights = (
        "beta_embedder.mlp.0.weight" in state_dict or
        "model.thermo_embedder.mlp.0.weight" in state_dict or
        any("thermo_embedder" in k for k in state_dict.keys())
    )
    
    if args.model_type == "cuau":
        # Load CuAu model (MultiOutputTransformer wrapped in TransformerWrapper)
        num_sites = args.size[0] * args.size[1] * args.size[2]
        
        # Setup energy model for CuAu first (needed to get fixed_positions)
        energy_model = setup_cuau_energy_model(args, device)
        
        # Determine reference positions for RoPE from atoms
        fixed_positions = None
        if hasattr(energy_model, "atoms"):
            ref_atoms = energy_model.atoms
        else:
            ref_atoms = getattr(energy_model, "atoms", None)
        
        # Check checkpoint to see if it was trained with fixed_positions or grid mode
        # by checking for position_embedder buffers (they exist in grid mode, not in fixed_positions mode)
        # Note: keys may have "model." prefix if saved from TransformerWrapper
        checkpoint_has_position_buffers = any(
            "position_embedder.default_input_pos" in k or "position_embedder.grid_dims" in k or
            "model.position_embedder.default_input_pos" in k or "model.position_embedder.grid_dims" in k
            for k in state_dict.keys()
        )
        
        if checkpoint_has_position_buffers:
            # Checkpoint has position buffers, so it was trained in grid mode
            logger.info("Checkpoint has position_embedder buffers - model was trained in grid mode")
            logger.info("Using grid_shape mode (fixed_positions=None) to match checkpoint")
            fixed_positions = None  # Use grid mode to match checkpoint
        elif ref_atoms is not None:
            # Checkpoint doesn't have buffers, so it was trained with fixed_positions
            # Extract scaled positions from atoms to match training configuration
            fixed_positions = torch.tensor(
                ref_atoms.get_scaled_positions(), dtype=torch.float32
            )
            logger.info(f"Checkpoint appears to be from fixed_positions mode")
            logger.info(f"Extracted fixed_positions from atoms: shape {fixed_positions.shape}")
        else:
            logger.warning("No atoms found in energy_model - using grid mode as fallback")
            fixed_positions = None
        
        vocab_size = 3
        num_scalars = vocab_size
        
        base_net = MultiOutputTransformer(
            num_scalars=num_scalars,
            num_marginal=1,
            n_layers=args.depth,
            n_heads=args.num_heads,
            n_embed=args.embed_dim,
            max_src_len=num_sites,
            physical_dim=3,
            grid_shape=tuple(args.size),
            fixed_positions=fixed_positions,
            num_atom_types=vocab_size,
        ).to(device)
        
        model = TransformerWrapper(base_net, vocab_size=vocab_size, length=num_sites, device=device)
    elif args.model_type == "potts":
        # Load Potts model (RopeVIT, similar to Ising but with vocab_size=q+1)
        vocab_size = args.q + 1  # Potts uses q+1 vocab size (0..q-1 states + padding)
        model = get_rope_vit_model(
            L=args.L,
            embed_dim=args.embed_dim,
            depth=args.depth,
            num_heads=args.num_heads,
            vocab_size=vocab_size,
            device=device
        )
        energy_model = None
    else:
        # Load Ising model (RopeVIT)
        model = get_rope_vit_model(
            L=args.L,
            embed_dim=args.embed_dim,
            depth=args.depth,
            num_heads=args.num_heads,
            vocab_size=args.vocab_size,
            device=device
        )
        energy_model = None
    
    # Initialize EMA wrapper often used in MDNS
    ema = ExponentialMovingAverage(model.parameters(), decay=0.9999)
    
    # Load model weights
    # For CuAu models, always use strict=False because position_embedder buffers may differ
    # depending on whether fixed_positions was used during training
    if args.model_type == "cuau" or not ckpt_has_weights:
        logger.info("Loading state dict with strict=False (allowing buffer/parameter mismatches)")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        
        # Verify only expected keys are missing
        expected_missing = {"beta_embedder", "h_embedder", "thermo_proj", "thermo_embedder"}
        # For CuAu, position_embedder buffers may differ between fixed_positions and grid modes
        if args.model_type == "cuau":
            expected_missing.update({"default_input_pos", "grid_dims", "head_splits"})
            # Filter out position_embedder buffer mismatches from missing keys
            real_missing = [k for k in missing 
                          if not any(e in k for e in expected_missing) 
                          and "position_embedder" not in k]
        else:
            real_missing = [k for k in missing if not any(e in k for e in expected_missing)]
        
        if real_missing:
            logger.warning(f"Missing keys: {len(real_missing)}")
            for k in real_missing[:10]:  # Show first 10
                logger.warning(f"  - {k}")
        else:
            logger.info("All critical parameters loaded successfully")
        
        # Log unexpected keys (buffers/params that exist in checkpoint but not in model)
        if unexpected:
            logger.info(f"Unexpected keys in checkpoint (not loaded into model): {len(unexpected)}")
            # Filter out position_embedder buffer mismatches as these are expected
            if args.model_type == "cuau":
                real_unexpected = [k for k in unexpected 
                                 if "position_embedder" not in k 
                                 or ("default_input_pos" not in k and "grid_dims" not in k)]
            else:
                real_unexpected = unexpected
            if real_unexpected:
                for k in real_unexpected[:10]:  # Show first 10
                    logger.info(f"  - {k}")
    else:
        try:
            model.load_state_dict(state_dict)
        except RuntimeError as e:
            logger.warning(f"Strict loading failed: {e}")
            logger.info("Attempting non-strict loading...")
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded with {len(missing)} missing keys and {len(unexpected)} unexpected keys")

    if 'ema_state_dict' in checkpoint:
        ema.load_state_dict(checkpoint['ema_state_dict'])
        ema.store(model.parameters())
        ema.copy_to(model.parameters())
        logger.info("Loaded EMA weights into model")
    else:
        logger.warning("No EMA state dict found in checkpoint, using standard weights")
    
    # Initialize BiasPotential if requested
    bias_pot = None
    if args.use_bias:
        # Reference temperature for bias-potential initialization. Defined here so it
        # is available to both the checkpoint-bias branch and the CLI-fallback branch.
        # Checkpoint T should already be in eV for CuAu (converted during training).
        T_init = args.temps[0] if args.temps else 1.0
        if args.model_type == "cuau":
            T_init = K_B * T_init  # Convert Kelvin to eV for fallback initialization
        if 'bias_potential' in checkpoint:
            logger.info("Loading BiasPotential from checkpoint")
            bias_state = checkpoint['bias_potential']

            # Check if params exist in state_dict (saved by modern bias.py)
            if 'params' in bias_state:
                params = bias_state['params']
                logger.info(f"Using bias params from checkpoint: {params}")
                
                # Get T from params
                # For CuAu checkpoints, T is already in eV (converted during training in train_cuau.py line 375)
                # For Ising/Potts checkpoints, T might be in different units - check if > 100 (likely Kelvin)
                T_val = params.get('T', T_init)
                if args.model_type == "cuau":
                    # CuAu checkpoints always store T in eV - don't convert
                    # Only convert if T_val > 100 (which would indicate it's in Kelvin, not eV)
                    if isinstance(T_val, (int, float)) and T_val > 100.0:
                        logger.warning(f"Bias potential T value {T_val} seems to be in Kelvin. Converting to eV.")
                        T_val = K_B * T_val
                    else:
                        logger.info(f"Using bias T from checkpoint as-is: {T_val} eV")
                elif args.model_type in ["ising", "potts"]:
                    # For Ising/Potts, if T > 100, likely in Kelvin (though typically uses dimensionless beta)
                    if isinstance(T_val, (int, float)) and T_val > 100.0:
                        logger.warning(f"Bias potential T value {T_val} seems to be in Kelvin. Converting.")
                        T_val = K_B * T_val
                
                # Use BiasPotentialMultiDim for Potts (2D CV) or CuAu with composition_order (2D CV)
                # BiasPotential for Ising (1D CV) or CuAu with composition (1D CV)
                use_2d_cv = (args.model_type == "potts") or (args.model_type == "cuau" and args.cv_type == "composition_order")
                if use_2d_cv:
                    # Convert cv_min, cv_max, grid_size, sigma to lists if needed
                    def to_list(val):
                        if isinstance(val, tuple):
                            return list(val)
                        elif isinstance(val, (int, float)):
                            return [val]
                        return val
                    
                    cv_min_val = params.get('cv_min', args.cv_min)
                    cv_max_val = params.get('cv_max', args.cv_max)
                    grid_size_val = params.get('grid_size', args.bias_grid_size)
                    sigma_val = params.get('sigma', args.bias_sigma)
                    
                    # Parse grid_size if it's a string
                    if isinstance(grid_size_val, str):
                        if ',' in grid_size_val:
                            grid_size_val = [int(x) for x in grid_size_val.split(',')]
                        else:
                            grid_size_val = int(grid_size_val)
                    
                    # Parse sigma if it's a string
                    if isinstance(sigma_val, str):
                        if ',' in sigma_val:
                            sigma_val = [float(x) for x in sigma_val.split(',')]
                        else:
                            sigma_val = float(sigma_val)
                    
                    bias_pot = BiasPotentialMultiDim(
                        cv_min=to_list(cv_min_val),
                        cv_max=to_list(cv_max_val),
                        grid_size=grid_size_val,
                        sigma=to_list(sigma_val) if not isinstance(sigma_val, list) else sigma_val,
                        initial_height=params.get('initial_height', args.bias_height),
                        bias_factor=params.get('bias_factor', args.bias_factor),
                        T=T_val,
                        kernel_type=params.get('kernel_type', args.kernel_type),
                        device=device
                    )
                else:
                    # 1D CV: parse string inputs if needed
                    grid_size_val = params.get('grid_size', args.bias_grid_size)
                    if isinstance(grid_size_val, str):
                        grid_size_val = int(grid_size_val)
                    sigma_val = params.get('sigma', args.bias_sigma)
                    if isinstance(sigma_val, str):
                        sigma_val = float(sigma_val)
                    
                    bias_pot = BiasPotential(
                        cv_min=params.get('cv_min', args.cv_min), 
                        cv_max=params.get('cv_max', args.cv_max), 
                        grid_size=grid_size_val,
                        sigma=sigma_val,
                        initial_height=params.get('initial_height', args.bias_height),
                        bias_factor=params.get('bias_factor', args.bias_factor),
                        T=T_val,
                        kernel_type=params.get('kernel_type', args.kernel_type),
                        device=device
                    )
            else:
                # Fallback to CLI args if params not inside state_dict
                logger.warning("Bias params not found in checkpoint state dict! Using CLI arguments.")
                use_2d_cv = (args.model_type == "potts") or (args.model_type == "cuau" and args.cv_type == "composition_order")
                if use_2d_cv:
                    # Parse comma-separated values for Potts 2D CV
                    def parse_list_arg(arg):
                        if isinstance(arg, tuple):
                            return list(arg)
                        elif isinstance(arg, str) and ',' in arg:
                            return [float(x.strip()) for x in arg.split(',')]
                        elif isinstance(arg, (int, float)):
                            return [float(arg)]
                        return arg
                    
                    # Parse grid_size
                    grid_size_val = args.bias_grid_size
                    if isinstance(grid_size_val, str) and ',' in grid_size_val:
                        grid_size_val = [int(x.strip()) for x in grid_size_val.split(',')]
                    elif isinstance(grid_size_val, (int, str)):
                        grid_size_val = int(grid_size_val) if isinstance(grid_size_val, str) else grid_size_val
                    
                    # Parse sigma
                    sigma_val = args.bias_sigma
                    if isinstance(sigma_val, str) and ',' in sigma_val:
                        sigma_val = [float(x.strip()) for x in sigma_val.split(',')]
                    elif isinstance(sigma_val, (int, float, str)):
                        sigma_val = float(sigma_val) if isinstance(sigma_val, str) else sigma_val
                    
                    bias_pot = BiasPotentialMultiDim(
                        cv_min=parse_list_arg(args.cv_min),
                        cv_max=parse_list_arg(args.cv_max),
                        grid_size=grid_size_val,
                        sigma=parse_list_arg(sigma_val) if not isinstance(sigma_val, list) else sigma_val,
                        initial_height=args.bias_height,
                        bias_factor=args.bias_factor,
                        T=T_init,
                        kernel_type=args.kernel_type,
                        device=device
                    )
                else:
                    # 1D CV: parse string inputs if needed
                    grid_size_val = args.bias_grid_size
                    if isinstance(grid_size_val, str):
                        grid_size_val = int(grid_size_val)
                    sigma_val = args.bias_sigma
                    if isinstance(sigma_val, str):
                        sigma_val = float(sigma_val)
                    
                    bias_pot = BiasPotential(
                        cv_min=args.cv_min, cv_max=args.cv_max, 
                        grid_size=grid_size_val,
                        sigma=sigma_val,
                        initial_height=args.bias_height,
                        bias_factor=args.bias_factor,
                        T=T_init,
                        kernel_type=args.kernel_type,
                        device=device
                    )
            
            bias_pot.load_state_dict(bias_state)
            
            # Validate that all sampling temperatures match training temperature for metadynamics
            if args.use_bias and args.model_type == "cuau" and args.temps:
                # bias_pot.T is in eV (energy units) - this is the training temperature
                training_T_ev = bias_pot.T  # Already in eV
                training_T_k = training_T_ev / K_B  # Convert back to Kelvin for comparison
                
                # Check all sampling temperatures match
                for sampling_temp_k in args.temps:
                    tolerance_k = 0.1  # Allow 0.1K difference
                    if abs(sampling_temp_k - training_T_k) > tolerance_k:
                        raise ValueError(
                            f"Sampling temperature ({sampling_temp_k}K) does not match "
                            f"bias potential training temperature ({training_T_k:.2f}K = {training_T_ev:.6f} eV). "
                            f"For metadynamics, all sampling temperatures must match the training temperature."
                        )
                logger.info(f"✓ All sampling temperatures match training temperature ({training_T_k:.2f}K = {training_T_ev:.6f} eV)")
            elif args.use_bias and args.model_type in ["ising", "potts"] and args.temps:
                # For Ising/Potts, temperature is dimensionless (beta = 1/T)
                training_T = bias_pot.T  # Already in dimensionless units
                
                # Check all sampling temperatures match
                for sampling_temp in args.temps:
                    tolerance = 0.01  # Allow small difference
                    if abs(sampling_temp - training_T) > tolerance:
                        raise ValueError(
                            f"Sampling temperature ({sampling_temp}) does not match "
                            f"bias potential training temperature ({training_T}). "
                            f"For metadynamics, all sampling temperatures must match the training temperature."
                        )
                logger.info(f"✓ All sampling temperatures match training temperature ({training_T})")
            
            # Normalize bias potential (shift min to 0) to avoid numerical explosion in weights
            # Note: BiasPotentialMultiDim doesn't have normalize() method
            if hasattr(bias_pot, 'normalize'):
                logger.info("Normalizing bias potential (shifting min to 0)...")
                bias_pot.normalize()
            else:
                logger.info("Skipping normalization (BiasPotentialMultiDim doesn't support it)")
            
        else:
            logger.warning("Bias potential requested but not found in checkpoint! Using initialized (empty/initial) bias from CLI args.")
            use_2d_cv = (args.model_type == "potts") or (args.model_type == "cuau" and args.cv_type == "composition_order")
            if use_2d_cv:
                # Parse comma-separated values for Potts 2D CV
                def parse_list_arg(arg):
                    if isinstance(arg, tuple):
                        return list(arg)
                    elif isinstance(arg, str) and ',' in arg:
                        return [float(x.strip()) for x in arg.split(',')]
                    elif isinstance(arg, (int, float)):
                        return [float(arg)]
                    return arg
                
                # Parse grid_size
                grid_size_val = args.bias_grid_size
                if isinstance(grid_size_val, str) and ',' in grid_size_val:
                    grid_size_val = [int(x.strip()) for x in grid_size_val.split(',')]
                elif isinstance(grid_size_val, (int, str)):
                    grid_size_val = int(grid_size_val) if isinstance(grid_size_val, str) else grid_size_val
                
                # Parse sigma
                sigma_val = args.bias_sigma
                if isinstance(sigma_val, str) and ',' in sigma_val:
                    sigma_val = [float(x.strip()) for x in sigma_val.split(',')]
                elif isinstance(sigma_val, (int, float, str)):
                    sigma_val = float(sigma_val) if isinstance(sigma_val, str) else sigma_val
                
                bias_pot = BiasPotentialMultiDim(
                    cv_min=parse_list_arg(args.cv_min),
                    cv_max=parse_list_arg(args.cv_max),
                    grid_size=grid_size_val,
                    sigma=parse_list_arg(sigma_val) if not isinstance(sigma_val, list) else sigma_val,
                    initial_height=args.bias_height,
                    bias_factor=args.bias_factor,
                    T=T_init,
                    kernel_type=args.kernel_type,
                    device=device
                )
            else:
                # 1D CV: coerce string CLI inputs to numeric (as the other branches do)
                grid_size_val = args.bias_grid_size
                if isinstance(grid_size_val, str):
                    grid_size_val = int(grid_size_val)
                sigma_val = args.bias_sigma
                if isinstance(sigma_val, str):
                    sigma_val = float(sigma_val)
                bias_pot = BiasPotential(
                    cv_min=args.cv_min, cv_max=args.cv_max,
                    grid_size=grid_size_val,
                    sigma=sigma_val,
                    initial_height=args.bias_height,
                    bias_factor=args.bias_factor,
                    T=T_init,
                    kernel_type=args.kernel_type,
                    device=device
                )

    model.eval()
    
    # Determine if we SHOULD condition based on user arguments (Sweep -> Condition)
    # "check if either the number of input --temps and --fields is greater than 1"
    force_conditioning = len(args.temps) > 1 or len(args.fields) > 1
    
    if force_conditioning and not ckpt_has_weights:
        logger.warning("Requesting a sweep (conditioning needed) but checkpoint lacks conditioning weights. Proceeding with random embeddings (IS with unconditioned proposal).")
        
    return model, force_conditioning, bias_pot, energy_model


def _format_key(temp: float, field: float) -> str:
    return f"{temp:.4f}K_h{field:.4f}"


def run_sampling(
    model: torch.nn.Module,
    args: argparse.Namespace,
    device: torch.device,
    has_conditioning: bool,
    bias_pot: Optional[BiasPotential] = None,
    energy_model: Optional[Any] = None
) -> Dict[str, Dict[str, Any]]:
    
    configs = defaultdict(list)
    energies = defaultdict(list)
    x_up = defaultdict(list)
    log_rnd_values = defaultdict(list) # Restore log_rnd
    log_rw_values = defaultdict(list) 
    logp_x_values = defaultdict(list)
    logf_t_values = defaultdict(list)
    weights_values = defaultdict(list) 
    ness_values: Dict[str, float] = {}
    free_energies: Dict[str, float] = {}
    free_energy_profiles: Dict[str, np.ndarray] = {}
    cv_grids: Dict[str, np.ndarray] = {}
    
    # Define CV computation functions based on model type
    def compute_cv_ising(x):
        """Compute magnetization CV for Ising model."""
        spins = 2 * x.float() - 1
        return ising2d_mag(spins)
    
    def compute_cv_potts(x):
        """Compute CV for Potts model.
        
        For q=3, returns 2D projection: [proj_x, proj_y]
        For other q, returns first q-1 concentrations.
        
        Args:
            x: Input configurations [B, L*L] with values in {0, 1, ..., q-1}
        """
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = x
        
        # Reshape to B, D
        if x_np.ndim == 1:
            x_np = x_np.reshape(1, -1)
        
        B, D_ = x_np.shape
        q = args.q
        
        # Count frequencies for each state 0..q-1
        # counts: [B, q]
        counts = np.zeros((B, q))
        for i in range(q):
            counts[:, i] = np.sum(x_np == i, axis=1)
        
        concentrations = counts / D_  # [B, q]
        
        if q == 3:
            # 2D projection for q=3:
            # x = c1 - 0.5 * (c2 + c3)
            # y = (sqrt(3)/2) * (c2 - c3)
            c1 = concentrations[:, 0]
            c2 = concentrations[:, 1]
            c3 = concentrations[:, 2]
            
            proj_x = c1 - 0.5 * (c2 + c3)
            proj_y = (np.sqrt(3)/2) * (c2 - c3)
            
            # Stack [B, 2]
            cv = np.stack([proj_x, proj_y], axis=1)
            return torch.tensor(cv, device=x.device if isinstance(x, torch.Tensor) else device, dtype=torch.float32)
        else:
            # Fallback for q!=3: return first q-1 concentrations
            return torch.tensor(concentrations[:, :-1], device=x.device if isinstance(x, torch.Tensor) else device, dtype=torch.float32)
    
    def compute_cv_cuau(x, energy_model_arg=None):
        """Compute Au concentration CV for CuAu alloy (1D).
        
        Args:
            x: Input configurations [B, L]
            energy_model_arg: Energy model instance (optional, for CuAuRewardWrapper compatibility)
        """
        # If energy_model_arg is provided (from CuAuRewardWrapper), use it
        # Otherwise use the energy_model from closure
        if energy_model_arg is not None:
            return energy_model_arg.get_concentrations(x)
        else:
            if energy_model is None:
                raise ValueError("Energy model required for CuAu sampling")
            return energy_model.get_concentrations(x)
    
    def compute_cv_cuau_2d(x, energy_model_arg=None, sublattice_map=None, num_sites=None):
        """Compute 2D CV [composition, order_parameter] for CuAu alloy.
        
        Args:
            x: Input configurations [B, L]
            energy_model_arg: Energy model instance (optional)
            sublattice_map: Sublattice mapping tensor [L] (required for 2D CV)
            num_sites: Total number of sites (required for 2D CV)
        """
        # Get composition (1D)
        if energy_model_arg is not None:
            comp = energy_model_arg.get_concentrations(x)
        else:
            if energy_model is None:
                raise ValueError("Energy model required for CuAu sampling")
            comp = energy_model.get_concentrations(x)
        
        # Get order parameter
        if sublattice_map is None or num_sites is None:
            raise ValueError("sublattice_map and num_sites required for composition_order CV")
        order = compute_order_parameter(x, sublattice_map, num_sites)
        
        # Stack to [B, 2]
        return torch.stack([comp, order], dim=1)
    
    # Select CV computation function based on model type
    # For CuAu, set up CV function based on cv_type
    cuau_sublattice_map = None
    cuau_num_sites = None
    if args.model_type == "cuau":
        if args.cv_type == "composition_order":
            # 2D CV: need to precompute sublattice map
            if energy_model is None:
                raise ValueError("Energy model required for CuAu sampling with composition_order CV")
            cuau_num_sites = args.size[0] * args.size[1] * args.size[2]
            cuau_sublattice_map = get_sublattice_map(energy_model.atoms, tuple(args.size)).to(device)
            # Create closure with sublattice_map
            def cv_compute_fn_cuau_2d(x):
                return compute_cv_cuau_2d(x, energy_model_arg=energy_model, 
                                        sublattice_map=cuau_sublattice_map, 
                                        num_sites=cuau_num_sites)
            cv_compute_fn = cv_compute_fn_cuau_2d
        else:
            # 1D CV: composition only
            cv_compute_fn = compute_cv_cuau
    
    # Select CV computation function and reward function based on model type
    if args.model_type == "cuau":
        # CuAu reward function using CuAuRewardWrapper
        def get_reward_fn(default_temp_k, default_field, bias_pot=None):
            # Create CuAuRewardWrapper instance
            reward_wrapper = CuAuRewardWrapper(
                energy_model=energy_model,
                default_temp_k=default_temp_k
            )
            reward_wrapper.set_default_field(default_field)
            
            def reward_fn(x, beta=None, h=None, J=1, **kwargs):
                """Reward function for CuAu model."""
                # CuAuRewardWrapper expects beta and h, converts internally
                return reward_wrapper(x, beta=beta, h=h, J=J, use_bias=False, 
                                    bias_potential=None, cv_compute_fn=None)
            
            def biased_reward_fn(x, beta=None, h=None, J=1, use_bias=True):
                """Biased reward function for CuAu model."""
                # Use CuAuRewardWrapper with bias
                return reward_wrapper(x, beta=beta, h=h, J=J, use_bias=use_bias,
                                    bias_potential=bias_pot, cv_compute_fn=cv_compute_fn)
            
            return biased_reward_fn if args.use_bias else reward_fn
    elif args.model_type == "potts":
        cv_compute_fn = compute_cv_potts
        # Potts reward function
        def get_reward_fn(default_beta, default_h, J=1, bias_pot=None):
            def reward_fn(x, beta=None, h=None, J=J, **kwargs):
                """Reward function for Potts model."""
                beta_val = beta if beta is not None else default_beta
                # Potts reward: -beta * H
                return -beta_val * potts2d_ham(x, J=J, q=args.q)
            
            def biased_reward_fn(x, beta=None, h=None, J=J, use_bias=True):
                # 1. Standard reward
                r = reward_fn(x, beta=beta, h=h, J=J)
                
                # 2. Add Bias: R' = R - beta * V(s)
                if bias_pot is not None and use_bias:
                    s = compute_cv_potts(x)
                    v = bias_pot.evaluate(s)
                    
                    # Get beta for scaling
                    beta_val = beta if beta is not None else default_beta
                    # Handle tensor/scalar beta
                    if isinstance(beta_val, (int, float)):
                        beta_tensor = torch.tensor(beta_val, device=x.device)
                    elif isinstance(beta_val, torch.Tensor):
                        beta_tensor = beta_val.to(x.device)
                    else:
                        beta_tensor = torch.tensor(beta_val, device=x.device)
                    
                    r = r - beta_tensor * v
                return r
            
            return biased_reward_fn if args.use_bias else reward_fn
    else:
        # Ising reward function
        cv_compute_fn = compute_cv_ising
        
        def get_reward_fn(default_beta, default_h, J=1, bias_pot=None):
            def reward_fn(x, beta=None, h=None, J=J, **kwargs):
                """Reward function wrapper that handles scalar or per-sample betas/fields."""
                beta_val = beta if beta is not None else default_beta
                h_val = h if h is not None else default_h
                
                spin = 2 * x.float() - 1
                return -beta_val * ising2d_ham(spin, J=J, h=h_val)

            def biased_reward_fn(x, beta=None, h=None, J=J, use_bias=True):
                # 1. Standard reward
                r = reward_fn(x, beta=beta, h=h, J=J)
                
                # 2. Add Bias: R' = R - beta * V(s)
                if bias_pot is not None and use_bias:
                    # Convert x (0,1) to spins (-1,1) for CV calc
                    s = compute_cv_ising(x)
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
                    
                    r = r - beta_tensor * v
                return r
            
            return biased_reward_fn if args.use_bias else reward_fn

    # Loop over conditions
    for temp in tqdm(args.temps, desc="Temps"):
        # Convert temperature to beta based on model type
        if args.model_type == "cuau":
            # For CuAu, temp is in Kelvin, beta = 1/(kB*T) in 1/eV
            beta = 1.0 / (K_B * temp)
            temp_k = temp  # Store in Kelvin for CuAu
        else:
            # For Ising, temp is dimensionless (equivalent to 1/beta)
            beta = 1.0 / temp
            temp_k = None  # Not used for Ising
        
        for field in tqdm(args.fields, desc="Fields", leave=False):
            key = _format_key(temp, field)
            
            samples_collected = 0
            
            batch_configs = []
            batch_energies = []
            batch_x_up = []
            batch_log_rnd = [] # Restore log_rnd
            batch_log_rw = [] 
            batch_logp_x = []
            batch_logf_t = []
            batch_weights = []
            
            pbar = tqdm(total=args.num_samples, desc=f"Sampling {key}", leave=False)
            
            # Create condition-specific reward function (closure over beta/field or temp/field)
            if args.model_type == "cuau":
                current_reward_fn = get_reward_fn(default_temp_k=temp_k, default_field=field, bias_pot=bias_pot)
            elif args.model_type == "potts":
                current_reward_fn = get_reward_fn(default_beta=beta, default_h=field, J=args.J, bias_pot=bias_pot)
            else:  # ising
                current_reward_fn = get_reward_fn(default_beta=beta, default_h=field, J=args.J, bias_pot=bias_pot)
            
            while samples_collected < args.num_samples:
                current_batch_size = min(args.batch_size, args.num_samples - samples_collected)
                with torch.no_grad():
                    # Handle conditional vs unconditional
                    if has_conditioning:
                        # For CuAu, pass temp in Kelvin; for Ising/Potts, pass beta
                        if args.model_type == "cuau":
                            temp_batch = torch.full((current_batch_size,), temp_k, device=device).float()
                            field_batch = torch.full((current_batch_size,), field, device=device).float()
                            # For rnd, we need to pass beta_batch and h_batch
                            # CuAuRewardWrapper will handle the conversion
                            beta_batch = 1.0 / (K_B * temp_batch)
                            h_batch = field_batch
                        else:  # ising or potts
                            beta_batch = torch.full((current_batch_size,), beta, device=device).float()
                            h_batch = torch.full((current_batch_size,), field, device=device).float()
                        
                        x, log_rnd = rnd(
                            model, 
                            current_reward_fn, 
                            batch_size=current_batch_size, 
                            device=device,
                            beta_batch=beta_batch,
                            h_batch=h_batch,
                            J=args.J
                        )
                        # Compute stats
                        logf_t, logp_x = _compute_log_stats(
                            x, log_rnd, current_reward_fn, model,
                            beta_batch=beta_batch,
                            h_batch=h_batch,
                            J=args.J,
                            bias_potential=bias_pot,
                            cv_compute_fn=cv_compute_fn if args.model_type == "cuau" else None
                        )
                    else:
                        # Unconditional model
                        x, log_rnd = rnd(
                            model, 
                            current_reward_fn,
                            batch_size=current_batch_size, 
                            device=device,
                            beta_batch=None, 
                            h_batch=None,
                            J=args.J
                        )
                        # Compute stats
                        logf_t, logp_x = _compute_log_stats(
                            x, log_rnd, current_reward_fn, model,
                            beta_batch=None,
                            h_batch=None,
                            J=args.J,
                            bias_potential=bias_pot,
                            cv_compute_fn=cv_compute_fn if args.model_type in ["cuau", "potts"] else None
                        )
                    
                    log_rw = logf_t - logp_x

                    # Compute energies (unweighted, raw energy)
                    if args.model_type == "cuau":
                        # For CuAu, use energy_model.get_energy
                        if energy_model is None:
                            raise ValueError("Energy model required for CuAu sampling")
                        raw_energy = energy_model.get_energy(x)  # [B] in eV
                        # Au concentration (x_up)
                        x_up_batch = energy_model.get_concentrations(x)  # [B]
                        # Use cv_compute_fn for bias evaluation (handles both 1D and 2D CVs)
                        cv_val_for_bias = cv_compute_fn(x)  # [B] for 1D, [B, 2] for 2D
                    elif args.model_type == "potts":
                        # For Potts, use potts2d_ham
                        raw_energy = potts2d_ham(x, J=args.J, q=args.q)  # [B]
                        # For x_up: compute fraction of sites in the most frequent state
                        # x is [B, L*L] with values in {0, 1, ..., q-1}
                        x_reshaped = x.reshape(x.shape[0], args.L, args.L)  # [B, L, L]
                        # For each sample, count occurrences of each state and find max
                        x_up_batch = torch.zeros(x.shape[0], device=x.device)
                        for i in range(x.shape[0]):
                            sample = x_reshaped[i]  # [L, L]
                            # Count occurrences of each state
                            counts = torch.bincount(sample.flatten().long(), minlength=args.q)
                            # Get the maximum count (most frequent state)
                            max_count = counts.max().float()
                            # Fraction of sites in most frequent state
                            x_up_batch[i] = max_count / (args.L * args.L)
                        # Use full CV for bias evaluation (2D projection)
                        cv_val_for_bias = compute_cv_potts(x)
                    else:  # ising
                        # For Ising, use ising2d_ham to compute raw energy
                        # Convert x from {0, 1} to {-1, 1} format for ising2d_ham
                        spin = 2 * x.float() - 1  # [B, L*L] with values in {-1, 1}
                        h_val = args.h if hasattr(args, 'h') else 0.0
                        raw_energy = ising2d_ham(spin, J=args.J, h=h_val)  # [B]
                        # For Ising, CV is fraction of up spins (between 0 and 1)
                        # x is [B, L*L] with values in {0, 1}, so mean gives fraction of up spins
                        x_up_batch = x.float().mean(dim=1)  # [B]
                        cv_val_for_bias = compute_cv_ising(x) 

                    # Store batch
                    batch_configs.append(x.cpu().numpy())
                    batch_energies.append(raw_energy.cpu().numpy())
                    batch_x_up.append(x_up_batch.cpu().numpy())
                    batch_log_rnd.append(log_rnd.cpu().numpy()) # Store log_rnd
                    batch_log_rw.append(log_rw.cpu().numpy()) # Store log_rw
                    batch_logp_x.append(logp_x.cpu().numpy())
                    batch_logf_t.append(logf_t.cpu().numpy())
                    
                    if bias_pot is not None:
                        # Calculate unbiasing weights = exp(beta_current * V(s))
                        # where beta_current is for the CURRENT sampling temperature (not training temperature!)
                        # s is the CV value (magnetization for Ising, Au concentration for CuAu, or 2D projection for Potts)
                        # Use full CV for bias evaluation (important for Potts with 2D CV)
                        v_s = bias_pot.evaluate(cv_val_for_bias)  # [B] in eV (bias potential values)
                        
                        # Use bias_pot.T for unbiasing (validated to match sampling temperature above)
                        # Since sampling temp matches training temp, we can use the stored T
                        beta_bias = 1.0 / bias_pot.T  # in 1/eV for CuAu, or dimensionless for Ising/Potts
                        
                        # Log-weights: log_w = beta_bias * V(s)
                        # This removes the bias: p_unbiased(x) = p_biased(x) * exp(beta * V(s))
                        log_w = beta_bias * v_s
                        
                        # Clamp log_w to prevent numerical overflow
                        # exp(700) is near float32 max, so clamp to reasonable range
                        log_w_max = 50.0  # exp(50) ≈ 5e21, still manageable
                        log_w_clamped = torch.clamp(log_w, max=log_w_max)
                        
                        # Log warning if values were clamped (indicates potential issue)
                        if (log_w > log_w_max).any():
                            num_clamped = (log_w > log_w_max).sum().item()
                            if samples_collected == 0:  # Only log once per condition
                                max_v = v_s.max().item()
                                max_logw = log_w.max().item()
                                logger.warning(f"Clamped {num_clamped} unbiasing weights (log_w > {log_w_max}). "
                                             f"Max V(s)={max_v:.2f} eV, max log_w={max_logw:.2f}, "
                                             f"beta_bias={beta_bias:.4f}")
                        
                        w = torch.exp(log_w_clamped).detach().cpu().numpy()
                        batch_weights.append(w)
                    
                    samples_collected += current_batch_size
                    pbar.update(current_batch_size)
            
            pbar.close()
            
            # Concatenate
            configs[key] = np.concatenate(batch_configs, axis=0)
            energies[key] = np.concatenate(batch_energies, axis=0)
            x_up[key] = np.concatenate(batch_x_up, axis=0)
            log_rnd_values[key] = np.concatenate(batch_log_rnd, axis=0)
            log_rw_values[key] = np.concatenate(batch_log_rw, axis=0)
            logp_x_values[key] = np.concatenate(batch_logp_x, axis=0)
            logf_t_values[key] = np.concatenate(batch_logf_t, axis=0)
            if batch_weights:
                weights_values[key] = np.concatenate(batch_weights, axis=0)
            
            # Compute aggregate metrics
            # NESS using log_rnd (as requested)
            log_rnd_tensor = torch.tensor(log_rnd_values[key], device=device)
            
            # Diagnostic: log statistics about log_rnd distribution
            log_rnd_mean = log_rnd_tensor.mean().item()
            log_rnd_std = log_rnd_tensor.std().item()
            log_rnd_min = log_rnd_tensor.min().item()
            log_rnd_max = log_rnd_tensor.max().item()
            log_rnd_range = log_rnd_max - log_rnd_min
            logger.info(f"[{key}] log_rnd stats: mean={log_rnd_mean:.4f}, std={log_rnd_std:.4f}, "
                       f"min={log_rnd_min:.4f}, max={log_rnd_max:.4f}, range={log_rnd_range:.4f}")
            
            ness_val = ess(log_rnd_tensor, normalize=True)
            ness_values[key] = float(ness_val) if isinstance(ness_val, (float, int)) else ness_val.item()
            
            # Free Energy F = -1/beta * log(Z) using log_rw (as requested)
            # log Z = logsumexp(log_rw) - log(N)
            log_rw_tensor = torch.tensor(log_rw_values[key], device=device)
            log_Z = torch.logsumexp(log_rw_tensor, dim=0) - np.log(len(log_rw_tensor))
            
            # For CuAu, use temperature in Kelvin; for Ising/Potts, use beta
            if args.model_type == "cuau":
                f_val = -(K_B * temp_k) * log_Z  # F = -kB*T*log(Z) in eV
            else:  # ising or potts
                f_val = -(1.0/beta) * log_Z  # F = -(1/beta)*log(Z)
            
            free_energies[key] = f_val.item()
            
            logger.info(f"[{key}] NESS: {ness_values[key]:.4f}, Free Energy: {free_energies[key]:.4f}")
            
            # If bias was used, calculate and log corrected metrics
            if bias_pot is not None:
                # log_rw here is w.r.t BIASED target.
                # log_weights (unbiasing) = beta * V(s).
                # log_rw_total = log_rw + log_weights.
                # This total weight targets the UNBIASED distribution.
                
                # Reconstruct log_unbias_weights from stored weights
                # weights = exp(log_w). log_w = log(weights)
                w_np = weights_values[key]
                log_w_np = np.log(w_np + 1e-10) # Avoid log(0)
                
                # Physical NESS using log_rnd (consistent with NESS choice)
                log_rnd_total_tensor = torch.tensor(log_rnd_values[key], device=device) + torch.tensor(log_w_np, device=device)
                ness_phys = ess(log_rnd_total_tensor, normalize=True)
                
                # Physical Free Energy using log_rw (consistent with Free Energy choice)
                log_rw_total_tensor = torch.tensor(log_rw_values[key], device=device) + torch.tensor(log_w_np, device=device)
                log_Z_phys = torch.logsumexp(log_rw_total_tensor, dim=0) - np.log(len(log_rw_total_tensor))
                
                # For CuAu, use temperature in Kelvin for free energy calculation
                if args.model_type == "cuau":
                    f_phys = -(K_B * temp_k) * log_Z_phys  # F = -kB*T*log(Z)
                else:
                    f_phys = -(1.0/beta) * log_Z_phys  # F = -(1/beta)*log(Z)
                
                logger.info(f"[{key}] Physical NESS (Corrected): {ness_phys:.4f}, Physical Free Energy: {f_phys.item():.4f}")
                
                # Store these as well (optional, maybe overwrite?)
                # For now let's just log them so user sees the explanation.
                # To make them available in output:
                ness_values[f"{key}_physical"] = float(ness_phys)
                free_energies[f"{key}_physical"] = f_phys.item()

                # Calculate Free Energy Profile from Bias Potential if available
                grid_vals, bias_vals = bias_pot.get_bias_grid_np()
                gamma = bias_pot.gamma
                
                # F(s) ~ - (gamma / (gamma - 1)) * V(s)
                # For CuAu, convert from energy units (eV) to physical units
                if gamma > 1.0:
                    free_energy_profile = - (gamma / (gamma - 1)) * bias_vals
                else:
                    free_energy_profile = - bias_vals 
                
                # For CuAu, convert from eV to kB*T units if needed for consistency
                if args.model_type == "cuau":
                    # bias_vals is in eV (from BiasPotential with T in eV)
                    # Convert to kB*T units for display
                    free_energy_profile = free_energy_profile / (K_B * temp_k)
                # For Ising/Potts, bias_vals is already in dimensionless units, no conversion needed
                    
                # Shift so that the ends are zero (as requested)
                # Average of ends to be robust
                end_avg = (free_energy_profile[0] + free_energy_profile[-1]) / 2.0
                free_energy_profile = free_energy_profile - end_avg
                
                free_energy_profiles[key] = free_energy_profile
                cv_grids[key] = grid_vals

    results = {
        "configs": dict(configs),
        "energies": dict(energies),
        "x_up": dict(x_up),
        "log_rnd": dict(log_rnd_values), 
        "log_rw": dict(log_rw_values), 
        "logp_x": dict(logp_x_values),
        "logf_t": dict(logf_t_values),
        "ness": ness_values,
        "free_energies": free_energies,
        "free_energy_profiles": free_energy_profiles,
        "cv_grids": cv_grids,
        "weights": dict(weights_values),
    }

    return results


def save_results(args, results):
    output_dir = Path(args.output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output_name
    
    logger.info(f"Saving results to {output_path}")
    with open(output_path, "wb") as f:
        pkl.dump(results, f)


def main():
    args = parse_args()
    
    # Set device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Load model
    model, has_conditioning, bias_pot, energy_model = load_model(args, device)
    
    # Run sampling
    
    # Check for conditioning logic requested by user
    # Logic now handled inside load_model and returned as has_conditioning (force_conditioning)
    results = run_sampling(model, args, device, has_conditioning, bias_pot, energy_model)
    
    # Save results
    save_results(args, results)


if __name__ == "__main__":
    main()
