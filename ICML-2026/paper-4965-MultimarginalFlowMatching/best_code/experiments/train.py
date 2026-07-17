#!/usr/bin/env python
"""
Unified training script for OTP-FM experiments.

Usage:
    # Run with dataset defaults
    python experiments/train.py --dataset singlecell

    # Run with specific potential config (layers on top of defaults)
    python experiments/train.py --dataset singlecell --potential W2Inf

    # Run with custom config file
    python experiments/train.py --dataset singlecell --config path/to/config.json

    # Override specific options via CLI
    python experiments/train.py --dataset singlecell --potential KL --epochs 500 --seed 123

    # List all available config options
    python experiments/train.py --list-options

Config files are loaded in order (later overrides earlier):
    1. configs/{dataset}/defaults.json (always loaded)
    2. configs/{dataset}/{potential}.json (if --potential specified)
    3. --config file (if specified)
    4. CLI overrides (--epochs, --seed, --lr, etc.)

Author(s): Raghav Kansal
"""

import argparse
import json
import logging
import re
import sys
from collections import OrderedDict
from pathlib import Path
from textwrap import dedent

import numpy as np
import torch
from otpfm import OTPFM
from otpfm.potentials import (
    KLPotential,
    MMDRBFPotential,
    W2InfPotential,
    W2Potential,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# =============================================================================
# Config Options Documentation
# =============================================================================

CONFIG_OPTIONS = """
Available Configuration Options
===============================

Data Options:
  data_dir              Directory containing data files (default: "data")
  normalize             Normalize data (default: true)
  ot_coupling           Use OT-coupled sampling (default: true)
  ot_method             OT method: "emd" or "sinkhorn" (default: "emd")
  batch_size            Training batch size (default: 256)
  val_split             Validation split fraction (default: 0.1)

Dataset-Specific:
  pca_dim               [singlecell] PCA dimension (default: 100)
  holdout_times         Times to hold out for evaluation
  train_times           Times to use for training

Network Architecture:
  hidden_dim            Hidden layer dimension (default: 256)
  num_hidden_layers     Number of hidden layers (default: 4)
  x_emb_dim             Input embedding dimension (default: 64)
  t_emb_dim             Time embedding dimension (default: 64)
  activation            Activation: "silu", "relu", "gelu" (default: "silu")
  layernorm             Use layer normalization (default: true)
  dropout               Dropout rate (default: 0.0)
  residual_every        Residual connection frequency (default: 2)

Potential Options:
  potential             Type: "w2inf", "w2", "mmd", "kl" (default: "w2inf")
  tks                   List of intermediate times, e.g. [0.5] or [0.33, 0.67]
  strength              Potential strength (default: 100.0)
  strengths             Per-tk strengths (overrides strength)
  lambda_type           Lambda function: "gaussian", "delta", "triangle", "box" (default: "gaussian")
  width                 Lambda width (default: "auto")
  widths                Per-tk widths (overrides width)

Potential-Specific:
  mmd_bandwidth         [mmd] List of bandwidths (default: [3.0])
  kl_rho_method         [kl] Method: "kde", "gaussian" (default: "kde")
  kl_bandwidth          [kl] KDE bandwidth or "silverman" (default: 3.0)

Training Options:
  epochs                Number of epochs (default: 100)
  lr                    Learning rate (default: 0.001)
  optimizer             Optimizer: "adam", "adamw" (default: "adam")
  grad_clip             Gradient clipping (0 = disabled) (default: 0.0)
  seed                  Random seed (default: 42)

OTP-FM Options:
  consistency_loss      Consistency loss type: "meanflow", "imf", "lsd" (default: "meanflow")
  lossfn                Loss function: "adaptive", "mse" (default: "adaptive")
  time_sampler          Time sampling: "uniform_scaled_marginal" (default)
  diag_prob             Probability of diagonal time samples (default: 0.75)
  euler_steps           Euler integration steps (default: 2)
  ema_decay             EMA decay rate (default: 0.99)
  jvp_clamp             JVP clamping value (default: 100.0)
  solver_type           Fixed-point solver (default: null = auto)
  picard_steps          Picard iteration steps (default: 5)

Progressive Training:
  otp_alpha_type        Alpha schedule: "sigmoid", "constant" (default: "sigmoid")
  otp_alpha_slope       Sigmoid slope (default: 6.0)
  otp_alpha_mean_scale  Sigmoid midpoint scale (default: 0.5)

Evaluation:
  sampling_steps        Steps for trajectory sampling (default: 50)
  eval_n_steps          Evaluation trajectory steps (default: 50)
  eval_num_samples      Number of samples for evaluation (default: 2000)
"""


# =============================================================================
# Config Loading
# =============================================================================

CONFIGS_ROOT = Path("OTP-FM/configs")

# Map dataset names to config directory names
DATASET_CONFIG_DIRS = {
    "gaussian": "gaussian",
    "singlecell": "singlecell",
    "citeseq": "citeseq",
    "gulfofmexico": "gom",
    "beijingair": "beijing",
}


def load_json_config(config_path: Path) -> dict:
    """Load a single JSON config file."""
    with open(config_path) as f:
        config = json.load(f)
    # Filter out description/comment fields
    return {k: v for k, v in config.items() if not k.startswith("_") and k != "description"}


def load_config(dataset: str, potential: str = None, config_path: Path = None) -> dict:
    """
    Load config with layered overrides.

    Order of precedence (later overrides earlier):
        1. configs/{dataset}/defaults.json
        2. configs/{dataset}/{potential}.json (if potential specified)
        3. Custom config file (if config_path specified)
    """
    config_dir = DATASET_CONFIG_DIRS.get(dataset, dataset)
    base_config_path = CONFIGS_ROOT / config_dir / "defaults.json"

    # Start with defaults
    if base_config_path.exists():
        config = load_json_config(base_config_path)
        logger.info(f"Loaded defaults from: {base_config_path}")
    else:
        logger.warning(f"No defaults found at {base_config_path}, using empty config")
        config = {}

    # Layer potential-specific config (case-insensitive match)
    if potential:
        potential_dir = CONFIGS_ROOT / config_dir
        potential_config_path = None
        if potential_dir.is_dir():
            for p in potential_dir.glob("*.json"):
                if p.stem.lower() == potential.lower() and p.stem != "defaults":
                    potential_config_path = p
                    break
        if potential_config_path:
            potential_config = load_json_config(potential_config_path)
            config = merge_configs(config, potential_config)
            logger.info(f"Loaded potential config from: {potential_config_path}")
        else:
            logger.warning(f"Potential config not found in {potential_dir} for '{potential}'")

    # Layer custom config file
    if config_path:
        custom_config = load_json_config(config_path)
        config = merge_configs(config, custom_config)
        logger.info(f"Loaded custom config from: {config_path}")

    return config


def merge_configs(base: dict, override: dict) -> dict:
    """Merge override config into base config. Explicit null values override."""
    result = base.copy()
    result.update(override)
    return result


def list_available_configs():
    """List available dataset and potential configs."""
    print("\nAvailable Configurations")
    print("=" * 50)

    if not CONFIGS_ROOT.exists():
        print(f"No configs directory found at {CONFIGS_ROOT}")
        return

    for dataset_dir in sorted(CONFIGS_ROOT.iterdir()):
        if dataset_dir.is_dir():
            print(f"\n{dataset_dir.name}/")
            for config_file in sorted(dataset_dir.glob("*.json")):
                name = config_file.stem
                marker = " (default)" if name == "defaults" else ""
                print(f"  - {name}{marker}")

    print("\n" + CONFIG_OPTIONS)


# =============================================================================
# Potential Creation
# =============================================================================


def create_potential(config: dict, tk: float, strength: float = None, width: float = None):
    """Create a potential based on config."""
    # Support both "width" and "lambda_width" in config for compatibility
    default_width = config.get("width") or config.get("lambda_width", "auto")

    common_kwargs = {
        "tk": tk,
        "strength": strength if strength is not None else config["strength"],
        "lambda_type": config.get("lambda_type", "gaussian"),
        "width": width if width is not None else default_width,
    }

    potential_type = config.get("potential", "w2inf").lower()

    match potential_type:
        case "w2inf":
            return W2InfPotential(**common_kwargs)
        case "w2":
            return W2Potential(**common_kwargs)
        case "mmd":
            return MMDRBFPotential(
                **common_kwargs,
                sigma=config.get("mmd_bandwidth", [3.0]),
            )
        case "kl":
            # Parse bandwidth
            bandwidth = config.get("kl_bandwidth", 3.0)
            try:
                bandwidth = float(bandwidth)
            except (ValueError, TypeError):
                pass  # Keep as string

            return KLPotential(
                **common_kwargs,
                rho_method=config.get("kl_rho_method", "kde"),
                mu_method=config.get("kl_mu_method"),
                bandwidth=bandwidth,
            )
        case _:
            raise ValueError(f"Unknown potential type: {potential_type}")


def create_potentials(config: dict) -> OrderedDict:
    """Create all potentials from config."""
    potentials = OrderedDict()
    tks = config["tks"]

    # Handle per-potential strengths/widths
    strengths = config.get("strengths") or [config["strength"]] * len(tks)
    default_w = config.get("width") or config.get("lambda_width", 0.2)
    widths = config.get("widths") or [default_w] * len(tks)

    for tk, strength, width in zip(tks, strengths, widths):
        potentials[tk] = create_potential(config, tk, strength=strength, width=width)

    return potentials


# =============================================================================
# Model Creation
# =============================================================================


def create_model(config: dict, dim: int, potentials: OrderedDict, device: str) -> OTPFM:
    """Create OTPFM model from config."""
    flownet_args = {
        "x_emb_dim": config.get("x_emb_dim", 64),
        "t_emb_dim": config.get("t_emb_dim", 64),
        "hidden_dim": config.get("hidden_dim", 256),
        "num_hidden_layers": config.get("num_hidden_layers", 4),
        "x_hidden_layers": config.get("x_hidden_layers", 0),
        "t_hidden_layers": config.get("t_hidden_layers", 0),
        "activation_fn": config.get("activation", "silu"),
        "layernorm": config.get("layernorm", True),
        "dropout": config.get("dropout", 0.0),
        "residual_every": config.get("residual_every", 2),
    }

    model = OTPFM(
        d=dim,
        tks=list(potentials.keys()),
        potentials=potentials,
        flownet_args=flownet_args,
        consistency_loss=config.get("consistency_loss", "meanflow"),
        lossfn=config.get("lossfn", "adaptive"),
        time_sampler=config.get("time_sampler", "uniform_scaled_marginal"),
        diag_prob=config.get("diag_prob", 0.75),
        euler_steps=config.get("euler_steps", 2),
        ema_decay=config.get("ema_decay", 0.99),
        ema_ot=config.get("ema_ot", True),
        jvp_clamp=config.get("jvp_clamp", 100.0),
        solver_type=config.get("solver_type"),
        picard_steps=config.get("picard_steps", 5),
        adaptive_exp=config.get("adaptive_exp", 1.0),
        x_pred=config.get("x_pred", False),
    ).to(device)

    return model


# =============================================================================
# Run Number Management
# =============================================================================


def get_next_run_number(parent_dir: Path) -> int:
    """Get next run number from existing directories."""
    if not parent_dir.exists():
        return 1

    max_num = 0
    for item in parent_dir.iterdir():
        if item.is_dir():
            match = re.match(r"^(\d+)_", item.name)
            if match:
                max_num = max(max_num, int(match.group(1)))

    return max_num + 1


def build_tag(config: dict, base_tag: str) -> str:
    """Build a descriptive tag from config."""
    parts = [base_tag]
    parts.append(config.get("potential", "w2inf"))
    parts.append(f"s{config.get('strength', 100)}")
    parts.append(f"w{config.get('width') or config.get('lambda_width', 0.2)}")
    parts.append(f"lr{config.get('lr', 0.001)}")
    parts.append(f"nl{config.get('num_hidden_layers', 4)}")
    return "_".join(parts)


# =============================================================================
# Dataset-Specific Setup
# =============================================================================


def get_dataset_trainer_args(dataset: str, setup_result: dict, config: dict) -> dict:
    """Get dataset-specific trainer kwargs.

    Returns only the kwargs needed for the given dataset's trainer,
    avoiding passing unnecessary args across trainers.
    """
    if dataset == "gaussian":
        return {}

    # Real-world datasets: singlecell, gulfofmexico, beijingair
    args = {}
    for key in ("marginals", "scaler", "holdout_times", "train_times"):
        if key in setup_result:
            args[key] = setup_result[key]

    # Only pass if explicitly configured; otherwise use the trainer's per-dataset defaults
    for key in ("eval_n_steps", "eval_num_samples", "eval_metrics", "traj_skips", "save_skips"):
        if key in config:
            args[key] = config[key]

    return args


def setup_gaussian(config: dict, device: str):
    """Set up Gaussian experiment."""
    from experiments.gaussian import data
    from experiments.gaussian import trainer as gaussian_trainer

    # Load data
    train_loader, val_loader, marginals = data.create_gaussian_dataloaders(
        means=config.get("means", [0.0, 1.0, 0.5]),
        stds=config.get("stds", [0.3, 0.2, 0.4]),
        n_samples=config.get("n_samples", 2000),
        batch_size=config.get("batch_size", 256),
        val_split=config.get("val_split", 0.1),
    )

    dim = config.get("dim", 1)
    potentials = create_potentials(config)
    model = create_model(config, dim, potentials, device)

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "marginals": marginals,
        "model": model,
        "potentials": potentials,
        "trainer_class": gaussian_trainer.GaussianTrainer,
        "dim": dim,
    }


def setup_singlecell(config: dict, device: str):
    """Set up single-cell experiment."""
    from experiments.singlecell import data
    from experiments.singlecell import trainer as singlecell_trainer

    # Load data
    result = data.load_eb_data(
        data_dir=Path(config.get("data_dir", "data")),
        pca_dim=config.get("pca_dim", 100),
        normalize=config.get("normalize", True),
        ot_coupling=config.get("ot_coupling", True),
        ot_method=config.get("ot_method", "emd"),
        holdout_times=config.get("holdout_times", [1, 3]),
    )

    train_loader, val_loader = data.create_eb_dataloaders(
        result["pcs"],
        result["labels"],
        holdout_times=config.get("holdout_times", [1, 3]),
        batch_size=config.get("batch_size", 256),
        val_split=config.get("val_split", 0.1),
        ot_alignments=result.get("ot_alignments"),
    )

    dim = config.get("pca_dim", 100)

    # Auto-compute evenly-spaced tks when the number of intermediate training
    # marginals differs from the configured number of potentials.
    train_times = sorted(result.get("train_times", [0, 2, 4]))
    n_intermediate = len(train_times) - 2  # exclude source and target
    n_configured = len(config.get("tks", [0.5]))
    if n_intermediate != n_configured and n_intermediate > 0:
        auto_tks = [(i + 1) / (n_intermediate + 1) for i in range(n_intermediate)]
        logger.info(
            f"Auto-computed evenly-spaced tks for {n_intermediate} intermediate "
            f"marginals (train_times={train_times}): {auto_tks} "
            f"(overriding config tks={config.get('tks')})"
        )
        config["tks"] = auto_tks

    potentials = create_potentials(config)
    model = create_model(config, dim, potentials, device)

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "marginals": result["marginals"],
        "scaler": result.get("scaler"),
        "model": model,
        "potentials": potentials,
        "trainer_class": singlecell_trainer.EBTrainer,
        "dim": dim,
        "holdout_times": config.get("holdout_times", [1, 3]),
        "train_times": train_times,
    }


def setup_citeseq(config: dict, device: str):
    """Set up CITE-seq experiment."""
    from experiments.citeseq import data
    from experiments.citeseq import trainer as citeseq_trainer

    result = data.load_citeseq_data(
        data_dir=Path(config.get("data_dir", "data")),
        pca_dim=config.get("pca_dim", 50),
        normalize=config.get("normalize", True),
        ot_coupling=config.get("ot_coupling", True),
        ot_method=config.get("ot_method", "emd"),
        holdout_times=config.get("holdout_times", [1]),
    )

    train_loader, val_loader = data.create_citeseq_dataloaders(
        result["pcs"],
        result["labels"],
        holdout_times=config.get("holdout_times", [1]),
        batch_size=config.get("batch_size", 256),
        val_split=config.get("val_split", 0.1),
        ot_alignments=result.get("ot_alignments"),
    )

    dim = config.get("pca_dim", 50)

    # Auto-compute tks: cite-seq has 4 timepoints, with holdout the remaining
    # train times determine the number of intermediate marginals.
    train_times = sorted(result.get("train_times", [0, 2, 3]))
    n_intermediate = len(train_times) - 2
    n_configured = len(config.get("tks", [0.5]))
    if n_intermediate != n_configured and n_intermediate > 0:
        auto_tks = [(i + 1) / (n_intermediate + 1) for i in range(n_intermediate)]
        logger.info(
            f"Auto-computed evenly-spaced tks for {n_intermediate} intermediate "
            f"marginals (train_times={train_times}): {auto_tks} "
            f"(overriding config tks={config.get('tks')})"
        )
        config["tks"] = auto_tks

    potentials = create_potentials(config)
    model = create_model(config, dim, potentials, device)

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "marginals": result["marginals"],
        "scaler": result.get("scaler"),
        "model": model,
        "potentials": potentials,
        "trainer_class": citeseq_trainer.CiteSeqTrainer,
        "dim": dim,
        "holdout_times": config.get("holdout_times", [1]),
        "train_times": train_times,
    }


def setup_gulfofmexico(config: dict, device: str):
    """Set up Gulf of Mexico experiment."""
    from experiments.gulfofmexico import data
    from experiments.gulfofmexico import trainer as gom_trainer

    # Load data
    result = data.load_gom_data(
        data_dir=Path(config.get("data_dir", "data")),
        normalize=config.get("normalize", True),
        ot_coupling=config.get("ot_coupling", True),
        ot_method=config.get("ot_method", "emd"),
        train_times=config.get("train_times", [0, 2, 4, 6, 8]),
        holdout_times=config.get("holdout_times", [1, 3, 5, 7]),
    )

    train_loader, val_loader = data.create_gom_dataloaders(
        result["marginals_list"],
        batch_size=config.get("batch_size", 64),
        holdout_times=config.get("holdout_times", [1, 3, 5, 7]),
        val_split=config.get("val_split", 0.0),
        ot_alignments=result.get("ot_alignments"),
    )

    dim = 2  # GoM is always 2D
    potentials = create_potentials(config)
    model = create_model(config, dim, potentials, device)

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "marginals": result["marginals"],
        "scaler": result.get("scaler"),
        "model": model,
        "potentials": potentials,
        "trainer_class": gom_trainer.GoMTrainer,
        "dim": dim,
        "holdout_times": config.get("holdout_times", [1, 3, 5, 7]),
        "train_times": config.get("train_times", [0, 2, 4, 6, 8]),
    }


def setup_beijingair(config: dict, device: str):
    """Set up Beijing Air Quality experiment."""
    from experiments.beijingair import data
    from experiments.beijingair import trainer as beijing_trainer

    # Load data
    result = data.load_beijing_data(
        data_dir=Path(config.get("data_dir", "data/beijing")),
        normalize=config.get("normalize", True),
        ot_coupling=config.get("ot_coupling", True),
        ot_method=config.get("ot_method", "emd"),
        train_times=config.get("train_times", [0, 1, 3, 4, 6, 7, 9, 10, 12]),
        holdout_times=config.get("holdout_times", [2, 5, 8, 11]),
    )

    train_loader, val_loader = data.create_beijing_dataloaders(
        result["marginals_list"],
        batch_size=config.get("batch_size", 128),
        holdout_times=config.get("holdout_times", [2, 5, 8, 11]),
        val_split=config.get("val_split", 0.0),
        ot_alignments=result.get("ot_alignments"),
    )

    dim = result.get("dim", 1)
    potentials = create_potentials(config)
    model = create_model(config, dim, potentials, device)

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "marginals": result["marginals"],
        "scaler": result.get("scaler"),
        "model": model,
        "potentials": potentials,
        "trainer_class": beijing_trainer.BeijingTrainer,
        "dim": dim,
        "holdout_times": config.get("holdout_times", [2, 5, 8, 11]),
        "train_times": config.get("train_times", [0, 1, 3, 4, 6, 7, 9, 10, 12]),
    }


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Train OTP-FM on various datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=dedent(
            """
            Examples:
              python experiments/train.py --dataset singlecell
              python experiments/train.py --dataset singlecell --potential W2Inf
              python experiments/train.py --dataset gulfofmexico --potential KL --epochs 500
              python experiments/train.py --list-options
        """
        ),
    )

    # Main arguments
    parser.add_argument(
        "--dataset",
        type=str.lower,
        choices=["gaussian", "singlecell", "citeseq", "gulfofmexico", "beijingair"],
        help="Dataset to train on",
    )
    parser.add_argument(
        "--potential",
        type=str,
        help="Potential config to load (e.g., W2, W2Inf, KL, MMD)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to custom JSON config file",
    )

    # Run configuration
    parser.add_argument("--tag", type=str, default="run", help="Tag for this run")
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to save results (default: results/{dataset}/{tag})",
    )
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu/mps)")

    # Common overrides
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--strength", type=float, help="Potential strength")
    parser.add_argument("--width", type=float, help="Lambda width")
    parser.add_argument("--holdout-times", type=int, nargs="+", help="Times to hold out")
    parser.add_argument("--pca-dim", type=int, help="[singlecell] PCA dimensions")
    parser.add_argument("--lossfn", type=str, help="Loss function")
    parser.add_argument("--consistency-loss", type=str, help="Consistency loss type")
    parser.add_argument("--hidden-dim", type=int, help="Hidden layer dimension")
    parser.add_argument("--num-hidden-layers", type=int, help="Number of hidden layers")
    parser.add_argument("--dropout", type=float, help="Dropout rate")
    parser.add_argument(
        "--otp-alpha-mean-scale",
        type=float,
        help="Sigmoid midpoint scale (higher = more time at low alpha)",
    )
    parser.add_argument(
        "--otp-alpha-slope", type=float, help="Sigmoid slope (steepness of transition)"
    )
    parser.add_argument(
        "--x-pred",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable x-prediction mode",
    )

    # Utility
    parser.add_argument(
        "--list-options",
        action="store_true",
        help="List all available config options and exit",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    # Handle --list-options
    if args.list_options:
        list_available_configs()
        sys.exit(0)

    # Require dataset if not listing options
    if not args.dataset:
        parser.error("--dataset is required (or use --list-options)")

    # Set debug logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load config with layered overrides
    config = load_config(
        dataset=args.dataset,
        potential=args.potential,
        config_path=Path(args.config) if args.config else None,
    )

    # Override with CLI args
    cli_overrides = {
        "seed": args.seed,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "strength": args.strength,
        "width": args.width,
        "holdout_times": args.holdout_times,
        "pca_dim": args.pca_dim,
        "lossfn": args.lossfn,
        "consistency_loss": args.consistency_loss,
        "hidden_dim": args.hidden_dim,
        "num_hidden_layers": args.num_hidden_layers,
        "dropout": args.dropout,
        "otp_alpha_mean_scale": args.otp_alpha_mean_scale,
        "otp_alpha_slope": args.otp_alpha_slope,
        "x_pred": args.x_pred,
    }
    for key, value in cli_overrides.items():
        if value is not None:
            config[key] = value

    # Set random seed
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    logger.info(f"Random seed: {seed}")

    # Set device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    logger.info(f"Device: {device}")

    # Dataset-specific setup
    setup_funcs = {
        "gaussian": setup_gaussian,
        "singlecell": setup_singlecell,
        "citeseq": setup_citeseq,
        "gulfofmexico": setup_gulfofmexico,
        "beijingair": setup_beijingair,
    }

    setup_result = setup_funcs[args.dataset](config, device)

    # Create save directory - automatically adds important hyperparameters and a run number
    base_save_dir = Path(args.save_dir) if args.save_dir else (Path("results") / args.dataset)
    if args.tag:
        save_dir = base_save_dir / args.tag
        parent_dir = save_dir.parent
        tag = save_dir.name
    else:
        parent_dir = base_save_dir
        tag = None
    parent_dir.mkdir(parents=True, exist_ok=True)

    run_num = get_next_run_number(parent_dir)
    tag_suffix = build_tag(config, tag)
    full_save_dir = parent_dir / f"{run_num}_{tag_suffix}"
    full_save_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Save directory: {full_save_dir}")

    # Save config
    config_save_path = full_save_dir / "config.json"
    with open(config_save_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Config saved to {config_save_path}")

    # Create trainer
    TrainerClass = setup_result["trainer_class"]
    dataset_args = get_dataset_trainer_args(args.dataset, setup_result, config)
    trainer = TrainerClass(
        model=setup_result["model"],
        train_loader=setup_result["train_loader"],
        val_loader=setup_result["val_loader"],
        save_dir=full_save_dir,
        lr=config.get("lr", 0.001),
        epochs=config.get("epochs", 100),
        optimizer=config.get("optimizer", "adam"),
        grad_clip=config.get("grad_clip", 0.0),
        weight_decay=config.get("weight_decay", 0.0),
        lr_schedule=config.get("lr_schedule", None),
        do_otp=True,
        otp_alpha_type=config.get("otp_alpha_type", "sigmoid"),
        otp_alpha_slope=config.get("otp_alpha_slope", 6.0),
        otp_alpha_mean_scale=config.get("otp_alpha_mean_scale", 0.5),
        sampling_steps=config.get("sampling_steps", 50),
        potentials=setup_result["potentials"],
        device=device,
        **dataset_args,
    )

    # Train
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting training: {args.dataset}")
    logger.info(f"{'='*60}")

    trainer.train()
    trainer.post_training(show=False)

    logger.info(f"\n{'='*60}")
    logger.info("Training complete!")
    logger.info(f"Results saved to: {full_save_dir}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
