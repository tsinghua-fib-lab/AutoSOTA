#!/usr/bin/env python
"""Blockwise MCMC sampling utility for the SnowyFlow Ising benchmarks.

* Generates samples for each (temperature, chemical potential) pair.
* Runs the sampler in sequential blocks of fixed length (``steps-per-block``).
* Persists the requested statistics after each block so downstream notebooks
  can quantify convergence using KL/JS divergence against the final block.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle as pkl
import random
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from clease.calculator import attach_calculator  # noqa: F401
from clease.settings import CEBulk, Concentration

from baselines.constants import K_B
from baselines.energy.ce import AuCuAlloyModel  # type: ignore
from baselines.energy.ising import LatticeIsingModel  # type: ignore
from baselines.energy.ising import LatticePottsModel


@dataclass
class BlockSummary:
    """Small metadata bundle describing a saved block."""

    block_index: int
    start_step: int
    end_step: int
    num_samples: int
    path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run blockwise Gibbs sampling for Ising or CuAu models and persist intermediate results."
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["ising", "cuau", "potts"],
        default="ising",
        help="Type of model to use: 'ising' for Ising model, 'cuau' for Au-Cu alloy model, 'potts' for Potts model.",
    )
    parser.add_argument(
        "--temps",
        type=float,
        nargs="+",
        default=[1.5, 2.0, 2.5, 3.0, 3.5],
        help="Temperatures (in Kelvin) to evaluate.",
    )
    parser.add_argument(
        "--chem-pots",
        type=float,
        nargs="+",
        default=[-0.4, -0.2, 0.0, 0.2, 0.4],
        help="Chemical potentials to evaluate.",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=10,
        help="Linear lattice dimension; the sampler runs on a dim x dim lattice (for Ising and Potts models).",
    )
    parser.add_argument(
        "--q",
        type=int,
        default=3,
        help="Number of spin states for Potts model (q >= 2). Defaults to 3. Only used for --model-type potts.",
    )
    parser.add_argument(
        "--eci-file",
        type=str,
        default=None,
        help="Path to ECI JSON file for CuAu model initialization (required for --model-type cuau).",
    )
    parser.add_argument(
        "--supercell",
        type=int,
        nargs=3,
        default=[2, 2, 4],
        help="Supercell size [nx, ny, nz] for CuAu model (default: [2, 2, 4]).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10_000,
        help="Total number of parallel chains to evolve per block.",
    )
    parser.add_argument(
        "--samples-per-block",
        type=int,
        default=10_000,
        help="Number of samples to persist from each block (taken from the tail).",
    )
    parser.add_argument(
        "--steps-per-block",
        type=int,
        default=200,
        help="Number of Gibbs steps to run inside each block.",
    )
    parser.add_argument(
        "--num-blocks",
        type=int,
        default=10,
        help="How many sequential blocks to run per (temp, chem_pot) pair.",
    )
    parser.add_argument(
        "--init-sigma",
        type=float,
        default=1.0,
        help="Standard deviation used to seed the Ising sampler.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/mcmc_blocks",
        help="Directory to store block outputs and metadata (created if missing).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducibility.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing outputs for a (temp, chem_pot) pair.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Verbosity for console logging.",
    )
    parser.add_argument(
        "--continue",
        type=str,
        default=None,
        dest="continue_from",
        help="Path to metadata.json file to continue sampling from. Loads configuration and last block states.",
    )
    parser.add_argument(
        "--arm-seeds",
        type=str,
        default=None,
        help="Path to ARM sampling results pickle file to use as initial seeds. Configs from ARM will be treated as effective step 1.",
    )
    args = parser.parse_args()

    # Validate CuAu-specific arguments
    if args.model_type == "cuau" and args.eci_file is None:
        raise ValueError(
            "--eci-file is required when --model-type is 'cuau'. "
            "Please provide the path to the ECI JSON file."
        )

    return args


def set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def create_model(
    model_type: str,
    dim: int | None = None,
    init_sigma: float = 1.0,
    batch_size: int = 1250,
    eci_file: str | None = None,
    supercell: list[int] | None = None,
    q: int = 3,
) -> LatticeIsingModel | AuCuAlloyModel | LatticePottsModel:
    """Create and return a model instance based on model_type.

    Args:
        model_type: Type of model to create ("ising", "cuau", or "potts").
        dim: Linear dimension for Ising/Potts model (required for "ising" and "potts").
        init_sigma: Initial sigma for Ising/Potts model.
        batch_size: Batch size for model initialization.
        eci_file: Path to ECI JSON file (required for "cuau").
        supercell: Supercell size [nx, ny, nz] (required for "cuau").
        q: Number of spin states for Potts model (required for "potts").

    Returns:
        Model instance (LatticeIsingModel, AuCuAlloyModel, or LatticePottsModel).
    """
    if model_type == "ising":
        if dim is None:
            raise ValueError("dim is required for Ising model")
        return LatticeIsingModel(dim, init_sigma=init_sigma, n_samples=batch_size)
    elif model_type == "potts":
        if dim is None:
            raise ValueError("dim is required for Potts model")
        if q < 2:
            raise ValueError("q must be >= 2 for Potts model")
        return LatticePottsModel(dim, q=q, init_sigma=init_sigma, n_samples=batch_size)
    elif model_type == "cuau":
        if eci_file is None:
            raise ValueError("eci_file is required for CuAu model")
        if supercell is None:
            supercell = [2, 2, 4]

        # Load ECI file
        eci_path = Path(eci_file).expanduser().resolve()
        if not eci_path.exists():
            raise FileNotFoundError(f"ECI file not found: {eci_path}")
        with eci_path.open("r", encoding="utf-8") as f:
            eci = json.load(f)

        # Create CEBulk settings
        conc = Concentration(basis_elements=[["Au", "Cu"]])
        settings = CEBulk(
            crystalstructure="fcc",
            a=3.8,
            size=supercell,
            concentration=conc,
            db_name="aucu_dft.db",
            max_cluster_dia=[6.0, 4.5, 4.5],
        )
        atoms = settings.atoms.copy()

        # Create and return AuCuAlloyModel
        return AuCuAlloyModel(structure=atoms, settings=settings, eci=eci)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Must be 'ising', 'cuau', or 'potts'.")


def _format_key(temp: float, chem_pot: float) -> str:
    return f"{temp:.1f}K_mu{chem_pot:.2f}"


def _subset_tail(tensor: torch.Tensor, num_items: int) -> torch.Tensor:
    if num_items >= tensor.shape[0]:
        return tensor
    return tensor[-num_items:]


def _save_block(
    *,
    block_dir: Path,
    block_index: int,
    configs: dict[str, np.ndarray],
    energies: dict[str, np.ndarray],
    x_up: dict[str, np.ndarray],
    metadata: dict[str, object],
) -> Path:
    block_dir.mkdir(parents=True, exist_ok=True)
    block_path = block_dir / f"block_{block_index:04d}.pkl"

    data = {
        "configs": configs,
        "energies": energies,
        "x_up": x_up,
        "metadata": metadata,
    }
    with block_path.open("wb") as f:
        pkl.dump(data, f)
    return block_path


def _initialise_states(
    model: LatticeIsingModel | AuCuAlloyModel | LatticePottsModel,
    temps: list[float],
    chem_pots: list[float],
    *,
    batch_size: int,
    steps_per_block: int,
    logger: logging.Logger,
) -> dict[tuple[float, float], torch.Tensor]:
    states: dict[tuple[float, float], torch.Tensor] = {}
    for chem_pot in chem_pots:
        logger.info("Initialising annealed state for chem_pot=%.2f", chem_pot)
        samples = model.init_sample(batch_size)
        for temp in reversed(temps):
            temps_tensor = torch.full(
                (batch_size,), temp, dtype=torch.float32, device=samples.device
            )
            fields_tensor = torch.full(
                (batch_size,), chem_pot, dtype=torch.float32, device=samples.device
            )
            samples = model.generate_samples(
                n_samples=batch_size,
                temps=temps_tensor,
                fields=fields_tensor,
                gt_steps=steps_per_block,
                rand=True,
                starting_samples=samples,
            )
            states[(temp, chem_pot)] = samples.clone().detach()
    return states


def _load_arm_states(
    arm_results_path: Path,
    temps: list[float],
    chem_pots: list[float],
    batch_size: int,
    num_sites: int,
    logger: logging.Logger,
) -> dict[tuple[float, float], torch.Tensor]:
    """Load states from ARM sampling results to use as seeds.

    ARM results contain configs with shape (num_samples, num_sites).
    The model expects flattened tensors (batch_size, num_sites),
    so we keep them flattened (don't reshape to 3D).

    Args:
        arm_results_path: Path to ARM results pickle file.
        temps: List of temperatures.
        chem_pots: List of chemical potentials.
        batch_size: Batch size for samples.
        num_sites: Number of sites (dim*dim for Ising, num_sites for CuAu).
        logger: Logger instance.
    """
    logger.info("Loading ARM states from %s", arm_results_path)
    with arm_results_path.open("rb") as f:
        arm_data = pkl.load(f)

    if "configs" not in arm_data:
        raise KeyError(
            f"'configs' key not found in ARM results. Available keys: {list(arm_data.keys())}"
        )

    configs = arm_data["configs"]
    states: dict[tuple[float, float], torch.Tensor] = {}

    for chem_pot in chem_pots:
        for temp in temps:
            key = _format_key(temp, chem_pot)
            if key not in configs:
                raise KeyError(
                    f"Key {key} not found in ARM results. Available keys: {list(configs.keys())}"
                )

            # Load the configs from ARM results
            config_array = configs[key]  # shape: (num_samples, num_sites)
            # Handle different possible shapes
            if config_array.ndim == 2:
                # Shape is (num_samples, num_sites) - keep flattened, model expects this
                num_samples, config_num_sites = config_array.shape
                if config_num_sites != num_sites:
                    raise ValueError(
                        f"ARM configs for {key} have {config_num_sites} sites, "
                        f"but model expects {num_sites} sites. Shape: {config_array.shape}"
                    )
                # Keep as (num_samples, num_sites) - model expects flattened tensors
            elif config_array.ndim == 3:
                # Already in (num_samples, dim1, dim2) format - flatten it
                num_samples = config_array.shape[0]
                # Flatten to (num_samples, num_sites) for the model
                config_array = config_array.reshape(num_samples, -1)
                if config_array.shape[1] != num_sites:
                    raise ValueError(
                        f"ARM configs for {key} have {config_array.shape[1]} sites after flattening, "
                        f"but model expects {num_sites} sites."
                    )
            else:
                raise ValueError(
                    f"Unexpected config shape for {key}: {config_array.shape}. "
                    "Expected (num_samples, num_sites) or (num_samples, dim, dim)."
                )

            # If we have fewer samples than batch_size, repeat to fill batch
            if num_samples < batch_size:
                logger.warning(
                    "ARM results have %d samples for %s, but batch_size is %d. "
                    "Repeating samples to fill batch.",
                    num_samples,
                    key,
                    batch_size,
                )
                repeat_factor = (batch_size // num_samples) + 1
                config_array = np.tile(config_array, (repeat_factor, 1))[:batch_size]
            elif num_samples > batch_size:
                # Take the last batch_size samples
                logger.info(
                    "ARM results have %d samples for %s, taking last %d samples.",
                    num_samples,
                    key,
                    batch_size,
                )
                config_array = config_array[-batch_size:]

            # Convert to torch tensor and ensure correct dtype
            # ARM configs are typically int/long, convert to float32 for model
            config_tensor = torch.from_numpy(config_array).to(torch.float32)
            states[(temp, chem_pot)] = config_tensor

    logger.info("Loaded ARM states for %d (temp, chem_pot) pairs", len(states))
    logger.info("ARM states are treated as effective step 1 for MCMC continuation.")
    return states


def _load_last_block_states(
    last_block_path: Path,
    temps: list[float],
    chem_pots: list[float],
    batch_size: int,
    logger: logging.Logger,
) -> dict[tuple[float, float], torch.Tensor]:
    """Load states from the last saved block to continue sampling."""
    logger.info("Loading states from last block: %s", last_block_path)
    with last_block_path.open("rb") as f:
        block_data = pkl.load(f)

    configs = block_data["configs"]
    states: dict[tuple[float, float], torch.Tensor] = {}

    for chem_pot in chem_pots:
        for temp in temps:
            key = _format_key(temp, chem_pot)
            if key not in configs:
                raise KeyError(
                    f"Key {key} not found in last block. Available keys: {list(configs.keys())}"
                )

            # Load the configs and convert to torch tensor
            config_array = configs[key]  # shape: (num_samples, dim, dim)
            num_samples = config_array.shape[0]

            # If we have fewer samples than batch_size, we'll need to handle this
            if num_samples < batch_size:
                logger.warning(
                    "Last block has %d samples for %s, but batch_size is %d. "
                    "Repeating samples to fill batch.",
                    num_samples,
                    key,
                    batch_size,
                )
                # Repeat samples to fill batch_size
                repeat_factor = (batch_size // num_samples) + 1
                config_array = np.tile(config_array, (repeat_factor, 1, 1))[:batch_size]
            elif num_samples > batch_size:
                # Take the last batch_size samples
                config_array = config_array[-batch_size:]

            # Convert to torch tensor and ensure correct dtype
            # Configs are saved as int8, convert to float32 for model
            config_tensor = torch.from_numpy(config_array).to(torch.float32)
            states[(temp, chem_pot)] = config_tensor

    logger.info("Loaded states for %d (temp, chem_pot) pairs", len(states))
    return states


def _load_metadata(
    metadata_path: Path, logger: logging.Logger
) -> tuple[dict, list[BlockSummary], int]:
    """Load metadata.json and return configuration, existing blocks, and next block index."""
    logger.info("Loading metadata from %s", metadata_path)
    with metadata_path.open("r") as f:
        metadata = json.load(f)

    # Convert block summaries
    block_summaries = [BlockSummary(**block_dict) for block_dict in metadata.get("blocks", [])]

    # Determine next block index
    if block_summaries:
        next_block_index = max(block.block_index for block in block_summaries) + 1
    else:
        next_block_index = 0

    logger.info(
        "Found %d existing blocks. Will continue from block %d",
        len(block_summaries),
        next_block_index,
    )

    return metadata, block_summaries, next_block_index


def run_blockwise_sampler(
    temps: Iterable[float],
    chem_pots: Iterable[float],
    *,
    model_type: str,
    dim: int | None = None,
    batch_size: int,
    samples_per_block: int,
    steps_per_block: int,
    num_blocks: int,
    init_sigma: float,
    output_dir: Path,
    overwrite: bool,
    continue_from: str | None,
    arm_seeds: str | None,
    eci_file: str | None = None,
    supercell: list[int] | None = None,
    q: int = 3,
    logger: logging.Logger,
) -> None:
    temps = list(temps)
    chem_pots = list(chem_pots)

    # Determine num_sites based on model type
    if model_type == "ising":
        if dim is None:
            raise ValueError("dim is required for Ising model")
        num_sites = dim * dim
    elif model_type == "potts":
        if dim is None:
            raise ValueError("dim is required for Potts model")
        num_sites = dim * dim
    elif model_type == "cuau":
        if supercell is None:
            supercell = [2, 2, 4]
        num_sites = supercell[0] * supercell[1] * supercell[2]
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Create model
    model = create_model(
        model_type=model_type,
        dim=dim,
        init_sigma=init_sigma,
        batch_size=batch_size,
        eci_file=eci_file,
        supercell=supercell,
        q=q,
    )

    # Handle continuation from existing metadata
    block_summaries: list[BlockSummary] = []
    start_block_index = 0

    if continue_from is not None:
        metadata_path = Path(continue_from).expanduser().resolve()
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        existing_metadata, existing_blocks, start_block_index = _load_metadata(
            metadata_path, logger
        )
        block_summaries = existing_blocks

        # Override parameters from metadata
        temps = existing_metadata["temps"]
        chem_pots = existing_metadata["chem_pots"]
        # Default to ising for backward compat
        model_type = existing_metadata.get("model_type", "ising")
        batch_size = existing_metadata["batch_size"]
        steps_per_block = existing_metadata["steps_per_block"]
        samples_per_block = existing_metadata["samples_per_block"]
        init_sigma = existing_metadata["init_sigma"]
        output_dir = metadata_path.parent  # Use the directory containing metadata.json

        # Get model-specific parameters
        if model_type == "ising":
            dim = existing_metadata["dim"]
            num_sites = dim * dim
            eci_file = None
            supercell = None
            q = None
        elif model_type == "potts":
            dim = existing_metadata["dim"]
            num_sites = dim * dim
            eci_file = None
            supercell = None
            q = existing_metadata.get("q", 3)
        elif model_type == "cuau":
            dim = None
            supercell = existing_metadata.get("supercell", [2, 2, 4])
            num_sites = supercell[0] * supercell[1] * supercell[2]
            eci_file = existing_metadata.get("eci_file")
            q = None
        else:
            raise ValueError(f"Unknown model_type in metadata: {model_type}")

        # Determine target number of blocks
        # If user specified a higher num_blocks, use that; otherwise use original target
        original_num_blocks = existing_metadata.get("num_blocks", start_block_index)
        if num_blocks <= original_num_blocks:
            # Use original target (user didn't override or wants fewer, which we'll warn about)
            if num_blocks < original_num_blocks:
                logger.warning(
                    "Requested num_blocks (%d) is less than original target (%d). "
                    "Using original target of %d blocks.",
                    num_blocks,
                    original_num_blocks,
                    original_num_blocks,
                )
            num_blocks = original_num_blocks
        # else: user wants more blocks than original, use their specified num_blocks

        logger.info(
            "Continuing from existing run: model_type=%s, temps=%s, chem_pots=%s, "
            "batch_size=%d, steps_per_block=%d, starting at block %d, target total blocks: %d",
            model_type,
            temps,
            chem_pots,
            batch_size,
            steps_per_block,
            start_block_index,
            num_blocks,
        )

        # Recreate model with correct parameters
        model_q = q if model_type == "potts" and q is not None else (3 if model_type == "potts" else 3)
        model = create_model(
            model_type=model_type,
            dim=dim,
            init_sigma=init_sigma,
            batch_size=batch_size,
            eci_file=eci_file,
            supercell=supercell,
            q=model_q,
        )

        # Get device from model
        if isinstance(model, (LatticeIsingModel, LatticePottsModel)):
            device = (
                next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")
            )
        else:  # AuCuAlloyModel
            # For AuCuAlloyModel, check if sampler has parameters, otherwise use CPU
            if hasattr(model, "sampler") and list(model.sampler.parameters()):
                device = next(model.sampler.parameters()).device
            else:
                device = torch.device("cpu")

        # Load states from last block
        if existing_blocks:
            last_block_path = Path(existing_blocks[-1].path)
            states = _load_last_block_states(
                last_block_path,
                temps,
                chem_pots,
                batch_size,
                logger,
            )
            # Move states to the correct device
            states = {k: v.to(device) for k, v in states.items()}
        else:
            logger.warning("No existing blocks found, initializing from scratch")
            states = _initialise_states(
                model,
                temps,
                chem_pots,
                batch_size=batch_size,
                steps_per_block=steps_per_block,
                logger=logger,
            )
    else:
        logger.info(
            "Starting blockwise sampling: model_type=%s, temps=%s, chem_pots=%s, "
            "batch_size=%d, steps_per_block=%d, num_blocks=%d, num_sites=%d",
            model_type,
            temps,
            chem_pots,
            batch_size,
            steps_per_block,
            num_blocks,
            num_sites,
        )

        block_dir = output_dir
        block_dir.mkdir(parents=True, exist_ok=True)
        existing_blocks = list(block_dir.glob("block_*.pkl"))
        if existing_blocks and not overwrite:
            raise FileExistsError(
                f"Block outputs already exist in '{block_dir}'. Use --overwrite to regenerate them or --continue to resume."
            )

        if existing_blocks and overwrite:
            for existing in existing_blocks:
                logger.warning("Overwriting existing block output %s", existing)
                existing.unlink()

        # Load states from ARM results if provided, otherwise initialize from scratch
        if arm_seeds is not None:
            arm_results_path = Path(arm_seeds).expanduser().resolve()
            if not arm_results_path.exists():
                raise FileNotFoundError(f"ARM seeds file not found: {arm_results_path}")
            logger.info("Initializing states from ARM results: %s", arm_results_path)
            states = _load_arm_states(
                arm_results_path,
                temps,
                chem_pots,
                batch_size,
                num_sites,
                logger,
            )
            # Move states to the correct device
            if isinstance(model, (LatticeIsingModel, LatticePottsModel)):
                device = (
                    next(model.parameters()).device
                    if list(model.parameters())
                    else torch.device("cpu")
                )
            else:  # AuCuAlloyModel
                if hasattr(model, "sampler") and list(model.sampler.parameters()):
                    device = next(model.sampler.parameters()).device
                else:
                    device = torch.device("cpu")
            states = {k: v.to(device) for k, v in states.items()}
            logger.info(
                "Using ARM states as seeds (effective step 1). MCMC will start from block 0."
            )
        else:
            states = _initialise_states(
                model,
                temps,
                chem_pots,
                batch_size=batch_size,
                steps_per_block=steps_per_block,
                logger=logger,
            )

    block_dir = output_dir
    block_dir.mkdir(parents=True, exist_ok=True)

    # Calculate how many blocks to run
    blocks_to_run = num_blocks - start_block_index
    if blocks_to_run <= 0:
        logger.warning("All %d blocks already completed. Nothing to do.", num_blocks)
        return

    logger.info(
        "Running %d additional blocks (from block %d to block %d)",
        blocks_to_run,
        start_block_index,
        start_block_index + blocks_to_run - 1,
    )

    for block_index in range(start_block_index, start_block_index + blocks_to_run):
        logger.info("Running block %d / %d", block_index + 1, num_blocks)
        block_configs: dict[str, np.ndarray] = {}
        block_energies: dict[str, np.ndarray] = {}
        block_x_up: dict[str, np.ndarray] = {}
        num_samples_total = 0

        block_stats: dict[str, dict[str, float]] = {}
        for chem_pot in chem_pots:
            for temp in temps:
                key = _format_key(temp, chem_pot)
                current_samples = states[(temp, chem_pot)]
                device = current_samples.device
                temps_tensor = torch.full((batch_size,), temp, dtype=torch.float32, device=device)
                fields_tensor = torch.full(
                    (batch_size,), chem_pot, dtype=torch.float32, device=device
                )

                current_samples = model.generate_samples(
                    n_samples=batch_size,
                    temps=temps_tensor,
                    fields=fields_tensor,
                    gt_steps=steps_per_block,
                    rand=True,
                    starting_samples=current_samples,
                )
                states[(temp, chem_pot)] = current_samples.clone().detach()

                tail_samples = _subset_tail(current_samples, samples_per_block)
                zero_fields_tail = torch.zeros(
                    tail_samples.shape[0], dtype=torch.float32, device=tail_samples.device
                )
                if model_type == "cuau":
                    tail_energies = model(tail_samples, temp, zero_fields_tail) * (temp * K_B)
                elif model_type == "potts":
                    # For Potts model, energy computation is similar to Ising
                    tail_energies = model(tail_samples, temp, zero_fields_tail) * temp
                else:  # ising
                    tail_energies = model(tail_samples, temp, zero_fields_tail) * temp
                
                # For Potts model, compute fraction of sites in the most frequent state
                # For Ising, this is the magnetization
                if model_type == "potts":
                    # Use the same logic as swendsen_wang_sampling: fraction of sites in most frequent state
                    q = model.q if hasattr(model, 'q') else 3
                    # Compute L from the shape: tail_samples is (num_samples, L*L)
                    num_sites = tail_samples.shape[1]
                    L = int(np.sqrt(num_sites))
                    if L * L != num_sites:
                        raise ValueError(f"Expected square lattice, but num_sites={num_sites} is not a perfect square")
                    num_samples_batch = tail_samples.shape[0]
                    # Reshape to (num_samples, L, L) for processing
                    tail_samples_2d = tail_samples.reshape(num_samples_batch, L, L)
                    # Convert to numpy for bincount (more efficient for this operation)
                    tail_samples_np = tail_samples_2d.detach().cpu().numpy()
                    # Compute fraction of sites in most frequent state for each sample
                    tail_x_up = torch.tensor([
                        np.bincount(sample.flatten(), minlength=q).max() / (L * L)
                        for sample in tail_samples_np
                    ], dtype=torch.float32, device=tail_samples.device)
                else:
                    tail_x_up = tail_samples.float().mean(dim=1)

                # For Potts model, ensure values are in correct range [0, q-1]
                if model_type == "potts":
                    # Potts samples are already integers in [0, q-1]
                    block_configs[key] = tail_samples.detach().cpu().numpy().astype(np.int8)
                else:
                    block_configs[key] = tail_samples.detach().cpu().numpy().astype(np.int8)
                block_energies[key] = tail_energies.detach().cpu().numpy().astype(np.float32)
                block_x_up[key] = tail_x_up.detach().cpu().numpy().astype(np.float32)
                num_samples_total += int(tail_samples.shape[0])

                # Compute statistics for this key
                energies_array = block_energies[key]
                x_up_array = block_x_up[key]
                block_stats[key] = {
                    "energies_mean": float(np.mean(energies_array)),
                    "energies_std": float(np.std(energies_array)),
                    "x_up_mean": float(np.mean(x_up_array)),
                    "x_up_std": float(np.std(x_up_array)),
                }

        block_metadata = {
            "block_index": block_index,
            "steps_per_block": steps_per_block,
            "num_samples_total": num_samples_total,
            "keys": list(block_configs.keys()),
            "samples_per_key": {key: int(block_configs[key].shape[0]) for key in block_configs},
            "statistics": block_stats,
        }

        # Print statistics for this block
        logger.info("Block %d statistics:", block_index)
        for key in sorted(block_stats.keys()):
            stats = block_stats[key]
            logger.info(
                "  %s: E_mean=%.4f, E_std=%.4f, x_up_mean=%.4f, x_up_std=%.4f",
                key,
                stats["energies_mean"],
                stats["energies_std"],
                stats["x_up_mean"],
                stats["x_up_std"],
            )

        block_path = _save_block(
            block_dir=block_dir,
            block_index=block_index,
            configs=block_configs,
            energies=block_energies,
            x_up=block_x_up,
            metadata=block_metadata,
        )

        summary = BlockSummary(
            block_index=block_index,
            start_step=block_index * steps_per_block,
            end_step=(block_index + 1) * steps_per_block,
            num_samples=num_samples_total,
            path=str(block_path),
        )
        block_summaries.append(summary)

        logger.debug("Block %d saved to %s", block_index, block_path)

    # Collect statistics from all blocks (existing + new)
    all_block_statistics: dict[int, dict[str, dict[str, float]]] = {}

    # Load statistics from all blocks in block_summaries
    for block_summary in block_summaries:
        block_path = Path(block_summary.path)
        if block_path.exists():
            with block_path.open("rb") as f:
                block_data = pkl.load(f)
            if "metadata" in block_data and "statistics" in block_data["metadata"]:
                all_block_statistics[block_summary.block_index] = block_data["metadata"][
                    "statistics"
                ]

    # Update metadata.json with all blocks (existing + new)
    metadata_path = block_dir / "metadata.json"
    metadata = {
        "model_type": model_type,
        "temps": temps,
        "chem_pots": chem_pots,
        "batch_size": batch_size,
        "steps_per_block": steps_per_block,
        # Total blocks including existing ones
        "num_blocks": start_block_index + blocks_to_run,
        "samples_per_block": samples_per_block,
        "init_sigma": init_sigma,
        "reference_block": block_summaries[-1].block_index if block_summaries else None,
        "blocks": [asdict(block) for block in block_summaries],
        "block_statistics": all_block_statistics,
    }
    # Add model-specific parameters
    if model_type == "ising":
        metadata["dim"] = dim
    elif model_type == "potts":
        metadata["dim"] = dim
        metadata["q"] = q
    elif model_type == "cuau":
        metadata["supercell"] = supercell
        if eci_file is not None:
            metadata["eci_file"] = str(eci_file)
    metadata_path.write_text(json.dumps(metadata, indent=2))
    logger.info("Metadata written to %s", metadata_path)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir = output_dir.resolve()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logger = logging.getLogger("mcmc_block_sampling")

    set_seed(args.seed)

    # Validate that --continue and --arm-seeds are not both specified
    if args.continue_from is not None and args.arm_seeds is not None:
        raise ValueError(
            "Cannot specify both --continue and --arm-seeds. "
            "Use --continue to resume an existing MCMC run, or --arm-seeds to start a new run from ARM results."
        )

    run_blockwise_sampler(
        args.temps,
        args.chem_pots,
        model_type=args.model_type,
        dim=args.dim,
        batch_size=args.batch_size,
        samples_per_block=args.samples_per_block,
        steps_per_block=args.steps_per_block,
        num_blocks=args.num_blocks,
        init_sigma=args.init_sigma,
        output_dir=output_dir,
        overwrite=args.overwrite,
        continue_from=args.continue_from,
        arm_seeds=args.arm_seeds,
        eci_file=args.eci_file,
        supercell=args.supercell,
        q=getattr(args, 'q', 3),
        logger=logger,
    )


if __name__ == "__main__":
    main()
