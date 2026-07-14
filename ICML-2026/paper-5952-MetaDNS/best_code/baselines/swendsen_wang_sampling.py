#!/usr/bin/env python
"""Blockwise Swendsen-Wang sampling utility for the SnowyFlow Ising and Potts benchmarks.

* Generates samples for each (temperature, chemical potential) pair using the Swendsen-Wang algorithm.
* Supports both Ising (q=2) and Potts (q>2) models.
* Runs the sampler in sequential blocks of fixed length (``steps-per-block``).
* Persists the requested statistics after each block so downstream notebooks
  can quantify convergence using KL/JS divergence against the final block.

The Swendsen-Wang algorithm is a cluster Monte Carlo method that:
1. Identifies bonds between aligned spins (same state for Potts)
2. Activates bonds with probability p = 1 - exp(-2*beta*J) for Ising, p = 1 - exp(-beta*J) for Potts
3. Identifies clusters of connected spins
4. Flips entire clusters (Ising) or assigns new random state (Potts)

This algorithm is particularly effective near critical temperatures where it
reduces critical slowing down compared to single-site update methods.

Parameter Mapping from original ising2d_swendsen_wang/potts2d_swendsen_wang functions:
- L → --dim
- J → --J
- beta → --temps (converted to β=1/T)
- B → --batch-size
- num_collect → --num-blocks (number of collection events/blocks)
- burn_in → --burn-in
- collect_every → --steps-per-block (steps between collections)
- q → --q (number of Potts states, only for Potts model)
- --samples-per-block is a new parameter (how many samples to save from each block)
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
from tqdm import tqdm

from baselines.energy.ising import LatticeIsingModel


def compute_potts_energy(S: np.ndarray, L: int, J: float, q: int = 3) -> np.ndarray:
    """Compute Potts model energy for configurations.
    
    Args:
        S: Configurations of shape (B, L, L) or (B, L*L) with values in {0, 1, ..., q-1}
        L: Linear lattice dimension
        J: Coupling constant
        q: Number of Potts states (unused in computation but kept for API consistency)
    
    Returns:
        Energy values of shape (B,)
    """
    # Reshape if needed
    if S.ndim == 2:
        B, _ = S.shape
        S = S.reshape(B, L, L)
    else:
        B, L_dim, L_dim = S.shape
        if L_dim != L:
            raise ValueError(f"Lattice dimension mismatch: expected {L}, got {L_dim}")
    
    energies = np.zeros(B)
    for b in range(B):
        S_2d = S[b]
        
        # Periodic boundary conditions
        s_left = np.roll(S_2d, shift=1, axis=1)
        s_top = np.roll(S_2d, shift=1, axis=0)
        s_right = np.roll(S_2d, shift=-1, axis=1)
        s_down = np.roll(S_2d, shift=-1, axis=0)
        
        # Count matching neighbors
        equal_left = (S_2d == s_left).astype(int)
        equal_right = (S_2d == s_right).astype(int)
        equal_top = (S_2d == s_top).astype(int)
        equal_down = (S_2d == s_down).astype(int)
        
        interaction_per_node = equal_left + equal_right + equal_top + equal_down
        energies[b] = -J * interaction_per_node.sum() / 2
    
    return energies


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
        description="Run blockwise Swendsen-Wang sampling for Ising/Potts models and persist intermediate results."
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["ising", "potts"],
        default="ising",
        help="Type of model to use: 'ising' for Ising model (q=2), 'potts' for Potts model (q>2).",
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
        help="Linear lattice dimension; the sampler runs on a dim x dim lattice.",
    )
    parser.add_argument(
        "--J",
        type=float,
        default=1.0,
        help="Coupling constant J for the model (default: 1.0).",
    )
    parser.add_argument(
        "--q",
        type=int,
        default=3,
        help="Number of Potts states (only used for Potts model, default: 3).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Total number of parallel chains to evolve per block (B parameter).",
    )
    parser.add_argument(
        "--samples-per-block",
        type=int,
        default=32,
        help="Number of samples to persist from each block (taken from the tail). Note: In original function, this would be B (batch_size) samples per collection.",
    )
    parser.add_argument(
        "--steps-per-block",
        type=int,
        default=128,
        help="Number of Swendsen-Wang steps to run inside each block (maps to collect_every parameter).",
    )
    parser.add_argument(
        "--num-blocks",
        type=int,
        default=10,
        help="How many sequential blocks to run per (temp, chem_pot) pair (maps to num_collect parameter).",
    )
    parser.add_argument(
        "--burn-in",
        type=int,
        default=1024,
        help="Number of initial Swendsen-Wang steps to discard before collecting samples.",
    )
    parser.add_argument(
        "--init-sigma",
        type=float,
        default=1.0,
        help="Standard deviation used to seed the Ising sampler (for energy computation).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/sw_blocks",
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
    args = parser.parse_args()
    return args


def set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _format_key(temp: float, chem_pot: float) -> str:
    return f"{temp:.1f}K_mu{chem_pot:.2f}"


def _subset_tail(tensor: np.ndarray, num_items: int) -> np.ndarray:
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


def _convert_to_ising_format(spins: np.ndarray) -> np.ndarray:
    """Convert spins from {-1,1} to {0,1} format for LatticeIsingModel.

    Args:
        spins: Array of shape (B, L, L) or (B, L*L) with values in {-1, 1}

    Returns:
        Array of same shape with values in {0, 1}
    """
    return ((spins + 1) // 2).astype(np.int8)


def _convert_from_ising_format(spins: np.ndarray) -> np.ndarray:
    """Convert spins from {0,1} to {-1,1} format for Swendsen-Wang.

    Args:
        spins: Array of shape (B, L, L) or (B, L*L) with values in {0, 1}

    Returns:
        Array of same shape with values in {-1, 1}
    """
    return (2 * spins - 1).astype(np.int8)


def swendsen_wang_step_ising(S: np.ndarray, L: int, J: float, beta: float) -> np.ndarray:
    """Perform one Swendsen-Wang step on a single Ising configuration.

    Args:
        S: Configuration of shape (L, L) with values in {-1, 1}
        L: Linear lattice dimension
        J: Coupling constant
        beta: Inverse temperature (1/T)

    Returns:
        Updated configuration of shape (L, L) with values in {-1, 1}
    """
    # Pre-compute bond probability for Ising
    p = 1 - np.exp(-2 * beta * J)

    # Step 1: Identify bonds between aligned spins
    h_bonds = np.zeros((L, L), dtype=bool)  # horizontal bonds
    v_bonds = np.zeros((L, L), dtype=bool)  # vertical bonds

    # Check horizontal bonds (same spin alignment)
    h_bonds[:, :-1] = S[:, :-1] == S[:, 1:]
    h_bonds[:, -1] = S[:, -1] == S[:, 0]  # periodic BC

    # Check vertical bonds (same spin alignment)
    v_bonds[:-1, :] = S[:-1, :] == S[1:, :]
    v_bonds[-1, :] = S[-1, :] == S[0, :]  # periodic BC

    # Step 2: Activate bonds with probability p
    h_bonds = h_bonds & (np.random.random((L, L)) < p)
    v_bonds = v_bonds & (np.random.random((L, L)) < p)

    # Step 3: Identify clusters using Union-Find
    parent = np.arange(L * L).reshape(L, L)
    rank = np.zeros((L, L), dtype=int)

    def find(x, y):
        if parent[x, y] != x * L + y:
            px, py = parent[x, y] // L, parent[x, y] % L
            parent[x, y] = find(px, py)
        return parent[x, y]

    def union(x1, y1, x2, y2):
        root1 = find(x1, y1)
        root2 = find(x2, y2)
        if root1 != root2:
            r1, c1 = root1 // L, root1 % L
            r2, c2 = root2 // L, root2 % L
            if rank[r1, c1] < rank[r2, c2]:
                parent[r1, c1] = root2
            else:
                parent[r2, c2] = root1
                if rank[r1, c1] == rank[r2, c2]:
                    rank[r1, c1] += 1

    # Process horizontal bonds
    for i in range(L):
        for j in range(L):
            if h_bonds[i, j]:
                union(i, j, i, (j + 1) % L)

    # Process vertical bonds
    for i in range(L):
        for j in range(L):
            if v_bonds[i, j]:
                union(i, j, (i + 1) % L, j)

    # Step 4: Identify clusters
    clusters = {}
    for i in range(L):
        for j in range(L):
            root = find(i, j)
            if root not in clusters:
                clusters[root] = []
            clusters[root].append((i, j))

    # Step 5: Flip clusters with probability 0.5
    S_new = S.copy()
    for cluster in clusters.values():
        if np.random.random() < 0.5:
            for i, j in cluster:
                S_new[i, j] *= -1

    return S_new


def swendsen_wang_step_potts(S: np.ndarray, L: int, J: float, beta: float, q: int) -> np.ndarray:
    """Perform one Swendsen-Wang step on a single Potts configuration.

    Args:
        S: Configuration of shape (L, L) with values in {0, 1, ..., q-1}
        L: Linear lattice dimension
        J: Coupling constant
        beta: Inverse temperature (1/T)
        q: Number of Potts states

    Returns:
        Updated configuration of shape (L, L) with values in {0, 1, ..., q-1}
    """
    # Pre-compute bond probability for Potts
    p = 1 - np.exp(-beta * J)

    # Step 1: Identify bonds between same-state neighbors
    h_bonds = np.zeros((L, L), dtype=bool)  # horizontal bonds
    v_bonds = np.zeros((L, L), dtype=bool)  # vertical bonds

    # Check horizontal bonds
    h_bonds[:, :-1] = S[:, :-1] == S[:, 1:]
    h_bonds[:, -1] = S[:, -1] == S[:, 0]  # periodic BC

    # Check vertical bonds
    v_bonds[:-1, :] = S[:-1, :] == S[1:, :]
    v_bonds[-1, :] = S[-1, :] == S[0, :]  # periodic BC

    # Step 2: Activate bonds with probability p
    h_bonds = h_bonds & (np.random.random((L, L)) < p)
    v_bonds = v_bonds & (np.random.random((L, L)) < p)

    # Step 3: Identify clusters using Union-Find
    parent = np.arange(L * L).reshape(L, L)
    rank = np.zeros((L, L), dtype=int)

    def find(x, y):
        if parent[x, y] != x * L + y:
            px, py = parent[x, y] // L, parent[x, y] % L
            parent[x, y] = find(px, py)
        return parent[x, y]

    def union(x1, y1, x2, y2):
        root1 = find(x1, y1)
        root2 = find(x2, y2)
        if root1 != root2:
            r1, c1 = root1 // L, root1 % L
            r2, c2 = root2 // L, root2 % L
            if rank[r1, c1] < rank[r2, c2]:
                parent[r1, c1] = root2
            else:
                parent[r2, c2] = root1
                if rank[r1, c1] == rank[r2, c2]:
                    rank[r1, c1] += 1

    # Process horizontal bonds
    for i in range(L):
        for j in range(L):
            if h_bonds[i, j]:
                union(i, j, i, (j + 1) % L)

    # Process vertical bonds
    for i in range(L):
        for j in range(L):
            if v_bonds[i, j]:
                union(i, j, (i + 1) % L, j)

    # Step 4: Identify clusters
    clusters = {}
    for i in range(L):
        for j in range(L):
            root = find(i, j)
            if root not in clusters:
                clusters[root] = []
            clusters[root].append((i, j))

    # Step 5: Assign new random state to each cluster
    S_new = S.copy()
    for cluster in clusters.values():
        # Randomly choose new state for the cluster
        new_state = np.random.randint(0, q)
        # Update all spins in the cluster
        for i, j in cluster:
            S_new[i, j] = new_state

    return S_new


# Note: The standalone swendsen_wang_sampling function has been replaced by
# the blockwise version in run_blockwise_sampler. The step functions
# swendsen_wang_step_ising and swendsen_wang_step_potts are used directly.


def _load_last_block_states(
    last_block_path: Path,
    temps: list[float],
    chem_pots: list[float],
    batch_size: int,
    L: int,
    model_type: str,
    logger: logging.Logger,
) -> dict[tuple[float, float], np.ndarray]:
    """Load states from the last saved block to continue sampling."""
    logger.info("Loading states from last block: %s", last_block_path)
    with last_block_path.open("rb") as f:
        block_data = pkl.load(f)

    configs = block_data["configs"]
    states: dict[tuple[float, float], np.ndarray] = {}

    for chem_pot in chem_pots:
        for temp in temps:
            key = _format_key(temp, chem_pot)
            if key not in configs:
                raise KeyError(
                    f"Key {key} not found in last block. Available keys: {list(configs.keys())}"
                )

            # Load the configs
            config_array = configs[key]  # shape: (num_samples, L*L)
            num_samples = config_array.shape[0]

            # Reshape to (num_samples, L, L)
            config_array = config_array.reshape(num_samples, L, L)

            # Convert format based on model type
            if model_type == "ising":
                # Convert from {0,1} to {-1,1} format
                config_array = _convert_from_ising_format(config_array)
            # For Potts, configs are already in {0,1,...,q-1} format, no conversion needed

            # Handle batch size
            if num_samples < batch_size:
                logger.warning(
                    "Last block has %d samples for %s, but batch_size is %d. "
                    "Repeating samples to fill batch.",
                    num_samples,
                    key,
                    batch_size,
                )
                repeat_factor = (batch_size // num_samples) + 1
                config_array = np.tile(config_array, (repeat_factor, 1, 1))[:batch_size]
            elif num_samples > batch_size:
                config_array = config_array[-batch_size:]

            states[(temp, chem_pot)] = config_array

    logger.info("Loaded states for %d (temp, chem_pot) pairs", len(states))
    return states


def _load_metadata(
    metadata_path: Path, logger: logging.Logger
) -> tuple[dict, list[BlockSummary], int]:
    """Load metadata.json and return configuration, existing blocks, and next block index."""
    logger.info("Loading metadata from %s", metadata_path)
    with metadata_path.open("r") as f:
        metadata = json.load(f)

    block_summaries = [BlockSummary(**block_dict) for block_dict in metadata.get("blocks", [])]

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
    dim: int,
    J: float,
    q: int,
    batch_size: int,
    samples_per_block: int,
    steps_per_block: int,
    num_blocks: int,
    burn_in: int,
    init_sigma: float,
    output_dir: Path,
    overwrite: bool,
    continue_from: str | None,
    logger: logging.Logger,
) -> None:
    temps = list(temps)
    chem_pots = list(chem_pots)
    L = dim

    # Create model for energy computation (only for Ising)
    model = None
    if model_type == "ising":
        model = LatticeIsingModel(dim=dim, init_sigma=init_sigma, n_samples=batch_size)

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
        batch_size = existing_metadata["batch_size"]
        steps_per_block = existing_metadata["steps_per_block"]
        samples_per_block = existing_metadata["samples_per_block"]
        init_sigma = existing_metadata["init_sigma"]
        output_dir = metadata_path.parent
        dim = existing_metadata["dim"]
        L = dim
        J = existing_metadata.get("J", 1.0)
        burn_in = existing_metadata.get("burn_in", 1024)
        model_type = existing_metadata.get("model_type", "ising")
        q = existing_metadata.get("q", 3)

        logger.info(
            "Continuing from existing run: model_type=%s, temps=%s, chem_pots=%s, "
            "batch_size=%d, steps_per_block=%d, starting at block %d",
            model_type,
            temps,
            chem_pots,
            batch_size,
            steps_per_block,
            start_block_index,
        )

        # Recreate model (only for Ising)
        if model_type == "ising":
            model = LatticeIsingModel(dim=dim, init_sigma=init_sigma, n_samples=batch_size)

        # Load states from last block
        if existing_blocks:
            last_block_path = Path(existing_blocks[-1].path)
            states = _load_last_block_states(
                last_block_path,
                temps,
                chem_pots,
                batch_size,
                L,
                model_type,
                logger,
            )
        else:
            logger.warning("No existing blocks found, initializing from scratch")
            states = {}
            for chem_pot in chem_pots:
                for temp in temps:
                    if model_type == "ising":
                        states[(temp, chem_pot)] = np.random.choice([-1, 1], size=(batch_size, L, L))
                    else:  # potts
                        states[(temp, chem_pot)] = np.random.randint(0, q, size=(batch_size, L, L))
    else:
        logger.info(
            "Starting blockwise Swendsen-Wang sampling: model_type=%s, temps=%s, chem_pots=%s, "
            "batch_size=%d, steps_per_block=%d, num_blocks=%d, L=%d, J=%.2f, q=%d",
            model_type,
            temps,
            chem_pots,
            batch_size,
            steps_per_block,
            num_blocks,
            L,
            J,
            q,
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

        # Initialize states
        states = {}
        for chem_pot in chem_pots:
            for temp in temps:
                if model_type == "ising":
                    states[(temp, chem_pot)] = np.random.choice([-1, 1], size=(batch_size, L, L))
                else:  # potts
                    states[(temp, chem_pot)] = np.random.randint(0, q, size=(batch_size, L, L))

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
                current_state = states[(temp, chem_pot)]  # shape: (B, L, L)

                # Convert beta from temperature
                beta = 1.0 / temp

                # Run Swendsen-Wang sampling for this block
                S = current_state.copy()  # Work with copy

                # For the first block, apply burn_in before collecting
                if block_index == 0 and burn_in > 0:
                    logger.debug("Applying burn_in of %d steps for %s", burn_in, key)
                    for _ in range(burn_in):
                        for b in range(batch_size):
                            if model_type == "ising":
                                S[b] = swendsen_wang_step_ising(S[b], L, J, beta)
                            else:  # potts
                                S[b] = swendsen_wang_step_potts(S[b], L, J, beta, q)

                # Collect samples during this block
                # We'll collect samples periodically throughout the block
                block_samples = []
                collect_every = max(1, steps_per_block // max(1, samples_per_block // batch_size))

                for step in range(steps_per_block):
                    # Perform Swendsen-Wang step on all configurations
                    for b in range(batch_size):
                        if model_type == "ising":
                            S[b] = swendsen_wang_step_ising(S[b], L, J, beta)
                        else:  # potts
                            S[b] = swendsen_wang_step_potts(S[b], L, J, beta, q)

                    # Collect sample periodically (aim for samples_per_block total)
                    if step % collect_every == 0:
                        block_samples.append(S.reshape(batch_size, L * L).copy())

                # Always collect the final state if we haven't collected it yet
                final_state = S.reshape(batch_size, L * L)
                if len(block_samples) == 0 or not np.array_equal(block_samples[-1], final_state):
                    block_samples.append(final_state.copy())

                # Update state for next block
                states[(temp, chem_pot)] = S.copy()

                # Take tail samples
                if len(block_samples) > 0:
                    all_samples = np.concatenate(block_samples, axis=0)  # (num_collected*B, L*L)
                    tail_samples = _subset_tail(all_samples, samples_per_block)
                else:
                    # Fallback: use current state
                    tail_samples = S.reshape(batch_size, L * L)[-samples_per_block:]

                # Convert format and compute energies based on model type
                if model_type == "ising":
                    # Convert to {0,1} format for energy computation and saving
                    tail_samples_01 = _convert_to_ising_format(tail_samples.reshape(-1, L, L)).reshape(
                        -1, L * L
                    )

                    # Compute energies using the Ising model
                    tail_samples_torch = torch.from_numpy(tail_samples_01).float()
                    temps_tensor = torch.full((tail_samples_torch.shape[0],), temp, dtype=torch.float32)
                    fields_tensor = torch.full(
                        (tail_samples_torch.shape[0],), chem_pot, dtype=torch.float32
                    )

                    # Compute energies (model expects {0,1} format)
                    tail_energies = model(tail_samples_torch, temps_tensor, fields_tensor) * temp

                    # Compute x_up (magnetization/concentration)
                    tail_x_up = tail_samples_01.mean(axis=1)  # Already in {0,1} format
                    
                    # Save in {0,1} format
                    tail_samples_save = tail_samples_01
                else:  # potts
                    # Potts samples are already in {0,1,...,q-1} format
                    tail_samples_potts = tail_samples.reshape(-1, L, L)
                    
                    # Compute energies using Potts energy function
                    # compute_potts_energy returns raw energy, so we don't multiply by temp
                    # (matching the convention where Ising model divides by temp then multiplies by temp)
                    tail_energies = compute_potts_energy(tail_samples_potts, L, J, q)
                    
                    # Compute x_up (average state value, or most frequent state fraction)
                    # For Potts, we compute the fraction of sites in the most frequent state
                    tail_x_up = np.array([
                        np.bincount(sample.flatten(), minlength=q).max() / (L * L)
                        for sample in tail_samples_potts
                    ])
                    
                    # Save in {0,1,...,q-1} format
                    tail_samples_save = tail_samples.astype(np.int8)

                block_configs[key] = tail_samples_save
                if model_type == "ising":
                    block_energies[key] = tail_energies.detach().cpu().numpy().astype(np.float32)
                else:  # potts
                    block_energies[key] = tail_energies.astype(np.float32)
                block_x_up[key] = tail_x_up.astype(np.float32)
                num_samples_total += int(tail_samples_save.shape[0])

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
        "num_blocks": start_block_index + blocks_to_run,
        "samples_per_block": samples_per_block,
        "init_sigma": init_sigma,
        "dim": dim,
        "J": J,
        "burn_in": burn_in,
        "q": q if model_type == "potts" else None,
        "reference_block": block_summaries[-1].block_index if block_summaries else None,
        "blocks": [asdict(block) for block in block_summaries],
        "block_statistics": all_block_statistics,
    }
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
    logger = logging.getLogger("swendsen_wang_sampling")

    set_seed(args.seed)

    run_blockwise_sampler(
        args.temps,
        args.chem_pots,
        model_type=args.model_type,
        dim=args.dim,
        J=args.J,
        q=args.q,
        batch_size=args.batch_size,
        samples_per_block=args.samples_per_block,
        steps_per_block=args.steps_per_block,
        num_blocks=args.num_blocks,
        burn_in=args.burn_in,
        init_sigma=args.init_sigma,
        output_dir=output_dir,
        overwrite=args.overwrite,
        continue_from=args.continue_from,
        logger=logger,
    )


if __name__ == "__main__":
    main()
