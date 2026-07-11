# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
import json
from itertools import combinations, product
from math import comb
from pathlib import Path

import numpy as np
from tqdm.rich import tqdm

from .globals import get_rng
from .logger import get_logger

CACHE_DIR = Path(".cache/grids")


def _get_cache_key(
    output_dim: int,
    r: float,
    grid_size: int | None,
    top_k: int | None,
    seed: int,
) -> str:
    """Generate a unique cache key for grid parameters."""
    params = {
        "output_dim": output_dim,
        "r": r,
        "grid_size": grid_size,
        "top_k": top_k,
        "seed": seed,
    }
    params_str = json.dumps(params, sort_keys=True)
    return hashlib.sha256(params_str.encode()).hexdigest()[:16]


def _load_cached_grid(cache_key: str) -> np.ndarray | None:
    """Load grid from cache if it exists."""
    cache_path = CACHE_DIR / f"{cache_key}.npy"
    if cache_path.exists():
        return np.load(cache_path)
    return None


def _save_grid_to_cache(cache_key: str, grid: np.ndarray) -> None:
    """Save grid to cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{cache_key}.npy"
    np.save(cache_path, grid)


def create_grid(output_dim: int, r: float, top_k: int | None = None) -> np.ndarray:
    choices = np.arange(0, 1 + r, r)
    all_combinations = list(product(choices, repeat=output_dim))

    unique_grid = []
    seen_ratios = set()

    if top_k is not None:
        for indices in combinations(range(output_dim), top_k):
            custom_row = np.zeros(output_dim)
            custom_row[list(indices)] = 1.0
            ratio = tuple(indices)
            if ratio not in seen_ratios:
                seen_ratios.add(ratio)
                unique_grid.append(custom_row)

    for combo in all_combinations:
        combo = np.array(combo)

        if np.allclose(combo, 0):
            continue

        nonzero_mask = combo != 0
        if np.any(nonzero_mask):
            nonzero_elements = combo[nonzero_mask]
            first_nonzero = nonzero_elements[0]
            normalized = combo.copy()
            normalized[nonzero_mask] = normalized[nonzero_mask] / first_nonzero
            ratio = tuple(round(x, 6) for x in normalized)
        else:
            ratio = tuple(combo)

        if ratio not in seen_ratios:
            seen_ratios.add(ratio)
            unique_grid.append(combo)

    grid = np.array(unique_grid)

    equal_weight_mask = np.all(grid == grid[:, [0]], axis=1)
    grid = grid[~equal_weight_mask]

    return grid


def create_grid_sampled(
    output_dim: int,
    r: float,
    grid_size: int,
    top_k: int | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Create a sampled grid with caching based on parameters and seed.

    Args:
        output_dim: Dimension of output space.
        r: Grid resolution.
        grid_size: Number of grid points to sample.
        top_k: Optional top-k constraint.
        seed: RNG seed for cache key. If None, caching is disabled.

    Returns:
        Grid matrix of shape (grid_size, output_dim).
    """
    # Check cache if seed is provided
    cache_key: str | None = None
    if seed is not None:
        cache_key = _get_cache_key(output_dim, r, grid_size, top_k, seed)
        cached = _load_cached_grid(cache_key)
        if cached is not None:
            return cached

    choices = np.arange(0, 1 + r, r)
    grid = []
    seen_ratios = set()
    rng = get_rng()

    if top_k is not None:
        n_custom = comb(output_dim, top_k)
        if n_custom > grid_size / 2:
            raise ValueError(f"Comb({output_dim}, {top_k}) = {n_custom} > grid_size/2 = {grid_size / 2}")
        for indices in combinations(range(output_dim), top_k):
            custom_row = np.zeros(output_dim)
            custom_row[list(indices)] = 1.0
            ratio = tuple(indices)
            if ratio not in seen_ratios:
                seen_ratios.add(ratio)
                grid.append(custom_row)

    logger = get_logger(__name__)
    tab = 17 * " "
    progress = tqdm(total=grid_size, initial=len(grid), desc=tab + "Sampling grid")
    fail_count = 0
    max_failures = 100
    while len(grid) < grid_size:
        combo = rng.choice(choices, size=output_dim, replace=True)

        if np.allclose(combo, 0):
            continue

        if np.all(combo == combo[0]):
            continue

        nonzero_mask = combo != 0
        nonzero_elements = combo[nonzero_mask]
        first_nonzero = nonzero_elements[0]
        normalized = combo.copy()
        normalized[nonzero_mask] = normalized[nonzero_mask] / first_nonzero
        ratio = tuple(round(x, 6) for x in normalized)

        if ratio not in seen_ratios:
            seen_ratios.add(ratio)
            grid.append(combo)
            progress.update(1)
            fail_count = 0
        else:
            fail_count += 1

        if fail_count >= max_failures:
            logger.info(
                "Early stop in grid sampling after %s consecutive duplicates; returning %s/%s samples.",
                max_failures,
                len(grid),
                grid_size,
            )
            break

    progress.close()
    result = np.array(grid)

    # Save to cache if seed is provided
    if cache_key is not None:
        _save_grid_to_cache(cache_key, result)

    return result
