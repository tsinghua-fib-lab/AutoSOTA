"""
FILE: synthetic.py
INPUT: Numeric experiment configuration and random seeds.
OUTPUT: Synthetic random-subspace samples, true labels, and normalized design matrices.
POS: Data generation helper for reproducible ICML experiments.
NOTE: Update this header and the folder's _MANIFEST.md explicitly if logic changes.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]


def _as_int_vector(value: int | Sequence[int], total: int, name: str) -> IntArray:
    if isinstance(value, int):
        if value <= 0:
            raise ValueError(f"{name} must be positive.")
        if total % value != 0:
            raise ValueError(f"{name}={value} must divide total={total}.")
        return np.full(total // value, value, dtype=int)

    vector = np.asarray(value, dtype=int)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional sequence.")
    if np.any(vector < 0):
        raise ValueError(f"{name} entries must be non-negative.")
    if int(vector.sum()) != total:
        raise ValueError(f"{name} must sum to {total}; got {int(vector.sum())}.")
    return vector


def sample_cluster_sizes(total_variables: int, n_clusters: int, seed: int) -> IntArray:
    """Draw cluster sizes with the same multinomial model used by the paper simulations."""
    if total_variables <= 0 or n_clusters <= 0:
        raise ValueError("total_variables and n_clusters must be positive.")
    if n_clusters > total_variables:
        raise ValueError("n_clusters cannot exceed total_variables.")
    rng = np.random.RandomState(seed)
    return rng.multinomial(total_variables, np.ones(n_clusters) / n_clusters, size=1)[0].astype(int)


def make_cluster_labels(cluster_sizes: Sequence[int]) -> IntArray:
    """Return one integer label per variable for contiguous synthetic clusters."""
    sizes = np.asarray(cluster_sizes, dtype=int)
    if sizes.ndim != 1 or sizes.size == 0 or np.any(sizes < 0) or sizes.sum() <= 0:
        raise ValueError("cluster_sizes must be a non-empty non-negative vector with positive total size.")
    return np.repeat(np.arange(sizes.size, dtype=int), sizes)


def sample_random_subspace(
    n_samples: int = 100,
    n_variables: int = 500,
    cluster_sizes: int | Sequence[int] = 25,
    n_group_factors: int | Sequence[int] | None = 1,
    n_group_factors_range: tuple[int, int] | None = None,
    rho_range: tuple[float, float] = (0.0, 0.0),
    noise_range: tuple[float, float] = (1.0, 3.0),
    seed: int = 2020,
    positive_loadings_only: bool = False,
    group_noise: bool = False,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    """Generate the random-subspace factor model used in the synthetic experiments."""
    if n_samples <= 0 or n_variables <= 0:
        raise ValueError("n_samples and n_variables must be positive.")
    sizes = _as_int_vector(cluster_sizes, n_variables, "cluster_sizes")
    n_clusters = sizes.size
    rng = np.random.RandomState(seed)

    if n_group_factors is not None:
        if isinstance(n_group_factors, int):
            if n_group_factors <= 0:
                raise ValueError("n_group_factors must be positive.")
            factor_counts = np.full(n_clusters, n_group_factors, dtype=int)
        else:
            factor_counts = np.asarray(n_group_factors, dtype=int)
            if factor_counts.shape != (n_clusters,) or np.any(factor_counts <= 0):
                raise ValueError("n_group_factors must have one positive entry per cluster.")
    else:
        if n_group_factors_range is None:
            n_group_factors_range = (1, n_samples)
        min_factors, max_factors = n_group_factors_range
        if min_factors <= 0 or min_factors >= max_factors:
            raise ValueError("n_group_factors_range must be an increasing positive tuple.")
        factor_counts = np.zeros(n_clusters, dtype=int)

    factors = rng.randn(n_samples, min(n_variables, n_samples))
    factors /= np.linalg.norm(factors, ord=2, axis=0)
    global_factor = rng.randn(n_samples)
    global_factor /= np.linalg.norm(global_factor, ord=2)
    global_rho = rng.uniform(*rho_range, size=n_variables)
    cluster_bounds = np.concatenate([[0], sizes.cumsum()]).astype(int)
    loadings_matrix = np.zeros((n_variables, factors.shape[1]))

    for cluster_idx in range(n_clusters):
        if sizes[cluster_idx] == 0:
            continue
        if n_group_factors is None:
            upper_exclusive = min(int(sizes[cluster_idx]), max_factors)
            if upper_exclusive <= min_factors:
                raise ValueError("n_group_factors_range is incompatible with at least one non-empty cluster.")
            n_factors = int(rng.randint(max(1, min_factors), upper_exclusive))
        else:
            n_factors = int(factor_counts[cluster_idx])
        factor_indices = rng.choice(np.arange(factors.shape[1], dtype=int), size=n_factors, replace=False)
        for variable_idx in range(cluster_bounds[cluster_idx], cluster_bounds[cluster_idx + 1]):
            loadings = rng.randn(factor_indices.shape[0])
            if positive_loadings_only:
                loadings = np.abs(loadings)
            loadings /= np.linalg.norm(loadings, ord=2)
            loadings_matrix[variable_idx, factor_indices] = np.sqrt(1.0 - global_rho[variable_idx]) * loadings

    if group_noise:
        group_noise_level = rng.uniform(*noise_range, size=n_clusters)
        noise_level = np.ones(n_variables)
        for cluster_idx in range(n_clusters):
            noise_level[cluster_bounds[cluster_idx] : cluster_bounds[cluster_idx + 1]] = group_noise_level[cluster_idx]
    else:
        noise_level = rng.uniform(*noise_range, size=n_variables)

    noise = rng.randn(n_samples, n_variables)
    noise = noise / np.linalg.norm(noise, ord=2, axis=0) * np.sqrt(noise_level)
    samples = np.expand_dims(global_factor, axis=1) * np.sqrt(global_rho) + factors @ loadings_matrix.T + noise
    return samples, global_rho, noise_level, loadings_matrix, factors


def remove_first_k_pcs(samples: FloatArray, n_components: int) -> FloatArray:
    """Remove leading principal components from a centered design matrix."""
    if n_components == 0:
        return samples.copy()
    if n_components < 0 or n_components >= min(samples.shape):
        raise ValueError(f"n_components must be in [0, {min(samples.shape) - 1}].")
    centered = samples - samples.mean(axis=0)
    left, singular_values, right = np.linalg.svd(centered, full_matrices=False)
    singular_values[:n_components] = 0.0
    return (left * singular_values) @ right + samples.mean(axis=0)


def normalize_design(samples: FloatArray, n_removed_pcs: int = 0) -> FloatArray:
    """Apply the paper preprocessing: optional PC removal, centering, and column normalization."""
    processed = remove_first_k_pcs(samples, n_removed_pcs)
    processed = processed - processed.mean(axis=0)
    norms = np.linalg.norm(processed, ord=2, axis=0)
    if np.any(norms == 0):
        raise ValueError("Cannot normalize a design matrix with zero-norm columns.")
    return processed / norms
