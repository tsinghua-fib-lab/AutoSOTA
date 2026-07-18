"""
Ensemble-based reference selection methods for cross-lingual alignment.

This module provides different ensemble selection strategies including:
- Standard voting-based ensemble selection
- Bernoulli trial-based ensemble selection with posterior distributions
- Voting matrices are maintained in-memory as scipy.sparse matrices (no save/load)
"""

import numpy as np
import torch
import argparse
import os
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from loguru import logger
from scipy.stats import beta as beta_dist
from scipy.sparse import lil_matrix, csr_matrix

from graph_utils.distance_encoder import compute_distance_encoding
from utils.retrieval_util import find_mutual_pairs, deduplicate_pairs, get_topk
from utils.graph_util import get_dists
from utils.memory_util import (
    estimate_matrix_memory_gb,
    get_available_memory_gb,
)
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.mixture import GaussianMixture


# =============================================================================
# ADAPTIVE OVERLAP INFERENCE METHODS
# =============================================================================
# These methods infer the number of true overlapping pairs from the posterior
# probability distribution, instead of using a fixed threshold.

def estimate_overlap_otsu(posterior_means: np.ndarray) -> tuple:
    """
    Use Otsu's method to find optimal threshold that maximizes inter-class variance.

    This is a classic thresholding method from image processing that finds the
    threshold that best separates a bimodal distribution into two classes.

    Args:
        posterior_means: Array of posterior mean probabilities for each pair

    Returns:
        threshold: Optimal threshold value
        n_selected: Number of pairs above threshold
        method_info: Dict with diagnostic information
    """
    if len(posterior_means) == 0:
        return 0.5, 0, {'method': 'otsu', 'status': 'empty_input'}

    # Discretize to histogram (100 bins between 0 and 1)
    n_bins = 100
    hist, bin_edges = np.histogram(posterior_means, bins=n_bins, range=(0, 1))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Normalize histogram
    hist = hist.astype(float)
    total = hist.sum()
    if total == 0:
        return 0.5, 0, {'method': 'otsu', 'status': 'zero_histogram'}
    hist = hist / total

    # Compute cumulative sums and means
    cumsum = np.cumsum(hist)
    cumsum_mean = np.cumsum(hist * bin_centers)

    global_mean = cumsum_mean[-1]

    # Compute inter-class variance for each threshold
    inter_class_var = np.zeros(n_bins)
    for t in range(n_bins):
        w0 = cumsum[t]  # Weight of class 0
        w1 = 1 - w0      # Weight of class 1

        if w0 < 1e-10 or w1 < 1e-10:
            continue

        mu0 = cumsum_mean[t] / w0  # Mean of class 0
        mu1 = (global_mean - cumsum_mean[t]) / w1  # Mean of class 1

        inter_class_var[t] = w0 * w1 * (mu0 - mu1) ** 2

    # Find threshold with maximum inter-class variance
    optimal_idx = np.argmax(inter_class_var)
    threshold = bin_centers[optimal_idx]

    n_selected = np.sum(posterior_means >= threshold)

    return threshold, n_selected, {
        'method': 'otsu',
        'inter_class_variance': inter_class_var[optimal_idx],
        'status': 'success'
    }


def estimate_overlap_gmm(posterior_means: np.ndarray, n_components: int = 2) -> tuple:
    """
    Use Gaussian Mixture Model to separate true pairs from false pairs.

    Fits a 2-component GMM to the posterior means. The component with higher
    mean represents true pairs. Pairs are assigned to the "true" component
    based on posterior probability.

    Args:
        posterior_means: Array of posterior mean probabilities for each pair
        n_components: Number of mixture components (default: 2 for true/false)

    Returns:
        threshold: Estimated threshold (intersection of two Gaussians)
        n_selected: Number of pairs classified as "true"
        method_info: Dict with diagnostic information
    """
    if len(posterior_means) < n_components * 2:
        return 0.5, 0, {'method': 'gmm', 'status': 'insufficient_data'}

    X = posterior_means.reshape(-1, 1)

    try:
        gmm = GaussianMixture(n_components=n_components, random_state=42, max_iter=200)
        gmm.fit(X)

        # Get component parameters
        means = gmm.means_.flatten()
        stds = np.sqrt(gmm.covariances_.flatten())
        weights = gmm.weights_

        # Identify the "true" component (higher mean)
        true_component = np.argmax(means)
        false_component = 1 - true_component

        # Predict component membership
        labels = gmm.predict(X)
        probs = gmm.predict_proba(X)

        # Count pairs assigned to true component
        n_selected = np.sum(labels == true_component)

        # Estimate threshold as intersection point of two Gaussians
        # Approximate: use the point where posterior prob of true component > 0.5
        if means[true_component] > means[false_component]:
            # Find threshold where P(true|x) = 0.5
            # This is approximately the midpoint weighted by variances
            mu_t, mu_f = means[true_component], means[false_component]
            std_t, std_f = stds[true_component], stds[false_component]

            # Quadratic formula for intersection
            a = 1/(2*std_f**2) - 1/(2*std_t**2)
            b = mu_t/std_t**2 - mu_f/std_f**2
            c = mu_f**2/(2*std_f**2) - mu_t**2/(2*std_t**2) + np.log(std_t/std_f)

            if abs(a) > 1e-10:
                discriminant = b**2 - 4*a*c
                if discriminant >= 0:
                    x1 = (-b + np.sqrt(discriminant)) / (2*a)
                    x2 = (-b - np.sqrt(discriminant)) / (2*a)
                    # Choose intersection between the two means
                    for x in [x1, x2]:
                        if mu_f < x < mu_t:
                            threshold = x
                            break
                    else:
                        threshold = (mu_t + mu_f) / 2
                else:
                    threshold = (mu_t + mu_f) / 2
            else:
                threshold = (mu_t + mu_f) / 2
        else:
            threshold = 0.5

        return threshold, n_selected, {
            'method': 'gmm',
            'means': means.tolist(),
            'stds': stds.tolist(),
            'weights': weights.tolist(),
            'true_component': int(true_component),
            'bic': gmm.bic(X),
            'status': 'success'
        }

    except Exception as e:
        logger.warning(f"GMM fitting failed: {e}")
        return 0.5, 0, {'method': 'gmm', 'status': f'failed: {e}'}


def estimate_overlap_elbow(posterior_means: np.ndarray, sensitivity: float = 1.0) -> tuple:
    """
    Find the "elbow" in sorted posterior means to determine cutoff.

    Sorts posterior means in descending order and finds the point where
    the rate of decrease changes most dramatically (the elbow/knee point).

    Args:
        posterior_means: Array of posterior mean probabilities
        sensitivity: Higher values = more sensitive to changes (default: 1.0)

    Returns:
        threshold: Posterior mean value at the elbow
        n_selected: Number of pairs above the elbow
        method_info: Dict with diagnostic information
    """
    if len(posterior_means) < 3:
        return 0.5, 0, {'method': 'elbow', 'status': 'insufficient_data'}

    # Sort in descending order
    sorted_means = np.sort(posterior_means)[::-1]
    n = len(sorted_means)

    # Compute curvature using finite differences
    # First derivative (slope)
    dx = 1.0 / n  # Normalized x-axis
    dy = np.diff(sorted_means)
    slope = dy / dx

    # Second derivative (curvature approximation)
    d2y = np.diff(slope)
    curvature = d2y / dx

    # Find the point of maximum curvature (most negative second derivative)
    # This is where the curve bends most sharply
    if len(curvature) > 0:
        # Weight by position to prefer earlier elbows
        position_weight = np.exp(-np.arange(len(curvature)) / (n * sensitivity))
        weighted_curvature = curvature * position_weight

        elbow_idx = np.argmin(weighted_curvature)  # Most negative = sharpest bend

        # The threshold is the value at the elbow point
        threshold = sorted_means[elbow_idx + 1]  # +1 due to diff offset
        n_selected = elbow_idx + 2  # Number of points before/at elbow

        return threshold, n_selected, {
            'method': 'elbow',
            'elbow_index': int(elbow_idx),
            'max_curvature': float(curvature[elbow_idx]),
            'status': 'success'
        }

    return sorted_means[n // 2], n // 2, {'method': 'elbow', 'status': 'fallback'}


def estimate_overlap_expected_count(posterior_means: np.ndarray) -> tuple:
    """
    Estimate the expected number of true pairs from posterior means.

    The posterior mean represents P(true pair | data). The sum of all posterior
    means gives the expected number of true pairs. Select top-k pairs where
    k = round(sum(posterior_means)).

    This is a principled Bayesian approach: E[# true pairs] = sum(P(true|data))

    Args:
        posterior_means: Array of posterior mean probabilities

    Returns:
        threshold: The posterior mean value of the k-th pair
        n_selected: Expected number of true pairs
        method_info: Dict with diagnostic information
    """
    if len(posterior_means) == 0:
        return 0.5, 0, {'method': 'expected_count', 'status': 'empty_input'}

    # Expected number of true pairs
    expected_count = np.sum(posterior_means)
    n_selected = int(np.round(expected_count))

    # Clamp to valid range
    n_selected = max(0, min(n_selected, len(posterior_means)))

    # Find the threshold: sort and take the n_selected-th value
    sorted_means = np.sort(posterior_means)[::-1]

    if n_selected > 0 and n_selected <= len(sorted_means):
        threshold = sorted_means[n_selected - 1]
    else:
        threshold = 0.5

    return threshold, n_selected, {
        'method': 'expected_count',
        'expected_count_raw': float(expected_count),
        'status': 'success'
    }


def estimate_overlap_gap_statistic(posterior_means: np.ndarray, min_gap_ratio: float = 0.1) -> tuple:
    """
    Find natural gaps in the sorted posterior distribution.

    Looks for large gaps (jumps) in the sorted posterior means that might
    indicate a natural separation between true and false pairs.

    Args:
        posterior_means: Array of posterior mean probabilities
        min_gap_ratio: Minimum gap size as ratio of max gap to consider

    Returns:
        threshold: Posterior mean value at the largest gap
        n_selected: Number of pairs above the gap
        method_info: Dict with diagnostic information
    """
    if len(posterior_means) < 3:
        return 0.5, 0, {'method': 'gap', 'status': 'insufficient_data'}

    sorted_means = np.sort(posterior_means)[::-1]
    gaps = -np.diff(sorted_means)  # Negative because sorted descending

    # Find significant gaps
    max_gap = np.max(gaps)
    significant_gaps = gaps > (max_gap * min_gap_ratio)

    if np.any(significant_gaps):
        # Find the first significant gap (prefer earlier cutoffs)
        first_sig_gap = np.argmax(significant_gaps)
        threshold = sorted_means[first_sig_gap + 1]
        n_selected = first_sig_gap + 1
    else:
        # No significant gap found, use median
        n_selected = len(sorted_means) // 2
        threshold = sorted_means[n_selected] if n_selected < len(sorted_means) else 0.5

    return threshold, n_selected, {
        'method': 'gap',
        'max_gap': float(max_gap),
        'gap_index': int(first_sig_gap) if np.any(significant_gaps) else -1,
        'status': 'success' if np.any(significant_gaps) else 'fallback'
    }


def infer_overlap_adaptive(
    posterior_stats: dict,
    method: str = 'ensemble',
    fallback_threshold: float = 0.5,
    min_pairs: int = 1,
    max_pairs_ratio: float = 1.0
) -> tuple:
    """
    Adaptively infer the overlapping pairs from posterior distribution.

    This is the main entry point for adaptive overlap inference. It combines
    multiple methods and uses ensemble voting to determine the final selection.

    Args:
        posterior_stats: Dict mapping pair_key -> {'posterior_mean': float, ...}
        method: Selection method:
            - 'otsu': Otsu's thresholding
            - 'gmm': Gaussian Mixture Model
            - 'elbow': Elbow/knee detection
            - 'expected': Expected count from posterior sum
            - 'gap': Gap statistic
            - 'ensemble': Combine multiple methods (recommended)
        fallback_threshold: Threshold to use if method fails
        min_pairs: Minimum number of pairs to select
        max_pairs_ratio: Maximum ratio of pairs to select (1.0 = all)

    Returns:
        selected_pair_keys: List of pair keys selected as true overlaps
        threshold_used: The threshold value used for selection
        method_info: Dict with diagnostic information from all methods
    """
    if not posterior_stats:
        return [], fallback_threshold, {'status': 'empty_input'}

    # Extract posterior means
    pair_keys = list(posterior_stats.keys())
    posterior_means = np.array([
        posterior_stats[k].get('posterior_mean', 0.0) for k in pair_keys
    ])

    if len(posterior_means) == 0:
        return [], fallback_threshold, {'status': 'no_posterior_means'}

    max_pairs = int(len(posterior_means) * max_pairs_ratio)

    all_methods_info = {}

    if method == 'ensemble':
        # Run all methods and combine results
        results = {}

        # Otsu
        thresh_otsu, n_otsu, info_otsu = estimate_overlap_otsu(posterior_means)
        results['otsu'] = (thresh_otsu, n_otsu)
        all_methods_info['otsu'] = info_otsu

        # GMM
        thresh_gmm, n_gmm, info_gmm = estimate_overlap_gmm(posterior_means)
        results['gmm'] = (thresh_gmm, n_gmm)
        all_methods_info['gmm'] = info_gmm

        # Elbow
        thresh_elbow, n_elbow, info_elbow = estimate_overlap_elbow(posterior_means)
        results['elbow'] = (thresh_elbow, n_elbow)
        all_methods_info['elbow'] = info_elbow

        # Expected count
        thresh_exp, n_exp, info_exp = estimate_overlap_expected_count(posterior_means)
        results['expected'] = (thresh_exp, n_exp)
        all_methods_info['expected'] = info_exp

        # Gap statistic
        thresh_gap, n_gap, info_gap = estimate_overlap_gap_statistic(posterior_means)
        results['gap'] = (thresh_gap, n_gap)
        all_methods_info['gap'] = info_gap

        # Ensemble: use median of n_selected values (robust to outliers)
        n_values = [n for _, n in results.values() if n > 0]
        if n_values:
            n_selected = int(np.median(n_values))
        else:
            n_selected = len(posterior_means) // 2

        # Compute corresponding threshold
        sorted_indices = np.argsort(posterior_means)[::-1]
        n_selected = max(min_pairs, min(n_selected, max_pairs))

        if n_selected > 0:
            threshold_used = posterior_means[sorted_indices[n_selected - 1]]
        else:
            threshold_used = fallback_threshold

        all_methods_info['ensemble'] = {
            'n_from_methods': {k: v[1] for k, v in results.items()},
            'final_n_selected': n_selected,
            'method': 'median_ensemble'
        }

    elif method == 'otsu':
        threshold_used, n_selected, info = estimate_overlap_otsu(posterior_means)
        all_methods_info['otsu'] = info

    elif method == 'gmm':
        threshold_used, n_selected, info = estimate_overlap_gmm(posterior_means)
        all_methods_info['gmm'] = info

    elif method == 'elbow':
        threshold_used, n_selected, info = estimate_overlap_elbow(posterior_means)
        all_methods_info['elbow'] = info

    elif method == 'expected':
        threshold_used, n_selected, info = estimate_overlap_expected_count(posterior_means)
        all_methods_info['expected'] = info

    elif method == 'gap':
        threshold_used, n_selected, info = estimate_overlap_gap_statistic(posterior_means)
        all_methods_info['gap'] = info

    else:
        # Fallback to fixed threshold
        threshold_used = fallback_threshold
        n_selected = np.sum(posterior_means > threshold_used)
        all_methods_info['fallback'] = {'threshold': threshold_used}

    # Enforce bounds
    n_selected = max(min_pairs, min(n_selected, max_pairs))

    # Select top-n pairs by posterior mean
    sorted_indices = np.argsort(posterior_means)[::-1][:n_selected]
    selected_pair_keys = [pair_keys[i] for i in sorted_indices]

    logger.debug(f"Adaptive overlap inference ({method}): selected {len(selected_pair_keys)} pairs "
                f"from {len(pair_keys)} candidates, threshold={threshold_used:.4f}")

    return selected_pair_keys, threshold_used, all_methods_info


def precompute_full_distance_matrices(emb1_unique, emb2_unique, ref_emb1, ref_emb2,
                                       ori_ref_emb1, ori_ref_emb2, args, device, use_gpu,
                                       is_normalized=False):
    """
    OPTIMIZATION: Precompute full distance matrices once before ensemble loop.

    Instead of computing distances to subset_ref_emb in each ensemble iteration
    (which is redundant since subset_ref_emb = ref_emb[subset_indices]), we compute
    distances to the full ref_emb once and extract subsets by column indexing.

    This reduces distance computation from O(n_ensembles * n_unique * n_subset)
    to O(n_unique * n_ref), a significant speedup for multiple ensembles.

    Args:
        emb1_unique, emb2_unique: Unique embeddings to compute distances for
        ref_emb1, ref_emb2: Full reference embeddings
        ori_ref_emb1, ori_ref_emb2: Original reference embeddings (for concat_seed_pairs)
        args: Arguments with distance_metric, transformation, etc.
        device: Computation device
        use_gpu: Whether to use GPU
        is_normalized: If True, skip normalization for cosine distance

    Returns:
        full_dist_vec1: Distance matrix (n_unique1, n_ref)
        full_dist_vec2: Distance matrix (n_unique2, n_ref)
        ori_dist_vec1: Distance to ori_ref (n_unique1, n_ori) or None
        ori_dist_vec2: Distance to ori_ref (n_unique2, n_ori) or None
    """
    # Get transformation parameters
    transformation = getattr(args, 'transformation', None)
    transformation_params = getattr(args, 'transformation_params', None)
    multi_gpu_config = getattr(args, 'multi_gpu_config', None)

    # Backward compatibility
    if transformation is None and getattr(args, 'use_rbf_distance_encoding', False):
        transformation = 'rbf'
        rbf_sigma_val = getattr(args, 'rbf_sigma', None)
        if rbf_sigma_val is not None:
            transformation_params = {'sigma': rbf_sigma_val}

    logger.debug(f"Precomputing full distance matrices: ({len(emb1_unique)}, {len(ref_emb1)}) and ({len(emb2_unique)}, {len(ref_emb2)})")

    # Compute full distance matrices (ONCE instead of n_ensembles times)
    full_dist_vec1 = compute_distance_encoding(
        emb=emb1_unique, ref_embeddings=ref_emb1, distance_metric=args.distance_metric,
        use_gpu=use_gpu, device=device, multi_gpu_config=multi_gpu_config,
        transformation=transformation, transformation_params=transformation_params,
        is_normalized=is_normalized)

    full_dist_vec2 = compute_distance_encoding(
        emb=emb2_unique, ref_embeddings=ref_emb2, distance_metric=args.distance_metric,
        use_gpu=use_gpu, device=device, multi_gpu_config=multi_gpu_config,
        transformation=transformation, transformation_params=transformation_params,
        is_normalized=is_normalized)

    # Also precompute distances to original refs if needed for concat_seed_pairs
    ori_dist_vec1 = None
    ori_dist_vec2 = None
    if ori_ref_emb1 is not None and ori_ref_emb2 is not None:
        logger.debug(f"Precomputing ori_ref distance matrices: ({len(emb1_unique)}, {len(ori_ref_emb1)})")
        ori_dist_vec1 = compute_distance_encoding(
            emb=emb1_unique, ref_embeddings=ori_ref_emb1, distance_metric=args.distance_metric,
            use_gpu=use_gpu, device=device, multi_gpu_config=multi_gpu_config,
            transformation=transformation, transformation_params=transformation_params,
            is_normalized=is_normalized)
        ori_dist_vec2 = compute_distance_encoding(
            emb=emb2_unique, ref_embeddings=ori_ref_emb2, distance_metric=args.distance_metric,
            use_gpu=use_gpu, device=device, multi_gpu_config=multi_gpu_config,
            transformation=transformation, transformation_params=transformation_params,
            is_normalized=is_normalized)

    # Harmonize devices: compute_distance_encoding may place matrices on different
    # devices depending on available GPU memory. If there's a mismatch, move all to
    # CPU (since CPU placement was due to memory pressure — moving back to GPU risks OOM).
    all_tensors = [full_dist_vec1, full_dist_vec2, ori_dist_vec1, ori_dist_vec2]
    devices = {t.device for t in all_tensors if isinstance(t, torch.Tensor)}
    if len(devices) > 1:
        logger.debug(f"Device mismatch in precomputed matrices: {devices}. Moving all to CPU.")
        if isinstance(full_dist_vec1, torch.Tensor) and full_dist_vec1.device.type != 'cpu':
            full_dist_vec1 = full_dist_vec1.cpu()
        if isinstance(full_dist_vec2, torch.Tensor) and full_dist_vec2.device.type != 'cpu':
            full_dist_vec2 = full_dist_vec2.cpu()
        if isinstance(ori_dist_vec1, torch.Tensor) and ori_dist_vec1.device.type != 'cpu':
            ori_dist_vec1 = ori_dist_vec1.cpu()
        if isinstance(ori_dist_vec2, torch.Tensor) and ori_dist_vec2.device.type != 'cpu':
            ori_dist_vec2 = ori_dist_vec2.cpu()

    return full_dist_vec1, full_dist_vec2, ori_dist_vec1, ori_dist_vec2


def extend_precomputed_distance_matrices(prev_full_dist_vec1, prev_full_dist_vec2,
                                          emb1_unique, emb2_unique,
                                          new_ref_emb1, new_ref_emb2,
                                          args, device, use_gpu, is_normalized=False):
    """
    Incrementally extend precomputed distance matrices by computing only new columns.

    When reference embeddings grow from R_{N-1} to R_N, the first R_{N-1} columns
    of the distance matrix are unchanged. This function computes only the new columns
    for the newly added references and concatenates them with the cached matrices.

    Args:
        prev_full_dist_vec1/2: Cached (n_unique, prev_n_ref) distance matrices
        emb1_unique, emb2_unique: Unique embeddings (unchanged between iterations)
        new_ref_emb1, new_ref_emb2: Only the NEW reference embeddings to add
        args, device, use_gpu, is_normalized: Standard params

    Returns:
        extended_dist_vec1, extended_dist_vec2: (n_unique, prev_n_ref + n_new_ref) matrices
    """
    transformation = getattr(args, 'transformation', None)
    transformation_params = getattr(args, 'transformation_params', None)
    multi_gpu_config = getattr(args, 'multi_gpu_config', None)

    if transformation is None and getattr(args, 'use_rbf_distance_encoding', False):
        transformation = 'rbf'
        rbf_sigma_val = getattr(args, 'rbf_sigma', None)
        if rbf_sigma_val is not None:
            transformation_params = {'sigma': rbf_sigma_val}

    n_new = len(new_ref_emb1)
    logger.debug(f"Extending distance matrices: adding {n_new} new ref columns")

    # Compute distances only for new reference columns
    new_dist_vec1 = compute_distance_encoding(
        emb=emb1_unique, ref_embeddings=new_ref_emb1, distance_metric=args.distance_metric,
        use_gpu=use_gpu, device=device, multi_gpu_config=multi_gpu_config,
        transformation=transformation, transformation_params=transformation_params,
        is_normalized=is_normalized)

    new_dist_vec2 = compute_distance_encoding(
        emb=emb2_unique, ref_embeddings=new_ref_emb2, distance_metric=args.distance_metric,
        use_gpu=use_gpu, device=device, multi_gpu_config=multi_gpu_config,
        transformation=transformation, transformation_params=transformation_params,
        is_normalized=is_normalized)

    # Concatenate with cached matrices, harmonizing devices if needed
    if isinstance(prev_full_dist_vec1, torch.Tensor):
        if not isinstance(new_dist_vec1, torch.Tensor):
            new_dist_vec1 = torch.from_numpy(new_dist_vec1)
        if not isinstance(new_dist_vec2, torch.Tensor):
            new_dist_vec2 = torch.from_numpy(new_dist_vec2)
        # Check for device mismatch across all tensors
        devices = {prev_full_dist_vec1.device, prev_full_dist_vec2.device,
                   new_dist_vec1.device, new_dist_vec2.device}
        if len(devices) > 1:
            target_device = torch.device('cpu')
            logger.debug(f"Device mismatch in extend: {devices}. Harmonizing to CPU.")
        else:
            target_device = prev_full_dist_vec1.device
        extended_dist_vec1 = torch.cat([prev_full_dist_vec1.to(target_device), new_dist_vec1.to(target_device)], dim=1)
        extended_dist_vec2 = torch.cat([prev_full_dist_vec2.to(target_device), new_dist_vec2.to(target_device)], dim=1)
    else:
        if isinstance(new_dist_vec1, torch.Tensor):
            new_dist_vec1 = new_dist_vec1.cpu().numpy()
            new_dist_vec2 = new_dist_vec2.cpu().numpy()
        extended_dist_vec1 = np.concatenate([prev_full_dist_vec1, new_dist_vec1], axis=1)
        extended_dist_vec2 = np.concatenate([prev_full_dist_vec2, new_dist_vec2], axis=1)

    return extended_dist_vec1, extended_dist_vec2


def build_spatial_tiles(pool_emb1, pool_emb2, anchor_local_e1, anchor_local_e2,
                        n_tiles, overlap_k=2, device=None):
    """Partition pool into n_tiles overlapping spatial tiles using anchor pairs.

    Uses k-means on anchor embeddings (emb1 side) to define tile centers,
    then assigns each pool point to its top-overlap_k nearest tile centers.
    Corresponding emb2 tile centers are derived from paired anchor embeddings.

    Args:
        pool_emb1: (n_pool1, dim1) pool embeddings in space 1
        pool_emb2: (n_pool2, dim2) pool embeddings in space 2
        anchor_local_e1: pool-local indices of anchor points in emb1
        anchor_local_e2: pool-local indices of anchor points in emb2 (paired with anchor_local_e1)
        n_tiles: number of spatial tiles (G)
        overlap_k: assign each point to top-k nearest tile centers (default 2)
        device: torch device for GPU-accelerated assignment

    Returns:
        (tile_indices_e1, tile_indices_e2): lists of n_tiles np.arrays of pool-local indices
    """
    import time as _time
    _t0 = _time.time()

    n_anchors = len(anchor_local_e1)
    n_tiles = min(n_tiles, max(n_anchors, 3))

    # Extract anchor embeddings (used for emb2 center derivation via pairing)
    anchor_emb1 = pool_emb1[anchor_local_e1]  # (n_anchors, dim1)
    anchor_emb2 = pool_emb2[anchor_local_e2]  # (n_anchors, dim2)

    # K-means on a UNIFORM SUBSAMPLE of the pool (not just anchors) for spatial uniformity.
    # Anchors cluster in "already found" regions; pool subsample covers the full space.
    from sklearn.cluster import MiniBatchKMeans
    n_subsample = min(20000, len(pool_emb1))
    rng = np.random.RandomState(42)
    subsample_idx = rng.choice(len(pool_emb1), n_subsample, replace=False)
    subsample_emb1 = pool_emb1[subsample_idx]
    subsample_emb1_normed = subsample_emb1 / (np.linalg.norm(subsample_emb1, axis=1, keepdims=True) + 1e-8)

    kmeans = MiniBatchKMeans(n_clusters=n_tiles, batch_size=min(10000, n_subsample),
                             n_init=3, max_iter=50, random_state=42)
    kmeans.fit(subsample_emb1_normed)

    # E1 centers: directly from k-means (uniform spatial coverage)
    centers_e1 = kmeans.cluster_centers_.astype(np.float32)  # already normed by k-means input

    # E2 centers: for each e1 cluster, find anchors closest to that cluster center,
    # use their paired emb2 embeddings to derive the corresponding e2 center.
    anchor_emb1_normed = anchor_emb1 / (np.linalg.norm(anchor_emb1, axis=1, keepdims=True) + 1e-8)
    anchor_cluster_sims = anchor_emb1_normed @ centers_e1.T  # (n_anchors, n_tiles)
    anchor_cluster_labels = anchor_cluster_sims.argmax(axis=1)  # each anchor → nearest tile

    centers_e2 = np.zeros((n_tiles, pool_emb2.shape[1]), dtype=np.float32)
    for g in range(n_tiles):
        mask = anchor_cluster_labels == g
        if mask.sum() > 0:
            centers_e2[g] = anchor_emb2[mask].mean(axis=0)
        else:
            # No anchors near this tile — use overall anchor mean as fallback
            centers_e2[g] = anchor_emb2.mean(axis=0)

    # Normalize centers for cosine similarity
    centers_e1 /= (np.linalg.norm(centers_e1, axis=1, keepdims=True) + 1e-8)
    centers_e2 /= (np.linalg.norm(centers_e2, axis=1, keepdims=True) + 1e-8)

    # Assign pool points to top-overlap_k nearest tile centers via GPU cosine similarity
    use_gpu = device is not None and device.type == 'cuda' and torch.cuda.is_available()
    overlap_k_eff = min(overlap_k, n_tiles)

    def _assign_to_tiles(pool_emb, centers, n_pool, n_tiles_local):
        """Assign each pool point to its top-overlap_k nearest tile centers."""
        tile_members = [[] for _ in range(n_tiles_local)]

        if use_gpu and n_pool > 10000:
            # GPU path: chunked matmul for large pools
            centers_t = torch.from_numpy(centers).float().to(device)  # (n_tiles, dim)
            pool_normed = pool_emb / (np.linalg.norm(pool_emb, axis=1, keepdims=True) + 1e-8)
            chunk_size = min(500_000, n_pool)
            for start in range(0, n_pool, chunk_size):
                end = min(start + chunk_size, n_pool)
                chunk = torch.from_numpy(pool_normed[start:end].astype(np.float32)).to(device)
                sims = chunk @ centers_t.T  # (chunk, n_tiles)
                _, topk_idx = sims.topk(overlap_k_eff, dim=1)  # (chunk, overlap_k)
                topk_idx = topk_idx.cpu().numpy()
                for local_i in range(end - start):
                    for tile_g in topk_idx[local_i]:
                        tile_members[tile_g].append(start + local_i)
                del chunk, sims, topk_idx
        else:
            # CPU path
            pool_normed = pool_emb / (np.linalg.norm(pool_emb, axis=1, keepdims=True) + 1e-8)
            sims = pool_normed @ centers.T  # (n_pool, n_tiles)
            topk_idx = np.argpartition(-sims, overlap_k_eff, axis=1)[:, :overlap_k_eff]
            for i in range(n_pool):
                for tile_g in topk_idx[i]:
                    tile_members[tile_g].append(i)

        return [np.array(members, dtype=np.int64) for members in tile_members]

    tile_indices_e1 = _assign_to_tiles(pool_emb1, centers_e1, len(pool_emb1), n_tiles)
    tile_indices_e2 = _assign_to_tiles(pool_emb2, centers_e2, len(pool_emb2), n_tiles)

    _elapsed = _time.time() - _t0
    sizes_e1 = [len(t) for t in tile_indices_e1]
    sizes_e2 = [len(t) for t in tile_indices_e2]
    logger.info(f"Built {n_tiles} spatial tiles in {_elapsed:.1f}s "
                f"(e1: {min(sizes_e1):,}-{max(sizes_e1):,}, e2: {min(sizes_e2):,}-{max(sizes_e2):,}, "
                f"overlap_k={overlap_k_eff}, anchors={n_anchors:,})")

    return tile_indices_e1, tile_indices_e2


def process_ensemble_with_precomputed_distances(
    ensemble_idx, full_dist_vec1, full_dist_vec2, ori_dist_vec1, ori_dist_vec2,
    ind_emb1_unique, ind_emb2_unique, ref_subset_indices, args, device, use_gpu,
    concat_seed_pairs, anchor_mode, max_view_pool=None, spatial_tiles=None):
    """
    OPTIMIZED: Process a single ensemble using precomputed full distance matrices.

    Instead of computing distances from emb_unique to subset_ref_emb, this function
    extracts the relevant columns from precomputed full distance matrices.

    This is mathematically equivalent because:
        dist(emb[i], ref[subset_indices][j]) == dist(emb[i], ref[subset_indices[j]])

    So: compute_distances(emb, ref[subset_indices]) == compute_distances(emb, ref)[:, subset_indices]

    Args:
        ensemble_idx: Index of this ensemble
        full_dist_vec1: Precomputed (n_unique1, n_ref) distance matrix
        full_dist_vec2: Precomputed (n_unique2, n_ref) distance matrix
        ori_dist_vec1/2: Precomputed distances to ori_ref (or None)
        ind_emb1_unique, ind_emb2_unique: Original indices
        ref_subset_indices: Which columns to extract for this ensemble
        args, device, use_gpu: Standard arguments
        concat_seed_pairs, anchor_mode: For supervised/ood modes
        spatial_tiles: Optional (tile_indices_e1, tile_indices_e2) from build_spatial_tiles.
            When provided, MNN is run per-tile and results are unioned.

    Returns:
        (ensemble_idx, subset_mutual_pairs, subset_accuracy, mutual_nn)
    """
    # OPTIMIZATION: Extract subset by column indexing (very fast!)
    # This replaces the expensive compute_distance_encoding call that was done per-ensemble
    if isinstance(full_dist_vec1, torch.Tensor):
        ref_subset_indices_t = torch.tensor(ref_subset_indices, dtype=torch.long, device=full_dist_vec1.device)
        dist_vec1_subset = full_dist_vec1[:, ref_subset_indices_t].clone()
        if isinstance(full_dist_vec2, torch.Tensor) and full_dist_vec2.device != full_dist_vec1.device:
            ref_subset_indices_t2 = ref_subset_indices_t.to(full_dist_vec2.device)
            dist_vec2_subset = full_dist_vec2[:, ref_subset_indices_t2].clone().to(full_dist_vec1.device)
        else:
            dist_vec2_subset = full_dist_vec2[:, ref_subset_indices_t].clone()
    else:
        dist_vec1_subset = full_dist_vec1[:, ref_subset_indices].copy()
        dist_vec2_subset = full_dist_vec2[:, ref_subset_indices].copy()

    # Handle concat_seed_pairs case (concatenate ori_ref distances)
    # NOTE: When full distance matrices are pre-normalized (L2 row-wise), column subsets
    # are NOT unit-norm. However, concatenating pre-normalized ori columns with pre-normalized
    # subset columns requires re-normalization. Only skip if no concat.
    needs_renorm = False
    if concat_seed_pairs and anchor_mode in ("supervised", "ood") and ori_dist_vec1 is not None:
        if isinstance(dist_vec1_subset, torch.Tensor):
            ori_dist_vec1_t = torch.as_tensor(ori_dist_vec1, device=dist_vec1_subset.device, dtype=dist_vec1_subset.dtype)
            ori_dist_vec2_t = torch.as_tensor(ori_dist_vec2, device=dist_vec2_subset.device, dtype=dist_vec2_subset.dtype)
            dist_vec1_subset = torch.cat((ori_dist_vec1_t, dist_vec1_subset), dim=1)
            dist_vec2_subset = torch.cat((ori_dist_vec2_t, dist_vec2_subset), dim=1)
        else:
            dist_vec1_subset = np.concatenate((ori_dist_vec1, dist_vec1_subset), axis=1)
            dist_vec2_subset = np.concatenate((ori_dist_vec2, dist_vec2_subset), axis=1)
        needs_renorm = True  # Concatenation breaks row normalization

    # Per-view local selection: restrict MNN to pool points most relevant to this view's refs.
    # Points far from the view's refs have degenerate distance vectors → noise in MNN.
    # The row norm of the column subset (before re-normalization) measures how well a point
    # is represented by this view's refs — high norm = discriminative, low norm = noise.
    view_idx1 = None
    view_idx2 = None
    if max_view_pool is not None and len(dist_vec1_subset) > max_view_pool:
        if isinstance(dist_vec1_subset, torch.Tensor):
            norms1 = torch.norm(dist_vec1_subset, dim=1)
            norms2 = torch.norm(dist_vec2_subset, dim=1)
            _, view_idx1 = norms1.topk(min(max_view_pool, len(norms1)))
            _, view_idx2 = norms2.topk(min(max_view_pool, len(norms2)))
            view_idx1 = view_idx1.cpu().numpy()
            view_idx2 = view_idx2.cpu().numpy()
        else:
            norms1 = np.linalg.norm(dist_vec1_subset, axis=1)
            norms2 = np.linalg.norm(dist_vec2_subset, axis=1)
            view_idx1 = np.argpartition(norms1, -max_view_pool)[-max_view_pool:]
            view_idx2 = np.argpartition(norms2, -max_view_pool)[-max_view_pool:]
        dist_vec1_subset = dist_vec1_subset[view_idx1]
        dist_vec2_subset = dist_vec2_subset[view_idx2]
        ind_emb1_view = ind_emb1_unique[view_idx1]
        ind_emb2_view = ind_emb2_unique[view_idx2]
    else:
        ind_emb1_view = ind_emb1_unique
        ind_emb2_view = ind_emb2_unique

    # Re-normalize: column subsets of a row-normalized matrix are NOT row-normalized.
    if isinstance(dist_vec1_subset, torch.Tensor):
        dist_vec1_subset = torch.nn.functional.normalize(dist_vec1_subset, p=2, dim=1)
        dist_vec2_subset = torch.nn.functional.normalize(dist_vec2_subset, p=2, dim=1)
    else:
        n1 = np.linalg.norm(dist_vec1_subset, axis=1, keepdims=True); n1[n1 < 1e-8] = 1.0; dist_vec1_subset /= n1
        n2 = np.linalg.norm(dist_vec2_subset, axis=1, keepdims=True); n2[n2 < 1e-8] = 1.0; dist_vec2_subset /= n2

    # Find mutual pairs — either tiled (forward per tile + global mutual) or single-pool
    if spatial_tiles is not None:
        # Tiled forward NN + global mutual check:
        # 1. Per tile: forward search only (p2→best p1, p1→best p2)
        # 2. Global merge: for each point, keep best match across all tiles
        # 3. Mutual check: if p2's global best is p1 AND p1's global best is p2 → mutual pair
        tile_indices_e1, tile_indices_e2 = spatial_tiles
        n_pool1 = len(ind_emb1_view) if view_idx1 is None else len(ind_emb1_view)
        n_pool2 = len(ind_emb2_view) if view_idx2 is None else len(ind_emb2_view)

        # Global best-match arrays (pool-level indices)
        # For each p2 (pool idx), track best p1 (pool idx) and similarity
        # For each p1 (pool idx), track best p2 (pool idx) and similarity
        n_e1 = len(dist_vec1_subset)  # number of emb1 points (view-restricted or full pool)
        n_e2 = len(dist_vec2_subset)  # number of emb2 points
        best_fwd_idx = np.full(n_e2, -1, dtype=np.int64)   # p2 → best p1 (view-local idx)
        best_fwd_sim = np.full(n_e2, -np.inf, dtype=np.float32)
        best_rev_idx = np.full(n_e1, -1, dtype=np.int64)   # p1 → best p2 (view-local idx)
        best_rev_sim = np.full(n_e1, -np.inf, dtype=np.float32)

        # Map tile pool-level indices to view-local indices (for indexing dist_vec_subset)
        if view_idx1 is not None:
            view_set1 = set(view_idx1.tolist())
            view_set2 = set(view_idx2.tolist())
            pool_to_view1 = {int(p): v for v, p in enumerate(view_idx1)}
            pool_to_view2 = {int(p): v for v, p in enumerate(view_idx2)}
        else:
            view_set1 = view_set2 = None
            pool_to_view1 = pool_to_view2 = None

        for t in range(len(tile_indices_e1)):
            t_pool_idx1 = tile_indices_e1[t]
            t_pool_idx2 = tile_indices_e2[t]

            # Intersect with view if view restriction active
            if view_set1 is not None:
                t_pool_idx1 = np.array([p for p in t_pool_idx1 if p in view_set1], dtype=np.int64)
                t_pool_idx2 = np.array([p for p in t_pool_idx2 if p in view_set2], dtype=np.int64)

            if len(t_pool_idx1) < 10 or len(t_pool_idx2) < 10:
                continue

            # Get view-local indices
            if pool_to_view1 is not None:
                t_view_idx1 = np.array([pool_to_view1[int(p)] for p in t_pool_idx1], dtype=np.int64)
                t_view_idx2 = np.array([pool_to_view2[int(p)] for p in t_pool_idx2], dtype=np.int64)
            else:
                t_view_idx1 = t_pool_idx1
                t_view_idx2 = t_pool_idx2

            tile_dv1 = dist_vec1_subset[t_view_idx1]  # (tile_n1, subset_refs)
            tile_dv2 = dist_vec2_subset[t_view_idx2]  # (tile_n2, subset_refs)

            # Forward search: for each p2 in tile, find top-k p1 candidates
            tile_topk = min(3, len(t_view_idx1))
            fwd_idx, fwd_sim = get_topk(tile_dv1, tile_dv2, k=tile_topk,
                                         metric=args.distance_metric, return_dist=True,
                                         use_faiss=True, is_normalized=True)
            # fwd_idx: (tile_n2, k), fwd_sim: (tile_n2, k)
            if isinstance(fwd_idx, torch.Tensor):
                fwd_idx = fwd_idx.cpu().numpy()
                fwd_sim = fwd_sim.cpu().numpy()

            # Reverse search: for each p1 in tile, find top-k p2 candidates
            rev_topk = min(3, len(t_view_idx2))
            rev_idx, rev_sim = get_topk(tile_dv2, tile_dv1, k=rev_topk,
                                         metric=args.distance_metric, return_dist=True,
                                         use_faiss=True, is_normalized=True)
            if isinstance(rev_idx, torch.Tensor):
                rev_idx = rev_idx.cpu().numpy()
                rev_sim = rev_sim.cpu().numpy()

            # Update global best matches from ALL top-k candidates (vectorized)
            # Forward: for each p2, check each of its k candidates against global best
            for ki in range(tile_topk):
                fwd_idx_k = fwd_idx[:, ki] if fwd_idx.ndim > 1 else fwd_idx
                fwd_sim_k = fwd_sim[:, ki] if fwd_sim.ndim > 1 else fwd_sim
                fwd_matched_view1 = t_view_idx1[fwd_idx_k.astype(np.int64)]
                fwd_sim_f32 = fwd_sim_k.astype(np.float32)
                better_fwd = fwd_sim_f32 > best_fwd_sim[t_view_idx2]
                if better_fwd.any():
                    update_idx2 = t_view_idx2[better_fwd]
                    best_fwd_sim[update_idx2] = fwd_sim_f32[better_fwd]
                    best_fwd_idx[update_idx2] = fwd_matched_view1[better_fwd]

            # Reverse: for each p1, check each of its k candidates against global best
            for ki in range(rev_topk):
                rev_idx_k = rev_idx[:, ki] if rev_idx.ndim > 1 else rev_idx
                rev_sim_k = rev_sim[:, ki] if rev_sim.ndim > 1 else rev_sim
                rev_matched_view2 = t_view_idx2[rev_idx_k.astype(np.int64)]
                rev_sim_f32 = rev_sim_k.astype(np.float32)
                better_rev = rev_sim_f32 > best_rev_sim[t_view_idx1]
                if better_rev.any():
                    update_idx1 = t_view_idx1[better_rev]
                    best_rev_sim[update_idx1] = rev_sim_f32[better_rev]
                    best_rev_idx[update_idx1] = rev_matched_view2[better_rev]

        # Global mutual check (vectorized): p2's best is p1 AND p1's best is p2
        has_match = best_fwd_idx >= 0  # p2 has a forward match
        candidate_i2 = np.where(has_match)[0]
        candidate_j1 = best_fwd_idx[candidate_i2]
        # Check mutuality: p1's best p2 must be this p2
        is_mutual = best_rev_idx[candidate_j1] == candidate_i2
        mutual_i2 = candidate_i2[is_mutual]
        mutual_j1 = candidate_j1[is_mutual]
        mutual_sims = best_fwd_sim[mutual_i2]
        subset_mutual_pairs = list(zip(mutual_i2.tolist(), mutual_j1.tolist(), mutual_sims.tolist()))

        # Map back to pool-level indices if view restriction was applied
        if view_idx1 is not None:
            mapped_pairs = []
            for i2_view, j1_view, sim in subset_mutual_pairs:
                mapped_pairs.append((view_idx2[i2_view], view_idx1[j1_view], sim))
            subset_mutual_pairs = mapped_pairs

        mutual_nn = len(subset_mutual_pairs)
        # Compute correct count using pool-level indices
        if view_idx1 is not None:
            correct = sum(1 for i, j, _ in subset_mutual_pairs
                          if ind_emb1_unique[j] == ind_emb2_unique[i])
        else:
            correct = sum(1 for i, j, _ in subset_mutual_pairs
                          if ind_emb1_unique[j] == ind_emb2_unique[i])
    else:
        # Standard single-pool MNN
        n_unique_max = max(len(ind_emb1_view), len(ind_emb2_view))
        use_approx = (n_unique_max > 100_000)
        subset_mutual_pairs, mutual_nn, correct = find_mutual_pairs(
            dist_vec1_subset, dist_vec2_subset, ind_emb1_view, ind_emb2_view, args, device, use_gpu,
            is_normalized=True, approximate_mnn=use_approx)

        # Map pairs back to pool-level indices so Bernoulli aggregation is consistent across views
        if view_idx1 is not None:
            mapped_pairs = []
            for i_local, j_local, dist in subset_mutual_pairs:
                mapped_pairs.append((view_idx2[i_local], view_idx1[j_local], dist))
            subset_mutual_pairs = mapped_pairs

    subset_accuracy = correct / mutual_nn if mutual_nn > 0 else 0.0

    return ensemble_idx, subset_mutual_pairs, subset_accuracy, mutual_nn, view_idx1, view_idx2


def calculate_safe_max_workers(n_ensembles, n_unique_samples, n_ref_subset, embedding_dim_avg, use_gpu, device, safety_factor=0.25, max_parallel_workers=None):
    """
    Calculate safe max_workers based on available memory to prevent OOM.

    Args:
        max_parallel_workers: Optional hard limit on parallel workers (useful for large datasets
                             where multiprocessing overhead causes OOM even with memory-based limits)

    Returns:
        tuple: (max_workers, should_use_sequential)
            - max_workers: Number of safe parallel workers (1 to n_ensembles)
            - should_use_sequential: True if memory too low for parallel execution
    """
    # Memory per worker: ref embeddings + unique embeddings + temporary computation buffers
    ref_memory_gb = 2 * n_ref_subset * embedding_dim_avg * 4 / (1024**3)
    unique_memory_gb = 2 * n_unique_samples * embedding_dim_avg * 4 / (1024**3)

    # For GPU: distance matrices are stored on CPU, but we need temporary GPU buffers for computation
    # Each worker needs space for input tensors + intermediate computation (~30% of distance matrix size)
    if use_gpu:
        # Be MUCH more conservative for GPU - each worker needs significant memory
        # Account for: input tensors, output tensors, intermediate computations, and PyTorch overhead
        dist_matrix_memory_gb = 2 * estimate_matrix_memory_gb(n_unique_samples, n_ref_subset) * 0.5
    else:
        # Full distance matrix in memory for CPU
        dist_matrix_memory_gb = 2 * estimate_matrix_memory_gb(n_unique_samples, n_ref_subset)

    serialization_overhead_gb = (ref_memory_gb + unique_memory_gb) * 0.5  # Increase overhead estimate
    memory_per_worker_gb = ref_memory_gb + unique_memory_gb + dist_matrix_memory_gb + serialization_overhead_gb

    available_memory_gb = get_available_memory_gb(use_gpu=use_gpu, device=device)
    memory_type = "GPU" if (use_gpu and device is not None and device.type == 'cuda') else "RAM"

    # Check if we have enough memory for even 1 worker with safety margin
    min_required_memory_gb = memory_per_worker_gb * 1.5  # Need 50% headroom (increased from 20%)

    if available_memory_gb < min_required_memory_gb:
        # Critical memory shortage - force sequential execution
        logger.warning(f"CRITICAL MEMORY SHORTAGE: {available_memory_gb:.2f} GB {memory_type} available, "
                      f"but need {min_required_memory_gb:.2f} GB for safe parallel execution")
        logger.warning(f"Automatically switching to SEQUENTIAL mode (no multiprocessing)")
        logger.warning(f"This will be slower but won't cause OOM errors")

        # Check for competing processes
        if use_gpu and device is not None and device.type == 'cuda':
            try:
                import subprocess
                result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                                       '--format=csv,noheader,nounits'],
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0 and result.stdout.strip():
                    processes = result.stdout.strip().split('\n')
                    if len(processes) > 1:
                        logger.warning(f"Detected {len(processes)} competing GPU processes:")
                        for proc in processes[:5]:  # Show first 5
                            logger.warning(f"  - PID using GPU memory: {proc}")
            except Exception:
                pass  # Ignore errors in process detection

        return 1, True  # Use sequential mode

    # For GPU, use balanced safety factor that allows reasonable parallelism
    if use_gpu and device is not None and device.type == 'cuda':
        safety_factor = 0.2  # Balanced for GPU with parallel workers (allows 4-5 workers)
        logger.debug(f"GPU mode: using balanced safety_factor={safety_factor} for performance")
    else:
        # Adaptive safety factor for CPU
        if available_memory_gb < memory_per_worker_gb * 3:
            # Low memory: be more aggressive (use up to 40% of available)
            adjusted_safety_factor = 0.4
            logger.debug(f"Low memory detected, increasing safety_factor: {safety_factor:.2f} -> {adjusted_safety_factor:.2f}")
            safety_factor = adjusted_safety_factor

    usable_memory_gb = available_memory_gb * safety_factor
    max_workers = max(1, int(usable_memory_gb / memory_per_worker_gb))
    max_workers = min(max_workers, n_ensembles)

    # For GPU, cap at a reasonable number regardless of memory calculation
    if use_gpu and device is not None and device.type == 'cuda':
        gpu_max_workers = 12  # Maximum 6 parallel GPU workers for good parallelism without excessive contention
        if max_workers > gpu_max_workers:
            logger.debug(f"GPU mode: capping workers at {gpu_max_workers} for optimal performance (calculated {max_workers})")
            max_workers = gpu_max_workers

    # Apply user-specified limit for large datasets (reduces multiprocessing overhead)
    if max_parallel_workers is not None and max_parallel_workers > 0:
        if max_workers > max_parallel_workers:
            logger.debug(f"Applying max_parallel_workers limit: {max_workers} -> {max_parallel_workers}")
            max_workers = max_parallel_workers

    logger.debug(f"Memory-aware worker calculation: {available_memory_gb:.2f} GB {memory_type} available, "
                f"{memory_per_worker_gb:.2f} GB per worker, max_workers={max_workers}/{n_ensembles}")
    if max_workers < n_ensembles:
        logger.warning(f"Reducing workers from {n_ensembles} to {max_workers} due to memory constraints")

    return max_workers, False  # Use parallel mode with limited workers


def run_ensembles_in_batches(ensemble_args, max_workers, n_ensembles, ctx):
    """Run ensembles in batches to avoid OOM errors with better GPU load balancing."""
    import gc
    import time
    results_dict = {}

    # Detect number of GPUs from ensemble_args
    n_gpus = 1
    if ensemble_args:
        # gpu_id is at index 10 in the args tuple
        gpu_ids_in_use = set()
        for args in ensemble_args:
            if len(args) > 10 and args[10] is not None:
                gpu_ids_in_use.add(args[10])
        n_gpus = len(gpu_ids_in_use) if gpu_ids_in_use else 1

    # For multi-GPU, increase batch size to better utilize both GPUs
    if n_gpus > 1:
        # Use at least 2x the number of GPUs to ensure both are busy
        effective_max_workers = max(max_workers, n_gpus * 2)
        logger.debug(f"Multi-GPU detected ({n_gpus} GPUs): increasing batch size from {max_workers} to {effective_max_workers} for better utilization")
        max_workers = effective_max_workers

    n_batches = (n_ensembles + max_workers - 1) // max_workers
    logger.debug(f"Running {n_ensembles} ensembles in {n_batches} batch(es) with {max_workers} workers per batch")

    for batch_idx in range(n_batches):
        batch_start = batch_idx * max_workers
        batch_end = min(batch_start + max_workers, n_ensembles)
        batch_size = batch_end - batch_start
        batch_args = ensemble_args[batch_start:batch_end]

        logger.debug(f"Batch {batch_idx + 1}/{n_batches}: ensembles {batch_start}-{batch_end-1}")

        # Log GPU distribution in this batch
        if n_gpus > 1:
            gpu_counts = {}
            for args in batch_args:
                gpu_id = args[10] if len(args) > 10 else 0
                gpu_counts[gpu_id] = gpu_counts.get(gpu_id, 0) + 1
            logger.debug(f"  GPU distribution in batch: {gpu_counts}")

        # Clean up before starting new batch (only between batches, not first batch)
        if batch_idx > 0:
            gc.collect()
            if torch.cuda.is_available():
                for gpu_id in range(n_gpus):
                    with torch.cuda.device(gpu_id):
                        torch.cuda.empty_cache()
                time.sleep(0.3)

        with ProcessPoolExecutor(max_workers=batch_size, mp_context=ctx) as executor:
            future_to_idx = {}
            # Submit all jobs at once for better parallelism across GPUs
            for args in batch_args:
                future = executor.submit(run_single_ensemble_gpu, args)
                future_to_idx[future] = args[0]

            for future in as_completed(future_to_idx):
                ensemble_idx, subset_mutual_pairs, subset_accuracy, mutual_nn = future.result()
                results_dict[ensemble_idx] = (subset_mutual_pairs, subset_accuracy, mutual_nn)

        logger.debug(f"Batch {batch_idx + 1}/{n_batches} completed")

    return results_dict


def _prep_ref_for_dist(ref_emb1, distance_metric, use_gpu, device):
    """Move ref embeddings to GPU/CPU and pre-normalize for cosine.

    Returns (ref_tensor_or_array, on_gpu_bool). For cosine, ref is L2-normalized
    so that ref @ ref[i] yields cosine similarity in [−1, 1]; we use 1 − sim
    as cosine distance. For euclidean we use squared L2 (monotonic with L2,
    sqrt is unnecessary for argmax/argmin).
    """
    on_gpu = bool(torch.cuda.is_available() and use_gpu)
    if on_gpu:
        ref = torch.from_numpy(ref_emb1).float().to(device, non_blocking=True)
        if distance_metric == 'cosine':
            ref = torch.nn.functional.normalize(ref, p=2, dim=1)
    else:
        ref = ref_emb1.astype(np.float32, copy=False)
        if distance_metric == 'cosine':
            n = np.linalg.norm(ref, axis=1, keepdims=True)
            n[n < 1e-8] = 1.0
            ref = ref / n
    return ref, on_gpu


def _row_dist_gpu(ref_norm_t, idx, distance_metric):
    """Distance from ref[idx] to all rows. Returns a 1-D tensor on the same device."""
    if distance_metric == 'cosine':
        # Cosine distance = 1 - cosine similarity. Adding constant 1.0 doesn't
        # change argmax/argmin, so we return -similarity to save one op (smaller
        # value = closer = lower distance).
        return -(ref_norm_t @ ref_norm_t[idx])
    # Squared euclidean (monotone with L2).
    diff = ref_norm_t - ref_norm_t[idx].unsqueeze(0)
    return (diff * diff).sum(dim=1)


def _row_dist_cpu(ref_norm_a, idx, distance_metric):
    if distance_metric == 'cosine':
        return -(ref_norm_a @ ref_norm_a[idx])
    diff = ref_norm_a - ref_norm_a[idx][None, :]
    return np.einsum('ij,ij->i', diff, diff)


def _generate_furthest_subsets(ref_emb1, n_ensembles, subset_size,
                               distance_metric, use_gpu, device,
                               force_include_first_k=0):
    """Greedy FPS with running min-distance maintenance — no N×N matrix.

    Preserves prior semantics: maintains a global pool of "unused" indices,
    drawing each ensemble's points from the pool. When the pool empties it is
    refilled (excluding indices already chosen in the current ensemble), so
    n_ensembles × subset_size > n_ref still works.

    Paper Section 4: "we include the seed set S in every view". When
    `force_include_first_k > 0`, every ensemble's subset begins with indices
    [0, 1, ..., force_include_first_k-1] (which the caller arranges to be the
    seed rows of ref_emb1), and FPS expands from there to subset_size.

    Per ensemble: subset_size argmax/min-update steps, each O(n_ref · dim).
    Total: O(n_ensembles · subset_size · n_ref · dim) — vs the old
    O(n_ref²) memory + O(n_ref²) per-step indexing.
    """
    n_ref = len(ref_emb1)
    ref, on_gpu = _prep_ref_for_dist(ref_emb1, distance_metric, use_gpu, device)
    subset_indices_list = []

    if on_gpu:
        # Mirror pool state on CPU and GPU so we can branch on emptiness without
        # forcing a GPU→CPU sync each step. After every selection both copies
        # are updated; CPU tracks `n_avail` (avoids `.any()`), GPU mask is used
        # by `torch.where` for the masked argmax.
        pool_mask_gpu = torch.ones(n_ref, dtype=torch.bool, device=device)
        pool_mask_cpu = np.ones(n_ref, dtype=bool)
        n_avail = n_ref
        neg_inf = torch.tensor(float('-inf'), device=device, dtype=torch.float32)

        for _ in range(n_ensembles):
            if n_avail == 0:
                pool_mask_gpu.fill_(True)
                pool_mask_cpu[:] = True
                n_avail = n_ref

            # Force-include the first `force_include_first_k` indices (seed set S
            # at the head of ref_emb1) — paper Section 4: "seeds in every view".
            forced = min(force_include_first_k, subset_size, n_ref)
            if forced > 0:
                selected = list(range(forced))
                pool_mask_gpu[:forced] = False
                pool_mask_cpu[:forced] = False
                n_avail = max(0, n_avail - forced)
                # Initialize min_dist as the min over distances from each forced anchor.
                min_dist = _row_dist_gpu(ref, 0, distance_metric)
                for _f in range(1, forced):
                    min_dist = torch.minimum(min_dist, _row_dist_gpu(ref, _f, distance_metric))
            else:
                # Pick a random starting point from the available pool. Use the
                # CPU mirror to avoid a sync on `pool_mask.nonzero()`.
                avail_np = np.flatnonzero(pool_mask_cpu)
                first = int(np.random.choice(avail_np))
                min_dist = _row_dist_gpu(ref, first, distance_metric)
                selected = [first]
                pool_mask_gpu[first] = False
                pool_mask_cpu[first] = False
                n_avail -= 1

            for _ in range(subset_size - len(selected)):
                if n_avail == 0:
                    # Pool exhausted: refill, then re-exclude already-selected refs.
                    pool_mask_gpu.fill_(True)
                    pool_mask_cpu[:] = True
                    sel_arr = np.asarray(selected, dtype=np.int64)
                    pool_mask_cpu[sel_arr] = False
                    n_avail = n_ref - len(selected)
                    if n_avail <= 0:
                        # subset_size > n_ref: allow a random repeat (no further
                        # FPS info to add — pool is fully consumed).
                        next_idx = int(np.random.randint(n_ref))
                        selected.append(next_idx)
                        new_d = _row_dist_gpu(ref, next_idx, distance_metric)
                        min_dist = torch.minimum(min_dist, new_d)
                        continue
                    sel_t = torch.as_tensor(sel_arr, dtype=torch.long, device=device)
                    pool_mask_gpu[sel_t] = False

                masked = torch.where(pool_mask_gpu, min_dist, neg_inf)
                next_idx = int(masked.argmax().item())
                selected.append(next_idx)
                pool_mask_gpu[next_idx] = False
                pool_mask_cpu[next_idx] = False
                n_avail -= 1
                new_d = _row_dist_gpu(ref, next_idx, distance_metric)
                min_dist = torch.minimum(min_dist, new_d)

            subset_indices_list.append(np.asarray(selected, dtype=np.int64))
    else:
        pool_mask = np.ones(n_ref, dtype=bool)
        for _ in range(n_ensembles):
            if not pool_mask.any():
                pool_mask[:] = True

            # Force-include the first `force_include_first_k` indices (seed set S).
            forced = min(force_include_first_k, subset_size, n_ref)
            if forced > 0:
                selected = list(range(forced))
                pool_mask[:forced] = False
                min_dist = _row_dist_cpu(ref, 0, distance_metric)
                for _f in range(1, forced):
                    new_d = _row_dist_cpu(ref, _f, distance_metric)
                    np.minimum(min_dist, new_d, out=min_dist)
            else:
                avail = np.flatnonzero(pool_mask)
                first = int(np.random.choice(avail))
                min_dist = _row_dist_cpu(ref, first, distance_metric)
                selected = [first]
                pool_mask[first] = False

            for _ in range(subset_size - len(selected)):
                if not pool_mask.any():
                    pool_mask[:] = True
                    pool_mask[np.asarray(selected, dtype=np.int64)] = False
                    if not pool_mask.any():
                        next_idx = int(np.random.randint(n_ref))
                        selected.append(next_idx)
                        new_d = _row_dist_cpu(ref, next_idx, distance_metric)
                        np.minimum(min_dist, new_d, out=min_dist)
                        continue

                masked = np.where(pool_mask, min_dist, -np.inf)
                next_idx = int(np.argmax(masked))
                selected.append(next_idx)
                pool_mask[next_idx] = False
                new_d = _row_dist_cpu(ref, next_idx, distance_metric)
                np.minimum(min_dist, new_d, out=min_dist)

            subset_indices_list.append(np.asarray(selected, dtype=np.int64))

    logger.debug(f"Furthest-point sampling: {len(subset_indices_list)} subsets of size {subset_size} "
                 f"(n_ref={n_ref}, gpu={on_gpu}, metric={distance_metric})")
    return subset_indices_list


def _torch_gpu_cosine_kmeans(ref_normed: np.ndarray,
                             n_clusters: int,
                             top_k: int,
                             n_iter: int = 20,
                             device: str = 'cuda',
                             seed: int = 42,
                             chunk: int = 131072) -> np.ndarray:
    """GPU Lloyd's algorithm for cosine k-means on L2-normalized inputs.

    Equivalent to faiss.Kmeans on normalized vectors (cosine = inner product),
    but runs on GPU via torch so GPU utilization stays high during the partition
    step. faiss-gpu 1.14.1 Kmeans crashes (CUDA error 209) on A100, so we
    bypass it. Uses fp16 for the data tensor + matmul to keep memory bounded;
    centroid accumulation stays in fp32 for numerical stability.

    Returns (n, top_k) int64 array of top-k nearest centroid indices per point.
    """
    n, d = ref_normed.shape
    rng = np.random.RandomState(seed)
    init_idx = rng.choice(n, size=n_clusters, replace=False)

    X = torch.from_numpy(ref_normed).to(device, non_blocking=True).half()
    centroids = torch.from_numpy(ref_normed[init_idx]).to(device).float()

    ones_chunk = torch.ones(chunk, dtype=torch.float32, device=device)

    for _ in range(n_iter):
        new_centroids = torch.zeros((n_clusters, d), dtype=torch.float32, device=device)
        counts = torch.zeros(n_clusters, dtype=torch.float32, device=device)
        c_half = centroids.half().T.contiguous()
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            sims = X[start:end] @ c_half
            assign = sims.argmax(dim=1)
            new_centroids.index_add_(0, assign, X[start:end].float())
            counts.index_add_(0, assign, ones_chunk[:end - start])
        empty_mask = counts == 0
        if bool(empty_mask.any()):
            n_empty = int(empty_mask.sum().item())
            reinit_idx = rng.choice(n, size=n_empty, replace=False)
            new_centroids[empty_mask] = torch.from_numpy(ref_normed[reinit_idx]).to(device).float()
            counts[empty_mask] = 1.0
        new_centroids = new_centroids / counts[:, None]
        new_centroids = new_centroids / (new_centroids.norm(dim=1, keepdim=True) + 1e-8)
        centroids = new_centroids

    out = torch.empty((n, top_k), dtype=torch.long)
    c_half = centroids.half().T.contiguous()
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        sims = X[start:end] @ c_half
        _, topk = sims.topk(top_k, dim=1)
        out[start:end] = topk.cpu()
    del X, centroids, c_half
    torch.cuda.empty_cache()
    return out.numpy()


def generate_ensemble_subsets(ref_emb1, n_ensembles, subset_size, strategy='random', distance_metric='cosine', use_gpu=False, device=None, force_include_first_k=0):
    """
    Generate n_ensembles subsets of reference embeddings based on different strategies.

    Args:
        ref_emb1: Reference embeddings (n_ref, dim)
        n_ensembles: Number of ensembles (subsets) to generate
        subset_size: Size of each subset
        strategy: One of ['random', 'cluster', 'furthest', 'nearest']
        distance_metric: Distance metric for 'furthest' and 'nearest' strategies

    Returns:
        List of n_ensembles arrays, each containing indices of subset members
    """
    n_ref = len(ref_emb1)

    if strategy == 'random':
        # Random sampling: each ensemble randomly samples subset_size points
        subset_indices_list = []
        for _ in range(n_ensembles):
            indices = np.random.choice(n_ref, size=subset_size, replace=False)
            subset_indices_list.append(indices)
        return subset_indices_list

    elif strategy == 'cluster':
        # Cluster-based: cluster into n_ensembles clusters, each cluster is one subset
        ref_normed = (ref_emb1 / (np.linalg.norm(ref_emb1, axis=1, keepdims=True) + 1e-8)).astype(np.float32)
        d = ref_normed.shape[1]
        cluster_overlap_k = 2  # paper ρ=2

        # Prefer torch GPU Lloyd's (keeps GPU busy at scale; faiss.Kmeans(gpu=True)
        # crashes with CUDA 209 on faiss-gpu 1.14.1 + A100, and CPU faiss.Kmeans on
        # ~1M×1500 with hundreds of clusters takes minutes/iter and starves the GPU).
        used_gpu_km = False
        if use_gpu and torch.cuda.is_available():
            try:
                dev = device if (device is not None and str(device).startswith('cuda')) else 'cuda'
                topk_clusters = _torch_gpu_cosine_kmeans(
                    ref_normed, n_ensembles, top_k=cluster_overlap_k,
                    n_iter=20, device=str(dev), seed=42)
                used_gpu_km = True
            except Exception as e:
                logger.warning(f"GPU k-means failed ({e!r}); falling back to CPU faiss")
        if not used_gpu_km:
            import faiss
            kmeans = faiss.Kmeans(d, n_ensembles, niter=20, verbose=False, seed=42)
            kmeans.train(ref_normed)
            _, topk_clusters = kmeans.index.search(ref_normed, cluster_overlap_k)
        # topk_clusters: (n_ref, cluster_overlap_k)

        # Group cluster members (each ref appears in cluster_overlap_k clusters)
        subset_indices_list = []
        cluster_members = {c: [] for c in range(n_ensembles)}
        for ref_idx in range(n_ref):
            for ki in range(cluster_overlap_k):
                c = int(topk_clusters[ref_idx, ki])
                if c >= 0:
                    cluster_members[c].append(ref_idx)
        for cluster_id in range(n_ensembles):
            if len(cluster_members[cluster_id]) > 0:
                subset_indices_list.append(np.array(cluster_members[cluster_id], dtype=np.int64))

        # If some clusters are empty, fill with remaining random samples
        while len(subset_indices_list) < n_ensembles:
            indices = np.random.choice(n_ref, size=subset_size, replace=False)
            subset_indices_list.append(indices)

        logger.debug(f"Cluster strategy: created {len(subset_indices_list)} clusters")
        for i, indices in enumerate(subset_indices_list):
            logger.debug(f"  Cluster {i}: {len(indices)} members")

        return subset_indices_list

    elif strategy == 'furthest':
        # Greedy farthest-point sampling with running min-distance maintenance:
        # avoids materializing the n_ref × n_ref pairwise distance matrix.
        # Cost per ensemble: O(subset_size · n_ref · dim) — one matmul row per pick.
        return _generate_furthest_subsets(
            ref_emb1, n_ensembles, subset_size, distance_metric, use_gpu, device,
            force_include_first_k=force_include_first_k)

    elif strategy == 'nearest':
        # Nearest neighbors: randomly sample seed points, each with its nearest neighbors
        # Each ensemble consists of one seed point and its (subset_size - 1) nearest neighbors
        subset_indices_list = []

        # Compute pairwise distances once (optimized for GPU if available)
        if torch.cuda.is_available() and use_gpu:
            # GPU-accelerated distance computation
            ref_emb_tensor = torch.from_numpy(ref_emb1).float().to(device)

            if distance_metric == 'cosine':
                # Normalize for cosine distance
                ref_emb_normalized = torch.nn.functional.normalize(ref_emb_tensor, p=2, dim=1)
                dist_matrix = 1 - torch.mm(ref_emb_normalized, ref_emb_normalized.T)
            else:
                # Euclidean distance
                dist_matrix = torch.cdist(ref_emb_tensor, ref_emb_tensor, p=2)

            # Keep on GPU for fast indexing
            dist_matrix_gpu = dist_matrix
            use_gpu_indexing = True
        else:
            # CPU fallback
            if distance_metric == 'cosine':
                # Normalize for cosine distance
                ref_emb_normalized = ref_emb1 / (np.linalg.norm(ref_emb1, axis=1, keepdims=True) + 1e-8)
                dist_matrix = 1 - np.dot(ref_emb_normalized, ref_emb_normalized.T)
            else:
                # Euclidean distance
                dist_matrix = euclidean_distances(ref_emb1, ref_emb1)
            use_gpu_indexing = False

        # Randomly sample n_ensembles seed points
        seed_indices = np.random.choice(n_ref, size=n_ensembles, replace=False)

        for seed_idx in seed_indices:
            # Find k nearest neighbors (including the seed point itself)
            if use_gpu_indexing:
                # GPU path
                seed_dists = dist_matrix_gpu[seed_idx]
                # Sort and get top k indices (including self at distance 0)
                _, nearest_indices = torch.topk(seed_dists, k=min(subset_size, n_ref), largest=False)
                nearest_indices = nearest_indices.cpu().numpy()
            else:
                # CPU path
                seed_dists = dist_matrix[seed_idx]
                # Sort and get top k indices (including self at distance 0)
                nearest_indices = np.argsort(seed_dists)[:min(subset_size, n_ref)]

            subset_indices_list.append(nearest_indices)

        logger.debug(f"Nearest neighbors strategy: created {len(subset_indices_list)} subsets of size {subset_size}")
        for i, indices in enumerate(subset_indices_list):
            logger.debug(f"  Ensemble {i}: seed={seed_indices[i]}, {len(indices)} neighbors")

        return subset_indices_list

    else:
        raise ValueError(f"Unknown ensemble strategy: {strategy}")


def run_single_ensemble_gpu(args_tuple):
    """
    GPU-accelerated single ensemble iteration for multiprocessing.

    Supports both random sampling and pre-specified subset indices:
    - If ref_subset_indices is None in args_tuple, randomly sample subset_size points
    - If ref_subset_indices is provided, use those indices directly (for cluster-based selection)
    """
    # Support both old (18-element) and new (19-element with is_normalized) tuple formats
    if len(args_tuple) == 19:
        (ensemble_idx, ref_emb1, ref_emb2, emb1_unique, emb2_unique,
         ind_emb1_unique, ind_emb2_unique, subset_size, args_dict, use_gpu, gpu_id,
         ref_indices1, ref_indices2, ori_ref_emb1, ori_ref_emb2, anchor_mode, concat_seed_pairs,
         ref_subset_indices, is_normalized) = args_tuple
    else:
        # Old format with 18 elements (no is_normalized)
        (ensemble_idx, ref_emb1, ref_emb2, emb1_unique, emb2_unique,
         ind_emb1_unique, ind_emb2_unique, subset_size, args_dict, use_gpu, gpu_id,
         ref_indices1, ref_indices2, ori_ref_emb1, ori_ref_emb2, anchor_mode, concat_seed_pairs,
         ref_subset_indices) = args_tuple
        is_normalized = False  # Default for backward compatibility

    args = argparse.Namespace(**args_dict)

    if use_gpu and torch.cuda.is_available():
        if gpu_id is not None and gpu_id < torch.cuda.device_count():
            device = torch.device(f"cuda:{gpu_id}")
        else:
            device = torch.device("cuda:0")  # Default to GPU 0
    else:
        device = torch.device("cpu")

    # If ref_subset_indices not provided, do random sampling
    if ref_subset_indices is None:
        n_ref = len(ref_emb1)
        ref_subset_indices = np.random.choice(n_ref, size=subset_size, replace=False)

    subset_ref_emb1 = ref_emb1[ref_subset_indices]
    subset_ref_emb2 = ref_emb2[ref_subset_indices]
    
    if concat_seed_pairs and anchor_mode in ("supervised", "ood"):
        subset_ref_emb1 = np.concatenate((ori_ref_emb1, subset_ref_emb1))
        subset_ref_emb2 = np.concatenate((ori_ref_emb2, subset_ref_emb2))

    # Prepare transformation parameters
    transformation = getattr(args, 'transformation', None)
    transformation_params = getattr(args, 'transformation_params', None)
    multi_gpu_config = getattr(args, 'multi_gpu_config', None)

    # Backward compatibility: handle deprecated use_rbf_distance_encoding
    if transformation is None and getattr(args, 'use_rbf_distance_encoding', False):
        transformation = 'rbf'
        rbf_sigma_val = getattr(args, 'rbf_sigma', None)
        if rbf_sigma_val is not None:
            transformation_params = {'sigma': rbf_sigma_val}

    # Compute distance vectors (skip normalization if already pre-normalized for cosine distance)
    dist_vec1_subset = compute_distance_encoding(
        emb=emb1_unique, ref_embeddings=subset_ref_emb1, distance_metric=args.distance_metric,
        use_gpu=use_gpu, device=device, multi_gpu_config=multi_gpu_config,
        transformation=transformation,
        transformation_params=transformation_params,
        is_normalized=is_normalized)
    dist_vec2_subset = compute_distance_encoding(
        emb=emb2_unique, ref_embeddings=subset_ref_emb2, distance_metric=args.distance_metric,
        use_gpu=use_gpu, device=device, multi_gpu_config=multi_gpu_config,
        transformation=transformation,
        transformation_params=transformation_params,
        is_normalized=is_normalized)

    # OPTIMIZATION 1.3: Pre-normalize distance vectors here to avoid redundant normalization
    # in find_mutual_pairs (called n_ensembles times)
    if use_gpu and isinstance(dist_vec1_subset, torch.Tensor):
        dist_vec1_subset = torch.nn.functional.normalize(dist_vec1_subset, p=2, dim=1)
        dist_vec2_subset = torch.nn.functional.normalize(dist_vec2_subset, p=2, dim=1)
    else:
        # CPU path - normalize using numpy
        dist_vec1_subset = dist_vec1_subset / (np.linalg.norm(dist_vec1_subset, axis=1, keepdims=True) + 1e-8)
        dist_vec2_subset = dist_vec2_subset / (np.linalg.norm(dist_vec2_subset, axis=1, keepdims=True) + 1e-8)

    # Find mutual pairs using unified function
    subset_mutual_pairs, mutual_nn, correct = find_mutual_pairs(
        dist_vec1_subset, dist_vec2_subset, ind_emb1_unique, ind_emb2_unique, args, device,
        use_gpu, is_normalized=True)

    subset_accuracy = correct / mutual_nn if mutual_nn > 0 else 0.0

    # Clean up memory after processing
    import gc
    del dist_vec1_subset, dist_vec2_subset, subset_ref_emb1, subset_ref_emb2
    gc.collect()
    if use_gpu and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return ensemble_idx, subset_mutual_pairs, subset_accuracy, mutual_nn


def ensemble_reference_selection_voting(ref_emb1, ref_emb2, emb1_unique, emb2_unique, ind_emb1_unique, ind_emb2_unique,
                                        args, device, ind_nonref, n_ensembles=10, subset_ratio=0.3,
                                        ref_indices1=None, ref_indices2=None,
                                        vote_threshold=0.6, ori_ref_emb1=None, ori_ref_emb2=None, return_vote_matrix=False,
                                        ensemble_strategy='random', skip_adaptive_scaling=False):
    """
    Ensemble-based reference selection method using voting.

    For each ensemble:
    1. Sample/select a subset of points from ref_emb1/ref_emb2 based on strategy
    2. Compute distance vectors using the subset
    3. Find mutual pairs and compute accuracy
    4. Track which mutual pairs appear most frequently across ensembles
    5. Return the final ensemble accuracy/recall

    Args:
        ref_emb1, ref_emb2: Reference embeddings
        emb1_unique, emb2_unique: Unique embeddings to compute distance vectors for
        ind_emb1_unique, ind_emb2_unique: Original indices of unique embeddings
        args: Arguments containing topk, distance_metric, etc.
        device: Device for computations
        n_ensembles: Number of ensemble runs (also number of clusters for 'cluster' strategy)
        subset_ratio: Percentage of reference points to use in each ensemble (for 'random' and 'furthest')
        ensemble_strategy: Strategy for selecting reference subsets:
            - 'random': Random sampling (default)
            - 'cluster': Cluster into n_ensembles clusters, each cluster votes
            - 'furthest': Each subset contains maximally dispersed points
        return_vote_matrix: If True, return vote matrix along with mutual pairs

    Returns:
        mutual_pair: Final mutual pairs selected by ensemble (list of (idx2, idx1, dist))
        vote_matrix (optional): Sparse CSR matrix (n2, n1) if return_vote_matrix=True
    """

    if len(ref_emb1) == 0:
        if return_vote_matrix:
            # Return empty sparse matrix
            n1 = len(emb1_unique)
            n2 = len(emb2_unique)
            return [], csr_matrix((n2, n1), dtype=np.int32)
        else:
            return []

    n_ref = len(ref_emb1)

    use_gpu = device.type == 'cuda' if hasattr(device, 'type') else False

    # For cluster strategy, subset_size is determined by clustering
    # For random and furthest, use subset_ratio
    if ensemble_strategy == 'cluster':
        subset_size = None  # Will be determined by cluster sizes
    else:
        subset_size = max(1, int(n_ref * subset_ratio))

    # Generate subsets based on strategy
    logger.debug(f"Generating {n_ensembles} subsets using '{ensemble_strategy}' strategy")
    if ensemble_strategy == 'cluster':
        # For cluster strategy, we don't need to specify subset_size
        # Each cluster will be its own subset
        subset_indices_list = generate_ensemble_subsets(
            ref_emb1, n_ensembles, subset_size=0, strategy=ensemble_strategy,
            distance_metric=args.distance_metric, 
            use_gpu=use_gpu,
            device=device
        )
    else:
        subset_indices_list = generate_ensemble_subsets(
            ref_emb1, n_ensembles, subset_size, strategy=ensemble_strategy,
            distance_metric=args.distance_metric,
            use_gpu=use_gpu,
            device=device
        )

    # Initialize sparse vote matrix: vote_matrix[i, j] = number of votes for pair (i, j)
    # Use lil_matrix for efficient incremental construction
    n1 = len(emb1_unique)
    n2 = len(emb2_unique)
    vote_matrix = lil_matrix((n2, n1), dtype=np.int32)  # Sparse matrix: (emb2_unique, emb1_unique)

    mutual_pair_dist = dict()
    subset_accuracies = []
    all_subset_pairs = []

    use_gpu = device.type == 'cuda' if hasattr(device, 'type') else False
    enable_parallel = getattr(args, 'enable_parallel_ensemble', True)
    n_gpus = torch.cuda.device_count() if use_gpu else 1

    logger.debug(f"Running ensemble reference selection: {n_ensembles} ensembles using '{ensemble_strategy}' strategy")
    logger.debug(f"Parameters: n_ensembles={n_ensembles}, subset_ratio={subset_ratio:.2f}, vote_threshold={vote_threshold:.2f}")
    logger.debug(f"Sparse vote matrix shape: {vote_matrix.shape} (emb2_unique x emb1_unique)")
    logger.debug(f"Using {'GPU' if use_gpu else 'CPU'} acceleration with {'parallel' if enable_parallel else 'sequential'} execution")

    start_time = time.time()

    # OPTIMIZATION: Pre-normalize embeddings once for cosine distance to avoid redundant normalization
    # in each ensemble worker. This is a major speedup for large n_ensembles.
    is_normalized = False
    if args.distance_metric == 'cosine':
        logger.debug("Pre-normalizing embeddings for cosine distance (avoids redundant normalization in workers)")
        # Normalize unique embeddings
        emb1_unique = emb1_unique / (np.linalg.norm(emb1_unique, axis=1, keepdims=True) + 1e-8)
        emb2_unique = emb2_unique / (np.linalg.norm(emb2_unique, axis=1, keepdims=True) + 1e-8)
        # Normalize reference embeddings
        ref_emb1 = ref_emb1 / (np.linalg.norm(ref_emb1, axis=1, keepdims=True) + 1e-8)
        ref_emb2 = ref_emb2 / (np.linalg.norm(ref_emb2, axis=1, keepdims=True) + 1e-8)
        # Normalize original reference embeddings if present
        if ori_ref_emb1 is not None:
            ori_ref_emb1 = ori_ref_emb1 / (np.linalg.norm(ori_ref_emb1, axis=1, keepdims=True) + 1e-8)
        if ori_ref_emb2 is not None:
            ori_ref_emb2 = ori_ref_emb2 / (np.linalg.norm(ori_ref_emb2, axis=1, keepdims=True) + 1e-8)
        is_normalized = True

    # OPTIMIZATION: Precompute full distance matrices ONCE instead of per-ensemble
    # This is the key optimization: computing distances once and extracting subsets by indexing
    # Complexity reduction: O(n_ensembles * n_unique * n_subset) -> O(n_unique * n_ref)
    use_precomputed_distances = getattr(args, 'use_precomputed_distances', True)  # Enable by default
    full_dist_vec1, full_dist_vec2, ori_dist_vec1, ori_dist_vec2 = None, None, None, None

    if use_precomputed_distances:
        logger.debug("OPTIMIZATION: Using precomputed distance matrices (compute once, index per-ensemble)")
        full_dist_vec1, full_dist_vec2, ori_dist_vec1, ori_dist_vec2 = precompute_full_distance_matrices(
            emb1_unique, emb2_unique, ref_emb1, ref_emb2,
            ori_ref_emb1, ori_ref_emb2, args, device, use_gpu, is_normalized)

    # With precomputed distances, sequential execution is often faster than parallel
    # because the per-ensemble work (column indexing + normalization + find_mutual_pairs)
    # is lightweight compared to multiprocessing overhead
    if use_precomputed_distances and full_dist_vec1 is not None:
        logger.debug("Using sequential execution with precomputed distances (optimal for this case)")
        enable_parallel = False

    if enable_parallel and n_ensembles > 1:
        # Calculate safe max_workers based on memory and user limits
        emb_dim1 = emb1_unique.shape[1] if len(emb1_unique.shape) > 1 else 1
        emb_dim2 = emb2_unique.shape[1] if len(emb2_unique.shape) > 1 else 1
        embedding_dim_avg = (emb_dim1 + emb_dim2) // 2

        # Get the average subset size
        avg_subset_size = int(np.mean([len(s) for s in subset_indices_list]))

        max_parallel_workers = getattr(args, 'max_parallel_workers', None)
        max_workers, should_use_sequential = calculate_safe_max_workers(
            n_ensembles=n_ensembles,
            n_unique_samples=len(emb1_unique),
            n_ref_subset=avg_subset_size,
            embedding_dim_avg=embedding_dim_avg,
            use_gpu=use_gpu,
            device=device,
            safety_factor=0.25,
            max_parallel_workers=max_parallel_workers
        )

        # Force sequential execution if memory too low
        if should_use_sequential:
            logger.warning(f"Memory too low for parallel execution, using sequential mode")
            enable_parallel = False

        # Use multiprocessing with spawn for both CPU and GPU
        try:
            if use_gpu and torch.cuda.is_available():
                # For GPU, use ProcessPoolExecutor with spawn method for true parallelism
                logger.debug(f"Using ProcessPoolExecutor for GPU-based parallel ensembles across {n_gpus} GPU(s)")
                logger.debug(f"max_workers={max_workers} (limited from {n_ensembles} ensembles)")

                # Set spawn method for multiprocessing
                ctx = mp.get_context('spawn')

                args_dict = vars(args)  # Convert args to dictionary for pickling

                # OPTIMIZATION 2.1: Ensure arrays are float32 to avoid dtype conversion overhead
                # in each worker process
                if not isinstance(ref_emb1, torch.Tensor):
                    ref_emb1 = np.asarray(ref_emb1, dtype=np.float32)
                    ref_emb2 = np.asarray(ref_emb2, dtype=np.float32)
                    emb1_unique = np.asarray(emb1_unique, dtype=np.float32)
                    emb2_unique = np.asarray(emb2_unique, dtype=np.float32)

                # Prepare arguments for each ensemble
                ensemble_args = []
                for ensemble_idx in range(n_ensembles):
                    gpu_id = ensemble_idx % n_gpus if n_gpus > 1 else 0
                    # Use pre-generated subset indices
                    ref_subset_indices = subset_indices_list[ensemble_idx]
                    ensemble_args.append((
                        ensemble_idx, ref_emb1, ref_emb2, emb1_unique, emb2_unique,
                        ind_emb1_unique, ind_emb2_unique, len(ref_subset_indices), args_dict, True, gpu_id,
                        ref_indices1, ref_indices2, ori_ref_emb1, ori_ref_emb2, args.anchor_mode, args.concat_seed_pairs,
                        ref_subset_indices, is_normalized  # Pass pre-normalization flag
                    ))

                with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
                    future_to_idx = {
                        executor.submit(run_single_ensemble_gpu, args): args[0]
                        for args in ensemble_args
                    }

                    results_dict = {}
                    for future in as_completed(future_to_idx):
                        ensemble_idx, subset_mutual_pairs, subset_accuracy, mutual_nn = future.result()
                        results_dict[ensemble_idx] = (subset_mutual_pairs, subset_accuracy, mutual_nn)

                    # Process results in order
                    for ensemble_idx in range(n_ensembles):
                        if ensemble_idx in results_dict:
                            subset_mutual_pairs, subset_accuracy, mutual_nn = results_dict[ensemble_idx]

                            # Update vote matrix
                            for i, nearest_i, dist_between_pair in subset_mutual_pairs:
                                vote_matrix[i, nearest_i] += 1
                                pair_key = (i, nearest_i)
                                mutual_pair_dist[pair_key] = dist_between_pair

                            subset_accuracies.append(subset_accuracy)
                            all_subset_pairs.append(subset_mutual_pairs)

                            logger.info(f"Ensemble {ensemble_idx+1}: {mutual_nn} mutual pairs, accuracy: {subset_accuracy:.3f}")

            else:
                # For CPU, use ProcessPoolExecutor with spawn method
                logger.debug("Using ProcessPoolExecutor for CPU-based parallel ensembles")
                logger.debug(f"max_workers={max_workers} (limited from {n_ensembles} ensembles)")

                # Set spawn method for multiprocessing
                ctx = mp.get_context('spawn')

                args_dict = vars(args)  # Convert args to dictionary for pickling

                # Prepare arguments for each ensemble
                ensemble_args = []
                for ensemble_idx in range(n_ensembles):
                    # Use pre-generated subset indices
                    ref_subset_indices = subset_indices_list[ensemble_idx]
                    ensemble_args.append((
                        ensemble_idx, ref_emb1, ref_emb2, emb1_unique, emb2_unique,
                        ind_emb1_unique, ind_emb2_unique, len(ref_subset_indices), args_dict, False, None,
                        ref_indices1, ref_indices2, ori_ref_emb1, ori_ref_emb2, args.anchor_mode, args.concat_seed_pairs,
                        ref_subset_indices, is_normalized  # Pass pre-normalization flag
                    ))

                with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
                    future_to_idx = {
                        executor.submit(run_single_ensemble_gpu, args): args[0]
                        for args in ensemble_args
                    }

                    results_dict = {}
                    for future in as_completed(future_to_idx):
                        ensemble_idx, subset_mutual_pairs, subset_accuracy, mutual_nn = future.result()
                        results_dict[ensemble_idx] = (subset_mutual_pairs, subset_accuracy, mutual_nn)

                    # Process results in order
                    for ensemble_idx in range(n_ensembles):
                        if ensemble_idx in results_dict:
                            subset_mutual_pairs, subset_accuracy, mutual_nn = results_dict[ensemble_idx]

                            # Update vote matrix
                            for i, nearest_i, dist_between_pair in subset_mutual_pairs:
                                vote_matrix[i, nearest_i] += 1
                                pair_key = (i, nearest_i)
                                mutual_pair_dist[pair_key] = dist_between_pair

                            subset_accuracies.append(subset_accuracy)
                            all_subset_pairs.append(subset_mutual_pairs)

                            logger.info(f"Ensemble {ensemble_idx+1}: {mutual_nn} mutual pairs, accuracy: {subset_accuracy:.3f}")

        except Exception as e:
            logger.warning(f"Parallel execution failed: {e}. Falling back to sequential execution.")
            enable_parallel = False

    if not enable_parallel or n_ensembles <= 1:
        # Sequential execution with GPU optimization
        aggressive_memory_clear = getattr(args, 'aggressive_memory_clear', False)
        # Early stopping state
        prev_nnz = 0
        stagnation_count = 0
        early_stop_patience = 3
        early_stop_min_new_fraction = 0.02
        min_ensembles = max(5, n_ensembles // 3)
        actual_ensembles_run = n_ensembles

        for ensemble_idx in range(n_ensembles):
            # Use pre-generated subset indices
            ref_subset_indices = subset_indices_list[ensemble_idx]

            # OPTIMIZATION: Use precomputed distances if available (much faster!)
            if use_precomputed_distances and full_dist_vec1 is not None:
                _, subset_mutual_pairs, subset_accuracy, mutual_nn, _, _ = process_ensemble_with_precomputed_distances(
                    ensemble_idx, full_dist_vec1, full_dist_vec2, ori_dist_vec1, ori_dist_vec2,
                    ind_emb1_unique, ind_emb2_unique, ref_subset_indices, args, device, use_gpu,
                    args.concat_seed_pairs, args.anchor_mode,
                    max_view_pool=getattr(args, 'max_view_pool', None),
                    spatial_tiles=None)  # No tiling in blocked selection path
            else:
                # Fallback to legacy method (computes distances per-ensemble)
                args_tuple = (
                    ensemble_idx, ref_emb1, ref_emb2, emb1_unique, emb2_unique,
                    ind_emb1_unique, ind_emb2_unique, len(ref_subset_indices), vars(args), use_gpu, 0,
                    ref_indices1, ref_indices2, ori_ref_emb1, ori_ref_emb2, args.anchor_mode, args.concat_seed_pairs,
                    ref_subset_indices, is_normalized  # Pass pre-normalization flag
                )
                _, subset_mutual_pairs, subset_accuracy, mutual_nn = run_single_ensemble_gpu(args_tuple)

            # Update vote matrix
            for i, nearest_i, dist_between_pair in subset_mutual_pairs:
                vote_matrix[i, nearest_i] += 1
                pair_key = (i, nearest_i)
                mutual_pair_dist[pair_key] = dist_between_pair

            subset_accuracies.append(subset_accuracy)
            all_subset_pairs.append(subset_mutual_pairs)

            logger.info(f"Ensemble {ensemble_idx+1}: {mutual_nn} mutual pairs, accuracy: {subset_accuracy:.3f}")

            # Aggressive memory clearing for large datasets
            if aggressive_memory_clear:
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Early stopping check
            if ensemble_idx >= min_ensembles - 1:
                current_nnz = vote_matrix.nnz
                new_pairs = current_nnz - prev_nnz
                if current_nnz > 0 and new_pairs / current_nnz < early_stop_min_new_fraction:
                    stagnation_count += 1
                    if stagnation_count >= early_stop_patience:
                        actual_ensembles_run = ensemble_idx + 1
                        logger.info(f"Early stopping at ensemble {actual_ensembles_run}/{n_ensembles}: "
                                   f"last {early_stop_patience} ensembles added <{early_stop_min_new_fraction*100}% new pairs")
                        break
                else:
                    stagnation_count = 0
                prev_nnz = current_nnz
    else:
        actual_ensembles_run = n_ensembles

    elapsed_time = time.time() - start_time
    logger.debug(f"Ensemble computation completed in {elapsed_time:.2f} seconds "
                f"({actual_ensembles_run}/{n_ensembles} ensembles run)")

    # Convert to CSR format for efficient operations
    vote_matrix = vote_matrix.tocsr()

    # Select pairs using vote threshold (scale with actual ensembles run)
    min_votes = max(1, int(actual_ensembles_run * vote_threshold))

    # Count total number of pairs with votes > 0
    total_pairs_with_votes = vote_matrix.nnz
    min_threshold_pairs = max(5, total_pairs_with_votes // 20)

    # Start with threshold, then fall back if needed
    frequent_pairs = []
    final_min_votes = min_votes

    # OPTIMIZED: Convert to COO once, then filter with vectorized operations
    cx = vote_matrix.tocoo()
    coo_rows, coo_cols, coo_data = cx.row, cx.col, cx.data

    for min_votes_loop in range(min_votes, 0, -1):
        # Vectorized filtering instead of Python loop
        mask = coo_data >= min_votes_loop
        n_candidates = mask.sum()

        logger.debug(f"Threshold {min_votes_loop}: Found {n_candidates} pairs with >= {min_votes_loop} votes")

        # Use this threshold if we have enough pairs or if we're at minimum threshold
        if n_candidates >= min_threshold_pairs or min_votes_loop == 1:
            # Build candidate pairs only when we find the right threshold
            filtered_rows = coo_rows[mask]
            filtered_cols = coo_cols[mask]
            filtered_data = coo_data[mask]
            # Sort by vote count (descending)
            sort_idx = np.argsort(-filtered_data)
            candidate_pairs = [((filtered_rows[i], filtered_cols[i]), filtered_data[i]) for i in sort_idx]
            frequent_pairs = candidate_pairs
            final_min_votes = min_votes_loop
            break

    logger.debug(f"Selected {len(frequent_pairs)} pairs using min_votes={final_min_votes} (initial_min_votes={min_votes}, out of {total_pairs_with_votes} total pairs)")

    # Log sparse vote matrix statistics
    if total_pairs_with_votes > 0:
        vote_data = vote_matrix.data
        max_votes = vote_data.max()
        mean_votes = vote_data.mean()
        sparsity = 1 - (total_pairs_with_votes / (n2 * n1))
        logger.debug(f"Sparse vote matrix statistics: non-zero entries={total_pairs_with_votes}/{n2*n1}, sparsity={sparsity:.4f}, max votes={max_votes}, mean votes (non-zero)={mean_votes:.2f}")
    else:
        logger.debug(f"Sparse vote matrix statistics: no pairs received any votes")

    # Extract indices from frequent_pairs: list of ((idx2, idx1), votes)
    pairs_only = [pair for pair, _ in frequent_pairs]
    mutual_pair_dist = [(pair[0], pair[1], mutual_pair_dist[pair]) for pair in pairs_only]
    mutual_pair = deduplicate_pairs(mutual_pair_dist)

    if return_vote_matrix:
        return mutual_pair, vote_matrix
    else:
        return mutual_pair


def ensemble_reference_selection_bernoulli(ref_emb1, ref_emb2, emb1_unique, emb2_unique, ind_emb1_unique, ind_emb2_unique,
                                           args, device, n_ensembles=10, subset_ratio=0.3,
                                           ref_indices1=None, ref_indices2=None, ori_ref_emb1=None, ori_ref_emb2=None,
                                           pair_history=None, posterior_threshold=0.1, ensemble_strategy='random',
                                           posterior_strategy='iteration_based', current_iteration=1, max_iterations=100,
                                           total_ensembles_run=None, overlap_inference_method='threshold',
                                           use_fixed_posterior_threshold=False,
                                           cached_dist_matrices=None,
                                           embeddings_prenormalized=False,
                                           per_view_neighborhoods=False,
                                           nn_search_fn=None,
                                           view_k_neighbors=200,
                                           full_emb1=None,
                                           full_emb2=None):
    """
    Ensemble-based reference selection method using Bernoulli trials with posterior distributions.

    For each ensemble:
    1. Select a subset of points from ref_emb1/ref_emb2 based on strategy (random/cluster/furthest)
    2. Compute distance vectors using the subset
    3. Find mutual pairs and treat as Bernoulli trials (success/failure for each candidate pair)
    4. Update Beta posterior distributions for each pair's success probability
    5. Sample from posterior distributions to select pairs

    Args:
        ref_emb1, ref_emb2: Reference embeddings
        emb1_unique, emb2_unique: Unique embeddings to compute distance vectors for
        ind_emb1_unique, ind_emb2_unique: Original indices of unique embeddings
        args: Arguments containing topk, distance_metric, etc.
        device: Device for computations
        n_ensembles: Number of ensemble runs (also number of clusters for 'cluster' strategy)
        subset_ratio: Percentage of reference points to use in each ensemble (for 'random' and 'furthest')
        ensemble_strategy: Strategy for selecting reference subsets:
            - 'random': Random sampling (default)
            - 'cluster': Cluster into n_ensembles clusters, each cluster votes
            - 'furthest': Each subset contains maximally dispersed points
        pair_history: Dictionary tracking Beta parameters (alpha, beta) for each pair
        posterior_threshold: Base threshold for posterior sampling
        posterior_strategy: Strategy for posterior selection:
            - 'standard': Use fixed posterior_threshold
            - 'iteration_based': Adjust threshold based on iteration progress
        current_iteration: Current iteration number (1-based)
        max_iterations: Maximum number of iterations for normalization
        total_ensembles_run: Total number of ensembles run across all iterations (for intrinsic strategy)
        overlap_inference_method: Method for inferring which pairs are true overlaps:
            - 'threshold': Use fixed/iteration-based threshold (default, original behavior)
            - 'adaptive': Use adaptive inference combining multiple methods
            - 'otsu': Use Otsu's thresholding
            - 'gmm': Use Gaussian Mixture Model
            - 'elbow': Use elbow/knee detection
            - 'expected': Use expected count from posterior sum
            - 'gap': Use gap statistic

    Returns:
        mutual_pair: Final mutual pairs selected by posterior sampling
        pair_history: Updated Beta parameters for next iteration
        posterior_stats: Dictionary with credibility metrics for each selected pair
    """

    if len(ref_emb1) == 0:
        return [], {}, {}, {}, None

    n_ref = len(ref_emb1)

    # Initialize pair history if not provided (first iteration)
    if pair_history is None:
        pair_history = {}
    use_gpu = device.type == 'cuda' if hasattr(device, 'type') else False

    # Per-view: use cluster strategy for spatially diverse views
    # - furthest/nearest are O(n_ref²) — infeasible for large ref sets
    # - random gives overlapping neighborhoods (all refs in same dense region)
    # - cluster: k-means into N groups → each view covers a different spatial region → better total coverage
    if per_view_neighborhoods and n_ref > 5000:
        if ensemble_strategy in ('furthest', 'nearest', 'random'):
            # Target cluster size = max embedding dimension for balanced coverage
            target_cluster_size = max(ref_emb1.shape[1], ref_emb2.shape[1])
            n_ensembles = max(n_ensembles, n_ref // target_cluster_size)
            logger.info(f"Per-view: switching '{ensemble_strategy}' → 'cluster' (n_ref={n_ref:,}, "
                       f"n_clusters={n_ensembles}, ~{n_ref // n_ensembles} refs/cluster)")
            ensemble_strategy = 'cluster'

    # For cluster strategy, subset_size is determined by clustering
    # For random and furthest, use subset_ratio
    if ensemble_strategy == 'cluster':
        subset_size = None  # Will be determined by cluster sizes
    else:
        subset_size = max(5, int(n_ref * subset_ratio))

    # Generate subsets based on strategy
    logger.debug(f"Generating {n_ensembles} subsets using '{ensemble_strategy}' strategy for Bernoulli trials")
    if ensemble_strategy == 'cluster':
        # For cluster strategy, we don't need to specify subset_size
        # Each cluster will be its own subset
        subset_indices_list = generate_ensemble_subsets(
            ref_emb1, n_ensembles, subset_size=0, strategy=ensemble_strategy,
            distance_metric=args.distance_metric,
            use_gpu=use_gpu,
            device=device
        )
        # Calculate average subset size from generated clusters for memory estimation
        subset_size = int(np.mean([len(indices) for indices in subset_indices_list]))
        logger.debug(f"Cluster strategy: average subset size = {subset_size}")
    else:
        # Paper Section 4: "we include the seed set S in every view". The caller
        # arranges the original seeds at the head of ref_emb1 (ori_ref_emb1
        # concatenated first), so forcing FPS to include the first n_seeds rows
        # in every view realises that spec. The internal min() in
        # _generate_furthest_subsets caps to subset_size when n_seeds > s_t.
        _n_seeds_force = len(ori_ref_emb1) if ori_ref_emb1 is not None else 0
        subset_indices_list = generate_ensemble_subsets(
            ref_emb1, n_ensembles, subset_size, strategy=ensemble_strategy,
            distance_metric=args.distance_metric,
            use_gpu=use_gpu,
            device=device,
            force_include_first_k=_n_seeds_force,
        )

    enable_parallel = getattr(args, 'enable_parallel_ensemble', True)
    n_gpus = torch.cuda.device_count() if use_gpu else 1

    logger.debug(f"Running Bernoulli trial ensemble selection: {n_ensembles} ensembles using '{ensemble_strategy}' strategy")
    logger.debug(f"Parameters: n_ensembles={n_ensembles}, subset_ratio={subset_ratio:.2f}, current pairs tracked: {len(pair_history)}")
    logger.debug(f"Using {'GPU' if use_gpu else 'CPU'} acceleration with {'parallel' if enable_parallel else 'sequential'} execution")

    start_time = time.time()

    # OPTIMIZATION: Pre-normalize embeddings once for cosine distance (in-place to save memory)
    is_normalized = embeddings_prenormalized
    if args.distance_metric == 'cosine' and not embeddings_prenormalized:
        if per_view_neighborhoods:
            # Per-view: only normalize refs (pool embeddings are extracted per-view, normalized there)
            logger.debug("Pre-normalizing ref embeddings only (per-view mode)")
            for arr in [ref_emb1, ref_emb2]:
                norms = np.linalg.norm(arr, axis=1, keepdims=True)
                norms[norms < 1e-8] = 1.0
                arr /= norms
        else:
            logger.debug("Pre-normalizing embeddings in-place for cosine distance")
            for arr in [emb1_unique, emb2_unique, ref_emb1, ref_emb2]:
                norms = np.linalg.norm(arr, axis=1, keepdims=True)
                norms[norms < 1e-8] = 1.0
                arr /= norms
        if ori_ref_emb1 is not None:
            norms = np.linalg.norm(ori_ref_emb1, axis=1, keepdims=True); norms[norms < 1e-8] = 1.0; ori_ref_emb1 /= norms
        if ori_ref_emb2 is not None:
            norms = np.linalg.norm(ori_ref_emb2, axis=1, keepdims=True); norms[norms < 1e-8] = 1.0; ori_ref_emb2 /= norms
        is_normalized = True

    # OPTIMIZATION: Precompute full distance matrices ONCE instead of per-ensemble
    # Support incremental extension from cached matrices
    # Skip when per_view_neighborhoods=True (each view computes its own distances)
    use_precomputed_distances = getattr(args, 'use_precomputed_distances', True)
    full_dist_vec1, full_dist_vec2, ori_dist_vec1, ori_dist_vec2 = None, None, None, None

    if per_view_neighborhoods:
        use_precomputed_distances = False
        enable_parallel = False  # Per-view requires sequential (each view has its own neighborhood)
        # Strategy already switched to 'random' above if n_ref > 5000
        logger.info(f"Per-view neighborhoods: skipping global distance precomputation "
                   f"(view_k={view_k_neighbors}, n_ref={len(ref_emb1):,}, nn_search={'available' if nn_search_fn else 'MISSING'})")

    if use_precomputed_distances:
        n_unique_max = max(len(emb1_unique), len(emb2_unique))
        precompute_memory_gb = 2 * n_unique_max * n_ref * 4 / (1024**3)
        n_ori_ref = len(ori_ref_emb1) if ori_ref_emb1 is not None else 0
        if n_ori_ref > 0:
            precompute_memory_gb += 2 * n_unique_max * n_ori_ref * 4 / (1024**3)
        available_ram_gb = get_available_memory_gb(use_gpu=False)

        # For large datasets, use chunked GPU computation and store results on CPU.
        # This is faster than CPU-only computation while avoiding GPU OOM for storage.
        force_cpu_precompute = False
        use_chunked_gpu_precompute = False
        if use_gpu:
            available_gpu_gb = get_available_memory_gb(use_gpu=True, device=device)
            if precompute_memory_gb > available_gpu_gb * 0.3:
                # Matrices too large to store on GPU, but we can still COMPUTE on GPU in chunks
                use_chunked_gpu_precompute = True
                logger.debug(f"Precomputed matrices (~{precompute_memory_gb:.1f} GB) too large for GPU storage ({available_gpu_gb:.1f} GB). Will compute on GPU in chunks, store on CPU.")

        if precompute_memory_gb > available_ram_gb * 0.5:
            logger.warning(
                f"Precomputed distance matrices would require ~{precompute_memory_gb:.1f} GB "
                f"(available RAM: {available_ram_gb:.1f} GB). Disabling precomputation to avoid OOM."
            )
            use_precomputed_distances = False
        else:
            # Try incremental extension from cached matrices
            if cached_dist_matrices is not None:
                prev_dist1, prev_dist2, ori_dist_vec1, ori_dist_vec2, prev_n_ref, prev_fingerprint = cached_dist_matrices
                # Verify prefix: check that the first and mid ref embeddings are unchanged
                prev_first, prev_mid = prev_fingerprint
                prefix_valid = False
                if prev_n_ref <= n_ref and prev_n_ref > 0:
                    prev_mid_idx = prev_n_ref // 2
                    if np.array_equal(ref_emb1[0], prev_first) and np.array_equal(ref_emb1[prev_mid_idx], prev_mid):
                        prefix_valid = True

                precompute_gpu = use_gpu and not force_cpu_precompute
                precompute_device = device if precompute_gpu else torch.device('cpu')

                if prefix_valid and prev_n_ref < n_ref:
                    # Refs grew with same prefix — compute only new columns
                    new_ref_emb1 = ref_emb1[prev_n_ref:]
                    new_ref_emb2 = ref_emb2[prev_n_ref:]
                    logger.debug(f"OPTIMIZATION: Extending cached distance matrices ({prev_n_ref} -> {n_ref} refs, computing {n_ref - prev_n_ref} new columns)")
                    full_dist_vec1, full_dist_vec2 = extend_precomputed_distance_matrices(
                        prev_dist1, prev_dist2, emb1_unique, emb2_unique,
                        new_ref_emb1, new_ref_emb2, args, precompute_device, precompute_gpu, is_normalized)
                elif prefix_valid and prev_n_ref == n_ref:
                    logger.debug("OPTIMIZATION: Reusing cached distance matrices (refs unchanged)")
                    full_dist_vec1 = prev_dist1
                    full_dist_vec2 = prev_dist2
                else:
                    logger.debug(f"Refs changed incompatibly (prev={prev_n_ref}, cur={n_ref}, prefix_valid={prefix_valid}) — recomputing from scratch")
                    cached_dist_matrices = None

            if full_dist_vec1 is None:
                if use_chunked_gpu_precompute:
                    # Compute on GPU in chunks, store results as CPU numpy → frees GPU for MNN
                    logger.debug(f"OPTIMIZATION: Computing full distance matrices via chunked GPU (store on CPU)")
                    chunk_size = 500_000
                    full_dist_vec1 = _compute_distance_encoding_chunked(
                        emb1_unique, ref_emb1, args, device, True, chunk_size=chunk_size)
                    full_dist_vec2 = _compute_distance_encoding_chunked(
                        emb2_unique, ref_emb2, args, device, True, chunk_size=chunk_size)
                    ori_dist_vec1 = None
                    ori_dist_vec2 = None
                    if ori_ref_emb1 is not None and ori_ref_emb2 is not None:
                        ori_dist_vec1 = _compute_distance_encoding_chunked(
                            emb1_unique, ori_ref_emb1, args, device, True, chunk_size=chunk_size)
                        ori_dist_vec2 = _compute_distance_encoding_chunked(
                            emb2_unique, ori_ref_emb2, args, device, True, chunk_size=chunk_size)
                    # Free GPU memory after precomputation
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    precompute_gpu = use_gpu and not force_cpu_precompute
                    precompute_device = device if precompute_gpu else torch.device('cpu')
                    logger.debug(f"OPTIMIZATION: Computing full distance matrices from scratch (on {'GPU' if precompute_gpu else 'CPU'})")
                    full_dist_vec1, full_dist_vec2, ori_dist_vec1, ori_dist_vec2 = precompute_full_distance_matrices(
                        emb1_unique, emb2_unique, ref_emb1, ref_emb2,
                        ori_ref_emb1, ori_ref_emb2, args, precompute_device, precompute_gpu, is_normalized)

        # OPTIMIZATION: Pre-normalize full distance matrices ONCE (row-wise L2)
        # This avoids creating new normalized copies in every per-ensemble call.
        # After this, process_ensemble_with_precomputed_distances can skip normalization.
        if full_dist_vec1 is not None:
            logger.debug("Pre-normalizing full distance matrices (avoids per-ensemble copies)")
            if isinstance(full_dist_vec1, torch.Tensor):
                full_dist_vec1 = torch.nn.functional.normalize(full_dist_vec1, p=2, dim=1)
                full_dist_vec2 = torch.nn.functional.normalize(full_dist_vec2, p=2, dim=1)
                if ori_dist_vec1 is not None:
                    ori_dist_vec1 = torch.nn.functional.normalize(ori_dist_vec1, p=2, dim=1)
                if ori_dist_vec2 is not None:
                    ori_dist_vec2 = torch.nn.functional.normalize(ori_dist_vec2, p=2, dim=1)
            else:
                norms = np.linalg.norm(full_dist_vec1, axis=1, keepdims=True); norms[norms < 1e-8] = 1.0; full_dist_vec1 /= norms
                norms = np.linalg.norm(full_dist_vec2, axis=1, keepdims=True); norms[norms < 1e-8] = 1.0; full_dist_vec2 /= norms
                if ori_dist_vec1 is not None:
                    norms = np.linalg.norm(ori_dist_vec1, axis=1, keepdims=True); norms[norms < 1e-8] = 1.0; ori_dist_vec1 /= norms
                if ori_dist_vec2 is not None:
                    norms = np.linalg.norm(ori_dist_vec2, axis=1, keepdims=True); norms[norms < 1e-8] = 1.0; ori_dist_vec2 /= norms

        # With precomputed distances, sequential is faster than parallel
        logger.debug("Using sequential execution with precomputed distances (optimal for this case)")
        enable_parallel = False

    # Build spatial tiles for tiled MNN if pool is large enough
    spatial_tiles = None
    tile_threshold = getattr(args, 'tile_threshold', 500_000)
    pool_size = max(len(emb1_unique), len(emb2_unique))

    logger.info(f"Tiling check: precomputed={use_precomputed_distances}, dist_available={full_dist_vec1 is not None}, "
                f"pool={pool_size:,}, threshold={tile_threshold:,}")
    if use_precomputed_distances and full_dist_vec1 is not None and pool_size > tile_threshold:
        # Target ~100K per tile after accounting for overlap (each point in overlap_k tiles)
        # With overlap_k=2, effective tile size ≈ pool * overlap_k / n_tiles
        # So n_tiles = pool * overlap_k / target → more tiles for same target
        tile_overlap_k = getattr(args, 'tile_overlap_k', 2)
        target_tile_size = 100_000
        n_tiles = max(3, (pool_size * tile_overlap_k) // target_tile_size)

        # Map ref_indices (global) to pool-local indices
        if ref_indices1 is not None and ref_indices2 is not None:
            g2pool_e1 = {}
            for l, g in enumerate(ind_emb1_unique):
                g2pool_e1[int(g)] = l
            g2pool_e2 = {}
            for l, g in enumerate(ind_emb2_unique):
                g2pool_e2[int(g)] = l

            # Keep only anchor pairs where BOTH sides are in pool (preserves pairing)
            anchor_local_e1_list = []
            anchor_local_e2_list = []
            for g1, g2 in zip(ref_indices1, ref_indices2):
                g1_int, g2_int = int(g1), int(g2)
                if g1_int in g2pool_e1 and g2_int in g2pool_e2:
                    anchor_local_e1_list.append(g2pool_e1[g1_int])
                    anchor_local_e2_list.append(g2pool_e2[g2_int])
            anchor_local_e1 = np.array(anchor_local_e1_list, dtype=np.int64)
            anchor_local_e2 = np.array(anchor_local_e2_list, dtype=np.int64)
            min_anchors = len(anchor_local_e1)

            if min_anchors >= n_tiles * 2:
                spatial_tiles = build_spatial_tiles(
                    emb1_unique, emb2_unique, anchor_local_e1, anchor_local_e2,
                    n_tiles=n_tiles, overlap_k=tile_overlap_k, device=device)
                logger.info(f"Spatial tiling active: {n_tiles} tiles for pool={pool_size:,} "
                           f"(threshold={tile_threshold:,}, overlap_k={tile_overlap_k})")
            else:
                logger.debug(f"Not enough anchors for tiling ({min_anchors} < {n_tiles * 2}), using full-pool MNN")

    # Collect all candidate pairs across ensembles and their distances
    all_candidate_pairs = set()
    pair_distances = {}
    pair_discovery_map = {}  # Maps pair_key -> ensemble_idx that first discovered it
    pair_ensemble_votes = {}  # Maps pair_key -> list of ensemble indices that found this pair
    ensemble_view_indices = {}  # Maps ensemble_idx -> (set(view_idx1), set(view_idx2)) for visibility-aware Bernoulli
    ensemble_mutual_pairs = {}  # Store mutual pairs for reuse in Phase 2 (avoid re-running ensembles)

    _used_single_search = False  # Track if single-search optimization was used

    # Pre-allocate visibility count arrays for per-view mode (updated inline during ensemble loop)
    _pv_e1_vis_count = None
    _pv_e2_vis_count = None
    _pv_found_count = None  # dict: pair_key → found count
    if per_view_neighborhoods:
        max_g1 = int(ind_emb1_unique.max()) + 1 if len(ind_emb1_unique) > 0 else 1
        max_g2 = int(ind_emb2_unique.max()) + 1 if len(ind_emb2_unique) > 0 else 1
        _pv_e1_vis_count = np.zeros(max_g1, dtype=np.int32)
        _pv_e2_vis_count = np.zeros(max_g2, dtype=np.int32)
        _pv_found_count = {}

    # Run ensembles to collect all possible candidate pairs (Phase 1)
    if enable_parallel and n_ensembles > 1 and use_gpu and torch.cuda.is_available():
        # Check memory first before attempting parallel execution
        ctx = mp.get_context('spawn')
        args_dict = vars(args)

        # Calculate embedding dimensions for memory estimation
        emb_dim1 = ref_emb1.shape[1] if len(ref_emb1.shape) > 1 else ref_emb1.shape[0]
        emb_dim2 = ref_emb2.shape[1] if len(ref_emb2.shape) > 1 else ref_emb2.shape[0]
        embedding_dim_avg = (emb_dim1 + emb_dim2) // 2

        # Calculate safe max_workers and check if should use sequential
        max_parallel_workers = getattr(args, 'max_parallel_workers', None)
        max_workers, should_use_sequential = calculate_safe_max_workers(
            n_ensembles=n_ensembles,
            n_unique_samples=len(emb1_unique),
            n_ref_subset=subset_size,
            embedding_dim_avg=embedding_dim_avg,
            use_gpu=use_gpu,
            device=device,
            safety_factor=0.25,
            max_parallel_workers=max_parallel_workers
        )

        # Force sequential execution if memory too low
        if should_use_sequential:
            logger.warning(f"Phase 1: Memory too low for parallel execution, using sequential mode")
            enable_parallel = False
        else:
            # Parallel GPU execution
            logger.debug(f"Phase 1: Collecting candidate pairs in parallel across {n_gpus} GPU(s)")

            ensemble_args = []
            for ensemble_idx in range(n_ensembles):
                gpu_id = ensemble_idx % n_gpus if n_gpus > 1 else 0
                # Use pre-generated subset indices
                ref_subset_indices = subset_indices_list[ensemble_idx]
                ensemble_args.append((
                    ensemble_idx, ref_emb1, ref_emb2, emb1_unique, emb2_unique,
                    ind_emb1_unique, ind_emb2_unique, len(ref_subset_indices), args_dict, True, gpu_id,
                    ref_indices1, ref_indices2, ori_ref_emb1, ori_ref_emb2, args.anchor_mode, args.concat_seed_pairs,
                    ref_subset_indices, is_normalized  # Pass pre-normalization flag
                ))

            # Run in batches if memory-constrained
            results_dict = run_ensembles_in_batches(ensemble_args, max_workers, n_ensembles, ctx)

            # Collect results
            for ensemble_idx in range(n_ensembles):
                if ensemble_idx in results_dict:
                    subset_mutual_pairs, subset_accuracy, mutual_nn = results_dict[ensemble_idx]

                    # Store mutual pairs for reuse in Phase 2 (avoid re-running ensembles)
                    ensemble_mutual_pairs[ensemble_idx] = subset_mutual_pairs

                    # Collect all pairs that appear in this ensemble
                    for i, nearest_i, dist_between_pair in subset_mutual_pairs:
                        pair_key = (i, nearest_i)
                        all_candidate_pairs.add(pair_key)
                        pair_distances[pair_key] = dist_between_pair

                        # Track which ensemble discovered this pair (first discovery)
                        if pair_key not in pair_discovery_map:
                            pair_discovery_map[pair_key] = ensemble_idx

                    logger.info(f"Ensemble {ensemble_idx+1}: {mutual_nn} mutual pairs, accuracy: {subset_accuracy:.3f}")
    else:
        # Sequential per-ensemble MNN with per-view local selection
        _used_single_search = False
        logger.debug("Phase 1: Collecting candidate pairs sequentially (per-view)")
        aggressive_memory_clear = getattr(args, 'aggressive_memory_clear', False)

        # Adaptive view_k: k = N_db / n_ref × s
        # s = expected number of refs jointly covering each matching pair (Poisson model).
        # Higher s → larger neighborhoods, more recall, slower.
        _N_db = len(emb1_unique)
        _s = getattr(args, 'adaptive_s', 5)
        _adaptive_k = min(view_k_neighbors, int(_N_db / max(1, n_ref) * _s))
        if _adaptive_k < view_k_neighbors:
            logger.info(f"Adaptive view_k: {view_k_neighbors} → {_adaptive_k} (n_ref={n_ref:,})")
        view_k_neighbors = _adaptive_k

        # Batch k-NN: concatenate all cluster refs, one GPU call per view, split results back
        _batched_nn_results = None
        if per_view_neighborhoods and nn_search_fn is not None:
            import time as _batch_time
            _batch_t0 = _batch_time.time()
            max_view_refs = 2 * max(ref_emb1.shape[1], ref_emb2.shape[1])

            # Build per-cluster ref indices (with same cap as per-ensemble loop)
            # Track which ref_emb rows each cluster uses (indices into ref_emb, not global)
            _cluster_ref_rows = []
            for _ci in range(n_ensembles):
                _sub = subset_indices_list[_ci]
                if len(_sub) > max_view_refs:
                    _rng = np.random.RandomState(_ci)
                    _sub = _sub[_rng.choice(len(_sub), max_view_refs, replace=False)]
                _cluster_ref_rows.append(_sub)

            # Deduplicate: find unique ref rows across all clusters, search once
            _all_rows = np.concatenate(_cluster_ref_rows)
            _unique_rows, _inv_idx = np.unique(_all_rows, return_inverse=True)
            _dedup_emb1 = ref_emb1[_unique_rows]
            _dedup_emb2 = ref_emb2[_unique_rows]
            logger.info(f"Batched k-NN: {len(_all_rows)} total → {len(_unique_rows)} unique queries "
                       f"({100*(1-len(_unique_rows)/len(_all_rows)):.0f}% dedup) across {n_ensembles} clusters, k={view_k_neighbors}")

            _dedup_nn_e1 = nn_search_fn(_dedup_emb1, 'e1', view_k_neighbors)
            _dedup_nn_e2 = nn_search_fn(_dedup_emb2, 'e2', view_k_neighbors)

            # Expand back to full (with duplicates) then split by cluster
            _full_nn_e1 = _dedup_nn_e1[_inv_idx]
            _full_nn_e2 = _dedup_nn_e2[_inv_idx]
            _batched_nn_results = {'e1': {}, 'e2': {}}
            _offset = 0
            for _ci in range(n_ensembles):
                _sz = len(_cluster_ref_rows[_ci])
                _batched_nn_results['e1'][_ci] = _full_nn_e1[_offset:_offset+_sz]
                _batched_nn_results['e2'][_ci] = _full_nn_e2[_offset:_offset+_sz]
                _offset += _sz

            del _dedup_emb1, _dedup_emb2, _dedup_nn_e1, _dedup_nn_e2, _full_nn_e1, _full_nn_e2
            _batch_elapsed = _batch_time.time() - _batch_t0
            logger.info(f"Batched k-NN done in {_batch_elapsed:.1f}s")

        # --- Full-GPU ensemble processing: zero CPU in hot path ---
        _use_emb1 = full_emb1 if full_emb1 is not None else emb1_unique
        _use_emb2 = full_emb2 if full_emb2 is not None else emb2_unique
        _gpu_mode = isinstance(_use_emb1, torch.Tensor) and use_gpu

        if per_view_neighborhoods and nn_search_fn is not None and _gpu_mode:
            _ens_t0 = time.time()

            # Move shared data to GPU once
            _ref_emb1_g = torch.from_numpy(ref_emb1).float().to(device)
            _ref_emb2_g = torch.from_numpy(ref_emb2).float().to(device)
            _ind1_g = torch.from_numpy(ind_emb1_unique.astype(np.int64)).to(device)
            _ind2_g = torch.from_numpy(ind_emb2_unique.astype(np.int64)).to(device)

            # Convert batched k-NN results to GPU tensors
            _nn_results_g = None
            if _batched_nn_results is not None:
                _nn_results_g = {
                    'e1': {k: torch.from_numpy(v.astype(np.int64)).to(device) for k, v in _batched_nn_results['e1'].items()},
                    'e2': {k: torch.from_numpy(v.astype(np.int64)).to(device) for k, v in _batched_nn_results['e2'].items()},
                }

            # Convert cluster ref rows to GPU
            _cluster_rows_g = None
            if _cluster_ref_rows is not None:
                _cluster_rows_g = [torch.from_numpy(r.astype(np.int64)).to(device) for r in _cluster_ref_rows]

            # Pre-compute per-ensemble data: ref embeddings, pool indices, pool embeddings
            _ens_data = []  # (ref_sub, pi1, pi2, vr1, vr2, pe1, pe2)
            for eidx in range(n_ensembles):
                if _cluster_rows_g is not None:
                    _vr1 = _ref_emb1_g[_cluster_rows_g[eidx]]
                    _vr2 = _ref_emb2_g[_cluster_rows_g[eidx]]
                else:
                    _sub_g = torch.from_numpy(subset_indices_list[eidx].astype(np.int64)).to(device)
                    _vr1 = _ref_emb1_g[_sub_g]
                    _vr2 = _ref_emb2_g[_sub_g]
                    _mvr = 2 * max(_vr1.shape[1], _vr2.shape[1])
                    if len(_vr1) > _mvr:
                        _keep = torch.randperm(len(_vr1), device=device)[:_mvr]
                        _vr1 = _vr1[_keep]; _vr2 = _vr2[_keep]

                if _nn_results_g is not None:
                    _pi1 = torch.unique(_nn_results_g['e1'][eidx].reshape(-1))
                    _pi2 = torch.unique(_nn_results_g['e2'][eidx].reshape(-1))
                else:
                    _nne1 = nn_search_fn(_vr1.cpu().numpy(), 'e1', view_k_neighbors)
                    _nne2 = nn_search_fn(_vr2.cpu().numpy(), 'e2', view_k_neighbors)
                    _pi1 = torch.from_numpy(np.unique(_nne1.ravel()).astype(np.int64)).to(device)
                    _pi2 = torch.from_numpy(np.unique(_nne2.ravel()).astype(np.int64)).to(device)

                _ens_data.append((_pi1, _pi2, _vr1, _vr2))

            _use_fp16 = getattr(args, 'fp16', True)
            _csls_k = getattr(args, 'csls_neighborhood', 0)
            _use_rbf_hash = getattr(args, 'transformation', None) == 'rbf'
            _half_dtype = torch.float16 if _use_fp16 else torch.float32
            _NEG_INF = torch.finfo(_half_dtype).min
            _mnn_bytes = 2 if _use_fp16 else 4  # bytes per element in MNN matmul

            # Decide bmm batch size based on GPU memory
            _pool_sizes = [(len(d[0]), len(d[1])) for d in _ens_data]
            _max_p1 = max(s[0] for s in _pool_sizes)
            _max_p2 = max(s[1] for s in _pool_sizes)
            _max_ref = max(len(d[2]) for d in _ens_data)
            _gpu_free = torch.cuda.mem_get_info(device)[0]
            _edim1 = _use_emb1.shape[1]
            _edim2 = _use_emb2.shape[1]
            # Memory per batch item: sim(p2×p1×bytes) + 2 dv(p×ref×bytes) + pool_emb(p×dim×4) + ref_emb(ref×dim×4)
            _bmm_mem_per = (_max_p2 * _max_p1 * _mnn_bytes
                          + (_max_p1 + _max_p2) * _max_ref * _mnn_bytes
                          + _max_p1 * _edim1 * 4 + _max_p2 * _edim2 * 4
                          + _max_ref * (_edim1 + _edim2) * 4)
            _batch_size = max(1, min(n_ensembles, int(_gpu_free * 0.5 / max(1, _bmm_mem_per))))
            _use_bmm = _batch_size >= 2

            if _use_bmm:
                logger.info(f"Fused bmm: batch={_batch_size}, pool=({_max_p1:,},{_max_p2:,}), "
                           f"ref={_max_ref}, {_bmm_mem_per/1e9:.1f}GB/ens, {_gpu_free/1e9:.1f}GB free")
            else:
                logger.info(f"Sequential GPU: {n_ensembles} ensembles, pool=({_max_p1:,},{_max_p2:,})")
            # Sort ensembles by pool size so similar-sized ones are batched together
            _ens_order = sorted(range(n_ensembles), key=lambda i: len(_ens_data[i][0]) * len(_ens_data[i][1]))
            _ens_gpu_results = [None] * n_ensembles
            _n_bmm = 0
            _n_seq = 0

            # Greedily form batches: accumulate sorted ensembles until memory fills
            _b0 = 0
            while _b0 < n_ensembles:
                _cur_free = torch.cuda.mem_get_info(device)[0] if use_gpu else 0
                _mem_budget = _cur_free * 0.5

                # Find how many ensembles fit in one bmm batch
                _b1 = _b0 + 1
                while _b1 < n_ensembles:
                    _cand = [_ens_data[_ens_order[j]] for j in range(_b0, _b1 + 1)]
                    _cp1 = max(len(d[0]) for d in _cand)
                    _cp2 = max(len(d[1]) for d in _cand)
                    _cref = max(len(d[2]) for d in _cand)
                    # Full per-item peak matching the initial estimate (_bmm_mem_per):
                    # sim(p2×p1) + 2 dv(p×ref) + pool_emb(p×dim×4) + ref_emb(ref×dim×4).
                    # The old estimate counted only the sim matrix and ignored the dv
                    # distance-encode tensors (p×ref); when ref grows large at late
                    # iters those dominate and OOM the 1.0 - bmm distance encode.
                    _cmem = (_cp2 * _cp1 * _mnn_bytes
                             + (_cp1 + _cp2) * _cref * _mnn_bytes
                             + _cp1 * _edim1 * 4 + _cp2 * _edim2 * 4
                             + _cref * (_edim1 + _edim2) * 4) * len(_cand)
                    if _cmem > _mem_budget:
                        break
                    _b1 += 1

                _batch_indices = _ens_order[_b0:_b1]
                _batch_data = [_ens_data[i] for i in _batch_indices]
                _bsize = len(_batch_data)
                _bp1 = max(len(d[0]) for d in _batch_data)
                _bp2 = max(len(d[1]) for d in _batch_data)
                _bref = max(len(d[2]) for d in _batch_data)
                _can_bmm = _bsize >= 2

                if _can_bmm:
                    _n_bmm += _bsize
                else:
                    _n_seq += _bsize

                if _can_bmm:
                    # --- Fused bmm with masking ---
                    _bp1 = max(len(d[0]) for d in _batch_data)
                    _bp2 = max(len(d[1]) for d in _batch_data)
                    _bref = max(len(d[2]) for d in _batch_data)

                    _pool1_batch = torch.zeros(_bsize, _bp1, _edim1, device=device)
                    _pool2_batch = torch.zeros(_bsize, _bp2, _edim2, device=device)
                    _ref1_batch = torch.zeros(_bsize, _bref, _edim1, device=device)
                    _ref2_batch = torch.zeros(_bsize, _bref, _edim2, device=device)
                    _real_p1 = []
                    _real_p2 = []
                    _real_ref = []

                    for i, (_pi1, _pi2, _vr1, _vr2) in enumerate(_batch_data):
                        _p1 = torch.nn.functional.normalize(_use_emb1[_pi1].float(), p=2, dim=1)
                        _p2 = torch.nn.functional.normalize(_use_emb2[_pi2].float(), p=2, dim=1)
                        _r1 = torch.nn.functional.normalize(_vr1.float(), p=2, dim=1)
                        _r2 = torch.nn.functional.normalize(_vr2.float(), p=2, dim=1)
                        _pool1_batch[i, :len(_pi1)] = _p1
                        _pool2_batch[i, :len(_pi2)] = _p2
                        _ref1_batch[i, :len(_vr1)] = _r1
                        _ref2_batch[i, :len(_vr2)] = _r2
                        _real_p1.append(len(_pi1))
                        _real_p2.append(len(_pi2))
                        _real_ref.append(len(_vr1))

                    # Batched distance encoding: cosine distance = 1 - cosine_sim
                    _dv1_batch = 1.0 - torch.bmm(_pool1_batch, _ref1_batch.transpose(1, 2))  # (B, bp1, bref)
                    _dv2_batch = 1.0 - torch.bmm(_pool2_batch, _ref2_batch.transpose(1, 2))  # (B, bp2, bref)
                    del _pool1_batch, _pool2_batch, _ref1_batch, _ref2_batch

                    if _use_rbf_hash:
                        # Paper Section 4 local-isometry RBF hash: exp(-d/sigma), with a
                        # per-view median-heuristic bandwidth sigma_{t,k} over the view's own
                        # nonzero distance-to-anchor entries (matches apply_distance_transformation).
                        # Applied per (view, space); padded ref columns then zeroed so they don't
                        # bias the hash (exp(0)=1 would otherwise pollute the signature/norm).
                        for i in range(_bsize):
                            _rr = _real_ref[i]
                            _rp1 = _real_p1[i]
                            _rp2 = _real_p2[i]
                            _v1 = _dv1_batch[i, :_rp1, :_rr]
                            _s1 = torch.median(_v1[_v1 > 0]) if bool((_v1 > 0).any()) else _v1.new_tensor(1.0)
                            _s1 = torch.clamp(_s1, min=1e-8)
                            _dv1_batch[i, :, :_rr] = torch.exp(-_dv1_batch[i, :, :_rr] / _s1)
                            _v2 = _dv2_batch[i, :_rp2, :_rr]
                            _s2 = torch.median(_v2[_v2 > 0]) if bool((_v2 > 0).any()) else _v2.new_tensor(1.0)
                            _s2 = torch.clamp(_s2, min=1e-8)
                            _dv2_batch[i, :, :_rr] = torch.exp(-_dv2_batch[i, :, :_rr] / _s2)
                            if _rr < _bref:
                                _dv1_batch[i, :, _rr:] = 0
                                _dv2_batch[i, :, _rr:] = 0
                    else:
                        # Mask padded ref columns to 0 before normalize (so they don't affect norm)
                        for i in range(_bsize):
                            _rr = _real_ref[i]
                            if _rr < _bref:
                                _dv1_batch[i, :, _rr:] = 0
                                _dv2_batch[i, :, _rr:] = 0

                    _dv1_batch = torch.nn.functional.normalize(_dv1_batch, p=2, dim=2).to(_half_dtype)
                    _dv2_batch = torch.nn.functional.normalize(_dv2_batch, p=2, dim=2).to(_half_dtype)

                    # Batched MNN: bmm(dv2, dv1.T) → (B, bp2, bp1)
                    _sim_batch = torch.bmm(_dv2_batch, _dv1_batch.transpose(1, 2))
                    del _dv1_batch, _dv2_batch

                    # Mask padded positions to -inf so they never win max
                    for i in range(_bsize):
                        _rp1 = _real_p1[i]
                        _rp2 = _real_p2[i]
                        if _rp1 < _bp1:
                            _sim_batch[i, :, _rp1:] = _NEG_INF  # padded p1 columns
                        if _rp2 < _bp2:
                            _sim_batch[i, _rp2:, :] = _NEG_INF  # padded p2 rows

                    # CSLS correction: subtract mean of top-k similarities per row/col
                    if _csls_k > 0:
                        for i in range(_bsize):
                            _rp1 = _real_p1[i]
                            _rp2 = _real_p2[i]
                            _s_i = _sim_batch[i, :_rp2, :_rp1]
                            _ck = min(_csls_k, _rp1, _rp2)
                            _knn_fwd = _s_i.topk(_ck, dim=1)[0].mean(dim=1)  # (rp2,)
                            _knn_bwd = _s_i.topk(_ck, dim=0)[0].mean(dim=0)  # (rp1,)
                            _sim_batch[i, :_rp2, :_rp1] = _s_i - _knn_fwd.unsqueeze(1) / 2 - _knn_bwd.unsqueeze(0) / 2

                    _fwd_sim_b, _fwd_idx_b = _sim_batch.max(dim=2)  # (B, bp2) → best p1 for each p2
                    _, _rev_idx_b = _sim_batch.max(dim=1)            # (B, bp1) → best p2 for each p1
                    del _sim_batch

                    for i in range(_bsize):
                        _rp1 = _real_p1[i]
                        _rp2 = _real_p2[i]
                        _fwd_idx = _fwd_idx_b[i, :_rp2]
                        _fwd_sim = _fwd_sim_b[i, :_rp2]
                        _rev_idx = _rev_idx_b[i, :_rp1]

                        _is_m = _rev_idx[_fwd_idx] == torch.arange(_rp2, device=device)
                        _mi2 = torch.where(_is_m)[0]
                        _mj1 = _fwd_idx[_mi2]
                        _ms = _fwd_sim[_mi2].float()

                        _pi1, _pi2, _, _ = _batch_data[i]
                        _ens_gpu_results[_batch_indices[i]] = (
                            _pi1.cpu().numpy(), _pi2.cpu().numpy(),
                            _mi2.cpu().numpy(), _mj1.cpu().numpy(), _ms.cpu().numpy())

                    del _fwd_sim_b, _fwd_idx_b, _rev_idx_b

                else:
                    # --- Sequential for large pools or batch_size=1 ---
                    for i, (_pi1, _pi2, _vr1, _vr2) in enumerate(_batch_data):
                        torch.cuda.empty_cache()
                        _pe1 = _use_emb1[_pi1].float()
                        _pe2 = _use_emb2[_pi2].float()

                        _seq_transform = 'rbf' if _use_rbf_hash else None
                        _dv1 = compute_distance_encoding(
                            emb=_pe1, ref_embeddings=_vr1,
                            distance_metric=args.distance_metric,
                            transformation=_seq_transform,
                            use_gpu=True, device=device, is_normalized=is_normalized)
                        del _pe1
                        _dv2 = compute_distance_encoding(
                            emb=_pe2, ref_embeddings=_vr2,
                            distance_metric=args.distance_metric,
                            transformation=_seq_transform,
                            use_gpu=True, device=device, is_normalized=is_normalized)
                        del _pe2

                        # Ensure tensors are on GPU (compute_distance_encoding may fall back to CPU under memory pressure)
                        if _dv1.device.type != 'cuda':
                            _dv1 = _dv1.to(device)
                        if _dv2.device.type != 'cuda':
                            _dv2 = _dv2.to(device)

                        _dv1 = torch.nn.functional.normalize(_dv1, p=2, dim=1).to(_half_dtype).contiguous()
                        _dv2 = torch.nn.functional.normalize(_dv2, p=2, dim=1).to(_half_dtype).contiguous()

                        _n1, _n2 = _dv1.shape[0], _dv2.shape[0]
                        _bytes_per = 2 if _use_fp16 else 4
                        # CSLS needs ~3x sim memory (sim + broadcast temps); without CSLS ~2x (sim + results)
                        _mem_mult = 3 if _csls_k > 0 else 2
                        _seq_gpu_free = torch.cuda.mem_get_info(device)[0] if use_gpu else 0
                        if _n2 * _n1 * _bytes_per * _mem_mult < _seq_gpu_free * 0.5:
                            _sim = torch.mm(_dv2, _dv1.T)
                            if _csls_k > 0:
                                _ck = min(_csls_k, _n1, _n2)
                                _knn_fwd = _sim.topk(_ck, dim=1)[0].mean(dim=1)
                                _knn_bwd = _sim.topk(_ck, dim=0)[0].mean(dim=0)
                                _sim = _sim - _knn_fwd.unsqueeze(1) / 2 - _knn_bwd.unsqueeze(0) / 2
                            _fwd_sim, _fwd_idx = _sim.max(dim=1)
                            _, _rev_idx = _sim.max(dim=0)
                            del _sim
                        else:
                            _fwd_idx = torch.empty(_n2, dtype=torch.long, device=device)
                            _fwd_sim = torch.full((_n2,), _NEG_INF, dtype=_half_dtype, device=device)
                            _rev_idx = torch.empty(_n1, dtype=torch.long, device=device)
                            _rev_sim = torch.full((_n1,), _NEG_INF, dtype=_half_dtype, device=device)

                            if _csls_k > 0:
                                # Two-pass tiled CSLS on GPU:
                                # Pass 1 (lightweight): column top-k for _knn_bwd only
                                # Pass 2: compute row fwd means, apply CSLS, find MNN
                                _tile_free = torch.cuda.mem_get_info(device)[0]
                                _tile = max(256, min(8192, int(_tile_free * 0.3 / max(1, _n1 * _bytes_per))))
                                _ck = min(_csls_k, _n1, _n2)

                                # Pass 1: accumulate per-column top-k across row tiles
                                _col_topk = torch.full((_n1, _ck), _NEG_INF, dtype=_half_dtype, device=device)
                                for _q0 in range(0, _n2, _tile):
                                    _q1 = min(_q0 + _tile, _n2)
                                    _sim = torch.mm(_dv2[_q0:_q1], _dv1.T)  # (tile, _n1)
                                    # Merge: stack tile rows (transposed) with running buffer, keep top-k
                                    # _sim.T is (_n1, tile), _col_topk is (_n1, _ck)
                                    _tile_sz = _q1 - _q0
                                    if _tile_sz >= _ck:
                                        # Tile has enough rows: just topk the tile, then merge with buffer
                                        _tile_topk = _sim.T.topk(_ck, dim=1)[0]  # (_n1, _ck)
                                        _col_topk = torch.cat([_tile_topk, _col_topk], dim=1).topk(_ck, dim=1)[0]
                                        del _tile_topk
                                    else:
                                        _col_topk = torch.cat([_sim.T, _col_topk], dim=1).topk(_ck, dim=1)[0]
                                    del _sim
                                _knn_bwd = _col_topk.float().mean(dim=1)  # (_n1,)
                                del _col_topk

                                # Pass 2: row fwd means + CSLS correction + MNN
                                for _q0 in range(0, _n2, _tile):
                                    _q1 = min(_q0 + _tile, _n2)
                                    _sim = torch.mm(_dv2[_q0:_q1], _dv1.T)
                                    _knn_fwd_tile = _sim.topk(_ck, dim=1)[0].mean(dim=1)
                                    _sim -= _knn_fwd_tile.unsqueeze(1) / 2
                                    _sim -= _knn_bwd.unsqueeze(0).to(_sim.dtype) / 2
                                    del _knn_fwd_tile
                                    _v, _ix = _sim.max(dim=1)
                                    _fwd_sim[_q0:_q1] = _v; _fwd_idx[_q0:_q1] = _ix
                                    _vr, _ir = _sim.max(dim=0)
                                    _b = _vr > _rev_sim
                                    _rev_sim[_b] = _vr[_b]; _rev_idx[_b] = _ir[_b] + _q0
                                    del _sim
                                del _knn_bwd
                            else:
                                # Tiled GPU MNN (no CSLS)
                                _tile_free = torch.cuda.mem_get_info(device)[0]
                                _tile = max(256, min(8192, int(_tile_free * 0.3 / max(1, _n1 * _bytes_per))))
                                for _q0 in range(0, _n2, _tile):
                                    _q1 = min(_q0 + _tile, _n2)
                                    _sim = torch.mm(_dv2[_q0:_q1], _dv1.T)
                                    _v, _ix = _sim.max(dim=1)
                                    _fwd_sim[_q0:_q1] = _v; _fwd_idx[_q0:_q1] = _ix
                                    _vr, _ir = _sim.max(dim=0)
                                    _b = _vr > _rev_sim
                                    _rev_sim[_b] = _vr[_b]; _rev_idx[_b] = _ir[_b] + _q0
                                    del _sim
                                del _dv1, _dv2

                        _is_m = _rev_idx[_fwd_idx] == torch.arange(_n2, device=device)
                        _mi2 = torch.where(_is_m)[0]
                        _mj1 = _fwd_idx[_mi2]
                        _ms = _fwd_sim[_mi2].float()
                        del _fwd_idx, _fwd_sim, _rev_idx, _is_m

                        _ens_gpu_results[_batch_indices[i]] = (
                            _pi1.cpu().numpy(), _pi2.cpu().numpy(),
                            _mi2.cpu().numpy(), _mj1.cpu().numpy(), _ms.cpu().numpy())

                _b0 = _b1

            logger.info(f"Ensemble processing: {_n_bmm} bmm + {_n_seq} sequential")

            # Results already transferred to CPU per-batch. Clean up GPU shared data.
            del _ref_emb1_g, _ref_emb2_g, _nn_results_g, _cluster_rows_g
            torch.cuda.empty_cache()

            # Collect results on CPU
            for eidx in range(n_ensembles):
                _pi1_np, _pi2_np, _mi2_np, _mj1_np, _ms_np = _ens_gpu_results[eidx]

                _vg1 = ind_emb1_unique[_pi1_np]
                _vg2 = ind_emb2_unique[_pi2_np]
                _g_i2 = _vg2[_mi2_np]
                _g_j1 = _vg1[_mj1_np]
                subset_mutual_pairs = list(zip(_g_i2.tolist(), _g_j1.tolist(), _ms_np.tolist()))

                np.add.at(_pv_e1_vis_count, _vg1.astype(np.int64), 1)
                np.add.at(_pv_e2_vis_count, _vg2.astype(np.int64), 1)
                for _gi2, _gj1, _ in subset_mutual_pairs:
                    _pv_found_count[(_gi2, _gj1)] = _pv_found_count.get((_gi2, _gj1), 0) + 1

                mutual_nn = len(subset_mutual_pairs)
                correct = int((_g_i2 == _g_j1).sum())
                subset_accuracy = correct / mutual_nn if mutual_nn > 0 else 0.0

                ensemble_mutual_pairs[eidx] = subset_mutual_pairs
                for i, nearest_i, dist_between_pair in subset_mutual_pairs:
                    pair_key = (i, nearest_i)
                    all_candidate_pairs.add(pair_key)
                    pair_distances[pair_key] = dist_between_pair
                    if pair_key not in pair_discovery_map:
                        pair_discovery_map[pair_key] = eidx

                if eidx < 5 or (eidx + 1) % 50 == 0 or eidx == n_ensembles - 1:
                    logger.info(f"  Ensemble {eidx+1}/{n_ensembles}: refs={len(subset_indices_list[eidx])}, "
                               f"pool=({len(_pi1_np):,},{len(_pi2_np):,}), "
                               f"pairs={mutual_nn:,}, acc={subset_accuracy:.3f}")

            _ens_gpu_results = None
            logger.info(f"Ensemble loop: {n_ensembles} ensembles in {time.time()-_ens_t0:.1f}s (full-GPU)")

        else:
            # Non-per-view paths (precomputed distances or single ensemble GPU)
            for ensemble_idx in range(n_ensembles):
                ref_subset_indices = subset_indices_list[ensemble_idx]
                if use_precomputed_distances and full_dist_vec1 is not None:
                    _, subset_mutual_pairs, subset_accuracy, mutual_nn, _, _ = process_ensemble_with_precomputed_distances(
                        ensemble_idx, full_dist_vec1, full_dist_vec2, ori_dist_vec1, ori_dist_vec2,
                        ind_emb1_unique, ind_emb2_unique, ref_subset_indices, args, device, use_gpu,
                        args.concat_seed_pairs, args.anchor_mode,
                        max_view_pool=None if spatial_tiles is not None else getattr(args, 'max_view_pool', None),
                        spatial_tiles=spatial_tiles)
                else:
                    args_tuple = (
                        ensemble_idx, ref_emb1, ref_emb2, emb1_unique, emb2_unique,
                        ind_emb1_unique, ind_emb2_unique, len(ref_subset_indices), vars(args), use_gpu, 0,
                        ref_indices1, ref_indices2, ori_ref_emb1, ori_ref_emb2, args.anchor_mode, args.concat_seed_pairs,
                        ref_subset_indices, is_normalized
                    )
                    _, subset_mutual_pairs, subset_accuracy, mutual_nn = run_single_ensemble_gpu(args_tuple)

                ensemble_mutual_pairs[ensemble_idx] = subset_mutual_pairs

                for i, nearest_i, dist_between_pair in subset_mutual_pairs:
                    pair_key = (i, nearest_i)
                    all_candidate_pairs.add(pair_key)
                    pair_distances[pair_key] = dist_between_pair
                    if pair_key not in pair_discovery_map:
                        pair_discovery_map[pair_key] = ensemble_idx

                logger.info(f"Ensemble {ensemble_idx+1}: {mutual_nn} mutual pairs, accuracy: {subset_accuracy:.3f}")

                if aggressive_memory_clear:
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    logger.debug(f"Total unique candidate pairs found: {len(all_candidate_pairs)}")

    # Phase 2: Bernoulli posterior computation
    # Fast path: single-search already computed alpha/beta arrays vectorized
    if _used_single_search:
        logger.debug("Phase 2: Using vectorized alpha/beta from single-search (fast path)")
        _pl = _single_search_pair_list
        for idx, pair_key in enumerate(_pl):
            if pair_key not in pair_history:
                pair_history[pair_key] = {
                    'alpha': float(_single_search_alpha[idx]),
                    'beta': float(_single_search_beta[idx])
                }
    else:
        # Slow path: iterate over stored Phase 1 results
        logger.debug("Phase 2: Running Bernoulli trials (vectorized voting)")
        # Vectorized Phase 2: convert pairs to arrays, use set intersection for voting
        pair_list_p2 = list(all_candidate_pairs)
        n_cand = len(pair_list_p2)
        # Per-iteration vote increments. These are *added* to the persistent
        # pair_history below so the Beta-Bernoulli posterior accumulates evidence
        # across all iterations (cumulative), rather than being overwritten each
        # iteration. successes -> alpha, failures -> beta.
        delta_alpha = np.zeros(n_cand, dtype=np.float32)
        delta_beta = np.zeros(n_cand, dtype=np.float32)
        # Build pair → index lookup
        pair_to_idx_p2 = {pk: idx for idx, pk in enumerate(pair_list_p2)}

        if per_view_neighborhoods and _pv_found_count is not None:
            # Visibility-aware voting using pre-accumulated arrays from ensemble loop.
            # No re-iteration needed — just lookup.
            import time as _vt
            _vt0 = _vt.time()

            # 1. Found count: lookup from pre-built dict
            found_count = np.array([_pv_found_count.get(pk, 0) for pk in pair_list_p2], dtype=np.float32)

            # 2. Visibility: lookup from pre-built flat arrays (already accumulated inline)
            _pair_e2 = np.array([pk[0] for pk in pair_list_p2], dtype=np.int64)
            _pair_e1 = np.array([pk[1] for pk in pair_list_p2], dtype=np.int64)
            visible_count = np.minimum(
                _pv_e1_vis_count[_pair_e1],
                _pv_e2_vis_count[_pair_e2]
            ).astype(np.float32)

            # 3. This iteration's votes: successes = found, failures = visible - found
            delta_alpha = found_count
            delta_beta = np.maximum(0, visible_count - found_count)

            _vt1 = _vt.time()
            logger.info(f"Visibility voting: {n_cand:,} cands, {n_ensembles} ens, "
                       f"avg_vis={visible_count.mean():.1f}, avg_found={found_count.mean():.2f}, "
                       f"time={_vt1-_vt0:.1f}s")
        else:
            # Original voting: every ensemble counts as a trial for every pair
            for ensemble_idx in range(n_ensembles):
                if ensemble_idx in ensemble_mutual_pairs:
                    subset_mutual_pairs = ensemble_mutual_pairs[ensemble_idx]
                    found_mask = np.zeros(n_cand, dtype=bool)
                    for i, nearest_i, _ in subset_mutual_pairs:
                        idx = pair_to_idx_p2.get((i, nearest_i))
                        if idx is not None:
                            found_mask[idx] = True
                    delta_alpha[found_mask] += 1.0
                    delta_beta[~found_mask] += 1.0

        # Accumulate this iteration's votes into pair_history (cumulative Beta-Bernoulli):
        # each new pair starts at the Beta(1, 1) prior, then evidence is added every
        # iteration it is a candidate. Pairs already in pair_history carry their priors.
        for idx, pair_key in enumerate(pair_list_p2):
            entry = pair_history.get(pair_key)
            if entry is None:
                entry = {'alpha': 1.0, 'beta': 1.0}
                pair_history[pair_key] = entry
            entry['alpha'] += float(delta_alpha[idx])
            entry['beta'] += float(delta_beta[idx])
        logger.debug(f"Vectorized voting done for {n_cand:,} candidates × {n_ensembles} ensembles")

    # Sample from posterior distributions to select final pairs
    selected_pairs = []
    posterior_stats = {}

    # Vectorized posterior computation — avoids expensive per-pair scipy calls
    n_ph = len(pair_history)
    if n_ph > 10_000:
        logger.debug(f"Computing posteriors vectorized for {n_ph:,} candidates")
        ph_keys = list(pair_history.keys())
        _alpha = np.array([pair_history[k]['alpha'] for k in ph_keys], dtype=np.float32)
        _beta = np.array([pair_history[k]['beta'] for k in ph_keys], dtype=np.float32)
        _pmean = _alpha / (_alpha + _beta)
        _pvar = (_alpha * _beta) / ((_alpha + _beta)**2 * (_alpha + _beta + 1))
        _pstd = np.sqrt(_pvar)
        _ci_low = np.maximum(0, _pmean - 1.96 * _pstd)
        _ci_high = np.minimum(1, _pmean + 1.96 * _pstd)

        min_observations = 3 if ensemble_view_indices else 0
        for idx, pair_key in enumerate(ph_keys):
            n_trials = _alpha[idx] + _beta[idx] - 2
            if n_trials < min_observations:
                continue
            posterior_stats[pair_key] = {
                'posterior_mean': float(_pmean[idx]),
                'posterior_std': float(_pstd[idx]),
                'credible_interval_95': (float(_ci_low[idx]), float(_ci_high[idx])),
                'n_successes': float(_alpha[idx] - 1),
                'n_trials': float(n_trials),
            }
        logger.debug(f"Vectorized posterior computation done ({len(posterior_stats):,} pairs)")
    else:
        # Scalar path for small candidate sets
        min_observations = 3 if ensemble_view_indices else 0
        for pair_key, params in pair_history.items():
            alpha, beta = params['alpha'], params['beta']
            n_trials = alpha + beta - 2
            if n_trials < min_observations:
                continue
            posterior_mean = alpha / (alpha + beta)
            posterior_var = (alpha * beta) / ((alpha + beta)**2 * (alpha + beta + 1))
            posterior_std = np.sqrt(posterior_var)
            credible_interval = beta_dist.interval(0.95, alpha, beta)

            posterior_stats[pair_key] = {
                'posterior_mean': posterior_mean,
            'posterior_std': posterior_std,
            'credible_interval_95': credible_interval,
            'n_successes': alpha - 1,
            'n_trials': alpha + beta - 2,
        }

    _dump_dir = os.environ.get("VECLINK_DUMP_POSTERIORS_DIR")
    if _dump_dir and len(posterior_stats) > 0:
        os.makedirs(_dump_dir, exist_ok=True)
        _tag = os.environ.get("VECLINK_DUMP_TAG", "run")
        _it = int(os.environ.get("VECLINK_DUMP_ITER", "0"))
        _arr = np.fromiter((s['posterior_mean'] for s in posterior_stats.values()),
                           dtype=np.float32, count=len(posterior_stats))
        np.save(f"{_dump_dir}/{_tag}_iter{_it:02d}_posteriors.npy", _arr)

    # Check if using adaptive overlap inference methods
    if overlap_inference_method != 'threshold':
        # Map method names for adaptive inference
        method_map = {
            'adaptive': 'ensemble',
            'otsu': 'otsu',
            'gmm': 'gmm',
            'elbow': 'elbow',
            'expected': 'expected',
            'gap': 'gap'
        }
        inference_method = method_map.get(overlap_inference_method, 'ensemble')

        # Use adaptive inference to select pairs
        selected_pair_keys, adaptive_threshold, method_info = infer_overlap_adaptive(
            posterior_stats,
            method=inference_method,
            fallback_threshold=posterior_threshold,
            min_pairs=1,
            max_pairs_ratio=1.0
        )

        # Build selected_pairs list from selected keys
        for pair_key in selected_pair_keys:
            stats = posterior_stats[pair_key]
            selected_pairs.append((pair_key, stats['posterior_mean'], stats['posterior_std']**2, stats['posterior_mean']))

        logger.debug(f"Adaptive overlap inference ({overlap_inference_method}): selected {len(selected_pairs)} pairs, threshold={adaptive_threshold:.4f}")
        if 'ensemble' in method_info:
            logger.debug(f"  Method estimates: {method_info['ensemble'].get('n_from_methods', {})}")

        # Skip the threshold-based selection below
        effective_threshold = adaptive_threshold

    else:
        # Threshold-based selection
        if use_fixed_posterior_threshold:
            effective_threshold = posterior_threshold
            logger.debug(f"Fixed posterior threshold: {effective_threshold:.6f}")
        else:
            # Iteration-based adaptive threshold
            if total_ensembles_run is not None:
                total_ensembles_with_current = total_ensembles_run
            else:
                total_ensembles_with_current = current_iteration * n_ensembles

            effective_threshold = (2 * current_iteration + 1) / (2 + total_ensembles_with_current)

            logger.debug(f"Iteration-based posterior: iteration {current_iteration}, "
                       f"total_ensembles={total_ensembles_with_current}, "
                       f"effective_threshold={effective_threshold:.6f} (base={posterior_threshold:.3f})")

        # Select pairs based on threshold using posterior mean
        for pair_key, stats in posterior_stats.items():
            if stats['posterior_mean'] > effective_threshold:
                selected_pairs.append((pair_key, stats['posterior_mean'], stats['posterior_std']**2, stats['posterior_mean']))

        # Sort by posterior mean
        selected_pairs.sort(key=lambda x: x[1], reverse=True)

        if use_fixed_posterior_threshold:
            logger.debug(f"Selected {len(selected_pairs)} pairs using fixed posterior threshold {effective_threshold:.6f}")
        else:
            # Limit to reasonable number of pairs (only for adaptive threshold)
            max_pairs = min(len(selected_pairs), int(len(all_candidate_pairs) * 0.5))
            selected_pairs = selected_pairs[:max_pairs]

            logger.debug(f"Selected {len(selected_pairs)} pairs using Bernoulli trial posterior mean (iteration_based strategy)")
            logger.debug(f"Effective threshold: {effective_threshold:.6f} (base: {posterior_threshold}), max pairs: {max_pairs}")

    # Convert back to the expected format
    mutual_pair = []
    final_posterior_stats = {}
    pair_voting_refs = {}  # Track which references voted for each pair

    for (i, nearest_i), post_mean, post_var, _ in selected_pairs:
        if (i, nearest_i) in pair_distances:
            dist = pair_distances[(i, nearest_i)]
            mutual_pair.append((i, nearest_i, dist))
            final_posterior_stats[(i, nearest_i)] = posterior_stats[(i, nearest_i)]

            # Track which reference subsets voted for this pair
            # Get the ensemble indices that voted for this pair
            voting_ensembles = pair_ensemble_votes.get((i, nearest_i), [])

            # Collect all reference indices from the voting ensembles
            voting_ref_indices = set()
            for ensemble_idx in voting_ensembles:
                # Get the reference subset indices used in this ensemble
                ref_subset_indices = subset_indices_list[ensemble_idx]
                voting_ref_indices.update(ref_subset_indices)

            # Store the reference indices that voted
            pair_voting_refs[(i, nearest_i)] = sorted(list(voting_ref_indices))

    # Paper Section 4, p. 6: "we enforce one-to-one matching by greedily selecting
    # non-conflicting promoted pairs in decreasing theta_hat". selected_pairs is
    # already sorted by posterior mean DESC, so mutual_pair preserves that order.
    used_first, used_second = set(), set()
    mutual_pair_greedy = []
    for i, nearest_i, dist in mutual_pair:
        if i in used_first or nearest_i in used_second:
            continue
        used_first.add(i)
        used_second.add(nearest_i)
        mutual_pair_greedy.append((i, nearest_i, dist))
    mutual_pair = mutual_pair_greedy

    elapsed_time = time.time() - start_time
    logger.debug(f"Bernoulli trial ensemble computation completed in {elapsed_time:.2f} seconds")

    # Package cached distance matrices for reuse in next iteration
    # Include fingerprints of ref embeddings to detect when refs changed incompatibly
    dist_cache = None
    if use_precomputed_distances and full_dist_vec1 is not None:
        # Store fingerprints: first ref, mid ref, and last ref (small memory)
        mid_idx = n_ref // 2
        ref_fingerprint = (ref_emb1[0].copy(), ref_emb1[mid_idx].copy())
        dist_cache = (full_dist_vec1, full_dist_vec2, ori_dist_vec1, ori_dist_vec2, n_ref, ref_fingerprint)

    return mutual_pair, pair_history, final_posterior_stats, pair_voting_refs, dist_cache


def _compute_distance_encoding_chunked(emb_unique, ref_sub, args, device, use_gpu, chunk_size=500_000):
    """
    Compute distance encoding in chunks to avoid materializing large matrices on GPU.
    Returns the full distance encoding as a CPU numpy array.

    Optimizations:
    - Precomputes RBF sigma from a sample so all chunks use the same sigma
    - Distributes chunks across multiple GPUs via ThreadPoolExecutor
    """
    n = len(emb_unique)
    distance_metric = getattr(args, 'distance_metric', 'cosine')
    transformation = getattr(args, 'transformation', None)
    transformation_params = getattr(args, 'transformation_params', None)

    # Precompute RBF sigma from a sample so all chunks are consistent
    if transformation == 'rbf' and (transformation_params is None or 'sigma' not in (transformation_params or {})):
        sample_size = min(50_000, n)
        sample_dists = get_dists(emb_unique[:sample_size], ref_sub,
                                  metric=distance_metric, use_gpu=use_gpu, device=device)
        if isinstance(sample_dists, torch.Tensor):
            sample_dists = sample_dists.cpu()
        from utils.graph_util import _safe_median_positive
        sigma = float(_safe_median_positive(sample_dists))
        sigma = max(sigma, 1e-8)
        transformation_params = {"sigma": sigma}
        logger.debug(f"  Precomputed RBF sigma={sigma:.6f} from {sample_size:,} sample")
        del sample_dists

    if n <= chunk_size:
        dist_vec = compute_distance_encoding(
            emb_unique, ref_embeddings=ref_sub,
            distance_metric=distance_metric,
            transformation=transformation,
            transformation_params=transformation_params,
            use_gpu=use_gpu, device=device
        )
        if isinstance(dist_vec, torch.Tensor):
            dist_vec = dist_vec.cpu().numpy()
        return dist_vec

    n_ref = len(ref_sub)
    result = np.empty((n, n_ref), dtype=np.float32)
    n_gpus = torch.cuda.device_count() if use_gpu else 0

    chunks = [(start, min(start + chunk_size, n)) for start in range(0, n, chunk_size)]

    def process_chunk_on_device(start, end, chunk_device):
        chunk = emb_unique[start:end]
        dist_chunk = compute_distance_encoding(
            chunk, ref_embeddings=ref_sub,
            distance_metric=distance_metric,
            transformation=transformation,
            transformation_params=transformation_params,
            use_gpu=use_gpu, device=chunk_device
        )
        if isinstance(dist_chunk, torch.Tensor):
            dist_chunk = dist_chunk.cpu().numpy()
        result[start:end] = dist_chunk
        if use_gpu:
            torch.cuda.empty_cache()

    if n_gpus > 1:
        from concurrent.futures import ThreadPoolExecutor

        chunk_tasks = [(s, e, torch.device(f'cuda:{i % n_gpus}'))
                       for i, (s, e) in enumerate(chunks)]
        # Process n_gpus chunks at a time (one per GPU)
        for i in range(0, len(chunk_tasks), n_gpus):
            batch = chunk_tasks[i:i + n_gpus]
            with ThreadPoolExecutor(max_workers=len(batch)) as executor:
                futures = [executor.submit(process_chunk_on_device, s, e, d)
                           for s, e, d in batch]
                for f in futures:
                    f.result()  # wait and propagate exceptions
    else:
        for start, end in chunks:
            process_chunk_on_device(start, end, device)

    return result

