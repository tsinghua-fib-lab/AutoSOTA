import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='pkg_resources')

import os
import sys
import faulthandler
import traceback
import gc
import time
import argparse
import random
from typing import Tuple

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from loguru import logger

from utils.load_data import load_npy, DataPartitioner, save_npy, load_beir_qrels, BEIR_AVAILABLE, convert_global_to_local_indices
from utils.clustering import Clusterer
from graph_utils.cluster import Cluster
from utils.graph_util import get_dists
from utils.retrieval_util import compute_accuracy_recall, analyze_distance_based_accuracy
from utils.sample_methods import sample_ref_points
from utils.ensemble_selection import (
    ensemble_reference_selection_voting,
    ensemble_reference_selection_bernoulli,
)
from utils.procrustes_util import cluster_wise_procrustes_refinement


def optimize_for_large_datasets(args):
    """
    Optimize execution strategy for large datasets to prevent CUDA OOM while maintaining speed.
    
    Only changes HOW ensembles are executed (parallelism, memory clearing), 
    NOT the algorithm parameters (n_ensembles, subset_ratio, etc.).
    """
    large_datasets = {"scidocs", "fiqa", "fever"}
    dataset_name = getattr(args, "dataset", None)
    ref_dataset_name = getattr(args, "ref_dataset", None)

    if dataset_name in large_datasets or ref_dataset_name in large_datasets:
        logger.debug(f"Large dataset detected ({dataset_name or ref_dataset_name}); optimizing execution strategy")
        
        # Enable aggressive memory clearing between ensembles
        args.aggressive_memory_clear = True
        logger.debug("  - Enabled aggressive GPU memory clearing between ensembles")

        # Cap max_iter for large datasets if user hasn't explicitly set a low value
        # Only cap if the main dataset (not ref_dataset) is large
        if dataset_name in large_datasets and getattr(args, 'max_iter', 100) >= 100:
            args.max_iter = 30
            logger.debug(f"  - Capped max_iter to {args.max_iter} for large dataset")


def auto_tune_for_scale(n_unique1, n_unique2, args):
    """
    Auto-tune parameters based on actual dataset size to prevent infeasible computations.
    CSLS requires materializing a full (N_unique x N_unique) distance matrix in get_topk(),
    which is O(N^2) in memory. For large datasets this is infeasible (e.g., 3M x 3M = 36 TB).
    When N exceeds a threshold, we disable CSLS and rely on FAISS approximate NN instead.
    """
    n_max = max(n_unique1, n_unique2)

    if n_max > 50_000 and getattr(args, 'csls_neighborhood', 0) > 0:
        logger.warning(
            f"Auto-disabling CSLS (N_unique={n_max:,}). "
            f"CSLS requires O(N^2) memory which is infeasible at this scale. "
            f"Using FAISS approximate NN instead."
        )
        args.csls_neighborhood = 0

    if n_max > 500_000:
        if not getattr(args, 'aggressive_memory_clear', False):
            args.aggressive_memory_clear = True
            logger.debug(f"Enabled aggressive memory clearing for large dataset (N_unique={n_max:,})")

    # Auto-set max_refs based on available RAM if not user-specified
    large_threshold = getattr(args, 'large_dataset_threshold', 500_000)
    if n_max > large_threshold and getattr(args, 'max_refs', None) is None:
        from utils.memory_util import compute_max_refs
        auto_max_refs = compute_max_refs(n_max)
        # Clamp to a reasonable range
        auto_max_refs = min(auto_max_refs, 5000)
        args.max_refs = auto_max_refs
        logger.warning(
            f"Auto-setting max_refs={auto_max_refs} for large dataset (N_unique={n_max:,}). "
            f"Override with --max_refs if needed."
        )


def run_subsampled_warmstart(emb1_unique, emb2_unique, ind_emb1_unique, ind_emb2_unique,
                             ori_ref_emb1, ori_ref_emb2, ind_nonref, args, device,
                             subsample_size=200_000, n_iters=3):
    """
    Run VecLink on a subsample to warm-start the reference set for large datasets.

    Takes a random subsample of the corpus, runs a few iterations of ensemble matching,
    and returns discovered matches mapped back to full corpus indices.

    Args:
        emb1_unique, emb2_unique: Full unique embeddings
        ind_emb1_unique, ind_emb2_unique: Global indices of unique embeddings
        ori_ref_emb1, ori_ref_emb2: Original seed reference embeddings
        ind_nonref: Overlapping indices
        args: Arguments
        device: Computation device
        subsample_size: Number of points to subsample
        n_iters: Number of warm-start iterations

    Returns:
        warmstart_ref_indices1, warmstart_ref_indices2: Discovered match indices (global)
    """
    n1 = len(emb1_unique)
    n2 = len(emb2_unique)

    # Subsample: take the same random indices from both sets to maintain correspondence
    actual_sub_size = min(subsample_size, n1, n2)
    sub_idx1 = np.sort(np.random.choice(n1, actual_sub_size, replace=False))
    sub_idx2 = np.sort(np.random.choice(n2, actual_sub_size, replace=False))

    sub_emb1 = emb1_unique[sub_idx1]
    sub_emb2 = emb2_unique[sub_idx2]
    sub_ind1 = ind_emb1_unique[sub_idx1]
    sub_ind2 = ind_emb2_unique[sub_idx2]

    logger.debug(f"Warm-start: subsampled {actual_sub_size:,} points from each set")

    use_gpu = args.use_gpu and torch.cuda.is_available()

    # Normalize if using cosine
    if getattr(args, 'distance_metric', 'cosine') == 'cosine':
        sub_emb1_norm = sub_emb1 / (np.linalg.norm(sub_emb1, axis=1, keepdims=True) + 1e-8)
        sub_emb2_norm = sub_emb2 / (np.linalg.norm(sub_emb2, axis=1, keepdims=True) + 1e-8)
        ref1_norm = ori_ref_emb1 / (np.linalg.norm(ori_ref_emb1, axis=1, keepdims=True) + 1e-8)
        ref2_norm = ori_ref_emb2 / (np.linalg.norm(ori_ref_emb2, axis=1, keepdims=True) + 1e-8)
    else:
        sub_emb1_norm = sub_emb1
        sub_emb2_norm = sub_emb2
        ref1_norm = ori_ref_emb1
        ref2_norm = ori_ref_emb2

    ref_emb1 = ref1_norm
    ref_emb2 = ref2_norm

    all_ref_indices1 = np.array([], dtype=np.int64)
    all_ref_indices2 = np.array([], dtype=np.int64)

    # Build global→local map once; reused across all warm-start iterations.
    n_total = max(int(ind_emb1_unique.max()), int(ind_emb2_unique.max())) + 1
    g2l1 = np.full(n_total, -1, dtype=np.int64)
    g2l1[ind_emb1_unique] = np.arange(len(ind_emb1_unique))
    g2l2 = np.full(n_total, -1, dtype=np.int64)
    g2l2[ind_emb2_unique] = np.arange(len(ind_emb2_unique))

    for it in range(n_iters):
        # Use voting-based ensemble on the subsample
        mutual_pairs = ensemble_reference_selection_voting(
            ref_emb1, ref_emb2, sub_emb1_norm, sub_emb2_norm, sub_ind1, sub_ind2,
            args, device, ind_nonref, vote_threshold=args.ensemble_vote_threshold,
            n_ensembles=args.ensemble_n_ensembles, subset_ratio=args.ensemble_subset_ratio,
            ref_indices1=all_ref_indices1 if len(all_ref_indices1) > 0 else None,
            ref_indices2=all_ref_indices2 if len(all_ref_indices2) > 0 else None,
            ori_ref_emb1=ori_ref_emb1, ori_ref_emb2=ori_ref_emb2,
            ensemble_strategy=args.ensemble_strategy
        )

        if len(mutual_pairs) == 0:
            logger.debug(f"Warm-start iteration {it+1}: no mutual pairs found, stopping")
            break

        # Extract discovered indices (these are global indices via sub_ind1/sub_ind2)
        new_ref1 = np.array([sub_ind1[j] for _, j, _ in mutual_pairs])
        new_ref2 = np.array([sub_ind2[i] for i, _, _ in mutual_pairs])

        # Merge with existing
        if len(all_ref_indices1) > 0:
            all_ref_indices1 = np.unique(np.concatenate([all_ref_indices1, new_ref1]))
            all_ref_indices2 = np.unique(np.concatenate([all_ref_indices2, new_ref2]))
        else:
            all_ref_indices1 = np.unique(new_ref1)
            all_ref_indices2 = np.unique(new_ref2)

        logger.debug(f"Warm-start iteration {it+1}: found {len(mutual_pairs)} pairs, "
                     f"total refs: {len(all_ref_indices1)}")

        # Update reference embeddings for next iteration (using the full embeddings).
        # Map global → local emb1_unique/emb2_unique indices via the precomputed g2l.
        local1 = g2l1[all_ref_indices1]
        local2 = g2l2[all_ref_indices2]
        discovered_local1 = local1[local1 >= 0]
        discovered_local2 = local2[local2 >= 0]

        if len(discovered_local1) > 0:
            disc_emb1 = emb1_unique[discovered_local1]
            disc_emb2 = emb2_unique[discovered_local2]
            if getattr(args, 'distance_metric', 'cosine') == 'cosine':
                disc_emb1 = disc_emb1 / (np.linalg.norm(disc_emb1, axis=1, keepdims=True) + 1e-8)
                disc_emb2 = disc_emb2 / (np.linalg.norm(disc_emb2, axis=1, keepdims=True) + 1e-8)
            ref_emb1 = np.concatenate([ref1_norm, disc_emb1])
            ref_emb2 = np.concatenate([ref2_norm, disc_emb2])

            # Cap refs for warmstart too
            max_refs = getattr(args, 'max_refs', None)
            min_len = min(len(ref_emb1), len(ref_emb2))
            if max_refs is not None and min_len > max_refs:
                keep = np.random.choice(min_len, max_refs, replace=False)
                ref_emb1 = ref_emb1[keep]
                ref_emb2 = ref_emb2[keep]
            elif len(ref_emb1) != len(ref_emb2):
                # Truncate to matching length
                ref_emb1 = ref_emb1[:min_len]
                ref_emb2 = ref_emb2[:min_len]

        if use_gpu:
            torch.cuda.empty_cache()

    logger.debug(f"Warm-start complete: discovered {len(all_ref_indices1)} reference pairs")
    return all_ref_indices1, all_ref_indices2


def compute_annealed_ref_filter_ratio(iteration, max_iter, initial_ratio, final_ratio,
                                    annealing_type="linear", quality_history=None):
    """
    Compute annealed ref_filter_ratio based on iteration progress and quality metrics.
    
    Args:
        iteration: Current iteration (1-based)
        max_iter: Maximum number of iterations
        initial_ratio: Starting ref_filter_ratio
        final_ratio: Final ref_filter_ratio  
        annealing_type: Type of annealing ("linear", "exponential", "cosine", "quality_adaptive")
        quality_history: List of quality metrics (mean_quality, kept_quality) for adaptive annealing
        
    Returns:
        float: Annealed ref_filter_ratio
    """
    if annealing_type == "none" or max_iter <= 1:
        return initial_ratio
    
    # Normalize iteration progress to [0, 1]
    progress = min((iteration - 1) / (max_iter - 1), 1.0)
    
    if annealing_type == "linear":
        # Linear interpolation from initial to final
        ratio = initial_ratio + (final_ratio - initial_ratio) * progress
        
    elif annealing_type == "exponential":
        # Exponential decay: initial_ratio * (final_ratio/initial_ratio)^progress
        if final_ratio > 0 and initial_ratio > 0:
            ratio = initial_ratio * (final_ratio / initial_ratio) ** progress
        else:
            ratio = initial_ratio * (1 - progress) + final_ratio * progress
            
    elif annealing_type == "cosine":
        # Cosine annealing for smooth transitions - inverted to increase over time
        cosine_progress = 0.5 * (1 - np.cos(np.pi * progress))
        ratio = initial_ratio + (final_ratio - initial_ratio) * cosine_progress
        
    elif annealing_type == "quality_adaptive":
        # Adaptive annealing based on pairwise distance quality improvement
        if quality_history is None or len(quality_history) < 3:
            # Fall back to linear if not enough history
            ratio = initial_ratio + (final_ratio - initial_ratio) * progress
        else:
            # Check quality trends in recent iterations
            recent_qualities = quality_history[-3:]
            mean_qualities = [q[0] for q in recent_qualities]  # mean_quality
            kept_qualities = [q[1] for q in recent_qualities]  # kept_quality
            
            # Check if mean quality is improving
            quality_improving = len(mean_qualities) >= 2 and mean_qualities[-1] > mean_qualities[0]
            
            # Check if kept quality is above a threshold (good references available)
            high_quality = kept_qualities[-1] > 0.5 if kept_qualities else False
            
            if quality_improving and high_quality:
                # Quality is good and improving, be less aggressive with filtering
                ratio = initial_ratio + (final_ratio - initial_ratio) * (progress * 0.3)
            elif quality_improving:
                # Quality improving but not high, moderate filtering
                ratio = initial_ratio + (final_ratio - initial_ratio) * (progress * 0.7)
            elif high_quality:
                # High quality but not improving, standard annealing
                ratio = initial_ratio + (final_ratio - initial_ratio) * progress
            else:
                # Low quality and not improving, be more aggressive with filtering
                ratio = initial_ratio + (final_ratio - initial_ratio) * min(progress * 1.5, 1.0)
                
    else:
        raise ValueError(f"Unknown annealing type: {annealing_type}")
    
    # Ensure ratio stays within reasonable bounds
    ratio = max(0.1, min(1.0, ratio))
    return ratio

def filter_references_by_pairwise_distance_quality(ref_indices1, ref_indices2, emb1, emb2,
                                                   distance_metric="cosine", top_k_ratio=0.7, device=None, return_metrics=False,
                                                   previous_mutual_pairs=None, ind_emb1_unique=None, ind_emb2_unique=None,
                                                   use_multi_gpu=False, gpu_ids=None, multi_gpu_config=None,
                                                   cached_dist_matrices=None,
                                                   emb1_g2l=None, emb2_g2l=None):
    """
    Filter reference pairs based on correlation quality and mutual nearest neighbor contribution.
    Uses only reference-to-reference distances (no data leakage) and prioritizes references
    that have good distance correlation and help find more mutual pairs.

    Args:
        ref_indices1, ref_indices2: Current reference indices
        emb1, emb2: Embedding matrices (compact unique subsets when g2l mappings provided)
        distance_metric: Distance metric to use
        top_k_ratio: Keep top fraction of pairs with best combined score (0.7 = keep top 70%)
        device: Computing device
        return_metrics: If True, return quality metrics along with filtered indices
        previous_mutual_pairs: Previous iteration's mutual pairs for contribution tracking
        ind_emb1_unique, ind_emb2_unique: Unique indices for tracking mutual pair contributions
        use_multi_gpu: Whether to use multiple GPUs
        gpu_ids: List of GPU IDs to use
        multi_gpu_config: Optional dict configuring multi-GPU chunking for get_dists
        emb1_g2l, emb2_g2l: Global→local index mappings. When provided, emb1/emb2 are
            compact arrays and ref_indices are mapped via g2l before indexing.

    Returns:
        filtered_ref_indices1, filtered_ref_indices2: Filtered reference indices
        If return_metrics=True, also returns: (mean_quality, kept_quality, min_quality, max_quality)
    """
    # Ensure indices are NumPy arrays for advanced indexing
    ref_indices1 = np.asarray(ref_indices1, dtype=np.int32)
    ref_indices2 = np.asarray(ref_indices2, dtype=np.int32)

    if len(ref_indices1) < 5:  # Need minimum references for meaningful comparison
        if return_metrics:
            return ref_indices1, ref_indices2, (0.0, 0.0, 0.0, 0.0)
        return ref_indices1, ref_indices2

    use_gpu = device is not None and device.type == 'cuda'
    n_refs = len(ref_indices1)

    # Get reference embeddings only - no data leakage from non-reference points
    if emb1_g2l is not None:
        ref_emb1 = emb1[emb1_g2l[ref_indices1]]
        ref_emb2 = emb2[emb2_g2l[ref_indices2]]
    else:
        ref_emb1 = emb1[ref_indices1]
        ref_emb2 = emb2[ref_indices2]

    # Check if we can reuse cached distance matrices
    current_hash = hash((tuple(ref_indices1.tolist()), tuple(ref_indices2.tolist())))
    dist_matrix1 = None
    dist_matrix2 = None
    cache_hit = False

    if cached_dist_matrices is not None:
        prev_hash, cached_dist1, cached_dist2 = cached_dist_matrices
        if current_hash == prev_hash:
            # Reuse cached matrices - reference indices haven't changed
            logger.debug("Reusing cached reference distance matrices (no reference changes detected)")
            dist_matrix1 = cached_dist1
            dist_matrix2 = cached_dist2
            cache_hit = True

    # Configure multi-GPU settings for this call if not provided
    if multi_gpu_config is None and use_multi_gpu and gpu_ids:
        multi_gpu_config = {
            "enabled": True,
            "gpu_ids": gpu_ids
        }
    multi_gpu_enabled = bool(multi_gpu_config and multi_gpu_config.get("enabled") and multi_gpu_config.get("gpu_ids"))

    # Compute distance matrices only if not cached
    if not cache_hit:
        if multi_gpu_enabled:
            dist_matrix1 = get_dists(
                ref_emb1,
                ref_emb1,
                metric=distance_metric,
                use_gpu=use_gpu,
                device=device,
                multi_gpu_config=multi_gpu_config
            )
            dist_matrix2 = get_dists(
                ref_emb2,
                ref_emb2,
                metric=distance_metric,
                use_gpu=use_gpu,
                device=device,
                multi_gpu_config=multi_gpu_config
            )
        else:
            if use_gpu:
                ref_emb1_t = torch.tensor(ref_emb1, device=device, dtype=torch.float32)
                ref_emb2_t = torch.tensor(ref_emb2, device=device, dtype=torch.float32)
                dist_matrix1 = get_dists(ref_emb1_t, ref_emb1_t, metric=distance_metric, use_gpu=True, device=device)
                dist_matrix2 = get_dists(ref_emb2_t, ref_emb2_t, metric=distance_metric, use_gpu=True, device=device)
            else:
                dist_matrix1 = get_dists(ref_emb1, ref_emb1, metric=distance_metric, use_gpu=False)
                dist_matrix2 = get_dists(ref_emb2, ref_emb2, metric=distance_metric, use_gpu=False)
    
    # Score each reference based on correlation + mutual NN contribution
    # Use GPU-resident vectorized correlation computation for efficiency

    # Criterion 1: Distance vector correlation between embeddings (VECTORIZED ON GPU)
    if use_gpu and isinstance(dist_matrix1, torch.Tensor):
        # Keep distance matrices on GPU for vectorized correlation
        # dist_matrix1, dist_matrix2: (n_refs, n_refs) on GPU

        # Create mask to exclude diagonal (self-distances)
        mask = ~torch.eye(n_refs, dtype=torch.bool, device=device)

        # Apply mask to get filtered distance vectors for all references at once
        dist1_filtered = dist_matrix1[:, mask[0]].reshape(n_refs, n_refs - 1)
        dist2_filtered = dist_matrix2[:, mask[0]].reshape(n_refs, n_refs - 1)

        # Vectorized standardization (z-score normalization)
        mean1 = dist1_filtered.mean(dim=1, keepdim=True)
        mean2 = dist2_filtered.mean(dim=1, keepdim=True)
        std1 = dist1_filtered.std(dim=1, keepdim=True) + 1e-8
        std2 = dist2_filtered.std(dim=1, keepdim=True) + 1e-8

        dist1_standardized = (dist1_filtered - mean1) / std1
        dist2_standardized = (dist2_filtered - mean2) / std2

        # Vectorized correlation: Pearson correlation coefficient
        correlations = (dist1_standardized * dist2_standardized).mean(dim=1)
        correlations = torch.nan_to_num(correlations, nan=0.0)
        correlations_np = correlations.cpu().numpy()
    else:
        # CPU fallback: use vectorized numpy operations
        if isinstance(dist_matrix1, torch.Tensor):
            dist_matrix1 = dist_matrix1.cpu().numpy()
        if isinstance(dist_matrix2, torch.Tensor):
            dist_matrix2 = dist_matrix2.cpu().numpy()

        # Vectorized approach: extract all non-diagonal distances at once
        dist1_pairs = []
        dist2_pairs = []
        for i in range(n_refs):
            dist1_pairs.append(np.delete(dist_matrix1[i], i))
            dist2_pairs.append(np.delete(dist_matrix2[i], i))

        dist1_filtered = np.array(dist1_pairs, dtype=np.float32)
        dist2_filtered = np.array(dist2_pairs, dtype=np.float32)

        # Vectorized standardization (z-score normalization)
        mean1 = dist1_filtered.mean(axis=1, keepdims=True)
        mean2 = dist2_filtered.mean(axis=1, keepdims=True)
        std1 = dist1_filtered.std(axis=1, keepdims=True) + 1e-8
        std2 = dist2_filtered.std(axis=1, keepdims=True) + 1e-8

        dist1_std = (dist1_filtered - mean1) / std1
        dist2_std = (dist2_filtered - mean2) / std2

        correlations_np = (dist1_std * dist2_std).mean(axis=1)
        correlations_np = np.nan_to_num(correlations_np, nan=0.0)

    # Criterion 2: Mutual pair contribution tracking
    # OPTIMIZED: O(n_refs + n_pairs) instead of O(n_refs × n_pairs)
    mutual_contributions = np.zeros(n_refs, dtype=np.float32)
    if previous_mutual_pairs is not None and ind_emb1_unique is not None and ind_emb2_unique is not None:
        # Pre-compute counts: how many mutual pairs reference each index
        ref1_counts = {}  # ref_idx -> count of appearances in emb1
        ref2_counts = {}  # ref_idx -> count of appearances in emb2
        for i_mutual, j_mutual, _ in previous_mutual_pairs:
            idx1 = ind_emb1_unique[j_mutual]
            idx2 = ind_emb2_unique[i_mutual]
            ref1_counts[idx1] = ref1_counts.get(idx1, 0) + 1
            ref2_counts[idx2] = ref2_counts.get(idx2, 0) + 1

        # Now O(1) lookup per reference instead of O(n_pairs)
        n_mutual_pairs = len(previous_mutual_pairs)
        for i in range(n_refs):
            # Count contributions from both embedding spaces
            mutual_contribution = ref1_counts.get(ref_indices1[i], 0) + ref2_counts.get(ref_indices2[i], 0)
            # Normalize by number of mutual pairs to get contribution rate
            if n_mutual_pairs > 0:
                mutual_contribution /= n_mutual_pairs
            mutual_contributions[i] = mutual_contribution

    # Combine criteria: correlation + mutual NN contribution
    reference_scores = correlations_np + mutual_contributions
    reference_scores = list(reference_scores)
    
    # Select top k% references based on combined score
    reference_scores = np.array(reference_scores)
    n_to_keep = max(2, int(top_k_ratio * n_refs))  # Keep at least 2 references
    
    # Get indices of top k references
    top_indices = np.argsort(reference_scores)[-n_to_keep:]
    
    filtered_indices1 = ref_indices1[top_indices]
    filtered_indices2 = ref_indices2[top_indices]
    
    # Compute quality metrics
    mean_quality = np.mean(reference_scores)
    kept_quality = np.mean(reference_scores[top_indices])
    min_quality = np.min(reference_scores)
    max_quality = np.max(reference_scores)
    
    n_filtered = len(ref_indices1) - len(filtered_indices1)
    if n_filtered > 0:
        logger.debug(f"Filtered out {n_filtered} references using correlation + mutual NN contribution")
        logger.debug(f"Score range: [{min_quality:.4f}, {max_quality:.4f}]")
        logger.debug(f"Mean score: {mean_quality:.4f}, Kept score: {kept_quality:.4f}")
    
    # Prepare cache for next call
    new_cache = (current_hash, dist_matrix1, dist_matrix2)

    if return_metrics:
        return filtered_indices1, filtered_indices2, (mean_quality, kept_quality, min_quality, max_quality), new_cache
    else:
        return filtered_indices1, filtered_indices2, new_cache

# Ensemble functions moved to utils/ensemble_selection.py


def test_clu(args, seed=None):
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    optimize_for_large_datasets(args)

    # Always use RBF distance transformation (sigma auto-computed downstream)
    args.transformation = "rbf"
    args.transformation_params = None

    # Multi-GPU setup
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    use_multi_gpu = n_gpus > 1 and args.use_gpu

    if use_multi_gpu:
        logger.debug(f"Multi-GPU mode enabled: {n_gpus} GPUs detected")
        device = torch.device("cuda:0")  # Primary device
        gpu_ids = list(range(n_gpus))
        logger.debug(f"Using GPUs: {gpu_ids}")
        for i in range(n_gpus):
            logger.debug(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        device = torch.device("cuda" if (torch.cuda.is_available() and args.use_gpu) else "cpu")
        gpu_ids = None

    logger.debug(f"Using device: {device}")
    logger.debug(f"args.use_gpu: {args.use_gpu}")
    logger.debug(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.debug(f"CUDA device count: {n_gpus}")
        if not use_multi_gpu and n_gpus > 0:
            logger.debug(f"CUDA device name: {torch.cuda.get_device_name()}")
    if args.use_gpu and not torch.cuda.is_available():
        logger.warning("GPU requested but CUDA not available, using CPU")

    # Store multi-GPU config in args for downstream functions
    args.use_multi_gpu = use_multi_gpu
    args.gpu_ids = gpu_ids
    args.n_gpus = n_gpus
    multi_gpu_chunk_size = getattr(args, "multi_gpu_chunk_size", None)
    if use_multi_gpu:
        args.multi_gpu_config = {
            "enabled": True,
            "gpu_ids": gpu_ids,
            "chunk_size": multi_gpu_chunk_size
        }
    else:
        args.multi_gpu_config = None

    # Apply memory-efficient mode if enabled
    if getattr(args, 'memory_efficient', False):
        logger.debug("Memory-efficient mode enabled: adjusting parameters for large datasets")

    base_dir = args.base_dir
    large_datasets = {"scidocs", "fiqa", "fever"}

    def load_embedding_file(file_name: str, dataset_name: str):
        # Sequential read (no mmap) — avoids slow random page faults on network storage
        return load_npy(base_dir, file_name, mmap_mode=None)

    # Load embeddings
    if args.dataset in ["scifact", "scidocs", "fiqa", "nfcorpus", "arguana", "fever"]:
        emb1 = load_embedding_file(f"corpus_embeddings_{args.emb1}_{args.dataset}.npy", args.dataset)
        emb2 = load_embedding_file(f"corpus_embeddings_{args.emb2}_{args.dataset}.npy", args.dataset)
    elif args.dataset in ["cifar10"]:
        emb1 = load_npy(base_dir, f"emb_{args.dataset}/{args.dataset}_embeddings_{args.emb1}_{args.emb_dim1}.npy")
        emb2 = load_npy(base_dir, f"emb_{args.dataset}/{args.dataset}_embeddings_{args.emb2}_{args.emb_dim2}.npy")
    elif args.dataset in ["Citeseer", "Cora", "PubMed"]:
        emb1 = load_npy(base_dir, f"emb_{args.dataset}/{args.emb1}.npy")
        emb2 = load_npy(base_dir, f"emb_{args.dataset}/{args.emb2}.npy")
    elif args.dataset in ["biorxiv", "alloprof", "big_patent", "arxivp2p", "plsc", "wikicities", "stack_exchange"]:
        emb1 = load_npy(base_dir, f"{args.emb1}_{args.dataset}/{args.emb1}_{args.dataset}_embeddings.npy")
        emb2 = load_npy(base_dir, f"{args.emb2}_{args.dataset}/{args.emb2}_{args.dataset}_embeddings.npy")
    elif args.dataset in ["coco"]:
        emb1 = load_npy(base_dir, f"{args.dataset}_image_embeddings_{args.emb1}.npy")
        emb2 = load_npy(base_dir, f"{args.dataset}_text_embeddings_{args.emb1}.npy")
    elif args.dataset in ["StackExchangeClustering", "StackExchangeClustering.v2", "TwentyNewsgroups", "TwentyNewsgroupsClustering", "TwentyNewsgroupsClustering.v2", "RedditClustering.v2"]:
        emb1 = load_npy(base_dir, f"texts_embeddings_{args.emb1}_{args.dataset}.npy")
        emb2 = load_npy(base_dir, f"texts_embeddings_{args.emb2}_{args.dataset}.npy")
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    # Load indices
    # Add seed suffix to cache name if seed is provided
    seed_suffix = f"_seed{seed}" if seed is not None else ""

    if args.partition == "cluster_partial":
        ind_file_name = os.path.join(args.cache_dir, f"{args.dataset}_{args.emb1}_{args.partition}{args.n_clusters}_{args.nonref_clu_choices}{seed_suffix}")
    elif args.partition == "random":
        ind_file_name = os.path.join(args.cache_dir, f"{args.dataset}_{args.partition}_{args.overlap_ratio}{seed_suffix}")
    elif args.partition == "la2m":
        ind_file_name = os.path.join(args.cache_dir, f"{args.dataset}_la2m_{args.overlap_ratio}{seed_suffix}")

    if os.path.exists(os.path.join(ind_file_name, "ind1.npy")):
        ind_emb1_unique = load_npy(ind_file_name, "ind1")
        ind_emb2_unique = load_npy(ind_file_name, "ind2")
        # For random partition, overlap is the shared prefix of both index arrays.
        # Use set intersection via a boolean mask instead of sorted np.intersect1d.
        n_total = max(ind_emb1_unique.max(), ind_emb2_unique.max()) + 1
        mask = np.zeros(n_total, dtype=bool)
        mask[ind_emb1_unique] = True
        ind_nonref = ind_emb2_unique[mask[ind_emb2_unique]]
        del mask
        data_partitioner = DataPartitioner.from_indices(ind_emb1_unique, ind_emb2_unique, ind_nonref)

        # Compute cluster labels for Procrustes if enabled
        if args.use_procrustes:
            clusterer = Clusterer(method=args.cluster_method, n_clusters=args.n_clusters, use_gpu=args.use_gpu)
            emb1_cluster_labels = clusterer.fit(emb1)
            emb1_cluster_labels = LabelEncoder().fit_transform(emb1_cluster_labels)
        else:
            emb1_cluster_labels = None
    else:
        if args.partition == "la2m":
            # LA2M partition requires BEIR qrels
            if not BEIR_AVAILABLE:
                raise ImportError("BEIR is required for la2m partition. Install with: pip install beir")
            logger.debug(f"Creating LA2M partition for dataset {args.dataset}")
            qrels, dataset_index, dataset_obj = load_beir_qrels(args.dataset, data_path=args.base_dir)
            labels = np.zeros(len(emb1), dtype=np.int32)
            data_partitioner = DataPartitioner(
                labels=labels,
                total_ind=np.arange(len(emb1)),
                partition_type="la2m",
                overlap_ratio=args.overlap_ratio,
                qrels=qrels,
                dataset_index=dataset_index,
                dataset_obj=dataset_obj,
                select_top_1=True,
                remove_dup_answer=True
            )
            emb1_cluster_labels = None
        else:
            if args.partition == "random" and not args.use_procrustes:
                logger.debug("Creating random partition directly without clustering")
                data_partitioner = DataPartitioner(
                    labels=None,
                    total_ind=np.arange(len(emb1), dtype=np.int32),
                    partition_type=args.partition,
                    nonref_clu_choices=args.nonref_clu_choices,
                    overlap_ratio=args.overlap_ratio
                )
                emb1_cluster_labels = None
            else:
                clusterer = Clusterer(method=args.cluster_method, n_clusters=args.n_clusters, use_gpu=args.use_gpu)
                labels = clusterer.fit(emb1)
                labels = LabelEncoder().fit_transform(labels)
                data_partitioner = DataPartitioner(labels, partition_type=args.partition, nonref_clu_choices=args.nonref_clu_choices, overlap_ratio=args.overlap_ratio)
                # Save cluster labels for later use in Procrustes refinement
                emb1_cluster_labels = labels

        ind_emb1_unique = data_partitioner.ind_emb1_unique
        ind_emb2_unique = data_partitioner.ind_emb2_unique
        ind_nonref = data_partitioner.ind_emb1_nonref

        # Always save partition indices to cache (including seed-specific ones)
        save_npy(ind_file_name, "ind1", ind_emb1_unique)
        save_npy(ind_file_name, "ind2", ind_emb2_unique)
        save_npy(ind_file_name, "ind_nonref", ind_nonref)
    
    ref_indices1 = np.array([], dtype=np.int32)
    ref_indices2 = np.array([], dtype=np.int32) 
    
    if args.anchor_mode.startswith("ood"):
        logger.debug("Using OOD anchor generation")
        ref_dataset = args.ref_dataset
        ref_emb1 = load_embedding_file(f"corpus_embeddings_{args.emb1}_{ref_dataset}.npy", ref_dataset)
        ref_emb2 = load_embedding_file(f"corpus_embeddings_{args.emb2}_{ref_dataset}.npy", ref_dataset)

        # Calculate overlap size (number of points to be matched)
        overlap_size = len(ind_nonref)

        if args.partition == "cluster_partial":
            ref_ind_file_name = os.path.join(args.cache_dir, f"{args.ref_dataset}_{args.emb1}_{args.partition}{args.n_clusters}_{args.nonref_clu_choices}{seed_suffix}")
        elif args.partition == "random":
            ref_ind_file_name = os.path.join(args.cache_dir, f"{args.ref_dataset}_{args.partition}_{args.overlap_ratio}{seed_suffix}")
        elif args.partition == "la2m":
            ref_ind_file_name = os.path.join(args.cache_dir, f"{args.ref_dataset}_la2m_{args.overlap_ratio}{seed_suffix}")

        # Determine cache key based on whether n_seeds is provided
        if args.n_seeds is not None:
            ref_cache_key = f"ref_ind_n{args.n_seeds}"
        else:
            ref_cache_key = f"ref_ind_{args.ref_ratio}"

        if os.path.exists(os.path.join(ref_ind_file_name, f"{ref_cache_key}.npy")):
            ref_ind = load_npy(ref_ind_file_name, ref_cache_key)
        else:
            if args.partition == "cluster_partial":
                total_candidates = len(ref_emb1)
                if total_candidates == 0:
                    ref_ind = np.array([], dtype=np.int32)
                else:
                    ref_emb1_cluster = Cluster(
                        ref_emb1,
                        np.arange(total_candidates),
                        args.n_clusters_overlap,
                        args.cluster_method,
                        graph_method=args.graph_method,
                        knn_k=args.knn_k,
                        sample=args.sample,
                        use_gpu=getattr(args, "use_gpu", True)
                    )
                    label_list = ref_emb1_cluster.label_list
                    if isinstance(label_list, torch.Tensor):
                        label_list = label_list.cpu().numpy()
                    else:
                        label_list = np.asarray(label_list)

                    cluster_choices = getattr(args, "nonref_clu_choices", None)
                    choices_array = None
                    if cluster_choices is not None:
                        if isinstance(cluster_choices, str):
                            stripped = cluster_choices.strip()
                            if stripped.endswith("]"):
                                stripped = stripped[1:-1]
                            if stripped:
                                try:
                                    choices_array = np.array([int(item.strip()) for item in stripped.split(',') if item.strip()], dtype=np.int32)
                                except ValueError:
                                    choices_array = None
                        else:
                            try:
                                choices_array = np.array(cluster_choices, dtype=np.int32)
                            except (TypeError, ValueError):
                                choices_array = None

                    if choices_array is not None and choices_array.size > 0:
                        available_labels = np.intersect1d(label_list, choices_array)
                        if available_labels.size == 0:
                            available_labels = label_list
                    else:
                        available_labels = label_list

                    available_labels = np.asarray(available_labels, dtype=np.int32)
                    if available_labels.size == 0:
                        ref_ind = np.array([], dtype=np.int32)
                    else:
                        # Determine target count: use n_seeds if provided, otherwise ref_ratio
                        if args.n_seeds is not None:
                            target_total = min(args.n_seeds, total_candidates)
                        else:
                            target_total = max(1, int(round(args.ref_ratio * overlap_size)))
                        selected_mask = np.zeros(total_candidates, dtype=bool)
                        selected_indices = []
                        per_cluster = max(1, int(np.ceil(target_total / len(available_labels))))

                        for label in available_labels:
                            cluster_indices = ref_emb1_cluster.get_ori_ind(int(label))
                            if len(cluster_indices) == 0:
                                continue
                            remaining_slots = target_total - len(selected_indices)
                            if remaining_slots <= 0:
                                break
                            n_select = min(len(cluster_indices), per_cluster, remaining_slots)
                            if n_select <= 0:
                                continue
                            if len(cluster_indices) <= n_select:
                                chosen = cluster_indices
                            else:
                                chosen = np.random.choice(cluster_indices, size=n_select, replace=False)
                            selected_indices.extend(chosen.tolist())
                            selected_mask[chosen] = True

                        if len(selected_indices) < target_total:
                            remaining_slots = target_total - len(selected_indices)
                            if remaining_slots > 0:
                                remaining_indices = np.where(~selected_mask)[0]
                                if len(remaining_indices) > 0:
                                    if len(remaining_indices) <= remaining_slots:
                                        extra = remaining_indices
                                    else:
                                        extra = np.random.choice(remaining_indices, size=remaining_slots, replace=False)
                                    selected_indices.extend(extra.tolist())

                        ref_ind = np.array(selected_indices[:target_total], dtype=np.int32)
            elif args.partition == "random" or args.partition == "la2m":
                # Determine target count: use n_seeds if provided, otherwise ref_ratio
                if args.n_seeds is not None:
                    target_total = args.n_seeds
                else:
                    target_total = max(1, int(round(args.ref_ratio * overlap_size)))
                # Ensure we don't try to select more than available
                target_total = min(target_total, len(ref_emb1))
                ref_ind = np.random.choice(np.arange(len(ref_emb1)), size=target_total, replace=False)
            # Always save ref_ind to cache (including seed-specific ones)
            save_npy(ref_ind_file_name, ref_cache_key, ref_ind)
        
        ori_ref_emb1 = np.asarray(ref_emb1[ref_ind], dtype=np.float32)
        ori_ref_emb2 = np.asarray(ref_emb2[ref_ind], dtype=np.float32)
        # Use boolean mask for union size instead of sorted union
        _n = max(ind_emb1_unique.max(), ind_emb2_unique.max()) + 1
        _um = np.zeros(_n, dtype=bool)
        _um[ind_emb1_unique] = True
        _um[ind_emb2_unique] = True
        union_dataset_size = int(_um.sum())
        del _um
        actual_ref_ratio = len(ref_ind) / union_dataset_size if union_dataset_size > 0 else 0


    else:  # supervised mode (original method)
        logger.debug("Using supervised anchor initialization")

        # Determine cache key based on whether n_seeds is provided
        if args.n_seeds is not None:
            sup_cache_key = f"ref_ind_n{args.n_seeds}"
        else:
            sup_cache_key = f"ref_ind_{args.ref_ratio}"

        # Only load/save from cache for random sampling
        if args.ref_method == "random":
            # Try to load from cache first (seed-specific if seed is set)
            ref_ind = load_npy(ind_file_name, sup_cache_key)
            if ref_ind is None:
                # Generate new ref_ind
                ref_ind = sample_ref_points(
                    method=args.ref_method,
                    embeddings=emb1,
                    candidate_indices=data_partitioner.ind_emb1_nonref,
                    ref_ratio=args.ref_ratio,
                    n_samples=args.n_seeds,
                    use_gpu=args.use_gpu if hasattr(args, 'use_gpu') else False
                )
                # Save to cache for random method
                save_npy(ind_file_name, sup_cache_key, ref_ind)
        else:
            # For non-random methods, always generate fresh (no caching)
            ref_ind = sample_ref_points(
                method=args.ref_method,
                embeddings=emb1,
                candidate_indices=data_partitioner.ind_emb1_nonref,
                ref_ratio=args.ref_ratio,
                n_samples=args.n_seeds,
                use_gpu=args.use_gpu if hasattr(args, 'use_gpu') else False
            )

        # For supervised mode, we use the same anchors for both embeddings
        ori_ref_emb1 = np.asarray(emb1[ref_ind], dtype=np.float32)
        ori_ref_emb2 = np.asarray(emb2[ref_ind], dtype=np.float32)
        
        # Calculate actual ref_ratio for supervised mode
        # Use boolean mask instead of sorted union for speed on large arrays
        n_total = max(ind_emb1_unique.max(), ind_emb2_unique.max()) + 1
        union_mask = np.zeros(n_total, dtype=bool)
        union_mask[ind_emb1_unique] = True
        union_mask[ind_emb2_unique] = True
        union_dataset_size = int(union_mask.sum())
        actual_ref_ratio = len(ref_ind) / union_dataset_size if union_dataset_size > 0 else 0
        del union_mask

        # Remove supervised references from the partitioned data to avoid data leakage
        # Use boolean mask instead of sorted setdiff1d
        ref_set = np.zeros(n_total, dtype=bool)
        ref_set[ref_ind] = True
        ind_emb1_unique = ind_emb1_unique[~ref_set[ind_emb1_unique]]
        ind_emb2_unique = ind_emb2_unique[~ref_set[ind_emb2_unique]]
        ind_nonref = ind_nonref[~ref_set[ind_nonref]]
        del ref_set

    # Extract embeddings after all anchor mode processing to ensure supervised refs are excluded
    emb1_unique = np.asarray(emb1[ind_emb1_unique], dtype=np.float32)
    emb2_unique = np.asarray(emb2[ind_emb2_unique], dtype=np.float32)

    # Build global→local index mappings so we can free the full embedding arrays.
    # All discovered pair indices (ref_indices1/2) are subsets of ind_emb1/2_unique,
    # so we only need mappings for those index sets.
    n_total_emb1 = len(emb1)
    n_total_emb2 = len(emb2)
    emb1_g2l = np.full(n_total_emb1, -1, dtype=np.int64)
    emb1_g2l[ind_emb1_unique] = np.arange(len(ind_emb1_unique))
    emb2_g2l = np.full(n_total_emb2, -1, dtype=np.int64)
    emb2_g2l[ind_emb2_unique] = np.arange(len(ind_emb2_unique))

    # Free the full embedding arrays (saves ~111GB for fever float64 / ~55GB if float32).
    # For mmap-backed arrays this releases page cache; for regular arrays it frees heap.
    del emb1, emb2
    gc.collect()
    logger.debug(f"Freed full embedding arrays. Mapping arrays: {emb1_g2l.nbytes/1e6:.0f}MB + {emb2_g2l.nbytes/1e6:.0f}MB")

    auto_tune_for_scale(len(emb1_unique), len(emb2_unique), args)

    logger.debug(f"overlap_ratio: {args.overlap_ratio}")
    initial_working_mem_gb = (
        emb1_unique.nbytes +
        emb2_unique.nbytes +
        ori_ref_emb1.nbytes +
        ori_ref_emb2.nbytes
    ) / (1024**3)
    initial_precompute_mem_gb = 2 * len(emb1_unique) * len(ori_ref_emb1) * 4 / (1024**3)
    logger.debug(
        f"Working set sizes: emb1_unique={len(emb1_unique):,}, emb2_unique={len(emb2_unique):,}, "
        f"seed_refs={len(ori_ref_emb1):,}"
    )
    logger.debug(
        f"Initial memory estimate: unique+seed embeddings={initial_working_mem_gb:.2f} GiB, "
        f"precomputed distance matrices={initial_precompute_mem_gb:.2f} GiB"
    )

    # Warm-start phase for large datasets: run VecLink on a subsample first
    n_max_unique = max(len(emb1_unique), len(emb2_unique))
    large_threshold = getattr(args, 'large_dataset_threshold', 500_000)
    warmstart_ref_indices1 = None
    warmstart_ref_indices2 = None

    if n_max_unique > large_threshold:
        warmstart_size = getattr(args, 'warmstart_size', 50_000)
        warmstart_iters = getattr(args, 'warmstart_iters', 3)
        if warmstart_size < 100:
            logger.debug("Warm-start disabled (warmstart_size too small)")
            warmstart_size = 0
        if warmstart_size > 0:
            logger.info(f"Large dataset detected (N={n_max_unique:,}). Running warm-start on {warmstart_size:,} subsample...")

        if warmstart_size > 0:
            warmstart_ref_indices1, warmstart_ref_indices2 = run_subsampled_warmstart(
                emb1_unique, emb2_unique, ind_emb1_unique, ind_emb2_unique,
                ori_ref_emb1, ori_ref_emb2, ind_nonref, args, device,
                subsample_size=warmstart_size, n_iters=warmstart_iters
            )

        if warmstart_ref_indices1 is not None and len(warmstart_ref_indices1) > 0:
            logger.info(f"Warm-start discovered {len(warmstart_ref_indices1)} reference pairs")
        else:
            logger.warning("Warm-start found no pairs, proceeding with seeds only")
            warmstart_ref_indices1 = None
            warmstart_ref_indices2 = None

    # Prepare GPU tensors for local neighborhood retrieval on large datasets.
    # This enables the Bernoulli pipeline to work with few seeds on large corpora
    # by restricting MNN search to local neighborhoods around anchors.
    # Uses GPU brute-force matmul+topk instead of FAISS IVF (no slow training step).
    use_local_neighborhoods = (n_max_unique > large_threshold)
    nn_emb1_gpu = None
    nn_emb2_gpu = None
    k_neighbors = getattr(args, 'k_neighbors', 5000)

    if use_local_neighborhoods:
        logger.info(f"Preparing GPU tensors for local neighborhood search (k={k_neighbors})...")

        # k-NN databases are ALWAYS FP16 (neighbor search only needs approximate
        # indices, and fp16 halves resident GPU memory). Normalize in fp32 on CPU,
        # cast to fp16 on CPU, THEN move to GPU — so the GPU never materializes the
        # fp32 full-corpus tensor. A 4096-dim corpus is ~51GB in fp32; the old code
        # put BOTH sides' fp32 on GPU and then made fp16 copies, peaking ~96GB and
        # OOMing at load for small×large pairs (e.g. mistral×qwen). Building fp16
        # directly caps resident GPU use at the fp16 databases (~32GB for 1024+4096).
        def _build_fp16_nn_db(emb_np):
            t = torch.from_numpy(emb_np).float()
            t /= t.norm(dim=1, keepdim=True).clamp(min=1e-8)
            return t.half().to(device)
        nn_emb1_gpu = _build_fp16_nn_db(emb1_unique)
        torch.cuda.empty_cache()
        nn_emb2_gpu = _build_fp16_nn_db(emb2_unique)
        torch.cuda.empty_cache()

        logger.info(f"  GPU neighborhood tensors (fp16): E1={nn_emb1_gpu.shape} ({nn_emb1_gpu.element_size()*nn_emb1_gpu.nelement()/1e9:.1f}GB), "
                   f"E2={nn_emb2_gpu.shape} ({nn_emb2_gpu.element_size()*nn_emb2_gpu.nelement()/1e9:.1f}GB)")
        gc.collect()

        # The --fp16 flag controls ensemble MNN precision (distance encoding + CSLS), not k-NN.
        _use_fp16 = getattr(args, 'fp16', True)
        _nn_databases = {'e1': nn_emb1_gpu, 'e2': nn_emb2_gpu}
        _ens_prec_str = "FP16" if _use_fp16 else "FP32"
        logger.info(f"Neighborhood search ready (torch FP16 GPU matmul). Ensemble MNN: {_ens_prec_str}")

        # Incremental k-NN cache: maps (space, row_hash) → neighbor indices (max k seen).
        # Database never changes, so a query embedding always gets the same neighbors.
        # Cache is k-independent: stores results at max k, truncates on lookup.
        _nn_cache = {}  # key: (space, bytes) → value: np.array of shape (cached_k,)

        def nn_search_fn(query_emb_np, space, k):
            """Search via chunked GPU matmul + topk, with cross-iteration caching."""
            db = _nn_databases[space]
            k_eff = min(k, db.shape[0])
            query_f32 = np.ascontiguousarray(query_emb_np, dtype=np.float32)
            n_q = len(query_f32)

            result = np.empty((n_q, k_eff), dtype=np.int64)
            uncached_mask = np.ones(n_q, dtype=bool)

            # Use first 8 floats as a fast hash key
            _hash_dim = min(8, query_f32.shape[1])
            for i in range(n_q):
                _key = (space, query_f32[i, :_hash_dim].tobytes())
                if _key in _nn_cache:
                    _cached = _nn_cache[_key]
                    if len(_cached) >= k_eff:
                        result[i] = _cached[:k_eff]
                        uncached_mask[i] = False

            n_uncached = uncached_mask.sum()
            if n_uncached == 0:
                logger.debug(f"  nn_search {space}: 100% cache hit ({n_q:,} queries)")
                return result

            # Search uncached queries at max(k_eff, existing cache k) for future reuse
            _search_k = k_eff
            uncached_q = query_f32[uncached_mask]
            query = torch.from_numpy(uncached_q).to(db.device).half()
            query = query / query.norm(dim=1, keepdim=True).clamp(min=1e-4)
            n_db = db.shape[0]
            max_chunk = max(1, int(8e9 / (n_db * 2)))
            all_idx = []
            for qs in range(0, len(query), max_chunk):
                qe = min(qs + max_chunk, len(query))
                sims = torch.mm(query[qs:qe], db.T)
                _, idx = sims.topk(_search_k, dim=1)
                all_idx.append(idx.cpu())
                del sims
            del query
            uncached_results = torch.cat(all_idx, dim=0).numpy()

            # Store in cache and fill result
            uncached_indices = np.where(uncached_mask)[0]
            for j, orig_i in enumerate(uncached_indices):
                _key = (space, query_f32[orig_i, :_hash_dim].tobytes())
                _nn_cache[_key] = uncached_results[j]
                result[orig_i] = uncached_results[j][:k_eff]

            hit_pct = 100 * (n_q - n_uncached) / n_q
            logger.debug(f"  nn_search {space}: {hit_pct:.0f}% cache hit ({n_q - n_uncached:,}/{n_q:,}), "
                        f"searched {n_uncached:,} new queries")
            return result

    # Initialize tracking for quality-based annealing
    quality_history = []

    # Initialize convergence tracking (paper Appendix C.2.1: burn-in T_min=10, Δ<0.01)
    mutual_nn_history = []
    convergence_threshold = 0.01  # Stop if mutual_nn_ratio change < 1%
    min_convergence_iters = 10  # Burn-in T_min: need 10 consecutive stable iterations
    gt_concat_min_ratio = 0.1  # Disable GT concat in ensemble subsets once mutual_nn_ratio drops below this

    # Initialize Bernoulli trials history if enabled
    pair_history = None
    posterior_stats = None
    pair_voting_refs = None  # Track which references voted for each pair
    bernoulli_dist_cache = None  # Cached distance matrices for incremental extension

    # Track total ensembles run across all iterations (for adaptive posterior threshold)
    total_ensembles_run = 0
    concat_seed_pairs_enabled = args.concat_seed_pairs

    # Initialize distance cache for reference filtering
    ref_dist_cache = None

    # Track ALL discovered pairs for neighborhood expansion (not capped).
    # Distance encoding uses the capped ref set (limited dimensionality),
    # but neighborhood queries use ALL discoveries for maximum spatial coverage.
    all_discovered_g1 = np.array([], dtype=np.int64)  # global indices in emb1
    all_discovered_g2 = np.array([], dtype=np.int64)  # global indices in emb2

    # Timing & memory tracking (excludes data loading)
    import resource as _resource
    _iter_wall_start = time.time()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    _cpu_rss_before = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss  # KB on Linux

    # Iterative refinement loop with anchor updating
    iteration = 0
    while True:
        if iteration == 0:
            ref_emb1 = ori_ref_emb1
            ref_emb2 = ori_ref_emb2
            # Incorporate warm-start references if available
            if warmstart_ref_indices1 is not None and len(warmstart_ref_indices1) > 0:
                ws_local1 = np.array([emb1_g2l[g] for g in warmstart_ref_indices1 if emb1_g2l[g] >= 0])
                ws_local2 = np.array([emb2_g2l[g] for g in warmstart_ref_indices2 if emb2_g2l[g] >= 0])
                if len(ws_local1) > 0:
                    ref_emb1 = np.concatenate([ori_ref_emb1, emb1_unique[ws_local1]])
                    ref_emb2 = np.concatenate([ori_ref_emb2, emb2_unique[ws_local2]])
                    # Initialize ref_indices with warm-start pairs
                    ref_indices1 = warmstart_ref_indices1[:len(ws_local1)]
                    ref_indices2 = warmstart_ref_indices2[:len(ws_local2)]
                    logger.debug(f"Iteration 0: seeded with {len(ws_local1)} warm-start refs + {len(ori_ref_emb1)} seeds")
                # Clear warm-start data (only used once)
                warmstart_ref_indices1 = None
                warmstart_ref_indices2 = None
        else:
            if use_local_neighborhoods:
                # Per-view mode: skip mmap gather — use GPU tensors directly
                # Build a lightweight ref_emb from GPU tensors (already in memory)
                _ref_local1 = np.array([emb1_g2l[g] for g in ref_indices1], dtype=np.int64)
                _ref_local2 = np.array([emb2_g2l[g] for g in ref_indices2], dtype=np.int64)
                ref_emb1 = _nn_databases['e1'][_ref_local1].float().cpu().numpy()
                ref_emb2 = _nn_databases['e2'][_ref_local2].float().cpu().numpy()
                # Prepend original seeds
                ref_emb1 = np.concatenate((ori_ref_emb1 / (np.linalg.norm(ori_ref_emb1, axis=1, keepdims=True) + 1e-8), ref_emb1))
                ref_emb2 = np.concatenate((ori_ref_emb2 / (np.linalg.norm(ori_ref_emb2, axis=1, keepdims=True) + 1e-8), ref_emb2))
                max_refs = None
            else:
                ref_emb1 = np.concatenate((ori_ref_emb1, emb1_unique[emb1_g2l[ref_indices1]]))
                ref_emb2 = np.concatenate((ori_ref_emb2, emb2_unique[emb2_g2l[ref_indices2]]))
                max_refs = getattr(args, 'max_refs', None)
            if max_refs is not None and len(ref_emb1) > max_refs:
                logger.debug(f"Capping references: {len(ref_emb1)} -> {max_refs}")
                n_ori = len(ori_ref_emb1)
                n_discovered = len(ref_emb1) - n_ori
                n_keep_discovered = max(0, max_refs - n_ori)

                if n_keep_discovered < n_discovered:
                    # Use quality-based selection if we have enough refs and filtering is feasible
                    # For very large ref sets, fall back to random to avoid O(n_ref^2) filtering cost
                    if n_discovered > 10_000:
                        # Random downsample for very large ref sets
                        keep_idx = np.random.choice(n_discovered, n_keep_discovered, replace=False)
                        logger.debug(f"  Using random downsample (n_discovered={n_discovered} too large for quality filter)")
                    else:
                        # Quality-based: score by distance correlation and keep best
                        try:
                            discovered_emb1 = emb1_unique[emb1_g2l[ref_indices1]]
                            discovered_emb2 = emb2_unique[emb2_g2l[ref_indices2]]
                            # Compute pairwise distance correlation as quality score
                            from utils.graph_util import get_dists
                            d1 = get_dists(discovered_emb1, ori_ref_emb1, metric=args.distance_metric, use_gpu=False)
                            d2 = get_dists(discovered_emb2, ori_ref_emb2, metric=args.distance_metric, use_gpu=False)
                            if isinstance(d1, torch.Tensor):
                                d1 = d1.cpu().numpy()
                            if isinstance(d2, torch.Tensor):
                                d2 = d2.cpu().numpy()
                            # Pearson correlation per discovered ref
                            d1_z = (d1 - d1.mean(axis=1, keepdims=True)) / (d1.std(axis=1, keepdims=True) + 1e-8)
                            d2_z = (d2 - d2.mean(axis=1, keepdims=True)) / (d2.std(axis=1, keepdims=True) + 1e-8)
                            corr = (d1_z * d2_z).mean(axis=1)
                            keep_idx = np.argsort(corr)[-n_keep_discovered:]
                            logger.debug(f"  Using quality-based selection (corr range: {corr.min():.3f} to {corr.max():.3f})")
                        except Exception as e:
                            logger.warning(f"  Quality-based ref capping failed ({e}), falling back to random")
                            keep_idx = np.random.choice(n_discovered, n_keep_discovered, replace=False)

                    # Rebuild ref arrays: keep all original seeds + selected discovered
                    ref_indices1 = ref_indices1[keep_idx]
                    ref_indices2 = ref_indices2[keep_idx]
                    ref_emb1 = np.concatenate((ori_ref_emb1, emb1_unique[emb1_g2l[ref_indices1]]))
                    ref_emb2 = np.concatenate((ori_ref_emb2, emb2_unique[emb2_g2l[ref_indices2]]))

        iteration += 1

        # Compute ensemble parameters with scaling for both strategies
        current_ref_size = len(ref_emb1)

        # For very small ref sets (< 50), use all refs per ensemble (no subsetting)
        # to maximize the dimensionality of distance vectors. Once refs grow, switch to subsetting.
        force_full_refs = current_ref_size < 50

        # Paper view schedule (Appendix B.2): with anchor-pool growth ratio
        # g_t = |L_{t-1}| / |S|, the scale factor is sf_t = 1 + c·log(g_t), and the
        # schedule is m_t = ceil(m0·sf_t) views of size s_t = ceil(ρ0·|L_{t-1}|/sf_t).
        n_seeds_eff = max(len(ori_ref_emb1), 1)
        g_t = current_ref_size / n_seeds_eff
        schedule_c = getattr(args, 'schedule_c', 0.3)
        sf_t = 1.0 + schedule_c * np.log(max(g_t, 1.0))

        # m0 = args.ensemble_n_ensembles (default 5 = ceil(2/ρ0)), ρ0 = args.ensemble_subset_ratio
        ensemble_n_ensembles = int(np.ceil(args.ensemble_n_ensembles * sf_t))
        if force_full_refs:
            ensemble_subset_ratio = 1.0
        else:
            subset_size = int(np.ceil(args.ensemble_subset_ratio * current_ref_size / sf_t))
            subset_size = max(1, min(subset_size, current_ref_size))
            ensemble_subset_ratio = subset_size / current_ref_size
        logger.debug(f"Iteration {iteration}: paper schedule g_t={g_t:.2f}, sf_t={sf_t:.3f}, "
                   f"ref_size={current_ref_size}, subset_ratio={ensemble_subset_ratio:.3f}, "
                   f"n_ensembles={ensemble_n_ensembles}"
                   f"{' (FULL REFS: too few for subsetting)' if force_full_refs else ''}")

        # Accumulate total ensembles run
        total_ensembles_run += ensemble_n_ensembles

        args.concat_seed_pairs = concat_seed_pairs_enabled

        # OOM retry logic: if we get OOM, reduce max_parallel_workers and retry this iteration
        oom_retry_count = 0
        max_oom_retries = 3
        original_max_parallel_workers = args.max_parallel_workers

        while oom_retry_count <= max_oom_retries:
            try:
                per_view_active = use_local_neighborhoods
                pool_emb1 = emb1_unique
                pool_emb2 = emb2_unique
                pool_ind1 = ind_emb1_unique
                pool_ind2 = ind_emb2_unique
                if per_view_active:
                    logger.info(f"Per-view neighborhoods active (iteration={iteration})")
                    bernoulli_dist_cache = None
                    # pair_history persists across iterations: per-view pair keys are
                    # stable global indices, so the Beta-Bernoulli posterior accumulates
                    # evidence over all iterations (cumulative), matching the paper.

                if os.environ.get("VECLINK_DUMP_POSTERIORS_DIR"):
                    os.environ["VECLINK_DUMP_ITER"] = str(iteration)

                # Bernoulli trial-based ensemble selection
                mutual_pairs, pair_history, posterior_stats, pair_voting_refs, bernoulli_dist_cache = ensemble_reference_selection_bernoulli(
                    ref_emb1, ref_emb2, pool_emb1, pool_emb2, pool_ind1, pool_ind2,
                    args, device,
                    n_ensembles=ensemble_n_ensembles, subset_ratio=ensemble_subset_ratio,
                    ref_indices1=ref_indices1, ref_indices2=ref_indices2,
                    ori_ref_emb1=ori_ref_emb1, ori_ref_emb2=ori_ref_emb2,
                    pair_history=pair_history, posterior_threshold=args.posterior_threshold,
                    ensemble_strategy=args.ensemble_strategy,
                    posterior_strategy="iteration_based",
                    current_iteration=iteration,
                    max_iterations=args.max_iter,
                    total_ensembles_run=total_ensembles_run,
                    overlap_inference_method=args.overlap_inference_method,
                    cached_dist_matrices=bernoulli_dist_cache if (iteration > 1 and not use_local_neighborhoods) else None,
                    per_view_neighborhoods=per_view_active,
                    nn_search_fn=nn_search_fn if per_view_active else None,
                    view_k_neighbors=getattr(args, 'per_view_k', 200),
                    full_emb1=_nn_databases['e1'] if per_view_active else None,
                    full_emb2=_nn_databases['e2'] if per_view_active else None,
                )
                # Success! Break out of retry loop
                break

            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                # Check if it's actually an OOM error
                if isinstance(e, RuntimeError) and "out of memory" not in str(e).lower():
                    raise  # Re-raise if not OOM

                oom_retry_count += 1

                # Clear CUDA cache and run garbage collection
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

                if oom_retry_count > max_oom_retries:
                    logger.error(f"Iteration {iteration}: OOM retry limit ({max_oom_retries}) reached. Cannot continue.")
                    raise

                # Reduce max_parallel_workers for retry
                if args.max_parallel_workers is None:
                    # If it was None (auto), start with 4 workers
                    args.max_parallel_workers = 4
                else:
                    # Reduce by half, minimum 1
                    args.max_parallel_workers = max(1, args.max_parallel_workers // 2)

                # Reset peak memory stats so OOM spike doesn't inflate the final metric
                if device.type == 'cuda':
                    torch.cuda.reset_peak_memory_stats(device)

                logger.warning(f"Iteration {iteration}: OOM detected (retry {oom_retry_count}/{max_oom_retries}). "
                             f"Reducing max_parallel_workers to {args.max_parallel_workers} and retrying...")

        # Restore original max_parallel_workers for next iteration
        args.max_parallel_workers = original_max_parallel_workers
        
        # Update reference indices and embeddings
        # For Bernoulli path with local neighborhoods, mutual_pairs indices are relative
        # to the pool (pool_ind1/pool_ind2), not the full ind_emb1_unique/ind_emb2_unique.
        # Exception: per-view mode returns global indices directly.
        if per_view_active:
            # Phase 2 per-view: pairs already contain global indices
            new_ref_indices1 = np.array([mutual_pairs[i][1] for i in range(len(mutual_pairs))])
            new_ref_indices2 = np.array([mutual_pairs[i][0] for i in range(len(mutual_pairs))])
        else:
            new_ref_indices1 = np.array([pool_ind1[mutual_pairs[i][1]] for i in range(len(mutual_pairs))])
            new_ref_indices2 = np.array([pool_ind2[mutual_pairs[i][0]] for i in range(len(mutual_pairs))])
        
        # Accumulate ALL discovered pairs for neighborhood expansion (uncapped)
        if use_local_neighborhoods and len(new_ref_indices1) > 0:
            all_discovered_g1 = np.unique(np.concatenate([all_discovered_g1, new_ref_indices1]))
            all_discovered_g2 = np.unique(np.concatenate([all_discovered_g2, new_ref_indices2]))

        # Convert ensemble result to expected format
        mutual_nn = len(mutual_pairs)

        if mutual_nn == 0:
            logger.warning("No mutual nearest neighbors, break")
            break

        total_points = len(emb1_unique)
        mutual_nn_ratio = mutual_nn / total_points if total_points > 0 else 0.0

        # Apply cluster-wise Procrustes transformation if enabled
        if args.use_procrustes and mutual_nn > 0 and emb1_cluster_labels is not None:
            # Get cluster labels for emb1_unique (subset)
            emb1_unique_cluster_labels = emb1_cluster_labels[ind_emb1_unique]

            # Define wrapper for finding mutual pairs using existing ensemble method
            def find_mutual_pairs_wrapper(emb1_cluster, emb2, ind1_cluster, ind2):
                """GPU-accelerated wrapper to find mutual NNs between cluster and emb2"""
                import torch
                from loguru import logger

                # Get device from args or use cuda if available
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                use_gpu = torch.cuda.is_available()

                if use_gpu:
                    try:
                        logger.debug(f"GPU mutual NN: cluster={len(emb1_cluster)} pts, emb2={len(emb2)} pts")

                        # Convert to PyTorch tensors and move to GPU
                        if not torch.is_tensor(emb1_cluster):
                            emb1_t = torch.from_numpy(emb1_cluster).to(device).float()
                        else:
                            emb1_t = emb1_cluster.to(device)

                        if not torch.is_tensor(emb2):
                            emb2_t = torch.from_numpy(emb2).to(device).float()
                        else:
                            emb2_t = emb2.to(device)

                        # Normalize vectors on GPU
                        emb1_norm = torch.nn.functional.normalize(emb1_t, p=2, dim=1)
                        emb2_norm = torch.nn.functional.normalize(emb2_t, p=2, dim=1)

                        # Forward pass: cluster -> emb2 (single GPU matrix multiply, no chunking needed)
                        # Cosine similarity via normalized dot product
                        sim_1to2 = emb1_norm @ emb2_norm.T  # (n1, n2)
                        nn_1to2 = torch.argmax(sim_1to2, dim=1)  # (n1,)

                        # Backward pass: emb2 -> cluster (single GPU matrix multiply)
                        sim_2to1 = emb2_norm @ emb1_norm.T  # (n2, n1)
                        nn_2to1 = torch.argmax(sim_2to1, dim=1)  # (n2,)

                        # Vectorized mutual pair detection on GPU (NO Python loop!)
                        n1 = emb1_norm.shape[0]
                        idx = torch.arange(n1, device=device)  # [0, 1, 2, ..., n1-1]

                        # For each cluster point i, check if nn_2to1[nn_1to2[i]] == i
                        # This is the mutual NN condition, fully vectorized
                        is_mutual = nn_2to1[nn_1to2] == idx  # Boolean tensor (n1,)

                        # Extract mutual pairs
                        mutual_i = idx[is_mutual]  # Cluster indices with mutual NNs
                        mutual_j = nn_1to2[is_mutual]  # Their corresponding emb2 indices

                        # Convert to original indices and move to CPU (single transfer!)
                        mutual_i_cpu = mutual_i.cpu().numpy()
                        mutual_j_cpu = mutual_j.cpu().numpy()

                        # Build result list with original indices
                        mutual = [(int(ind1_cluster[i]), int(ind2[j]))
                                  for i, j in zip(mutual_i_cpu, mutual_j_cpu)]

                        # Cleanup GPU memory
                        del emb1_t, emb2_t, emb1_norm, emb2_norm, sim_1to2, sim_2to1
                        del nn_1to2, nn_2to1, is_mutual, mutual_i, mutual_j
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                        logger.debug(f"GPU mutual NN complete: found {len(mutual)} pairs")
                        return mutual

                    except Exception as e:
                        logger.warning(f"GPU mutual NN failed, falling back to CPU: {e}")
                        # Fall through to CPU implementation below

                # CPU fallback (original implementation - used when GPU unavailable or fails)
                logger.debug("Using CPU mutual NN finding")
                chunk_size = 500
                n1, n2 = len(emb1_cluster), len(emb2)

                emb1_norm = emb1_cluster / (np.linalg.norm(emb1_cluster, axis=1, keepdims=True) + 1e-8)
                emb2_norm = emb2 / (np.linalg.norm(emb2, axis=1, keepdims=True) + 1e-8)

                nn_1to2 = np.zeros(n1, dtype=np.int32)
                for i in range(0, n1, chunk_size):
                    chunk = emb1_norm[i:i+chunk_size]
                    sim_chunk = chunk @ emb2_norm.T
                    nn_1to2[i:i+chunk_size] = np.argmax(sim_chunk, axis=1)
                    del sim_chunk

                nn_2to1 = np.zeros(n2, dtype=np.int32)
                for j in range(0, n2, chunk_size):
                    chunk = emb2_norm[j:j+chunk_size]
                    sim_chunk = chunk @ emb1_norm.T
                    nn_2to1[j:j+chunk_size] = np.argmax(sim_chunk, axis=1)
                    del sim_chunk

                mutual = []
                for i, j in enumerate(nn_1to2):
                    if nn_2to1[j] == i:
                        mutual.append((ind1_cluster[i], ind2[j]))

                return mutual

            # Apply cluster-wise Procrustes refinement (includes deduplication internally)
            refined_ind1, refined_ind2, _ = cluster_wise_procrustes_refinement(
                emb1_unique,
                emb2_unique,
                ind_emb1_unique,
                ind_emb2_unique,
                new_ref_indices1,  # mutual pairs from emb1
                new_ref_indices2,  # mutual pairs from emb2
                emb1_unique_cluster_labels,
                find_mutual_pairs_wrapper,
                allow_scale=True,
                allow_translation=True,
                min_pairs_per_cluster=3,
                verbose=True,
                use_gpu=args.use_gpu,
                device=device
            )

            # Update reference indices with refined mutual NNs (keep original embeddings)
            if len(refined_ind1) > 0:
                ref_indices1 = refined_ind1
                ref_indices2 = refined_ind2
                logger.debug(f"Iteration {iteration}: Procrustes refinement found {len(refined_ind1)} refined mutual NN pairs")
            else:
                logger.warning(f"Iteration {iteration}: Procrustes refinement found no pairs, keeping original")

        else:
            ref_indices1 = new_ref_indices1
            ref_indices2 = new_ref_indices2

        # Track mutual NN ratio for convergence detection
        mutual_nn_history.append(mutual_nn_ratio)

        if concat_seed_pairs_enabled and mutual_nn_ratio < gt_concat_min_ratio:
            concat_seed_pairs_enabled = False
            logger.debug(f"Disabling seed pair concat in ensemble subsets: mutual_nn_ratio {mutual_nn_ratio:.4f} < {gt_concat_min_ratio:.4f}")
        
        # Check for convergence: stable mutual_nn_ratio over multiple iterations
        if len(mutual_nn_history) >= min_convergence_iters + 1:
            recent_ratios = mutual_nn_history[-(min_convergence_iters + 1):]
            ratio_changes = [abs(recent_ratios[i] - recent_ratios[i-1]) for i in range(1, len(recent_ratios))]
            max_change = max(ratio_changes) if ratio_changes else float('inf')

            if max_change < convergence_threshold:
                logger.debug(f"Stopping: converged (mutual_nn_ratio stable at {mutual_nn_ratio:.4f}, max change: {max_change:.4f})")
                break

        if len(mutual_pairs) == 0:
            logger.debug("Stopping: no mutual pairs")
            break
            
        if iteration >= args.max_iter:
            logger.debug(f"Stopping: reached maximum iterations ({args.max_iter})")
            break
        
        if(len(ref_indices1) != len(np.unique(ref_indices1))):
            logger.error(f"ref_indices1: {ref_indices1}")
            logger.error(f"ref_indices2: {ref_indices2}")
            raise ValueError("ref_indices1 has duplicates")
        if (len(ref_indices2) != len(np.unique(ref_indices2))):
            logger.error(f"ref_indices1: {ref_indices1}")
            logger.error(f"ref_indices2: {ref_indices2}")
            raise ValueError("ref_indices2 has duplicates")
                
        # Apply improved reference filtering based on pairwise distance quality
        if args.enable_ref_filtering and len(ref_indices1) >= 5:  # Only filter if enabled and we have enough references
            prev_accuracy, prev_recall, prev_correct = compute_accuracy_recall(ref_indices1, ref_indices2, ind_nonref)
            
            # Compute annealed ref_filter_ratio based on quality metrics
            if hasattr(args, 'ref_filter_annealing') and args.ref_filter_annealing != "none":
                initial_ratio = args.ref_filter_ratio
                final_ratio = getattr(args, 'ref_filter_final_ratio', None)
                if final_ratio is None:
                    final_ratio = min(1.0, args.ref_filter_ratio * 1.5)  # Increase filter ratio over time
                current_filter_ratio = compute_annealed_ref_filter_ratio(
                    iteration, args.max_iter, initial_ratio, final_ratio,
                    args.ref_filter_annealing, quality_history
                )
                logger.debug(f"Iteration {iteration}: Using annealed ref_filter_ratio={current_filter_ratio:.3f} (initial={initial_ratio:.3f}, final={final_ratio:.3f})")
            else:
                current_filter_ratio = args.ref_filter_ratio
            
            # Apply filtering and get quality metrics
            # Pass previous mutual pairs for contribution tracking (if available)
            previous_mutual_pairs_for_filtering = mutual_pairs if iteration > 1 else None
            
            filter_result = filter_references_by_pairwise_distance_quality(
                ref_indices1, ref_indices2, emb1_unique, emb2_unique,
                distance_metric=args.distance_metric, top_k_ratio=current_filter_ratio, device=device,
                return_metrics=True, previous_mutual_pairs=previous_mutual_pairs_for_filtering,
                ind_emb1_unique=ind_emb1_unique, ind_emb2_unique=ind_emb2_unique,
                use_multi_gpu=args.use_multi_gpu, gpu_ids=args.gpu_ids,
                multi_gpu_config=args.multi_gpu_config,
                cached_dist_matrices=ref_dist_cache,
                emb1_g2l=emb1_g2l, emb2_g2l=emb2_g2l
            )

            if len(filter_result) == 4:  # Got metrics + cache
                ref_indices1, ref_indices2, quality_metrics, ref_dist_cache = filter_result
                mean_quality, kept_quality, min_quality, max_quality = quality_metrics
                quality_history.append((mean_quality, kept_quality, min_quality, max_quality))
                logger.debug(f"Iteration {iteration}: Quality metrics - mean: {mean_quality:.4f}, kept: {kept_quality:.4f}")
            elif len(filter_result) == 3:  # No metrics but has cache
                ref_indices1, ref_indices2, ref_dist_cache = filter_result
                
            current_accuracy, current_recall, current_correct = compute_accuracy_recall(ref_indices1, ref_indices2, ind_nonref)
            logger.debug(f"Iteration {iteration}: prev_accuracy={prev_accuracy:.4f}, prev_recall={prev_recall:.4f}, current_accuracy={current_accuracy:.4f}, current_recall={current_recall:.4f}")
            # Invalidate Bernoulli dist cache after filtering (refs reordered/removed)
            bernoulli_dist_cache = None
        # EVALUATION ONLY: ground truth used for monitoring, not for algorithm decisions
        accuracy, recall, correct = compute_accuracy_recall(ref_indices1, ref_indices2, ind_nonref)
        logger.info(f"Iteration {iteration}: accuracy={correct}/{len(ref_indices1)}={accuracy:.4f}, recall={correct}/{len(ind_nonref)}={recall:.4f}")

        # Write per-iteration results to CSV log
        import time as _time_mod
        iter_log_path = f"cache/iter_log_{args.dataset}_{args.emb1}_{args.emb2}_s{args.n_seeds}.csv"
        os.makedirs(os.path.dirname(iter_log_path), exist_ok=True)
        write_header = not os.path.exists(iter_log_path) or iteration == 1
        with open(iter_log_path, 'a' if not write_header else 'w') as f:
            if write_header:
                f.write("iteration,precision,recall,correct,total_pairs,pool_e1,pool_e2,n_refs,time\n")
            pool_e1_size = len(pool_emb1) if 'pool_emb1' in dir() else len(emb1_unique)
            pool_e2_size = len(pool_emb2) if 'pool_emb2' in dir() else len(emb2_unique)
            f.write(f"{iteration},{accuracy:.6f},{recall:.6f},{correct},{len(ref_indices1)},{pool_e1_size},{pool_e2_size},{len(ref_emb1)},{_time_mod.strftime('%H:%M:%S')}\n")

        # Perform distance-based analysis for supervised mode
        if args.anchor_mode == "supervised" and 'ref_ind' in locals():
            try:
                analysis_result = analyze_distance_based_accuracy(
                    ref_indices1=ref_indices1,
                    ref_indices2=ref_indices2,
                    emb1=emb1_unique,
                    emb2=emb2_unique,
                    anchor_indices=ref_ind,
                    distance_metric=args.distance_metric,
                    use_gpu=args.use_gpu,
                    device=device,
                    emb1_g2l=emb1_g2l,
                    emb2_g2l=emb2_g2l,
                    anchor_emb1=ori_ref_emb1,
                    anchor_emb2=ori_ref_emb2
                )

                avg_corr = analysis_result['avg_distance_correlation']
                min_corr = analysis_result['min_distance_correlation']

                logger.debug(f"DISTANCE-BASED ANALYSIS (Iteration {iteration})")
                if not np.isnan(avg_corr):
                    logger.debug(f"  Average distance to anchors: r={avg_corr:.4f}, p={analysis_result['avg_distance_p_value']:.4e}")
                else:
                    logger.debug(f"  Average distance to anchors: Cannot compute (insufficient variance)")
                if not np.isnan(min_corr):
                    logger.debug(f"  Minimum distance to anchors: r={min_corr:.4f}, p={analysis_result['min_distance_p_value']:.4e}")
                else:
                    logger.debug(f"  Minimum distance to anchors: Cannot compute (insufficient variance)")

                for (min_pct, max_pct), stats in sorted(analysis_result['percentile_breakdown'].items()):
                    range_str = f"{min_pct}-{max_pct}%"
                    acc_str = f"{stats['accuracy']:.4f}" if stats['n_pairs'] > 0 else "N/A"
                    logger.debug(f"  {range_str:<15} acc={acc_str}, pairs={stats['n_pairs']}, correct={stats['n_correct']}")

            except Exception as e:
                logger.warning(f"Distance-based analysis failed: {e}")

            ref_indices1_local = convert_global_to_local_indices(ref_indices1, ind_emb1_unique)
            ref_indices2_local = convert_global_to_local_indices(ref_indices2, ind_emb2_unique)

    # Record metrics (excludes data loading time)
    args._runtime_s = time.time() - _iter_wall_start
    args._peak_gpu_mb = (torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                         if device.type == 'cuda' else 0.0)
    args._peak_cpu_rss_mb = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss / 1024  # Linux: KB → MB

    accuracy, recall, correct = compute_accuracy_recall(ref_indices1, ref_indices2, ind_nonref)
    logger.debug(f"accuracy: {correct}/{len(ref_indices1)} = {accuracy}, recall: {correct}/{len(ind_nonref)} = {recall}")

    # Save ref_indices1 and ref_indices2 if save_ref_indices is enabled
    if getattr(args, 'save_ref_indices', False):
        # Determine base directory for saving
        if getattr(args, 'save_ref_indices_dir', None):
            # Use custom directory with partition subdirectory structure
            base_dir = args.save_ref_indices_dir
            partition_subdir = f"{args.dataset}_{args.partition}_{args.overlap_ratio}"
            emb_subdir_name = f"{args.emb1}_{args.emb2}_pred_ind"
            ref_indices_dir = os.path.join(base_dir, partition_subdir, emb_subdir_name)
        else:
            # Default: create subdirectory under the same directory where ind1.npy and ind2.npy are stored
            # Directory structure: cache/ind/{dataset}_{partition}_{overlap_ratio}/{emb1}_{emb2}_pred_ind/
            emb_subdir_name = f"{args.emb1}_{args.emb2}_pred_ind"
            ref_indices_dir = os.path.join(ind_file_name, emb_subdir_name)

        os.makedirs(ref_indices_dir, exist_ok=True)

        # Create filename based on method
        method_str = "bernoulli"
        anchor_str = args.anchor_mode
        # Use n_seeds in filename if specified, otherwise use ref_ratio
        if getattr(args, 'n_seeds', None) is not None:
            filename = f"{anchor_str}_{method_str}_n{args.n_seeds}_pred_ind"
        else:
            filename = f"{anchor_str}_{method_str}_ref{args.ref_ratio}_pred_ind"

        # Save as .npy files
        if args.anchor_mode == "supervised":
            ref_indices1 = np.concatenate([ref_indices1, ref_ind], axis=0)
            ref_indices2 = np.concatenate([ref_indices2, ref_ind], axis=0)

        ref_indices1_path = os.path.join(ref_indices_dir, f"{filename}1.npy")
        ref_indices2_path = os.path.join(ref_indices_dir, f"{filename}2.npy")

        np.save(ref_indices1_path, ref_indices1)
        np.save(ref_indices2_path, ref_indices2)

        logger.debug(f"Saved ref_indices to {ref_indices1_path} and {ref_indices2_path}")
        logger.debug(f"ref_indices1 shape: {ref_indices1.shape}, ref_indices2 shape: {ref_indices2.shape}")

    return accuracy, recall, ref_indices1, ref_indices2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="scifact")
    parser.add_argument("--ref_dataset", type=str, default="fiqa")
    parser.add_argument("--ref_ratio", type=float, default=0.01)
    parser.add_argument("--n_seeds", type=int, default=None,
                        help="Fixed number of seed pairs. If provided, overrides ref_ratio.")
    parser.add_argument("--base_dir", type=str, default="embeddings")
    parser.add_argument("--cache_dir", type=str, default="cache/ind")
    parser.add_argument("--emb1", type=str, default="mistral")
    parser.add_argument("--emb2", type=str, default="openai")
    parser.add_argument("--emb_dim1", type=int, default=768)
    parser.add_argument("--emb_dim2", type=int, default=768)
    
    parser.add_argument("--partition", type=str, default="random")
    parser.add_argument("--overlap_ratio", type=float, default=0.3)
    parser.add_argument("--nonref_clu_choices", type=int, nargs='+', default=[0])
    parser.add_argument("--n_clusters", type=int, default=10)
    parser.add_argument("--n_clusters_overlap", type=int, default=20)

    parser.add_argument("--csls_neighborhood", type=int, default=50, help="use CSLS for dictionary induction")
    parser.add_argument("--cluster_method", type=str, default="kmeans")
    parser.add_argument("--distance_metric", type=str, default="cosine")
    parser.add_argument("--ref_method", type=str, default="random")
    parser.add_argument("--graph_method", type=str, default="knn")
    parser.add_argument("--knn_k", type=int, default=500)
    parser.add_argument("--topk", type=int, default=5,
                        help="Top-k neighbors used by find_mutual_pairs in utils/retrieval_util.py")
    parser.add_argument("--k_neighbors", type=int, default=5000,
                        help="Local neighborhood size per anchor for large datasets (default: 5000)")
    parser.add_argument("--per_view_k", type=int, default=200,
                        help="k-NN per ref point per view in per-view neighborhood mode (default: 200)")
    def str2bool(x):
        """Parse boolean CLI arguments."""
        return x.lower() not in ('false', '0', 'no', 'n')
    parser.add_argument("--use_gpu", type=str2bool, default=True)
    parser.add_argument("--multi_gpu_chunk_size", type=int, default=None,
                        help="Rows per GPU chunk when using multi-GPU distance computations (default: auto)")
    parser.add_argument("--sample", type=str2bool, default=False)

    # Anchor generation arguments
    parser.add_argument("--anchor_mode", type=str, default="supervised", choices=["supervised", "ood"],
                        help="Mode for anchor generation: supervised (original) or ood (out-of-distribution)")
    parser.add_argument("--concat_seed_pairs", type=str2bool, default=False,
                        help="Whether to concatenate initial seed pairs to reference embeddings in supervised/OOD mode")

    # Ensemble reference selection parameters
    parser.add_argument("--ensemble_n_ensembles", type=int, default=5,
                        help="Number of ensemble runs for reference selection (default: auto based on ref/subset sizes)")
    parser.add_argument("--ensemble_subset_ratio", type=float, default=0.4, help="Base per-view anchor fraction ρ0 (default: 0.4)")
    parser.add_argument("--schedule_c", type=float, default=0.3, help="View-schedule growth constant c in sf_t = 1 + c·log(g_t) (paper Appendix B.2, default: 0.3)")
    parser.add_argument("--max_parallel_workers", type=int, default=None, help="Max parallel workers for ensemble (None=auto, 2=recommended for large datasets like scidocs/fiqa)")
    parser.add_argument("--ensemble_vote_threshold", type=float, default=0.6, help="Vote threshold for ensemble selection (0.0=all pairs from any ensemble, 0.6=majority, 1.0=unanimous)")
    parser.add_argument("--ensemble_strategy", type=str, default="furthest", choices=['random', 'cluster', 'furthest', 'nearest'], help="Ensemble strategy for reference selection: furthest (default, dispersed anchors), random, cluster (localized anchors), or nearest (local neighborhoods)")
    
    # Training control parameters
    parser.add_argument("--max_iter", type=int, default=100, help="Maximum number of iterations")
    parser.add_argument("--max_refs", type=int, default=None,
                        help="Maximum number of reference points to keep. Caps the reference set to prevent OOM "
                             "from growing distance matrices. If None, auto-determined based on available memory.")
    parser.add_argument("--large_dataset_threshold", type=int, default=500_000,
                        help="N_unique threshold above which warm-start and local-neighborhood search are enabled (default: 500000)")
    parser.add_argument("--warmstart_size", type=int, default=0,
                        help="Subsample size for warm-start phase on large datasets (0=disabled, default: 0)")
    parser.add_argument("--warmstart_iters", type=int, default=3,
                        help="Number of iterations for warm-start phase (default: 3)")

    # Procrustes refinement parameters
    parser.add_argument("--use_procrustes", action="store_true", help="Apply orthogonal Procrustes transformation after finding mutual NNs to align embedding spaces")

    # Reference filtering parameters
    parser.add_argument("--enable_ref_filtering", type=str2bool, default=False, help="Enable reference filtering based on distance quality")
    parser.add_argument("--ref_filter_ratio", type=float, default=0.9, help="Keep top fraction of references (0.8 = keep top 80%%)")
    
    # Reference filtering annealing parameters
    parser.add_argument("--ref_filter_annealing", type=str, default="quality_adaptive", 
                        choices=["none", "linear", "exponential", "cosine", "quality_adaptive"], 
                        help="Annealing strategy for ref_filter_ratio")
    # Bernoulli posterior ensemble selection parameters
    parser.add_argument("--posterior_threshold", type=float, default=0.1, help="Posterior threshold for Bernoulli trial-based ensemble selection")
    parser.add_argument("--overlap_inference_method", type=str, default="otsu",
                        choices=["threshold", "adaptive", "otsu", "gmm", "elbow", "expected", "gap"],
                        help="Method to infer overlapping pairs from posterior distribution: "
                             "'threshold' (default) uses fixed/iteration-based threshold, "
                             "'adaptive' combines multiple methods (recommended for unknown overlap), "
                             "'otsu' uses Otsu's thresholding, 'gmm' uses Gaussian Mixture Model, "
                             "'elbow' uses elbow/knee detection, 'expected' uses sum of posteriors, "
                             "'gap' uses gap statistic")

    # Precision control
    parser.add_argument("--fp16", type=str2bool, default=True,
                        help="Use FP16 for GPU k-NN search and ensemble computation (default: True). Set False for FP32.")

    parser.add_argument("--save_ref_indices", action="store_true", help="Save final reference indices to disk")
    parser.add_argument("--save_ref_indices_dir", type=str, default=None, help="Custom directory to save reference indices (overrides default cache location)")

    # Seed parameter for reproducibility
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility. If set, creates seed-specific cache files.")

    # Experiment output
    parser.add_argument("--experiment_csv", type=str, default=None,
                        help="Path to CSV file for appending experiment summary (precision, recall, runtime, memory)")

    # Debug logging
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging (loguru output)")

    args = parser.parse_args()

    # Configure loguru log level
    logger.remove()  # Remove default handler
    if args.debug:
        logger.add(sys.stderr, level="DEBUG")
    else:
        # Only show INFO from test_clu (iteration accuracy line), plus WARNING+, plus tile/ensemble INFO
        logger.add(
            sys.stderr,
            level="INFO",
            filter=lambda record: record["level"].no >= 30 or record["function"] == "test_clu" or record["function"] == "<module>" or "tile" in record["message"].lower() or "Tiling" in record["message"] or "Ensemble " in record["message"] or "spatial" in record["message"].lower()
        )

    faulthandler.enable()
    try:
        # Use seed from args if provided
        seed = args.seed
        accuracy, recall, ref_indices1, ref_indices2 = test_clu(args, seed=seed)
        logger.info(f"Final results: accuracy={accuracy:.4f}, recall={recall:.4f}")

        # Write experiment summary to CSV if --experiment_csv is set
        if getattr(args, 'experiment_csv', None):
            import csv
            csv_path = args.experiment_csv
            write_header = not os.path.exists(csv_path)
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(['precision', 'recall', 'runtime_s',
                                     'peak_gpu_mb', 'peak_cpu_rss_mb', 'n_seeds', 'overlap_ratio', 'seed'])
                writer.writerow([
                    f"{accuracy:.6f}",
                    f"{recall:.6f}",
                    f"{getattr(args, '_runtime_s', -1):.1f}",
                    f"{getattr(args, '_peak_gpu_mb', -1):.0f}",
                    f"{getattr(args, '_peak_cpu_rss_mb', -1):.0f}",
                    args.n_seeds,
                    args.overlap_ratio,
                    seed,
                ])
            logger.info(f"Experiment results appended to {csv_path}")
    except Exception as e:
        logger.error("Exception during test_clu execution")
        traceback.print_exc()
        sys.exit(1)
