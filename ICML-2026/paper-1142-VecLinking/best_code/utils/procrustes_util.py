"""
Procrustes Analysis Utilities

Provides functions for computing orthogonal Procrustes transformations
to align embedding spaces, with support for cluster-wise transformations.
"""

import numpy as np
import torch
from typing import Tuple, Dict, Optional, Union
from scipy.linalg import orthogonal_procrustes
from loguru import logger
from utils.dimension_alignment import align_dimensions
from utils.retrieval_util import deduplicate_pairs


def compute_procrustes_transform_gpu(
    X: Union[np.ndarray, torch.Tensor],
    Y: Union[np.ndarray, torch.Tensor],
    device: torch.device,
    allow_scale: bool = False,
    allow_translation: bool = False
) -> Dict[str, torch.Tensor]:
    """
    GPU-accelerated Procrustes transformation using PyTorch SVD.

    Solves: min ||Y - XR||_F subject to R'R = I
    Solution: R = V @ U.T where U, S, Vh = svd(X.T @ Y)

    Args:
        X: Source embeddings (n_pairs, d) - numpy array or torch tensor
        Y: Target embeddings (n_pairs, d) - numpy array or torch tensor
        device: GPU device to use
        allow_scale: If True, allow uniform scaling
        allow_translation: If True, allow translation

    Returns:
        Dictionary containing:
            - 'rotation': Rotation matrix (d, d) as torch tensor
            - 'scale': Scaling factor as torch tensor
            - 'translation': Translation vector (d,) as torch tensor
    """
    # Convert to tensors if needed
    if not torch.is_tensor(X):
        X = torch.tensor(X, dtype=torch.float32, device=device)
    else:
        X = X.to(device=device, dtype=torch.float32)
    if not torch.is_tensor(Y):
        Y = torch.tensor(Y, dtype=torch.float32, device=device)
    else:
        Y = Y.to(device=device, dtype=torch.float32)

    d = X.shape[1]

    # Center if translation allowed
    if allow_translation:
        X_mean = X.mean(dim=0)
        Y_mean = Y.mean(dim=0)
        X_centered = X - X_mean
        Y_centered = Y - Y_mean
        translation = Y_mean - X_mean
    else:
        X_centered = X
        Y_centered = Y
        translation = torch.zeros(d, device=device, dtype=torch.float32)

    # Compute optimal rotation R such that X @ R ≈ Y
    n_pairs = X_centered.shape[0]

    if n_pairs < d // 2:
        # Low-rank optimization: when n_pairs << d, work in the combined subspace
        # M = X.T @ Y has rank ≤ n_pairs, so full SVD of (d,d) matrix is wasteful
        # Instead, compute SVD in the subspace spanned by X and Y

        # Stack X and Y to find their combined column space
        Z = torch.cat([X_centered, Y_centered], dim=0)  # (2n, d)

        # QR decomposition to get orthonormal basis for column space
        # Q has shape (d, min(2n, d)) with orthonormal columns
        Q, _ = torch.linalg.qr(Z.T)  # (d, 2n) if 2n < d
        k = Q.shape[1]  # Actual rank of subspace

        # Project X and Y onto this subspace
        X_proj = X_centered @ Q  # (n, k)
        Y_proj = Y_centered @ Q  # (n, k)

        # Procrustes in reduced k-dimensional space (k ≤ 2n << d)
        M_reduced = X_proj.T @ Y_proj  # (k, k)
        U_r, S_r, Vh_r = torch.linalg.svd(M_reduced, full_matrices=False)
        R_reduced = U_r @ Vh_r  # (k, k) orthogonal rotation in subspace

        # Extend to full space: rotate within subspace, identity on orthogonal complement
        # R_full = Q @ R_reduced @ Q.T + (I - Q @ Q.T)
        #        = I + Q @ (R_reduced - I_k) @ Q.T
        R = torch.eye(d, device=device, dtype=torch.float32) + Q @ (R_reduced - torch.eye(k, device=device, dtype=torch.float32)) @ Q.T
    else:
        # Original full SVD for when d is small or n_pairs is large
        M = X_centered.T @ Y_centered
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)
        R = U @ Vh

    # Compute scale if allowed
    if allow_scale:
        X_rotated = X_centered @ R
        numerator = (Y_centered * X_rotated).sum()
        denominator = (X_rotated * X_rotated).sum()
        scale = numerator / denominator if denominator > 0 else torch.tensor(1.0, device=device)
    else:
        scale = torch.tensor(1.0, device=device, dtype=torch.float32)

    return {
        'rotation': R,
        'scale': scale,
        'translation': translation
    }


def compute_procrustes_transform(
    X: np.ndarray,
    Y: np.ndarray,
    allow_scale: bool = False,
    allow_translation: bool = False,
    use_gpu: bool = False,
    device: torch.device = None
) -> Dict[str, np.ndarray]:
    """
    Compute Procrustes transformation to align X to Y.

    Args:
        X: Source embeddings (n_pairs, d)
        Y: Target embeddings (n_pairs, d)
        allow_scale: If True, allow uniform scaling
        allow_translation: If True, allow translation
        use_gpu: If True, use GPU-accelerated computation
        device: GPU device to use (required if use_gpu=True)

    Returns:
        Dictionary containing:
            - 'rotation': Rotation/reflection matrix (d, d)
            - 'scale': Scaling factor (scalar)
            - 'translation': Translation vector (d,)
    """

    if len(X) == 0 or len(Y) == 0:
        raise ValueError("Cannot compute Procrustes with empty arrays")

    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"X and Y must have same number of points: {X.shape[0]} vs {Y.shape[0]}")

    # Use GPU path if requested and available
    if use_gpu and torch.cuda.is_available():
        if device is None:
            device = torch.device('cuda')

        result = compute_procrustes_transform_gpu(
            X, Y, device,
            allow_scale=allow_scale,
            allow_translation=allow_translation
        )

        # Convert back to numpy
        return {
            'rotation': result['rotation'].cpu().numpy(),
            'scale': float(result['scale'].cpu().item()) if torch.is_tensor(result['scale']) else float(result['scale']),
            'translation': result['translation'].cpu().numpy()
        }

    # CPU path using scipy
    # Center the data if translation is allowed
    if allow_translation:
        X_mean = X.mean(axis=0)
        Y_mean = Y.mean(axis=0)
        X_centered = X - X_mean
        Y_centered = Y - Y_mean
        translation = Y_mean - X_mean
    else:
        X_centered = X.copy()
        Y_centered = Y.copy()
        translation = np.zeros(X.shape[1])

    # Compute optimal rotation using scipy's orthogonal_procrustes
    # This solves: min ||Y_centered - X_centered @ R||_F
    R, _ = orthogonal_procrustes(X_centered, Y_centered)

    # Compute optimal scale if allowed
    if allow_scale:
        X_rotated = X_centered @ R
        scale = np.sum(Y_centered * X_rotated) / np.sum(X_rotated * X_rotated)
    else:
        scale = 1.0

    return {
        'rotation': R,
        'scale': scale,
        'translation': translation
    }


def apply_procrustes_transform(
    X: np.ndarray,
    transform: Dict[str, np.ndarray]
) -> np.ndarray:
    """
    Apply Procrustes transformation to embeddings.

    Args:
        X: Embeddings to transform (n, d)
        transform: Transform dictionary from compute_procrustes_transform

    Returns:
        Transformed embeddings (n, d)
    """
    # Apply: X_transformed = scale * X @ R + translation
    X_transformed = X @ transform['rotation']
    X_transformed = X_transformed * transform['scale']
    X_transformed = X_transformed + transform['translation']

    return X_transformed


def cluster_wise_procrustes_refinement(
    emb1: np.ndarray,
    emb2: np.ndarray,
    ind_emb1: np.ndarray,
    ind_emb2: np.ndarray,
    mutual_pairs_ind1: np.ndarray,
    mutual_pairs_ind2: np.ndarray,
    cluster_labels: np.ndarray,
    find_mutual_pairs_fn,
    allow_scale: bool = False,
    allow_translation: bool = False,
    min_pairs_per_cluster: int = 3,
    verbose: bool = True,
    use_gpu: bool = True,
    device: torch.device = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply cluster-wise Procrustes transformations and find refined mutual NNs.

    Workflow:
    1. Align dimensions if needed using zero padding
    2. For each cluster in emb1:
       a. Find mutual NN pairs in this cluster
       b. Compute Procrustes transform using these pairs
       c. Transform all points in the cluster
       d. Re-compute mutual NNs in transformed space
    3. Return refined mutual NN pairs (original embeddings unchanged)

    Args:
        emb1: First embeddings (n, d)
        emb2: Second embeddings (m, d)
        ind_emb1: Original indices for emb1 (n,)
        ind_emb2: Original indices for emb2 (m,)
        mutual_pairs_ind1: Original indices from emb1 in mutual pairs
        mutual_pairs_ind2: Original indices from emb2 in mutual pairs
        cluster_labels: Cluster labels for emb1 (n,)
        find_mutual_pairs_fn: Function to find mutual pairs (e.g., from retrieval_util)
        allow_scale: Allow scaling in Procrustes
        allow_translation: Allow translation in Procrustes
        min_pairs_per_cluster: Minimum pairs needed to compute transform
        verbose: Print progress
        use_gpu: Use GPU-accelerated Procrustes computation
        device: GPU device to use (auto-detected if None)

    Returns:
        (refined_ind1, refined_ind2, emb1_transformed):
            - refined_ind1: Original indices from emb1 in refined mutual pairs
            - refined_ind2: Original indices from emb2 in refined mutual pairs
            - emb1_transformed: Transformed emb1 (n, d)
    """
    # Initialize GPU device if use_gpu is enabled and device not provided
    if use_gpu and device is None and torch.cuda.is_available():
        device = torch.device('cuda')

    # Align dimensions if needed using zero padding
    if emb1.shape[1] != emb2.shape[1]:
        if verbose:
            logger.debug(f"Aligning dimensions using zero padding: emb1={emb1.shape[1]}, emb2={emb2.shape[1]}")
        emb1_work, emb2_work = align_dimensions(emb1, emb2, method="padding", padding_mode="zero")
        if verbose:
            logger.debug(f"After alignment: emb1={emb1_work.shape[1]}, emb2={emb2_work.shape[1]}")
    else:
        # No copy needed when dimensions match - use original arrays
        # We only read from these, never modify them
        emb1_work = emb1
        emb2_work = emb2

    unique_clusters = np.unique(cluster_labels)
    emb1_transformed = emb1_work.copy()

    all_refined_pairs = []

    # Build reverse index mappings ONCE: original_index -> position_in_array
    # This enables O(1) lookup instead of O(n) np.where() search per pair
    ind_emb1_to_pos = {orig_idx: pos for pos, orig_idx in enumerate(ind_emb1)}
    ind_emb2_to_pos = {orig_idx: pos for pos, orig_idx in enumerate(ind_emb2)}

    if verbose:
        logger.debug(f"Processing {len(unique_clusters)} clusters for Procrustes refinement")

    # Pre-compute cluster assignments for O(1) lookup instead of O(n*m) np.isin() per cluster
    # Build mapping: original_index -> cluster_id
    orig_idx_to_cluster = {}
    for cluster_id in unique_clusters:
        cluster_mask = cluster_labels == cluster_id
        for orig_idx in ind_emb1[cluster_mask]:
            orig_idx_to_cluster[orig_idx] = cluster_id

    # Pre-group mutual pairs by cluster (O(n) total, not O(n*m) per cluster)
    pairs_by_cluster = {c: ([], []) for c in unique_clusters}
    for idx1, idx2 in zip(mutual_pairs_ind1, mutual_pairs_ind2):
        if idx1 in orig_idx_to_cluster:
            c = orig_idx_to_cluster[idx1]
            pairs_by_cluster[c][0].append(idx1)
            pairs_by_cluster[c][1].append(idx2)

    if verbose:
        logger.debug(f"Pre-grouped {len(mutual_pairs_ind1)} pairs into {len(unique_clusters)} clusters")

    import time as time_module
    for cluster_idx, cluster_id in enumerate(unique_clusters):
        cluster_start_time = time_module.time()
        if verbose:
            logger.debug(f"Starting cluster {cluster_id} ({cluster_idx + 1}/{len(unique_clusters)})...")

        # Get points in this cluster
        cluster_mask = cluster_labels == cluster_id
        cluster_indices_in_emb1 = np.where(cluster_mask)[0]
        cluster_orig_indices = ind_emb1[cluster_indices_in_emb1]

        # Find mutual pairs that belong to this cluster - O(1) lookup from pre-computed dict
        cluster_pairs_ind1 = np.array(pairs_by_cluster[cluster_id][0], dtype=np.int64)
        cluster_pairs_ind2 = np.array(pairs_by_cluster[cluster_id][1], dtype=np.int64)

        if len(cluster_pairs_ind1) < min_pairs_per_cluster:
            if verbose:
                logger.debug(f"Cluster {cluster_id}: Only {len(cluster_pairs_ind1)} pairs, skipping Procrustes")
            continue

        # Convert original indices to positions in emb1 and emb2
        # Use O(1) dictionary lookup instead of O(n) np.where() search
        pairs_pos_in_emb1 = np.array([ind_emb1_to_pos[idx] for idx in cluster_pairs_ind1], dtype=np.int64)
        pairs_pos_in_emb2 = np.array([ind_emb2_to_pos[idx] for idx in cluster_pairs_ind2], dtype=np.int64)

        # Get paired embeddings for Procrustes (use aligned versions)
        X_pairs = emb1_work[pairs_pos_in_emb1]
        Y_pairs = emb2_work[pairs_pos_in_emb2]

        try:
            # Compute Procrustes transformation for this cluster
            transform = compute_procrustes_transform(
                X_pairs, Y_pairs,
                allow_scale=allow_scale,
                allow_translation=allow_translation,
                use_gpu=use_gpu,
                device=device
            )

            # Apply transformation to ALL points in this cluster
            X_cluster = emb1_work[cluster_indices_in_emb1]
            X_cluster_transformed = apply_procrustes_transform(X_cluster, transform)
            emb1_transformed[cluster_indices_in_emb1] = X_cluster_transformed

            # Find mutual NNs for transformed cluster using existing function
            # The provided function is already optimized with chunking, FAISS, etc.
            cluster_mutual_pairs = find_mutual_pairs_fn(
                X_cluster_transformed,
                emb2_work,
                ind_emb1[cluster_indices_in_emb1],
                ind_emb2
            )

            # Add to refined pairs
            all_refined_pairs.extend(cluster_mutual_pairs)

            if verbose:
                elapsed = time_module.time() - cluster_start_time
                logger.debug(f"Cluster {cluster_id}: {len(cluster_indices_in_emb1)} points, "
                          f"{len(cluster_pairs_ind1)} pairs for Procrustes, "
                          f"{len(cluster_mutual_pairs)} refined mutual pairs (took {elapsed:.1f}s)")

        except Exception as e:
            logger.warning(f"Cluster {cluster_id}: Failed to compute Procrustes: {e}")
            continue

    # Memory cleanup after all clusters (instead of per-cluster to reduce overhead)
    if use_gpu and torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Convert refined pairs to arrays
    if len(all_refined_pairs) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), emb1_transformed

    all_refined_pairs = np.array(all_refined_pairs)
    refined_ind1 = all_refined_pairs[:, 0]
    refined_ind2 = all_refined_pairs[:, 1]

    # Build maps from original indices to positions in current embedding arrays
    orig_to_pos_emb1 = {orig_idx: pos for pos, orig_idx in enumerate(ind_emb1)}
    orig_to_pos_emb2 = {orig_idx: pos for pos, orig_idx in enumerate(ind_emb2)}

    # Compute distances for each pair using transformed emb1 and aligned emb2
    pair_with_dist = []
    for i1, i2 in zip(refined_ind1, refined_ind2):
        if i1 in orig_to_pos_emb1 and i2 in orig_to_pos_emb2:
            p1 = orig_to_pos_emb1[i1]
            p2 = orig_to_pos_emb2[i2]
            dist = float(np.linalg.norm(emb1_transformed[p1] - emb2_work[p2]))
            pair_with_dist.append((int(i1), int(i2), dist))

    # Use retrieval_util's deduplication to keep the shortest-distance pairs (unique on both sides)
    deduped = deduplicate_pairs(pair_with_dist)

    if len(deduped) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), emb1_transformed

    refined_ind1 = np.array([p[0] for p in deduped], dtype=np.int64)
    refined_ind2 = np.array([p[1] for p in deduped], dtype=np.int64)

    if verbose:
        logger.debug(f"Found {len(refined_ind1)} unique refined mutual NN pairs after cluster-wise Procrustes")

    return refined_ind1, refined_ind2, emb1_transformed
