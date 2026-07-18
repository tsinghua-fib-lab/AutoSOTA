import numpy as np
import torch
from loguru import logger
from typing import Union, Optional
from utils.clustering import Clusterer


def sample_cluster_centroids(
    embeddings: Union[np.ndarray, torch.Tensor],
    candidate_indices: np.ndarray,
    ref_ratio: float = 0.1,
    n_clusters: Optional[int] = None,
    use_gpu: bool = False,
    n_samples: Optional[int] = None,
    **kwargs
) -> np.ndarray:
    """
    Sample reference points by clustering and selecting points closest to centroids.

    This method uses k-means clustering to partition the candidate points into clusters,
    then selects the point closest to each cluster centroid as a reference point.

    Algorithm:
    1. Extract embeddings for candidate indices
    2. Run k-means clustering with k determined by n_samples or ref_ratio
    3. For each cluster, find the point closest to the centroid
    4. Return indices of these representative points

    Args:
        embeddings: Full embedding matrix (N, D)
        candidate_indices: Indices of candidate points to sample from
        ref_ratio: Ratio of candidate points to sample (default: 0.1)
        n_clusters: Number of clusters (if None, computed from n_samples or ref_ratio)
        use_gpu: Use GPU acceleration for clustering (default: False)
        n_samples: Fixed number of samples. If provided, overrides ref_ratio.
        **kwargs: Additional arguments (ignored for compatibility)

    Returns:
        Array of sampled reference point indices

    Example:
        >>> ref_indices = sample_cluster_centroids(
        ...     embeddings, candidate_indices, ref_ratio=0.1
        ... )
    """
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()

    # Extract embeddings for candidate indices
    candidate_embs = embeddings[candidate_indices]
    n_candidates = len(candidate_indices)

    # Determine number of clusters from n_samples, n_clusters, or ref_ratio
    if n_samples is not None:
        # Use n_samples directly as number of clusters
        n_clusters = min(max(1, n_samples), n_candidates)
        logger.debug(f"Cluster centroids sampling: {n_clusters} clusters from {n_candidates} candidates (fixed n_seeds={n_samples})")
    elif n_clusters is None:
        n_clusters = max(1, int(n_candidates * ref_ratio))
        # Ensure n_clusters doesn't exceed number of candidates
        n_clusters = min(n_clusters, n_candidates)
        logger.debug(f"Cluster centroids sampling: {n_clusters} clusters from {n_candidates} candidates (ratio={ref_ratio:.3f})")
    else:
        # n_clusters was explicitly provided
        n_clusters = min(n_clusters, n_candidates)
        logger.debug(f"Cluster centroids sampling: {n_clusters} clusters from {n_candidates} candidates (explicit n_clusters)")

    # Perform k-means clustering
    clusterer = Clusterer(method='kmeans', n_clusters=n_clusters, use_gpu=use_gpu)
    cluster_labels = clusterer.fit(candidate_embs)

    # Get cluster centers from the fitted model
    if isinstance(clusterer.model, dict) and clusterer.model.get('type') == 'pytorch':
        # GPU/PyTorch model
        centroids = clusterer.model['centroids'].cpu().numpy()
    else:
        # CPU/sklearn model
        centroids = clusterer.model['sklearn_model'].cluster_centers_

    # For each cluster, find the point closest to the centroid
    ref_points = []
    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_member_indices = np.where(cluster_mask)[0]

        if len(cluster_member_indices) == 0:
            logger.warning(f"Cluster {cluster_id} is empty, skipping")
            continue

        # Get embeddings of cluster members
        cluster_member_embs = candidate_embs[cluster_member_indices]

        # Find the member closest to centroid
        centroid = centroids[cluster_id]
        distances = np.linalg.norm(cluster_member_embs - centroid, axis=1)
        closest_idx = cluster_member_indices[np.argmin(distances)]

        # Map back to original indices
        ref_points.append(candidate_indices[closest_idx])

    ref_points = np.array(ref_points)

    logger.debug(f"Cluster centroids sampling: selected {len(ref_points)} reference points")

    return ref_points


def sample_furthest_point(
    embeddings: Union[np.ndarray, torch.Tensor],
    candidate_indices: np.ndarray,
    ref_ratio: float = 0.1,
    metric: str = "euclidean",
    use_gpu: bool = False,
    n_samples: Optional[int] = None,
    **kwargs
) -> np.ndarray:
    """
    Sample reference points using Furthest Point Sampling (FPS).

    Furthest Point Sampling is a greedy algorithm that iteratively selects points
    that are maximally separated from all previously selected points. This produces
    a diverse and well-distributed set of reference points.

    Algorithm:
    1. Start with a random point
    2. Iteratively select the point that is farthest from all previously selected points
    3. Continue until n_samples (or ref_ratio * len(candidate_indices)) points are selected

    Args:
        embeddings: Full embedding matrix (N, D)
        candidate_indices: Indices of candidate points to sample from
        ref_ratio: Ratio of candidate points to sample (default: 0.1)
        metric: Distance metric ('euclidean' or 'cosine', default: 'euclidean')
        use_gpu: Use GPU acceleration (default: False)
        n_samples: Fixed number of samples. If provided, overrides ref_ratio.
        **kwargs: Additional arguments (ignored for compatibility)

    Returns:
        Array of sampled reference point indices

    Example:
        >>> ref_indices = sample_furthest_point(
        ...     embeddings, candidate_indices, ref_ratio=0.1, metric='euclidean'
        ... )
    """
    if isinstance(embeddings, torch.Tensor):
        embeddings_np = embeddings.cpu().numpy()
    else:
        embeddings_np = embeddings

    # Extract embeddings for candidate indices
    candidate_embs = embeddings_np[candidate_indices]
    n_candidates = len(candidate_indices)

    # Determine number of points to sample from n_samples or ref_ratio
    if n_samples is not None:
        n_to_sample = min(max(1, n_samples), n_candidates)
        logger.debug(f"Furthest point sampling: selecting {n_to_sample} from {n_candidates} candidates (fixed n_seeds={n_samples}, metric={metric})")
    else:
        n_to_sample = max(1, int(n_candidates * ref_ratio))
        n_to_sample = min(n_to_sample, n_candidates)
        logger.debug(f"Furthest point sampling: selecting {n_to_sample} from {n_candidates} candidates (ratio={ref_ratio:.3f}, metric={metric})")
    
    # Use n_to_sample for the rest of the function
    n_samples = n_to_sample

    # Initialize with a random point
    selected_mask = np.zeros(n_candidates, dtype=bool)
    first_idx = np.random.randint(0, n_candidates)
    selected_mask[first_idx] = True
    selected_indices = [first_idx]

    # Use GPU if available and requested
    if use_gpu and torch.cuda.is_available():
        try:
            device = torch.device("cuda")
            candidate_embs_gpu = torch.from_numpy(candidate_embs).float().to(device)

            # Normalize for cosine distance
            if metric == "cosine":
                candidate_embs_gpu = torch.nn.functional.normalize(candidate_embs_gpu, dim=1)

            # Initialize distances to infinity
            min_distances = torch.full((n_candidates,), float('inf'), device=device)

            # Update distances with first point
            first_point = candidate_embs_gpu[first_idx:first_idx+1]
            if metric == "cosine":
                # Cosine distance = 1 - cosine similarity
                distances = 1.0 - torch.mm(candidate_embs_gpu, first_point.T).squeeze()
            else:  # euclidean
                distances = torch.norm(candidate_embs_gpu - first_point, dim=1)
            min_distances = torch.minimum(min_distances, distances)

            # Iteratively select furthest points
            for _ in range(1, n_samples):
                # Mask out already selected points
                min_distances_masked = min_distances.clone()
                min_distances_masked[selected_mask] = -float('inf')

                # Select the furthest point
                furthest_idx = torch.argmax(min_distances_masked).item()
                selected_mask[furthest_idx] = True
                selected_indices.append(furthest_idx)

                # Update distances
                new_point = candidate_embs_gpu[furthest_idx:furthest_idx+1]
                if metric == "cosine":
                    distances = 1.0 - torch.mm(candidate_embs_gpu, new_point.T).squeeze()
                else:  # euclidean
                    distances = torch.norm(candidate_embs_gpu - new_point, dim=1)
                min_distances = torch.minimum(min_distances, distances)

            # Clean up GPU memory
            del candidate_embs_gpu, min_distances
            torch.cuda.empty_cache()

        except Exception as e:
            logger.warning(f"GPU FPS failed: {e}, falling back to CPU")
            use_gpu = False

    if not use_gpu or not torch.cuda.is_available():
        # CPU implementation
        # Normalize for cosine distance
        if metric == "cosine":
            norms = np.linalg.norm(candidate_embs, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            candidate_embs_norm = candidate_embs / norms

        # Initialize distances to infinity
        min_distances = np.full(n_candidates, float('inf'))

        # Update distances with first point
        if metric == "cosine":
            distances = 1.0 - np.dot(candidate_embs_norm, candidate_embs_norm[first_idx])
        else:  # euclidean
            distances = np.linalg.norm(candidate_embs - candidate_embs[first_idx], axis=1)
        min_distances = np.minimum(min_distances, distances)

        # Iteratively select furthest points
        for _ in range(1, n_samples):
            # Mask out already selected points
            min_distances_masked = min_distances.copy()
            min_distances_masked[selected_mask] = -float('inf')

            # Select the furthest point
            furthest_idx = np.argmax(min_distances_masked)
            selected_mask[furthest_idx] = True
            selected_indices.append(furthest_idx)

            # Update distances
            if metric == "cosine":
                distances = 1.0 - np.dot(candidate_embs_norm, candidate_embs_norm[furthest_idx])
            else:  # euclidean
                distances = np.linalg.norm(candidate_embs - candidate_embs[furthest_idx], axis=1)
            min_distances = np.minimum(min_distances, distances)

    # Map back to original indices
    ref_points = candidate_indices[np.array(selected_indices)]

    logger.debug(f"Furthest point sampling: selected {len(ref_points)} reference points")

    return ref_points


def sample_random(
    embeddings: Union[np.ndarray, torch.Tensor],
    candidate_indices: np.ndarray,
    ref_ratio: float = 0.1,
    seed: Optional[int] = None,
    n_samples: Optional[int] = None,
    **kwargs
) -> np.ndarray:
    """
    Sample reference points randomly from candidate indices.

    This is the simplest sampling method, which randomly selects a subset
    of candidate points without replacement.

    Args:
        embeddings: Full embedding matrix (N, D) - not used but kept for API consistency
        candidate_indices: Indices of candidate points to sample from
        ref_ratio: Ratio of candidate points to sample (default: 0.1)
        seed: Random seed for reproducibility (default: None)
        n_samples: Fixed number of samples. If provided, overrides ref_ratio.
        **kwargs: Additional arguments (ignored for compatibility)

    Returns:
        Array of sampled reference point indices

    Example:
        >>> ref_indices = sample_random(
        ...     embeddings, candidate_indices, ref_ratio=0.1, seed=42
        ... )
    """
    n_candidates = len(candidate_indices)
    if n_samples is not None:
        n_samples = min(max(1, n_samples), n_candidates)
        logger.debug(f"Random sampling: selecting {n_samples} from {n_candidates} candidates (fixed n_seeds)")
    else:
        n_samples = max(1, int(n_candidates * ref_ratio))
        n_samples = min(n_samples, n_candidates)
        logger.debug(f"Random sampling: selecting {n_samples} from {n_candidates} candidates (ratio={ref_ratio:.3f})")

    # Set seed if provided
    if seed is not None:
        np.random.seed(seed)

    # Random sampling without replacement
    ref_points = np.random.choice(candidate_indices, size=n_samples, replace=False)

    logger.debug(f"Random sampling: selected {len(ref_points)} reference points")

    return ref_points


def sample_localized_cluster(
    embeddings: Union[np.ndarray, torch.Tensor],
    candidate_indices: np.ndarray,
    ref_ratio: float = 0.1,
    n_clusters: Optional[int] = None,
    use_gpu: bool = False,
    n_samples: Optional[int] = None,
    **kwargs
) -> np.ndarray:
    """
    Sample reference points by selecting all points from a single localized cluster.

    This method maximizes localization by clustering the embeddings and selecting
    all points from the cluster whose size best matches the desired number of anchors.
    Unlike other methods that spread points across the embedding space, this method
    concentrates all reference points in a single dense region.

    Algorithm:
    1. Extract embeddings for candidate indices
    2. Determine desired number of samples from n_samples or ref_ratio
    3. Cluster candidates into k clusters
    4. Select the cluster whose size is closest to desired number
    5. Return all points from that cluster

    Args:
        embeddings: Full embedding matrix (N, D)
        candidate_indices: Indices of candidate points to sample from
        ref_ratio: Ratio of candidate points to sample (default: 0.1)
        n_clusters: Number of clusters (if None, auto-determined)
        use_gpu: Use GPU acceleration for clustering (default: False)
        n_samples: Fixed number of samples. If provided, overrides ref_ratio.
        **kwargs: Additional arguments (ignored for compatibility)

    Returns:
        Array of sampled reference point indices from a single localized cluster

    Example:
        >>> ref_indices = sample_localized_cluster(
        ...     embeddings, candidate_indices, ref_ratio=0.1
        ... )
    """
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()

    # Extract embeddings for candidate indices
    candidate_embs = embeddings[candidate_indices]
    n_candidates = len(candidate_indices)

    # Determine desired number of samples from n_samples or ref_ratio
    if n_samples is not None:
        n_desired = min(max(1, n_samples), n_candidates)
        logger.debug(f"Localized cluster sampling: target size {n_desired} (fixed n_seeds={n_samples})")
    else:
        n_desired = max(1, int(n_candidates * ref_ratio))
        logger.debug(f"Localized cluster sampling: target size {n_desired} (ratio={ref_ratio:.3f})")

    # Auto-determine number of clusters if not provided
    # Use more clusters to have better granularity for finding right-sized cluster
    if n_clusters is None:
        # Heuristic: create clusters of average size ~n_desired/2
        # This gives us flexibility to find a cluster close to n_desired
        n_clusters = max(3, min(n_candidates // max(1, n_desired // 2), 20))

    # Ensure n_clusters is valid
    n_clusters = min(n_clusters, n_candidates)

    logger.debug(f"Localized cluster sampling: {n_clusters} clusters from {n_candidates} candidates")

    # Perform k-means clustering
    clusterer = Clusterer(method='kmeans', n_clusters=n_clusters, use_gpu=use_gpu)
    cluster_labels = clusterer.fit(candidate_embs)

    # Count points in each cluster
    cluster_sizes = {}
    cluster_members = {}
    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_member_indices = np.where(cluster_mask)[0]
        cluster_sizes[cluster_id] = len(cluster_member_indices)
        cluster_members[cluster_id] = cluster_member_indices

    # Find cluster with size closest to desired number
    best_cluster_id = min(cluster_sizes.keys(),
                         key=lambda cid: abs(cluster_sizes[cid] - n_desired))
    best_cluster_size = cluster_sizes[best_cluster_id]

    logger.debug(f"Selected cluster {best_cluster_id} with {best_cluster_size} points "
                f"(target was {n_desired})")

    # Get all points from the selected cluster
    selected_member_indices = cluster_members[best_cluster_id]
    ref_points = candidate_indices[selected_member_indices]

    logger.debug(f"Localized cluster sampling: selected {len(ref_points)} reference points "
                f"from single cluster")

    return ref_points


def sample_nearest(
    embeddings: Union[np.ndarray, torch.Tensor],
    candidate_indices: np.ndarray,
    ref_ratio: float = 0.1,
    n_samples: Optional[int] = None,
    metric: str = "euclidean",
    use_gpu: bool = False,
    seed: Optional[int] = None,
    **kwargs
) -> np.ndarray:
    """
    Sample reference points by selecting k-nearest neighbors of a random seed point.

    This method creates a localized reference set by randomly selecting one point
    from the candidates, then finding its k-nearest neighbors. This produces a
    concentrated set of reference points in a single region of the embedding space.

    Algorithm:
    1. Randomly select one seed point from candidates
    2. Compute distances from seed to all other candidates
    3. Select the top k (n_samples or ref_ratio * n_candidates) nearest neighbors
    4. Return the seed point plus its nearest neighbors as the reference set

    Args:
        embeddings: Full embedding matrix (N, D)
        candidate_indices: Indices of candidate points to sample from
        ref_ratio: Ratio of candidate points to sample (default: 0.1)
        n_samples: Fixed number of samples. If provided, overrides ref_ratio.
        metric: Distance metric ('euclidean' or 'cosine', default: 'euclidean')
        use_gpu: Use GPU acceleration (default: False)
        seed: Random seed for reproducibility (default: None)
        **kwargs: Additional arguments (ignored for compatibility)

    Returns:
        Array of sampled reference point indices (seed + its nearest neighbors)

    Example:
        >>> ref_indices = sample_nearest(
        ...     embeddings, candidate_indices, n_samples=15, metric='euclidean'
        ... )
    """
    if isinstance(embeddings, torch.Tensor):
        embeddings_np = embeddings.cpu().numpy()
    else:
        embeddings_np = embeddings

    # Extract embeddings for candidate indices
    candidate_embs = embeddings_np[candidate_indices]
    n_candidates = len(candidate_indices)

    # Determine number of points to sample from n_samples or ref_ratio
    if n_samples is not None:
        n_to_sample = min(max(1, n_samples), n_candidates)
        logger.debug(f"Nearest sampling: selecting {n_to_sample} from {n_candidates} candidates (fixed n_seeds={n_samples}, metric={metric})")
    else:
        n_to_sample = max(1, int(n_candidates * ref_ratio))
        n_to_sample = min(n_to_sample, n_candidates)
        logger.debug(f"Nearest sampling: selecting {n_to_sample} from {n_candidates} candidates (ratio={ref_ratio:.3f}, metric={metric})")

    # Set seed if provided
    if seed is not None:
        np.random.seed(seed)

    # Randomly select a seed point
    seed_local_idx = np.random.randint(0, n_candidates)
    seed_emb = candidate_embs[seed_local_idx]

    logger.debug(f"Nearest sampling: using candidate {seed_local_idx} (global idx {candidate_indices[seed_local_idx]}) as seed")

    # Compute distances from seed to all candidates
    if use_gpu and torch.cuda.is_available():
        try:
            device = torch.device("cuda")
            candidate_embs_gpu = torch.from_numpy(candidate_embs).float().to(device)
            seed_emb_gpu = torch.from_numpy(seed_emb).float().to(device).unsqueeze(0)

            if metric == "cosine":
                # Normalize for cosine similarity
                candidate_embs_gpu = torch.nn.functional.normalize(candidate_embs_gpu, dim=1)
                seed_emb_gpu = torch.nn.functional.normalize(seed_emb_gpu, dim=1)
                # Cosine distance = 1 - cosine similarity
                distances = 1.0 - torch.mm(candidate_embs_gpu, seed_emb_gpu.T).squeeze()
            else:  # euclidean
                distances = torch.norm(candidate_embs_gpu - seed_emb_gpu, dim=1)

            distances = distances.cpu().numpy()

            # Clean up GPU memory
            del candidate_embs_gpu, seed_emb_gpu
            torch.cuda.empty_cache()

        except Exception as e:
            logger.warning(f"GPU nearest sampling failed: {e}, falling back to CPU")
            use_gpu = False

    if not use_gpu or not torch.cuda.is_available():
        # CPU implementation
        if metric == "cosine":
            # Normalize for cosine similarity
            norms = np.linalg.norm(candidate_embs, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            candidate_embs_norm = candidate_embs / norms
            seed_norm = seed_emb / max(np.linalg.norm(seed_emb), 1e-12)
            # Cosine distance = 1 - cosine similarity
            distances = 1.0 - np.dot(candidate_embs_norm, seed_norm)
        else:  # euclidean
            distances = np.linalg.norm(candidate_embs - seed_emb, axis=1)

    # Get indices of the k nearest neighbors (including the seed point itself)
    # argsort returns indices that would sort the array in ascending order
    nearest_local_indices = np.argsort(distances)[:n_to_sample]

    # Map back to original (global) indices
    ref_points = candidate_indices[nearest_local_indices]

    logger.debug(f"Nearest sampling: selected {len(ref_points)} reference points around seed")

    return ref_points


def get_sample_method(method_name: str):
    """
    Get sampling method function by name.

    Args:
        method_name: Name of the sampling method
                    ('cluster_centroids', 'furthest_point', 'random', 'localized_cluster', 'nearest')

    Returns:
        Sampling function

    Raises:
        ValueError: If method_name is not recognized
    """
    methods = {
        'cluster_centroids': sample_cluster_centroids,
        'furthest_point': sample_furthest_point,
        'random': sample_random,
        'localized_cluster': sample_localized_cluster,
        'nearest': sample_nearest,
    }

    if method_name not in methods:
        raise ValueError(f"Unknown sampling method: {method_name}. "
                        f"Available methods: {list(methods.keys())}")

    return methods[method_name]


def sample_ref_points(
    method: str,
    embeddings: Union[np.ndarray, torch.Tensor],
    candidate_indices: np.ndarray,
    ref_ratio: float = 0.1,
    n_samples: Optional[int] = None,
    **kwargs
) -> np.ndarray:
    """
    Sample reference points using the specified method.

    This is a convenience wrapper that calls the appropriate sampling method.

    Args:
        method: Sampling method name ('cluster_centroids', 'furthest_point', 'random', 'localized_cluster', 'nearest')
        embeddings: Full embedding matrix (N, D)
        candidate_indices: Indices of candidate points to sample from
        ref_ratio: Ratio of candidate points to sample (default: 0.1)
        n_samples: Fixed number of samples. If provided, overrides ref_ratio.
        **kwargs: Additional method-specific arguments

    Returns:
        Array of sampled reference point indices

    Example:
        >>> ref_indices = sample_ref_points(
        ...     'cluster_centroids', embeddings, candidate_indices, ref_ratio=0.1
        ... )
        >>> # Or with fixed number of samples:
        >>> ref_indices = sample_ref_points(
        ...     'random', embeddings, candidate_indices, n_samples=10
        ... )
    """
    sample_fn = get_sample_method(method)
    return sample_fn(embeddings, candidate_indices, ref_ratio, n_samples=n_samples, **kwargs)
