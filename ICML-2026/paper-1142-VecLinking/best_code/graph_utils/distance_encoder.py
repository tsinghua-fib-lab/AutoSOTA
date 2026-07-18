import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import torch
from utils.graph_util import get_dists
from utils.load_data import align_dim
from utils.memory_util import estimate_matrix_memory_gb, get_available_memory_gb
from loguru import logger


def get_ref_points(emb, method="centroid", ratio=0.1, ref_num=None):
    """
    Select reference points from embeddings using various methods.
    
    Args:
        emb: Input embeddings (numpy array or tensor)
        method: Selection method ("centroid", "medoid", "random", "farthest")
        ratio: Proportion of points to select as references (when ref_num is None)
        ref_num: Exact number of reference points to select
        
    Returns:
        Array of indices of selected reference points
    """
    # Handle tensor vs numpy operations
    if isinstance(emb, torch.Tensor):
        emb_work = emb.cpu().numpy()  # Convert for numpy operations
    else:
        emb_work = emb

    if ref_num is not None:
        select_num = ref_num
    else:
        select_num = int(len(emb_work) * ratio)

    if method == "centroid":
        centroid = np.mean(emb_work, axis=0)
        distances = np.linalg.norm(emb_work - centroid, axis=1)
        closest_indices = np.argsort(distances)[:select_num]
        return closest_indices
    elif method == "medoid":
        dist_matrix = euclidean_distances(emb_work)
        total_distances = np.sum(dist_matrix, axis=1)
        medoid_idx = np.argsort(total_distances)[:select_num]
        return medoid_idx
    elif method == "random":
        return np.random.choice(range(len(emb_work)), size=select_num, replace=False)
    elif method == "farthest":
        # Farthest point sampling - select points as far apart as possible
        if select_num >= len(emb_work):
            return np.arange(len(emb_work))

        # Compute distance matrix
        dist_matrix = euclidean_distances(emb_work)

        # Start with a random point (or could use point farthest from centroid)
        selected_indices = [np.random.randint(len(emb_work))]

        for _ in range(select_num - 1):
            # For each remaining point, find minimum distance to already selected points
            min_dists = np.full(len(emb_work), np.inf)
            for i in range(len(emb_work)):
                if i not in selected_indices:
                    # Find minimum distance to any selected point
                    min_dist_to_selected = min(dist_matrix[i, j] for j in selected_indices)
                    min_dists[i] = min_dist_to_selected

            # Select the point with maximum minimum distance
            next_point = int(np.argmax(min_dists))
            selected_indices.append(next_point)

        return np.array(selected_indices)
    else:
        raise ValueError(f"Invalid method: {method}")


def compute_distance_encoding(emb, distance_metric='euclidean', use_gpu=True, device=None,
                                  multi_gpu_config=None, ref_indices=None, ref_embeddings=None, emb_indices=None,
                                  select="all", ref_method="centroid", ratio=0.1, ref_num=None,
                                  reduced_dim=None, cluster_labels=None,
                                  cluster_label_list=None,
                                  transformation=None, transformation_params=None,
                                  is_normalized=False,
                                  # Deprecated parameters - for backward compatibility
                                  use_rbf_encoding=None, rbf_sigma=None):
    """
    Unified function for distance-based encoding that combines the functionality of
    dist2vec, encode_dist, and encode_dist_with_ref.

    Args:
        emb: Input embeddings (numpy array or tensor)
        distance_metric: Distance metric to use
        use_gpu: Whether to use GPU for computations
        device: Specific device to use (if None, auto-detected)
        multi_gpu_config: Optional dict passed to get_dists for multi-GPU distance computation
        ref_indices: Specific reference point indices to use
        ref_embeddings: Reference embeddings to use directly
        emb_indices: Specific embedding indices to encode (if None, encodes all)
        select: Reference selection strategy ("cluster" or "all") when auto-selecting
        ref_method: Method for selecting reference points ("centroid", "medoid", "random", "farthest")
        ratio: Proportion of points to select as references (when ref_num is None)
        ref_num: Exact number of reference points to select
        reduced_dim: Apply PCA dimension reduction to output
        cluster_labels: Cluster labels for cluster-based reference selection
        cluster_label_list: List of unique cluster labels
        transformation: Distance transformation type (None, "rbf", "inverse", "sigmoid")
        transformation_params: Dict of transformation-specific parameters
            - For "rbf": {"sigma": float} (default: None = auto-compute)
            - For "inverse": {"scale": float} (default: 1.0)
            - For "sigmoid": {"scale": float, "midpoint": float} (defaults: 1.0, auto)
        is_normalized: If True, skip normalization for cosine distance (inputs already L2-normalized)
        use_rbf_encoding: [DEPRECATED] Use transformation="rbf" instead
        rbf_sigma: [DEPRECATED] Use transformation_params={"sigma": value} instead

    Returns:
        Distance vectors (embeddings encoded as distances to reference points)
    """
    # Handle backward compatibility for deprecated parameters
    if use_rbf_encoding is not None:
        import warnings
        warnings.warn(
            "use_rbf_encoding is deprecated and will be removed in v2.0. "
            "Use transformation='rbf' instead.",
            DeprecationWarning,
            stacklevel=2
        )
        if transformation is None:
            transformation = "rbf" if use_rbf_encoding else None

    if rbf_sigma is not None:
        import warnings
        warnings.warn(
            "rbf_sigma is deprecated and will be removed in v2.0. "
            "Use transformation_params={'sigma': value} instead.",
            DeprecationWarning,
            stacklevel=2
        )
        if transformation == "rbf" and transformation_params is None:
            transformation_params = {"sigma": rbf_sigma}

    # Normalize "none" to None
    if transformation == "none":
        transformation = None

    # Handle device setup
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() and use_gpu else "cpu")

    # Determine tensor properties
    is_tensor = isinstance(emb, torch.Tensor)
    if is_tensor:
        device = emb.device
            
    # Mode 1: Use provided reference embeddings directly
    if ref_embeddings is not None:
        dist_matrix = get_dists(
            emb,
            ref_embeddings,
            metric=distance_metric,
            use_gpu=use_gpu,
            device=device,
            multi_gpu_config=multi_gpu_config,
            is_normalized=is_normalized
        )
        
    # Mode 2: Use provided reference indices
    elif ref_indices is not None:
        # Extract reference embeddings from indices
        if isinstance(emb, torch.Tensor) and isinstance(ref_indices, (list, np.ndarray)):
            ref_emb = emb[ref_indices]
        elif isinstance(emb, torch.Tensor) and isinstance(ref_indices, torch.Tensor):
            ref_emb = emb[ref_indices]
        else:
            ref_emb = emb[ref_indices]
            
        # Handle specific embedding indices if provided
        if emb_indices is not None:
            if isinstance(emb, torch.Tensor):
                emb_subset = emb[emb_indices]
            else:
                emb_subset = emb[emb_indices]
            dist_matrix = get_dists(
                emb_subset,
                ref_emb,
                metric=distance_metric,
                use_gpu=use_gpu,
                device=device,
                multi_gpu_config=multi_gpu_config,
                is_normalized=is_normalized
            )
        else:
            dist_matrix = get_dists(
                emb,
                ref_emb,
                metric=distance_metric,
                use_gpu=use_gpu,
                device=device,
                multi_gpu_config=multi_gpu_config,
                is_normalized=is_normalized
            )
            
    # Mode 3: Auto-select reference points
    else:
        # Compute full distance matrix first
        dist_matrix_full = get_dists(
            emb,
            metric=distance_metric,
            use_gpu=use_gpu,
            device=device,
            multi_gpu_config=multi_gpu_config,
            is_normalized=is_normalized
        )
        
        if select == "cluster" and cluster_labels is not None:
            # select ref points from each cluster
            all_ref_ind = []

            # Get unique labels in appropriate format
            if cluster_label_list is not None:
                labels_unique = cluster_label_list
            else:
                if is_tensor:
                    labels_unique = torch.unique(cluster_labels)
                else:
                    labels_unique = np.unique(cluster_labels)

            for label in labels_unique:
                # Get cluster indices
                if is_tensor:
                    emb_cluster_indices = torch.where(cluster_labels == label)[0].cpu().numpy()
                else:
                    emb_cluster_indices = np.where(cluster_labels == label)[0]

                # Get cluster embeddings
                if is_tensor:
                    clu_emb = emb[emb_cluster_indices]
                else:
                    clu_emb = emb[emb_cluster_indices]

                ref_ind = get_ref_points(clu_emb, method=ref_method, ratio=ratio, ref_num=ref_num)
                ref_ind = emb_cluster_indices[ref_ind]
                all_ref_ind.extend(ref_ind)

            # Use tensor operations if available
            if is_tensor and isinstance(dist_matrix_full, torch.Tensor):
                all_ref_ind_tensor = torch.tensor(all_ref_ind, device=device, dtype=torch.long)
                dist_matrix = dist_matrix_full[:, all_ref_ind_tensor]
            else:
                dist_matrix = dist_matrix_full[:, all_ref_ind]

        elif select == "all":
            ref_ind = get_ref_points(emb, method=ref_method, ratio=ratio, ref_num=ref_num)

            # Use tensor operations if available
            if is_tensor and isinstance(dist_matrix_full, torch.Tensor):
                ref_ind_tensor = torch.tensor(ref_ind, device=device)
                dist_matrix = dist_matrix_full[:, ref_ind_tensor]
            else:
                dist_matrix = dist_matrix_full[:, ref_ind]
                
        # Handle specific embedding indices if provided for auto-selection
        if emb_indices is not None:
            if isinstance(dist_matrix, torch.Tensor):
                emb_ind_tensor = torch.tensor(emb_indices, device=dist_matrix.device)
                dist_matrix = dist_matrix[emb_ind_tensor]
            else:
                dist_matrix = dist_matrix[emb_indices]

    # Convert to tensor for output if using GPU
    # Only move to GPU if it's not already a tensor (computation already happened on GPU)
    if use_gpu and not isinstance(dist_matrix, torch.Tensor):
        # For large matrices, keep on CPU to avoid OOM during conversion
        # The actual distance computation should have been done on GPU already
        # If we need it on GPU later, it will be moved in chunks
        matrix_size_gb = estimate_matrix_memory_gb(dist_matrix.shape[0], dist_matrix.shape[1])
        available_gpu_gb = get_available_memory_gb(use_gpu=True, device=device)

        # Only move to GPU if it fits comfortably (use 30% threshold)
        if matrix_size_gb < available_gpu_gb * 0.3:
            dist_matrix = torch.tensor(dist_matrix, dtype=torch.float32, device=device)
        else:
            logger.warning(
                f"Distance matrix ({matrix_size_gb:.2f} GB) too large for GPU "
                f"({available_gpu_gb:.2f} GB available). Keeping on CPU."
            )
            dist_matrix = torch.tensor(dist_matrix, dtype=torch.float32, device='cpu')
    elif use_gpu and isinstance(dist_matrix, torch.Tensor) and dist_matrix.device != device:
        # Already a tensor but on wrong device - move only if it fits
        matrix_size_gb = estimate_matrix_memory_gb(dist_matrix.shape[0], dist_matrix.shape[1])
        available_gpu_gb = get_available_memory_gb(use_gpu=True, device=device)

        if matrix_size_gb < available_gpu_gb * 0.3:
            dist_matrix = dist_matrix.to(device)
        else:
            logger.warning(
                f"Distance matrix ({matrix_size_gb:.2f} GB) too large to move to GPU "
                f"({available_gpu_gb:.2f} GB available). Keeping on current device."
            )

    # Apply distance transformation if requested
    if transformation is not None:
        from utils.graph_util import apply_distance_transformation

        dist_matrix = apply_distance_transformation(
            dist_matrix,
            transformation=transformation,
            params=transformation_params,
            use_gpu=use_gpu,
            device=device
        )

    # Apply dimensionality reduction if requested
    if reduced_dim is not None:
        dist_matrix = align_dim(dist_matrix, method='pca', n_components=reduced_dim)

    return dist_matrix
