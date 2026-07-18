import torch
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from queue import Queue
import threading
from utils.memory_util import estimate_matrix_memory_gb, get_available_memory_gb
from loguru import logger

_INT_MAX = 2_147_483_647
_MEDIAN_SAMPLE_SIZE = 1_000_000


def _safe_median_positive(data, default=None):
    """Median of positive elements, sampling if tensor exceeds INT_MAX."""
    is_tensor = isinstance(data, torch.Tensor)
    numel = data.numel() if is_tensor else data.size

    if numel < _INT_MAX:
        # Small path: original behavior
        if is_tensor:
            mask = data > 0
            if mask.any():
                return torch.median(data[mask])
            return default if default is not None else torch.tensor(1.0, device=data.device)
        else:
            mask = data > 0
            if np.any(mask):
                return np.median(data[mask])
            return default if default is not None else 1.0

    # Large path: sample random flat indices to avoid INT_MAX limit
    if is_tensor:
        indices = torch.randint(0, numel, (_MEDIAN_SAMPLE_SIZE,))
        samples = data.reshape(-1)[indices.to(data.device)]
        samples = samples[samples > 0]
        if samples.numel() == 0:
            return default if default is not None else torch.tensor(1.0, device=data.device)
        return torch.median(samples)
    else:
        indices = np.random.randint(0, numel, _MEDIAN_SAMPLE_SIZE)
        samples = data.reshape(-1)[indices]
        samples = samples[samples > 0]
        if len(samples) == 0:
            return default if default is not None else 1.0
        return np.median(samples)


def chunked_cosine_similarity(x_norm, y_norm, device, chunk_size=None, available_memory_gb=None):
    """
    Compute cosine similarity matrix in chunks to avoid OOM.

    Args:
        x_norm: Normalized tensor (n, d)
        y_norm: Normalized tensor (m, d)
        device: Computing device
        chunk_size: Number of rows to process at once (auto-computed if None)
        available_memory_gb: Available GPU memory (auto-detected if None)

    Returns:
        Similarity matrix (n, m)
    """
    n, d = x_norm.shape
    m = y_norm.shape[0]

    # Estimate memory required for full matrix
    matrix_memory_gb = estimate_matrix_memory_gb(n, m)

    # Get available memory
    if available_memory_gb is None:
        available_memory_gb = get_available_memory_gb(use_gpu=True, device=device)

    # Check if we need chunking (use 1 GB threshold for safety)
    # Also check if result matrix itself will fit in memory
    result_fits_on_gpu = matrix_memory_gb < available_memory_gb * 0.2  # Only use 20% for result

    if matrix_memory_gb > 1.0 or available_memory_gb < matrix_memory_gb * 1.5 or not result_fits_on_gpu:
        # Calculate safe chunk size
        if chunk_size is None:
            # Use 30% of available memory for safety (less aggressive for parallel workers)
            usable_memory_gb = available_memory_gb * 0.3
            # chunk_size × m × 4 bytes <= usable_memory_gb
            chunk_size = max(100, int(usable_memory_gb * (1024**3) / (m * 4)))
            chunk_size = min(chunk_size, n)  # Don't exceed n

        logger.debug(f"Chunked cosine similarity: matrix {n}×{m} = {matrix_memory_gb:.2f} GB, "
                   f"available {available_memory_gb:.2f} GB, using chunk_size={chunk_size}")

        # If result matrix doesn't fit on GPU, allocate on CPU
        if not result_fits_on_gpu:
            logger.debug(f"Result matrix ({matrix_memory_gb:.2f} GB) too large for GPU "
                       f"({available_memory_gb:.2f} GB avail). Allocating on CPU, computing on GPU.")
            result = torch.zeros(n, m, device='cpu', dtype=torch.float32)

            # Process in chunks, compute on GPU but store on CPU
            n_chunks = (n + chunk_size - 1) // chunk_size
            for i in range(0, n, chunk_size):
                end_i = min(i + chunk_size, n)
                chunk = torch.mm(x_norm[i:end_i], y_norm.T)
                result[i:end_i] = chunk.cpu()  # Move chunk to CPU
                del chunk  # Free GPU memory immediately
                torch.cuda.empty_cache()

                if n_chunks > 5:  # Only log if many chunks
                    logger.debug(f"  Chunk {i//chunk_size + 1}/{n_chunks} completed")
        else:
            # Result fits on GPU
            result = torch.zeros(n, m, device=device, dtype=torch.float32)

            # Process in chunks
            n_chunks = (n + chunk_size - 1) // chunk_size
            for i in range(0, n, chunk_size):
                end_i = min(i + chunk_size, n)
                chunk = torch.mm(x_norm[i:end_i], y_norm.T)
                result[i:end_i] = chunk

                if n_chunks > 5:  # Only log if many chunks
                    logger.debug(f"  Chunk {i//chunk_size + 1}/{n_chunks} completed")

        return result
    else:
        # Memory sufficient, compute directly
        return torch.mm(x_norm, y_norm.T)


def chunked_euclidean_distance(x, y, device, chunk_size=None, available_memory_gb=None):
    """
    Compute Euclidean distance matrix in chunks to avoid OOM.

    Uses the efficient formula: ||a-b||^2 = ||a||^2 + ||b||^2 - 2⟨a,b⟩
    """
    n, d = x.shape
    m = y.shape[0]

    # Estimate memory
    matrix_memory_gb = estimate_matrix_memory_gb(n, m)
    if available_memory_gb is None:
        available_memory_gb = get_available_memory_gb(use_gpu=True, device=device)

    # Balanced memory usage - enough safety for parallel workers but not too conservative
    # Use 20% of available memory per worker (reasonable for 4-5 parallel workers on 79GB GPU)
    safety_factor = 0.2

    # Check if result matrix will fit on GPU
    result_fits_on_gpu = matrix_memory_gb < available_memory_gb * safety_factor

    # Always use chunking for GPU operations to be safe with parallel workers
    if device.type == 'cuda':
        if chunk_size is None:
            # Use 15% of available memory per chunk - balanced between speed and safety
            usable_memory_gb = available_memory_gb * 0.15
            chunk_size = max(100, int(usable_memory_gb * (1024**3) / (m * 4)))
            chunk_size = min(chunk_size, n)

        # logger.debug(f"Chunked euclidean distance: matrix {n}×{m} = {matrix_memory_gb:.2f} GB, "
        #            f"available {available_memory_gb:.2f} GB, using chunk_size={chunk_size}")

        # Precompute ||b||^2 for all y
        y_sq = (y**2).sum(dim=1, keepdim=True).T  # (1, m)

        # Allocate result on GPU if it fits, otherwise CPU
        if result_fits_on_gpu:
            # logger.debug(f"Allocating result matrix on GPU ({matrix_memory_gb:.2f} GB)")
            result = torch.zeros(n, m, device=device, dtype=torch.float32)
        else:
            # logger.debug(f"Allocating result matrix on CPU to save GPU memory")
            result = torch.zeros(n, m, device='cpu', dtype=torch.float32)

        # Process in chunks
        for i in range(0, n, chunk_size):
            end_i = min(i + chunk_size, n)
            x_chunk = x[i:end_i]

            # ||a||^2 for chunk
            x_sq = (x_chunk**2).sum(dim=1, keepdim=True)  # (chunk, 1)

            # -2⟨a,b⟩
            ab = torch.mm(x_chunk, y.T)  # (chunk, m)

            # ||a-b||^2 = ||a||^2 + ||b||^2 - 2⟨a,b⟩
            dists_sq = x_sq + y_sq - 2*ab
            chunk_result = dists_sq.clamp(min=0).sqrt()

            # Move to CPU if result is on CPU, otherwise keep on GPU
            if result.device.type == 'cpu':
                result[i:end_i] = chunk_result.cpu()
            else:
                result[i:end_i] = chunk_result

            # Clean up intermediate tensors
            del chunk_result, dists_sq, ab, x_sq, x_chunk

        # Single cache clear at the end instead of every iteration
        torch.cuda.empty_cache()

        # Clean up y_sq
        del y_sq
        torch.cuda.empty_cache()

        return result
    else:
        # CPU case - check if we need chunking for large matrices
        if matrix_memory_gb > 1.0:
            if chunk_size is None:
                chunk_size = max(100, min(1000, n))

            logger.debug(f"Chunked euclidean distance (CPU): matrix {n}×{m} = {matrix_memory_gb:.2f} GB, using chunk_size={chunk_size}")

            result = torch.zeros(n, m, device='cpu', dtype=torch.float32)
            y_sq = (y**2).sum(dim=1, keepdim=True).T

            for i in range(0, n, chunk_size):
                end_i = min(i + chunk_size, n)
                x_chunk = x[i:end_i]
                x_sq = (x_chunk**2).sum(dim=1, keepdim=True)
                ab = torch.mm(x_chunk, y.T)
                dists_sq = x_sq + y_sq - 2*ab
                result[i:end_i] = dists_sq.clamp(min=0).sqrt()
                del x_chunk, x_sq, ab, dists_sq

            del y_sq
            return result
        else:
            # Small enough to compute directly
            return torch.cdist(x, y, p=2)


def _multi_gpu_distance_worker(task_queue, result_array, source_array, target_tensor,
                               metric, sigma, gpu_id):
    """
    Worker that processes queued row chunks on a single GPU.

    OPTIMIZATION 2.2: target_tensor is now pre-loaded on the GPU instead of loading from target_array.
    This avoids redundant GPU memory transfers when multiple workers share the same target data.
    """
    device = torch.device(f"cuda:{gpu_id}")
    # target_tensor is already on the correct GPU device, no need to reload

    while True:
        task = task_queue.get()
        if task is None:
            task_queue.task_done()
            break

        chunk_start, chunk_end = task
        chunk_tensor = torch.tensor(source_array[chunk_start:chunk_end], device=device, dtype=torch.float32)

        # Reuse standard get_dists on a single GPU for each chunk
        dist_chunk = get_dists(
            chunk_tensor,
            target_tensor,
            metric=metric,
            sigma=sigma,
            use_gpu=True,
            device=device,
            multi_gpu_config=None
        )

        if isinstance(dist_chunk, torch.Tensor):
            dist_chunk_np = dist_chunk.detach().cpu().numpy()
        else:
            dist_chunk_np = dist_chunk

        result_array[chunk_start:chunk_end] = dist_chunk_np

        del chunk_tensor, dist_chunk, dist_chunk_np
        torch.cuda.empty_cache()
        task_queue.task_done()

    del target_tensor
    torch.cuda.empty_cache()


def _multi_gpu_pairwise_distance(source_array, target_array, metric, sigma, gpu_ids, chunk_size=None):
    """
    Compute pairwise distances by distributing row chunks across multiple GPUs.

    OPTIMIZATION 2.2: Pre-load target_array to each GPU once before starting workers,
    reducing redundant GPU memory transfers.
    """
    n_rows = source_array.shape[0]
    n_cols = target_array.shape[0]

    if chunk_size is None:
        # Aim for at least 4 chunks per GPU to keep them busy
        approx_chunk = max(1, n_rows // (len(gpu_ids) * 4))
        chunk_size = max(256, approx_chunk)
        chunk_size = min(chunk_size, n_rows)

    logger.debug(
        f"Multi-GPU distance computation across {len(gpu_ids)} GPUs: "
        f"{n_rows}x{n_cols} matrix, chunk_size={chunk_size}"
    )

    # OPTIMIZATION 2.2: Pre-load target array to each GPU once
    # This avoids each worker loading the same data repeatedly
    shared_target_tensors = {}
    for gpu_id in gpu_ids:
        device = torch.device(f"cuda:{gpu_id}")
        shared_target_tensors[gpu_id] = torch.tensor(target_array, device=device, dtype=torch.float32)

    task_queue = Queue()
    for start in range(0, n_rows, chunk_size):
        end = min(start + chunk_size, n_rows)
        task_queue.put((start, end))
    for _ in gpu_ids:
        task_queue.put(None)

    result = np.zeros((n_rows, n_cols), dtype=np.float32)
    threads = []
    for gpu_id in gpu_ids:
        # Pass pre-loaded tensor instead of numpy array
        target_tensor = shared_target_tensors[gpu_id]
        worker = threading.Thread(
            target=_multi_gpu_distance_worker,
            args=(task_queue, result, source_array, target_tensor, metric, sigma, gpu_id),
            daemon=True
        )
        worker.start()
        threads.append(worker)

    task_queue.join()
    for worker in threads:
        worker.join()

    return result


def get_dists(dist_vec1, dist_vec2=None, metric='euclidean', sigma=None, use_gpu=False, device=None,
              multi_gpu_config=None, is_normalized=False):
    """
    Unified function to compute distances between embeddings.

    Args:
        dist_vec1, dist_vec2: Embedding matrices (can be single vectors or matrices)
        metric: "cosine", "euclidean", or "rbf"
        sigma: RBF kernel parameter (auto-computed if None)
        use_gpu: Whether to use GPU computation
        device: Computing device (for GPU)
        multi_gpu_config: Optional dict with keys {enabled: bool, gpu_ids: List[int], chunk_size: int}
        is_normalized: If True, skip normalization for cosine distance (inputs already L2-normalized)

    Returns:
        Distance matrix or vector
    """
    if dist_vec2 is None:
        dist_vec2 = dist_vec1

    # Determine if multi-GPU execution is requested/available
    multi_gpu_enabled = False
    multi_gpu_chunk_size = None
    gpu_ids = None
    if multi_gpu_config:
        gpu_ids = multi_gpu_config.get("gpu_ids")
        if gpu_ids is None or len(gpu_ids) == 0:
            gpu_ids = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        multi_gpu_enabled = (
            multi_gpu_config.get("enabled", False)
            and torch.cuda.is_available()
            and len(gpu_ids) > 1
        )
        multi_gpu_chunk_size = multi_gpu_config.get("chunk_size")

    if multi_gpu_enabled:
        # Multi-GPU path only handles matrix-to-matrix distances
        if isinstance(dist_vec1, torch.Tensor):
            source_array = dist_vec1.detach().cpu().numpy()
        else:
            source_array = np.asarray(dist_vec1)
        if isinstance(dist_vec2, torch.Tensor):
            target_array = dist_vec2.detach().cpu().numpy()
        else:
            target_array = np.asarray(dist_vec2)

        if source_array.ndim == 2 and target_array.ndim == 2:
            source_array = source_array.astype(np.float32, copy=False)
            target_array = target_array.astype(np.float32, copy=False)

            dist_matrix = _multi_gpu_pairwise_distance(
                source_array,
                target_array,
                metric,
                sigma,
                gpu_ids,
                chunk_size=multi_gpu_chunk_size
            )

            if use_gpu and device is not None:
                try:
                    return torch.tensor(dist_matrix, device=device)
                except RuntimeError as exc:
                    logger.warning(
                        f"Unable to move multi-GPU result to {device}: {exc}. Keeping on CPU."
                    )
                    return torch.tensor(dist_matrix, device='cpu')
            return dist_matrix
        else:
            logger.debug("Multi-GPU requested but inputs not 2D matrices; falling back to single-device computation.")

    # Handle GPU/CPU conversion based on use_gpu parameter
    if use_gpu and device is not None:
        if not torch.is_tensor(dist_vec1):
            dist_vec1 = torch.tensor(dist_vec1, device=device, dtype=torch.float32)
        if not torch.is_tensor(dist_vec2):
            dist_vec2 = torch.tensor(dist_vec2, device=device, dtype=torch.float32)

    if type(dist_vec1) == torch.Tensor:
        if metric == 'cosine':
            # Normalize embeddings (skip if already normalized)
            if is_normalized:
                dist_vec1_norm = dist_vec1
                dist_vec2_norm = dist_vec2
            elif dist_vec1 is dist_vec2:
                # Self-distance case: normalize only once
                if dist_vec1.dim() == 1:
                    dist_vec1_norm = dist_vec1 / torch.norm(dist_vec1, keepdim=True)
                else:
                    dist_vec1_norm = torch.nn.functional.normalize(dist_vec1, dim=-1)
                dist_vec2_norm = dist_vec1_norm
            else:
                # Different inputs: normalize both
                if dist_vec1.dim() == 1:
                    dist_vec1_norm = dist_vec1 / torch.norm(dist_vec1, keepdim=True)
                else:
                    dist_vec1_norm = torch.nn.functional.normalize(dist_vec1, dim=-1)

                if dist_vec2.dim() == 1:
                    dist_vec2_norm = dist_vec2 / torch.norm(dist_vec2, keepdim=True)
                else:
                    dist_vec2_norm = torch.nn.functional.normalize(dist_vec2, dim=-1)

            # Handle different dimensionality combinations
            if dist_vec1.dim() == 1 and dist_vec2.dim() == 2:
                # Single vector to multiple vectors
                similarities = torch.mv(dist_vec2_norm, dist_vec1_norm)
            elif dist_vec1.dim() == 2 and dist_vec2.dim() == 1:
                # Multiple vectors to single vector 
                similarities = torch.mv(dist_vec1_norm, dist_vec2_norm)
            elif dist_vec1.dim() == 2 and dist_vec2.dim() == 2:
                # Matrix to matrix - use chunked computation for large matrices
                similarities = chunked_cosine_similarity(dist_vec1_norm, dist_vec2_norm, device)
            else:
                # Both single vectors
                similarities = torch.dot(dist_vec1_norm, dist_vec2_norm)
            
            # Clamp and convert to distance
            if torch.is_tensor(similarities):
                similarities = torch.clamp(similarities, -1.0, 1.0)
            dists = 1.0 - similarities
            
        elif metric == 'euclidean':
            # Handle different dimensionality combinations
            if dist_vec1.dim() == 1 and dist_vec2.dim() == 2:
                # Single vector to multiple vectors
                dists = torch.norm(dist_vec2 - dist_vec1.unsqueeze(0), dim=-1)
            elif dist_vec1.dim() == 2 and dist_vec2.dim() == 1:
                # Multiple vectors to single vector
                dists = torch.norm(dist_vec1 - dist_vec2.unsqueeze(0), dim=-1)
            elif dist_vec1.dim() == 2 and dist_vec2.dim() == 2:
                # Matrix to matrix - use chunked computation for large matrices
                dists = chunked_euclidean_distance(dist_vec1, dist_vec2, device)
            else:
                # Both single vectors
                dists = torch.norm(dist_vec1 - dist_vec2)
                
        elif metric == 'rbf':
            # Use chunked euclidean distance for large matrices
            if dist_vec1.dim() == 2 and dist_vec2.dim() == 2:
                euclidean_dists = chunked_euclidean_distance(dist_vec1, dist_vec2, device)
            else:
                euclidean_dists = torch.cdist(dist_vec1, dist_vec2, p=2)
            if sigma is None:
                sigma = _safe_median_positive(euclidean_dists)
                sigma = torch.clamp(sigma, min=1e-8)

            sim = torch.exp(-euclidean_dists**2 / (2 * sigma**2))
            # Convert similarity to distance consistently
            dists = 1.0 - sim
        else:
            raise ValueError("Unsupported metric")
            
        # Convert to CPU if requested and not using GPU
        if not use_gpu and torch.is_tensor(dists):
            return dists.cpu().numpy()
            
    elif type(dist_vec1) == np.ndarray:
        if metric == 'cosine':
            # Normalize embeddings (skip if already normalized)
            if is_normalized:
                dist_vec1_norm = dist_vec1
                dist_vec2_norm = dist_vec2
            elif dist_vec1 is dist_vec2:
                # Self-distance case: normalize only once
                dist_vec1_norm = dist_vec1 / np.linalg.norm(dist_vec1, axis=-1, keepdims=True)
                dist_vec2_norm = dist_vec1_norm
            else:
                # Different inputs: normalize both
                dist_vec1_norm = dist_vec1 / np.linalg.norm(dist_vec1, axis=-1, keepdims=True)
                dist_vec2_norm = dist_vec2 / np.linalg.norm(dist_vec2, axis=-1, keepdims=True)

            # Handle different dimensionality combinations
            if dist_vec1.ndim == 1 and dist_vec2.ndim == 2:
                # Single vector to multiple vectors
                similarities = np.dot(dist_vec2_norm, dist_vec1_norm)
            elif dist_vec1.ndim == 2 and dist_vec2.ndim == 1:
                # Multiple vectors to single vector
                similarities = np.dot(dist_vec1_norm, dist_vec2_norm)
            elif dist_vec1.ndim == 2 and dist_vec2.ndim == 2:
                # Matrix to matrix
                similarities = np.dot(dist_vec1_norm, dist_vec2_norm.T)
            else:
                # Both single vectors
                similarities = np.dot(dist_vec1_norm, dist_vec2_norm)
            dists = 1.0 - similarities
            
        elif metric == 'euclidean':
            # Handle different dimensionality combinations  
            if dist_vec1.ndim == 1 and dist_vec2.ndim == 2:
                # Single vector to multiple vectors
                dists = np.linalg.norm(dist_vec2 - dist_vec1, axis=-1)
            elif dist_vec1.ndim == 2 and dist_vec2.ndim == 1:
                # Multiple vectors to single vector
                dists = np.linalg.norm(dist_vec1 - dist_vec2, axis=-1)
            elif dist_vec1.ndim == 2 and dist_vec2.ndim == 2:
                # Matrix to matrix
                from scipy.spatial.distance import cdist
                dists = cdist(dist_vec1, dist_vec2, metric='euclidean')
            else:
                # Both single vectors
                dists = np.linalg.norm(dist_vec1 - dist_vec2)
                
        elif metric == 'rbf':
            euclidean_dists = euclidean_distances(dist_vec1, dist_vec2)
            if sigma is None:
                sigma = _safe_median_positive(euclidean_dists)
                sigma = max(sigma, 1e-8)

            sim = np.exp(-euclidean_dists**2 / (2 * sigma**2))
            # Convert similarity to distance consistently
            dists = 1.0 - sim
        else:
            raise ValueError("Unsupported metric")
    return dists


def apply_distance_transformation(dist_matrix, transformation="rbf", params=None,
                                  use_gpu=False, device=None):
    """
    Apply a transformation to distance matrix.

    Transforms raw distances to similarity-like values using various functions.
    All transformations produce values in (0, 1] or (0, 1) range where higher
    values indicate closer/more similar points.

    Args:
        dist_matrix: Distance matrix (numpy array or torch tensor)
        transformation: Type of transformation to apply
            - "rbf": Radial Basis Function using exp(-d / sigma)
            - "inverse": Inverse distance using 1 / (1 + scale * d)
            - "sigmoid": Sigmoid-based using 1 / (1 + exp(scale * (d - midpoint)))
            - "inverse_rbf": Inverse RBF using 1 - exp(-d / sigma), emphasizes long distances
        params: Dict of transformation-specific parameters
            - For "rbf": {"sigma": float} (default: None = auto-compute median)
            - For "inverse": {"scale": float} (default: 1.0)
            - For "sigmoid": {"scale": float, "midpoint": float} (defaults: 1.0, auto-median)
            - For "inverse_rbf": {"sigma": float} (default: None = auto-compute median)
        use_gpu: Whether to use GPU
        device: Computing device (for GPU)

    Returns:
        Transformed distance matrix (same type as input)

    Examples:
        >>> # RBF transformation with auto-computed sigma
        >>> transformed = apply_distance_transformation(dists, "rbf")

        >>> # Inverse distance with custom scale
        >>> transformed = apply_distance_transformation(
        ...     dists, "inverse", params={"scale": 2.0})

        >>> # Sigmoid with custom parameters
        >>> transformed = apply_distance_transformation(
        ...     dists, "sigmoid", params={"scale": 1.5, "midpoint": 0.8})
    """
    is_tensor = isinstance(dist_matrix, torch.Tensor)

    # Validate transformation type
    valid_transformations = ["rbf", "inverse", "sigmoid", "inverse_rbf"]
    if transformation not in valid_transformations:
        raise ValueError(f"Invalid transformation '{transformation}'. "
                        f"Must be one of {valid_transformations}")

    # Convert to tensor if needed
    if not is_tensor and use_gpu and device is not None:
        dist_matrix = torch.tensor(dist_matrix, device=device, dtype=torch.float32)
        is_tensor = True

    if transformation == "rbf":
        # RBF: exp(-d / sigma). Paper writes sigma_{t,k} as the median of
        # nonzero pairwise L2 distances "between hashes within the view"
        # (Section 4, p. 6), but that definition is circular (sigma → hash →
        # sigma) and collapses in practice (sparse pool hashes near 0 yield
        # tiny sigma → kernel collapses to one-hot). We use the standard
        # median heuristic on the raw distance-to-anchor matrix, which keeps
        # the kernel well-conditioned and matches the spirit of the spec
        # (per-view bandwidth set by the view's own scale).
        sigma = params.get("sigma") if params else None
        if sigma is None:
            sigma = _safe_median_positive(dist_matrix)
            if is_tensor:
                sigma = torch.clamp(sigma, min=1e-8)
            else:
                sigma = max(sigma, 1e-8)

        # Apply: exp(-d / sigma)  ← NOTE: NOT d²/sigma
        if is_tensor:
            return torch.exp(-dist_matrix / sigma)
        else:
            return np.exp(-dist_matrix / sigma)

    elif transformation == "inverse_rbf":
        # Inverse RBF: 1 - exp(-d / sigma) - emphasizes long distances
        sigma = params.get("sigma") if params else None
        if sigma is None:
            sigma = _safe_median_positive(dist_matrix)
            if is_tensor:
                sigma = torch.clamp(sigma, min=1e-8)
            else:
                sigma = max(sigma, 1e-8)

        # Apply: 1 - exp(-d / sigma)
        if is_tensor:
            return 1.0 - torch.exp(-dist_matrix / sigma)
        else:
            return 1.0 - np.exp(-dist_matrix / sigma)

    elif transformation == "inverse":
        # Inverse: 1 / (1 + scale * d)
        scale = params.get("scale", 1.0) if params else 1.0
        # logger.debug(f"Inverse transformation: scale={scale:.4f}")

        if is_tensor:
            return 1.0 / (1.0 + scale * dist_matrix)
        else:
            return 1.0 / (1.0 + scale * dist_matrix)

    elif transformation == "sigmoid":
        # Sigmoid: 1 / (1 + exp(scale * (d - midpoint)))
        scale = params.get("scale", 1.0) if params else 1.0
        midpoint = params.get("midpoint") if params else None

        if midpoint is None:
            if is_tensor:
                midpoint = _safe_median_positive(dist_matrix, default=torch.tensor(0.5, device=dist_matrix.device))
            else:
                midpoint = _safe_median_positive(dist_matrix, default=0.5)

        # logger.debug(f"Sigmoid transformation: scale={scale:.4f}, midpoint={midpoint:.4f if not is_tensor else midpoint.item():.4f}")

        if is_tensor:
            return 1.0 / (1.0 + torch.exp(scale * (dist_matrix - midpoint)))
        else:
            return 1.0 / (1.0 + np.exp(scale * (dist_matrix - midpoint)))

