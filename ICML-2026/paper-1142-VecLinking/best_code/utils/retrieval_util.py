import numpy as np
import torch
from loguru import logger


def compute_accuracy_recall(ref_indices1, ref_indices2, ind_nonref):
    """Compute accuracy and recall for reference indices."""
    correct = sum(1 for a, b in zip(ref_indices1, ref_indices2) if a == b)
    accuracy = correct / len(ref_indices1) if len(ref_indices1) > 0 else 0
    recall = correct / len(ind_nonref) if len(ind_nonref) > 0 else 0
    return accuracy, recall, correct


def analyze_distance_based_accuracy(
    ref_indices1: np.ndarray,
    ref_indices2: np.ndarray,
    emb1: np.ndarray,
    emb2: np.ndarray,
    anchor_indices: np.ndarray,
    distance_metric: str = 'cosine',
    use_gpu: bool = False,
    device: torch.device = None,
    percentile_ranges: list = None,
    emb1_g2l: np.ndarray = None,
    emb2_g2l: np.ndarray = None,
    anchor_emb1: np.ndarray = None,
    anchor_emb2: np.ndarray = None
) -> dict:
    """
    Analyze accuracy breakdown by distance to supervised anchor points.

    Computes per-pair correctness and correlates it with distances to anchors.
    Returns accuracy statistics for different distance percentile ranges.

    Args:
        ref_indices1: Discovered pair indices from embedding 1 (shape: [n_pairs])
        ref_indices2: Discovered pair indices from embedding 2 (shape: [n_pairs])
        emb1: Full embedding matrix 1 (shape: [n_emb1, d])
        emb2: Full embedding matrix 2 (shape: [n_emb2, d])
        anchor_indices: Indices of supervised anchors (shape: [n_anchors])
                       These are the ORIGINAL supervised reference points (ref_ind)
        distance_metric: 'cosine' or 'euclidean'
        use_gpu: Whether to use GPU for computation
        device: PyTorch device for GPU computation
        percentile_ranges: List of (min%, max%) tuples. Default: [(0,20), (20,40), (40,60), (60,80), (80,100)]

    Returns:
        dict with keys:
            - 'avg_distance_correlation': Pearson correlation for average distance
            - 'min_distance_correlation': Pearson correlation for minimum distance
            - 'avg_distance_p_value': p-value for average distance correlation
            - 'min_distance_p_value': p-value for minimum distance correlation
            - 'percentile_breakdown': dict mapping range tuple to {
                'accuracy': float,
                'n_pairs': int,
                'n_correct': int
              }
            - 'avg_distances': array of average distances for each pair
            - 'min_distances': array of minimum distances for each pair
            - 'correctness': array of 1/0 for each pair
    """
    from scipy.stats import pearsonr
    from utils.graph_util import get_dists
    from loguru import logger

    # Default percentile ranges
    if percentile_ranges is None:
        percentile_ranges = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]

    # Handle empty pairs case
    if len(ref_indices1) == 0 or len(anchor_indices) == 0:
        return {
            'avg_distance_correlation': np.nan,
            'min_distance_correlation': np.nan,
            'avg_distance_p_value': np.nan,
            'min_distance_p_value': np.nan,
            'percentile_breakdown': {range_tuple: {'accuracy': 0.0, 'n_pairs': 0, 'n_correct': 0}
                                    for range_tuple in percentile_ranges},
            'avg_distances': np.array([]),
            'min_distances': np.array([]),
            'correctness': np.array([])
        }

    n_pairs = len(ref_indices1)
    n_anchors = len(anchor_indices)

    # Compute correctness
    correctness = (ref_indices1 == ref_indices2).astype(np.float32)

    # Extract anchor embeddings (use pre-extracted if provided)
    if anchor_emb1 is None:
        anchor_emb1 = emb1[anchor_indices]
    if anchor_emb2 is None:
        anchor_emb2 = emb2[anchor_indices]

    # Compute distances with chunking
    avg_distances_list = []
    min_distances_list = []

    chunk_size = min(1000, n_pairs)  # Adaptive chunk size

    for chunk_start in range(0, n_pairs, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_pairs)

        # Get pair embeddings for this chunk
        chunk_indices1 = ref_indices1[chunk_start:chunk_end]
        chunk_indices2 = ref_indices2[chunk_start:chunk_end]
        if emb1_g2l is not None:
            chunk_emb1 = emb1[emb1_g2l[chunk_indices1]]
            chunk_emb2 = emb2[emb2_g2l[chunk_indices2]]
        else:
            chunk_emb1 = emb1[chunk_indices1]
            chunk_emb2 = emb2[chunk_indices2]

        try:
            # Distances in space 1: [chunk_size, n_anchors]
            dist_matrix1 = get_dists(
                chunk_emb1, anchor_emb1,
                metric=distance_metric,
                use_gpu=use_gpu,
                device=device
            )

            # Distances in space 2: [chunk_size, n_anchors]
            dist_matrix2 = get_dists(
                chunk_emb2, anchor_emb2,
                metric=distance_metric,
                use_gpu=use_gpu,
                device=device
            )

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning("GPU OOM in distance analysis, falling back to CPU")
                if use_gpu:
                    torch.cuda.empty_cache()
                dist_matrix1 = get_dists(chunk_emb1, anchor_emb1,
                                        metric=distance_metric, use_gpu=False, device=None)
                dist_matrix2 = get_dists(chunk_emb2, anchor_emb2,
                                        metric=distance_metric, use_gpu=False, device=None)
            else:
                raise

        # Convert to numpy if needed
        if torch.is_tensor(dist_matrix1):
            dist_matrix1 = dist_matrix1.cpu().numpy()
        if torch.is_tensor(dist_matrix2):
            dist_matrix2 = dist_matrix2.cpu().numpy()

        # Combine distances from both spaces: [chunk_size, 2*n_anchors]
        combined_distances = np.concatenate([dist_matrix1, dist_matrix2], axis=1)

        # Compute statistics per pair
        chunk_avg = np.mean(combined_distances, axis=1)
        chunk_min = np.min(combined_distances, axis=1)

        avg_distances_list.append(chunk_avg)
        min_distances_list.append(chunk_min)

    # Concatenate all chunks
    avg_distances = np.concatenate(avg_distances_list)
    min_distances = np.concatenate(min_distances_list)

    # Compute correlations
    if len(np.unique(correctness)) < 2:
        # All same correctness
        avg_corr, avg_p_value = np.nan, np.nan
        min_corr, min_p_value = np.nan, np.nan
    elif len(correctness) < 3:
        # Too few samples
        avg_corr, avg_p_value = np.nan, np.nan
        min_corr, min_p_value = np.nan, np.nan
    else:
        try:
            avg_corr, avg_p_value = pearsonr(correctness, avg_distances)
            min_corr, min_p_value = pearsonr(correctness, min_distances)
        except Exception as e:
            logger.warning(f"Pearson correlation computation failed: {e}")
            avg_corr, avg_p_value = np.nan, np.nan
            min_corr, min_p_value = np.nan, np.nan

    # Bucket by percentiles (using average distance)
    percentile_breakdown = {}
    for min_pct, max_pct in percentile_ranges:
        min_threshold = np.percentile(avg_distances, min_pct)
        max_threshold = np.percentile(avg_distances, max_pct)

        if max_pct == 100:
            mask = (avg_distances >= min_threshold) & (avg_distances <= max_threshold)
        else:
            mask = (avg_distances >= min_threshold) & (avg_distances < max_threshold)

        n_pairs_in_range = np.sum(mask)
        if n_pairs_in_range > 0:
            n_correct = np.sum(correctness[mask])
            accuracy = n_correct / n_pairs_in_range
        else:
            n_correct = 0
            accuracy = 0.0

        percentile_breakdown[(min_pct, max_pct)] = {
            'accuracy': float(accuracy),
            'n_pairs': int(n_pairs_in_range),
            'n_correct': int(n_correct)
        }

    return {
        'avg_distance_correlation': float(avg_corr) if not np.isnan(avg_corr) else np.nan,
        'min_distance_correlation': float(min_corr) if not np.isnan(min_corr) else np.nan,
        'avg_distance_p_value': float(avg_p_value) if not np.isnan(avg_p_value) else np.nan,
        'min_distance_p_value': float(min_p_value) if not np.isnan(min_p_value) else np.nan,
        'percentile_breakdown': percentile_breakdown,
        'avg_distances': avg_distances,
        'min_distances': min_distances,
        'correctness': correctness
    }


def topk_mean(m, k, inplace=False):
    """
    Compute the mean of the top k values for each row in matrix m
    """
    if isinstance(m, torch.Tensor):
        device = m.device
        n = m.shape[0]
        ans = torch.zeros(n, dtype=m.dtype, device=device)
        if k <= 0:
            return ans
        if not inplace:
            m = m.clone()
        minimum = m.min()
        for _ in range(k):
            ind1 = m.argmax(dim=1)
            ans += m[torch.arange(n, device=device), ind1]
            m[torch.arange(n, device=device), ind1] = minimum
        return ans / k
    else:
        # Handle numpy arrays
        n = m.shape[0]
        ans = np.zeros(n, dtype=m.dtype)
        if k <= 0:
            return ans
        if not inplace:
            m = np.array(m)
        ind0 = np.arange(n)
        ind1 = np.empty(n, dtype=int)
        minimum = m.min()
        for _ in range(k):
            np.argmax(m, axis=1, out=ind1)
            ans += m[ind0, ind1]
            m[ind0, ind1] = minimum
        return ans / k

def get_topk(dist_vec1, dist_vec2=None, k=5, metric='euclidean', return_dist=False, csls_neighborhood=0, use_faiss=True, approximate_nn="auto", is_normalized=False):
    """
    Get the top k nearest neighbors for each row in dist_vec2 in the space of dist_vec1.

    Automatically detects GPU availability and uses appropriate computation backend:
    - GPU with CUDA (small N): Uses PyTorch tiled GEMM with tensor cores
    - Large N: Uses FAISS CPU IVF with multithreading (O(N*nprobe) vs O(N^2))
    - CPU: Uses NumPy arrays + FAISS CPU or PyTorch CPU

    Args:
        dist_vec1: Reference vectors (database)
        dist_vec2: Query vectors (if None, uses dist_vec1)
        k: Number of nearest neighbors
        metric: Distance metric ('euclidean', 'cosine')
        return_dist: Whether to return distances along with indices
        csls_neighborhood: CSLS correction neighborhood size (0 to disable)
        use_faiss: Whether to use FAISS for k-NN search when possible
        approximate_nn: Use FAISS IVF approximate NN for large databases.
            "auto" (default): use IVF when database > 10k vectors.
            True: always use IVF. False: always use exact IndexFlat.
    """
    import faiss
    from utils.graph_util import get_dists

    # Use all CPU threads for FAISS (critical for large IVF searches)
    faiss.omp_set_num_threads(min(64, faiss.omp_get_max_threads()))

    if dist_vec2 is None:
        dist_vec2 = dist_vec1

    # Auto-detect GPU availability and prefer the device already holding the tensors.
    faiss_gpu_id = 0
    existing_device = None
    if isinstance(dist_vec1, torch.Tensor):
        existing_device = dist_vec1.device
    elif isinstance(dist_vec2, torch.Tensor):
        existing_device = dist_vec2.device

    use_gpu = torch.cuda.is_available() and torch.cuda.device_count() > 0
    if existing_device is not None and existing_device.type == 'cuda':
        device = existing_device
        faiss_gpu_id = existing_device.index if existing_device.index is not None else 0
    else:
        device = torch.device('cuda' if use_gpu else 'cpu')
        if device.type == 'cuda':
            faiss_gpu_id = device.index if device.index is not None else 0
    
    # Convert to appropriate format based on GPU availability
    if use_gpu:
        # Use GPU with PyTorch tensors
        if isinstance(dist_vec1, np.ndarray):
            dist_vec1 = torch.from_numpy(dist_vec1).float().to(device)
        elif isinstance(dist_vec1, torch.Tensor):
            dist_vec1 = dist_vec1.float().to(device)
            
        if isinstance(dist_vec2, np.ndarray):
            dist_vec2 = torch.from_numpy(dist_vec2).float().to(device)
        elif isinstance(dist_vec2, torch.Tensor):
            dist_vec2 = dist_vec2.float().to(device)
    else:
        # Use CPU with NumPy arrays
        if isinstance(dist_vec1, torch.Tensor):
            dist_vec1 = dist_vec1.cpu().numpy()
        if isinstance(dist_vec2, torch.Tensor):
            dist_vec2 = dist_vec2.cpu().numpy()
    
    # Apply CSLS correction if needed
    if csls_neighborhood > 0:
        if use_gpu:
            dists = get_dists(dist_vec2, dist_vec1, metric, use_gpu=True, device=device)
            sim = -dists
            
            # CSLS correction with PyTorch
            knn_sim_fwd = torch.topk(sim, k=csls_neighborhood, dim=1)[0].mean(dim=1)
            knn_sim_bwd = torch.topk(sim.T, k=csls_neighborhood, dim=1)[0].mean(dim=1)
            sim = sim - knn_sim_fwd.unsqueeze(1)/2 - knn_sim_bwd.unsqueeze(0)/2
            
            # Convert back to distances
            dists = -sim
            knn_dists, knn_indices = torch.topk(dists, k, largest=False)
        else:
            dists = get_dists(dist_vec2, dist_vec1, metric, use_gpu=False)
            sim = -dists
            
            # CSLS correction with NumPy
            knn_sim_fwd = topk_mean(sim, k=csls_neighborhood)
            knn_sim_bwd = topk_mean(sim.T, k=csls_neighborhood)
            sim = sim - knn_sim_fwd[:, np.newaxis]/2 - knn_sim_bwd[np.newaxis, :]/2
            
            # Convert back to distances  
            dists = -sim
            knn_indices = np.argpartition(dists, k, axis=1)[:, :k]
            knn_dists = np.take_along_axis(dists, knn_indices, axis=1)
            
            # Sort within each row
            sort_indices = np.argsort(knn_dists, axis=1)
            knn_indices = np.take_along_axis(knn_indices, sort_indices, axis=1)
            knn_dists = np.take_along_axis(knn_dists, sort_indices, axis=1)
    
    else:
        # No CSLS correction - use FAISS for efficient k-NN search
        if use_faiss and metric in ['euclidean', 'cosine']:
            gpu_success = False

            # Determine dimensionality
            if isinstance(dist_vec1, torch.Tensor):
                d = dist_vec1.shape[1]
                n_db = dist_vec1.shape[0]
                n_query = dist_vec2.shape[0] if isinstance(dist_vec2, torch.Tensor) else dist_vec2.shape[0]
            else:
                d = dist_vec1.shape[1]
                n_db = dist_vec1.shape[0]
                n_query = dist_vec2.shape[0]

            if use_gpu:
                try:
                    target_device = torch.device(f'cuda:{faiss_gpu_id}')

                    # Convert inputs — use float16 for large MNN (2x faster GEMM, half memory)
                    use_fp16 = (n_db > 50_000 or n_query > 50_000)
                    target_dtype = torch.float16 if use_fp16 else torch.float32
                    np_dtype = np.float16 if use_fp16 else np.float32
                    if isinstance(dist_vec1, torch.Tensor):
                        vec1_f32 = dist_vec1.to(target_dtype).contiguous()
                    else:
                        vec1_f32 = torch.from_numpy(np.ascontiguousarray(dist_vec1, dtype=np_dtype))
                    if isinstance(dist_vec2, torch.Tensor):
                        vec2_f32 = dist_vec2.to(target_dtype).contiguous()
                    else:
                        vec2_f32 = torch.from_numpy(np.ascontiguousarray(dist_vec2, dtype=np_dtype))

                    if metric == 'cosine' and not is_normalized:
                        vec1_f32 = torch.nn.functional.normalize(vec1_f32, p=2, dim=1)
                        vec2_f32 = torch.nn.functional.normalize(vec2_f32, p=2, dim=1)

                    use_ip = (metric == 'cosine')

                    # GPU brute-force with streaming queries: load database to GPU once,
                    # stream query chunks from CPU. sim_matrix = (chunk, n_db) fits in GPU.
                    vec1_gpu = vec1_f32.to(target_device)
                    torch.cuda.synchronize(faiss_gpu_id)
                    torch.cuda.empty_cache()
                    free_bytes = torch.cuda.mem_get_info(faiss_gpu_id)[0]
                    # Budget for sim matrix: ~50% of free (rest for query chunk + topk + overhead)
                    bytes_per_elem = 2 if use_fp16 else 4
                    sim_budget = int(free_bytes * 0.5)
                    search_chunk = max(256, min(sim_budget // (n_db * bytes_per_elem), n_query))
                    logger.debug(f"  get_topk GPU brute-force: n_db={n_db:,}, n_query={n_query:,}, chunk={search_chunk:,}, free={free_bytes/1e9:.1f}GB")

                    knn_dists_cpu = torch.empty((n_query, k), dtype=torch.float32)
                    knn_indices_cpu = torch.empty((n_query, k), dtype=torch.long)

                    for q_start in range(0, n_query, search_chunk):
                        q_end = min(q_start + search_chunk, n_query)
                        query_gpu = vec2_f32[q_start:q_end].to(target_device)

                        if use_ip:
                            sims = torch.mm(query_gpu, vec1_gpu.T)
                            topk_vals, topk_idx = sims.topk(k, dim=1, largest=True)
                            del sims
                        else:
                            dists_mat = (query_gpu.norm(dim=1, keepdim=True).square()
                                     + vec1_gpu.norm(dim=1).square().unsqueeze(0)
                                     - 2 * torch.mm(query_gpu, vec1_gpu.T))
                            topk_vals, topk_idx = dists_mat.topk(k, dim=1, largest=False)
                            del dists_mat

                        knn_dists_cpu[q_start:q_end] = topk_vals.cpu()
                        knn_indices_cpu[q_start:q_end] = topk_idx.cpu()
                        del query_gpu, topk_vals, topk_idx

                    knn_dists = knn_dists_cpu.to(device)
                    knn_indices = knn_indices_cpu.to(device)
                    del vec1_gpu

                    del vec1_f32, vec2_f32
                    gpu_success = True

                except Exception as e:
                    logger.warning(f"GPU search failed ({e}), falling back to CPU")
                    import traceback; traceback.print_exc()

            # CPU FAISS fallback
            if not gpu_success:
                # Convert to numpy for CPU path
                if isinstance(dist_vec1, torch.Tensor):
                    vec1_np = dist_vec1.cpu().numpy().astype(np.float32)
                elif isinstance(dist_vec1, np.ndarray):
                    vec1_np = dist_vec1.astype(np.float32)
                else:
                    vec1_np = np.asarray(dist_vec1, dtype=np.float32)
                if isinstance(dist_vec2, torch.Tensor):
                    vec2_np = dist_vec2.cpu().numpy().astype(np.float32)
                elif isinstance(dist_vec2, np.ndarray):
                    vec2_np = dist_vec2.astype(np.float32)
                else:
                    vec2_np = np.asarray(dist_vec2, dtype=np.float32)

                d = vec1_np.shape[1]
                n_db = vec1_np.shape[0]
                _use_ivf = (approximate_nn is True) or (approximate_nn == "auto" and n_db > 10000)

                if metric == 'cosine' and not is_normalized:
                    vec1_np = vec1_np.copy()
                    vec2_np = vec2_np.copy()
                    faiss.normalize_L2(vec1_np)
                    faiss.normalize_L2(vec2_np)
                if _use_ivf and n_db > 256:
                    nlist = min(int(np.sqrt(n_db)), 4096)
                    nlist = max(1, nlist)

                    if metric == 'cosine':
                        quantizer = faiss.IndexFlatIP(d)
                        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
                    else:
                        quantizer = faiss.IndexFlatL2(d)
                        index = faiss.IndexIVFFlat(quantizer, d, nlist)

                    # Train on subsample for large databases
                    train_size = min(n_db, max(nlist * 40, 200_000))
                    if train_size < n_db:
                        train_idx = np.random.choice(n_db, train_size, replace=False)
                        index.train(vec1_np[train_idx])
                    else:
                        index.train(vec1_np)
                    index.add(vec1_np)
                    index.nprobe = min(nlist, 64)
                else:
                    if metric == 'cosine':
                        index = faiss.IndexFlatIP(d)
                    else:
                        index = faiss.IndexFlatL2(d)
                    index.add(vec1_np)

                knn_dists, knn_indices = index.search(vec2_np, k)
        
        else:
            # Fallback to manual computation without FAISS
            if use_gpu:
                dists = get_dists(dist_vec2, dist_vec1, metric, use_gpu=True, device=device)
                knn_dists, knn_indices = torch.topk(dists, k, largest=False)
            else:
                dists = get_dists(dist_vec2, dist_vec1, metric, use_gpu=False)
                knn_indices = np.argpartition(dists, k, axis=1)[:, :k]
                knn_dists = np.take_along_axis(dists, knn_indices, axis=1)
                
                # Sort within each row
                sort_indices = np.argsort(knn_dists, axis=1)
                knn_indices = np.take_along_axis(knn_indices, sort_indices, axis=1)
                knn_dists = np.take_along_axis(knn_dists, sort_indices, axis=1)
    
    if return_dist:
        return knn_indices, knn_dists
    else:
        return knn_indices


def deduplicate_pairs(pairs):
    # Deduplicate pairs - keep only the shortest distance for each element in each position
    first_pos_best = {}  # first_position_element -> (second_element, distance, full_pair)
    second_pos_best = {}  # second_position_element -> (first_element, distance, full_pair)
    
    for i, j, dist in pairs:
        # Check if element i (first position) already has a better pair
        if i not in first_pos_best or dist < first_pos_best[i][1]:
            first_pos_best[i] = (j, dist, (i, j, dist))
        
        # Check if element j (second position) already has a better pair  
        if j not in second_pos_best or dist < second_pos_best[j][1]:
            second_pos_best[j] = (i, dist, (i, j, dist))
    
    # Find pairs that are optimal for both positions
    deduplicated_pairs = []
    for i, (j, dist, pair) in first_pos_best.items():
        if second_pos_best.get(j, (None, None, None))[2] == pair:
            deduplicated_pairs.append(pair)
    return deduplicated_pairs
        

def find_mutual_pairs(dist_vec1, dist_vec2, ind_emb1_unique, ind_emb2_unique,
                             args, device, use_gpu=True, is_normalized=False,
                             approximate_mnn=False):
    """
    Unified function to find mutual pairs using either GPU-optimized or CPU fallback method.

    Args:
        dist_vec1, dist_vec2: Distance vectors (L2-normalized when is_normalized=True)
        ind_emb1_unique, ind_emb2_unique: Original indices
        args: Arguments containing topk, distance_metric, etc.
        device: GPU device (unused for CPU path)
        use_gpu: Whether to use GPU acceleration
        is_normalized: If True, skip normalization check (vectors already L2-normalized)
        approximate_mnn: If True, use single FAISS search + reverse dot-product verification
                         (~2x faster, suitable for ensemble voting pipelines)

    Returns:
        mutual_pairs: List of (i, nearest_i, distance) tuples
        mutual_nn: Number of mutual pairs
        correct: Number of correct pairs
    """
    from utils.graph_util import get_dists
    if use_gpu and torch.cuda.is_available():
        # Keep both dist_vec1 and dist_vec2 as CPU tensors/numpy.
        # get_topk handles all GPU memory management internally (IVF or brute-force).
        if torch.is_tensor(dist_vec1):
            dist_vec1 = dist_vec1.float().cpu()
        if torch.is_tensor(dist_vec2):
            dist_vec2 = dist_vec2.float().cpu()

        ind_emb1_tensor = torch.from_numpy(ind_emb1_unique).to(device).long()
        ind_emb2_tensor = torch.from_numpy(ind_emb2_unique).to(device).long()

        # Skip normalization check if caller guarantees pre-normalized vectors
        if not is_normalized:
            dist_vec1 = torch.nn.functional.normalize(dist_vec1, p=2, dim=1)
            if torch.is_tensor(dist_vec2):
                dist_vec2 = torch.nn.functional.normalize(dist_vec2, p=2, dim=1)

        n1 = dist_vec1.shape[0]
        n2 = dist_vec2.shape[0] if torch.is_tensor(dist_vec2) else dist_vec2.shape[0]
        topk = args.topk

        # Forward search: for each point in vec2, find top-k in vec1
        import time as _time
        _t_faiss_start = _time.time()
        nearest_ind1, nearest_dist1 = get_topk(dist_vec1, dist_vec2, k=topk,
                          metric=args.distance_metric, return_dist=True,
                          csls_neighborhood=args.csls_neighborhood, use_faiss=True,
                          is_normalized=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        _t_faiss_end = _time.time()

        if not torch.is_tensor(nearest_ind1):
            nearest_ind1 = torch.from_numpy(nearest_ind1).to(device)
        else:
            nearest_ind1 = nearest_ind1.to(device)
        if not torch.is_tensor(nearest_dist1):
            nearest_dist1 = torch.from_numpy(nearest_dist1).to(device)
        else:
            nearest_dist1 = nearest_dist1.to(device)
        actual_device = nearest_ind1.device

        if approximate_mnn:
            # --- Single-search approximate MNN ---
            # Verify mutuality via batched dot products on candidate pairs only.

            # Ensure both are CPU tensors for indexing
            if not torch.is_tensor(dist_vec1):
                dist_vec1_t = torch.from_numpy(np.ascontiguousarray(dist_vec1, dtype=np.float32))
            else:
                dist_vec1_t = dist_vec1.float().cpu()
            if not torch.is_tensor(dist_vec2):
                dist_vec2_t = torch.from_numpy(np.ascontiguousarray(dist_vec2, dtype=np.float32))
            else:
                dist_vec2_t = dist_vec2.float().cpu()

            # Flatten candidate pairs (on GPU)
            i_all = torch.arange(n2, device=actual_device).unsqueeze(1).expand(-1, topk).flatten()
            j_all = nearest_ind1.long().flatten()
            total_pairs = i_all.shape[0]

            valid_mask = (j_all >= 0) & (j_all < n1)
            j_safe = j_all.clamp(min=0, max=n1 - 1)

            # Chunked dot-product: index CPU tensors, compute on GPU
            pair_sims = torch.full((total_pairs,), -float('inf'), device=actual_device)
            dot_chunk = 500_000
            for start in range(0, total_pairs, dot_chunk):
                end = min(start + dot_chunk, total_pairs)
                chunk_valid = valid_mask[start:end]
                if chunk_valid.any():
                    j_idx = j_safe[start:end][chunk_valid].cpu()
                    i_idx = i_all[start:end][chunk_valid].cpu()
                    v1_rows = dist_vec1_t[j_idx].to(actual_device)
                    v2_rows = dist_vec2_t[i_idx].to(actual_device)
                    pair_sims[start:end][chunk_valid] = (v1_rows * v2_rows).sum(dim=1)
                    del v1_rows, v2_rows

            # For each unique j, find which candidate i has highest similarity
            # Only scatter valid pairs (j_all >= 0)
            best_sim_per_j = torch.full((n1,), -float('inf'), device=actual_device)
            if valid_mask.all():
                best_sim_per_j.scatter_reduce_(0, j_all, pair_sims, reduce='amax')
            else:
                best_sim_per_j.scatter_reduce_(0, j_safe[valid_mask], pair_sims[valid_mask], reduce='amax')

            # A pair is "mutual" if it achieves the max similarity for its j
            is_best = (pair_sims == best_sim_per_j[j_safe]).reshape(n2, topk)
            # Invalidate any entries with bad FAISS indices
            is_best = is_best & valid_mask.reshape(n2, topk)

            # For each i, find the first k where it's the best reverse candidate
            has_mutual = is_best.any(dim=1)
            first_mutual_k = torch.where(
                is_best,
                torch.arange(topk, device=actual_device).unsqueeze(0).expand(n2, -1),
                torch.full((n2, topk), topk, device=actual_device)
            ).min(dim=1)[0]

        else:
            # --- Exact MNN with two FAISS searches ---
            nearest_ind2, nearest_dist2 = get_topk(dist_vec2, dist_vec1, k=topk,
                                    metric=args.distance_metric, return_dist=True,
                                    csls_neighborhood=args.csls_neighborhood, use_faiss=True,
                                    is_normalized=True)

            if not torch.is_tensor(nearest_ind2):
                nearest_ind2 = torch.from_numpy(nearest_ind2).to(actual_device)
            else:
                nearest_ind2 = nearest_ind2.to(actual_device)

            # Vectorized mutual NN detection
            idx_expanded = torch.arange(n2, device=actual_device).unsqueeze(1).expand(-1, topk)
            neighbors_flat = nearest_ind1.flatten()
            neighbor_lists = nearest_ind2[neighbors_flat].reshape(n2, topk, topk)
            is_mutual = (idx_expanded.unsqueeze(2) == neighbor_lists).any(dim=2)

            has_mutual = is_mutual.any(dim=1)
            first_mutual_k = torch.where(
                is_mutual,
                torch.arange(topk, device=actual_device).unsqueeze(0).expand(n2, -1),
                torch.full((n2, topk), topk, device=actual_device)
            ).min(dim=1)[0]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        _t_mnn_end = _time.time()
        from loguru import logger as _logger
        _logger.debug(f"  find_mutual_pairs timing: faiss={_t_faiss_end - _t_faiss_start:.1f}s, "
                     f"mnn={_t_mnn_end - _t_faiss_end:.1f}s")

        # Extract mutual pairs (shared by both paths)
        valid_points = torch.where(has_mutual)[0]
        valid_k = first_mutual_k[has_mutual]
        i_indices = valid_points
        j_indices = nearest_ind1[valid_points, valid_k]
        distances = nearest_dist1[valid_points, valid_k]

        mutual_pairs = list(zip(i_indices.cpu().numpy(),
                               j_indices.cpu().numpy(),
                               distances.cpu().numpy()))

        deduplicated_pairs = deduplicate_pairs(mutual_pairs)

        correct = 0
        for i, j, dist in deduplicated_pairs:
            orig_j = ind_emb1_tensor[j].item()
            orig_i = ind_emb2_tensor[i].item()
            if orig_i == orig_j:
                correct += 1

        return deduplicated_pairs, len(deduplicated_pairs), correct
    else:
        # CPU fallback
        nearest_ind1, nearest_dist1 = get_topk(dist_vec1, dist_vec2, k=args.topk,
                                   metric=args.distance_metric, return_dist=True,
                                   csls_neighborhood=args.csls_neighborhood)
        nearest_ind2, nearest_dist2 = get_topk(dist_vec2, dist_vec1, k=args.topk,
                                   metric=args.distance_metric, return_dist=True,
                                   csls_neighborhood=args.csls_neighborhood)

        # OPTIMIZED: Convert tensors to numpy ONCE before loop to avoid per-iteration .item() overhead
        if torch.is_tensor(nearest_ind1):
            nearest_ind1 = nearest_ind1.cpu().numpy()
        if torch.is_tensor(nearest_ind2):
            nearest_ind2 = nearest_ind2.cpu().numpy()
        if torch.is_tensor(nearest_dist1):
            nearest_dist1 = nearest_dist1.cpu().numpy()

        subset_mutual_pairs = []
        correct = 0
        mutual_nn = 0

        for i, neighbors_of_i in enumerate(nearest_ind1):
            for k_idx in range(args.topk):
                nearest_i = int(neighbors_of_i[k_idx])  # Direct numpy indexing, no .item()
                neighbors_of_nearest_i = nearest_ind2[nearest_i]
                if i in neighbors_of_nearest_i:
                    mutual_nn += 1
                    if ind_emb2_unique[i] == ind_emb1_unique[nearest_i]:
                        correct += 1

                    # Direct numpy indexing, no .item() needed
                    dist_between_pair = float(nearest_dist1[i, k_idx])
                    # (ind2, ind1, dist)
                    subset_mutual_pairs.append((i, nearest_i, dist_between_pair))
                    break

        deduplicated_pairs = deduplicate_pairs(subset_mutual_pairs)

        # Recalculate correct count for deduplicated pairs
        correct = 0
        for i, j, dist in deduplicated_pairs:
            if ind_emb2_unique[i] == ind_emb1_unique[j]:
                correct += 1

        return deduplicated_pairs, len(deduplicated_pairs), correct

