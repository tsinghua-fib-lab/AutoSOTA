"""
Multi-threaded pipeline for async training with SSD offloading.

Implements three-thread architecture:
1. GPU Training Thread: Forward/backward/optimization
2. D2H Sync Thread: Async GPU->RAM transfers
3. Prefetch/Loader Thread: SSD->RAM loading and eviction
"""

import threading
import time
from queue import Queue, Empty
from typing import List, Dict, Optional, Callable
import torch
import numpy as np
from fast_tsp import find_tour as fast_tsp_find_tour

from .tiered_cache_manager import TieredCacheManager
from .gaussian_block import GaussianBlock, FrustumCuller


class AsyncPipeline:
    """
    Asynchronous pipeline for billion-scale Gaussian Splatting training.

    Coordinates three parallel paths:
    - GPU training (forward/backward/step)
    - Urgent SSD->RAM fetch for the current working set
    - Background future prefetch for upcoming working sets
    """

    def __init__(
        self,
        cache_manager: TieredCacheManager,
        frustum_culler: FrustumCuller,
        block_size: int,
        point_dim: int = 59,
        prefetch_ahead: int = 3,
        verbose: bool = True,
    ):
        self.cache = cache_manager
        self.culler = frustum_culler
        self.block_size = block_size
        self.point_dim = point_dim
        self.prefetch_ahead = prefetch_ahead
        self.verbose = bool(verbose)

        # CUDA streams for async operations
        self.compute_stream = torch.cuda.Stream()
        self.d2h_stream = torch.cuda.Stream()

        # Communication queues
        self.urgent_prefetch_queue: Queue = Queue(maxsize=10)
        self.future_prefetch_queue: Queue = Queue(maxsize=20)
        self.completed_queue: Queue = Queue(maxsize=10)

        # Threads
        self.prefetch_thread = None
        self.future_prefetch_thread = None
        self.running = False

        # Active blocks on GPU
        self.gpu_blocks: Dict[int, torch.Tensor] = {}
        self.gpu_lock = threading.Lock()

        # Statistics
        self.stats = {
            'iterations': 0,
            'gpu_time': 0.0,
            'load_time': 0.0,
            'sync_time': 0.0,
            'urgent_prefetch_time': 0.0,
            'urgent_prefetch_jobs': 0,
            'urgent_prefetch_blocks': 0,
            'urgent_wait_time': 0.0,
            'urgent_wait_calls': 0,
            'urgent_wait_timeouts': 0,
            'future_prefetch_time': 0.0,
            'future_prefetch_jobs': 0,
            'future_prefetch_blocks': 0,
        }

    def start(self):
        """Start background threads."""
        self.running = True

        self.prefetch_thread = threading.Thread(
            target=self._urgent_prefetch_worker,
            daemon=True,
        )
        self.prefetch_thread.start()

        self.future_prefetch_thread = threading.Thread(
            target=self._future_prefetch_worker,
            daemon=True,
        )
        self.future_prefetch_thread.start()

        self._log("[Pipeline] Async pipeline started")

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def _urgent_prefetch_worker(self):
        while self.running:
            try:
                iteration, needed = self.urgent_prefetch_queue.get(timeout=0.1)
            except Empty:
                continue

            try:
                t0 = time.time()
                loaded = self.cache.prefetch(needed)
                t1 = time.time()
                elapsed = t1 - t0
                self.stats['load_time'] += elapsed
                self.stats['urgent_prefetch_time'] += elapsed
                self.stats['urgent_prefetch_jobs'] += 1
                self.stats['urgent_prefetch_blocks'] += int(len(loaded))
                self.completed_queue.put((iteration, loaded, elapsed))
            except Exception as e:
                print(f"[Pipeline] Urgent prefetch error: {e}")
            finally:
                self.urgent_prefetch_queue.task_done()

    def _future_prefetch_worker(self):
        while self.running:
            try:
                iteration, future = self.future_prefetch_queue.get(timeout=0.1)
            except Empty:
                continue

            try:
                if future:
                    t0 = time.time()
                    loaded_count = self.cache.prefetch_future(future)
                    t1 = time.time()
                    self.stats['future_prefetch_time'] += t1 - t0
                    self.stats['future_prefetch_jobs'] += 1
                    self.stats['future_prefetch_blocks'] += int(loaded_count)
            except Exception as e:
                print(f"[Pipeline] Future prefetch error: {e}")
            finally:
                self.future_prefetch_queue.task_done()

    def request_prefetch(
        self,
        iteration: int,
        needed_blocks: List[int],
        future_blocks: Optional[List[int]] = None
    ):
        """
        Request prefetch of blocks (non-blocking).

        Urgent blocks for the current working set and future blocks for background
        warm-up are scheduled independently so future prefetch cannot block the
        current iteration's SSD->RAM fetch.
        """
        try:
            self.urgent_prefetch_queue.put_nowait((iteration, needed_blocks))
        except Exception:
            pass

        if future_blocks:
            try:
                self.future_prefetch_queue.put_nowait((iteration, future_blocks))
            except Exception:
                pass

    def wait_for_prefetch(self, iteration: int, timeout: float = 10.0) -> Dict[int, torch.Tensor]:
        start_time = time.time()
        self.stats['urgent_wait_calls'] += 1

        while time.time() - start_time < timeout:
            try:
                iter_num, loaded_blocks, load_time = self.completed_queue.get(timeout=0.1)
                if iter_num == iteration:
                    self.stats['urgent_wait_time'] += time.time() - start_time
                    return loaded_blocks
            except Empty:
                continue

        self.stats['urgent_wait_timeouts'] += 1
        raise TimeoutError(f"Prefetch for iteration {iteration} timed out")

    def load_blocks_to_gpu(
        self,
        block_dict: Dict[int, torch.Tensor],
        device: torch.device
    ) -> Dict[int, torch.Tensor]:
        gpu_blocks = {}

        with torch.cuda.stream(self.compute_stream):
            for block_id, cpu_tensor in block_dict.items():
                gpu_tensor = cpu_tensor.to(device, non_blocking=True)
                gpu_blocks[block_id] = gpu_tensor

        with self.gpu_lock:
            self.gpu_blocks.update(gpu_blocks)

        return gpu_blocks

    def sync_blocks_to_ram(
        self,
        block_ids: List[int],
        updated_gpu_tensors: List[torch.Tensor]
    ):
        self.cache.sync_from_gpu(
            block_ids=block_ids,
            gpu_tensors=updated_gpu_tensors,
            cuda_stream=self.d2h_stream
        )

    def train_step(
        self,
        block_ids: List[int],
        train_fn: Callable[[Dict[int, torch.Tensor]], torch.Tensor],
        optimizer: torch.optim.Optimizer
    ) -> torch.Tensor:
        with self.gpu_lock:
            active_blocks = {
                block_id: self.gpu_blocks[block_id]
                for block_id in block_ids
                if block_id in self.gpu_blocks
            }

        if len(active_blocks) != len(block_ids):
            missing = set(block_ids) - set(active_blocks.keys())
            raise RuntimeError(f"Missing GPU blocks: {missing}")

        with torch.cuda.stream(self.compute_stream):
            t0 = time.time()
            loss = train_fn(active_blocks)
            optimizer.step()
            optimizer.zero_grad()
            t1 = time.time()
            self.stats['gpu_time'] += t1 - t0

        return loss

    def get_stats(self) -> Dict:
        total_time = (
            self.stats['gpu_time']
            + self.stats['load_time']
            + self.stats['sync_time']
            + self.stats['future_prefetch_time']
        )

        return {
            **self.stats,
            'cache_stats': self.cache.get_stats(),
            'storage_stats': self.cache.storage.get_stats(),
            'total_time': total_time,
        }

    def shutdown(self):
        """Shutdown pipeline and all threads."""
        self._log("[Pipeline] Shutting down...")

        self.running = False

        if self.prefetch_thread:
            self.prefetch_thread.join(timeout=5.0)
        if self.future_prefetch_thread:
            self.future_prefetch_thread.join(timeout=5.0)

        self.compute_stream.synchronize()
        self.d2h_stream.synchronize()

        stats = self.get_stats()
        cache_stats = stats.get("cache_stats", {})
        print(
            "[Pipeline] Shutdown summary: "
            f"urgent_blocks={stats.get('urgent_prefetch_blocks', 0)} "
            f"future_blocks={stats.get('future_prefetch_blocks', 0)} "
            f"cache_hit_rate={cache_stats.get('hit_rate', 0.0):.3f} "
            f"ram_mb={cache_stats.get('ram_usage_mb', 0.0):.1f}"
        )


class TSPScheduler:
    """
    TSP-based scheduler for determining optimal block loading order.

    Uses camera clustering, spatial distance, and **view-direction-aware
    hybrid distance** to minimize SSD seeks and cache misses.

    Hybrid Distance Formula:
        D(a, b) = ||P_a - P_b||_2  +  lambda * (1 - cos(theta_ab))

    where lambda is auto-scaled to the scene's median inter-camera distance,
    so the angular penalty is always commensurate with the spatial term
    regardless of absolute scene scale (MatrixCity ~hundreds of meters vs
    indoor ~0.1 meters).
    """

    def __init__(
        self,
        camera_positions: np.ndarray,
        camera_directions: np.ndarray,
        camera_clusters: np.ndarray,
        block_hotspots: Dict[int, float],
        angular_weight: Optional[float] = None,
    ):
        """
        Initialize TSP scheduler.

        Args:
            camera_positions: Array of camera positions (num_cameras, 3)
            camera_directions: Unit view direction vectors (num_cameras, 3).
                Extracted as the third column of the C2W rotation matrix
                (OpenCV convention: camera looks along +Z).
            camera_clusters: Cluster assignment for each camera (num_cameras,)
            block_hotspots: Dictionary mapping block_id -> visibility score
            angular_weight: Override for lambda.  If None (default), lambda
                is auto-computed as 0.5 * median(pairwise spatial distances).
        """
        self.camera_positions = camera_positions
        self.camera_directions = camera_directions
        self.camera_clusters = camera_clusters
        self.block_hotspots = block_hotspots

        # ====================================================================
        # [NEW] Adaptive lambda: scene-scale-aware angular penalty weight
        # ====================================================================
        # Rationale for median:
        #   - Mean is biased by outlier camera pairs at scene extremes.
        #   - Median represents the "typical" inter-camera hop distance,
        #     making the angular penalty ~50% of a typical hop when two
        #     cameras face completely opposite directions.
        #   - Factor 0.5: angular term max is 2 (anti-parallel), so peak
        #     penalty = 0.5 * median * 2 = median, i.e. at most one extra
        #     "typical hop" of penalty.
        # ====================================================================
        if angular_weight is not None:
            self.angular_weight = float(angular_weight)
            print(f"[TSP Schedule] Angular weight (manual): λ = {self.angular_weight:.4f}")
        else:
            self.angular_weight = self._auto_lambda(camera_positions)

        # Compute cluster order using fast_tsp (2-opt improved TSP)
        self.cluster_order = self._compute_tsp_order()

    # ====================================================================
    # Static / private helpers for distance matrix construction
    # ====================================================================

    @staticmethod
    def _auto_lambda(positions: np.ndarray, sample_cap: int = 5000) -> float:
        """
        Compute adaptive angular penalty weight from scene geometry.

        For very large camera sets (>sample_cap), we subsample to avoid
        O(N^2) memory on 50k cameras (~20 GB for float64).

        Returns:
            lambda = 0.5 * median(pairwise Euclidean distances)
        """
        n = len(positions)
        if n > sample_cap:
            rng = np.random.RandomState(42)
            idx = rng.choice(n, sample_cap, replace=False)
            pts = positions[idx]
        else:
            pts = positions

        sq = np.sum(pts ** 2, axis=1)
        dist_sq = sq[:, None] + sq[None, :] - 2 * np.dot(pts, pts.T)
        dist = np.sqrt(np.maximum(dist_sq, 0))

        # Extract upper triangle (exclude diagonal zeros)
        upper = dist[np.triu_indices_from(dist, k=1)]
        median_dist = float(np.median(upper))
        lam = 0.5 * median_dist

        print(f"[TSP Schedule] Auto λ: median_dist={median_dist:.4f}, λ=0.5×median={lam:.4f}")
        return lam

    @staticmethod
    def _spatial_distance_matrix(positions: np.ndarray) -> np.ndarray:
        """
        Vectorized pairwise Euclidean distance matrix (float64).

        Formula: ||a-b||² = ||a||² + ||b||² - 2<a,b>
        """
        sq = np.sum(positions ** 2, axis=1)
        dist_sq = sq[:, None] + sq[None, :] - 2 * np.dot(positions, positions.T)
        return np.sqrt(np.maximum(dist_sq, 0))

    @staticmethod
    def _angular_penalty_matrix(directions: np.ndarray) -> np.ndarray:
        """
        Vectorized pairwise angular penalty matrix.

        penalty(a, b) = 1 - cos(theta_ab) = 1 - d_a · d_b
        Range: [0, 2]  (0 = same direction, 2 = opposite)

        Args:
            directions: (N, 3) unit vectors
        Returns:
            (N, N) float64 matrix in [0, 2]
        """
        # cos_sim[i,j] = d_i · d_j  ∈ [-1, 1]
        cos_sim = np.dot(directions, directions.T)
        # Clamp for numerical safety
        np.clip(cos_sim, -1.0, 1.0, out=cos_sim)
        return 1.0 - cos_sim

    def _hybrid_distance_matrix(
        self,
        positions: np.ndarray,
        directions: np.ndarray,
    ) -> np.ndarray:
        """
        Build the full hybrid distance matrix (spatial + angular).

            D(a,b) = ||P_a - P_b||  +  λ · (1 - cos θ_ab)

        Args:
            positions:  (N, 3) camera positions
            directions: (N, 3) unit view directions
        Returns:
            (N, N) float64 hybrid distance matrix
        """
        spatial = self._spatial_distance_matrix(positions)
        angular = self._angular_penalty_matrix(directions)
        return spatial + self.angular_weight * angular

    @staticmethod
    def _to_int_distance_matrix(float_matrix: np.ndarray) -> np.ndarray:
        """
        Convert a float distance matrix to int32 (×1000, rounded).

        fast_tsp.find_tour requires integer distances.
        Same convention as BlockScheduler.
        """
        return np.round(float_matrix * 1000).astype(np.int32)

    def _compute_tsp_order(self) -> List[int]:
        """
        Compute TSP tour over camera clusters using fast_tsp (2-opt improved)
        with hybrid distance (spatial + mean angular penalty per cluster).

        Returns:
            List of cluster IDs in visit order
        """
        num_clusters = len(np.unique(self.camera_clusters))

        if num_clusters <= 1:
            return list(range(num_clusters))

        # Compute cluster centroids AND mean view directions
        centroids = np.zeros((num_clusters, 3))
        mean_dirs = np.zeros((num_clusters, 3))
        for cluster_id in range(num_clusters):
            mask = self.camera_clusters == cluster_id
            centroids[cluster_id] = self.camera_positions[mask].mean(axis=0)
            avg_dir = self.camera_directions[mask].mean(axis=0)
            norm = np.linalg.norm(avg_dir)
            mean_dirs[cluster_id] = avg_dir / (norm + 1e-12)

        # Build hybrid distance matrix for clusters
        hybrid = self._hybrid_distance_matrix(centroids, mean_dirs)
        dist_matrix_int = self._to_int_distance_matrix(hybrid)
        tour = fast_tsp_find_tour(dist_matrix_int)

        print(f"[TSP Schedule] Inter-Cluster TSP: {num_clusters} clusters, "
              f"solver=fast_tsp (2-opt + hybrid distance, λ={self.angular_weight:.4f})")
        return tour

    def get_tsp_camera_order(self) -> List[int]:
        """
        [CORE TSP ALGORITHM] Get camera training schedule following TWO-LEVEL TSP order.
        
        This is the low-level TSP scheduling implementation. User code should call
        CLMGSStorageAdapter.get_training_schedule() instead.
        
        Level 1: Inter-Cluster TSP (already computed in __init__)
        Level 2: Intra-Cluster TSP (computed here for each cluster)
        
        This mirrors the BlockScheduler's two-level TSP strategy for maximum
        spatial coherence within and across clusters.

        Returns:
            List of camera indices in TSP-optimized training order
            
        Example:
            3 clusters with TSP order [1, 0, 2]:
            - Cluster 1: TSP([cam5, cam2, cam8]) → [cam2, cam5, cam8]
            - Cluster 0: TSP([cam1, cam3, cam0]) → [cam0, cam1, cam3]  
            - Cluster 2: TSP([cam7, cam4, cam6]) → [cam4, cam6, cam7]
            Output: [cam2, cam5, cam8, cam0, cam1, cam3, cam4, cam6, cam7]
            
        Note:
            ALL cameras in each cluster are used (no sampling).
            K-Means clustering already ensures balanced cluster sizes.
        """
        schedule = []
        
        # ====================================================================
        # [PROGRESS TRACKING] Show progress for large camera sets
        # ====================================================================
        num_clusters = len(self.cluster_order)
        print(f"[TSP Schedule] Computing intra-cluster TSP for {num_clusters} clusters...")
        t_start = time.time()

        for idx, cluster_id in enumerate(self.cluster_order):
            # Print progress every 10% or for each cluster if < 10 clusters
            if num_clusters <= 10 or (idx + 1) % max(1, num_clusters // 10) == 0:
                print(f"  Progress: {idx + 1}/{num_clusters} clusters ({(idx + 1) / num_clusters * 100:.1f}%)", end='\r')
            # Get cameras in this cluster
            mask = self.camera_clusters == cluster_id
            cluster_camera_indices = np.where(mask)[0]
            
            # ================================================================
            # [FIX] Always use ALL cameras in the cluster
            # ================================================================
            # Original code had a bug: it would randomly sample if cluster size > cameras_per_cluster
            # This breaks TSP spatial coherence and causes incomplete training.
            # 
            # Correct behavior: Use ALL cameras (K-Means already balanced cluster sizes)
            selected_indices = cluster_camera_indices
            
            # ================================================================
            # [NEW] Intra-Cluster TSP Ordering
            # ================================================================
            # Sort cameras within cluster by spatial proximity (like BlockScheduler)
            if len(selected_indices) > 1:
                # Get positions AND directions for cameras in this cluster
                cluster_positions  = self.camera_positions[selected_indices]
                cluster_directions = self.camera_directions[selected_indices]
                n = len(cluster_positions)
                
                # ============================================================
                # Build hybrid distance matrix (spatial + angular penalty)
                # ============================================================
                # D(a,b) = ||P_a - P_b||  +  λ · (1 - cos θ_ab)
                #
                # ✅ Pure numpy vectorization (BLAS gemm + broadcast)
                # ✅ λ is auto-scaled to scene geometry (see _auto_lambda)
                # ============================================================
                hybrid = self._hybrid_distance_matrix(cluster_positions,
                                                      cluster_directions)
                dist_matrix_int = self._to_int_distance_matrix(hybrid)
                
                if n >= 3:
                    # fast_tsp needs ≥3 nodes for meaningful 2-opt improvement
                    intra_tsp_tour = fast_tsp_find_tour(dist_matrix_int)
                else:
                    # 2 cameras: trivial ordering
                    intra_tsp_tour = list(range(n))
                
                # Reorder cameras according to intra-cluster TSP
                sorted_indices = selected_indices[intra_tsp_tour]
            else:
                # Single camera, no ordering needed
                sorted_indices = selected_indices
            
            schedule.extend(sorted_indices.tolist())
        
        # ====================================================================
        # [PROGRESS] Print completion summary
        # ====================================================================
        t_end = time.time()
        total_cameras = len(self.camera_positions)
        elapsed_ms = (t_end - t_start) * 1000
        
        print(f"\n[TSP Schedule] Completed: {total_cameras} cameras, {num_clusters} clusters")
        print(f"[TSP Schedule] Time: {elapsed_ms:.1f} ms ({elapsed_ms/num_clusters:.1f} ms/cluster)")

        return schedule
    
    @staticmethod
    def _greedy_tsp(distance_matrix: np.ndarray) -> np.ndarray:
        """
        Greedy nearest neighbor TSP approximation (legacy fallback).
        
        Kept for clusters with < 3 cameras where fast_tsp is not applicable.
        For normal use, fast_tsp_find_tour is preferred.
        
        Args:
            distance_matrix: Square matrix of pairwise distances
            
        Returns:
            Array of indices representing TSP tour
        """
        n = distance_matrix.shape[0]
        if n <= 1:
            return np.arange(n)
        
        unvisited = set(range(1, n))
        tour = [0]  # Start from first point
        
        while unvisited:
            current = tour[-1]
            # Find nearest unvisited point
            nearest = min(unvisited, key=lambda x: distance_matrix[current, x])
            tour.append(nearest)
            unvisited.remove(nearest)
        
        return np.array(tour)
