"""Core component."""

from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import torch
import time
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import cdist
import logging
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
import random


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm < 1e-8:
        return v.astype(np.float32, copy=True)
    return (v / norm).astype(np.float32, copy=True)


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = a / (np.linalg.norm(a) + 1e-8)
    b_norm = b / (np.linalg.norm(b) + 1e-8)
    return 1.0 - float(np.dot(a_norm, b_norm))


class _Cluster:
    def __init__(self, center: np.ndarray, count: int, center_raw: Optional[np.ndarray] = None):
        """Handle init."""
        self.center = _normalize(center)
        self.count = int(count)
        if center_raw is not None:
            self.center_raw = center_raw.copy()
        else:
            self.center_raw = center.copy()
            
def _hierarchical_cluster(feats_norm: np.ndarray, feats_raw: np.ndarray, target_k: int, linkage_method: str, distance_metric: str) -> List[_Cluster]:
    """Handle hierarchical cluster."""
    if feats_norm.shape[0] == 0:
        return []
    if feats_norm.shape[0] <= target_k:
        return [_Cluster(feats_norm[i], 1, center_raw=feats_raw[i]) for i in range(feats_norm.shape[0])]

    try:
        Z = linkage(feats_norm, method=linkage_method, metric=distance_metric)
        cluster_ids = fcluster(Z, t=target_k, criterion="maxclust")
    except Exception:
        return [_Cluster(feats_norm[i], 1, center_raw=feats_raw[i]) for i in range(feats_norm.shape[0])]

    clusters_map = defaultdict(list)
    clusters_raw_map = defaultdict(list)
    
    for i, cid in enumerate(cluster_ids):
        clusters_map[cid].append(feats_norm[i])
        clusters_raw_map[cid].append(feats_raw[i])

    out: List[_Cluster] = []
    for cid in clusters_map:
        vectors = np.stack(clusters_map[cid])
        vectors_raw = np.stack(clusters_raw_map[cid])
        
        center = vectors.mean(axis=0)
        center_raw = vectors_raw.mean(axis=0)
        
        out.append(_Cluster(center, vectors.shape[0], center_raw=center_raw))
    return out

def _merge_close_clusters(clusters: List[_Cluster], tau: float) -> List[_Cluster]:
    """Handle merge close clusters."""
    if len(clusters) <= 1:
        return clusters

    centers = np.stack([c.center for c in clusters], axis=0)
    dmat = cdist(centers, centers, metric="cosine")

    merged = [False] * len(clusters)
    new_clusters: List[_Cluster] = []

    for i in range(len(clusters)):
        if merged[i]:
            continue
        close_idxs = [i]
        for j in range(i + 1, len(clusters)):
            if dmat[i, j] <= tau:
                merged[j] = True
                close_idxs.append(j)
        total_count = sum(clusters[k].count for k in close_idxs)
        weighted_center = sum(clusters[k].center * clusters[k].count for k in close_idxs) / float(
            max(total_count, 1)
        )
        weighted_center_raw = sum(clusters[k].center_raw * clusters[k].count for k in close_idxs) / float(
            max(total_count, 1)
        )
        new_clusters.append(_Cluster(weighted_center, total_count, center_raw=weighted_center_raw))

    return new_clusters


def _spherical_interpolate(v1: np.ndarray, v2: np.ndarray, t: float) -> np.ndarray:
    """Handle spherical interpolate."""
    v1_norm = _normalize(v1)
    v2_norm = _normalize(v2)
    
    dot = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
    
    if abs(dot) > 0.9995:
        return _normalize((1 - t) * v1_norm + t * v2_norm)
    
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    
    w1 = np.sin((1 - t) * theta) / sin_theta
    w2 = np.sin(t * theta) / sin_theta
    
    result = w1 * v1_norm + w2 * v2_norm
    return _normalize(result)

def _simplified_soinn_on_clusters(
    cluster_centers: List[np.ndarray],     # Normalized
    cluster_centers_raw: List[np.ndarray], # Raw (Unnormalized)
    cluster_counts: List[int],
    ad: int = 20,
    lam: int = 20,
    threshold_scale: float = 0.5,
    max_iterations: int = 3,
    max_degree_for_removal: int = 1,
) -> List[_Cluster]:
    """Handle simplified soinn on clusters."""
    if len(cluster_centers) == 0:
        return [], {}
    
    nodes = [_normalize(c.copy()) for c in cluster_centers]
    nodes_raw = [c.copy() for c in cluster_centers_raw]
    
    win_counts = cluster_counts.copy()
    edges = defaultdict(dict)
    
    n_nodes = len(nodes)
    if n_nodes >= 2:
        for i in range(n_nodes):
            dists = [_cosine_distance(nodes[i], nodes[j]) if i != j else float('inf') for j in range(n_nodes)]
            if len(dists) > 0:
                nearest_idx = np.argmin(dists)
                edges[i][nearest_idx] = 0
                edges[nearest_idx][i] = 0
    
    input_signals = list(zip(cluster_centers, cluster_centers_raw))
    
    # IMPORTANT (Determinism):
    # This refinement previously used python's global `random.shuffle(indices)`, which makes results
    # depend on the global RNG state of each worker process. Enabling/disabling STAR (or any other
    # feature) can consume RNG calls before `compress()`, changing the shuffle order and producing
    # different prototypes even on Task0 (where STAR hasn't aligned anything yet).
    #
    # Use a local RNG with a fixed seed so refinement order is reproducible and independent of
    # global RNG / multiprocessing for comparable experiments.
    _local_rng = np.random.RandomState(0)
    
    for iteration in range(max_iterations):
        indices = list(range(len(input_signals)))
        if len(indices) > 1:
            _local_rng.shuffle(indices)
        
        for t, idx in enumerate(indices, start=1):
            x, x_raw = input_signals[idx]
            x_norm = _normalize(x)
            
            if len(nodes) < 2: continue
            
            dists = np.array([_cosine_distance(x_norm, w) for w in nodes])
            sorted_idx = np.argsort(dists)
            s1, s2 = sorted_idx[0], sorted_idx[1]
            
            
            if s2 not in edges[s1]:
                edges[s1][s2] = 0
                edges[s2][s1] = 0
            
            for nbr in list(edges[s1].keys()):
                edges[s1][nbr] += 1
                if s1 in edges[nbr]: edges[nbr][s1] = edges[s1][nbr]
                if edges[s1][nbr] > ad:
                    del edges[s1][nbr]
                    if s1 in edges[nbr]: del edges[nbr][s1]
            
            eta1 = 1.0 / float(t + iteration * len(input_signals) + 1)
            eta2 = 1.0 / (100.0 * float(t + iteration * len(input_signals) + 1))
            
            nodes[s1] = _spherical_interpolate(nodes[s1], x_norm, eta1)
            
            # w_raw += eta * (x_raw - w_raw)
            nodes_raw[s1] = nodes_raw[s1] + eta1 * (x_raw - nodes_raw[s1])
            
            for nbr in list(edges[s1].keys()):
                nodes[nbr] = _spherical_interpolate(nodes[nbr], x_norm, eta2)
                nodes_raw[nbr] = nodes_raw[nbr] + eta2 * (x_raw - nodes_raw[nbr])
                
            win_counts[s1] += 1

        # ...
        to_remove = [i for i in range(len(nodes)) if len(edges[i]) <= max_degree_for_removal]
        if len(to_remove) > 0 and len(nodes) - len(to_remove) >= 2:
            keep_mask = [i not in to_remove for i in range(len(nodes))]
            
            old_to_new = {}
            new_nodes, new_nodes_raw, new_counts = [], [], []
            for old_idx, keep in enumerate(keep_mask):
                if keep:
                    old_to_new[old_idx] = len(new_nodes)
                    new_nodes.append(nodes[old_idx])
                    new_nodes_raw.append(nodes_raw[old_idx])
                    new_counts.append(win_counts[old_idx])
            
            new_edges = defaultdict(dict)
            for old_i, nbrs in edges.items():
                if old_i in old_to_new:
                    new_i = old_to_new[old_i]
                    for old_j in nbrs:
                        if old_j in old_to_new:
                            new_edges[new_i][old_to_new[old_j]] = edges[old_i][old_j]
            
            nodes, nodes_raw, win_counts, edges = new_nodes, new_nodes_raw, new_counts, new_edges
            
            for i in range(len(nodes)):
                if len(edges[i]) == 0 and len(nodes) > 1:
                    dists = [_cosine_distance(nodes[i], nodes[j]) if i != j else float('inf') for j in range(len(nodes))]
                    nearest = np.argmin(dists)
                    edges[i][nearest] = 0
                    edges[nearest][i] = 0

    result = []
    final_edges_map = {}
    
    final_indices = [i for i in range(len(nodes)) if len(edges[i]) > 0]
    if not final_indices: final_indices = range(len(nodes))
    
    for new_idx, old_idx in enumerate(final_indices):
        result.append(_Cluster(nodes[old_idx], win_counts[old_idx], center_raw=nodes_raw[old_idx]))
        final_edges_map[new_idx] = set()
        for nbr in edges[old_idx]:
             if nbr in final_indices:
                 final_edges_map[new_idx].add(final_indices.index(nbr))
                 
    return result, final_edges_map

def _compress_class_worker(args):
    
    (cls, feats, target_k, tau_merge, linkage_method, distance_metric, max_prototypes,
     use_soinn_refinement, soinn_ad, soinn_lam, soinn_threshold_scale, soinn_max_iter,
     soinn_max_degree_for_removal) = args
    
    feats_raw = feats.astype(np.float32)
    feats_norm = feats_raw / (np.linalg.norm(feats_raw, axis=1, keepdims=True) + 1e-8)
    
    clusters = _hierarchical_cluster(feats_norm, feats_raw, target_k, linkage_method, distance_metric)
    hierarchical_count = len(clusters)
    
    soinn_edges = {}
    if use_soinn_refinement and len(clusters) > 1:
        cluster_centers = [c.center for c in clusters]
        cluster_centers_raw = [c.center_raw for c in clusters] 
        cluster_counts = [c.count for c in clusters]
        
        clusters, soinn_edges = _simplified_soinn_on_clusters(
            cluster_centers,
            cluster_centers_raw,
            cluster_counts,
            ad=soinn_ad,
            lam=soinn_lam,
            threshold_scale=soinn_threshold_scale,
            max_iterations=soinn_max_iter,
            max_degree_for_removal=soinn_max_degree_for_removal,
        )
    else:
        clusters = sorted(clusters, key=lambda c: c.count, reverse=True)
    
    
    if len(clusters) > 0:
        raw_norms = [np.linalg.norm(c.center_raw) for c in clusters]
        avg_raw_norm = sum(raw_norms) / len(raw_norms)
        # print(f"DEBUG WORKER {cls}: Avg Raw Norm = {avg_raw_norm:.4f}")

    final_count = len(clusters)
    return cls, clusters, hierarchical_count, final_count, soinn_edges

class HCSOINNClassifier:
    """Handle init."""

    def __init__(
        self,
        max_prototypes_per_class: Optional[int] = 20,
        alpha: float = 0.5,
        tau_merge: float = 0.2,
        tau_reject: float = 2.0,
        linkage_method: str = "average",
        distance_metric: str = "cosine",
        use_soinn_refinement: bool = True,
        soinn_ad: int = 20,
        soinn_lam: int = 20,
        soinn_threshold_scale: float = 0.5,
        soinn_max_iter: int = 3,
        soinn_max_degree_for_removal: int = 1,
        coarse_topk: Optional[int] = None,
        enable_inference_profiling: bool = False,
        profile_sync_cuda: bool = True,
    ) -> None:
        self.max_prototypes_per_class = None if max_prototypes_per_class is None else int(
            max_prototypes_per_class
        )
        self.alpha = float(alpha)
        self.tau_merge = float(tau_merge)
        self.tau_reject = float(tau_reject)
        self.linkage_method = linkage_method
        self.distance_metric = distance_metric
        
        self.use_soinn_refinement = bool(use_soinn_refinement)
        self.soinn_ad = int(soinn_ad)
        self.soinn_lam = int(soinn_lam)
        self.soinn_threshold_scale = float(soinn_threshold_scale)
        self.soinn_max_iter = int(soinn_max_iter)
        self.soinn_max_degree_for_removal = int(soinn_max_degree_for_removal)

        # Adaptive prototype allocation
        self.use_adaptive_proto = False
        self.total_proto_budget = None

        # Two-stage coarse filtering: use NCM to shortlist candidate classes
        # before computing expensive sub-cluster distances.
        # None = disabled (evaluate all classes); int = number of NCM candidates.
        self.coarse_topk = None if coarse_topk is None else int(coarse_topk)

        self.class_mu: Dict[int, np.ndarray] = {}  # Normalized NCM centers for inference
        self.class_mu_raw: Dict[int, np.ndarray] = {}  # Raw (unnormalized) NCM centers for STAR
        self.class_count: Dict[int, int] = {}

        self.class_clusters: Dict[int, List[_Cluster]] = {}
        
        # STAR: Store ORIGINAL clusters and NCM centers (before any transformation)
        # This allows re-alignment from scratch to avoid cumulative errors
        self.class_clusters_original: Dict[int, List[_Cluster]] = {}
        self.class_mu_original: Dict[int, np.ndarray] = {}  # Normalized
        self.class_mu_raw_original: Dict[int, np.ndarray] = {}  # Unnormalized

        # DEBUG: Store snapshots of clusters at different tasks for comparison
        # Format: {task_id: {cls: [centers_normalized, centers_raw]}}
        self.cluster_snapshots: Dict[int, Dict[int, tuple]] = {}
        
        # STAR: Frozen clusters for hard restore (Force Freeze Policy)
        # {cls: List[_Cluster]} - Stores deep copy of clusters
        self.frozen_clusters: Dict[int, List[_Cluster]] = {}

        self.buffers: Dict[int, List[np.ndarray]] = {}

        # ------------------------------------------------------------------ #
        # Inference acceleration cache (built lazily on first predict_topk call)
        # ------------------------------------------------------------------ #
        # Cache is invalidated whenever class centers / prototypes change (compress, alignment, etc.)
        self._predict_cache_dirty: bool = True
        self._predict_cache: Dict[str, object] = {}

        # ------------------------------------------------------------------ #
        # Inference profiling (optional; focused on predict_topk internals)
        # ------------------------------------------------------------------ #
        self.enable_inference_profiling = bool(enable_inference_profiling)
        self.profile_sync_cuda = bool(profile_sync_cuda)
        self._profile_stats: Dict[str, object] = {}
        self.reset_profile_stats()

    # ------------------------------------------------------------------ #
    # Cache helpers
    # ------------------------------------------------------------------ #
    def invalidate_cache(self) -> None:
        """Invalidate cached tensors used by predict_topk(). Safe to call often."""
        self._predict_cache_dirty = True
        self._predict_cache.clear()

    # ------------------------------------------------------------------ #
    # Inference profiling helpers
    # ------------------------------------------------------------------ #
    def reset_profile_stats(self) -> None:
        self._profile_stats = {
            "calls": 0,
            "samples": 0,
            "total_sec": 0.0,
            "steps_sec": defaultdict(float),
        }

    def set_inference_profiling(
        self,
        enabled: bool,
        reset: bool = False,
        sync_cuda: Optional[bool] = None,
    ) -> None:
        self.enable_inference_profiling = bool(enabled)
        if sync_cuda is not None:
            self.profile_sync_cuda = bool(sync_cuda)
        if reset:
            self.reset_profile_stats()

    def get_profile_stats(self, reset: bool = False) -> Dict[str, object]:
        calls = int(self._profile_stats.get("calls", 0))
        samples = int(self._profile_stats.get("samples", 0))
        total_sec = float(self._profile_stats.get("total_sec", 0.0))
        steps_raw = self._profile_stats.get("steps_sec", {})
        steps_sec = {k: float(v) for k, v in sorted(steps_raw.items(), key=lambda x: x[1], reverse=True)}
        out = {
            "calls": calls,
            "samples": samples,
            "total_sec": total_sec,
            "avg_ms_per_call": (total_sec / calls * 1000.0) if calls > 0 else 0.0,
            "avg_ms_per_sample": (total_sec / samples * 1000.0) if samples > 0 else 0.0,
            "steps_sec": steps_sec,
            "steps_ratio": {k: (v / total_sec if total_sec > 0 else 0.0) for k, v in steps_sec.items()},
        }
        if reset:
            self.reset_profile_stats()
        return out

    def _profile_sync(self, device: torch.device) -> None:
        if (
            self.enable_inference_profiling
            and self.profile_sync_cuda
            and isinstance(device, torch.device)
            and device.type == "cuda"
            and torch.cuda.is_available()
        ):
            torch.cuda.synchronize(device=device)

    def _profile_tic(self, device: torch.device) -> float:
        self._profile_sync(device)
        return time.perf_counter()

    def _profile_toc(self, step_name: str, t0: float, device: torch.device) -> float:
        self._profile_sync(device)
        dt = time.perf_counter() - t0
        self._profile_stats["steps_sec"][step_name] += dt
        return dt

    def _ensure_predict_cache(
        self,
        device: torch.device,
        query_dim: int,
        valid_classes: List[int],
    ) -> None:
        """
        Build/cache prototype tensors on the target device to avoid per-call numpy concat and H2D copies.
        Cache key depends on: device, query_dim, and the ordered valid_classes list.
        """
        device_key = str(device)
        classes_key = tuple(int(c) for c in valid_classes)

        if (
            (not self._predict_cache_dirty)
            and self._predict_cache.get("device_key") == device_key
            and self._predict_cache.get("query_dim") == int(query_dim)
            and self._predict_cache.get("classes_key") == classes_key
        ):
            return

        # ---------- NCM centers (already normalized) ----------
        ncm_centers_np = np.stack([self.class_mu[cls] for cls in valid_classes]).astype(np.float32, copy=False)
        ncm_centers_t = torch.from_numpy(ncm_centers_np).to(device=device, dtype=torch.float32)

        # ---------- Prototypes (sub-clusters; fallback to NCM if empty) ----------
        all_protos: List[np.ndarray] = []
        proto_labels: List[int] = []       # original class id per proto
        proto_class_index: List[int] = []  # index in valid_classes per proto (0..C-1)

        for class_index, cls in enumerate(valid_classes):
            clusters = self.class_clusters.get(cls, [])
            if clusters:
                cluster_dims = [c.center.shape[0] for c in clusters]
                if all(dim == query_dim for dim in cluster_dims):
                    cls_protos = np.stack([c.center for c in clusters]).astype(np.float32, copy=False)
                    all_protos.append(cls_protos)
                    n_p = int(cls_protos.shape[0])
                    proto_labels.extend([int(cls)] * n_p)
                    proto_class_index.extend([int(class_index)] * n_p)
                    continue

            # Fallback: at least one proto per class (use NCM center)
            mu = self.class_mu[cls].astype(np.float32, copy=False)
            all_protos.append(mu[np.newaxis, :])
            proto_labels.append(int(cls))
            proto_class_index.append(int(class_index))

        all_protos_np = np.concatenate(all_protos, axis=0).astype(np.float32, copy=False)
        all_protos_t = torch.from_numpy(all_protos_np).to(device=device, dtype=torch.float32)
        proto_labels_t = torch.tensor(proto_labels, device=device, dtype=torch.long)
        proto_class_index_t = torch.tensor(proto_class_index, device=device, dtype=torch.long)

        # Prototypes and NCM centers are expected to already be L2-normalized.
        # We keep them as-is to avoid extra normalize() cost per call.
        self._predict_cache = {
            "device_key": device_key,
            "query_dim": int(query_dim),
            "classes_key": classes_key,
            "ncm_centers_t": ncm_centers_t,               # [C, D]
            "all_protos_t": all_protos_t,                 # [M, D]
            "proto_labels_t": proto_labels_t,             # [M]
            "proto_class_index_t": proto_class_index_t,   # [M] in 0..C-1
        }
        self._predict_cache_dirty = False

    # ------------------------------------------------------------------ #
    # STAR: Force Freeze Methods
    # ------------------------------------------------------------------ #
    def freeze_nodes(self, cls: int) -> None:
        """Force backup current nodes state."""
        if cls in self.class_clusters:
            # Deep copy to ensure independence
            self.frozen_clusters[cls] = [
                _Cluster(
                    center=c.center.copy(), 
                    count=c.count, 
                    center_raw=c.center_raw.copy() if c.center_raw is not None else None
                ) 
                for c in self.class_clusters[cls]
            ]
            logging.info(f"[HC-SOINN] Class {cls} nodes FROZEN.")

    def restore_frozen_nodes(self, cls: int) -> None:
        """Force restore nodes state from frozen backup."""
        if cls in self.frozen_clusters:
            self.class_clusters[cls] = []
            for c in self.frozen_clusters[cls]:
                # Deep copy raw center
                center_raw = c.center_raw.copy() if c.center_raw is not None else None
                
                # STRICTLY re-calculate normalized center from raw to ensure consistency
                # This fixes the "Raw identical but Normalized different" issue
                if center_raw is not None:
                    norm = np.linalg.norm(center_raw)
                    if norm > 1e-9:
                        center = center_raw / norm
                    else:
                        center = c.center.copy() # Fallback
                else:
                    center = c.center.copy()
                    
                self.class_clusters[cls].append(_Cluster(center, c.count, center_raw))
                
            logging.info(f"[HC-SOINN] Class {cls} nodes RESTORED from frozen state (Normalized re-calculated).")

    # ------------------------------------------------------------------ #
    # DEBUG: Snapshot and comparison methods
    # ------------------------------------------------------------------ #
    def save_cluster_snapshot(self, task_id: int, class_list: Optional[List[int]] = None) -> None:
        """
        Save a snapshot of cluster centers for debugging.
        
        Args:
            task_id: Current task ID
            class_list: List of classes to save (None = all classes)
        """
        if task_id not in self.cluster_snapshots:
            self.cluster_snapshots[task_id] = {}
        
        classes_to_save = class_list if class_list is not None else list(self.class_clusters.keys())
        
        for cls in classes_to_save:
            if cls in self.class_clusters and len(self.class_clusters[cls]) > 0:
                clusters = self.class_clusters[cls]
                centers_normalized = np.array([c.center.copy() for c in clusters])
                centers_raw = np.array([c.center_raw.copy() for c in clusters])
                self.cluster_snapshots[task_id][cls] = (centers_normalized, centers_raw)
                
    
    def compare_cluster_snapshots(self, task_id1: int, task_id2: int, class_list: Optional[List[int]] = None) -> None:
        """
        Compare cluster snapshots between two tasks.
        
        Args:
            task_id1: First task ID
            task_id2: Second task ID
            class_list: List of classes to compare (None = all common classes)
        """
        if task_id1 not in self.cluster_snapshots or task_id2 not in self.cluster_snapshots:
            return
        
        snapshot1 = self.cluster_snapshots[task_id1]
        snapshot2 = self.cluster_snapshots[task_id2]
        
        classes_to_compare = class_list if class_list is not None else list(set(snapshot1.keys()) & set(snapshot2.keys()))
        
        # Debug logging removed for performance
        for cls in sorted(classes_to_compare):
            if cls not in snapshot1 or cls not in snapshot2:
                continue
            
            centers_norm1, centers_raw1 = snapshot1[cls]
            centers_norm2, centers_raw2 = snapshot2[cls]
            
            # Check if dimensions match
            if centers_norm1.shape != centers_norm2.shape:
                logging.warning(
                    f"[DEBUG] Class {cls}: Shape mismatch! "
                    f"Task {task_id1}: {centers_norm1.shape}, Task {task_id2}: {centers_norm2.shape}"
                )
                continue
            
            # Compute differences (for internal use only, no logging)
            diff_norm = np.abs(centers_norm1 - centers_norm2)
            diff_raw = np.abs(centers_raw1 - centers_raw2)
    
    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    def add_features(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Handle add features."""
        features = np.asarray(features, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int64)
        if features.shape[0] == 0:
            return
        
        for cls in np.unique(labels):
            cls_mask = labels == cls
            cls_feats = features[cls_mask]
            if cls_feats.shape[0] == 0:
                continue

            if cls not in self.buffers:
                self.buffers[cls] = []
            self.buffers[cls].append(cls_feats)

            cls_count_old = self.class_count.get(cls, 0)
            if cls in self.class_mu_raw:
                cls_sum_old = self.class_mu_raw[cls] * cls_count_old
            else:
                cls_sum_old = np.zeros(cls_feats.shape[1], dtype=np.float32)
            cls_sum_new = cls_sum_old + cls_feats.sum(axis=0)
            cls_count_new = cls_count_old + cls_feats.shape[0]
            self.class_count[cls] = cls_count_new
            # Store both raw and normalized versions
            cls_mean_raw = cls_sum_new / float(max(cls_count_new, 1))
            self.class_mu_raw[cls] = cls_mean_raw
            self.class_mu[cls] = _normalize(cls_mean_raw)
            
            # STAR: Save original NCM centers (only once, when first created)
            if cls not in self.class_mu_original:
                self.class_mu_raw_original[cls] = cls_mean_raw.copy()
                self.class_mu_original[cls] = _normalize(cls_mean_raw)

        # class_mu / buffers updated => invalidate inference cache
        self.invalidate_cache()

    def compress(self) -> None:
        """Handle compress."""
        # Pre-collect features for dispersion analysis
        class_feats = {}
        for cls, chunk_list in list(self.buffers.items()):
            if len(chunk_list) == 0:
                continue
            feats = np.concatenate(chunk_list, axis=0).astype(np.float32, copy=False)
            if cls in self.class_clusters and len(self.class_clusters[cls]) > 0:
                old_centers_raw = np.stack([c.center_raw for c in self.class_clusters[cls]], axis=0)
                feats = np.concatenate([feats, old_centers_raw], axis=0)
            class_feats[cls] = feats

        # Compute adaptive target_k when enabled
        adaptive_k = {}
        if self.use_adaptive_proto and len(class_feats) > 1:
            dispersion = {}
            for cls, feats in class_feats.items():
                f_norm = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
                dispersion[cls] = float(np.mean(np.var(f_norm, axis=0))) + 1e-8
            total_disp = sum(dispersion.values())
            nc = len(class_feats)
            max_ppc = self.max_prototypes_per_class or 60
            total_budget = self.total_proto_budget or (nc * max_ppc)
            min_k = max(8, min(15, max_ppc // 4))
            for cls in class_feats:
                k = max(min_k, int(total_budget * dispersion[cls] / total_disp))
                if self.max_prototypes_per_class is not None:
                    k = min(k, class_feats[cls].shape[0])
                adaptive_k[cls] = k

        tasks = []
        for cls, feats in class_feats.items():
            if self.use_adaptive_proto and cls in adaptive_k:
                target_k = adaptive_k[cls]
            else:
                target_k = feats.shape[0] if self.max_prototypes_per_class is None else min(
                    self.max_prototypes_per_class, feats.shape[0]
                )

            tasks.append((
                cls, feats, target_k, self.tau_merge,
                self.linkage_method, self.distance_metric, self.max_prototypes_per_class,
                self.use_soinn_refinement, self.soinn_ad, self.soinn_lam,
                self.soinn_threshold_scale, self.soinn_max_iter, self.soinn_max_degree_for_removal
            ))
        if tasks:
            logging.info(f"[HC-SOINN] Compressing {len(tasks)} classes using multiprocessing...")
            with ProcessPoolExecutor() as executor:
                results = list(executor.map(_compress_class_worker, tasks))
            
            for cls, clusters, hierarchical_count, final_count, soinn_edges in results:
                
                self.class_clusters[cls] = clusters
                
                # STAR: Save original clusters (only once, when first created)
                if cls not in self.class_clusters_original:
                    # Deep copy clusters to preserve original state
                    self.class_clusters_original[cls] = [
                        _Cluster(
                            center=c.center.copy(),
                            count=c.count,
                            center_raw=c.center_raw.copy() if c.center_raw is not None else None
                        )
                        for c in clusters
                    ]
                
                # DEBUG: Log new cluster statistics after replacement
                if len(clusters) > 0:
                    new_centers = np.array([c.center for c in clusters])
                    new_centers_raw = np.array([c.center_raw for c in clusters])
                    new_mean_norm = np.mean(np.linalg.norm(new_centers, axis=1))
                    new_mean_norm_raw = np.mean(np.linalg.norm(new_centers_raw, axis=1))
                    logging.info(
                        f"[HC-SOINN] Class {cls}: Created {len(clusters)} new nodes "
                        # f"(center_mean_norm={new_mean_norm:.6f}, center_raw_mean_norm={new_mean_norm_raw:.6f})"
                    )
                
                if not hasattr(self, 'class_edges'):
                    self.class_edges: Dict[int, Dict[int, Set[int]]] = {}
                self.class_edges[cls] = soinn_edges
                self.buffers[cls] = []
                
                if self.use_soinn_refinement:
                    logging.info(
                        f"[HC-SOINN] class {cls}: hierarchical_clusters={hierarchical_count} -> "
                        f"soinn_refined={final_count} (reduction: {hierarchical_count - final_count})"
                    )
                else:
                    logging.info(
                        f"[HC-SOINN] class {cls}: hierarchical_clusters={hierarchical_count} -> "
                        f"merged={final_count} (reduction: {hierarchical_count - final_count})"
                    )
        else:
            logging.info("[HC-SOINN] No buffer to compress.")

        # Prototypes replaced => invalidate inference cache
        self.invalidate_cache()

    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    def predict_topk(
        self, query_features, topk: int, total_classes: int, device=None,
        use_ease_reweighting: bool = False,
        cur_task: int = 0,
        task_increment: int = 10,
        init_cls: int = 10,
        ease_alpha: float = 0.5
    ) -> np.ndarray:
        """Handle predict topk."""
        if query_features.shape[0] == 0:
            return np.zeros((0, topk), dtype=np.int64)
            
        if device is None:
            device = torch.device("cpu")

        total_t0 = None
        if self.enable_inference_profiling:
            total_t0 = self._profile_tic(device)

        query_features = np.asarray(query_features, dtype=np.float32)
        query_dim = query_features.shape[1]
        
        classes = sorted(self.class_mu.keys())
        if len(classes) == 0:
            fallback = np.arange(min(topk, max(total_classes, 1)), dtype=np.int64)
            return np.tile(fallback, (query_t.shape[0], 1))
        
        valid_classes = [cls for cls in classes if self.class_mu[cls].shape[0] == query_dim]
        
        if not valid_classes:
            return np.zeros((query_features.shape[0], topk), dtype=np.int64)

        stage_t0 = self._profile_tic(device) if self.enable_inference_profiling else None
        query_t = torch.from_numpy(query_features).to(device=device, dtype=torch.float32)
        if stage_t0 is not None:
            self._profile_toc("query_to_device", stage_t0, device)

        # Build/reuse cached prototype tensors on the target device
        stage_t0 = self._profile_tic(device) if self.enable_inference_profiling else None
        self._ensure_predict_cache(device=device, query_dim=query_dim, valid_classes=valid_classes)
        if stage_t0 is not None:
            self._profile_toc("build_or_reuse_cache", stage_t0, device)

        # ---- Optimization: normalize query ONCE for the entire call ----
        stage_t0 = self._profile_tic(device) if self.enable_inference_profiling else None
        q_norm = torch.nn.functional.normalize(query_t, p=2, dim=1)  # [N, D]
        if stage_t0 is not None:
            self._profile_toc("normalize_query", stage_t0, device)
        
        # -----------------------------------------------------------
        # -----------------------------------------------------------
        def compute_distance(protos_t, labels_t=None):
            """Handle compute distance."""
            if use_ease_reweighting and cur_task > 0:
                chunk_dim = 768
                num_chunks = query_dim // chunk_dim
                
                # Reshape: [N, T, C]
                q_chunks = query_t.view(query_t.shape[0], num_chunks, chunk_dim)
                p_chunks = protos_t.view(protos_t.shape[0], num_chunks, chunk_dim)
                
                # Normalize each chunk independently
                q_chunks = torch.nn.functional.normalize(q_chunks, p=2, dim=2)
                p_chunks = torch.nn.functional.normalize(p_chunks, p=2, dim=2)
                
                sim_per_chunk = (q_chunks.unsqueeze(1) * p_chunks.unsqueeze(0)).sum(dim=-1)
                
                if labels_t is None:
                    proto_task_ids = []
                    for c in valid_classes:
                        if c < init_cls:
                            tid = 0
                        else:
                            tid = (c - init_cls) // task_increment + 1
                        proto_task_ids.append(tid)
                    proto_task_ids = torch.tensor(proto_task_ids, device=device)
                else:
                    proto_task_ids = torch.zeros_like(labels_t)
                    mask_init = labels_t < init_cls
                    proto_task_ids[mask_init] = 0
                    proto_task_ids[~mask_init] = (labels_t[~mask_init] - init_cls) // task_increment + 1
                
                M = protos_t.shape[0]
                T = num_chunks
                weights = torch.ones(M, T, device=device)
                
                task_ids_expanded = proto_task_ids.unsqueeze(1).expand(M, T)
                chunk_indices = torch.arange(T, device=device).unsqueeze(0).expand(M, T)
                
                penalty_mask = (chunk_indices != task_ids_expanded)
                factor = ease_alpha / float(cur_task)
                weights[penalty_mask] = factor
                
                weighted_sim = (sim_per_chunk * weights.unsqueeze(0)).sum(dim=2)
                
                return 1.0 - weighted_sim
                
            else:
                sim = torch.mm(q_norm, protos_t.t())
                return 1.0 - sim

        # -----------------------------------------------------------
        # -----------------------------------------------------------
        ncm_centers_t = self._predict_cache["ncm_centers_t"]
        
        stage_t0 = self._profile_tic(device) if self.enable_inference_profiling else None
        dist_ncm = compute_distance(ncm_centers_t)  # [N, C]
        if stage_t0 is not None:
            self._profile_toc("compute_ncm_distance", stage_t0, device)
        
        # -----------------------------------------------------------
        # -----------------------------------------------------------
        all_protos_t = self._predict_cache["all_protos_t"]
        proto_labels_t = self._predict_cache["proto_labels_t"]
        proto_class_index_t = self._predict_cache["proto_class_index_t"]

        C = len(valid_classes)
        N = query_t.shape[0]

        coarse_k = self.coarse_topk
        use_coarse = (
            coarse_k is not None
            and coarse_k < C
            and not (use_ease_reweighting and cur_task > 0)
        )

        if use_coarse:
            # ---- Two-stage: NCM coarse filter → sub-cluster refine ----
            stage_t0 = self._profile_tic(device) if self.enable_inference_profiling else None
            _, coarse_indices = torch.topk(dist_ncm, k=coarse_k, dim=1, largest=False)  # [N, coarse_k]
            candidate_set = torch.unique(coarse_indices.reshape(-1))  # union across batch

            candidate_mask_C = torch.zeros(C, dtype=torch.bool, device=device)
            candidate_mask_C[candidate_set] = True
            proto_in_candidate = candidate_mask_C[proto_class_index_t]  # [M] bool mask

            filtered_indices = proto_in_candidate.nonzero(as_tuple=True)[0]
            filtered_protos = all_protos_t[filtered_indices]               # [M', D]
            filtered_class_index = proto_class_index_t[filtered_indices]   # [M']

            dist_filtered = 1.0 - torch.mm(q_norm, filtered_protos.t())    # [N, M']

            dist_sub = torch.full((N, C), float("inf"), device=device, dtype=dist_filtered.dtype)

            if hasattr(dist_sub, "scatter_reduce_"):
                idx = filtered_class_index.view(1, -1).expand(N, -1)
                dist_sub.scatter_reduce_(1, idx, dist_filtered, reduce="amin", include_self=True)
            else:
                for ci in candidate_set.tolist():
                    mask = (filtered_class_index == ci)
                    if mask.any():
                        min_d, _ = dist_filtered[:, mask].min(dim=1)
                        dist_sub[:, ci] = min_d
            if stage_t0 is not None:
                self._profile_toc("compute_subcluster_distance", stage_t0, device)

        else:
            # ---- Original: evaluate all prototypes ----
            stage_t0 = self._profile_tic(device) if self.enable_inference_profiling else None
            dist_proto_all = compute_distance(all_protos_t, proto_labels_t)  # [N, M]

            dist_sub = torch.full((N, C), float("inf"), device=device, dtype=dist_proto_all.dtype)

            if hasattr(dist_sub, "scatter_reduce_"):
                idx = proto_class_index_t.view(1, -1).expand(N, -1)  # [N, M]
                dist_sub.scatter_reduce_(1, idx, dist_proto_all, reduce="amin", include_self=True)
            else:
                for i, cls in enumerate(valid_classes):
                    mask = (proto_labels_t == cls)
                    if mask.any():
                        min_d, _ = dist_proto_all[:, mask].min(dim=1)
                        dist_sub[:, i] = min_d
                    else:
                        dist_sub[:, i] = dist_ncm[:, i]
            if stage_t0 is not None:
                self._profile_toc("compute_subcluster_distance", stage_t0, device)
        
        # score = alpha * d_ncm + (1 - alpha) * d_sub
        stage_t0 = self._profile_tic(device) if self.enable_inference_profiling else None
        final_scores = self.alpha * dist_ncm + (1.0 - self.alpha) * dist_sub
        
        k = min(topk, len(valid_classes))
        _, indices = torch.topk(final_scores, k=k, dim=1, largest=False)
        indices = indices.cpu().numpy()
        
        valid_classes_t = np.array(valid_classes)
        top_preds = valid_classes_t[indices]  # [N, topk]
        if stage_t0 is not None:
            self._profile_toc("fuse_and_topk", stage_t0, device)

        if total_t0 is not None:
            total_dt = self._profile_toc("predict_topk_total", total_t0, device)
            self._profile_stats["calls"] += 1
            self._profile_stats["samples"] += int(query_features.shape[0])
            self._profile_stats["total_sec"] += total_dt
        
        return top_preds

    def predict_class_logits(
        self,
        query_features,
        total_classes: int,
        device=None,
    ) -> torch.Tensor:
        """
        Return per-class logits aligned to [0, total_classes).
        Logits are the negative fused distance (higher is better).
        """
        if device is None:
            device = torch.device("cpu")

        if isinstance(query_features, torch.Tensor):
            query_t = query_features.detach().to(device=device, dtype=torch.float32)
        else:
            query_features = np.asarray(query_features, dtype=np.float32)
            query_t = torch.from_numpy(query_features).to(device=device, dtype=torch.float32)

        if query_t.shape[0] == 0:
            return torch.empty((0, total_classes), device=device, dtype=torch.float32)

        query_dim = query_t.shape[1]
        classes = sorted(self.class_mu.keys())
        valid_classes = [cls for cls in classes if self.class_mu[cls].shape[0] == query_dim]

        # Start from very small logits for all classes.
        full_logits = torch.full(
            (query_t.shape[0], total_classes),
            float("-inf"),
            device=device,
            dtype=torch.float32,
        )
        if len(valid_classes) == 0:
            return full_logits

        self._ensure_predict_cache(device=device, query_dim=query_dim, valid_classes=valid_classes)

        q_norm = torch.nn.functional.normalize(query_t, p=2, dim=1)
        ncm_centers_t = self._predict_cache["ncm_centers_t"]      # [C, D]
        all_protos_t = self._predict_cache["all_protos_t"]        # [M, D]
        proto_labels_t = self._predict_cache["proto_labels_t"]    # [M]
        proto_class_index_t = self._predict_cache["proto_class_index_t"]  # [M]

        # Distances in cosine space.
        dist_ncm = 1.0 - torch.mm(q_norm, ncm_centers_t.t())      # [N, C]
        dist_proto_all = 1.0 - torch.mm(q_norm, all_protos_t.t()) # [N, M]

        C = len(valid_classes)
        N = query_t.shape[0]
        dist_sub = torch.full((N, C), float("inf"), device=device, dtype=dist_proto_all.dtype)
        if hasattr(dist_sub, "scatter_reduce_"):
            idx = proto_class_index_t.view(1, -1).expand(N, -1)
            dist_sub.scatter_reduce_(1, idx, dist_proto_all, reduce="amin", include_self=True)
        else:
            for i, cls in enumerate(valid_classes):
                mask = (proto_labels_t == cls)
                if mask.any():
                    min_d, _ = dist_proto_all[:, mask].min(dim=1)
                    dist_sub[:, i] = min_d
                else:
                    dist_sub[:, i] = dist_ncm[:, i]

        final_scores = self.alpha * dist_ncm + (1.0 - self.alpha) * dist_sub  # lower is better
        valid_class_ids = torch.tensor(valid_classes, device=device, dtype=torch.long)
        full_logits[:, valid_class_ids] = -final_scores
        return full_logits

    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    def get_class_prototypes_info(self, cls: int, k: int = 5) -> Tuple[np.ndarray, List[int]]:
        """Handle get class prototypes info."""
        if cls not in self.class_clusters or len(self.class_clusters[cls]) == 0:
            return np.zeros((0, 0)), []
            
        clusters = self.class_clusters[cls]
        sorted_clusters = sorted(clusters, key=lambda c: c.count, reverse=True)
        top_clusters = sorted_clusters[:k]
        
        centers = np.stack([c.center for c in top_clusters], axis=0)  # [K, D]
        counts = [c.count for c in top_clusters]  # [K]
        
        return centers, counts

    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    def prototypes_per_class(self) -> Dict[int, int]:
        return {c: len(v) for c, v in self.class_clusters.items() if len(v) > 0}