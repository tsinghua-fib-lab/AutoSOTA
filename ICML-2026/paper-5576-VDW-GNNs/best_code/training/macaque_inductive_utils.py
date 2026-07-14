"""
Utilities shared by inductive macaque pipelines (MARBLE + VDW SupCon v2).

These helpers cover:
- mapping held-out trial sequences to embeddings via cached k-NN weights
- flattening per-trial embeddings into feature matrices for SVM probes
- fitting / evaluating SVMs
- persisting split embeddings for downstream analysis
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Callable

from config.train_config import TrainingConfig
import numpy as np
import torch
from sklearn.neighbors import BallTree, KDTree, NearestNeighbors
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from accelerate import Accelerator

from data_processing.cknn import cknneighbors_graph
from data_processing.macaque_reaching import (
    GEOMETRIC_MODE,
    NEURAL_STATE_ATTRIBUTE,
    MarbleFoldData,
    MarbleTrialSequence,
    _compute_day_O_frames,
    _build_Q_for_spatial_graph_parallel,
    _compute_q_block_metadata,
    compute_adapt_gauss_kernel_kth_neighbor_wts,
    get_graph_nbr_ct_stats,
)
from training.metrics_utils import METRIC_ALIASES
from models.custom_metrics import (
    compute_knn_probe_accuracy,
    compute_kmeans_probe_accuracy,
    compute_logistic_probe_accuracy,
    compute_spectral_probe_accuracy,
    compute_svm_probe_accuracy,
)
from train_utils import main_print
from os_utilities import ensure_dir_exists


def _format_day_token(day_index) -> str:
    if day_index is None:
        return "day_none"
    if isinstance(day_index, (list, tuple)):
        joined = "-".join(str(idx) for idx in day_index)
        return f"day{joined}"
    return f"day{day_index}"


def get_processed_dataset_path(
    *,
    data_dir: str,
    macaque_day_index,
    learnable_p: bool,
    fold_idx: Optional[int] = None,
    filename_prefix: str = "macaque_processed",
) -> Path:
    """
    Build a deterministic filepath for caching processed macaque train graphs.
    """
    if data_dir is None:
        raise ValueError("dataset_config.data_dir must be set to save processed datasets.")
    base_dir = Path(data_dir).expanduser()
    day_token = _format_day_token(macaque_day_index)
    fold_token = f"fold{int(fold_idx)}" if fold_idx is not None else "foldall"
    lp_token = "learnableP" if learnable_p else "fixedP"
    filename = f"{filename_prefix}_{day_token}_{lp_token}_{fold_token}.pt"
    return base_dir / filename


def save_processed_train_graph(
    data: Data,
    path: Path,
    *,
    log_fn: Optional[Callable[[str], None]] = None,
    holdout_cache: Optional[dict] = None,
) -> None:
    """
    Persist a processed PyG Data object to disk for reuse.
    """
    ensure_dir_exists(path.parent, raise_exception=True)
    payload = {"train_data": data}
    if holdout_cache is not None:
        payload["holdout_cache"] = holdout_cache
    torch.save(payload, path)
    if log_fn is not None:
        log_fn(f"[INFO] Saved processed dataset to {path}")


def load_processed_train_graph(path: Path) -> Data:
    """
    Load a processed PyG Data object from disk.
    """
    resolved = Path(path).expanduser()
    if not resolved.exists():
        raise FileNotFoundError(f"Processed dataset file not found: {resolved}")
    payload = torch.load(resolved, map_location="cpu")
    if isinstance(payload, Data):
        return payload
    if isinstance(payload, dict) and "train_data" in payload:
        train_data = payload["train_data"]
        if not isinstance(train_data, Data):
            raise ValueError(f"Processed dataset at {resolved} missing valid train_data.")
        holdout_cache = payload.get("holdout_cache")
        if holdout_cache is not None:
            train_data.holdout_cache = holdout_cache
        return train_data
    raise ValueError(f"Processed dataset at {resolved} not in expected format.")


def resolve_processed_dataset_override(base_path: Path, fold_idx: int) -> Path:
    """
    Expand a user-provided override path to a fold-specific filepath.

    Supports either:
    - explicit `{fold}` placeholder in the filename, e.g., data.pt -> data_{fold}.pt
    - automatic appending of `_fold{idx}` before the suffix when no placeholder is present.
    """
    fold_token = f"fold{int(fold_idx)}"
    path_str = str(base_path)
    if "{fold}" in path_str:
        replaced = path_str.replace("{fold}", fold_token)
        return Path(replaced).expanduser().resolve()

    suffix = ''.join(base_path.suffixes)
    if suffix:
        stem = base_path.name[:-len(suffix)]
    else:
        stem = base_path.name
    new_name = f"{stem}_{fold_token}{suffix}"
    return (base_path.parent / new_name).expanduser().resolve()


def build_train_graph(
    fold_data: MarbleFoldData,
    n_neighbors: int,
    delta: float,
    compute_q: bool = False,
    q_num_workers: int | None = None,
    scattering_n_neighbors: int | None = None,
    learnable_p: bool = False,
    min_neighbors: int | None = None,
    dist_metric: str = "euclidean",
    eval_mode: str = "weighted_average",
) -> Data:
    """
    Construct a PyG Data object for MARBLE/VDW training using train nodes only.

    Args:
        fold_data: Train split metadata and cached sequences.
        n_neighbors: Neighbor count for the cached CkNN graph (used for evaluation caches).
        delta: CkNN delta parameter.
        compute_q: Whether to build O-frames and the Q diffusion operator.
        q_num_workers: Optional override for parallel workers during O-frame/Q builds.
        scattering_n_neighbors: Optional neighbor count dedicated to scattering (O-frames/Q).
            Defaults to ``n_neighbors`` when None.
        learnable_p: If True, store the unweighted diffusion operator ('Q_unwt') and
            associated block metadata for use with a learnable P parameterization.
    """
    num_nodes = int(fold_data.train_positions.shape[0])
    max_neighbors_possible = max(1, num_nodes - 1)
    k_eff = int(max(1, min(n_neighbors, max_neighbors_possible)))

    if scattering_n_neighbors is not None and scattering_n_neighbors > n_neighbors:
        raise ValueError(
            "scattering_n_neighbors must be <= n_neighbors when constructing the CkNN graph."
        )

    metric = str(dist_metric or "euclidean").lower()

    def _build_cknn_graph(k: int) -> tuple[torch.Tensor, torch.Tensor]:
        print(f"[build_train_graph] CkNN metric={metric}, k={k}, delta={delta}")
        edge_index_np, edge_weight_np = cknneighbors_graph(
            fold_data.train_positions,
            n_neighbors=k,
            delta=delta,
            metric=metric,
            include_self=False,
            is_sparse=True,
            use_raw_distances=True,
        )
        ei = torch.from_numpy(edge_index_np).long()
        ew = torch.from_numpy(edge_weight_np).float()
        return ei, ew

    def _ensure_min_degree(
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        min_deg: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if min_deg is None or min_deg <= 0:
            return edge_index, edge_weight
        deg = torch.bincount(edge_index[0], minlength=num_nodes)
        if not bool((deg < min_deg).any()):
            return edge_index, edge_weight

        k_knn = min(num_nodes, max(min_deg, 1))
        if metric == "cosine":
            try:
                tree = BallTree(fold_data.train_positions, metric="angular")
                distances, indices = tree.query(fold_data.train_positions, k=k_knn)
            except Exception:
                nbrs = NearestNeighbors(
                    n_neighbors=k_knn,
                    metric="cosine",
                    algorithm="brute",
                )
                nbrs.fit(fold_data.train_positions)
                distances, indices = nbrs.kneighbors(fold_data.train_positions)
        else:

            nbrs = NearestNeighbors(
                n_neighbors=k_knn,
                metric=metric,
                algorithm="auto",
            )
            nbrs.fit(fold_data.train_positions)
            distances, indices = nbrs.kneighbors(fold_data.train_positions)

        edge_list = []
        for src in range(num_nodes):
            for j_idx, dst in enumerate(indices[src]):
                if src == dst:
                    continue
                edge_list.append((src, int(dst), float(distances[src, j_idx])))
        if edge_list:
            arr = np.array(edge_list)
            add_ei = torch.from_numpy(arr[:, :2].T.astype(np.int64))
            add_ew = torch.from_numpy(arr[:, 2].astype(np.float32))
            edge_index = torch.cat([edge_index, add_ei], dim=1)
            edge_weight = torch.cat([edge_weight, add_ew], dim=0)
            edge_index, edge_weight = to_undirected(edge_index, edge_weight, reduce="min")
        return edge_index, edge_weight

    def _limit_edges_per_node(
        edge_index_dir: torch.Tensor,
        edge_weight_dir: torch.Tensor,
        max_deg: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if max_deg >= max_neighbors_possible:
            return edge_index_dir, edge_weight_dir
        src = edge_index_dir[0]
        keep_mask = torch.zeros(edge_index_dir.shape[1], dtype=torch.bool)
        for node in range(num_nodes):
            node_mask = (src == node).nonzero(as_tuple=True)[0]
            if node_mask.numel() == 0:
                continue
            if node_mask.numel() <= max_deg:
                keep_mask[node_mask] = True
                continue
            node_weights = edge_weight_dir[node_mask]
            # Keep neighbors with the smallest distances (weights)
            topk_idx = torch.topk(node_weights, max_deg, largest=False).indices
            keep_mask[node_mask[topk_idx]] = True
        if not torch.any(keep_mask):
            raise ValueError(
                "CkNN restriction removed all edges; decrease scattering_n_neighbors constraint."
            )
        return edge_index_dir[:, keep_mask], edge_weight_dir[keep_mask]

    edge_index_dir, edge_weight_dir = _build_cknn_graph(k_eff)
    edge_index, edge_weight = to_undirected(edge_index_dir, edge_weight_dir, reduce="min")
    edge_index, edge_weight = _ensure_min_degree(edge_index, edge_weight, min_neighbors or 0)

    data = Data()
    train_positions = torch.from_numpy(fold_data.train_positions).float()
    data.pos = train_positions
    data.x = torch.from_numpy(fold_data.train_velocities).float()
    data[NEURAL_STATE_ATTRIBUTE] = train_positions.clone()
    data.edge_index = edge_index
    data.edge_weight = edge_weight
    data.num_nodes = num_nodes
    data.num_node_features = data.x.shape[1]
    data.mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.train_mask = torch.ones(data.num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)  # placeholder
    data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)  # placeholder
    data.degree = k_eff
    data.trial_ids = torch.from_numpy(
        np.asarray([seq.trial_id for seq in fold_data.train_trials], dtype=np.int64)
    )
    data.trial_ptr = torch.from_numpy(fold_data.train_trial_ptr)
    data.node_trial_id = torch.from_numpy(fold_data.train_trial_ids)
    data.node_time_idx = torch.from_numpy(fold_data.train_time_ids)
    data.condition_idx = torch.from_numpy(fold_data.train_condition_ids).long()
    data.trial_condition_idx = torch.from_numpy(fold_data.train_trial_conditions)
    # Log neighbor statistics (including undirected edge count) for debugging
    try:
        stats = get_graph_nbr_ct_stats({"train": data})["train"]
        print(
            "[Graph stats] undirected_edges="
            f"{int(stats['undirected_edge_count'])}, "
            f"min={stats['min']:.1f}, q1={stats['q1']:.1f}, "
            f"median={stats['median']:.1f}, q3={stats['q3']:.1f}, "
            f"max={stats['max']:.1f}"
        )
    except Exception as exc:
        print(f"[Graph stats] failed to compute neighbor stats: {exc}")
    holdout_cache: Dict[str, Any] = {}

    if compute_q:
        eval_mode_norm = str(eval_mode or "weighted_average").lower()
        worker_ct = q_num_workers
        if worker_ct is None:
            cpu_ct = os.cpu_count() or 2
            worker_ct = max(1, min(8, cpu_ct // 2))
        scatter_k = (
            int(scattering_n_neighbors)
            if scattering_n_neighbors is not None
            else int(n_neighbors)
        )
        scatter_k = int(max(1, min(scatter_k, max_neighbors_possible)))
        if scatter_k < data.degree:
            scattering_edge_index_dir, scattering_edge_weight_dir = _limit_edges_per_node(
                edge_index_dir,
                edge_weight_dir,
                scatter_k,
            )
        else:
            scattering_edge_index_dir = edge_index_dir
            scattering_edge_weight_dir = edge_weight_dir
        scattering_edge_index, scattering_edge_weight = to_undirected(
            scattering_edge_index_dir,
            scattering_edge_weight_dir,
            reduce="min",
        )

        scattering_graph = Data()
        scattering_graph.edge_index = scattering_edge_index
        scattering_graph.edge_weight = scattering_edge_weight
        scattering_graph.num_nodes = num_nodes
        scattering_graph[NEURAL_STATE_ATTRIBUTE] = data.pos.clone()

        scattering_graph.O_frames = _compute_day_O_frames(
            scattering_graph,
            vector_feat_key=NEURAL_STATE_ATTRIBUTE,
            target_dim=fold_data.state_dim,
            num_workers=worker_ct,
            backend="threads",
        )
        # Cache train O-frames and degrees for held-out evaluation paths.
        data.train_O_frames = scattering_graph.O_frames.clone()
        data.train_degrees = torch.bincount(
            scattering_edge_index[0],
            minlength=num_nodes,
        ).to(torch.long)
        data.diffusion_edge_index = scattering_edge_index
        data.diffusion_edge_weight = scattering_edge_weight
        Q_tensor = _build_Q_for_spatial_graph_parallel(
            scattering_graph,
            geometric_mode=GEOMETRIC_MODE,
            num_workers=worker_ct,
            backend="threads",
            apply_p_weights=not learnable_p,
        )
        if learnable_p:
            data.Q_unwt = Q_tensor
            vector_dim = int(scattering_graph[NEURAL_STATE_ATTRIBUTE].shape[1])
            meta = _compute_q_block_metadata(
                Q_tensor,
                scattering_edge_index,
                vector_dim=vector_dim,
                num_nodes=num_nodes,
            )
            data.Q_block_pairs = meta['block_pairs']
            data.Q_block_edge_ids = meta['block_edge_ids']
        else:
            data.Q = Q_tensor
        # Drop large intermediates not needed after Q is built.
        for attr in ("O_frames", NEURAL_STATE_ATTRIBUTE):
            if hasattr(scattering_graph, attr):
                delattr(scattering_graph, attr)
        for attr in ("mask", "train_mask", "val_mask", "test_mask"):
            if hasattr(data, attr):
                delattr(data, attr)

        # Precompute and cache held-out trial artifacts (O-frames, edges, weights, and optional Q scaffolds)
        if eval_mode_norm == "forward_insert":
            holdout_worker_ct = max(1, (os.cpu_count() or 2) - 2)
            holdout_cache = {"valid": {}, "test": {}}
            for split_name in ("valid", "test"):
                for seq in getattr(fold_data, f"{split_name}_trials"):
                    if seq.positions.shape[0] == 0:
                        continue
                    trial_O_frames, edge_index_dir, edge_weight_dir = _compute_trial_o_frames_and_edges(
                        train_positions=fold_data.train_positions,
                        train_degrees=data.train_degrees,
                        train_O_frames=data.train_O_frames,
                        seq=seq,
                        holdout_k_probe=int(getattr(data, "holdout_k_probe", 10)) if hasattr(data, "holdout_k_probe") else 10,
                    )
                    num_train_nodes = int(data.train_O_frames.shape[0])
                    if edge_index_dir.numel() > 0:
                        ei_full, ew_full = to_undirected(
                            edge_index_dir,
                            edge_weight_dir,
                            num_nodes=num_train_nodes + seq.positions.shape[0],
                            reduce="min",
                        )
                    else:
                        ei_full = edge_index_dir
                        ew_full = edge_weight_dir

                    cache_entry: Dict[str, Any] = {
                        "O_frames": trial_O_frames.cpu(),
                        "edge_index": ei_full.cpu(),
                        "edge_weight": ew_full.cpu(),
                        "trial_id": getattr(seq, "trial_id", None),
                        "num_nodes": seq.positions.shape[0],
                    }
                    if learnable_p:
                        temp_graph = Data()
                        temp_graph.edge_index = ei_full
                        temp_graph.edge_weight = ew_full
                        temp_graph.O_frames = torch.cat(
                            [data.train_O_frames, trial_O_frames.to(data.train_O_frames.device)], dim=0
                        )
                        temp_graph.num_nodes = num_train_nodes + seq.positions.shape[0]
                        temp_graph[NEURAL_STATE_ATTRIBUTE] = torch.cat(
                            [
                                data.pos,
                                torch.from_numpy(seq.positions).to(data.pos.device, data.pos.dtype),
                            ],
                            dim=0,
                        )
                        Q_unwt_raw = _build_Q_for_spatial_graph_parallel(
                            temp_graph,
                            geometric_mode=GEOMETRIC_MODE,
                            num_workers=holdout_worker_ct,
                            backend="threads",
                            apply_p_weights=False,
                        )
                        if isinstance(Q_unwt_raw, dict):
                            # Backward compatibility with older return type.
                            Q_unwt = torch.sparse_coo_tensor(
                                indices=torch.from_numpy(Q_unwt_raw["indices"]),
                                values=torch.from_numpy(Q_unwt_raw["values"]),
                                size=tuple(int(x) for x in Q_unwt_raw["size"].tolist()),
                            )
                        else:
                            Q_unwt = Q_unwt_raw.coalesce()
                        meta = _compute_q_block_metadata(
                            Q_unwt,
                            ei_full,
                            vector_dim=int(temp_graph.O_frames.shape[-1]),
                            num_nodes=int(temp_graph.num_nodes),
                        )
                        cache_entry["Q_unwt"] = Q_unwt.cpu()
                        cache_entry["Q_block_pairs"] = meta["block_pairs"].cpu()
                        cache_entry["Q_block_edge_ids"] = meta["block_edge_ids"].cpu()

                    holdout_cache[split_name][int(seq.trial_id)] = cache_entry

            if holdout_cache.get("valid") or holdout_cache.get("test"):
                data.holdout_cache = holdout_cache
    return data




def infer_trial_embeddings(
    seq: MarbleTrialSequence,
    train_embeddings: np.ndarray,
) -> np.ndarray:
    """
    Compute embeddings for a single trial in a hold-out set, as a weighted sum
    of the nearest neighbors' embeddings in the train set, using cached
    neighbor indices and weights.
    """
    if seq.node_indices is not None:
        return train_embeddings[seq.node_indices]
    if seq.neighbor_indices is None or seq.neighbor_weights is None:
        raise ValueError("Held-out trial missing neighbor cache for inductive embeddings.")
    neighbor_emb = train_embeddings[seq.neighbor_indices]  # (T, k, d)
    weights = seq.neighbor_weights[:, :, None]
    return np.sum(weights * neighbor_emb, axis=1)


def build_split_features(
    sequences: List[MarbleTrialSequence],
    train_embeddings: np.ndarray,
    expected_nodes: int | None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Flatten per-trial embeddings into feature vectors and align labels / ids.
    """
    features: List[np.ndarray] = []
    labels: List[int] = []
    trial_ids: List[int] = []
    for seq in sequences:
        emb = infer_trial_embeddings(seq, train_embeddings)
        if expected_nodes is not None and emb.shape[0] != expected_nodes:
            raise ValueError(
                f"Inconsistent node count ({emb.shape[0]}) for trial {seq.trial_id}; expected {expected_nodes}."
            )
        features.append(emb.reshape(-1))
        labels.append(seq.condition_idx)
        trial_ids.append(seq.trial_id)
    if not features:
        return (
            np.empty((0, 0), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
        )
    feats = np.vstack(features).astype(np.float32)
    out = (
        feats,
        np.asarray(labels, dtype=np.int64),
        np.asarray(trial_ids, dtype=np.int64),
    )
    return out


def build_split_features_adaptive_k(
    sequences: List[MarbleTrialSequence],
    train_embeddings: np.ndarray,
    *,
    train_positions: np.ndarray,
    train_degrees: Optional[torch.Tensor],
    holdout_k_probe: int,
    expected_nodes: int | None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Like build_split_features, but recompute held-out neighbor weights using an
    adaptive k per node: probe holdout_k_probe nearest train neighbors, take
    ceil(median(train_degree)) as k_i, then Gaussian-kernel weight the k_i
    closest neighbors.
    """
    if train_degrees is None:
        raise ValueError("train_degrees is required for adaptive-k embeddings.")
    deg_cpu = train_degrees.detach().cpu()
    if deg_cpu.numel() != train_positions.shape[0]:
        raise ValueError("train_degrees length does not match train_positions.")
    k_probe = max(1, min(int(holdout_k_probe), train_positions.shape[0]))
    kdt = KDTree(train_positions)

    features: List[np.ndarray] = []
    labels: List[int] = []
    trial_ids: List[int] = []
    for seq in sequences:
        if seq.positions.shape[0] == 0:
            continue
        pos = seq.positions
        dists, nbr_idx = kdt.query(pos, k=k_probe, return_distance=True)
        degs = deg_cpu[nbr_idx]
        k_dynamic = torch.ceil(degs.float().median(dim=1).values).to(torch.int64).numpy()
        emb_nodes: List[np.ndarray] = []
        for row_idx, row_deg in enumerate(k_dynamic.tolist()):
            ki = max(1, min(int(row_deg), train_positions.shape[0]))
            dist_row = dists[row_idx]
            idx_row = nbr_idx[row_idx]
            # sort by distance and truncate to ki
            order = np.argsort(dist_row)
            idx_use = idx_row[order[:ki]]
            dist_use = dist_row[order[:ki]]
            weights = compute_adapt_gauss_kernel_kth_neighbor_wts(dist_use.reshape(1, -1))[0]
            emb = train_embeddings[idx_use]
            emb_nodes.append(np.sum(weights[:, None] * emb, axis=0))
        emb_nodes_arr = np.stack(emb_nodes, axis=0)
        if expected_nodes is not None and emb_nodes_arr.shape[0] != expected_nodes:
            raise ValueError(
                f"Inconsistent node count ({emb_nodes_arr.shape[0]}) for trial {seq.trial_id}; expected {expected_nodes}."
            )
        features.append(emb_nodes_arr.reshape(-1))
        labels.append(seq.condition_idx)
        trial_ids.append(seq.trial_id)

    if not features:
        return (
            np.empty((0, 0), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
        )
    feats = np.vstack(features).astype(np.float32)
    return (
        feats,
        np.asarray(labels, dtype=np.int64),
        np.asarray(trial_ids, dtype=np.int64),
    )


def build_split_features_cebra(
    sequences: List[MarbleTrialSequence],
    cebra_model,
    expected_nodes: int | None,
    use_velocity_inputs: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-trial features by running the trained CEBRA model directly on
    held-out nodes (either velocities or positions). Trials are stacked for a
    single transform call (assumed to be the heaviest step), then sliced back
    per trial.
    """
    inputs_list: List[np.ndarray] = []
    lengths: List[int] = []
    labels: List[int] = []
    trial_ids: List[int] = []
    for seq in sequences:
        inputs = seq.velocities if use_velocity_inputs else seq.positions
        T = inputs.shape[0]
        if T == 0:
            continue
        if expected_nodes is not None and T != expected_nodes:
            raise ValueError(
                f"Inconsistent node count ({T}) for trial {seq.trial_id}; expected {expected_nodes}."
            )
        inputs_list.append(inputs)
        lengths.append(T)
        labels.append(seq.condition_idx)
        trial_ids.append(seq.trial_id)

    stacked = np.vstack(inputs_list)
    transformed = cebra_model.transform(stacked)  # (sum_T, d_cebra)

    features: List[np.ndarray] = []
    offset = 0
    for T in lengths:
        trial_emb = transformed[offset : offset + T]
        features.append(trial_emb.reshape(-1))
        offset += T

    feats = np.vstack(features).astype(np.float32)
    return (
        feats,
        np.asarray(labels, dtype=np.int64),
        np.asarray(trial_ids, dtype=np.int64),
    )


def _compute_trial_o_frames_and_edges(
    *,
    train_positions: np.ndarray,
    train_degrees: torch.Tensor,
    train_O_frames: torch.Tensor,
    seq: MarbleTrialSequence,
    holdout_k_probe: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute O-frames for a held-out trial and build directed edges from trial nodes
    to their neighbors (train + same-trial). Returns (trial_O_frames, edge_index_dir, edge_weight_dir).
    """
    if seq.positions.shape[0] == 0:
        raise ValueError("Trial has no nodes.")
    device = train_O_frames.device
    dtype = train_O_frames.dtype
    num_train = train_positions.shape[0]
    pos_trial = seq.positions  # (T, D)
    T, D = pos_trial.shape
    d = int(train_O_frames.shape[-1])

    # Probe degrees from nearest train neighbors
    k_probe = max(1, min(int(holdout_k_probe), num_train))
    kdt = KDTree(train_positions)
    probe_dists, probe_idx = kdt.query(pos_trial, k=k_probe, return_distance=True)
    probe_degrees = train_degrees[torch.from_numpy(probe_idx).to(train_degrees.device)]
    median_k = int(torch.ceil(probe_degrees.float().median()).item())
    k_i = max(1, min(median_k, num_train + T - 1))

    # Precompute intra-trial distances
    trial_kdt = KDTree(pos_trial)

    trial_O_list: List[torch.Tensor] = []
    edge_wts: List[float] = []
    edge_src: List[int] = []
    edge_dst: List[int] = []

    for t_idx in range(T):
        center = pos_trial[t_idx]  # (D,)
        # distances to train
        dist_train, idx_train = kdt.query(center[None, :], k=min(k_i, num_train), return_distance=True)
        dist_train = dist_train[0]
        idx_train = idx_train[0]
        # distances to same-trial (exclude self)
        dist_trial, idx_trial = trial_kdt.query(center[None, :], k=min(k_i + 1, T), return_distance=True)
        dist_trial = dist_trial[0]
        idx_trial = idx_trial[0]
        mask_self = idx_trial != t_idx
        dist_trial = dist_trial[mask_self]
        idx_trial = idx_trial[mask_self]

        # Merge candidates
        candidates: List[Tuple[float, int]] = []
        for dval, nbr in zip(dist_train, idx_train):
            candidates.append((float(dval), int(nbr)))
        for dval, nbr in zip(dist_trial, idx_trial):
            candidates.append((float(dval), num_train + int(nbr)))
        candidates.sort(key=lambda x: x[0])
        if len(candidates) > k_i:
            candidates = candidates[:k_i]

        # Build neighbor matrix
        neighbor_vecs: List[np.ndarray] = []
        for dval, nbr in candidates:
            if nbr < num_train:
                neighbor_vecs.append(train_positions[nbr])
            else:
                neighbor_vecs.append(pos_trial[nbr - num_train])
            edge_src.append(num_train + t_idx)
            edge_dst.append(nbr)
            edge_wts.append(dval)

        if not neighbor_vecs:
            # fallback identity
            trial_O_list.append(torch.eye(D, device=device, dtype=dtype)[:, :d])
            continue
        V = np.stack(neighbor_vecs, axis=1)  # (D, k)
        V_centered = V - center.reshape(D, 1)
        V_tensor = torch.from_numpy(V_centered).to(device=device, dtype=dtype)
        U, _, _ = torch.linalg.svd(V_tensor, full_matrices=True)
        trial_O_list.append(U[:, :d])

    trial_O_frames = torch.stack(trial_O_list, dim=0)  # (T, D, d)
    if edge_src:
        edge_index_dir = torch.tensor(
            [edge_src, edge_dst],
            dtype=torch.long,
            device=device,
        )
        edge_weight_dir = torch.tensor(edge_wts, dtype=dtype, device=device)
    else:
        edge_index_dir = torch.empty((2, 0), dtype=torch.long, device=device)
        edge_weight_dir = torch.empty((0,), dtype=dtype, device=device)
    return trial_O_frames, edge_index_dir, edge_weight_dir


def _build_augmented_data_for_trial(
    *,
    model,
    train_data: Data,
    seq: MarbleTrialSequence,
    holdout_k_probe: int,
) -> Data:
    """
    Construct an augmented Data object containing train nodes plus one held-out trial,
    with a diffusion operator built for the combined graph.
    """
    vector_feat_key = getattr(model, "vector_feat_key", None)
    if vector_feat_key is None:
        vector_feat_key = getattr(train_data, "vector_feat_key", None) or "x"
    learnable_P = bool(getattr(model, "learnable_P", False))

    train_positions = train_data[NEURAL_STATE_ATTRIBUTE].detach().cpu().numpy()
    train_O_frames = getattr(train_data, "train_O_frames", None)
    train_degrees = getattr(train_data, "train_degrees", None)
    if train_O_frames is None or train_degrees is None:
        raise ValueError("train_data missing train_O_frames or train_degrees for forward_insert.")
    train_O_frames = train_O_frames.to(train_data.edge_index.device)
    train_degrees = train_degrees.to(train_data.edge_index.device)

    cache = getattr(train_data, "holdout_cache", None)
    cache_entry = None
    if cache is not None:
        # try to find cached entry by trial_id in valid/test buckets
        for split_name in ("valid", "test"):
            bucket = cache.get(split_name, {})
            tid = getattr(seq, "trial_id", None)
            if tid is not None and int(tid) in bucket:
                cache_entry = bucket[int(tid)]
                break

    if cache_entry is not None:
        trial_O_frames = cache_entry["O_frames"].to(train_O_frames.device, train_O_frames.dtype)
        edge_index_dir = cache_entry["edge_index"].to(train_O_frames.device)
        edge_weight_dir = cache_entry["edge_weight"].to(train_O_frames.device)
        Q_unwt_cached = cache_entry.get("Q_unwt")
        Q_block_pairs_cached = cache_entry.get("Q_block_pairs")
        Q_block_edge_ids_cached = cache_entry.get("Q_block_edge_ids")
    else:
        trial_O_frames, edge_index_dir, edge_weight_dir = _compute_trial_o_frames_and_edges(
            train_positions=train_positions,
            train_degrees=train_degrees,
            train_O_frames=train_O_frames,
            seq=seq,
            holdout_k_probe=holdout_k_probe,
        )
        Q_unwt_cached = None
        Q_block_pairs_cached = None
        Q_block_edge_ids_cached = None
    num_train = int(train_positions.shape[0])
    T = int(seq.positions.shape[0])

    if edge_index_dir.numel() > 0:
        edge_index_full, edge_weight_full = to_undirected(
            edge_index_dir,
            edge_weight_dir,
            num_nodes=num_train + T,
            reduce="min",
        )
    else:
        edge_index_full = edge_index_dir
        edge_weight_full = edge_weight_dir

    O_frames_full = torch.cat([train_O_frames, trial_O_frames], dim=0)
    spatial_graph = Data()
    spatial_graph.edge_index = edge_index_full
    spatial_graph.edge_weight = edge_weight_full
    spatial_graph.O_frames = O_frames_full
    spatial_graph.num_nodes = num_train + T
    spatial_graph[NEURAL_STATE_ATTRIBUTE] = torch.cat(
        [
            torch.from_numpy(train_positions).to(train_O_frames.device, train_O_frames.dtype),
            torch.from_numpy(seq.positions).to(train_O_frames.device, train_O_frames.dtype),
        ],
        dim=0,
    )

    holdout_worker_ct = max(1, (os.cpu_count() or 2) - 2)
    if learnable_P:
        Q_unwt_dict = _build_Q_for_spatial_graph_parallel(
            spatial_graph,
            geometric_mode=GEOMETRIC_MODE,
            num_workers=holdout_worker_ct,
            backend="threads",
            apply_p_weights=False,
        )
        Q_unwt = torch.sparse_coo_tensor(
            indices=torch.from_numpy(Q_unwt_dict["indices"]).to(train_O_frames.device),
            values=torch.from_numpy(Q_unwt_dict["values"]).to(train_O_frames.device),
            size=tuple(int(x) for x in Q_unwt_dict["size"].tolist()),
        )
        meta = _compute_q_block_metadata(
            Q_unwt,
            edge_index_full,
            vector_dim=int(O_frames_full.shape[-1]),
            num_nodes=int(spatial_graph.num_nodes),
        )
    else:
        Q_dict = _build_Q_for_spatial_graph_parallel(
            spatial_graph,
            geometric_mode=GEOMETRIC_MODE,
            num_workers=holdout_worker_ct,
            backend="threads",
            apply_p_weights=True,
        )
        Q = torch.sparse_coo_tensor(
            indices=torch.from_numpy(Q_dict["indices"]).to(train_O_frames.device),
            values=torch.from_numpy(Q_dict["values"]).to(train_O_frames.device),
            size=tuple(int(x) for x in Q_dict["size"].tolist()),
        )

    vec_train = getattr(train_data, vector_feat_key)
    if vec_train is None:
        raise ValueError(f"train_data missing vector feature key '{vector_feat_key}' for forward_insert.")
    vec_trial = torch.from_numpy(seq.velocities).to(vec_train.device, vec_train.dtype)
    vec_full = torch.cat([vec_train, vec_trial], dim=0)

    aug = Data()
    aug[vector_feat_key] = vec_full
    if learnable_P:
        aug.Q_unwt = Q_unwt
        aug.Q_block_pairs = meta["block_pairs"]
        aug.Q_block_edge_ids = meta["block_edge_ids"]
        aug.diffusion_edge_index = edge_index_full
        aug.diffusion_edge_weight = edge_weight_full
    else:
        aug.Q = Q
    aug.num_nodes = num_train + T
    aug.trial_nodes_offset = num_train
    aug.trial_length = T
    aug.trial_id = getattr(seq, "trial_id", None)
    return aug


def infer_holdout_embeddings_forward_insert(
    *,
    model,
    train_data: Data,
    train_embeddings: np.ndarray,
    trial_sequences: List[MarbleTrialSequence],
    holdout_k_probe: int,
) -> Dict[str, np.ndarray]:
    """
    Build embeddings for held-out trials via forward insertion.
    """
    device = None
    if hasattr(train_data, "Q") and train_data.Q is not None:
        device = train_data.Q.device
    elif hasattr(train_data, "train_O_frames"):
        device = train_data.train_O_frames.device
    emb_list: List[np.ndarray] = []
    labels: List[int] = []
    trial_ids: List[int] = []
    for seq in trial_sequences:
        if seq.positions.shape[0] == 0:
            continue
        aug = _build_augmented_data_for_trial(
            model=model,
            train_data=train_data,
            seq=seq,
            holdout_k_probe=holdout_k_probe,
        )
        if device is not None and hasattr(aug, "to"):
            aug = aug.to(device)
        with torch.no_grad():
            out = model.compute_embeddings(aug)
        embs = out.detach().cpu().numpy()
        offset = int(getattr(aug, "trial_nodes_offset", embs.shape[0] - seq.positions.shape[0]))
        trial_emb = embs[offset : offset + seq.positions.shape[0]]
        emb_list.append(trial_emb.reshape(-1))
        labels.append(int(seq.condition_idx))
        trial_ids.append(int(seq.trial_id))

    if not emb_list:
        return {
            "features": np.empty((0, 0), dtype=np.float32),
            "labels": np.empty((0,), dtype=np.int64),
            "trial_ids": np.empty((0,), dtype=np.int64),
        }
    feats = np.stack(emb_list, axis=0).astype(np.float32)
    labels_arr = np.asarray(labels, dtype=np.int64)
    trial_ids_arr = np.asarray(trial_ids, dtype=np.int64)
    return {"features": feats, "labels": labels_arr, "trial_ids": trial_ids_arr}

def save_embeddings(
    embeddings_dir: Path,
    split_name: str,
    features: np.ndarray,
    labels: np.ndarray,
    trial_ids: np.ndarray,
) -> None:
    """
    Persist split-level embeddings/labels/trial ids for downstream probing.
    """
    ensure_dir_exists(embeddings_dir, raise_exception=True)
    payload = {
        "embeddings": torch.from_numpy(features).float(),
        "labels": torch.from_numpy(labels).long(),
        "trial_ids": torch.from_numpy(trial_ids).long(),
    }
    torch.save(payload, embeddings_dir / f"{split_name}_embeddings.pt")


def train_and_evaluate_svm(
    *,
    train_feats: np.ndarray,
    train_labels: np.ndarray,
    val_feats: np.ndarray,
    val_labels: np.ndarray,
    test_feats: np.ndarray,
    test_labels: np.ndarray,
    test_trial_ids: np.ndarray | None,
    kernel: str,
    C: float,
    gamma,
) -> Dict[str, float | np.ndarray | List[Dict[str, float]]]:
    """
    Fit an SVM on train embeddings and evaluate on val/test splits.
    """
    svm = SVC(
        kernel=kernel,
        C=C,
        gamma=gamma,
        probability=True,
    )
    if len(train_labels) > 0:
        svm.fit(train_feats, train_labels)

    def _predict_acc(feats: np.ndarray, labels: np.ndarray):
        if len(labels) == 0:
            return np.array([]), float("nan"), 0, 0
        preds = svm.predict(feats)
        acc = accuracy_score(labels, preds)
        correct = int(np.sum(preds == labels))
        total = len(labels)
        return preds, float(acc), correct, total

    train_preds, train_acc, train_correct, train_total = _predict_acc(train_feats, train_labels)
    val_preds, val_acc, val_correct, val_total = _predict_acc(val_feats, val_labels)
    test_preds, test_acc, test_correct, test_total = _predict_acc(test_feats, test_labels)

    if len(test_labels) > 0:
        test_probs = svm.predict_proba(test_feats)
        ids = test_trial_ids if test_trial_ids is not None else np.arange(len(test_labels))
        prob_records: List[Dict[str, float]] = [
            {
                "trial_id": int(tid),
                "condition_idx": int(lbl),
                "pred_label": int(pred),
                "probabilities": prob.tolist(),
            }
            for tid, lbl, pred, prob in zip(ids, test_labels, test_preds, test_probs)
        ]
    else:
        prob_records = []

    return {
        "svm": svm,
        "train_accuracy": train_acc,
        "train_correct": train_correct,
        "train_total": train_total,
        "val_accuracy": val_acc,
        "val_correct": val_correct,
        "val_total": val_total,
        "test_accuracy": test_acc,
        "test_correct": test_correct,
        "test_total": test_total,
        "test_prob_records": prob_records,
    }


def evaluate_supcon2_with_svm(
    *,
    model,
    train_data: Data,
    fold_data: MarbleFoldData,
    expected_nodes: int | None,
    svm_kernel: str,
    svm_C: float,
    svm_gamma,
    include_test: bool = False,
    return_split_features: bool = False,
) -> Tuple[np.ndarray, Dict[str, float], Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """
    Compute current train-node embeddings, build per-trial features, and run the SVM probe.
    """
    model.eval()
    eval_mode = str(getattr(model, "eval_mode", "weighted_average")).lower()
    holdout_k_probe = int(getattr(model, "holdout_k_probe", 10))
    use_forward_insert = eval_mode == "forward_insert"
    use_adaptive_weighted = eval_mode in ("weighted_average_adaptive", "weighted_average_adaptive_k")

    with torch.no_grad():
        embeddings = model.compute_embeddings(train_data.clone())
    train_embeddings = embeddings.detach().cpu().numpy()

    trial_nodes = expected_nodes if expected_nodes is not None else fold_data.nodes_per_trial
    train_feats, train_labels, train_trial_ids = build_split_features(
        fold_data.train_trials,
        train_embeddings,
        trial_nodes,
    )

    if use_forward_insert:
        val_embs = infer_holdout_embeddings_forward_insert(
            model=model,
            train_data=train_data,
            train_embeddings=train_embeddings,
            trial_sequences=fold_data.valid_trials,
            holdout_k_probe=holdout_k_probe,
        )
        val_feats = val_embs["features"]
        val_labels = val_embs["labels"]
        val_trial_ids = val_embs["trial_ids"]
    elif use_adaptive_weighted:
        train_degrees = getattr(train_data, "train_degrees", None)
        if train_degrees is None:
            train_degrees = getattr(train_data, "degree", None)
        val_feats, val_labels, val_trial_ids = build_split_features_adaptive_k(
            fold_data.valid_trials,
            train_embeddings,
            train_positions=train_data.pos.detach().cpu().numpy(),
            train_degrees=train_degrees,
            holdout_k_probe=holdout_k_probe,
            expected_nodes=trial_nodes,
        )
    else:
        val_feats, val_labels, val_trial_ids = build_split_features(
            fold_data.valid_trials,
            train_embeddings,
            trial_nodes,
        )

    feature_dim = 0
    for arr in (train_feats, val_feats):
        if arr.size > 0:
            feature_dim = arr.shape[1]
            break
    if feature_dim == 0:
        feature_dim = int(trial_nodes * train_embeddings.shape[1])

    if include_test:
        if use_forward_insert:
            test_embs = infer_holdout_embeddings_forward_insert(
                model=model,
                train_data=train_data,
                train_embeddings=train_embeddings,
                trial_sequences=fold_data.test_trials,
                holdout_k_probe=holdout_k_probe,
            )
            test_feats = test_embs["features"]
            test_labels = test_embs["labels"]
            test_trial_ids = test_embs["trial_ids"]
        elif use_adaptive_weighted:
            train_degrees = getattr(train_data, "train_degrees", None)
            if train_degrees is None:
                train_degrees = getattr(train_data, "degree", None)
            test_feats, test_labels, test_trial_ids = build_split_features_adaptive_k(
                fold_data.test_trials,
                train_embeddings,
                train_positions=train_data.pos.detach().cpu().numpy(),
                train_degrees=train_degrees,
                holdout_k_probe=holdout_k_probe,
                expected_nodes=trial_nodes,
            )
        else:
            test_feats, test_labels, test_trial_ids = build_split_features(
                fold_data.test_trials,
                train_embeddings,
                trial_nodes,
            )
    else:
        test_feats = np.empty((0, feature_dim), dtype=np.float32)
        test_labels = np.empty((0,), dtype=np.int64)
        test_trial_ids = np.empty((0,), dtype=np.int64)

    svm_stats = train_and_evaluate_svm(
        train_feats=train_feats,
        train_labels=train_labels,
        val_feats=val_feats,
        val_labels=val_labels,
        test_feats=test_feats,
        test_labels=test_labels,
        test_trial_ids=test_trial_ids,
        kernel=svm_kernel,
        C=svm_C,
        gamma=svm_gamma,
    )

    split_payloads: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    if return_split_features:
        split_payloads = {
            "train": (train_feats, train_labels, train_trial_ids),
            "valid": (val_feats, val_labels, val_trial_ids),
            "test": (test_feats, test_labels, test_trial_ids),
        }

    return train_embeddings, svm_stats, split_payloads


def _collect_split_embeddings_and_labels(
    split_key: str,
    *,
    config: TrainingConfig,
    dataloader_dict: Dict[str, Any],
    model_for_eval: torch.nn.Module,
    accelerator: Accelerator,
    target_attr: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Gather embeddings and labels for a given split across all accelerator ranks.
    """
    device = accelerator.device
    loader = dataloader_dict.get(split_key, None)
    # if (
    #     loader is None
    #     and fallback_to_train
    #     and (split_key != "train")
    #     and ("train" in dataloader_dict)
    #     and (len(dataloader_dict) == 1)
    # ):
    #     loader = dataloader_dict["train"]

    local_embeddings: List[torch.Tensor] = []
    local_labels: List[torch.Tensor] = []

    if loader is not None:
        with torch.no_grad():
            for batch in loader:
                if hasattr(batch, "to"):
                    batch = batch.to(device)

                outputs = model_for_eval(batch)

                if isinstance(outputs, dict) and ("embeddings" in outputs):
                    emb = outputs["embeddings"]
                elif isinstance(outputs, dict) and ("preds" in outputs):
                    emb = outputs["preds"]
                else:
                    continue

                if emb.dim() > 2:
                    emb = emb.view(emb.shape[0], -1)

                labels_tensor: Optional[torch.Tensor] = None
                # if hasattr(batch, "condition_idx"):
                #     labels_tensor = batch.condition_idx
                # else:
                labels_attr = target_attr if isinstance(target_attr, str) else None
                if labels_attr is None and hasattr(config, "dataset_config"):
                    if hasattr(config.dataset_config, "target_key"):
                        labels_attr = config.dataset_config.target_key
                if isinstance(labels_attr, str) and hasattr(batch, labels_attr):
                    labels_tensor = getattr(batch, labels_attr)

                # if labels_tensor is None:
                #     continue

                labels_tensor = labels_tensor.view(-1)
                local_embeddings.append(emb.detach())
                local_labels.append(labels_tensor.detach().to(torch.long))

    if len(local_embeddings) > 0:
        local_embeddings_tensor = torch.cat(local_embeddings, dim=0)
    else:
        local_embeddings_tensor = torch.empty(
            (0, 1),
            dtype=torch.float32,
            device=device,
        )

    if len(local_labels) > 0:
        local_labels_tensor = torch.cat(local_labels, dim=0)
    else:
        local_labels_tensor = torch.empty(
            (0,),
            dtype=torch.long,
            device=device,
        )

    accelerator.wait_for_everyone()
    gathered_embeddings = accelerator.gather(local_embeddings_tensor)
    gathered_labels = accelerator.gather(local_labels_tensor)
    return gathered_embeddings, gathered_labels


def _run_cluster_probes_on_embeddings(
    config: TrainingConfig,
    dataloader_dict: Dict[str, Any],
    trained_model: torch.nn.Module,
    accelerator: Accelerator,
    *,
    eval_metrics: Optional[List[str]] = None,
    num_clusters: Optional[int] = None,
    metric_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Run one or more clustering probes (SVM, logistic, k-means, spectral, etc.) on embeddings.

    eval_metrics: list of metric identifiers (e.g., ['svm', 'kmeans']).
    metric_kwargs: optional dict with per-metric kwargs, keyed by canonical metric name
        ('svm_accuracy', 'kmeans_accuracy', 'spectral_clustering_accuracy', 'logistic_linear_accuracy').
    """
    device = accelerator.device

    model_for_eval = trained_model.module if hasattr(trained_model, "module") else trained_model
    model_for_eval.eval()

    target_attr = None
    if hasattr(config, "dataset_config") and hasattr(config.dataset_config, "target_key"):
        target_attr = config.dataset_config.target_key

    train_embeddings, train_labels = _collect_split_embeddings_and_labels(
        "train",
        config=config,
        dataloader_dict=dataloader_dict,
        model_for_eval=model_for_eval,
        accelerator=accelerator,
        target_attr=target_attr,
    )

    if "test" in dataloader_dict and dataloader_dict["test"] is not None:
        eval_split_key = "test"
    elif "valid" in dataloader_dict and dataloader_dict["valid"] is not None:
        eval_split_key = "valid"
    else:
        eval_split_key = "train"

    eval_embeddings, eval_labels = _collect_split_embeddings_and_labels(
        eval_split_key,
        config=config,
        dataloader_dict=dataloader_dict,
        model_for_eval=model_for_eval,
        accelerator=accelerator,
        target_attr=target_attr,
    )

    metrics_to_use = eval_metrics or ['svm_accuracy']
    normalized_metrics: List[str] = []
    for metric in metrics_to_use:
        canonical = METRIC_ALIASES.get(str(metric).lower())
        if canonical is None:
            main_print(
                f"[WARNING] Unknown cluster eval metric '{metric}' - skipping.",
                acc=accelerator,
                config=config,
            )
            continue
        if canonical not in normalized_metrics:
            normalized_metrics.append(canonical)

    metric_specific_kwargs = metric_kwargs or {}
    resolved_num_clusters = num_clusters if num_clusters is not None else config.dataset_config.target_dim

    results: Dict[str, torch.Tensor] = {}

    if accelerator.is_main_process:
        for metric_name in normalized_metrics:
            if metric_name == 'svm_accuracy':
                results[metric_name] = compute_svm_probe_accuracy(
                    train_embeddings,
                    train_labels,
                    eval_embeddings,
                    eval_labels,
                    **metric_specific_kwargs.get('svm_accuracy', {}),
                )
            elif metric_name == 'logistic_linear_accuracy':
                results[metric_name] = compute_logistic_probe_accuracy(
                    train_embeddings,
                    train_labels,
                    eval_embeddings,
                    eval_labels,
                    **metric_specific_kwargs.get('logistic_linear_accuracy', {}),
                )
            elif metric_name == 'kmeans_accuracy':
                cfg = metric_specific_kwargs.get('kmeans_accuracy', {})
                results[metric_name] = compute_kmeans_probe_accuracy(
                    train_embeddings,
                    train_labels,
                    eval_embeddings,
                    eval_labels,
                    n_clusters=resolved_num_clusters,
                    n_init=cfg.get('n_init', 10),
                    max_iter=cfg.get('max_iter', 200),
                    random_state=cfg.get('random_state'),
                )
            elif metric_name == 'spectral_clustering_accuracy':
                cfg = metric_specific_kwargs.get('spectral_clustering_accuracy', {})
                results[metric_name] = compute_spectral_probe_accuracy(
                    train_embeddings,
                    train_labels,
                    eval_embeddings,
                    eval_labels,
                    n_clusters=resolved_num_clusters,
                    affinity=cfg.get('affinity', 'nearest_neighbors'),
                    n_neighbors=cfg.get('n_neighbors', 10),
                    assign_labels=cfg.get('assign_labels', 'cluster_qr'),
                    n_jobs=cfg.get('n_jobs', 1),
                )
            elif metric_name == 'knn_accuracy':
                cfg = metric_specific_kwargs.get('knn_accuracy', {})
                results[metric_name] = compute_knn_probe_accuracy(
                    train_embeddings,
                    train_labels,
                    eval_embeddings,
                    eval_labels,
                    k=cfg.get('knn_classifier_k', 1),
                )

    return results


def _maybe_save_fold_embeddings(
    config: TrainingConfig,
    dataloader_dict: Dict[str, Any],
    trained_model: torch.nn.Module,
    accelerator: Accelerator,
    fold_idx: int,
) -> None:
    """
    Save train/valid/test embeddings for a fold when requested.
    """
    save_embeddings_requested = bool(getattr(config, "save_embeddings", False))
    if not save_embeddings_requested:
        try:
            save_embeddings_requested = bool(getattr(config.model_config, "save_embeddings", False))
        except Exception:
            save_embeddings_requested = False

    if not save_embeddings_requested:
        return

    model_for_eval = trained_model.module if hasattr(trained_model, "module") else trained_model

    model_save_dir = config.model_save_dir
    if model_save_dir is None:
        main_print(
            "[WARNING] Cannot save embeddings because 'model_save_dir' is None.",
            acc=accelerator,
            config=config,
        )
        return

    fold_root = os.path.dirname(model_save_dir)
    embeddings_dir = os.path.join(fold_root, "embeddings")
    try:
        ensure_dir_exists(embeddings_dir, raise_exception=True)
    except Exception as exc:
        main_print(
            f"[WARNING] Failed to create embeddings directory '{embeddings_dir}': {exc}",
            acc=accelerator,
            config=config,
        )
        return

    model_for_eval.eval()

    target_attr = config.dataset_config.target_key
    if isinstance(target_attr, str) and ('+' in target_attr):
        parts = [p.strip() for p in target_attr.split('+') if p.strip()]
        target_attr = next((p for p in parts if 'vel' in p), parts[0] if parts else target_attr)

    split_pairs: List[Tuple[str, str]] = [
        ("train", "train"),
        ("valid", "valid"),
        ("test", "test"),
    ]
    if ("valid" not in dataloader_dict) and ("val" in dataloader_dict):
        split_pairs[1] = ("valid", "val")

    for display_name, lookup_key in split_pairs:
        loader = dataloader_dict.get(lookup_key)
        if loader is None:
            continue

        embeddings, labels = _collect_split_embeddings_and_labels(
            lookup_key,
            config=config,
            dataloader_dict=dataloader_dict,
            model_for_eval=model_for_eval,
            accelerator=accelerator,
            target_attr=target_attr,
        )

        if embeddings.numel() == 0:
            continue

        if accelerator.is_main_process:
            try:
                payload = {
                    "fold": fold_idx,
                    "split": display_name,
                    "embeddings": embeddings.cpu(),
                    "labels": labels.cpu(),
                }
                file_path = os.path.join(
                    embeddings_dir,
                    f"{display_name}_embeddings.pt"
                )
                torch.save(payload, file_path)
                main_print(
                    f"[INFO] Saved {display_name} embeddings for fold {fold_idx} to {file_path}",
                    acc=accelerator,
                    config=config,
                )
            except Exception as exc:
                main_print(
                    f"[WARNING] Failed to save {display_name} embeddings for fold {fold_idx}: {exc}",
                    acc=accelerator,
                    config=config,
                )

        accelerator.wait_for_everyone()