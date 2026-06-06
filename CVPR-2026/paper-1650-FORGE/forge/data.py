import math
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data


GLOBAL_UID = 0


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def reset_uid() -> None:
    global GLOBAL_UID
    GLOBAL_UID = 0


def next_uid() -> int:
    global GLOBAL_UID
    uid = GLOBAL_UID
    GLOBAL_UID += 1
    return uid


def to_np(x):
    x = np.asarray(x)
    if x.dtype == object:
        x = np.asarray(x.tolist())
    return x

def load_hospitals_pt(pt_path: str):
    if not os.path.isfile(pt_path):
        raise FileNotFoundError(f"PT file not found: {pt_path}")
    groups = torch.load(pt_path, map_location="cpu", weights_only=False)

    if isinstance(groups, dict) and any(str(k).startswith("roi_list_") for k in groups.keys()):
        suffixes = sorted(
            {
                str(k).split("roi_list_")[-1]
                for k in groups.keys()
                if str(k).startswith("roi_list_")
            },
            key=lambda x: int(x),
        )
        tmp = []
        for s in suffixes:
            tmp.append(
                {
                    "roi_list": groups[f"roi_list_{s}"],
                    "adj_list": groups[f"adj_list_{s}"],
                    "labels": groups[f"labels_{s}"],
                }
            )
        groups = tmp

    if not isinstance(groups, list):
        raise TypeError(f"PT content must be a list or roi_list_* dict, got: {type(groups)}")
    if len(groups) == 0:
        raise ValueError("No hospital/site groups found in PT file")
    return groups


def to_graph_tuples(group: Dict[str, Any]):
    out = []
    roi_list, adj_list, labels = group["roi_list"], group["adj_list"], group["labels"]
    for roi, adj, y in zip(roi_list, adj_list, labels):
        R = to_np(roi)
        A = to_np(adj)
        y = int(np.asarray(y).squeeze().item())
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError(f"Adjacency must be square, got shape={A.shape}")
        N = A.shape[0]
        if R.ndim == 1 and R.shape[0] == N:
            R = R.reshape(N, 1)
        elif R.ndim == 2:
            if R.shape[0] == N:
                pass
            elif R.shape[1] == N:
                R = R.T
            else:
                raise ValueError(f"ROI shape {R.shape}, N={N}")
        else:
            raise ValueError(f"Unexpected ROI ndim={R.ndim}")
        out.append(
            (
                torch.tensor(R, dtype=torch.float32),
                torch.tensor(A, dtype=torch.float32),
                y,
            )
        )
    return out


def normalize_features_tensor(X: torch.Tensor) -> torch.Tensor:
    mean = X.mean(0, keepdim=True)
    std = X.std(0, keepdim=True)
    std[std == 0] = 1
    return (X - mean) / std


def _symmetrize_zero_diag_torch(A: torch.Tensor) -> torch.Tensor:
    A = 0.5 * (A + A.T)
    A = A.clone()
    A.fill_diagonal_(0.0)
    return A


def _symmetrize_zero_diag_numpy(A: np.ndarray) -> np.ndarray:
    A = 0.5 * (A + A.T)
    A = A.copy()
    np.fill_diagonal(A, 0.0)
    return A


def _edge_index_from_thresholded_fc_torch(A: torch.Tensor, tau: float):
    A = _symmetrize_zero_diag_torch(A)
    keep = torch.triu(A >= float(tau), diagonal=1)
    src, dst = keep.nonzero(as_tuple=True)
    n = A.shape[0]
    if src.numel() == 0:
        edge_index = torch.stack([torch.arange(n), torch.arange(n)], dim=0).long()
        edge_weight = torch.ones(n, dtype=torch.float32)
        return edge_index, edge_weight
    w = A[src, dst].float()
    edge_index = torch.cat(
        [
            torch.stack([src, dst], dim=0),
            torch.stack([dst, src], dim=0),
        ],
        dim=1,
    ).long()
    edge_weight = torch.cat([w, w], dim=0).float()
    return edge_index, edge_weight


def _edge_index_from_thresholded_fc_numpy(A: np.ndarray, tau: float):
    A = _symmetrize_zero_diag_numpy(A)
    keep = np.triu(A >= float(tau), k=1)
    src, dst = np.nonzero(keep)
    n = A.shape[0]
    if src.size == 0:
        edge_index = torch.tensor(np.stack([np.arange(n), np.arange(n)], axis=0), dtype=torch.long)
        edge_weight = torch.ones(n, dtype=torch.float32)
        return edge_index, edge_weight
    w = A[src, dst].astype(np.float32)
    edge_index = torch.tensor(
        np.concatenate(
            [
                np.stack([src, dst], axis=0),
                np.stack([dst, src], axis=0),
            ],
            axis=1,
        ),
        dtype=torch.long,
    )
    edge_weight = torch.tensor(np.concatenate([w, w], axis=0), dtype=torch.float32)
    return edge_index, edge_weight


def dense_to_pyg(graphs: List[Tuple[torch.Tensor, torch.Tensor, int]], hospital_id: int, threshold: float) -> List[Data]:
    ds = []
    for X, A, y in graphs:
        X = normalize_features_tensor(X)
        edge_index, edge_weight = _edge_index_from_thresholded_fc_torch(A, threshold)
        d = Data(
            x=X.float(),
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=torch.tensor([int(y)], dtype=torch.long),
        )
        d.uid = torch.tensor([next_uid()], dtype=torch.long)
        d.hospital = torch.tensor([int(hospital_id)], dtype=torch.long)
        ds.append(d)
    return ds


def _safe_split_count(n: int, ratio: float) -> int:
    if n <= 1:
        return 0
    k = int(round(n * ratio))
    k = max(1, k)
    k = min(n - 1, k)
    return k


def stratified_train_val_indices(labels: List[int], val_ratio: float, seed: int):
    idx = np.arange(len(labels)).tolist()
    y = np.asarray(labels, dtype=np.int64)
    if len(idx) == 0:
        return [], []

    rng = random.Random(seed)
    per_class = {}
    for i, c in enumerate(y.tolist()):
        per_class.setdefault(int(c), []).append(i)

    val = []
    train = []
    for _, cls_idx in sorted(per_class.items()):
        cls_idx = cls_idx[:]
        rng.shuffle(cls_idx)
        n_val = _safe_split_count(len(cls_idx), val_ratio)
        val.extend(cls_idx[:n_val])
        train.extend(cls_idx[n_val:])

    if len(train) == 0 and len(val) > 0:
        train.append(val.pop())
    if len(val) == 0 and len(train) > 1:
        val.append(train.pop())

    train = sorted(train)
    val = sorted(val)
    return train, val


def _npz_pick_key(d: np.lib.npyio.NpzFile, cand: List[str]) -> Optional[str]:
    for k in cand:
        if k in d.files:
            return k
    return None


def _svd_row_project_np(X: np.ndarray, k: int) -> np.ndarray:
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    XS = U @ np.diag(S)
    return XS[:, : min(k, XS.shape[1])] if k > 0 else XS


def _align_features_numpy(R: np.ndarray, target_dim: int, svd_fallback: bool) -> np.ndarray:
    N, F = R.shape
    if F == target_dim:
        return R
    if F == N and svd_fallback and target_dim < F:
        return _svd_row_project_np(R, target_dim)
    if F > target_dim:
        return R[:, :target_dim]
    pad = np.zeros((N, target_dim - F), dtype=R.dtype)
    return np.concatenate([R, pad], axis=1)


def _to_square_fc_numpy(arr: np.ndarray) -> np.ndarray:
    A = np.asarray(arr, dtype=np.float32)
    if A.ndim == 1:
        n = int(round(math.sqrt(A.size)))
        if n * n != A.size:
            raise ValueError(f"cannot reshape flat FC of length {A.size} into a square matrix")
        A = A.reshape(n, n)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"expected FC matrix with shape [N, N], got {A.shape}")
    return _symmetrize_zero_diag_numpy(A)


def load_synth_npz_to_pyg(
    npz_path: str,
    in_dim: int,
    adj_threshold: float,
    hospital_id: int,
    uid_offset: int = 10_000_000,
    svd_fallback: bool = True,
) -> List[Data]:
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"npz file not found: {npz_path}")
    with np.load(npz_path) as d:
        k_fc = _npz_pick_key(d, ["fc_mats", "fc", "rois", "roi", "roi_data", "ROIs", "roi_list"])
        k_lbl = _npz_pick_key(d, ["labels", "label", "y", "ys"])
        if k_fc is None or k_lbl is None:
            raise KeyError(f"npz missing fc/label arrays: {list(d.files)}")
        fc_all = d[k_fc]
        labels = d[k_lbl].tolist() if isinstance(d[k_lbl], np.ndarray) else list(d[k_lbl])
    if len(fc_all) != len(labels):
        raise ValueError(f"npz length mismatch: len(fc_all)={len(fc_all)}, len(labels)={len(labels)}")

    ds = []
    for i in range(len(fc_all)):
        A_fc = _to_square_fc_numpy(fc_all[i])
        X = _align_features_numpy(A_fc, in_dim, svd_fallback)
        Xt = torch.tensor(X, dtype=torch.float32)
        Xt = normalize_features_tensor(Xt)
        edge_index, edge_weight = _edge_index_from_thresholded_fc_numpy(A_fc, adj_threshold)
        y = int(np.asarray(labels[i]).squeeze().item())
        d = Data(
            x=Xt,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=torch.tensor([y], dtype=torch.long),
        )
        d.uid = torch.tensor([uid_offset + i], dtype=torch.long)
        d.hospital = torch.tensor([int(hospital_id)], dtype=torch.long)
        ds.append(d)
    return ds


def prepare_data(cfg):
    set_global_seed(cfg.SEED)
    reset_uid()

    groups_real = load_hospitals_pt(cfg.PT_PATH)
    num_hospitals = len(groups_real)
    npz_paths = list(cfg.NPZ_HOSP_PATHS)
    if len(npz_paths) != num_hospitals:
        raise ValueError(
            f"Mismatch between real hospitals ({num_hospitals}) and NPZ_HOSP_PATHS ({len(npz_paths)})."
        )

    all_hosp_dense = [to_graph_tuples(g) for g in groups_real]
    feat_dims = {int(X.shape[1]) for graphs in all_hosp_dense for (X, _, _) in graphs}
    if len(feat_dims) != 1:
        raise ValueError(f"Inconsistent node feature dims across hospitals: {feat_dims}")
    in_dim = list(feat_dims)[0]

    splits = []
    for gi, graphs in enumerate(all_hosp_dense):
        labels = [int(y) for (_, _, y) in graphs]
        tr_idx, val_idx = stratified_train_val_indices(
            labels,
            val_ratio=cfg.VAL_RATIO,
            seed=cfg.SEED + 101 * gi,
        )
        train_pyg = dense_to_pyg([graphs[i] for i in tr_idx], gi + 1, cfg.ADJ_THRESHOLD)
        val_pyg = dense_to_pyg([graphs[i] for i in val_idx], gi + 1, cfg.ADJ_THRESHOLD)
        splits.append((train_pyg, val_pyg))

    synth_pyg_per_hosp = []
    for hi, p in enumerate(npz_paths, start=1):
        ds = load_synth_npz_to_pyg(
            p,
            in_dim,
            cfg.ADJ_THRESHOLD,
            hospital_id=hi,
            uid_offset=10_000_000 * hi,
            svd_fallback=cfg.SYN_SVD_FALLBACK,
        )
        synth_pyg_per_hosp.append(ds)

    return splits, synth_pyg_per_hosp, in_dim
