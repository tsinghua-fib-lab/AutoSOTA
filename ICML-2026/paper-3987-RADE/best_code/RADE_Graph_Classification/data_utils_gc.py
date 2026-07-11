# graph_classification/data_utils_gc.py
from __future__ import annotations

from typing import Dict

import numpy as np
import torch

# sklearn fallback (works even if OGB Evaluator is unavailable)
from sklearn.metrics import average_precision_score, roc_auc_score


def random_split_graph_idx(
    num_graphs: int,
    train_prop: float = 0.8,
    valid_prop: float = 0.1,
    seed: int = 0,
) -> Dict[str, torch.Tensor]:
    """
    Random graph-level split indices.

    Returns:
      {"train": LongTensor, "valid": LongTensor, "test": LongTensor}
    """
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    perm = torch.randperm(int(num_graphs), generator=g)

    n_train = int(round(train_prop * num_graphs))
    n_valid = int(round(valid_prop * num_graphs))

    train_idx = perm[:n_train].to(torch.long)
    valid_idx = perm[n_train:n_train + n_valid].to(torch.long)
    test_idx = perm[n_train + n_valid:].to(torch.long)
    return {"train": train_idx, "valid": valid_idx, "test": test_idx}


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def _is_labeled_mask(y: np.ndarray) -> np.ndarray:
    """
    OGB often uses -1 for missing labels; sometimes NaNs exist.
    Works for y shaped [N, T] or [N, 1].
    """
    mask = np.isfinite(y)
    mask = mask & (y != -1)
    return mask


def eval_acc_graph(y_true: torch.Tensor, logits: torch.Tensor) -> float:
    """
    Accuracy for TU graph classification (single-label, multiclass/binary).
    y_true: [B] or [B,1] long
    logits: [B, C]
    """
    if y_true.dim() == 2 and y_true.size(1) == 1:
        y_true = y_true.squeeze(1)
    y_true = y_true.view(-1).to(torch.long)

    pred = logits.argmax(dim=-1)
    return float((pred == y_true).float().mean().item())


def eval_ogb_rocauc(y_true: torch.Tensor, logits: torch.Tensor) -> float:
    """
    ROC-AUC for ogbg-molhiv style (binary, usually y is [B,1]).
    Uses sigmoid(logits) as scores. Ignores missing labels (-1 or NaN).
    """
    y = y_true
    if y.dim() == 1:
        y = y.view(-1, 1)
    y_np = _to_numpy(y).astype(np.float32)

    s = torch.sigmoid(logits)
    if s.dim() == 1:
        s = s.view(-1, 1)
    s_np = _to_numpy(s).astype(np.float32)

    mask = _is_labeled_mask(y_np)
    # flatten binary case
    yv = y_np[mask].reshape(-1)
    sv = s_np[mask].reshape(-1)

    # Guard: roc_auc_score requires both classes present
    if yv.size == 0:
        return float("nan")
    if np.unique(yv).size < 2:
        return float("nan")
    return float(roc_auc_score(yv, sv))


def eval_ogb_ap(y_true: torch.Tensor, logits: torch.Tensor) -> float:
    """
    Average Precision for ogbg-molpcba style (multi-label, y is [B, T]).
    Uses sigmoid(logits) as scores. Ignores missing labels (-1 or NaN).
    Returns mean AP over tasks that have at least one positive and one negative labeled example.
    """
    y = y_true
    if y.dim() == 1:
        y = y.view(-1, 1)
    y_np = _to_numpy(y).astype(np.float32)

    s = torch.sigmoid(logits)
    if s.dim() == 1:
        s = s.view(-1, 1)
    s_np = _to_numpy(s).astype(np.float32)

    N, T = y_np.shape
    ap_list = []

    for t in range(T):
        yt = y_np[:, t]
        st = s_np[:, t]
        mask = _is_labeled_mask(yt)

        yt = yt[mask]
        st = st[mask]

        if yt.size == 0:
            continue
        # AP is ill-defined if only one class present; skip such tasks (matches common OGB practice)
        if np.unique(yt).size < 2:
            continue

        ap_list.append(average_precision_score(yt, st))

    if len(ap_list) == 0:
        return float("nan")
    return float(np.mean(ap_list))

def get_primary_metric(dataset_name: str) -> str:
    """
    Defines the single metric you optimize/track as 'best/...' in wandb.
    """
    name = dataset_name.lower().strip()
    if name in {"ogbg-molhiv"}:
        return "rocauc"
    if name in {"ogbg-molpcba", "peptides-func"}:
        return "ap"
    return "acc"


def eval_graph_metric(dataset_name: str, y_true: torch.Tensor, logits: torch.Tensor) -> float:
    """
    Unified entry-point for evaluation.
    """
    metric = get_primary_metric(dataset_name)

    if metric == "acc":
        return eval_acc_graph(y_true, logits)
    if metric == "rocauc":
        return eval_ogb_rocauc(y_true, logits)
    if metric == "ap":
        return eval_ogb_ap(y_true, logits)

    raise ValueError(f"Unknown metric '{metric}' for dataset '{dataset_name}'.")

