# data_utils.py

from __future__ import annotations

from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

# Some datasets do not have official splits; create random splits here
def make_random_split_idx(num_nodes: int, train_prop: float, valid_prop: float, seed: int) -> dict:
    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(num_nodes, generator=g)
    n_train = int(round(train_prop * num_nodes))
    n_valid = int(round(valid_prop * num_nodes))
    train_idx = perm[:n_train].to(torch.long)
    valid_idx = perm[n_train:n_train + n_valid].to(torch.long)
    test_idx = perm[n_train + n_valid:].to(torch.long)
    return {"train": train_idx, "valid": valid_idx, "test": test_idx}

def rand_train_test_idx(
    label: torch.Tensor,
    train_prop: float = 0.6,
    valid_prop: float = 0.2,
    seed: int = 0,
    ignore_negative: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Randomly split nodes into train/valid/test indices.

    Args:
        label: Tensor of shape [N] or [N, 1]. If ignore_negative=True, nodes with label == -1 are excluded.
        train_prop: Fraction of labeled nodes used for training.
        valid_prop: Fraction of labeled nodes used for validation.
        seed: Random seed (torch generator).
        ignore_negative: If True, exclude nodes whose label == -1.

    Returns:
        (train_idx, valid_idx, test_idx): LongTensor index vectors.
    """
    y = label
    if y.dim() == 2 and y.size(1) == 1:
        y = y.squeeze(1)
    y = y.view(-1)

    if ignore_negative:
        labeled_nodes = torch.where(y != -1)[0]
    else:
        labeled_nodes = torch.arange(y.size(0), device=y.device)

    n = int(labeled_nodes.numel())
    n_train = int(round(train_prop * n))
    n_valid = int(round(valid_prop * n))

    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    perm = torch.randperm(n, generator=g)

    train_idx = labeled_nodes[perm[:n_train]].to(torch.long)
    valid_idx = labeled_nodes[perm[n_train:n_train + n_valid]].to(torch.long)
    test_idx = labeled_nodes[perm[n_train + n_valid:]].to(torch.long)

    return train_idx, valid_idx, test_idx


def class_rand_splits(
    label: torch.Tensor,
    label_num_per_class: int,
    valid_num: int = 500,
    test_num: int = 1000,
    seed: int = 0,
) -> Dict[str, torch.Tensor]:
    """
    Per-class balanced training split: sample `label_num_per_class` nodes from each class.
    The remaining nodes are shuffled and split into valid/test.

    Args:
        label: Tensor [N] or [N, 1] with integer class labels.
        label_num_per_class: Number of training nodes per class.
        valid_num: Number of validation nodes.
        test_num: Number of test nodes.
        seed: Random seed.

    Returns:
        split_idx: dict with keys {'train','valid','test'} and LongTensor node indices.
    """
    y = label
    if y.dim() == 2 and y.size(1) == 1:
        y = y.squeeze(1)
    y = y.view(-1)

    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    idx = torch.arange(y.size(0))
    class_list = y.unique(sorted=True)

    train_idx = []
    non_train_idx = []

    for c in class_list.tolist():
        idx_c = idx[y == c]
        n_c = int(idx_c.numel())
        if n_c == 0:
            continue
        perm_c = idx_c[torch.randperm(n_c, generator=g)]
        take = min(label_num_per_class, n_c)
        train_idx.append(perm_c[:take])
        if take < n_c:
            non_train_idx.append(perm_c[take:])

    train_idx = torch.cat(train_idx, dim=0).to(torch.long)
    if len(non_train_idx) > 0:
        non_train_idx = torch.cat(non_train_idx, dim=0).to(torch.long)
    else:
        non_train_idx = torch.empty((0,), dtype=torch.long)

    # shuffle remaining nodes and split valid/test
    non_train_idx = non_train_idx[torch.randperm(non_train_idx.numel(), generator=g)]
    valid_idx = non_train_idx[:valid_num]
    test_idx = non_train_idx[valid_num:valid_num + test_num]

    return {"train": train_idx, "valid": valid_idx, "test": test_idx}


# ---------------------------------------------------------------------
# Evaluation metrics (aligned with nc_datasets_simple.py: y is [N] long)
# ---------------------------------------------------------------------

def eval_acc_nc(y_true: torch.Tensor, logits: torch.Tensor) -> float:
    """
    Accuracy for single-label node classification.
    Args:
        y_true: LongTensor [N] (or [N,1]) ground-truth labels.
        logits: FloatTensor [N, C] model outputs (logits).
    """
    if y_true.dim() == 2 and y_true.size(1) == 1:
        y_true = y_true.squeeze(1)
    y_true = y_true.view(-1)

    pred = logits.argmax(dim=-1)
    return float((pred == y_true).float().mean().item())


def eval_f1_nc(y_true: torch.Tensor, logits: torch.Tensor, average: str = "micro") -> float:
    """
    F1 for single-label node classification (computed over predicted labels).
    Args:
        y_true: LongTensor [N] (or [N,1]).
        logits: FloatTensor [N, C].
        average: 'micro', 'macro', 'weighted', etc.
    """
    if y_true.dim() == 2 and y_true.size(1) == 1:
        y_true = y_true.squeeze(1)
    y_true_np = y_true.view(-1).detach().cpu().numpy()
    pred_np = logits.argmax(dim=-1).detach().cpu().numpy()
    return float(f1_score(y_true_np, pred_np, average=average))
