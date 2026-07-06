import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)

# ==============================
# Evaluation Metrics for Classification
# CHECKED
# ==============================
def cls_metrics(true_graph, pred_graph, ignore_diagonal=False, threshold=0.5):
    # Convert tensors to numpy
    if isinstance(true_graph, torch.Tensor):
        true_graph = true_graph.detach().cpu().numpy()
    if isinstance(pred_graph, torch.Tensor):
        pred_graph = pred_graph.detach().cpu().numpy()

    if true_graph.shape != pred_graph.shape:
        # 1. pred_graph is 2-d (d, L * d)
        if pred_graph.ndim == 2: # (d, L * d)
            dim_target, dim_source = true_graph.shape
            pred_graph = pred_graph.reshape((dim_target, -1, dim_source))
            pred_graph = np.max(pred_graph, axis=1)
        else:
            raise ValueError(f"Shape mismatch: true_graph {true_graph.shape} vs pred_graph {pred_graph.shape}")

    # Optionally ignore self-causality (diagonal)
    mask = ~np.eye(true_graph.shape[0], dtype=bool) if ignore_diagonal else np.ones_like(true_graph, dtype=bool)
    true_label = true_graph[mask].astype(int).ravel()
    pred_score = pred_graph[mask].astype(float).ravel()

    # Binarize predictions
    pred_label = (pred_score >= threshold).astype(int)

    # Confusion matrix elements
    TP = int(np.sum((true_label == 1) & (pred_label == 1)))
    TN = int(np.sum((true_label == 0) & (pred_label == 0)))
    FP = int(np.sum((true_label == 0) & (pred_label == 1)))
    FN = int(np.sum((true_label == 1) & (pred_label == 0)))

    # Handle degenerate cases (all 0 or all 1 labels)
    unique_labels = np.unique(true_label)
    if len(unique_labels) == 1:
        auroc, auprc = np.nan, np.nan
    else:
        auroc = roc_auc_score(true_label, pred_score)
        auprc = average_precision_score(true_label, pred_score)

    # Compute standard metrics
    out = {
        "TP": TP, "FP": FP, "TN": TN, "FN": FN,
        "accuracy": accuracy_score(true_label, pred_label),
        "precision": precision_score(true_label, pred_label, zero_division=0),
        "recall": recall_score(true_label, pred_label, zero_division=0),
        "f1": f1_score(true_label, pred_label, zero_division=0),
        "auroc": auroc,
        "auprc": auprc,
    }

    return out
