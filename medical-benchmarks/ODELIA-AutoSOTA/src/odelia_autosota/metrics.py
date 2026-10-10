from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

METRIC_ORDER = (
    "val_challenge_score",
    "val_micro_auc",
    "val_specificity_at_90_sensitivity",
    "val_sensitivity_at_90_specificity",
    "val_macro_auc_ovr",
    "val_balanced_accuracy",
    "val_macro_f1",
    "val_benign_recall",
    "val_malignant_recall",
    "val_nll",
)


def _validate(
    labels: np.ndarray, probabilities: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if labels.ndim != 1:
        raise ValueError(f"labels must be one-dimensional, got {labels.shape}")
    if probabilities.shape != (labels.size, 3):
        raise ValueError(
            f"probabilities must have shape ({labels.size}, 3), got {probabilities.shape}"
        )
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("probabilities must be finite")
    if not np.all(np.isin(labels, (0, 1, 2))):
        raise ValueError("labels must contain only classes 0, 1, and 2")
    row_sums = probabilities.sum(axis=1)
    if np.any(probabilities < 0.0) or not np.allclose(row_sums, 1.0, atol=1e-5):
        raise ValueError("each probability row must be non-negative and sum to one")
    return labels, probabilities


def compute_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    """Compute the locked ODELIA search metrics.

    The three official components use one flattened one-hot target vector and
    one flattened probability vector, matching the challenge micro-ROC rule.
    """

    labels, probabilities = _validate(labels, probabilities)
    one_hot = np.eye(3, dtype=np.int64)[labels]
    flat_targets = one_hot.reshape(-1)
    flat_probabilities = probabilities.reshape(-1)
    fpr, tpr, _ = roc_curve(flat_targets, flat_probabilities, drop_intermediate=False)

    micro_auc = float(roc_auc_score(flat_targets, flat_probabilities))
    # Match the released Grand Challenge evaluator exactly.
    sensitivity_at_90_specificity = float(np.interp(0.1, fpr, tpr))
    specificity_at_90_sensitivity = float(1.0 - np.interp(0.9, tpr, fpr))
    challenge_score = float(
        np.mean(
            [
                micro_auc,
                specificity_at_90_sensitivity,
                sensitivity_at_90_specificity,
            ]
        )
    )

    predictions = np.argmax(probabilities, axis=1)
    clipped = np.clip(probabilities[np.arange(labels.size), labels], 1e-12, 1.0)
    nll = float(-np.mean(np.log(clipped)))
    metrics = {
        "val_challenge_score": challenge_score,
        "val_micro_auc": micro_auc,
        "val_specificity_at_90_sensitivity": specificity_at_90_sensitivity,
        "val_sensitivity_at_90_specificity": sensitivity_at_90_specificity,
        "val_macro_auc_ovr": float(
            roc_auc_score(labels, probabilities, multi_class="ovr", average="macro")
        ),
        "val_balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "val_macro_f1": float(f1_score(labels, predictions, average="macro")),
        "val_benign_recall": float(
            recall_score(labels == 1, predictions == 1, zero_division=0)
        ),
        "val_malignant_recall": float(
            recall_score(labels == 2, predictions == 2, zero_division=0)
        ),
        "val_nll": nll,
    }
    if not all(math.isfinite(value) for value in metrics.values()):
        raise RuntimeError(f"non-finite metric encountered: {metrics}")
    return metrics


def format_metrics_line(metrics: dict[str, float]) -> str:
    missing = [name for name in METRIC_ORDER if name not in metrics]
    if missing:
        raise KeyError(f"missing metrics: {', '.join(missing)}")
    values = " ".join(f"{name}={float(metrics[name]):.12f}" for name in METRIC_ORDER)
    return f"AUTOSOTA_METRICS {values}"
