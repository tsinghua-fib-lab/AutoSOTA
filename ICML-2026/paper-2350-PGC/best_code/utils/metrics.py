import logging
from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    roc_auc_score,
)


logger = logging.getLogger(__name__)


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute plain (overall) accuracy."""
    return float(accuracy_score(y_true, y_pred))


def compute_real_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Accuracy on real samples (label = 0).

    Returns NaN when y_true contains no real samples.
    """
    mask = (y_true == 0)
    if mask.sum() == 0:
        return float("nan")
    return float((y_pred[mask] == 0).mean())


def compute_fake_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Accuracy on fake samples (label = 1).

    Returns NaN when y_true contains no fake samples.
    """
    mask = (y_true == 1)
    if mask.sum() == 0:
        return float("nan")
    return float((y_pred[mask] == 1).mean())


def compute_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute average precision (AP).

    Returns NaN if only one class is present in ``y_true`` (AP is undefined
    in that case).
    """
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def compute_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute ROC AUC.

    Returns NaN if only one class is present, or if scikit-learn raises a
    ValueError (degenerate cases such as all-identical scores).
    """
    if np.unique(y_true).size < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return float("nan")


def compute_all_metrics(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    """Compute the full metric bundle (acc, real_acc, fake_acc, ap, auc).

    Args:
        y_true: Ground-truth labels in {0, 1}.  0 = real, 1 = fake.
        y_score: Predicted "fake" probability for each sample.
        threshold: Probability threshold used to derive hard predictions for
            the accuracy-style metrics.  Defaults to 0.5.

    Returns:
        Dict with keys ``acc``, ``real_acc``, ``fake_acc``, ``ap``, ``auc``.
        When ``y_true`` is empty, every value is NaN.
    """
    if y_true.size == 0:
        return {
            "acc": float("nan"),
            "real_acc": float("nan"),
            "fake_acc": float("nan"),
            "ap": float("nan"),
            "auc": float("nan"),
        }

    y_pred = (y_score >= threshold).astype(np.int32)

    return {
        "acc": compute_accuracy(y_true, y_pred),
        "real_acc": compute_real_accuracy(y_true, y_pred),
        "fake_acc": compute_fake_accuracy(y_true, y_pred),
        "ap": compute_average_precision(y_true, y_score),
        "auc": compute_auc(y_true, y_score),
    }


def compute_mean_metrics(
    metrics_per_subset: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    """Compute macro-mean metrics across subsets (each subset weighs equally).

    Note:
        If any subset has NaN for a metric (e.g., AP/AUC when only one class
        is present), the corresponding mean becomes NaN as well.  This is
        intentional and matches the original implementation.
    """
    metric_names = ["acc", "real_acc", "fake_acc", "ap", "auc"]
    mean_metrics: Dict[str, float] = {}

    for metric_name in metric_names:
        values = [
            m.get(metric_name, float("nan")) for m in metrics_per_subset.values()
        ]
        if values:
            mean_metrics[metric_name] = float(np.mean(values))
        else:
            mean_metrics[metric_name] = float("nan")

    return mean_metrics


def log_metrics(metrics: Dict[str, float], prefix: str = "") -> None:

    if prefix:
        prefix = f"{prefix} - "

    logger.info(
        f"{prefix}ACC: %.4f | Real_ACC: %.4f | Fake_ACC: %.4f | AP: %.4f | AUC: %.4f",
        metrics.get("acc", float("nan")),
        metrics.get("real_acc", float("nan")),
        metrics.get("fake_acc", float("nan")),
        metrics.get("ap", float("nan")),
        metrics.get("auc", float("nan")),
    )
