from __future__ import annotations

import math

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


def create_empty_history() -> dict[str, list[float]]:
    return {
        "epoch": [],
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_balanced_accuracy": [],
        "val_macro_auc": [],
        "ema_val_balanced_accuracy": [],
        "lr": [],
        "epoch_time_sec": [],
    }


def aggregate_epoch_metrics(metrics: list[dict]) -> tuple[float, float]:
    total_samples = sum(metric["num_samples"] for metric in metrics)
    if total_samples == 0:
        raise ValueError("Cannot aggregate metrics for an empty epoch")

    total_loss = sum(
        _metric_value_to_float(
            metric["loss_sum"]
            if "loss_sum" in metric
            else float(metric["loss"]) * int(metric["num_samples"])
        )
        for metric in metrics
    )
    total_correct = sum(
        _metric_value_to_float(metric["num_correct"]) for metric in metrics
    )
    return total_loss / total_samples, total_correct / total_samples


def _metric_value_to_float(value) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(
                "Epoch metric tensors must be scalar values for aggregation"
            )
        return float(value.detach().cpu())
    return float(value)


def aggregate_validation_classification_metrics(
    metrics: list[dict],
) -> tuple[float, float, bool]:
    labels = torch.cat([metric["labels"] for metric in metrics], dim=0).numpy()
    predictions = torch.cat(
        [metric["predictions"] for metric in metrics], dim=0
    ).numpy()
    probabilities = torch.cat(
        [metric["probabilities"] for metric in metrics],
        dim=0,
    ).numpy()
    return compute_classification_metrics(labels, predictions, probabilities)


def compute_classification_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[float, float, bool]:
    balanced_accuracy = float(balanced_accuracy_score(labels, predictions))
    macro_auc = math.nan
    macro_auc_defined = True
    try:
        if probabilities.shape[1] == 2:
            macro_auc = float(roc_auc_score(labels, probabilities[:, 1]))
        else:
            macro_auc = float(
                roc_auc_score(
                    labels,
                    probabilities,
                    average="macro",
                    multi_class="ovr",
                    labels=list(range(probabilities.shape[1])),
                )
            )
    except ValueError:
        macro_auc_defined = False
    if math.isnan(macro_auc):
        macro_auc_defined = False
    return balanced_accuracy, macro_auc, macro_auc_defined


def compute_stratified_bootstrap_interval(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    *,
    metric_fn,
    n_bootstrap: int,
    confidence_level: float,
    seed: int,
) -> dict[str, float | int | bool | None]:
    if labels.size == 0:
        raise ValueError("Cannot compute a bootstrap interval for an empty dataset")
    if n_bootstrap < 1:
        raise ValueError(f"n_bootstrap must be at least 1, got {n_bootstrap}")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError(
            f"confidence_level must be strictly between 0 and 1, got {confidence_level}"
        )

    rng = np.random.default_rng(seed)
    unique_labels = np.unique(labels)
    class_indices = [
        np.flatnonzero(labels == class_label) for class_label in unique_labels
    ]

    bootstrap_values: list[float] = []
    for _ in range(n_bootstrap):
        sampled_indices = np.concatenate(
            [
                rng.choice(indices, size=len(indices), replace=True)
                for indices in class_indices
            ]
        )
        rng.shuffle(sampled_indices)
        try:
            metric_value = float(
                metric_fn(
                    labels[sampled_indices],
                    predictions[sampled_indices],
                    probabilities[sampled_indices],
                )
            )
        except ValueError:
            continue
        if math.isnan(metric_value):
            continue
        bootstrap_values.append(metric_value)

    if not bootstrap_values:
        return {
            "defined": False,
            "confidence_level": float(confidence_level),
            "lower": None,
            "upper": None,
            "n_bootstrap": int(n_bootstrap),
            "n_valid_bootstrap": 0,
        }

    alpha = 1.0 - confidence_level
    lower = float(np.percentile(bootstrap_values, 100.0 * alpha / 2.0))
    upper = float(np.percentile(bootstrap_values, 100.0 * (1.0 - alpha / 2.0)))
    return {
        "defined": True,
        "confidence_level": float(confidence_level),
        "lower": lower,
        "upper": upper,
        "n_bootstrap": int(n_bootstrap),
        "n_valid_bootstrap": len(bootstrap_values),
    }


def compute_ema(
    history: list[float],
    current_value: float,
    *,
    alpha: float,
) -> float:
    if not history:
        return current_value
    previous_value = history[-1]
    return alpha * current_value + (1 - alpha) * previous_value


def append_history(
    history: dict[str, list[float]],
    *,
    epoch: int,
    train_loss: float,
    train_accuracy: float,
    val_loss: float,
    val_accuracy: float,
    val_balanced_accuracy: float,
    val_macro_auc: float,
    ema_val_balanced_accuracy: float,
    lr: float,
    epoch_time_sec: float,
) -> None:
    history["epoch"].append(int(epoch))
    history["train_loss"].append(float(train_loss))
    history["train_accuracy"].append(float(train_accuracy))
    history["val_loss"].append(float(val_loss))
    history["val_accuracy"].append(float(val_accuracy))
    history["val_balanced_accuracy"].append(float(val_balanced_accuracy))
    history["val_macro_auc"].append(float(val_macro_auc))
    history["ema_val_balanced_accuracy"].append(float(ema_val_balanced_accuracy))
    history["lr"].append(float(lr))
    history["epoch_time_sec"].append(float(epoch_time_sec))


def should_update_best(
    best_state: dict[str, float | int | None],
    ema_val_balanced_accuracy: float,
    val_loss: float,
    *,
    tolerance: float,
) -> bool:
    best_ema = best_state["ema_val_balanced_accuracy"]
    best_val_loss = best_state["val_loss"]
    if best_ema is None:
        return True
    if ema_val_balanced_accuracy > best_ema + tolerance:
        return True
    if abs(ema_val_balanced_accuracy - best_ema) <= tolerance:
        return best_val_loss is None or val_loss < best_val_loss
    return False
