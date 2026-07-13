import json
import logging
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


LOGGER = logging.getLogger("rlie.core.reporting")


def save_json(path: str, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_dataset_csv(features: np.ndarray, labels: Sequence[int], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if features.size == 0:
        df = pd.DataFrame({"label": labels})
    else:
        feature_names = [f"rule_{idx}" for idx in range(features.shape[1])]
        df = pd.DataFrame(features, columns=feature_names)
        df.insert(0, "label", labels)
    df.to_csv(path, index=False)


def extract_rule_weights(model, rule_count: int) -> Tuple[List[float], float]:
    if model is None or rule_count <= 0:
        return [0.0 for _ in range(max(rule_count, 0))], 0.0

    try:
        named_steps = model.named_steps
        clf = named_steps["clf"]
    except (AttributeError, KeyError):  # pragma: no cover
        LOGGER.warning("模型管线缺少 'clf' 步骤，无法提取权重")
        return [0.0 for _ in range(rule_count)], 0.0

    weights = [0.0 for _ in range(rule_count)]
    coef = getattr(clf, "coef_", None)
    if coef is not None:
        flat = np.asarray(coef).ravel()
        limit = min(len(flat), rule_count)
        for idx in range(limit):
            weights[idx] = float(flat[idx])

    scaler = named_steps.get("scaler")
    scale_values = getattr(scaler, "scale_", None) if scaler is not None else None
    if scale_values is not None:
        scale_array = np.asarray(scale_values).ravel()
        limit = min(len(scale_array), rule_count)
        for idx in range(limit):
            scale = float(scale_array[idx])
            if scale != 0.0:
                weights[idx] = weights[idx] / scale

    bias = 0.0
    intercept = getattr(clf, "intercept_", None)
    if intercept is not None:
        intercept_array = np.asarray(intercept).ravel()
        if intercept_array.size > 0:
            bias = float(intercept_array[0])

    return weights, bias


def save_rule_snapshot(
    round_idx: int,
    feature_store,
    model,
    output_dir: str,
):
    snapshot_dir = os.path.join(output_dir, "rule_snapshots")
    os.makedirs(snapshot_dir, exist_ok=True)
    rule_count = len(feature_store.rules)
    weights, bias = extract_rule_weights(model, rule_count)

    rows: List[Dict[str, Optional[float]]] = []
    for idx, info in enumerate(feature_store.rules):
        rows.append(
            {
                "rule_index": idx,
                "rule": info.text,
                "source_round": info.source_round,
                "weight": weights[idx] if idx < len(weights) else 0.0,
                "train_accuracy": info.train_accuracy,
                "train_f1": info.train_f1,
                "train_coverage": info.train_coverage,
                "val_accuracy": info.val_accuracy,
                "val_f1": info.val_f1,
                "val_coverage": info.val_coverage,
            }
        )

    payload = {"round": round_idx, "bias": bias, "rules": rows}
    save_json(os.path.join(snapshot_dir, f"round_{round_idx:02d}_rules.json"), payload)

    columns = [
        "rule_index",
        "rule",
        "source_round",
        "weight",
        "train_accuracy",
        "train_f1",
        "train_coverage",
        "val_accuracy",
        "val_f1",
        "val_coverage",
    ]
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(os.path.join(snapshot_dir, f"round_{round_idx:02d}_rules.csv"), index=False)


def resolve_metric_name(metric_name: str) -> str:
    normalized = str(metric_name or "").strip().lower()
    if normalized in {"accuracy", "acc"}:
        return "accuracy"
    if normalized in {"f1", "f1_score"}:
        return "f1"
    raise ValueError("Metric must be 'accuracy' or 'f1'")


def should_update_best_snapshot(
    current_val_metric: float,
    current_train_metric: float,
    best_snapshot: Optional[Dict],
    best_metric_value: float,
    improvement_threshold: float,
    train_metric_name: str,
    tolerance: float = 1e-12,
) -> bool:
    if np.isnan(current_val_metric):
        return False

    current_train_metric = (
        float(current_train_metric)
        if current_train_metric is not None
        else float("nan")
    )

    if best_snapshot is None or best_metric_value == float("-inf") or np.isnan(best_metric_value):
        return True

    metric_diff = current_val_metric - best_metric_value
    if metric_diff > improvement_threshold + tolerance:
        return True
    if metric_diff < improvement_threshold - tolerance:
        return False
    if improvement_threshold > tolerance:
        return True

    best_train_metrics = best_snapshot.get("train_metrics", {})
    best_train_metric = best_train_metrics.get(train_metric_name)
    best_train_metric = float(best_train_metric) if best_train_metric is not None else float("nan")

    if np.isnan(best_train_metric) and not np.isnan(current_train_metric):
        return True
    if not np.isnan(best_train_metric) and np.isnan(current_train_metric):
        return False
    if np.isnan(best_train_metric) and np.isnan(current_train_metric):
        return True

    train_diff = current_train_metric - best_train_metric
    if train_diff > tolerance:
        return True
    if train_diff < -tolerance:
        return False
    return True
