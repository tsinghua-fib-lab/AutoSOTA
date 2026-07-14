from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import yaml

Number = Union[int, float]


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _weighted_mean_and_std(
    values: List[float],
    weights: List[float],
) -> Tuple[float, float]:
    """
    Compute weighted mean and population standard deviation.
    """
    safe_weights = [w if w > 0 else 1.0 for w in weights]
    total_weight = sum(safe_weights)
    if total_weight <= 0:
        return float("nan"), float("nan")
    mean = sum(w * v for w, v in zip(safe_weights, values)) / total_weight
    variance = sum(w * (v - mean) ** 2 for w, v in zip(safe_weights, values)) / total_weight
    std = math.sqrt(variance)
    return mean, std


def _unweighted_mean_and_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    std = math.sqrt(variance)
    return mean, std


def _median(values: List[float]) -> float:
    if not values:
        return float("nan")
    sorted_vals = sorted(values)
    mid = len(sorted_vals) // 2
    if len(sorted_vals) % 2 == 1:
        return sorted_vals[mid]
    return 0.5 * (sorted_vals[mid - 1] + sorted_vals[mid])


def _filter_finite(
    values: List[float],
    weights: Optional[List[float]] = None,
) -> Tuple[List[float], List[float]]:
    """
    Drop non-finite values (and their weights) before aggregation.
    """
    if weights is None:
        weights = [1.0] * len(values)
    filtered_values: List[float] = []
    filtered_weights: List[float] = []
    for value, weight in zip(values, weights):
        if math.isfinite(value):
            filtered_values.append(value)
            filtered_weights.append(weight)
    return filtered_values, filtered_weights


def _safe_weight(
    value: Any,
    weight_key: str,
    fold_name: str,
) -> float:
    """
    Coerce weight values to float, defaulting to 1.0 with a warning.
    """
    try:
        if value is None:
            raise TypeError("weight is None")
        return float(value)
    except (TypeError, ValueError) as exc:
        print(
            f"Warning: invalid {weight_key} in {fold_name} "
            f"(value={value}); defaulting weight to 1.0 ({exc})"
        )
        return 1.0


def compute_kfold_results(
    parent_metrics_dir: Path,
    *,
    timing_keys: Optional[Iterable[str]] = None,
    weight_key: str = "best_epoch",
) -> Dict[str, Any]:
    """
    Aggregate per-fold results.yaml files into a single kfold_results.yaml.

    Timing metrics use weighted mean/std with weights given by `weight_key`.
    All other numeric metrics use unweighted population statistics.
    """
    timing_keys = set(timing_keys or ("mean_train_time", "mean_infer_time"))
    parent_metrics_dir = Path(parent_metrics_dir)
    exp_dir = parent_metrics_dir.parent
    parent_metrics_dir.mkdir(parents=True, exist_ok=True)

    fold_entries: List[Tuple[str, Dict[str, Any]]] = []
    for fold_dir in sorted(exp_dir.glob("fold_*")):
        res_path = fold_dir / "metrics" / "results.yaml"
        pkl_path = res_path.with_suffix(".pkl")
        data: Dict[str, Any] = {}

        if res_path.exists():
            try:
                with open(res_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                if not isinstance(data, dict):
                    data = {}
            except Exception:
                data = {}

        # Backfill missing keys from pickle if available
        if pkl_path.exists():
            try:
                with open(pkl_path, "rb") as f:
                    pkl_data = pickle.load(f) or {}
                if isinstance(pkl_data, dict):
                    for key, value in pkl_data.items():
                        if key not in data:
                            data[key] = value
            except Exception:
                pass

        if data:
            best_epoch_value = data.get(weight_key)
            if (not _is_number(best_epoch_value)) or float(best_epoch_value) < 1:
                print(
                    f"Warning: invalid {weight_key} in {fold_dir.name} "
                    f"(value={best_epoch_value}); defaulting to 1"
                )
                data[weight_key] = 1
            fold_entries.append((fold_dir.name, data))

    # Require at least two folds to aggregate
    if len(fold_entries) < 2:
        return {}

    # Determine numeric keys common to all folds, excluding non-metric identifiers
    ignore_keys = {"fold", "fold_i", "model", "dataset"}
    common_numeric_keys: Optional[set[str]] = None
    for _, data in fold_entries:
        numeric_keys = {
            k
            for k, v in data.items()
            if _is_number(v)
            and k not in ignore_keys
            and not k.startswith("fold")
            and not k.startswith("model")
            and not k.startswith("dataset")
        }
        common_numeric_keys = numeric_keys if common_numeric_keys is None else (common_numeric_keys & numeric_keys)
    common_numeric_keys = common_numeric_keys or set()

    aggregate: Dict[str, Any] = {}
    for key in sorted(common_numeric_keys):
        values = [float(entry[key]) for _, entry in fold_entries]
        weights = [
            _safe_weight(entry.get(weight_key, 1.0), weight_key, fold_name)
            for fold_name, entry in fold_entries
        ]
        values, weights = _filter_finite(values, weights)
        if key in timing_keys:
            mean, std = _weighted_mean_and_std(values, weights)
        else:
            mean, std = _unweighted_mean_and_std(values)

        aggregate[key] = {
            "mean": mean,
            "std": std,
            "median": _median(values),
            "min": min(values) if values else float("nan"),
            "max": max(values) if values else float("nan"),
        }

    out_path = parent_metrics_dir / "kfold_results.yaml"
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(aggregate, f)

    return aggregate

