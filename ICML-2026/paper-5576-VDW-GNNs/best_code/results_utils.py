#!/usr/bin/env python3
"""
Shared utilities for summarizing experiment results across scripts.
"""

from __future__ import annotations

import math
from pathlib import Path
from statistics import mean, median
from typing import Dict, List, Iterable, Tuple, Any, Optional

import yaml


def read_yaml(path: Path) -> Dict[str, Any]:
    """
    Load a YAML file into a dictionary. Returns an empty dict on empty files.
    """
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def write_yaml(data: Dict[str, Any], path: Path) -> None:
    """
    Save a dictionary to YAML using safe dumping.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _round_value(value: float, decimals: int, force_int: bool = False) -> str:
    if force_int:
        return str(int(math.ceil(value)))
    fmt = f"{{:.{decimals}f}}"
    return fmt.format(value)


def format_metric(stats: Dict[str, float], mode: str, decimals: int, force_int: bool = False) -> str:
    if not stats:
        return "n/a"
    if mode == "mean_std":
        mean_v = stats.get("mean", float("nan"))
        std_v = stats.get("std", float("nan"))
        return f"{_round_value(mean_v, decimals, force_int)} ± {_round_value(std_v, decimals, force_int)}"
    if mode == "median_range":
        median_v = stats.get("median", float("nan"))
        vmin = stats.get("min", float("nan"))
        vmax = stats.get("max", float("nan"))
        return f"{_round_value(median_v, decimals, force_int)}[{_round_value(vmin, decimals, force_int)},{_round_value(vmax, decimals, force_int)}]"
    return "n/a"


def aggregate_values(values: Iterable[float]) -> Dict[str, float]:
    vals: List[float] = []
    for v in values:
        if v is None:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(fv):
            continue
        vals.append(fv)
    if not vals:
        return {}
    mu = mean(vals)
    return {
        "mean": mu,
        "std": float(math.sqrt(mean([(v - mu) ** 2 for v in vals]))) if len(vals) > 1 else 0.0,
        "median": median(vals),
        "min": min(vals),
        "max": max(vals),
    }


def aggregate_replication_results(
    result_paths: Iterable[Path],
    metrics: Iterable[str],
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate per-replication YAML results into summary statistics.
    """
    metric_values = {m: [] for m in metrics}
    for path in result_paths:
        if not path.exists():
            continue
        res = read_yaml(path)
        for m in metrics:
            metric_values[m].append(res.get(m))
    aggregated: Dict[str, Dict[str, float]] = {}
    for m, vals in metric_values.items():
        aggregated[m] = aggregate_values(vals)
    return aggregated


def parameter_std_zero_everywhere(
    results: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
) -> bool:
    """
    Returns True only if every parameter_count std is present and zero.
    """
    for models in results.values():
        for metrics in models.values():
            param_stats = metrics.get("parameter_count") or {}
            std = param_stats.get("std")
            if std is None:
                return False
            try:
                if not math.isclose(float(std), 0.0):
                    return False
            except (TypeError, ValueError):
                return False
    return True


def consistent_parameter_counts(
    results: Dict[str, Dict[str, Dict[str, Dict[str, float]]]]
) -> Tuple[bool, Dict[str, float]]:
    """
    Returns (is_consistent, model_to_param_count_mean).
    Consistency means every model has a finite parameter_count mean and the value
    is identical across all days.
    """
    model_counts: Dict[str, float] = {}
    for models in results.values():
        for model_name, metrics in models.items():
            param_stats = metrics.get("parameter_count") or {}
            mean_val = param_stats.get("mean")
            if mean_val is None or not math.isfinite(mean_val):
                return False, {}
            if model_name not in model_counts:
                model_counts[model_name] = float(mean_val)
            else:
                if not math.isclose(float(mean_val), model_counts[model_name]):
                    return False, {}
    return (len(model_counts) > 0), model_counts


def model_sort_key(model_name: str) -> Tuple[int, str]:
    lower = model_name.lower()
    if "vdw" in lower:
        return (1, lower)
    return (0, lower)


def prettify_model_name(model_name: str, desired_name: str) -> str:
    if "vdw" in model_name.lower():
        return f"{desired_name} (ours)"
    return model_name


def day_sort_key(day_name: str) -> int:
    base = day_name
    if base.startswith("day_"):
        base = base[len("day_"):]
    try:
        return int(base)
    except Exception:
        return 0


def format_day_label(day_name: str) -> str:
    if day_name.startswith("day_"):
        return day_name[len("day_"):]
    return day_name


def day_to_int(day_name: str):
    label = format_day_label(day_name)
    try:
        return int(label)
    except Exception:
        return label


def collect_results(
    root: Path,
    include_models: Optional[set[str]] = None,
) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    """
    Returns: {day: {model: {metric_key: stats_dict}}}
    """
    results: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    include_models_lower: Optional[set[str]] = None
    if include_models is not None:
        include_models_lower = {m.strip().lower() for m in include_models if m.strip()}
    for day_dir in sorted(p for p in root.glob("day_*") if p.is_dir()):
        day_key = day_dir.name
        for model_dir in sorted(p for p in day_dir.iterdir() if p.is_dir()):
            if include_models_lower is not None and model_dir.name.lower() not in include_models_lower:
                continue
            metrics_path = model_dir / "metrics" / "kfold_results.yaml"
            if not metrics_path.exists():
                print(f"Missing metrics file: {metrics_path}")
                continue
            try:
                data = read_yaml(metrics_path)
            except Exception as exc:
                print(f"Failed to read YAML: {metrics_path} ({exc})")
                continue
            if not isinstance(data, dict) or not data:
                print(f"Unexpected or empty results in YAML: {metrics_path}")
                continue
            model_key = model_dir.name
            results.setdefault(day_key, {})[model_key] = data
    return results
