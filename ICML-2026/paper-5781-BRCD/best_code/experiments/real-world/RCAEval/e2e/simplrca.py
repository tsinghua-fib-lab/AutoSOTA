from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from RCAEval.io.time_series import (
    convert_mem_mb,
    drop_constant,
    drop_extra,
    drop_near_constant,
    drop_time,
    preprocess,
    select_useful_cols,
)


def simplerca_helper( data, inject_time=None, dataset=None, num_loop=None, sli=None, anomalies=None, **kwargs):

    normal_df = data[data["time"] < inject_time]
    abnormal_df = data[data["time"] >= inject_time]

    normal_df = preprocess(
        data=normal_df, dataset=dataset, dk_select_useful=kwargs.get("dk_select_useful", False)
    )

    abnormal_df = preprocess(
        data= abnormal_df, dataset=dataset, dk_select_useful=kwargs.get("dk_select_useful", False)
    )
    intersects = [x for x in normal_df.columns if x in abnormal_df.columns]
    normal_df = normal_df[intersects]
    abnormal_df = abnormal_df[intersects]

    ranks = simplerca(normal_df, abnormal_df)
    return {
        "node_names": normal_df.columns.to_list(),
        "ranks": ranks,
    }


def simplerca(
    normal_df: pd.DataFrame,
    abnormal_df: pd.DataFrame,
    cpu_metric: Optional[str] = "cpu",
    memory_metric: Optional[str] = "mem",
    latency_keyword: str = "latency",
    latency_p_keyword: str = "lat",
    cpu_mem_multiplier: float = 1.19,
    latency_multiplier: float = 1.19,
    cpu_mem_min_value: float = 80.0,
    return_all_ranked: bool = True,
    top_k: Optional[int] = None,
) -> Dict[str, List[str]]:
    """
    SimpleRCA-style metric detector adapted for wide-form metric tables.

    Parameters
    ----------
    normal_df : pd.DataFrame
        Metrics from normal operation. Each column is a variable/metric.
    abnormal_df : pd.DataFrame
        Metrics from abnormal/failure operation. Each column is a variable/metric.
    cpu_metric : str | None
        Column name to treat as CPU metric. If absent or None, CPU priority is skipped.
    memory_metric : str | None
        Column name to treat as memory metric. If absent or None, memory priority is skipped.
    latency_keyword : str
        A column is treated as latency-like if this keyword appears in its name.
    latency_p_keyword : str
        Mimics the repo's "Latency" and "P9" filter.
    cpu_mem_multiplier : float
        Abnormal max must exceed this multiple of the normal 95th percentile
        for CPU and memory metrics.
    latency_multiplier : float
        Abnormal max must exceed this multiple of the normal 95th percentile
        for latency metrics.
    cpu_mem_min_value : float
        Additional absolute cutoff used for CPU and memory, matching the repo logic.
    return_all_ranked : bool
        If True, return all detected anomalous variables in ranked order.
        If False, return only the first top_k items (or top 5 if top_k is None).
    top_k : int | None
        Number of ranked variables to return when return_all_ranked=False.

    Returns
    -------
    dict
        {"ranks": sorted_root_causes}
    """
    if normal_df.empty or abnormal_df.empty:
        return {"ranks": []}

    # Work only on shared numeric columns
    common_cols = [c for c in normal_df.columns if c in abnormal_df.columns]
    if not common_cols:
        return {"ranks": []}

    numeric_cols: List[str] = []
    for c in common_cols:
        if pd.api.types.is_numeric_dtype(normal_df[c]) and pd.api.types.is_numeric_dtype(abnormal_df[c]):
            numeric_cols.append(c)

    if not numeric_cols:
        return {"ranks": []}

    cpu_metric = cpu_metric if cpu_metric in numeric_cols else None
    memory_metric = memory_metric if memory_metric in numeric_cols else None

    latency_metrics = [
        c for c in numeric_cols
        if latency_keyword in str(c) and latency_p_keyword in str(c)
    ]

    anomalies = []

    for col in numeric_cols:
        normal_series = pd.to_numeric(normal_df[col], errors="coerce").dropna()
        abnormal_series = pd.to_numeric(abnormal_df[col], errors="coerce").dropna()

        if normal_series.empty or abnormal_series.empty:
            continue

        threshold = float(normal_series.quantile(0.95))
        max_value = float(abnormal_series.max())

        # Avoid degenerate division issues later
        safe_threshold = threshold if threshold != 0 else 1e-8

        is_anomalous = False
        metric_type = "other"

        if cpu_metric is not None and col == cpu_metric:
            metric_type = "cpu"
            is_anomalous = (
                max_value > cpu_mem_multiplier * threshold and
                max_value > cpu_mem_min_value
            )

        elif memory_metric is not None and col == memory_metric:
            metric_type = "memory"
            is_anomalous = (
                max_value > cpu_mem_multiplier * threshold and
                max_value > cpu_mem_min_value
            )

        elif col in latency_metrics:
            metric_type = "latency"
            is_anomalous = max_value > latency_multiplier * threshold

        else:
            # Optional fallback:
            # For non-latency/non-CPU/non-memory variables, we can either ignore them
            # to stay closer to the repo, or score them like latency-style metrics.
            # Here I include them with the same relative threshold rule.
            metric_type = "other"
            is_anomalous = max_value > latency_multiplier * threshold

        if is_anomalous:
            anomalies.append(
                {
                    "variable": col,
                    "type": metric_type,
                    "value": max_value,
                    "threshold": threshold,
                    "ratio": max_value / safe_threshold,
                }
            )

    if not anomalies:
        return []

    def sort_key(item: Dict[str, float]):
        # Match repo priority:
        # CPU first, then memory, then others by anomaly ratio.
        type_priority = {
            "cpu": 0,
            "memory": 1,
            "latency": 2,
            "other": 3,
        }
        return (type_priority.get(item["type"], 99), -item["ratio"])

    anomalies.sort(key=sort_key)

    sorted_root_causes = [item["variable"] for item in anomalies]

    if not return_all_ranked:
        k = 5 if top_k is None else max(0, int(top_k))
        sorted_root_causes = sorted_root_causes[:k]

    return sorted_root_causes