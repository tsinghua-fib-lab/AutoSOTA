#!/usr/bin/env python3
"""
Compare suffix-aware detection performance of window-PP and CPD-online,
and also export global detection-only metrics (including PP).

Inputs:
    - CPD CSV: output of run_cpd_batch.py
    - PP CSV:  per-prompt window-PP metrics (e.g. *_pp_f1.csv)

For each detector:
    - CPD-online          : scans the full online W_plus trace (per-token)
    - window-PP w in {5,10,15,20}: scans the full non-overlapping window mean-NLL trace
    - PP (global_mean_nll): global per-prompt score (NO suffix locality)

Suffix-aware evaluation (CPD + window-PP only):
    - suffix_start = suffix_start_postprefix  (CPD) or suffix_start_token_idx (PP) in
      post-prefix coordinates.
    - suffix_len   = suffix_len_tokens (number of tokens in suffix)

Δ is defined as suffix_len (option A).

Definitions:
    TP_suffix   : adversarial prompt with suffix, detection interval overlaps [suffix_start, suffix_start + suffix_len)
    FP_benign   : benign prompt with any alarm (score >= thr and finite detection interval)
    FP_early_adv: adversarial prompt with suffix, detection interval ends before suffix_start
    FN_suffix   : adversarial prompt with suffix and
                  - no alarm, OR
                  - detection interval starts at/after suffix_end (late detection)

Suffix F1:
    precision_suffix = TP_suffix / (TP_suffix + FP_benign + FP_early_adv)
    recall_suffix    = TP_suffix / (TP_suffix + FN_suffix)
    F1_suffix        = 2 * P * R / (P + R)

Thresholds:
    - For CPD/window-PP, sweep unique finite scores and pick the threshold
      that maximizes F1_suffix.

ROC:
    - For suffix-aware summary we also compute AUROC for adversarial vs benign using scores
      (ignores locality), but with thresholds selected via suffix-aware F1.

Outputs:
    - suffix_eval_summary.csv (args.out_csv): one row per CPD/window-PP method with
      suffix-aware metrics and AUROC (NO PP row here).
    - detection_eval_summary.csv: global detection-only precision/recall/F1/AUROC for
      CPD, window-PP, and PP.
    - detection_counts.csv + detection_thresholds.csv + stacked timing plots:
      suffix-locality timing breakdowns for CPD/window-PP only (no PP).
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
import json
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def log(msg: str) -> None:
    print(f"[INFO] {msg}")

THRESHOLD_LABELS = {
    "overall_f1": "Best overall F1",
    "suffix_f1": "Best suffix F1",
    "fpr10": "10% FPR",
}
SEGMENT_ALGOS = ("gcg", "autodan", "advprompter")
METHOD_LABELS = {
    "cpd_online": "CPD Online",
    "window_pp_w5": "WPP (w=5)",
    "window_pp_w20": "WPP (w=20)",
}


# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------


def safe_div(num: float, den: float) -> float:
    return num / den if den else float("nan")


def compute_prf(labels: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    labels = labels.astype(int)
    preds = preds.astype(int)
    tp = int(((labels == 1) & (preds == 1)).sum())
    fp = int(((labels == 0) & (preds == 1)).sum())
    fn = int(((labels == 1) & (preds == 0)).sum())

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    if np.isfinite(precision + recall) and (precision + recall) > 0:
        f1 = 2.0 * precision * recall / (precision + recall)
    else:
        f1 = float("nan")
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def update_progress(current: int, total: int, label: str) -> None:
    total = max(int(total), 1)
    current = min(int(current), total)
    width = 30
    ratio = current / total
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    print(f"\r[{bar}] {current}/{total} {label}", end="", flush=True)
    if current >= total:
        print()


def update_subprogress(current: int, total: int, label: str) -> None:
    total = max(int(total), 1)
    current = min(int(current), total)
    width = 24
    ratio = current / total
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    print(f"\r    [{bar}] {current}/{total} {label}", end="", flush=True)
    if current >= total:
        print()


def compute_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """
    ROC AUC that does not depend on scikit-learn.
    y_true: 0/1 labels (1 = adversarial)
    scores: continuous scores (higher = more likely adversarial)
    """
    if len(y_true) == 0:
        return float("nan")
    positives = y_true == 1
    negatives = ~positives
    n_pos = int(positives.sum())
    n_neg = int(negatives.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores)
    ranked = np.empty_like(order, dtype=float)
    ranked[order] = np.arange(len(scores))
    sum_ranks_pos = float(ranked[positives].sum())
    auc = (sum_ranks_pos - n_pos * (n_pos - 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def compute_roc_curve(y_true: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Standard ROC curve for adversarial vs benign classification (global).
    Returns (fpr, tpr).
    """
    order = np.argsort(scores)[::-1]
    y_sorted = y_true[order]
    pos = y_sorted == 1
    neg = ~pos
    n_pos = float(pos.sum())
    n_neg = float(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    tps = np.cumsum(pos)
    fps = np.cumsum(neg)
    tpr = np.concatenate(([0.0], tps / n_pos, [1.0]))
    fpr = np.concatenate(([0.0], fps / n_neg, [1.0]))
    return fpr, tpr


def _filter_finite_scores(scores: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(scores)
    filtered_scores = scores[mask]
    filtered_labels = labels[mask]
    if filtered_scores.size == 0:
        raise RuntimeError("No finite scores available for threshold computation.")
    if len(np.unique(filtered_labels)) < 2:
        raise RuntimeError("Scores lack both label classes for threshold computation.")
    return filtered_scores, filtered_labels


def compute_classification_metrics(labels: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    labels = labels.astype(int)
    preds = preds.astype(int)
    tp = int(((labels == 1) & (preds == 1)).sum())
    fp = int(((labels == 0) & (preds == 1)).sum())
    tn = int(((labels == 0) & (preds == 0)).sum())
    fn = int(((labels == 1) & (preds == 0)).sum())
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall) if np.isfinite(precision + recall) else float("nan")
    fpr = safe_div(fp, fp + tn)
    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fpr),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }
    return metrics


def find_best_global_f1_threshold(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, Dict[str, float]]:
    scores_finite, labels_finite = _filter_finite_scores(scores, labels)
    unique_scores = np.unique(scores_finite)
    best_metrics: Optional[Dict[str, float]] = None
    best_threshold = unique_scores[0]

    for thr in unique_scores:
        preds = (scores_finite >= thr).astype(int)
        metrics = compute_classification_metrics(labels_finite, preds)
        if best_metrics is None or metrics["f1"] > best_metrics["f1"] + 1e-12:
            best_metrics = metrics
            best_threshold = thr
        elif best_metrics is not None and abs(metrics["f1"] - best_metrics["f1"]) <= 1e-12:
            if metrics["recall"] > best_metrics["recall"] + 1e-12:
                best_metrics = metrics
                best_threshold = thr
    assert best_metrics is not None
    return float(best_threshold), best_metrics


def find_threshold_at_fpr(scores: np.ndarray, labels: np.ndarray, target_fpr: float) -> Tuple[float, Dict[str, float]]:
    scores_finite, labels_finite = _filter_finite_scores(scores, labels)
    unique_scores = np.unique(scores_finite)
    target_fpr = float(target_fpr)
    best_thr: Optional[float] = None
    best_metrics: Optional[Dict[str, float]] = None
    fallback_thr: Optional[float] = None
    fallback_metrics: Optional[Dict[str, float]] = None
    best_diff = float("inf")
    fallback_diff = float("inf")

    for thr in unique_scores:
        preds = (scores_finite >= thr).astype(int)
        metrics = compute_classification_metrics(labels_finite, preds)
        fpr_val = metrics["fpr"]
        diff = abs(fpr_val - target_fpr)
        if fpr_val <= target_fpr and (target_fpr - fpr_val) < best_diff - 1e-12:
            best_diff = target_fpr - fpr_val
            best_thr = thr
            best_metrics = metrics
        if diff < fallback_diff - 1e-12:
            fallback_diff = diff
            fallback_thr = thr
            fallback_metrics = metrics

    if best_thr is not None and best_metrics is not None:
        return float(best_thr), best_metrics
    if fallback_thr is not None and fallback_metrics is not None:
        return float(fallback_thr), fallback_metrics
    raise RuntimeError("Unable to determine threshold at target FPR.")


def _parse_trace_value(value) -> np.ndarray:
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return np.asarray([], dtype=float)
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return np.asarray([], dtype=float)
    elif isinstance(value, (list, tuple, np.ndarray)):
        parsed = value
    else:
        return np.asarray([], dtype=float)
    try:
        arr = np.asarray(parsed, dtype=float)
    except Exception:
        return np.asarray([], dtype=float)
    return arr


def parse_trace_series(series: pd.Series) -> List[np.ndarray]:
    return [_parse_trace_value(val) for val in series]


def compute_trace_max(traces: List[np.ndarray]) -> np.ndarray:
    max_vals = np.full(len(traces), np.nan, dtype=float)
    for idx, trace in enumerate(traces):
        if trace.size == 0:
            continue
        finite = trace[np.isfinite(trace)]
        if finite.size == 0:
            continue
        max_vals[idx] = float(finite.max())
    return max_vals


def collect_unique_thresholds_from_traces(traces: List[np.ndarray]) -> np.ndarray:
    finite_chunks: List[np.ndarray] = []
    for trace in traces:
        if trace.size == 0:
            continue
        finite = trace[np.isfinite(trace)]
        if finite.size:
            finite_chunks.append(finite)
    if not finite_chunks:
        raise RuntimeError("No finite trace values available for threshold sweep.")
    values = np.concatenate(finite_chunks)
    unique_vals = np.unique(values)
    if unique_vals.size == 0:
        raise RuntimeError("No valid thresholds found for traces.")
    return unique_vals


def compute_alarm_positions_from_traces(
    traces: List[np.ndarray],
    threshold: float,
    step: float,
    index_offset: float = 0.0,
    interval_len: Optional[float] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    starts = np.full(len(traces), np.nan, dtype=float)
    ends = None
    if interval_len is not None:
        ends = np.full(len(traces), np.nan, dtype=float)
    if not np.isfinite(threshold):
        return starts, ends
    for idx_prompt, trace in enumerate(traces):
        if trace.size == 0:
            continue
        for idx_val, value in enumerate(trace):
            if not np.isfinite(value):
                continue
            if value >= threshold:
                start_val = index_offset + idx_val * step
                starts[idx_prompt] = start_val
                if ends is not None:
                    ends[idx_prompt] = start_val + interval_len
                break
    return starts, ends


def compute_all_alarm_intervals(
    traces: List[np.ndarray],
    threshold: float,
    step: float,
    index_offset: float = 0.0,
    interval_len: Optional[float] = None,
) -> List[List[Tuple[float, float]]]:
    """
    Return all (start, end) intervals where the trace meets/exceeds the threshold.
    Each position above threshold yields one interval; consecutive hits are not merged
    because locality accounting only needs the presence of before/overlap events.
    """
    if not np.isfinite(threshold):
        return [[] for _ in traces]
    dur = interval_len if interval_len is not None else step
    intervals: List[List[Tuple[float, float]]] = []
    for trace in traces:
        prompt_intervals: List[Tuple[float, float]] = []
        if trace is None or len(trace) == 0:
            intervals.append(prompt_intervals)
            continue
        for idx_val, value in enumerate(trace):
            if not np.isfinite(value):
                continue
            if value >= threshold:
                start_val = index_offset + idx_val * step
                prompt_intervals.append((start_val, start_val + dur))
        intervals.append(prompt_intervals)
    return intervals


@dataclass
class SuffixMetrics:
    threshold: float
    f1_suffix: float
    precision_suffix: float
    recall_suffix: float
    tp_suffix: int
    fp_benign: int
    fp_early_adv: int
    fn_suffix: int
    median_delay: float
    mean_delay: float
    n_benign: int
    n_adv_suffix: int
    n_ignored_adv_no_suffix: int
    auroc_adv: float


# ---------------------------------------------------------
# Core evaluation logic (suffix-aware)
# ---------------------------------------------------------


def evaluate_suffix_metrics_at_threshold(
    alarm_start: np.ndarray,
    alarm_end: Optional[np.ndarray],
    suffix_start: np.ndarray,
    suffix_len: np.ndarray,
    is_adv: np.ndarray,
    threshold: float,
    score_for_roc: np.ndarray,
) -> Tuple[SuffixMetrics, np.ndarray, np.ndarray]:
    """
    Compute suffix-aware metrics for a single detector at a given threshold.
    """
    assert (
        alarm_start.shape
        == suffix_start.shape
        == suffix_len.shape
        == is_adv.shape
        == score_for_roc.shape
    )
    if alarm_end is not None:
        assert alarm_end.shape == alarm_start.shape

    n = alarm_start.shape[0]

    tp_suffix = 0
    fp_benign = 0
    fp_early_adv = 0
    fn_suffix = 0

    n_benign = int((is_adv == 0).sum())
    n_adv_suffix = 0
    n_ignored_adv_no_suffix = 0

    delays_correct: List[float] = []
    early_offsets: List[float] = []

    for i in range(n):
        adv = int(is_adv[i]) == 1
        pos_start = alarm_start[i]
        pos_end = alarm_end[i] if alarm_end is not None else pos_start
        s_start = suffix_start[i]
        s_len = suffix_len[i]

        has_alarm = np.isfinite(pos_start)

        if not adv:
            if has_alarm:
                fp_benign += 1
            continue

        if not np.isfinite(s_start) or not np.isfinite(s_len) or s_len <= 0:
            n_ignored_adv_no_suffix += 1
            continue

        n_adv_suffix += 1

        start = float(s_start)
        end = float(s_start + max(s_len, 0.0))

        if not has_alarm:
            fn_suffix += 1
            continue

        # Treat alarm intervals as half-open [start, end); intervals ending exactly at
        # suffix_start are "before suffix" (not overlapping).
        if pos_end <= start:
            fp_early_adv += 1
            early_offsets.append(float(pos_end - start))
        elif pos_start >= end:
            fn_suffix += 1
        else:
            tp_suffix += 1
            delay = float(max(pos_start, start) - start)
            delays_correct.append(delay)

    denom_prec = tp_suffix + fp_benign + fp_early_adv
    denom_rec = tp_suffix + fn_suffix
    precision = safe_div(tp_suffix, denom_prec)
    recall = safe_div(tp_suffix, denom_rec)
    if np.isfinite(precision + recall) and (precision + recall) > 0:
        f1 = 2.0 * precision * recall / (precision + recall)
    else:
        f1 = float(0.0)

    if delays_correct:
        median_delay = float(np.median(delays_correct))
        mean_delay = float(np.mean(delays_correct))
    else:
        median_delay = float("nan")
        mean_delay = float("nan")

    valid_mask = np.isfinite(score_for_roc)
    if valid_mask.sum() >= 2:
        auroc_adv = compute_auroc(is_adv[valid_mask].astype(int), score_for_roc[valid_mask])
    else:
        auroc_adv = float("nan")

    metrics = SuffixMetrics(
        threshold=float(threshold),
        f1_suffix=float(f1),
        precision_suffix=float(precision),
        recall_suffix=float(recall),
        tp_suffix=int(tp_suffix),
        fp_benign=int(fp_benign),
        fp_early_adv=int(fp_early_adv),
        fn_suffix=int(fn_suffix),
        median_delay=float(median_delay),
        mean_delay=float(mean_delay),
        n_benign=int(n_benign),
        n_adv_suffix=int(n_adv_suffix),
        n_ignored_adv_no_suffix=int(n_ignored_adv_no_suffix),
        auroc_adv=float(auroc_adv),
    )
    return metrics, np.asarray(delays_correct, float), np.asarray(early_offsets, float)


def sweep_thresholds_suffix_f1(
    traces: List[np.ndarray],
    suffix_start: np.ndarray,
    suffix_len: np.ndarray,
    is_adv: np.ndarray,
    step: float,
    score_for_roc: np.ndarray,
    index_offset: float = 0.0,
    interval_len: Optional[float] = None,
    progress_label: Optional[str] = None,
) -> Tuple[SuffixMetrics, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Sweep thresholds over unique finite trace values and pick the one maximizing suffix-aware F1.
    """
    unique_scores = collect_unique_thresholds_from_traces(traces)

    best_metrics: Optional[SuffixMetrics] = None
    best_delays = np.array([], dtype=float)
    best_early_offsets = np.array([], dtype=float)
    best_alarm_start = np.full(len(traces), np.nan, dtype=float)
    best_alarm_end = np.full(len(traces), np.nan, dtype=float) if interval_len is not None else None
    thresholds_list: List[float] = []
    f1_list: List[float] = []

    progress_total = len(unique_scores)
    if progress_label:
        update_subprogress(0, progress_total, progress_label)

    for idx_thr, thr in enumerate(unique_scores, start=1):
        alarm_starts, alarm_ends = compute_alarm_positions_from_traces(
            traces,
            thr,
            step,
            index_offset=index_offset,
            interval_len=interval_len,
        )
        if alarm_ends is None and interval_len is not None:
            alarm_ends = alarm_starts + interval_len
        metrics, delays, early_offsets = evaluate_suffix_metrics_at_threshold(
            alarm_starts,
            alarm_ends,
            suffix_start,
            suffix_len,
            is_adv,
            thr,
            score_for_roc,
        )
        thresholds_list.append(float(thr))
        f1_list.append(metrics.f1_suffix)
        if progress_label:
            update_subprogress(idx_thr, progress_total, progress_label)

        if best_metrics is None:
            best_metrics = metrics
            best_delays = delays
            best_early_offsets = early_offsets
            best_alarm_start = alarm_starts
            if best_alarm_end is not None:
                best_alarm_end = alarm_ends
        else:
            if metrics.f1_suffix > best_metrics.f1_suffix + 1e-12:
                best_metrics = metrics
                best_delays = delays
                best_early_offsets = early_offsets
                best_alarm_start = alarm_starts
                if best_alarm_end is not None:
                    best_alarm_end = alarm_ends
            elif abs(metrics.f1_suffix - best_metrics.f1_suffix) <= 1e-12:
                if metrics.recall_suffix > best_metrics.recall_suffix + 1e-12:
                    best_metrics = metrics
                    best_delays = delays
                    best_early_offsets = early_offsets
                    best_alarm_start = alarm_starts
                    if best_alarm_end is not None:
                        best_alarm_end = alarm_ends
                elif abs(metrics.recall_suffix - best_metrics.recall_suffix) <= 1e-12:
                    if metrics.threshold < best_metrics.threshold:
                        best_metrics = metrics
                        best_delays = delays
                        best_early_offsets = early_offsets
                        best_alarm_start = alarm_starts
                        if best_alarm_end is not None:
                            best_alarm_end = alarm_ends

    assert best_metrics is not None
    return (
        best_metrics,
        np.asarray(thresholds_list, float),
        np.asarray(f1_list, float),
        best_delays,
        best_early_offsets,
        best_alarm_start,
        best_alarm_end,
    )


# ---------------------------------------------------------
# Data loading and merging
# ---------------------------------------------------------


def load_and_merge(pp_csv: str, cpd_csv: str) -> pd.DataFrame:
    """
    Load window-PP and CPD CSVs and align rows.
    """
    df_pp = pd.read_csv(pp_csv)
    df_cpd = pd.read_csv(cpd_csv)

    # Decide alignment key
    if "row_index" in df_pp.columns and "row_index" in df_cpd.columns:
        log("Aligning PP and CPD CSVs using 'row_index'")
        merged = pd.merge(
            df_pp,
            df_cpd,
            on="row_index",
            how="inner",
            suffixes=("_pp", "_cpd"),
        )
    else:
        if len(df_pp) != len(df_cpd):
            raise ValueError(
                f"PP CSV (n={len(df_pp)}) and CPD CSV (n={len(df_cpd)}) lengths differ "
                "and no row_index present in both."
            )
        log("Aligning PP and CPD CSVs by position (no 'row_index' found)")
        df_pp = df_pp.copy()
        df_cpd = df_cpd.copy()
        df_pp["__row_id"] = np.arange(len(df_pp))
        df_cpd["__row_id"] = np.arange(len(df_cpd))
        merged = pd.merge(
            df_pp,
            df_cpd,
            on="__row_id",
            how="inner",
            suffixes=("_pp", "_cpd"),
        )

    if merged.empty:
        raise ValueError("Merged dataframe is empty after alignment.")

    def norm_algo(col: pd.Series) -> pd.Series:
        return col.fillna("").astype(str).str.strip().str.lower()

    if "algorithm_pp" in merged.columns:
        log("Using algorithm_pp column for PP algorithms")
        algo_pp = norm_algo(merged["algorithm_pp"])
    elif "algorithm" in merged.columns:
        log("Using shared 'algorithm' column for PP algorithms (fallback)")
        algo_pp = norm_algo(merged["algorithm"])
    else:
        log("No algorithm column for PP; defaulting to empty strings")
        algo_pp = pd.Series([""] * len(merged), index=merged.index)

    if "algorithm_cpd" in merged.columns:
        log("Using algorithm_cpd column for CPD algorithms")
        algo_cpd = norm_algo(merged["algorithm_cpd"])
    elif "algorithm" in merged.columns:
        log("Using shared 'algorithm' column for CPD algorithms (fallback)")
        algo_cpd = norm_algo(merged["algorithm"])
    else:
        log("No algorithm column for CPD; reusing PP algorithm values")
        algo_cpd = algo_pp

    merged["algorithm_combined"] = np.where(algo_pp != "", algo_pp, algo_cpd)

    def extract_is_adv(frame: pd.DataFrame, col_name: str) -> Optional[pd.Series]:
        if col_name in frame.columns:
            return frame[col_name].fillna(0).astype(int)
        return None

    is_adv_pp = extract_is_adv(merged, "is_adversarial_pp")
    is_adv_cpd = extract_is_adv(merged, "is_adversarial_cpd")
    is_adv_plain = extract_is_adv(merged, "is_adversarial")

    if is_adv_pp is not None:
        log("Using is_adversarial_pp for labels")
        merged["is_adversarial_combined"] = is_adv_pp
    elif is_adv_cpd is not None:
        log("Using is_adversarial_cpd for labels")
        merged["is_adversarial_combined"] = is_adv_cpd
    elif is_adv_plain is not None:
        log("Using shared is_adversarial for labels (fallback)")
        merged["is_adversarial_combined"] = is_adv_plain
    else:
        raise ValueError("No is_adversarial column found in either CSV.")

    # suffix_start_postprefix and suffix_len_tokens from CPD if available
    suffix_start = None
    suffix_len = None
    if "suffix_start_postprefix" in merged.columns:
        log("Using suffix_start_postprefix from CPD CSV")
        suffix_start = merged["suffix_start_postprefix"].astype(float)
    elif "suffix_start_postprefix_cpd" in merged.columns:
        log("Using suffix_start_postprefix_cpd from merged CPD columns")
        suffix_start = merged["suffix_start_postprefix_cpd"].astype(float)

    if "suffix_len_tokens" in merged.columns:
        log("Using suffix_len_tokens from CPD CSV")
        suffix_len = merged["suffix_len_tokens"].astype(float)
    elif "suffix_len_tokens_cpd" in merged.columns:
        log("Using suffix_len_tokens_cpd from merged CPD columns")
        suffix_len = merged["suffix_len_tokens_cpd"].astype(float)

    # Fallback to PP's suffix_start_token_idx if needed
    if suffix_start is None:
        if "suffix_start_token_idx" in merged.columns:
            log("Suffix start fallback: suffix_start_token_idx (PP)")
            suffix_start = merged["suffix_start_token_idx"].astype(float)
        elif "suffix_start_token_idx_pp" in merged.columns:
            log("Suffix start fallback: suffix_start_token_idx_pp (PP)")
            suffix_start = merged["suffix_start_token_idx_pp"].astype(float)
        else:
            log("No suffix start found; filling NaN")
            suffix_start = pd.Series([np.nan] * len(merged), index=merged.index)
    if suffix_len is None:
        log("No suffix length found; filling NaN")
        suffix_len = pd.Series([np.nan] * len(merged), index=merged.index)

    merged["suffix_start_postprefix_combined"] = suffix_start
    merged["suffix_len_tokens_combined"] = suffix_len

    if "num_tokens_postprefix" in merged.columns:
        log("Using num_tokens_postprefix")
        merged["num_tokens_postprefix_combined"] = merged["num_tokens_postprefix"].astype(float)
    elif "num_tokens_postprefix_cpd" in merged.columns:
        log("Using num_tokens_postprefix_cpd")
        merged["num_tokens_postprefix_combined"] = merged["num_tokens_postprefix_cpd"].astype(float)
    else:
        log("No num_tokens_postprefix found; filling NaN")
        merged["num_tokens_postprefix_combined"] = np.nan

    # CPD fields
    if "online_max_W_plus" in merged.columns:
        log("Using online_max_W_plus for CPD scores")
        merged["cpd_score"] = merged["online_max_W_plus"].astype(float)
    elif "online_max_W_plus_cpd" in merged.columns:
        log("Using online_max_W_plus_cpd for CPD scores")
        merged["cpd_score"] = merged["online_max_W_plus_cpd"].astype(float)
    else:
        raise ValueError("online_max_W_plus column not found in CPD CSV.")

    if "online_t_alarm" in merged.columns:
        log("Using online_t_alarm for CPD alarm positions")
        merged["cpd_alarm_pos"] = merged["online_t_alarm"].astype(float)
    elif "online_t_alarm_cpd" in merged.columns:
        log("Using online_t_alarm_cpd for CPD alarm positions")
        merged["cpd_alarm_pos"] = merged["online_t_alarm_cpd"].astype(float)
    else:
        log("No CPD alarm position column; filling NaN")
        merged["cpd_alarm_pos"] = np.nan

    return merged


# ---------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------


def plot_f1_vs_threshold(
    thresholds: np.ndarray,
    f1_values: np.ndarray,
    best_metrics: SuffixMetrics,
    method_name: str,
    out_dir: str,
) -> None:
    plt.figure(figsize=(5, 3))
    plt.plot(thresholds, f1_values, linewidth=1.5)
    plt.axvline(best_metrics.threshold, linestyle="--")
    plt.xlabel("Threshold")
    plt.ylabel("Suffix F1")
    plt.title(f"F1 vs threshold — {method_name}\n(best F1={best_metrics.f1_suffix:.3f} @ h={best_metrics.threshold:.3f})")
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{method_name}_f1_vs_threshold.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    log(f"Saved plot: {out_path}")


def plot_roc_segmented(
    df: pd.DataFrame,
    scores: np.ndarray,
    method_name: str,
    out_dir: str,
) -> None:
    """
    Plot ROC segments: overall / no_gcg / gcg_only.
    """
    algo = df["algorithm_combined"].fillna("").astype(str).str.lower()
    is_adv_all = df["is_adversarial_combined"].fillna(0).astype(int).to_numpy()

    mask_gcg = (algo == "gcg")
    segments: Dict[str, np.ndarray] = {
        "overall": np.ones(len(df), dtype=bool),
        "no_gcg": ~mask_gcg,
        "gcg_only": mask_gcg | ((~mask_gcg) & (is_adv_all == 0)),
    }

    # Paper-friendly style: larger fonts/lines + vector PDF output (plus PNG for convenience).
    plt.figure(figsize=(4.2, 4.0))
    ax = plt.gca()

    for seg_name, seg_mask in segments.items():
        if seg_mask.sum() == 0:
            continue
        y = is_adv_all[seg_mask]
        s = scores[seg_mask].astype(float)
        finite_mask = np.isfinite(s)
        y = y[finite_mask]
        s = s[finite_mask]
        if len(y) < 2 or len(np.unique(y)) < 2:
            continue
        fpr, tpr = compute_roc_curve(y, s)
        auc_val = compute_auroc(y, s)
        ax.plot(fpr, tpr, label=f"{seg_name} (AUC={auc_val:.3f})", linewidth=2.0)

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.5, linewidth=1.5)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    ax.set_title(f"ROC (adv vs benign) — {method_name}", fontsize=11)
    ax.tick_params(axis="both", labelsize=9)
    ax.legend(loc="lower right", fontsize=8, frameon=True)
    plt.tight_layout()

    out_base = os.path.join(out_dir, f"{method_name}_roc_segmented")
    plt.savefig(f"{out_base}.pdf", bbox_inches="tight")
    plt.savefig(f"{out_base}.png", dpi=200, bbox_inches="tight")
    plt.close()
    log(f"Saved plot: {out_base}.pdf")


def plot_delay_hist(
    delays: np.ndarray,
    method_name: str,
    out_dir: str,
    kind: str,
) -> None:
    if delays.size == 0:
        return
    plt.figure(figsize=(8, 4))
    plt.hist(delays, bins=30, edgecolor="black", alpha=0.7)
    if kind == "delay":
        plt.xlabel("Alarm delay (tokens after suffix start)")
        plt.title(f"Delay histogram (TP_suffix) — {method_name}")
    else:
        plt.xlabel("Early alarm offset (interval end - suffix_start, tokens)")
        plt.title(f"Early alarm histogram (FP_early_adv) — {method_name}")
    plt.ylabel("Count")
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{method_name}_{kind}_hist.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    log(f"Saved plot: {out_path}")


def per_attack_breakdown(
    df: pd.DataFrame,
    alarm_start: np.ndarray,
    alarm_end: Optional[np.ndarray],
    suffix_start: np.ndarray,
    suffix_len: np.ndarray,
    is_adv: np.ndarray,
    method_name: str,
    threshold: float,
    out_dir: str,
    score_for_roc: np.ndarray,
) -> None:
    algos = df["algorithm_combined"].fillna("").astype(str)
    rows: List[Dict[str, object]] = []
    unique_algos = sorted(algos.unique())
    for algo in unique_algos:
        mask = algos == algo
        if mask.sum() == 0:
            continue
        alarm_start_masked = alarm_start[mask]
        alarm_end_masked = alarm_end[mask] if alarm_end is not None else None
        metrics, _, _ = evaluate_suffix_metrics_at_threshold(
            alarm_start_masked,
            alarm_end_masked,
            suffix_start[mask],
            suffix_len[mask],
            is_adv[mask],
            threshold,
            score_for_roc[mask],
        )
        rows.append(
            {
                "algorithm": algo,
                "method": method_name,
                "threshold": metrics.threshold,
                "F1_suffix": metrics.f1_suffix,
                "precision_suffix": metrics.precision_suffix,
                "recall_suffix": metrics.recall_suffix,
                "TP_suffix": metrics.tp_suffix,
                "FP_benign": metrics.fp_benign,
                "FP_early_adv": metrics.fp_early_adv,
                "FN_suffix": metrics.fn_suffix,
                "median_delay": metrics.median_delay,
                "mean_delay": metrics.mean_delay,
                "n_benign": metrics.n_benign,
                "n_adv_with_suffix": metrics.n_adv_suffix,
                "n_ignored_adv_no_suffix": metrics.n_ignored_adv_no_suffix,
            }
        )

    if not rows:
        return
    out_df = pd.DataFrame(rows)
    out_path = os.path.join(out_dir, f"suffix_breakdown_{method_name}.csv")
    out_df.to_csv(out_path, index=False)
    log(f"Wrote per-attack breakdown CSV: {out_path}")


def compute_detection_split(
    alarm_start: np.ndarray,
    alarm_end: Optional[np.ndarray],
    suffix_start: np.ndarray,
    suffix_len: np.ndarray,
    is_adv: np.ndarray,
    *,
    mask: Optional[np.ndarray] = None,
    all_intervals: Optional[List[List[Tuple[float, float]]]] = None,
) -> Tuple[float, float, float, float]:
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
    else:
        mask = np.ones(len(is_adv), dtype=bool)

    alarm_start_sel = alarm_start[mask]
    alarm_end_sel = alarm_end[mask] if alarm_end is not None else None
    suffix_start_sel = suffix_start[mask]
    suffix_len_sel = suffix_len[mask]
    is_adv_sel = is_adv[mask]

    mask_adv_suffix = (
        (is_adv_sel.astype(int) == 1)
        & np.isfinite(suffix_start_sel)
        & np.isfinite(suffix_len_sel)
        & (suffix_len_sel > 0)
    )
    suffix_end_sel = suffix_start_sel + suffix_len_sel

    if all_intervals is not None:
        intervals_masked = [all_intervals[i] for i in np.where(mask)[0]]

        # Auto-detect windowed detector by checking interval lengths.
        # NOTE: some operating points (e.g., strict FPR thresholds) may produce
        # many leading prompts with no alarms, so we scan until we observe a few
        # non-empty intervals instead of inspecting only the first N prompts.
        is_windowed = False
        interval_lengths: List[float] = []
        max_interval_samples = 32
        for intervals in intervals_masked:
            for start, end in intervals:
                interval_lengths.append(float(end - start))
                if len(interval_lengths) >= max_interval_samples:
                    break
            if len(interval_lengths) >= max_interval_samples:
                break
        if interval_lengths:
            median_len = float(np.median(interval_lengths))
            is_windowed = median_len > 1.5

        early_only = 0
        before_and_in = 0
        in_suffix_only = 0
        benign = 0
        for idx_row, intervals in enumerate(intervals_masked):
            has_alarm = len(intervals) > 0
            adv = int(is_adv_sel[idx_row]) == 1
            if not adv:
                if has_alarm:
                    benign += 1
                continue
            if not mask_adv_suffix[idx_row]:
                continue
            s_start = suffix_start_sel[idx_row]
            s_end = suffix_end_sel[idx_row]

            if is_windowed:
                # For windowed detectors: check for SINGLE straddling window
                # (window starts before ν and crosses into/over ν).
                has_straddle = any((start < s_start) and (end > s_start) for (start, end) in intervals)
                has_before = any(end <= s_start for (start, end) in intervals)
                has_in_only = any(
                    (start >= s_start) and (start < s_end)
                    for (start, end) in intervals
                )

                if has_straddle:
                    before_and_in += 1
                elif has_before:
                    early_only += 1
                elif has_in_only:
                    in_suffix_only += 1
            else:
                # For point detectors: check for MULTIPLE alarms (original logic)
                has_before = any(end <= s_start for (start, end) in intervals)
                has_overlap = any((start < s_end) and (end > s_start) for (start, end) in intervals)
                if has_before and has_overlap:
                    before_and_in += 1
                elif has_before:
                    early_only += 1
                elif has_overlap:
                    in_suffix_only += 1
        return float(early_only), float(before_and_in), float(in_suffix_only), float(benign)

    valid_adv = mask_adv_suffix & np.isfinite(alarm_start_sel)
    if alarm_end_sel is None:
        alarm_end_sel = alarm_start_sel
    early_only = np.sum(valid_adv & (alarm_end_sel < suffix_start_sel))
    overlap = valid_adv & (alarm_start_sel < suffix_end_sel) & (alarm_end_sel > suffix_start_sel)
    span_before = np.sum(overlap & (alarm_start_sel < suffix_start_sel))
    in_suffix_only = np.sum(overlap & ~(alarm_start_sel < suffix_start_sel))
    benign = np.sum((is_adv_sel.astype(int) == 0) & np.isfinite(alarm_start_sel))
    return float(early_only), float(span_before), float(in_suffix_only), float(benign)


def store_detection_counts(
    method_name: str,
    detection_counts: Dict[str, Tuple[float, float, float, float]],
    alarm_pairs: Dict[str, Tuple[np.ndarray, Optional[np.ndarray], List[List[Tuple[float, float]]]]],
    algo_arr: np.ndarray,
    suffix_start: np.ndarray,
    suffix_len: np.ndarray,
    is_adv: np.ndarray,
    storage: Dict[str, Dict[str, Dict[str, Tuple[float, float, float, float]]]],
    *,
    allowed_metrics: Optional[Set[str]] = None,
) -> None:
    if not detection_counts:
        return
    allowed_metrics = set(allowed_metrics) if allowed_metrics is not None else None
    entry = storage.setdefault(method_name, {"overall": {}, "segments": {}})
    for metric, vals in detection_counts.items():
        if allowed_metrics is not None and metric not in allowed_metrics:
            continue
        entry["overall"][metric] = vals
    for segment in SEGMENT_ALGOS:
        mask = algo_arr == segment
        if not np.any(mask):
            continue
        seg_counts = entry["segments"].setdefault(segment, {})
        for label, (start_arr, end_arr, intervals) in alarm_pairs.items():
            if allowed_metrics is not None and label not in allowed_metrics:
                continue
            seg_counts[label] = compute_detection_split(
                start_arr,
                end_arr,
                suffix_start,
                suffix_len,
                is_adv,
                mask=mask,
                all_intervals=intervals,
            )

    # Add benign-only segment based on non-adversarial prompts to keep benign alarms visible.
    mask_benign = is_adv.astype(int) == 0
    if np.any(mask_benign):
        seg_counts = entry["segments"].setdefault("benign", {})
        for label, (start_arr, end_arr, intervals) in alarm_pairs.items():
            if allowed_metrics is not None and label not in allowed_metrics:
                continue
            seg_counts[label] = compute_detection_split(
                start_arr,
                end_arr,
                suffix_start,
                suffix_len,
                is_adv,
                mask=mask_benign,
                all_intervals=intervals,
            )
        log(f"Stored benign segment counts for {method_name} (assumes non-adversarial prompts share algorithm label)")

def plot_detection_bars(
    method_name: str,
    counts: Dict[str, Tuple[float, float, float, float]],
    out_dir: str,
    *,
    suffix_label: Optional[str] = None,
    title: Optional[str] = None,
    allowed_metrics: Optional[Set[str]] = None,
    fpr_metric: str = "fpr10",
    cpd_threshold: Optional[float] = None,
) -> None:
    if not counts:
        return
    allowed_metrics = set(allowed_metrics) if allowed_metrics is not None else None
    sns.set_theme(style="white", context="talk")
    x_categories = ["Before suffix", "Before + In suffix", "In suffix", "In benign"]
    metric_list = ["overall_f1", "suffix_f1", fpr_metric]
    keys = [
        key
        for key in metric_list
        if key in counts and (allowed_metrics is None or key in allowed_metrics)
    ]
    if not keys:
        return
    x = np.arange(len(x_categories))
    width = 0.18
    colors = sns.color_palette("muted", n_colors=len(keys))
    plt.figure(figsize=(8, 6))
    for idx, key in enumerate(keys):
        values = counts[key]
        offset = (idx - (len(keys) - 1) / 2.0) * width
        label = THRESHOLD_LABELS.get(key, key)
        bars = plt.bar(x + offset, values, width=width, label=label, color=colors[idx])
        plt.bar_label(bars, fmt="%d", padding=4, fontsize=14)
    plt.xticks(x, x_categories, rotation=0, fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel("Count of triggers (adv + suffix)", fontsize=14)
    plot_title = title or f"Locality of Triggers — {method_name}"
    if method_name == "cpd_online" and cpd_threshold is not None:
        plot_title += f" [best τ={cpd_threshold:g}]"
    plt.title(plot_title, fontsize=16)
    plt.legend()
    plt.ylim(bottom=0)
    ax = plt.gca()
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout()
    filename = f"{method_name}_detection_split.pdf"
    if suffix_label:
        filename = f"{method_name}_{suffix_label}_detection_split.pdf"
    out_path = os.path.join(out_dir, filename)
    plt.savefig(out_path, dpi=300)
    plt.close()
    log(f"Saved plot: {out_path}")


def plot_detection_bars_stacked(
    method_name: str,
    segments: Dict[str, Dict[str, Tuple[float, float, float, float]]],
    out_dir: str,
    *,
    allowed_metrics: Optional[Set[str]] = None,
    fpr_metric: str = "fpr10",
    cpd_threshold: Optional[float] = None,
) -> None:
    if not segments:
        return
    allowed_metrics = set(allowed_metrics) if allowed_metrics is not None else None
    sns.set_theme(style="white", context="talk")
    x_categories = [
        "Before suffix",
        "Before + in suffix",
        "In suffix",
        "In benign",
    ]
    # Only plot overall F1 + FPR to keep the legend compact (omit suffix_F1 bars).
    metric_keys = ["overall_f1", fpr_metric]
    available_keys = []
    for k in metric_keys:
        if allowed_metrics is not None and k not in allowed_metrics:
            continue
        if any(k in seg_counts for seg_counts in segments.values()):
            available_keys.append(k)
    if not available_keys:
        return

    seg_names = sorted(segments.keys())
    hatches = ['//', 'oo', 'xx', '--', '++', 'OO', '..']
    hatch_map = {seg: hatches[i % len(hatches)] for i, seg in enumerate(seg_names)}

    muted = sns.color_palette("muted", n_colors=3)
    colors = {
        "overall_f1": muted[0],
        "suffix_f1": muted[1],
        fpr_metric: muted[2] if len(muted) > 2 else muted[-1],
    }
    edge_colors = {k: tuple(np.clip(np.array(v) * 0.6, 0, 1)) for k, v in colors.items()}

    x = np.arange(len(x_categories))
    width = 0.20
    group_spacing = 0.06

    plt.figure(figsize=(8, 6))
    for idx_key, key in enumerate(available_keys):
        offset = (idx_key - (len(available_keys) - 1) / 2.0) * (width + group_spacing)
        bars_total = np.zeros_like(x, dtype=float)
        for seg in seg_names:
            seg_counts = segments.get(seg, {})
            if key not in seg_counts:
                continue
            vals = np.array(seg_counts[key], dtype=float)
            bars = plt.bar(
                x + offset,
                vals,
                width=width,
                bottom=bars_total,
                color=colors.get(key, None),
                hatch=hatch_map[seg],
                edgecolor=edge_colors.get(key, "black"),
                linewidth=0.4,
                label=f"{THRESHOLD_LABELS.get(key, key)} ({seg})",
            )
            bars_total = bars_total + vals
        for xi, total in zip(x + offset, bars_total):
            if np.isfinite(total):
                plt.text(
                    xi,
                    total + 3,
                    f"{int(total)}",
                    ha="center",
                    va="bottom",
                    fontsize=14,
                )

    ax = plt.gca()
    for idx, xc in enumerate(x):
        ax.axvline(x=xc, color="gray", alpha=0.05, linewidth=1.0)
        if idx % 2 == 0:
            ax.axvspan(xc - 0.35, xc + 0.35, color="gray", alpha=0.03)

    plt.xticks(x, x_categories, rotation=0, fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel("Count of detections (adv + suffix)", fontsize=14)
    plt.title(f"Detection timing split — {method_name}", fontsize=16)
    ymax = plt.gca().get_ylim()[1]
    plt.ylim(bottom=0, top=ymax * 1.1)

    from matplotlib.patches import Patch

    metric_handles = [
        Patch(facecolor=colors.get(k, "#777777"), edgecolor="black", label=THRESHOLD_LABELS.get(k, k))
        for k in available_keys
    ]
    hatch_handles = [
        Patch(facecolor="white", edgecolor="black", hatch=hatch_map[seg], label=seg, linewidth=0.6)
        for seg in seg_names
    ]
    plt.legend(
        handles=metric_handles + hatch_handles,
        loc="upper right",
        framealpha=0.9,
        borderpad=0.6,
        handlelength=1.4,
        fontsize=14,
    )
    ax.grid(False)
    for spine in [ax.spines["top"], ax.spines["right"]]:
        spine.set_visible(False)
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{method_name}_detection_split_stacked.pdf")
    plt.savefig(out_path, dpi=300)
    plt.close()
    log(f"Saved plot: {out_path}")


def detection_counts_to_df(storage: Dict[str, Dict[str, Dict[str, Tuple[float, float, float, float]]]]) -> pd.DataFrame:
    rows = []
    for method, entry in storage.items():
        overall = entry.get("overall", {})
        segments = entry.get("segments", {})
        if not overall and segments:
            synth: Dict[str, Tuple[float, float, float, float]] = {}
            for seg_counts in segments.values():
                for metric, vals in seg_counts.items():
                    accum = np.array(synth.get(metric, (0.0, 0.0, 0.0, 0.0)), dtype=float)
                    synth[metric] = tuple(accum + np.array(vals, dtype=float))
            overall = synth
        for metric, vals in overall.items():
            rows.append(
                {
                    "method": method,
                    "segment": "overall",
                    "metric": metric,
                    "before_suffix": vals[0],
                    "before_in_suffix": vals[1],
                    "in_suffix": vals[2],
                    "in_benign": vals[3],
                }
            )
        for seg_name, seg_counts in segments.items():
            for metric, vals in seg_counts.items():
                rows.append(
                    {
                        "method": method,
                        "segment": seg_name,
                        "metric": metric,
                        "before_suffix": vals[0],
                        "before_in_suffix": vals[1],
                        "in_suffix": vals[2],
                        "in_benign": vals[3],
                    }
                )
    return pd.DataFrame(rows)


def df_to_detection_counts(df: pd.DataFrame) -> Dict[str, Dict[str, Dict[str, Tuple[float, float, float, float]]]]:
    storage: Dict[str, Dict[str, Dict[str, Tuple[float, float, float, float]]]] = {}
    for _, row in df.iterrows():
        method = str(row["method"])
        segment = str(row["segment"])
        metric = str(row["metric"])
        vals = (
            float(row["before_suffix"]),
            float(row["before_in_suffix"]),
            float(row["in_suffix"]),
            float(row["in_benign"]),
        )
        entry = storage.setdefault(method, {"overall": {}, "segments": {}})
        if segment == "overall":
            entry["overall"][metric] = vals
        else:
            entry["segments"].setdefault(segment, {})[metric] = vals

    # If no explicit benign segment exists, synthesize one so false alarms appear in stacked plots.
    for method, entry in storage.items():
        segments = entry.setdefault("segments", {})
        if "benign" in segments:
            continue
        overall = entry.get("overall", {})
        if not overall:
            continue
        benign_counts: Dict[str, Tuple[float, float, float, float]] = {}
        for metric, vals in overall.items():
            # Only the in_benign value is relevant for benign-only prompts.
            benign_counts[metric] = (0.0, 0.0, 0.0, float(vals[3]))
        if benign_counts:
            segments["benign"] = benign_counts
            log(f"Synthesized benign segment for method '{method}' from overall counts")
    return storage


def filter_detection_methods(
    detection_bar_data: Dict[str, Dict[str, Dict[str, Tuple[float, float, float, float]]]],
    allowed_methods: Optional[Set[str]],
) -> Dict[str, Dict[str, Dict[str, Tuple[float, float, float, float]]]]:
    if allowed_methods is None:
        return detection_bar_data
    return {m: data for m, data in detection_bar_data.items() if m in allowed_methods}


def merge_on_keys(existing: pd.DataFrame, new: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
    if existing is None or existing.empty:
        return new.copy()
    keep = existing[~existing.set_index(keys).index.isin(new.set_index(keys).index)]
    return pd.concat([keep, new], ignore_index=True)


def plot_detection_bars_combined(
    detection_bar_data: Dict[str, Dict[str, Dict[str, Tuple[float, float, float, float]]]],
    methods: List[str],
    out_dir: str,
    *,
    allowed_metrics: Optional[Set[str]] = None,
    fpr_metric: str = "fpr10",
    cpd_threshold: Optional[float] = None,
) -> None:
    """
    Combined grouped bar chart across methods for the same metric (overall segment only).
    """
    if len(methods) < 2:
        return
    allowed_metrics = set(allowed_metrics) if allowed_metrics is not None else None
    x_categories = ["Before suffix", "Before + In suffix", "In suffix", "In benign"]
    metrics = ["overall_f1", "suffix_f1", fpr_metric]
    if allowed_metrics is not None:
        metrics = [m for m in metrics if m in allowed_metrics]

    # Collect all segments present across methods (excluding overall; we want per-algorithm stacks).
    segment_names = set()
    for data in detection_bar_data.values():
        segment_names.update(data.get("segments", {}).keys())
    if not segment_names:
        return
    segment_names = sorted(segment_names)
    sns.set_theme(style="white", context="talk")
    metric_palette = sns.color_palette("muted", n_colors=3)
    colors = {
        "overall_f1": metric_palette[0],
        "suffix_f1": metric_palette[1],
        fpr_metric: metric_palette[2] if len(metric_palette) > 2 else metric_palette[-1],
    }
    hatches = ['//', 'oo', 'xx', '--', '++', 'OO', '..']
    hatch_map = {seg: hatches[i % len(hatches)] for i, seg in enumerate(segment_names)}
    method_palette = sns.color_palette("muted", n_colors=len(methods))
    method_colors = {m: method_palette[i % len(method_palette)] for i, m in enumerate(methods)}
    hatches = ['//', 'oo', 'xx', '--', '++', 'OO', '..']
    hatch_map = {}
    for i, seg in enumerate(segment_names):
        if seg == "benign":
            hatch_map[seg] = ""
        else:
            hatch_map[seg] = hatches[i % len(hatches)]

    # Pre-compute totals per method/metric to convert counts into percentages.
    totals = {}
    for m in methods:
        method_entry = detection_bar_data.get(m, {})
        for metric in metrics:
            total = 0.0
            for seg_counts in method_entry.get("segments", {}).values():
                vals = seg_counts.get(metric)
                if vals is not None:
                    total += float(np.nansum(np.array(vals, dtype=float)))
            # Fallback to overall counts if no per-segment totals exist.
            if total == 0.0 and metric in method_entry.get("overall", {}):
                total = float(np.nansum(np.array(method_entry["overall"][metric], dtype=float)))
            totals[(m, metric)] = total if total > 0 else float("nan")

    for metric in metrics:
        plt.figure(figsize=(10, 6))
        x = np.arange(len(x_categories))
        # Keep bars tight within each segment (no gap between methods).
        width = min(0.22, 0.9 / max(len(methods), 1))
        method_spacing = 0.0
        for idx_m, m in enumerate(methods):
            display_name = METHOD_LABELS.get(m, m)
            method_entry = detection_bar_data.get(m, {})
            offset_method = (idx_m - (len(methods) - 1) / 2.0) * (width + method_spacing)
            bottoms = np.zeros(len(x_categories), dtype=float)
            for seg in segment_names:
                counts = method_entry.get("segments", {}).get(seg, {})
                vals = counts.get(metric, (0.0, 0.0, 0.0, 0.0))
                vals = np.array(vals, dtype=float)
                total = totals.get((m, metric), float("nan"))
                if np.isfinite(total) and total > 0:
                    vals = vals / total * 100.0
                else:
                    vals = np.zeros_like(vals)
                bars = plt.bar(
                    x + offset_method,
                    vals,
                    width=width,
                    bottom=bottoms,
                    color=method_colors.get(m, "#777777"),
                    edgecolor=tuple(np.clip(np.array(method_colors.get(m, (0.5, 0.5, 0.5))) * 0.6, 0, 1)),
                    linewidth=0.6,
                    hatch=hatch_map.get(seg, None),
                    label=display_name if seg == segment_names[0] else None,
                )
                bottoms += vals
            # Add total labels on top of each method's stack
            for xi, total in zip(x + offset_method, bottoms):
                if np.isfinite(total) and total > 0:
                    plt.text(xi, total + 0.8, f"{total:.0f}%", ha="center", va="bottom", fontsize=14)

        plt.xticks(x, x_categories, rotation=0, fontsize=14)
        plt.yticks(fontsize=14)
        plt.ylabel("Share of triggers (%)", fontsize=14)
        title = "Per attack locality distributions at F1-optimal" if metric == "overall_f1" else f"Per attack locality distributions ({THRESHOLD_LABELS.get(metric, metric)})"
        plt.title(title, fontsize=16)
        from matplotlib.patches import Patch
        method_handles = [
            Patch(facecolor=method_colors[m], edgecolor="black", label=METHOD_LABELS.get(m, m))
            for m in methods
        ]
        segment_handles = []
        for seg in segment_names:
            if seg == "benign":
                continue
            segment_handles.append(
                Patch(facecolor="white", edgecolor="black", hatch=hatch_map[seg], label=seg, linewidth=0.6)
            )
        plt.legend(
            handles=method_handles + segment_handles,
            loc="upper left",
            bbox_to_anchor=(0.01, 0.99),
            framealpha=0.9,
            fontsize=14,
        )
        plt.ylim(bottom=0)
        ax = plt.gca()
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        plt.tight_layout()
        filename = f"combined_stacked_{metric}_detection_split.pdf"
        out_path = os.path.join(out_dir, filename)
        plt.savefig(out_path, dpi=300)
        plt.close()
        log(f"Saved plot: {out_path}")


def render_detection_plots(
    detection_bar_data: Dict[str, Dict[str, Dict[str, Tuple[float, float, float, float]]]],
    roc_dir: str,
    *,
    allowed_metrics: Optional[Set[str]] = None,
    allowed_methods: Optional[Set[str]] = None,
    fpr_metric: str = "fpr10",
    cpd_threshold: Optional[float] = None,
) -> None:
    methods_order = sorted(detection_bar_data.keys())
    if allowed_methods is not None:
        methods_order = [m for m in methods_order if m in allowed_methods]
    for method_name, entry in detection_bar_data.items():
        if allowed_methods is not None and method_name not in allowed_methods:
            continue
        segments = entry.get("segments", {})
        if segments:
            plot_detection_bars_stacked(
                method_name,
                segments,
                roc_dir,
                allowed_metrics=allowed_metrics,
                fpr_metric=fpr_metric,
                cpd_threshold=cpd_threshold,
            )
        overall_counts = entry.get("overall", {})
        if overall_counts:
            plot_detection_bars(
                method_name,
                overall_counts,
                roc_dir,
                allowed_metrics=allowed_metrics,
                fpr_metric=fpr_metric,
                cpd_threshold=cpd_threshold,
            )
        for segment, seg_counts in entry.get("segments", {}).items():
            if not seg_counts:
                continue
            title = f"Locality of Triggers — {method_name} ({segment})"
            plot_detection_bars(
                method_name,
                seg_counts,
                roc_dir,
                suffix_label=segment,
                title=title,
                allowed_metrics=allowed_metrics,
                fpr_metric=fpr_metric,
                cpd_threshold=cpd_threshold,
            )
    if len(methods_order) >= 2:
        plot_detection_bars_combined(
            detection_bar_data,
            methods_order,
            roc_dir,
            allowed_metrics=allowed_metrics,
            fpr_metric=fpr_metric,
            cpd_threshold=cpd_threshold,
        )


def compute_suffix_stats(
    df,
    alarm_start,
    alarm_end,
    suffix_start,
    suffix_len,
    is_adv,
    method_name,
    out_dir,
    all_intervals: Optional[List[List[Tuple[float, float]]]] = None,
):
    algos = df["algorithm_combined"].fillna("").astype(str)
    unique_algos = sorted(algos.unique())
    rows = []

    for algo in unique_algos:
        mask = (algos == algo)
        if mask.sum() == 0:
            continue

        idxs = np.where(mask)[0]
        pos_start = alarm_start[mask]
        pos_end = alarm_end[mask] if alarm_end is not None else pos_start
        ss = suffix_start[mask]
        sl = suffix_len[mask]
        adv = is_adv[mask]
        n_total = len(pos_start)

        intervals_subset = None
        if all_intervals is not None:
            intervals_subset = [all_intervals[i] for i in idxs]

        fired = np.isfinite(pos_start) if intervals_subset is None else np.array([len(iv) > 0 for iv in intervals_subset])

        mask_adv_suffix = (adv == 1) & np.isfinite(ss) & (sl > 0)

        n_fired = int(fired.sum())
        n_benign_fired = int(((adv == 0) & fired).sum())
        suffix_end = ss + sl

        n_early = 0
        n_in = 0
        n_late = 0
        n_no_alarm = 0
        n_before_and_in = 0

        for j in range(n_total):
            has_suffix = bool(mask_adv_suffix[j])
            if not has_suffix:
                continue
            s_start = ss[j]
            s_end = suffix_end[j]
            if intervals_subset is None:
                has_alarm = bool(np.isfinite(pos_start[j]))
                intervals = []
                if has_alarm:
                    intervals.append((pos_start[j], pos_end[j]))
            else:
                intervals = intervals_subset[j]
                has_alarm = len(intervals) > 0

            if not has_alarm:
                n_no_alarm += 1
                continue

            has_before = any(end <= s_start for (start, end) in intervals)
            has_overlap = any((start < s_end) and (end > s_start) for (start, end) in intervals)
            has_after_only = any(start >= s_end for (start, end) in intervals)

            if has_before and not has_overlap:
                n_early += 1
            if has_overlap:
                n_in += 1
            if has_before and has_overlap:
                n_before_and_in += 1
            if has_after_only and not has_overlap:
                n_late += 1

        rows.append({
            "algorithm": algo,
            "method": method_name,
            "num_prompts": n_total,
            "num_fired": n_fired,
            "num_benign_fired": n_benign_fired,
            "num_early": n_early,
            "num_before_and_in": n_before_and_in,
            "num_in_suffix": n_in,
            "num_late": n_late,
            "num_no_alarm": n_no_alarm,
        })

    out_df = pd.DataFrame(rows)
    out_path = os.path.join(out_dir, f"suffix_stats_{method_name}.csv")
    out_df.to_csv(out_path, index=False)
    print(f"[OK] Wrote suffix stats table → {out_path}")
    return rows


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare suffix-aware detection of window-PP and CPD-online, and export detection-only metrics.")
    parser.add_argument("--cpd-csv", required=True, help="CSV from run_cpd_batch.py")
    parser.add_argument("--pp-csv", required=True, help="Per-prompt window-PP CSV (e.g., *_pp_f1.csv)")
    parser.add_argument("--roc-dir", required=True, help="Directory to store plots and per-attack breakdowns")
    parser.add_argument("--out-csv", required=True, help="Suffix summary CSV output path (suffix_eval_summary.csv)")
    parser.add_argument(
        "--debug-merged-csv",
        default=None,
        help="Optional path to write the merged PP+CPD dataframe (can be large).",
    )
    parser.add_argument(
        "--window-sizes",
        nargs="*",
        type=int,
        default=[5, 10, 15, 20],
        help="Window sizes to evaluate for window-PP (default: 5 10 15 20)",
    )
    parser.add_argument(
        "--detection-csv-load",
        help="Load detection counts CSV (from a previous run) and render plots only.",
    )
    parser.add_argument(
        "--detection-csv-save",
        help="Save detection counts CSV for reuse (default: <roc_dir>/detection_counts.csv).",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["cpd", "pp", "window"],
        default=["cpd", "pp", "window"],
        help="Which detectors to run (default: all).",
    )
    parser.add_argument(
        "--merge-output",
        action="store_true",
        help="Merge new results into existing CSVs instead of overwriting other methods.",
    )
    parser.add_argument(
        "--timing-metrics",
        nargs="+",
        help="Which metrics to include in detection timing split plots (default: overall_f1 suffix_f1 fpr<target%%>).",
    )
    parser.add_argument(
        "--timing-methods",
        nargs="+",
        help="Restrict detection timing plots to these methods (e.g., cpd_online window_pp_w10). Default: all methods.",
    )
    parser.add_argument(
        "--timing-best-window",
        action="store_true",
        help="Include only the best window_pp_* method (by F1_suffix in --out-csv) plus any methods specified in --timing-methods.",
    )
    parser.add_argument(
        "--fpr-target",
        type=float,
        default=0.10,
        help="Target FPR (e.g., 0.09 for 9%%, 0.01 for 1%%). Default: 0.10.",
    )
    parser.add_argument(
        "--cpd-h",
        type=float,
        default=None,
        help="Optional CPD threshold annotation (overrides detected best τ for cpd_online).",
    )
    args = parser.parse_args()

    run_cpd = "cpd" in args.methods
    run_pp = "pp" in args.methods
    run_window = "window" in args.methods
    fpr_metric = f"fpr{int(round(args.fpr_target * 100))}"
    THRESHOLD_LABELS[fpr_metric] = f"{args.fpr_target * 100:g}% FPR"
    best_cpd_threshold: Optional[float] = None

    if args.timing_metrics is None:
        allowed_timing_metrics = {"overall_f1", "suffix_f1", fpr_metric}
    else:
        allowed_timing_metrics = set(args.timing_metrics)
    allowed_timing_methods = set(args.timing_methods) if args.timing_methods else None

    # If --window-sizes is specified but --timing-methods is not, build allowed methods from window sizes
    if allowed_timing_methods is None and run_window:
        window_sizes_parsed = sorted(set(int(w) for w in args.window_sizes if int(w) > 0))
        allowed_timing_methods = set()
        for w in window_sizes_parsed:
            allowed_timing_methods.add(f"window_pp_w{w}")
        if run_cpd:
            allowed_timing_methods.add("cpd_online")
        print(f"[INFO] Restricting visualization to window sizes: {window_sizes_parsed} (and CPD if requested)")

    if args.timing_best_window:
        try:
            suffix_df = pd.read_csv(args.out_csv)
            window_rows = suffix_df[suffix_df["method"].str.contains("window_pp", na=False)]
            if not window_rows.empty:
                idx = window_rows["F1_suffix"].idxmax()
                best_method = str(window_rows.loc[idx, "method"])
                if allowed_timing_methods is None:
                    allowed_timing_methods = set()
                allowed_timing_methods.add(best_method)
                # Always keep CPD alongside best window when requested.
                allowed_timing_methods.add("cpd_online")
                print(f"[INFO] timing_best_window enabled: selected {best_method} (plus cpd_online)")
            else:
                print("[WARN] timing_best_window enabled but no window_pp method found in suffix summary; skipping.")
        except Exception as exc:
            print(f"[WARN] timing_best_window failed to read {args.out_csv}: {exc}")

    if args.detection_csv_load:
        if not os.path.exists(args.detection_csv_load):
            print(f"[ERROR] detection-csv-load not found: {args.detection_csv_load}")
            return
        df_cache = pd.read_csv(args.detection_csv_load)
        detection_bar_data = df_to_detection_counts(df_cache)
        detection_bar_data = filter_detection_methods(detection_bar_data, allowed_timing_methods)
        cpd_label_threshold = None
        thr_path = os.path.join(os.path.dirname(args.detection_csv_load), "detection_thresholds.csv")
        if not os.path.exists(thr_path):
            alt_thr_path = os.path.join(args.roc_dir, "detection_thresholds.csv")
            if os.path.exists(alt_thr_path):
                thr_path = alt_thr_path
        if os.path.exists(thr_path):
            try:
                thr_df = pd.read_csv(thr_path)
                row = thr_df[
                    (thr_df["method"] == "cpd_online")
                    & (thr_df["metric"] == "overall_f1")
                ]
                if not row.empty:
                    cpd_label_threshold = float(row.iloc[0]["threshold"])
            except Exception as exc:
                print(f"[WARN] Failed to read CPD threshold from {thr_path}: {exc}")
        # If the requested FPR metric is missing from the cache, warn early.
        has_fpr_metric = any(
            fpr_metric in entry.get("overall", {}) or any(fpr_metric in seg for seg in entry.get("segments", {}).values())
            for entry in detection_bar_data.values()
        )
        if not has_fpr_metric:
            print(f"[WARN] Requested metric '{fpr_metric}' not found in cached detection_counts. "
                  f"Rerun without --detection-csv-load (or regenerate the cache) to compute it.")
        render_detection_plots(
            detection_bar_data,
            args.roc_dir,
            allowed_metrics=allowed_timing_metrics,
            allowed_methods=allowed_timing_methods,
            fpr_metric=fpr_metric,
            cpd_threshold=cpd_label_threshold,
        )
        print(f"[OK] Rendered detection plots from cache {args.detection_csv_load}")
        return

    os.makedirs(args.roc_dir, exist_ok=True)

    df = load_and_merge(args.pp_csv, args.cpd_csv)
    if args.debug_merged_csv:
        os.makedirs(os.path.dirname(args.debug_merged_csv) or ".", exist_ok=True)
        df.to_csv(args.debug_merged_csv, index=False)
        log(f"Wrote merged dataframe to {args.debug_merged_csv}")

    is_adv = df["is_adversarial_combined"].to_numpy(dtype=int)
    suffix_start = df["suffix_start_postprefix_combined"].to_numpy(dtype=float)
    suffix_len = df["suffix_len_tokens_combined"].to_numpy(dtype=float)
    algo_arr = df["algorithm_combined"].fillna("").astype(str).str.strip().str.lower().to_numpy()

    methods_results: List[SuffixMetrics] = []
    methods_meta: List[Dict[str, object]] = []

    all_suffix_stats_rows: List[Dict[str, object]] = []
    detection_bar_data: Dict[str, Dict[str, Dict[str, Tuple[float, float, float, float]]]] = {}
    detection_threshold_rows: List[Dict[str, object]] = []
    detection_summary_rows: List[Dict[str, object]] = []

    window_sizes = sorted(set(int(w) for w in args.window_sizes if int(w) > 0))
    cpd_trace_col = None
    for c in ["online_W_plus_trace", "online_W_plus_trace_cpd"]:
        if c in df.columns:
            cpd_trace_col = c
            log(f"Selected CPD trace column: {c}")
            break

    total_steps = 1
    if run_pp and "global_mean_nll" in df.columns:
        total_steps += 1
    if run_cpd and cpd_trace_col is not None:
        total_steps += 1
    if run_window:
        total_steps += len(window_sizes)
    progress_step = 1
    update_progress(progress_step, total_steps, "Merged inputs")

    # --------------------- PP: detection-only (NO suffix-aware evaluation) ---------------------
    if run_pp and "global_mean_nll" in df.columns:
        pp_scores = df["global_mean_nll"].astype(float).to_numpy()
        method_name = "pp"
        try:
            thr_global, _ = find_best_global_f1_threshold(pp_scores, is_adv)
            preds = (pp_scores >= thr_global).astype(int)
            prf = compute_prf(is_adv, preds)

            mask_finite = np.isfinite(pp_scores)
            if mask_finite.sum() >= 2 and len(np.unique(is_adv[mask_finite])) >= 2:
                auroc_det = compute_auroc(is_adv[mask_finite].astype(int), pp_scores[mask_finite])
            else:
                auroc_det = float("nan")

            detection_summary_rows.append(
                {
                    "method": method_name,
                    "kind": "pp",
                    "threshold": thr_global,
                    "precision_det": prf["precision"],
                    "recall_det": prf["recall"],
                    "f1_det": prf["f1"],
                    "auroc_adv_global": auroc_det,
                }
            )
            detection_threshold_rows.append(
                {"method": method_name, "metric": "overall_f1", "threshold": thr_global}
            )

            # Threshold at ~10% FPR (for reporting, no locality)
            try:
                thr_fpr, _ = find_threshold_at_fpr(pp_scores, is_adv, target_fpr=args.fpr_target)
                detection_threshold_rows.append(
                    {"method": method_name, "metric": fpr_metric, "threshold": thr_fpr}
                )
            except RuntimeError:
                print("[WARN] Skipping PP FPR10 threshold (insufficient data)")
        except RuntimeError as exc:
            print(f"[WARN] Skipping PP detection-only metrics: {exc}")

        progress_step += 1
        update_progress(progress_step, total_steps, "Processed PP (detection-only)")

    # --------------------- CPD-online: suffix-aware + detection-only -------------------------
    if run_cpd:
        if cpd_trace_col is None:
            print("[WARN] Skipping CPD-online: no W_plus trace column found.")
        else:
            cpd_traces = parse_trace_series(df[cpd_trace_col].fillna("[]"))
            cpd_scores = compute_trace_max(cpd_traces)
            detection_counts: Dict[str, Tuple[float, float, float, float]] = {}
            alarm_pairs: Dict[str, Tuple[np.ndarray, Optional[np.ndarray], List[List[Tuple[float, float]]]]] = {}

            def record_cpd_counts(
                label: str,
                start_arr: np.ndarray,
                end_arr: Optional[np.ndarray],
                intervals: List[List[Tuple[float, float]]],
            ) -> None:
                detection_counts[label] = compute_detection_split(
                    start_arr,
                    end_arr,
                    suffix_start,
                    suffix_len,
                    is_adv,
                    all_intervals=intervals,
                )
                alarm_pairs[label] = (start_arr, end_arr, intervals)

            try:
                (
                    cpd_best,
                    cpd_thresholds,
                    cpd_f1s,
                    cpd_delays_best,
                    cpd_early_best,
                    cpd_alarm_start_best,
                    cpd_alarm_end_best,
                ) = sweep_thresholds_suffix_f1(
                    cpd_traces,
                    suffix_start,
                    suffix_len,
                    is_adv,
                    step=1.0,
                    score_for_roc=cpd_scores,
                    progress_label="CPD threshold sweep",
                )
                if cpd_alarm_end_best is None:
                    cpd_alarm_end_best = cpd_alarm_start_best.copy()
                cpd_intervals_best = compute_all_alarm_intervals(
                    cpd_traces, cpd_best.threshold, step=1.0, interval_len=1.0
                )
                method_name = "cpd_online"
                methods_results.append(cpd_best)
                best_cpd_threshold = cpd_best.threshold
                methods_meta.append(
                    {
                        "method": method_name,
                        "kind": "cpd",
                    }
                )
                if "suffix_f1" in allowed_timing_metrics:
                    detection_threshold_rows.append(
                        {"method": method_name, "metric": "suffix_f1", "threshold": cpd_best.threshold}
                    )

                # Plots + suffix stats + timing counts
                plot_f1_vs_threshold(cpd_thresholds, cpd_f1s, cpd_best, method_name, args.roc_dir)
                plot_roc_segmented(df, cpd_scores, method_name, args.roc_dir)
                plot_delay_hist(cpd_delays_best, method_name, args.roc_dir, kind="delay")
                plot_delay_hist(cpd_early_best, method_name, args.roc_dir, kind="early")
                per_attack_breakdown(
                    df,
                    cpd_alarm_start_best,
                    cpd_alarm_end_best,
                    suffix_start,
                    suffix_len,
                    is_adv,
                    method_name,
                    cpd_best.threshold,
                    args.roc_dir,
                    cpd_scores,
                )
                rows = compute_suffix_stats(
                    df,
                    cpd_alarm_start_best,
                    cpd_alarm_end_best,
                    suffix_start,
                    suffix_len,
                    is_adv,
                    method_name,
                    args.roc_dir,
                    all_intervals=cpd_intervals_best,
                )
                all_suffix_stats_rows.extend(rows)
                record_cpd_counts("suffix_f1", cpd_alarm_start_best, cpd_alarm_end_best, cpd_intervals_best)

                # Global detection-only metrics (same scores)
                try:
                    thr_global, _ = find_best_global_f1_threshold(cpd_scores, is_adv)
                    alarm_start_global, alarm_end_global = compute_alarm_positions_from_traces(
                        cpd_traces, thr_global, step=1.0
                    )
                    if alarm_end_global is None:
                        alarm_end_global = alarm_start_global.copy()
                    cpd_intervals_global = compute_all_alarm_intervals(
                        cpd_traces, thr_global, step=1.0, interval_len=1.0
                    )
                    record_cpd_counts("overall_f1", alarm_start_global, alarm_end_global, cpd_intervals_global)
                    if "overall_f1" in allowed_timing_metrics:
                        detection_threshold_rows.append(
                            {"method": method_name, "metric": "overall_f1", "threshold": thr_global}
                        )
                    preds = (cpd_scores >= thr_global).astype(int)
                    prf = compute_prf(is_adv, preds)
                    detection_summary_rows.append(
                        {
                            "method": method_name,
                            "kind": "cpd",
                            "threshold": thr_global,
                            "precision_det": prf["precision"],
                            "recall_det": prf["recall"],
                            "f1_det": prf["f1"],
                            "auroc_adv_global": cpd_best.auroc_adv,
                        }
                    )
                except RuntimeError:
                    print("[WARN] Skipping CPD overall F1 threshold (insufficient data)")

                try:
                    thr_fpr, _ = find_threshold_at_fpr(cpd_scores, is_adv, target_fpr=args.fpr_target)
                    alarm_start_fpr, alarm_end_fpr = compute_alarm_positions_from_traces(
                        cpd_traces, thr_fpr, step=1.0
                    )
                    if alarm_end_fpr is None:
                        alarm_end_fpr = alarm_start_fpr.copy()
                    cpd_intervals_fpr = compute_all_alarm_intervals(
                        cpd_traces, thr_fpr, step=1.0, interval_len=1.0
                    )
                    record_cpd_counts(fpr_metric, alarm_start_fpr, alarm_end_fpr, cpd_intervals_fpr)
                    if fpr_metric in allowed_timing_metrics:
                        detection_threshold_rows.append(
                            {"method": method_name, "metric": fpr_metric, "threshold": thr_fpr}
                        )
                except RuntimeError:
                    print("[WARN] Skipping CPD FPR10 threshold (insufficient data)")

                store_detection_counts(
                    method_name,
                    detection_counts,
                    alarm_pairs,
                    algo_arr,
                    suffix_start,
                    suffix_len,
                    is_adv,
                    detection_bar_data,
                    allowed_metrics=allowed_timing_metrics,
                )
            except RuntimeError as exc:
                print(f"[WARN] Skipping CPD-online: {exc}")
        progress_step += 1
        update_progress(progress_step, total_steps, "Processed CPD-online")

    # --------------------- WINDOW-PP: suffix-aware + detection-only --------------------------
    if run_window:
        for w in window_sizes:
            trace_col_candidates = [f"window_mean_nll_w{w}_all", f"window_mean_nll_w{w}_all_pp"]
            trace_col = None
            for c in trace_col_candidates:
                if c in df.columns:
                    trace_col = c
                    log(f"Selected window trace column for w={w}: {c}")
                    break
            if trace_col is None:
                print(f"[WARN] Skipping window-PP w={w}: no trace column found.")
                progress_step += 1
                update_progress(progress_step, total_steps, f"Skipped window-PP w={w}")
                continue

            traces_w = parse_trace_series(df[trace_col].fillna("[]"))
            scores_w = compute_trace_max(traces_w)

            try:
                (
                    best_w,
                    thresholds_w,
                    f1s_w,
                    delays_w_best,
                    early_w_best,
                    alarm_start_w_best,
                    alarm_end_w_best,
                ) = sweep_thresholds_suffix_f1(
                    traces_w,
                    suffix_start,
                    suffix_len,
                    is_adv,
                    step=float(w),
                    score_for_roc=scores_w,
                    interval_len=float(w),
                    progress_label=f"Window-PP w={w} sweep",
                )
            except RuntimeError as exc:
                print(f"[WARN] Skipping window-PP w={w}: {exc}")
                progress_step += 1
                update_progress(progress_step, total_steps, f"Skipped window-PP w={w}")
                continue

            method_name = f"window_pp_w{w}"
            methods_results.append(best_w)
            methods_meta.append(
                {
                    "method": method_name,
                    "kind": "window_pp",
                    "window": w,
                }
            )

            # Suffix-aware plots and stats
            plot_f1_vs_threshold(thresholds_w, f1s_w, best_w, method_name, args.roc_dir)
            plot_roc_segmented(df, scores_w, method_name, args.roc_dir)
            plot_delay_hist(delays_w_best, method_name, args.roc_dir, kind="delay")
            plot_delay_hist(early_w_best, method_name, args.roc_dir, kind="early")
            per_attack_breakdown(
                df,
                alarm_start_w_best,
                alarm_end_w_best,
                suffix_start,
                suffix_len,
                is_adv,
                method_name,
                best_w.threshold,
                args.roc_dir,
                scores_w,
            )

            intervals_w_best = compute_all_alarm_intervals(
                traces_w, best_w.threshold, step=float(w), interval_len=float(w)
            )
            rows = compute_suffix_stats(
                df,
                alarm_start_w_best,
                alarm_end_w_best,
                suffix_start,
                suffix_len,
                is_adv,
                method_name,
                args.roc_dir,
                all_intervals=intervals_w_best,
            )
            all_suffix_stats_rows.extend(rows)

            detection_counts: Dict[str, Tuple[float, float, float, float]] = {}
            alarm_pairs: Dict[str, Tuple[np.ndarray, Optional[np.ndarray], List[List[Tuple[float, float]]]]] = {}

            def record_window_counts(
                label: str,
                start_arr: np.ndarray,
                end_arr: Optional[np.ndarray],
                intervals: List[List[Tuple[float, float]]],
            ) -> None:
                detection_counts[label] = compute_detection_split(
                    start_arr,
                    end_arr,
                    suffix_start,
                    suffix_len,
                    is_adv,
                    all_intervals=intervals,
                )
                alarm_pairs[label] = (start_arr, end_arr, intervals)

            record_window_counts("suffix_f1", alarm_start_w_best, alarm_end_w_best, intervals_w_best)
            if "suffix_f1" in allowed_timing_metrics:
                detection_threshold_rows.append(
                    {"method": method_name, "metric": "suffix_f1", "threshold": best_w.threshold}
                )

            # Detection-only global metrics for this window
            try:
                thr_global, _ = find_best_global_f1_threshold(scores_w, is_adv)
                alarm_start_global, alarm_end_global = compute_alarm_positions_from_traces(
                    traces_w, thr_global, step=float(w), interval_len=float(w)
                )
                intervals_w_global = compute_all_alarm_intervals(
                    traces_w, thr_global, step=float(w), interval_len=float(w)
                )
                record_window_counts("overall_f1", alarm_start_global, alarm_end_global, intervals_w_global)
                if "overall_f1" in allowed_timing_metrics:
                    detection_threshold_rows.append(
                        {"method": method_name, "metric": "overall_f1", "threshold": thr_global}
                    )
                preds = (scores_w >= thr_global).astype(int)
                prf = compute_prf(is_adv, preds)
                detection_summary_rows.append(
                    {
                        "method": method_name,
                        "kind": "window_pp",
                        "window_size": w,
                        "threshold": thr_global,
                        "precision_det": prf["precision"],
                        "recall_det": prf["recall"],
                        "f1_det": prf["f1"],
                        "auroc_adv_global": best_w.auroc_adv,
                    }
                )
            except RuntimeError:
                print(f"[WARN] Skipping window-PP w={w} overall F1 threshold (insufficient data)")

            try:
                thr_fpr, _ = find_threshold_at_fpr(scores_w, is_adv, target_fpr=args.fpr_target)
                alarm_start_fpr, alarm_end_fpr = compute_alarm_positions_from_traces(
                    traces_w, thr_fpr, step=float(w), interval_len=float(w)
                )
                intervals_w_fpr = compute_all_alarm_intervals(
                    traces_w, thr_fpr, step=float(w), interval_len=float(w)
                )
                record_window_counts(fpr_metric, alarm_start_fpr, alarm_end_fpr, intervals_w_fpr)
                if fpr_metric in allowed_timing_metrics:
                    detection_threshold_rows.append(
                        {"method": method_name, "metric": fpr_metric, "threshold": thr_fpr}
                    )
            except RuntimeError:
                print(f"[WARN] Skipping window-PP w={w} FPR10 threshold (insufficient data)")

            store_detection_counts(
                method_name,
                detection_counts,
                alarm_pairs,
                algo_arr,
                suffix_start,
                suffix_len,
                is_adv,
                detection_bar_data,
                allowed_metrics=allowed_timing_metrics,
            )

        progress_step += 1
        update_progress(progress_step, total_steps, "Processed window-PPs")

    # --------------------- Detection counts + thresholds ---------------------
    det_save_path = args.detection_csv_save or os.path.join(args.roc_dir, "detection_counts.csv")
    det_df = detection_counts_to_df(detection_bar_data)
    if not det_df.empty:
        if args.merge_output and os.path.exists(det_save_path):
            existing_det = pd.read_csv(det_save_path)
            det_df = merge_on_keys(existing_det, det_df, ["method", "segment", "metric"])
        det_df.to_csv(det_save_path, index=False)
        print(f"[OK] Wrote detection counts cache to {det_save_path}")
    else:
        print("[WARN] No detection counts to write (no suffix-aware methods?).")

    thr_path = os.path.join(args.roc_dir, "detection_thresholds.csv")
    thr_df = pd.DataFrame(detection_threshold_rows)
    if not thr_df.empty:
        if args.merge_output and os.path.exists(thr_path):
            existing_thr = pd.read_csv(thr_path)
            thr_df = merge_on_keys(existing_thr, thr_df, ["method", "metric"])
        thr_df.to_csv(thr_path, index=False)
        print(f"[OK] Wrote detection thresholds to {thr_path}")
    else:
        print("[WARN] No detection thresholds to write.")

    # --------------------- Suffix-aware summary (CPD + window-PP only, NO PP) ----------------
    out_dir = os.path.dirname(args.out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if methods_results:
        summary_rows: List[Dict[str, object]] = []
        for m, meta in zip(methods_results, methods_meta):
            row = {
                "method": meta["method"],
                "kind": meta["kind"],
                "best_threshold": m.threshold,
                "F1_suffix": m.f1_suffix,
                "precision_suffix": m.precision_suffix,
                "recall_suffix": m.recall_suffix,
                "TP_suffix": m.tp_suffix,
                "FP_benign": m.fp_benign,
                "FP_early_adv": m.fp_early_adv,
                "FN_suffix": m.fn_suffix,
                "median_delay": m.median_delay,
                "mean_delay": m.mean_delay,
                "n_correct": m.tp_suffix,
                "n_benign": m.n_benign,
                "n_adv_with_suffix": m.n_adv_suffix,
                "n_ignored_adv_no_suffix": m.n_ignored_adv_no_suffix,
                "auroc_adv_global": m.auroc_adv,
            }
            if "window" in meta:
                row["window_size"] = meta["window"]
            summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows)
        if args.merge_output and os.path.exists(args.out_csv):
            existing_sum = pd.read_csv(args.out_csv)
            summary_df = merge_on_keys(existing_sum, summary_df, ["method"])
        summary_df.to_csv(args.out_csv, index=False)
        print(f"[OK] Wrote suffix evaluation summary to {args.out_csv}")
    else:
        print("[WARN] No suffix-aware methods evaluated; skipping suffix summary CSV.")

    # --------------------- Detection-only summary (CPD + window + PP) ------------------------
    if detection_summary_rows:
        det_summary_path = os.path.join(out_dir, "detection_eval_summary.csv") if out_dir else "detection_eval_summary.csv"
        det_summary_df = pd.DataFrame(detection_summary_rows)
        if args.merge_output and os.path.exists(det_summary_path):
            existing_det_summary = pd.read_csv(det_summary_path)
            det_summary_df = merge_on_keys(existing_det_summary, det_summary_df, ["method"])
        det_summary_df.to_csv(det_summary_path, index=False)
        print(f"[OK] Wrote detection-only summary to {det_summary_path}")
    else:
        print("[WARN] No detection-only metrics to write.")

    # --------------------- Combined suffix-stats table (CPD + window only) -------------------
    if all_suffix_stats_rows:
        suffix_stats_df = pd.DataFrame(all_suffix_stats_rows)
        suffix_stats_path = os.path.join(out_dir, "suffix_stats_all_methods.csv")
        if args.merge_output and os.path.exists(suffix_stats_path):
            existing_stats = pd.read_csv(suffix_stats_path)
            suffix_stats_df = merge_on_keys(existing_stats, suffix_stats_df, ["method", "algorithm"])
        suffix_stats_df.to_csv(suffix_stats_path, index=False)
        print(f"[OK] Wrote combined suffix-stats table to {suffix_stats_path}")
    else:
        print("[WARN] No suffix stats rows to write.")

    # --------------------- Render detection timing plots --------------------
    detection_bar_data = filter_detection_methods(detection_bar_data, allowed_timing_methods)
    label_cpd_threshold = args.cpd_h if args.cpd_h is not None else best_cpd_threshold
    render_detection_plots(
        detection_bar_data,
        args.roc_dir,
        allowed_metrics=allowed_timing_metrics,
        allowed_methods=allowed_timing_methods,
        fpr_metric=fpr_metric,
        cpd_threshold=label_cpd_threshold,
    )


if __name__ == "__main__":
    main()
