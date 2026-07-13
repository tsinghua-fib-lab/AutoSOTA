#!/usr/bin/env python3
"""
Perplexity-based detector with F1-thresholding + locality
=========================================================

This version replaces the paper's unusable threshold rule
    threshold = max(benign window score)
with the scientifically correct:
    threshold = ROC F1-optimal score

This matches the CPD detector's threshold style and provides:
- meaningful locality alarms
- fair comparison between detectors
- stable thresholds on multilingual datasets

Everything else remains baseline-defenses style:
- uses mean NLL (log-perplexity)
- non-overlapping windows
- max window-mean as detection score
- evaluates malicious_prompt + suffix (prefix removed)
"""

import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support
import matplotlib.pyplot as plt

# Ensure local utils are importable when launched as a script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------

def extract_token_nlls_all(token_stream: str) -> np.ndarray:
    """
    Extract per-token NLLs excluding prefix tokens, matching your
    original implementation and ensuring we evaluate:
        malicious_prompt + suffix
    """
    payload = json.loads(token_stream)
    toks = payload.get("tokens", [])
    nlls = []
    for t in toks:
        if int(t.get("is_prefix", 0)) == 1:
            continue
        n = t.get("nll")
        if n is None:
            continue
        nlls.append(float(n))
    return np.asarray(nlls, float)


def get_window_mean_nlls(nlls: np.ndarray, w: int):
    """Non-overlapping window means, baseline-defenses style."""
    if w <= 0:
        return []
    N = len(nlls)
    out = []
    for i in range(0, N, w):
        win = nlls[i:i+w]
        if len(win) > 0:
            out.append(float(np.mean(win)))
    return out


def compute_auc(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores)
    return auc(fpr, tpr), fpr, tpr


def find_best_f1_threshold(labels, scores):
    """
    Compute the threshold that maximizes F1 score.
    """
    fpr, tpr, thresholds = roc_curve(labels, scores)

    best_f1 = -1
    best_thr = thresholds[0]

    for thr in thresholds:
        preds = (scores >= thr).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary", zero_division=0
        )
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr

    return float(best_thr), float(best_f1)


def get_window_alarm_pos(nlls: np.ndarray, w: int, threshold: float):
    """
    Return first window whose mean NLL > threshold (baseline locality).
    """
    if w <= 0 or not np.isfinite(threshold):
        return None
    N = len(nlls)
    for i in range(0, N, w):
        win = nlls[i:i+w]
        if len(win) == 0:
            continue
        if float(np.mean(win)) > threshold:
            return i
    return None


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats-csv", required=True)
    parser.add_argument("--per-prompt-out", default=None)
    parser.add_argument("--roc-dir", default=None)
    parser.add_argument("--window-sizes", nargs="+", type=int, default=[1])
    args = parser.parse_args()
    run = None

    df = pd.read_csv(args.stats_csv)
    required = {"token_stream", "is_adversarial",
                "prefix_len_tokens", "suffix_start_postprefix"}
    if missing := (required - set(df.columns)):
        raise ValueError(f"Missing cols: {missing}")

    labels = df["is_adversarial"].astype(int).to_numpy()

    df_out = pd.DataFrame(index=df.index)
    df_out["row_index"] = df.index.astype(int)
    df_out["is_adversarial"] = labels
    df_out["algorithm"] = df.get("algorithm", "normal")
    # -----------------------------
    # Extract NLLs + suffix starts
    # -----------------------------
    all_nlls = []
    global_scores = []
    suffix_start_idx = []

    for idx, row in df.iterrows():
        nlls = extract_token_nlls_all(row["token_stream"])
        all_nlls.append(nlls)

        global_scores.append(float(np.mean(nlls)) if len(nlls) else np.nan)

        ps = row["suffix_start_postprefix"]
        if np.isnan(ps):
            suffix_start_idx.append(np.nan)
        else:
            suffix_start_idx.append(int(row["prefix_len_tokens"] + ps))

    df_out["global_mean_nll"] = global_scores
    df_out["suffix_start_token_idx"] = suffix_start_idx

    # -----------------------------
    # Window scores
    # -----------------------------
    window_sizes = sorted(set(int(w) for w in args.window_sizes if w > 0))
    window_cols = {}

    for w in window_sizes:
        col = f"window_mean_nll_w{w}"
        col_all = f"{col}_all"
        vals = []
        trace_vals = []
        for nlls in all_nlls:
            if len(nlls) == 0:
                vals.append(np.nan)
                trace_vals.append(json.dumps([]))
                continue
            means = get_window_mean_nlls(nlls, w)
            vals.append(float(max(means)) if means else np.nan)
            trace_vals.append(json.dumps(means))
        df_out[col] = vals
        df_out[col_all] = trace_vals
        window_cols[w] = col

    # -----------------------------
    # FIND F1-OPTIMAL THRESHOLDS
    # -----------------------------
    thresholds = {}

    # Global threshold
    mask = np.isfinite(df_out["global_mean_nll"].to_numpy())
    if mask.sum() > 10:
        thr, f1 = find_best_f1_threshold(labels[mask],
                                         df_out["global_mean_nll"].to_numpy()[mask])
        thresholds["global"] = thr
        print(f"[GLOBAL] F1-optimal threshold={thr:.4f} F1={f1:.4f}")

    # Per-window thresholds
    for w, col in window_cols.items():
        scores = df_out[col].to_numpy()
        mask = np.isfinite(scores)
        if mask.sum() > 10 and len(np.unique(labels[mask])) >= 2:
            thr, f1 = find_best_f1_threshold(labels[mask], scores[mask])
            thresholds[w] = thr
            print(f"[w={w}] F1-optimal threshold={thr:.4f} F1={f1:.4f}")
        else:
            thresholds[w] = np.nan

    # -----------------------------
    # LOCALITY using F1-thresholds
    # -----------------------------
    for w, col in window_cols.items():
        thr = thresholds[w]
        alarm_pos = []
        alarm_delay = []
        scores = df_out[col].to_numpy()

        for i, nlls in enumerate(all_nlls):
            if len(nlls) == 0 or not np.isfinite(thr):
                alarm_pos.append(np.nan)
                alarm_delay.append(np.nan)
                continue

            ap = get_window_alarm_pos(nlls, w, thr)
            if ap is None:
                alarm_pos.append(np.nan)
                alarm_delay.append(np.nan)
                continue

            alarm_pos.append(ap)
            ss = df_out["suffix_start_token_idx"].iloc[i]
            alarm_delay.append(np.nan if np.isnan(ss) else float(ap - ss))

        df_out[f"alarm_pos_w{w}"] = alarm_pos
        df_out[f"alarm_delay_w{w}"] = alarm_delay

    # -----------------------------
    # OUTPUT CSV
    # -----------------------------
    if args.per_prompt_out:
        os.makedirs(os.path.dirname(args.per_prompt_out), exist_ok=True)
        df_out.to_csv(args.per_prompt_out, index=False)
        print(f"[OK] wrote {args.per_prompt_out}")

    # -----------------------------
    # ROC CURVES
    # -----------------------------
    if args.roc_dir:
        os.makedirs(args.roc_dir, exist_ok=True)

        # Global ROC
        mask = np.isfinite(df_out["global_mean_nll"])
        if mask.sum() > 10:
            A, fpr, tpr = compute_auc(labels[mask],
                                      df_out["global_mean_nll"].to_numpy()[mask])
            plt.figure(figsize=(6, 6))
            plt.plot(fpr, tpr, label=f"AUC={A:.3f}")
            plt.plot([0, 1], [0, 1], "--", color="gray")
            plt.xlabel("FPR")
            plt.ylabel("TPR")
            plt.title("Global Mean NLL ROC")
            plt.legend()
            fp = os.path.join(args.roc_dir, "roc_global_mean_nll.png")
            plt.savefig(fp, dpi=150)
            plt.close()
            print(f"[OK] wrote {fp}")

        # Window ROCs
        for w, col in window_cols.items():
            scores = df_out[col].to_numpy()
            mask = np.isfinite(scores)
            if mask.sum() < 10 or len(np.unique(labels[mask])) < 2:
                continue

            A, fpr, tpr = compute_auc(labels[mask], scores[mask])
            plt.figure(figsize=(6, 6))
            plt.plot(fpr, tpr, label=f"AUC={A:.3f}")
            plt.plot([0, 1], [0, 1], "--", color="gray")
            plt.xlabel("FPR")
            plt.ylabel("TPR")
            plt.title(f"Window Mean NLL ROC (w={w})")
            plt.legend()
            fp = os.path.join(args.roc_dir,
                               f"roc_window_mean_nll_w{w}.png")
            plt.savefig(fp, dpi=150)
            plt.close()
            print(f"[OK] wrote {fp}")


if __name__ == "__main__":
    main()
