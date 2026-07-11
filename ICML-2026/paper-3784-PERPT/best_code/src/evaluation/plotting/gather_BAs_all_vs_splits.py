#!/usr/bin/env python3
"""
gather_BAs_all_vs_splits.py
----------------------------
Produces a wide CSV comparing all gamba_onepass models across splits
(all / test / training) and tasks (random, upstream, random-noannot,
multiclass-roi, multiclass-roi100bp).

One row per (model, split). Columns are mean BA% ± SE% for each task,
plus per-category BA% columns for the binary tasks.

Usage:
    python gather_BAs_all_vs_splits.py -o /home/mica/gamba/data_processing/data/240-mammalian/final_representations/gamba_comparison.csv
"""

import os
import math
import argparse
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import confusion_matrix

# ── Config ─────────────────────────────────────────────────────────────────────

ONEPASS_ROOT = (
    "/home/mica/gamba/other-models/final_representations/gamba_onepass"
)

GLOBAL_MODELS = [
    "gamba_cons_only_ALLPOSstep_44000",
    "gamba_dual_ALLPOSstep_44000",
    "gamba_seq_only_ALLPOSstep_44000",
    # "gamba_cons_only_step0",
    # "gamba_dual_step0",
    # "gamba_seq_only_step0",
    "caduceus_cons_only_ALLPOSstep_44000",
    "caduceus_dual_ALLPOSstep_44000",
    "caduceus_seq_only_ALLPOSstep_44000",
    # "caduceus_cons_only_step0",
    # "caduceus_dual_step0",
    # "caduceus_seq_only_step0",
]

SPLITS = ["all", "test", "training"]

BINARY_TASKS = ["random", "upstream", "random-noannot"]

MULTICLASS_SCOPES = [
    ("multiclass_roi",       "roi"),
    ("multiclass_roi100bp",  "roi100bp"),
]

CATEGORIES = [
    "vista_enhancer",
    "UCNE",
    "repeats",
    "exons",
    "introns",
    "noncoding_regions",
    "coding_regions",
    "upstream_TSS",
    "UTR5",
    "UTR3",
    "promoters",
]

SCOPE = "roi"   # binary tasks always use roi

# ── Name normalisation (mirrors existing code) ─────────────────────────────────

def to_model_id(model_folder: str) -> str:
    """Convert folder name to the ID used inside file names."""
    s = model_folder
    s = s.replace("_ALLPOSstep_", "_step")
    s = s.replace("_step_random_init", "_step0")
    s = s.replace("-random-init", "_step0")
    s = s.replace("_random-init", "_step0")
    return s


# ── Path helpers ───────────────────────────────────────────────────────────────

def binary_npz_path(model_folder: str, split: str, category: str, task: str) -> str:
    mid = to_model_id(model_folder)
    fname = f"reps_{mid}_{split}_{category}_binary-{task}_{SCOPE}.npz"
    return os.path.join(
        ONEPASS_ROOT, mid, "tasks", "binary", task, category, fname
    )


def multiclass_npz_path(model_folder: str, split: str, scope: str) -> str:
    mid = to_model_id(model_folder)
    fname = f"reps_{mid}_{split}_multiclass_{scope}.npz"
    return os.path.join(
        ONEPASS_ROOT, mid, "tasks", "multiclass", fname
    )


# ── Core metric ────────────────────────────────────────────────────────────────

def compute_ba_se(X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """LOO 1-NN balanced accuracy (%) and SE (%) with cosine distance."""
    nn = NearestNeighbors(n_neighbors=2, metric="cosine")
    nn.fit(X)
    _, idx = nn.kneighbors(X)
    y_pred = y[idx[:, 1]]

    classes = np.unique(y)
    cm = confusion_matrix(y, y_pred, labels=classes)
    n_per = cm.sum(axis=1).astype(float)
    recalls = np.divide(
        np.diag(cm).astype(float),
        np.where(n_per == 0, 1.0, n_per),
    )
    K = len(classes)
    ba = float(np.nanmean(recalls))
    var = float(np.nansum(recalls * (1 - recalls) / np.where(n_per == 0, np.inf, n_per)) / K**2)
    se = math.sqrt(max(var, 0.0))
    return ba * 100.0, se * 100.0


def load_npz(path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if not os.path.exists(path):
        return None, None
    z = np.load(path, allow_pickle=True)
    if "embeddings" not in z or "labels" not in z:
        return None, None
    X = np.asarray(z["embeddings"])
    y = np.asarray(z["labels"]).astype(str)
    return (X, y) if X.shape[0] >= 3 else (None, None)


# ── Per-task evaluators ────────────────────────────────────────────────────────

def eval_binary_task(
    model_folder: str, split: str, task: str
) -> Tuple[Optional[float], Optional[float], Dict[str, Optional[float]]]:
    """
    Returns (mean_BA, mean_SE, {category: BA}) averaged across categories.
    SE is propagated from per-category SEs.
    """
    bas, ses, cat_bas = [], [], {}

    for cat in CATEGORIES:
        path = binary_npz_path(model_folder, split, cat, task)
        X, y = load_npz(path)
        cat_bas[cat] = None
        if X is None:
            continue

        neg_lbl = task          # labels in file: "feature" vs task name
        if not (np.any(y == "feature") and np.any(y == neg_lbl)):
            continue

        try:
            ba, se = compute_ba_se(X, y)
        except Exception:
            continue

        bas.append(ba)
        ses.append(se)
        cat_bas[cat] = round(ba, 2)

    if not bas:
        return None, None, cat_bas

    mean_ba = float(np.mean(bas))
    K = len(bas)
    var = sum((s / 100.0) ** 2 for s in ses) / (K ** 2)
    mean_se = math.sqrt(max(var, 0.0)) * 100.0
    return round(mean_ba, 2), round(mean_se, 2), cat_bas


def eval_multiclass_task(
    model_folder: str, split: str, scope: str
) -> Tuple[Optional[float], Optional[float]]:
    path = multiclass_npz_path(model_folder, split, scope)
    X, y = load_npz(path)
    if X is None:
        return None, None
    try:
        ba, se = compute_ba_se(X, y)
        return round(ba, 2), round(se, 2)
    except Exception:
        return None, None


# ── Main builder ───────────────────────────────────────────────────────────────

def build_wide_dataframe() -> pd.DataFrame:
    rows = []

    for model_folder in GLOBAL_MODELS:
        mid = to_model_id(model_folder)
        print(f"\n── {mid}")

        for split in SPLITS:
            row: Dict = {"model": mid, "split": split}
            any_data = False

            # Binary tasks
            for task in BINARY_TASKS:
                col = task.replace("-", "_")   # e.g. random-noannot → random_noannot
                ba, se, cat_bas = eval_binary_task(model_folder, split, task)

                row[f"{col}_mean_BA"]  = ba
                row[f"{col}_mean_SE"]  = se

                for cat in CATEGORIES:
                    row[f"{col}_{cat}_BA"] = cat_bas.get(cat)

                if ba is not None:
                    any_data = True
                    print(f"  [{split}] {task:20s}  BA={ba:.1f} ± {se:.1f}")

            # Multiclass tasks
            for task_name, scope in MULTICLASS_SCOPES:
                ba, se = eval_multiclass_task(model_folder, split, scope)
                row[f"{task_name}_BA"] = ba
                row[f"{task_name}_SE"] = se
                if ba is not None:
                    any_data = True
                    print(f"  [{split}] {task_name:20s}  BA={ba:.1f} ± {se:.1f}")

            if any_data:
                rows.append(row)
            else:
                print(f"  [{split}] no data found — skipping row")

    return pd.DataFrame(rows)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Wide gamba onepass model comparison CSV")
    ap.add_argument(
        "-o", "--output",
        default="/home/mica/gamba/data_processing/data/240-mammalian/final_representations/gamba_comparison.csv",
        help="output CSV path",
    )
    args = ap.parse_args()

    print("Building comparison table…")
    df = build_wide_dataframe()

    if df.empty:
        raise SystemExit("No data collected — check ONEPASS_ROOT and file paths.")

    # Sort: model name, then split order
    split_order = {s: i for i, s in enumerate(SPLITS)}
    df["_split_ord"] = df["split"].map(split_order)
    df = df.sort_values(["model", "_split_ord"]).drop(columns="_split_ord")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\n✓ Wrote {len(df)} rows × {len(df.columns)} columns → {out}")

    # Quick summary table to stdout
    summary_cols = (
        ["model", "split"]
        + [f"{t.replace('-','_')}_mean_BA" for t in BINARY_TASKS]
        + [f"{n}_BA" for n, _ in MULTICLASS_SCOPES]
    )
    present = [c for c in summary_cols if c in df.columns]
    print("\n── Summary ──")
    print(df[present].to_string(index=False))


if __name__ == "__main__":
    main()