#!/usr/bin/env python3
"""
Evaluation script for paper-1029: Subgroup Discovery with the Cox Model
Reproduces AIDS-Karnof metrics (Table 3): EPE, C-Index, Size, Rej@10%
"""
import sys
import os
import json
import time
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from evaluation.run_experiment import get_job_list
from utils.metrics import rej_frac
from data.load import load_data

CONFIG_NAME = "aids_karnof"
CONFIG_DIR = os.path.join(os.path.dirname(__file__), "experiments", "configs")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", CONFIG_NAME)
RAW_DIR = os.path.join(RESULTS_DIR, "raw")
SIZE_THRESHOLD = 0.1
REJ_THRESHOLDS = [0.01, 0.05, 0.10]


def run_experiments():
    """Run all experiment jobs for the AIDS-Karnof config."""
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "logs"), exist_ok=True)

    # Import locally
    from evaluation.run_experiment import run_experiment as run_one

    job_list = get_job_list(CONFIG_NAME, config_dir=CONFIG_DIR)
    print(f"Running {len(job_list)} jobs for config '{CONFIG_NAME}'...")

    for task_id, job in enumerate(job_list):
        method = job["method"]
        dataset = job["dataset"]
        subgp_cols = job["subgp_cols"]
        adjust_cols = job["adjust_cols"]
        seed = job["seed"]
        dataset_hyper = job["dataset_hyper"]
        R_star = job["R_star"]
        kwargs = job["kwargs"]

        print(f"[Job {task_id}/{len(job_list)-1}] {dataset}, {method}, seed={seed}")
        try:
            run_one(method, dataset, seed, subgp_cols, adjust_cols,
                    CONFIG_NAME, task_id, dataset_hyper, R_star, **kwargs)
        except Exception as e:
            print(f"  FAILED: {e}")


def select_best_regions():
    """Select best region per (method, seed) by lowest train_epe with train_size >= 0.1."""
    job_list = get_job_list(CONFIG_NAME, config_dir=CONFIG_DIR)
    method_map = {i: job["method"] for i, job in enumerate(job_list)}

    all_dfs = []
    for f in sorted(os.listdir(RAW_DIR)):
        if f.endswith("_results.pkl"):
            job_idx = int(f.split("_")[0])
            method = method_map[job_idx]
            df = pd.read_pickle(os.path.join(RAW_DIR, f))
            df["method"] = method
            all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)

    selected_rows = []
    for (method, seed), group in combined.groupby(["method", "seed"]):
        filtered = group[group["train_size"] >= SIZE_THRESHOLD]
        if len(filtered) == 0:
            continue
        best_idx = filtered["train_epe"].idxmin()
        selected_rows.append(filtered.loc[best_idx].copy())

    selected = pd.DataFrame(selected_rows)
    processed_dir = os.path.join(RESULTS_DIR, "processed")
    os.makedirs(processed_dir, exist_ok=True)
    selected.to_pickle(os.path.join(processed_dir, "selected_best.pkl"))
    return selected


def compute_metrics():
    """Compute all metrics from selected best regions."""
    selected = select_best_regions()

    metrics = {}
    for method in sorted(selected["method"].unique()):
        method_df = selected[selected["method"] == method]
        for col in ["test_epe", "test_c_ind", "test_size", "train_epe", "train_size"]:
            vals = method_df[col].dropna()
            metrics[f"{method}_{col}"] = {
                "mean": float(vals.mean()),
                "sem": float(vals.sem()),
                "n": len(vals),
            }

    # Compute rejection fractions for ddgroup
    ddgroup_best = selected[selected["method"] == "ddgroup"]
    rej_results = []
    for _, row in ddgroup_best.iterrows():
        seed = row["seed"]
        R, beta = row["R"], row["beta"]
        _, _, _, X_adj_test, X_subgp_test, Y_test, _, _ = load_data(
            "aids", [2], [0], int(seed)
        )
        test_rej = rej_frac(X_adj_test, X_subgp_test, Y_test, R, beta,
                            REJ_THRESHOLDS, n_jobs=-1)
        rej_results.append({
            "test_rej_01": test_rej[0],
            "test_rej_05": test_rej[1],
            "test_rej_10": test_rej[2],
        })

    for key in ["test_rej_01", "test_rej_05", "test_rej_10"]:
        vals = [r[key] for r in rej_results]
        metrics[f"ddgroup_{key}"] = {
            "mean": float(np.mean(vals)),
            "sem": float(np.std(vals, ddof=1) / np.sqrt(len(vals))),
            "n": len(vals),
        }

    # Primary metric
    ddg_epe = metrics.get("ddgroup_test_epe", {}).get("mean", None)
    ddg_cind = metrics.get("ddgroup_test_c_ind", {}).get("mean", None)
    ddg_size = metrics.get("ddgroup_test_size", {}).get("mean", None)
    ddg_rej10 = metrics.get("ddgroup_test_rej_10", {}).get("mean", None)

    result = {
        "primary_metric": "test_epe",
        "metric_direction": "lower",
        "ddgroup_test_epe": ddg_epe,
        "ddgroup_test_c_index": ddg_cind,
        "ddgroup_test_size": ddg_size,
        "ddgroup_test_rej10": ddg_rej10,
        "all_metrics": metrics,
    }

    # Save
    output_path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    return result


def main():
    started = time.time()

    # Step 1: Run experiments
    run_experiments()

    # Step 2: Compute metrics
    result = compute_metrics()

    elapsed = time.time() - started
    print(f"\nEvaluation complete in {elapsed:.1f}s")
    print(f"DG EPE: {result['ddgroup_test_epe']}")
    print(f"DG C-Index: {result['ddgroup_test_c_index']}")
    print(f"DG Size: {result['ddgroup_test_size']}")
    print(f"DG Rej@10%: {result['ddgroup_test_rej10']}")


if __name__ == "__main__":
    main()
