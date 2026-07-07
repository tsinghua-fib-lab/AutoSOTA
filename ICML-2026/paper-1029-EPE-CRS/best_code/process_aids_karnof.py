import sys
sys.path.insert(0, "/repo/src")
import pandas as pd
import numpy as np
import os
from evaluation.run_experiment import get_job_list

# Get job list to map index to method
job_list = get_job_list("aids_karnof", config_dir="/repo/experiments/configs")
method_map = {i: job["method"] for i, job in enumerate(job_list)}

# Load all results and add method column
results_dir = "/repo/results/aids_karnof/raw"
all_dfs = []
for f in sorted(os.listdir(results_dir)):
    if f.endswith("_results.pkl"):
        job_idx = int(f.split("_")[0])
        method = method_map[job_idx]
        df = pd.read_pickle(os.path.join(results_dir, f))
        df["method"] = method
        df["job_idx"] = job_idx
        all_dfs.append(df)

combined = pd.concat(all_dfs, ignore_index=True)
print("Total results:", len(combined))
print("Methods:", sorted(combined["method"].unique()))
print("Seeds:", sorted(combined["seed"].unique()))

# Select best region per (method, seed)
size_threshold = 0.1
selected_rows = []
for (method, seed), group in combined.groupby(["method", "seed"]):
    filtered = group[group["train_size"] >= size_threshold]
    if len(filtered) == 0:
        print("WARNING: No results with train_size >= %s for %s, seed %d" % (size_threshold, method, seed))
        continue
    best_idx = filtered["train_epe"].idxmin()
    best_row = filtered.loc[best_idx].copy()
    selected_rows.append(best_row)

selected = pd.DataFrame(selected_rows)
print("\nSelected %d best regions (one per method x seed)" % len(selected))

# Compute summary statistics
print("\n=== Summary Statistics (mean (sem) over 10 seeds) ===")
for method in sorted(selected["method"].unique()):
    method_df = selected[selected["method"] == method]
    n = len(method_df)
    print("\n%s (n=%d):" % (method, n))
    for col in ["test_epe", "test_c_ind", "test_size", "train_epe", "train_size"]:
        vals = method_df[col].dropna()
        if len(vals) > 0:
            print("  %s: %.4f (%.4f)" % (col, vals.mean(), vals.sem()))

# Save selected results
os.makedirs("/repo/results/aids_karnof/processed", exist_ok=True)
selected.to_pickle("/repo/results/aids_karnof/processed/selected_best.pkl")
print("\nSaved selected results to results/aids_karnof/processed/selected_best.pkl")
