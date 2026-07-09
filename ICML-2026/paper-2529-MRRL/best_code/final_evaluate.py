"""Evaluate all completed seeds and compute aggregate metrics."""
import subprocess
import sys
import os
import pandas as pd
import numpy as np
import json
import glob

THETA = 1.0  # true causal effect

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_MODE"] = "offline"
os.environ["SLURM_JOB_ID"] = "local"
os.environ["SLURM_NODELIST"] = "localhost"
os.environ["HOSTNAME"] = "localhost"

all_estimates = []

for seed in range(42, 62):
    stats_file = f"outputs/repro_poly3-ds{seed}/1/training_stats.json"
    if not os.path.exists(stats_file):
        print(f"Seed {seed}: SKIP (not completed)")
        continue
    
    print(f"\nSeed {seed}: evaluating...")
    
    # Clean up re-run checkpoints if any
    ckpt_dir = f"outputs/repro_poly3-ds{seed}/1/checkpoints"
    for f in glob.glob(f"{ckpt_dir}/*-v1.ckpt"):
        os.remove(f)
        print(f"  Removed re-run checkpoint: {f}")
    
    # Run evaluate.py
    result = subprocess.run([
        sys.executable, "evaluate.py",
        "--exp_id", "repro_poly3",
        "--data_seed", str(seed),
        "--ckpt_strategy", "best",
        "--metric_key", "val/tot_loss",
        "--selection_mode", "min",
        "--batch_size", "3000",
    ], capture_output=True, text=True, timeout=300)
    
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[-200:]}")
        continue
    
    # Read the results CSV
    result_files = glob.glob(f"results/repro_poly3-ds{seed}_bestsim*_insample_estimates.csv")
    if not result_files:
        print(f"  WARNING: No results file found")
        continue
    
    df = pd.read_csv(result_files[0])
    hw_rows = df[df["instrument"] == "hW"].copy()
    hw_rows["data_seed"] = seed
    all_estimates.append(hw_rows)
    
    for _, row in hw_rows.iterrows():
        bias = row["estimate"] - THETA
        print(f"  hW pop={int(row["pop_num"])}: estimate={row["estimate"]:.6f} bias={bias:.6f}")

if not all_estimates:
    print("No seeds evaluated!")
    sys.exit(1)

# Combine and compute stats
combined = pd.concat(all_estimates, ignore_index=True)

print("\n" + "="*60)
print("AGGREGATE RESULTS")
print("="*60)

for pop_num in [-1, 0, 1]:
    sub = combined[combined["pop_num"] == pop_num]
    if len(sub) == 0:
        continue
    biases = sub["estimate"].values - THETA
    mean_bias = np.mean(biases)
    sd_bias = np.std(biases, ddof=1)
    n = len(sub)
    label = "Combined" if pop_num == -1 else f"Pop {pop_num}"
    print(f"\n{label} (n={n} seeds):")
    print(f"  Mean Bias: {mean_bias:.6f}")
    print(f"  SD across Seeds: {sd_bias:.6f}")
    print(f"  Individual biases: {biases.round(6).tolist()}")

# Save results
combined.to_csv("results/final_aggregate_metrics.csv", index=False)
with open("results/final_summary.json", "w") as f:
    summary = {}
    for pop_num in [-1, 0, 1]:
        sub = combined[combined["pop_num"] == pop_num]
        if len(sub) == 0:
            continue
        biases = sub["estimate"].values - THETA
        label = "combined" if pop_num == -1 else f"pop{pop_num}"
        summary[f"{label}_n"] = int(len(sub))
        summary[f"{label}_mean_bias"] = float(np.mean(biases))
        summary[f"{label}_sd"] = float(np.std(biases, ddof=1))
    json.dump(summary, f, indent=2)

print(f"\nResults saved to results/final_aggregate_metrics.csv and results/final_summary.json")
