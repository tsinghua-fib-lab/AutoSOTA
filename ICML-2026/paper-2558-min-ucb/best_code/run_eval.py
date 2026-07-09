#!/usr/bin/env python3
"""Reproduction script for Paper 2558 - APUB Optimization.
Evaluates APUB-M at N=120, M=5000, alpha=0.1 over 30 runs,
and compares with SAA-M baseline.
"""
import numpy as np
import pickle
import gurobipy as gp
import time
import sys
import json
import os

sys.path.insert(0, '/repo')
from apub import APUB
from saa import SAA

# --- Configuration (matching paper Table 1) ---
N_SAMPLE = 120
M_BOOTSTRAP = 5000
ALPHA = 0.1  # (1-alpha) = 0.9
N_RUNS = 30
N_ITEMS = 20
N_MACHINES = 8
SEED = 1234  # matching evaluate_saa_time.py

# Cost vector c (from config.yaml)
c = [-14, -9, -20, -15, -4, -40, -18, -11, -13, -16, -17, -8, -9, -24, -10, -7, -12, -3, -4, -5]
A = np.zeros((N_MACHINES, N_ITEMS))
b = np.zeros(N_MACHINES)

# --- Load pre-generated data ---
data_path = "/repo/120.pkl"
print(f"Loading data from {data_path}...")
with open(data_path, "rb") as f:
    data = pickle.load(f)
train_samples_list = data["train_samples"]
print(f"Loaded {len(train_samples_list)} training samples")

# --- Select 30 random samples (matching paper's 30 runs) ---
np.random.seed(SEED)
random_indices = np.random.randint(0, len(train_samples_list), size=N_RUNS)
print(f"Selected {N_RUNS} random indices: {random_indices}")

# --- Run APUB-M evaluation ---
print(f"\n{'='*60}")
print(f"Running APUB-M: N={N_SAMPLE}, M={M_BOOTSTRAP}, alpha={ALPHA}")
print(f"{'='*60}")

apub_times = []
apub_iterations = []

for run_idx, sample_idx in enumerate(random_indices):
    train_sample = train_samples_list[sample_idx]
    
    apub = APUB(A, b, c=c, n_items=N_ITEMS, n_machines=N_MACHINES, model=gp.Model())
    start = time.perf_counter()
    x_opt, eta, obj_val, num_cuts = apub.solve_two_stage_apub(
        train_sample, alpha=ALPHA, M_bootstrap=M_BOOTSTRAP
    )
    elapsed = time.perf_counter() - start
    
    apub_times.append(elapsed)
    apub_iterations.append(num_cuts)
    print(f"  Run {run_idx+1}/{N_RUNS} (sample {sample_idx}): "
          f"Time={elapsed:.2f}s, Iterations={num_cuts}, Objective={obj_val:.2f}")

apub_time_mean = np.mean(apub_times)
apub_time_std = np.std(apub_times)
apub_iter_mean = np.mean(apub_iterations)
apub_iter_std = np.std(apub_iterations)

print(f"\nAPUB-M Results:")
print(f"  Time:      {apub_time_mean:.2f} ± {apub_time_std:.2f} s")
print(f"  Iteration: {apub_iter_mean:.2f} ± {apub_iter_std:.2f}")

# --- Run SAA-M evaluation (baseline) ---
print(f"\n{'='*60}")
print(f"Running SAA-M: N={N_SAMPLE}")
print(f"{'='*60}")

saa_times = []
saa_iterations = []

for run_idx, sample_idx in enumerate(random_indices):
    train_sample = train_samples_list[sample_idx]
    
    saa = SAA(model=gp.Model(), c=c, n_items=N_ITEMS, n_machines=N_MACHINES)
    start = time.perf_counter()
    x_val, ub, it = saa.solve_nf(train_sample, max_iter=30, tol=1e-4)
    elapsed = time.perf_counter() - start
    
    saa_times.append(elapsed)
    saa_iterations.append(it)
    print(f"  Run {run_idx+1}/{N_RUNS} (sample {sample_idx}): "
          f"Time={elapsed:.2f}s, Iterations={it}, UB={ub:.2f}")

saa_time_mean = np.mean(saa_times)
saa_time_std = np.std(saa_times)
saa_iter_mean = np.mean(saa_iterations)
saa_iter_std = np.std(saa_iterations)

print(f"\nSAA-M Results:")
print(f"  Time:      {saa_time_mean:.2f} ± {saa_time_std:.2f} s")
print(f"  Iteration: {saa_iter_mean:.2f} ± {saa_iter_std:.2f}")

# --- Summary ---
print(f"\n{'='*60}")
print(f"SUMMARY (Paper Table 1, N=120, M=5000, (1-alpha)=0.9)")
print(f"{'='*60}")
print(f"{'Method':<15} {'Time(s)':<20} {'Iteration':<20}")
print(f"{'-'*55}")
print(f"{'SAA-M':<15} {saa_time_mean:.1f} ± {saa_time_std:.1f} s      {saa_iter_mean:.1f} ± {saa_iter_std:.1f}")
print(f"{'APUB-M':<15} {apub_time_mean:.1f} ± {apub_time_std:.1f} s      {apub_iter_mean:.1f} ± {apub_iter_std:.1f}")
print(f"\nPaper reports:")
print(f"  SAA-M:   Time=5.7±1.0s,  Iteration=8.0±1.4")
print(f"  APUB-M:  Time=7.2±1.2s,  Iteration=9.2±1.4")

# --- Save results ---
results = {
    "paper_id": 2558,
    "config": {
        "N": N_SAMPLE,
        "M": M_BOOTSTRAP,
        "alpha": ALPHA,
        "nominal_level": f"{1-ALPHA}",
        "n_items": N_ITEMS,
        "n_machines": N_MACHINES,
        "n_runs": N_RUNS,
        "seed": SEED,
        "random_indices": [int(x) for x in random_indices]
    },
    "apub_m": {
        "time_mean": float(apub_time_mean),
        "time_std": float(apub_time_std),
        "time_values": [float(x) for x in apub_times],
        "iteration_mean": float(apub_iter_mean),
        "iteration_std": float(apub_iter_std),
        "iteration_values": [int(x) for x in apub_iterations]
    },
    "saa_m": {
        "time_mean": float(saa_time_mean),
        "time_std": float(saa_time_std),
        "time_values": [float(x) for x in saa_times],
        "iteration_mean": float(saa_iter_mean),
        "iteration_std": float(saa_iter_std),
        "iteration_values": [int(x) for x in saa_iterations]
    }
}

results_path = "/repo/reproduction_results.json"
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {results_path}")
print("DONE!")
