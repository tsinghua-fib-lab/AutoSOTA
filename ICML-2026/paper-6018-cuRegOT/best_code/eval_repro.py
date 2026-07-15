#!/usr/bin/env python3
"""
cuRegOT (paper 6018) Reproduction Evaluation Script
===================================================
Reproduces the paper's Marginal Error metric for Synthetic I (iid), 1600x1200,
eta=0.001, S=10, with cost normalized by maximum.

This script is designed to run inside the Docker container `autosota_repro_paper_6018`
from /repo with the curegot conda environment.

Usage:
    cd /repo
    conda run -n curegot python eval_repro.py

The script returns baseline metrics (median across 10 runs):
- Marginal Error (SPLR/cuRegOT)
- Marginal Error (BCD/Sinkhorn baseline)
- Runtime (SPLR)
- Runtime (BCD)
"""
import sys
import os

# Ensure we import the installed curegot, not the local source directory
cwd = os.getcwd()
sys.path = [p for p in sys.path if cwd not in p]

import numpy as np
import time

try:
    import curegot
except ImportError as e:
    print(f"ERROR: Cannot import curegot: {e}")
    print("Make sure the curegot conda environment is active and the package is installed.")
    sys.exit(1)

def generate_synthetic_iid(n, m, seed):
    """Synthetic I (iid): N(0,1) positions, absolute distance, normalized by max, uniform marginals."""
    rng = np.random.RandomState(seed)
    x = rng.randn(n).astype(np.float64)
    y = rng.randn(m).astype(np.float64)
    M = np.abs(x[:, np.newaxis] - y[np.newaxis, :])
    M = M / np.max(M)
    a = np.ones(n, dtype=np.float64) / n
    b = np.ones(m, dtype=np.float64) / m
    return M, a, b

def compute_marginal_error(plan, a, b):
    """Maximum absolute deviation from target marginals."""
    row_sums = np.sum(plan, axis=1)
    col_sums = np.sum(plan, axis=0)
    return max(np.max(np.abs(row_sums - a)), np.max(np.abs(col_sums - b)))

def main():
    # Paper parameters
    n, m = 1600, 1200          # Problem size
    eta = 0.001                # Regularization
    S = 10                     # Sparsity reuse interval
    n_runs = 10                # Number of independent runs
    max_iter = 5000            # Full iteration schedule (tol=0.0)
    seeds = list(range(n_runs))

    print("=" * 70)
    print("cuRegOT Reproduction Evaluation")
    print("=" * 70)
    print(f"Problem: Synthetic I (iid), {n}x{m}")
    print(f"eta={eta}, S={S}, tol=0.0, max_iter={max_iter}")
    print(f"Cost normalization: by_maximum")
    print(f"Runs: {n_runs}, measure: median")
    print()

    splr_errors = []
    splr_times = []
    bcd_errors = []
    bcd_times = []

    for run_idx, seed in enumerate(seeds):
        M, a, b = generate_synthetic_iid(n, m, seed)

        # SPLR (cuRegOT)
        t0 = time.perf_counter()
        r_splr = curegot.numpy.sinkhorn_splr(
            M, a, b, eta, tol=0.0, max_iter=max_iter, verbose=0,
            sparsity_pattern_cycle=5,
            density=0.10,
            candidate_sinkhorn_iter=10
        )
        t_splr = time.perf_counter() - t0
        err_splr = compute_marginal_error(r_splr["plan"], a, b)
        splr_errors.append(err_splr)
        splr_times.append(t_splr)

        # BCD (Sinkhorn baseline)
        t0 = time.perf_counter()
        r_bcd = curegot.numpy.sinkhorn_bcd(
            M, a, b, eta, tol=1e-8, max_iter=10000, verbose=0
        )
        t_bcd = time.perf_counter() - t0
        err_bcd = compute_marginal_error(r_bcd["plan"], a, b)
        bcd_errors.append(err_bcd)
        bcd_times.append(t_bcd)

        print(f"Run {run_idx+1:2d} seed={seed}: "
              f"SPLR err={err_splr:.6e} time={t_splr:.2f}s | "
              f"BCD err={err_bcd:.6e} time={t_bcd:.2f}s")

    splr_errors = np.array(splr_errors)
    splr_times = np.array(splr_times)
    bcd_errors = np.array(bcd_errors)
    bcd_times = np.array(bcd_times)

    print()
    print("=" * 70)
    print("BASELINE METRICS (10-run median)")
    print("=" * 70)
    print(f"Marginal_Error_SPLR: {np.median(splr_errors):.8e}")
    print(f"Marginal_Error_BCD:  {np.median(bcd_errors):.8e}")
    print(f"Runtime_SPLR:        {np.median(splr_times):.4f}s")
    print(f"Runtime_BCD:         {np.median(bcd_times):.4f}s")

    return 0

if __name__ == "__main__":
    sys.exit(main())
