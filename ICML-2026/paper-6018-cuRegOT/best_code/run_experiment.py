"""
Reproduction experiment for cuRegOT paper (6018)
Synthetic I (iid): 1600x1200, eta=0.001, cost_normalization=by_maximum
tol=0.0, S=10, 10 runs, median
"""
import numpy as np
import time
import sys
import os

try:
    import curegot
    sinkhorn_bcd = curegot.numpy.sinkhorn_bcd
    sinkhorn_splr = curegot.numpy.sinkhorn_splr
    print("Successfully imported curegot (CUDA) module")
except ImportError as e:
    print(f"Failed to import curegot: {e}")
    sys.exit(1)

def generate_synthetic_iid(n, m, seed):
    """
    Synthetic I (iid):
    - Source and target positions drawn from N(0,1) in 1D
    - Cost matrix M_ij = |x_i - y_j|, normalized by max
    - Uniform marginals a_i = 1/n, b_j = 1/m
    """
    rng = np.random.RandomState(seed)

    # Source positions from N(0,1)
    x = rng.randn(n).astype(np.float64)
    # Target positions from N(0,1)
    y = rng.randn(m).astype(np.float64)

    # Cost matrix: pairwise absolute distance
    M = np.abs(x[:, np.newaxis] - y[np.newaxis, :])

    # Normalize by maximum
    M = M / np.max(M)

    # Uniform marginals
    a = np.ones(n, dtype=np.float64) / n
    b = np.ones(m, dtype=np.float64) / m

    return M, a, b

def compute_marginal_error(plan, a, b):
    """Compute marginal error: max row/col deviation from target marginals"""
    row_sums = np.sum(plan, axis=1)
    col_sums = np.sum(plan, axis=0)

    # Marginal error: maximum absolute deviation
    row_err = np.max(np.abs(row_sums - a))
    col_err = np.max(np.abs(col_sums - b))

    # Combined marginal error (max of both)
    marginal_err = max(row_err, col_err)

    return marginal_err, row_err, col_err

def main():
    # Experiment parameters from rubric
    n, m = 1600, 1200
    eta = 0.001
    tol = 0.0  # Run to max_iter without early convergence check
    sparsity_pattern_cycle = 10  # S=10
    n_runs = 10
    max_iter = 200  # Full iteration schedule

    print(f"{'='*70}")
    print(f"cuRegOT Reproduction Experiment")
    print(f"{'='*70}")
    print(f"Problem: Synthetic I (iid), {n}x{m}")
    print(f"Regularization eta: {eta}")
    print(f"Cost normalization: by_maximum")
    print(f"Tolerance: {tol} (run to max_iter)")
    print(f"Sparsity reuse interval S: {sparsity_pattern_cycle}")
    print(f"Number of runs: {n_runs}")
    print(f"Max iterations: {max_iter}")
    print(f"{'='*70}")

    # Seeds: 0 through 9
    seeds = list(range(n_runs))

    # Results storage
    splr_times = []
    splr_marginal_errors = []
    splr_niters = []
    bcd_times = []
    bcd_marginal_errors = []
    bcd_niters = []

    for run_idx, seed in enumerate(seeds):
        print(f"\n--- Run {run_idx+1}/{n_runs} (seed={seed}) ---")

        # Generate data
        M, a, b = generate_synthetic_iid(n, m, seed)
        print(f"  Data generated: M shape={M.shape}, M max={np.max(M):.4f}")

        # Run SPLR (cuRegOT)
        print(f"  Running SPLR (cuRegOT)...")
        t0 = time.perf_counter()
        splr_result = sinkhorn_splr(
            M, a, b, eta,
            tol=tol, max_iter=max_iter, verbose=0,
            sparsity_pattern_cycle=sparsity_pattern_cycle
        )
        splr_time = time.perf_counter() - t0
        splr_plan = splr_result["plan"]
        splr_niter = splr_result["niter"]

        splr_marg_err, splr_row_err, splr_col_err = compute_marginal_error(splr_plan, a, b)
        splr_times.append(splr_time)
        splr_marginal_errors.append(splr_marg_err)
        splr_niters.append(splr_niter)

        print(f"  SPLR: time={splr_time:.4f}s, niter={splr_niter}, marg_err={splr_marg_err:.6e}")
        print(f"    row_err={splr_row_err:.6e}, col_err={splr_col_err:.6e}")

        # Run BCD (baseline Sinkhorn)
        print(f"  Running BCD (Sinkhorn)...")
        t0 = time.perf_counter()
        bcd_result = sinkhorn_bcd(
            M, a, b, eta,
            tol=tol, max_iter=max_iter, verbose=0
        )
        bcd_time = time.perf_counter() - t0
        bcd_plan = bcd_result["plan"]
        bcd_niter = bcd_result["niter"]

        bcd_marg_err, bcd_row_err, bcd_col_err = compute_marginal_error(bcd_plan, a, b)
        bcd_times.append(bcd_time)
        bcd_marginal_errors.append(bcd_marg_err)
        bcd_niters.append(bcd_niter)

        print(f"  BCD:  time={bcd_time:.4f}s, niter={bcd_niter}, marg_err={bcd_marg_err:.6e}")
        print(f"    row_err={bcd_row_err:.6e}, col_err={bcd_col_err:.6e}")

    # Compute median statistics
    print(f"\n{'='*70}")
    print(f"RESULTS (10-run median)")
    print(f"{'='*70}")

    splr_times = np.array(splr_times)
    splr_marginal_errors = np.array(splr_marginal_errors)
    bcd_times = np.array(bcd_times)
    bcd_marginal_errors = np.array(bcd_marginal_errors)

    print(f"\nSPLR (cuRegOT):")
    print(f"  Marginal Error - median: {np.median(splr_marginal_errors):.6e}")
    print(f"  Marginal Error - min:   {np.min(splr_marginal_errors):.6e}")
    print(f"  Marginal Error - max:   {np.max(splr_marginal_errors):.6e}")
    print(f"  Runtime - median: {np.median(splr_times):.4f}s")
    print(f"  Runtime - min:   {np.min(splr_times):.4f}s")
    print(f"  Runtime - max:   {np.max(splr_times):.4f}s")
    print(f"  Iterations - median: {np.median(splr_niters)}")

    print(f"\nBCD (Sinkhorn):")
    print(f"  Marginal Error - median: {np.median(bcd_marginal_errors):.6e}")
    print(f"  Marginal Error - min:   {np.min(bcd_marginal_errors):.6e}")
    print(f"  Marginal Error - max:   {np.max(bcd_marginal_errors):.6e}")
    print(f"  Runtime - median: {np.median(bcd_times):.4f}s")
    print(f"  Runtime - min:   {np.min(bcd_times):.4f}s")
    print(f"  Runtime - max:   {np.max(bcd_times):.4f}s")
    print(f"  Iterations - median: {np.median(bcd_niters)}")

    # Print all individual run results for transparency
    print(f"\n{'='*70}")
    print(f"DETAILED RUN RESULTS")
    print(f"{'='*70}")
    print(f"{'Run':>4s}  {'Seed':>4s}  {'SPLR_Err':>12s}  {'SPLR_Time':>10s}  {'SPLR_Iter':>10s}  {'BCD_Err':>12s}  {'BCD_Time':>10s}  {'BCD_Iter':>10s}")
    print("-" * 80)
    for i in range(n_runs):
        print(f"{i+1:4d}  {seeds[i]:4d}  {splr_marginal_errors[i]:12.6e}  {splr_times[i]:10.4f}  {splr_niters[i]:10d}  {bcd_marginal_errors[i]:12.6e}  {bcd_times[i]:10.4f}  {bcd_niters[i]:10d}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
