#!/usr/bin/env python3
"""Reproduction script for MiP-CRIM on SK Model (n=1000 spins, 100 trials).
Matches Table 1 of the paper:
  Local-Minima-Preserving Polynomial Relaxation of Ising Problems
"""
import time
import sys
import numpy as np
from iamp_sk_solver import sync_ratio
from mip_crim import MiP_CRIM


def make_sk_matrix(n, seed=0):
    """SK model: J_ij ~ N(0,1), J_ii = 0, J symmetric."""
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((n, n))
    J = (W + W.T) / 2
    np.fill_diagonal(J, 0)  # zero diagonal for SK model
    J = np.round(J, decimals=5)  # lsb ~ 1e-5
    return J


def run_mip_crim_single(J, seed=0):
    """Run MiP-CRIM on a given SK matrix with paper's tuned parameters."""
    n = J.shape[0]
    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal(n)

    # Best tuned parameters for the SK Model (from paper benchmark_SK.py)
    params = dict(
        T=10, K=200,
        alpha=0.000014996, beta=0.001, lambda_=0.0707,
        step=1.00, beta1=0.09, beta2=0.999, eps=1e-8,
        sigma_noise=1e-3
    )

    J_mat = J.copy()
    np.fill_diagonal(J_mat, 0)

    t0 = time.perf_counter()
    sigma = MiP_CRIM(J_mat, x0, rng=rng, **params)
    elapsed = time.perf_counter() - t0

    energy = -0.5 * float(sigma @ J @ sigma)
    sync = sync_ratio(sigma, J)
    return dict(energy=energy, sync=sync, time=elapsed)


def main():
    n = 1000
    n_trials = 100

    print("=" * 80)
    print(f"  MiP-CRIM Reproduction: SK Model, n={n} spins, {n_trials} trials")
    print("=" * 80)
    print(f"  Fixed parameters: alpha=1.4996e-5, beta=0.001, lambda=0.0707")
    print(f"  Adam: step=1.0, beta1=0.09, beta2=0.999")
    print(f"  K=200 epochs, T=10 inner steps, sigma_noise=1e-3")
    print("=" * 80)
    sys.stdout.flush()

    energies = []
    syncs = []
    times = []

    for trial in range(n_trials):
        seed = trial * 137 + 42
        J = make_sk_matrix(n, seed=seed)
        result = run_mip_crim_single(J, seed=seed)
        energies.append(result["energy"])
        syncs.append(result["sync"])
        times.append(result["time"])

        if (trial + 1) % 10 == 0 or trial == 0:
            print(f"  Trial {trial+1:3d}/{n_trials}: energy={result['energy']:12.2f}, "
                  f"sync={result['sync']:.4f}, time={result['time']:.4f}s")
            sys.stdout.flush()

    energies = np.array(energies)
    syncs = np.array(syncs)
    times = np.array(times)

    best_energy = np.min(energies)
    mean_energy = np.mean(energies)
    best_sync = np.max(syncs)
    mean_sync = np.mean(syncs)
    mean_time = np.mean(times)

    print()
    print("=" * 80)
    print("  REPRODUCTION RESULTS")
    print("=" * 80)
    print(f"  Best Energy:   {best_energy:12.2f}  (paper: -16689.49)")
    print(f"  Mean Energy:   {mean_energy:12.2f}  (paper: -16491.62)")
    print(f"  Best Sync:     {best_sync:12.4f}  (paper:   1.000)")
    print(f"  Mean Sync:     {mean_sync:12.4f}  (paper:   0.999)")
    print(f"  Mean Runtime:  {mean_time:12.4f}s  (paper:   0.21s)")
    print("=" * 80)

    # Check against CI bounds
    print()
    print("  Rubric CI Comparison:")
    print(f"  Best Energy:  [-16814.75, -16676.96] -> {'IN' if -16814.75 <= best_energy <= -16676.96 else 'OUT'}")
    print(f"  Mean Energy:  [-16593.16, -16481.47] -> {'IN' if -16593.16 <= mean_energy <= -16481.47 else 'OUT'}")
    print(f"  Best Sync:    [1.000, 1.000]          -> {'IN' if 1.000 <= best_sync <= 1.000 else 'OUT'}")
    print(f"  Mean Sync:    [0.995, 0.9994]         -> {'IN' if 0.995 <= mean_sync <= 0.9994 else 'OUT'}")
    print(f"  Runtime:      [0.01, 0.23]            -> {'IN' if 0.01 <= mean_time <= 0.23 else 'OUT'}")


if __name__ == "__main__":
    main()
