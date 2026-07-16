#!/usr/bin/env python3
"""Canonical evaluation script for MiP-CRIM on SK Model (n=1000 spins, 100 trials).

Reproduces the SK spin glass benchmark from:
  "Local-Minima-Preserving Polynomial Relaxation of Ising Problems" (ICML 2026)
  Table 1 (1000-spin SK Model) and Table 2 (IAMP vs MiP-CRIM).

Usage:
  cd /repo && python3 eval_sk_model.py

Metrics produced:
  - Best Energy (lower is better)
  - Mean Energy (lower is better)
  - Best sync (higher is better, max 1.0)
  - Mean sync (higher is better, max 1.0)
  - Mean Runtime (seconds)
"""
import numpy as np
import time
import sys
from mip_crim import MiP_CRIM
from iamp_sk_solver import sync_ratio


def make_sk_matrix(n, seed=0):
    """SK model: J_ij ~ N(0,1), J_ii = 0, J symmetric."""
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((n, n))
    J = (W + W.T) / 2
    np.fill_diagonal(J, 0)
    J = np.round(J, decimals=5)  # lsb ~ 1e-5, consistent with paper
    return J


def run_single_trial(J, seed, params):
    """Run one MiP-CRIM trial."""
    n = J.shape[0]
    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal(n)

    J_copy = J.copy()

    t0 = time.perf_counter()
    sigma = MiP_CRIM(J_copy, x0, rng=rng, **params)
    elapsed = time.perf_counter() - t0

    energy = -0.5 * float(sigma @ J @ sigma)
    sync = sync_ratio(sigma, J)
    return energy, sync, elapsed


def main():
    n = 1000
    n_trials = 100

    # Parameters from the paper's benchmark_SK.py (Table 2 fixed params)
    # Tuned on a 100-spin instance, satisfying admissibility:
    #   3 * beta * lambda^2 < alpha < beta * lambda^2 + gamma_0
    # where gamma_0 = 1e-5 (lsb from rounding to 5 decimals)
    params = dict(
        T=10,      # inner Adam steps per epoch
        K=600,     # epochs (basin-hopping restarts)
        alpha=0.0000099,
        beta=0.0005,
        lambda_=0.07,
        step=1.00,       # Adam learning rate
        beta1=0.9,      # Adam beta1
        beta2=0.999,     # Adam beta2
        eps=1e-8,        # Adam epsilon
        sigma_noise=1e-3, sigma_noise_start=1e-2, sigma_noise_end=1e-4  # Gaussian perturbation scale (annealed)
    )

    print("=" * 70)
    print("  MiP-CRIM: SK Model Benchmark (n=1000 spins, 100 trials)")
    print("=" * 70)
    print(f"  Parameters: alpha={params['alpha']:.6e}, beta={params['beta']}, "
          f"lambda={params['lambda_']}")
    print(f"  Adam: step={params['step']}, beta1={params['beta1']}, "
          f"beta2={params['beta2']}")
    sys.stdout.flush()

    energies = []
    syncs = []
    times = []

    for trial in range(n_trials):
        seed = trial * 137 + 42
        J = make_sk_matrix(n, seed=seed)
        energy, sync, elapsed = run_single_trial(J, seed, params)
        energies.append(energy)
        syncs.append(sync)
        times.append(elapsed)

        if (trial + 1) % 20 == 0:
            print(f"  Trial {trial+1:3d}/{n_trials}: energy={energy:.2f}, "
                  f"sync={sync:.4f}, time={elapsed:.4f}s")
            sys.stdout.flush()

    energies = np.array(energies)
    syncs = np.array(syncs)
    times = np.array(times)

    print()
    print("=" * 70)
    print("  RESULTS")
    print("=" * 70)
    print(f"  Best Energy:   {np.min(energies):12.2f}")
    print(f"  Mean Energy:   {np.mean(energies):12.2f}")
    print(f"  Best Sync:     {np.max(syncs):12.4f}")
    print(f"  Mean Sync:     {np.mean(syncs):12.4f}")
    print(f"  Mean Runtime:  {np.mean(times):12.4f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
