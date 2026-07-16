#!/usr/bin/env python3
"""Adaptive parameter tuning for MiP-CRIM on SK model, following paper Appendix E.
Tunes on a single SK instance, then benchmarks 100 trials with tuned params.
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
    np.fill_diagonal(J, 0)
    return J  # NOTE: no rounding for Table 1 (unquantized)


def adaptive_tune(J, rounds=5, tune_res=3, K_outer=200, T_inner=10):
    """Adaptive grid search following Appendix E of the paper."""
    n = J.shape[0]
    x0 = np.random.randn(n)

    # Initial search ranges (from demo notebook)
    gamma0 = 0.1
    beta0 = 10
    step0 = 1.0
    sigma_noise = 1e-3

    Steps = np.linspace(0.01, step0, tune_res)
    Gamma = np.linspace(0.001, gamma0, tune_res)
    Beta = np.linspace(0.001, beta0, tune_res)

    best_energy = np.inf
    best_params = None
    best_spin = None

    for round_num in range(rounds):
        found_better = False
        round_best_energy = best_energy

        for step in Steps:
            for gamma in Gamma:
                for beta in Beta:
                    if beta <= 0 or gamma <= 0:
                        continue

                    lambda0 = np.sqrt(gamma / (2 * beta))

                    for lambda_ in np.linspace(0.01, lambda0, tune_res):
                        if lambda_ <= 0:
                            continue

                        # alpha in admissible range: 3βλ² < α < βλ² + γ
                        alpha_lo = 3 * beta * lambda_**2
                        alpha_hi = beta * lambda_**2 + gamma
                        if alpha_lo >= alpha_hi:
                            continue

                        for alpha in np.linspace(alpha_lo, alpha_hi, tune_res):
                            if alpha <= 0:
                                continue

                            spin_vec = MiP_CRIM(
                                J.copy(), x0,
                                T=T_inner, K=K_outer,
                                alpha=alpha, beta=beta, lambda_=lambda_,
                                step=step, beta1=0.09, beta2=0.999, eps=1e-8,
                                sigma_noise=sigma_noise,
                                rng=None, return_all=False
                            )

                            energy = -0.5 * float(spin_vec @ J @ spin_vec)

                            if energy < best_energy:
                                best_energy = energy
                                best_spin = spin_vec.copy()
                                best_params = {
                                    "alpha": alpha, "beta": beta, "lambda_": lambda_,
                                    "step": step, "gamma": gamma
                                }
                                found_better = True

        if not found_better or best_params is None:
            print(f"  Round {round_num+1}: no improvement, keeping best params")
            continue

        print(f"  Round {round_num+1}/{rounds}: best energy={best_energy:.2f}, "
              f"alpha={best_params['alpha']:.6e}, beta={best_params['beta']:.4f}, "
              f"lambda={best_params['lambda_']:.4f}, step={best_params['step']:.4f}")

        # Refinement: zoom in around best
        bp = best_params
        i_alpha = bp["alpha"]

        # Refine beta
        beta_center = bp["beta"]
        if beta_center <= 2 * Beta[0]:
            Beta = np.linspace(max(1e-6, beta_center/4), beta_center * 2, tune_res)
        else:
            Beta = np.linspace(max(1e-6, beta_center/2), beta_center * 2, tune_res)

        # Refine gamma
        gamma_center = bp["gamma"]
        if gamma_center <= 2 * Gamma[0]:
            Gamma = np.linspace(max(1e-8, gamma_center/4), gamma_center * 2, tune_res)
        else:
            Gamma = np.linspace(max(1e-8, gamma_center/2), gamma_center * 2, tune_res)

        # Refine step
        step_center = bp["step"]
        Steps = np.linspace(max(0.001, step_center/2), min(2.0, step_center * 2), tune_res)

    return best_params, best_energy


def run_mip_crim_tuned(J, params, seed=0):
    """Run MiP-CRIM with tuned parameters."""
    n = J.shape[0]
    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal(n)

    t0 = time.perf_counter()
    sigma = MiP_CRIM(
        J.copy(), x0,
        T=10, K=200,
        alpha=params["alpha"], beta=params["beta"],
        lambda_=params["lambda_"],
        step=params["step"], beta1=0.09, beta2=0.999, eps=1e-8,
        sigma_noise=1e-3, rng=rng, return_all=False
    )
    elapsed = time.perf_counter() - t0

    energy = -0.5 * float(sigma @ J @ sigma)
    sync = sync_ratio(sigma, J)
    return dict(energy=energy, sync=sync, time=elapsed)


def main():
    n = 1000
    n_trials = 100

    print("=" * 80)
    print("  MiP-CRIM Tuned Reproduction: SK Model, n=1000 spins, 100 trials")
    print("=" * 80)

    # Step 1: Tune parameters on a single instance
    print("\n[Step 1] Tuning parameters on a single SK instance...")
    J_tune = make_sk_matrix(n, seed=9999)
    best_params, tune_energy = adaptive_tune(J_tune, rounds=5, tune_res=3)

    if best_params is None:
        print("ERROR: Tuning failed, using default parameters")
        best_params = dict(alpha=0.000014996, beta=0.001, lambda_=0.0707, step=1.00)
    else:
        print(f"\n  Tuned: alpha={best_params['alpha']:.6e}, beta={best_params['beta']:.6f}, "
              f"lambda={best_params['lambda_']:.6f}, step={best_params['step']:.6f}")

    print(f"  Tuning instance energy: {tune_energy:.2f}")
    sys.stdout.flush()

    # Step 2: Benchmark 100 trials with tuned parameters
    print(f"\n[Step 2] Running {n_trials} trials with tuned parameters...")
    sys.stdout.flush()

    energies, syncs, times = [], [], []

    for trial in range(n_trials):
        seed = trial * 137 + 42
        J = make_sk_matrix(n, seed=seed)
        result = run_mip_crim_tuned(J, best_params, seed=seed)
        energies.append(result["energy"])
        syncs.append(result["sync"])
        times.append(result["time"])

        if (trial + 1) % 10 == 0:
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
    print("  REPRODUCTION RESULTS (TUNED)")
    print("=" * 80)
    print(f"  Best Energy:   {best_energy:12.2f}  (paper: -16689.49)")
    print(f"  Mean Energy:   {mean_energy:12.2f}  (paper: -16491.62)")
    print(f"  Best Sync:     {best_sync:12.4f}  (paper:   1.000)")
    print(f"  Mean Sync:     {mean_sync:12.4f}  (paper:   0.999)")
    print(f"  Mean Runtime:  {mean_time:12.4f}s  (paper:   0.21s)")
    print("=" * 80)

    print()
    print("  Rubric CI Comparison:")
    print(f"  Best Energy:  [-16814.75, -16676.96] -> {'IN' if -16814.75 <= best_energy <= -16676.96 else 'OUT'}")
    print(f"  Mean Energy:  [-16593.16, -16481.47] -> {'IN' if -16593.16 <= mean_energy <= -16481.47 else 'OUT'}")
    print(f"  Best Sync:    [1.000, 1.000]          -> {'IN' if 1.000 <= best_sync <= 1.000 else 'OUT'}")
    print(f"  Mean Sync:    [0.995, 0.9994]         -> {'IN' if 0.995 <= mean_sync <= 0.9994 else 'OUT'}")
    print(f"  Runtime:      [0.01, 0.23]            -> {'IN' if 0.01 <= mean_time <= 0.23 else 'OUT'}")


if __name__ == "__main__":
    main()
