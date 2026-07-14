#!/usr/bin/env python3
"""
Directional Power Simulation for SPH Two-Sample Test

Reproduces Table 2 results from Section 6.2:
- vMF Directional data: d=2, m=n=25
- Concentration alternative: kappa_X=5, kappa_Y=8.75 (ratio 1.75)
- Test: SPH-sqrt with multi-p kernel aggregation and Bonferroni correction
- alpha = 0.05

Multi-p aggregation (ALGO-01): evaluates SPH kernel at p=2,4,8 and combines
via Bonferroni-adjusted minimum p-value, inspired by MMDAgg (Schrab et al.).

Usage:
    python power_directional.py [--R N] [--B_perm N] [--seed N]
"""

import numpy as np
from scipy import stats
from scipy.stats import vonmises_fisher, norm
import argparse
import time
import os
from datetime import datetime

from optimized_estimators import SphericalTestConfig, OptimizedTestStatistic


def run_power_simulation(
    d: int = 2,
    m: int = 25,
    n: int = 25,
    p: int = 2,
    p_values: list = None,
    kappa_X: float = 5.0,
    kappa_Y: float = 8.75,
    alpha: float = 0.05,
    R: int = 500,
    B_perm: int = 500,
    master_seed: int = 5086,
    output_dir: str = "simulation_results",
    use_multi_p: bool = False,
):
    """
    Run power simulation for directional two-sample test.

    When use_multi_p=True, aggregates across p_values using Bonferroni-adjusted
    minimum p-value (MMDAgg-style aggregation).
    """
    os.makedirs(output_dir, exist_ok=True)

    # Mean direction (same for both groups to test concentration differences)
    mu = np.zeros(d + 1)
    mu[0] = 1.0  # North pole

    if use_multi_p and p_values is None:
        p_values = [2, 4, 8]

    if use_multi_p:
        K = len(p_values)
        # Create configs and calculators for each p
        configs = [SphericalTestConfig(p=p_val, d=d) for p_val in p_values]
        calculators = [OptimizedTestStatistic(cfg) for cfg in configs]

        print("=" * 70)
        print("DIRECTIONAL POWER SIMULATION: SPH-sqrt Multi-p (Simes)")
        print("=" * 70)
        print("  Sphere dimension: d = %d (S^%d in R^%d)" % (d, d, d+1))
        print("  Truncation parameters: p = %s" % str(p_values))
        print("  Simes correction (less conservative than Bonferroni)")
        print("  Sample sizes: m = %d, n = %d" % (m, n))
        print("  vMF concentration: kappa_X = %.2f, kappa_Y = %.2f" % (kappa_X, kappa_Y))
        print("  kappa ratio: %.2f" % (kappa_Y/kappa_X))
        print("  Mean direction: matched (mu = [1, 0, ..., 0])")
        print("  Significance level: alpha = %.2f" % alpha)
        print("  Monte Carlo replications: R = %d" % R)
        print("  Permutation iterations: B_perm = %d" % B_perm)
        print("  Master seed: %d" % master_seed)
        print("=" * 70)

        # Results storage
        T_values = {p_val: np.zeros(R) for p_val in p_values}
        asymp_pvals = np.zeros(R)
        perm_pvals = np.zeros(R)

        start_time = time.time()

        for r in range(R):
            # Derive seeds from master seed
            seed_r = master_seed + r * 1000
            rng_data = np.random.default_rng(seed_r)
            rng_perm = np.random.default_rng(seed_r + 500)

            # Generate samples
            vmf_X = vonmises_fisher(mu=mu, kappa=kappa_X)
            X = vmf_X.rvs(size=m, random_state=rng_data)
            vmf_Y = vonmises_fisher(mu=mu, kappa=kappa_Y)
            Y = vmf_Y.rvs(size=n, random_state=rng_data)

            # Compute T_obs for each p
            T_obs_list = []
            for k_idx, calc in enumerate(calculators):
                T_k = calc.compute(X, Y, use_unbiased=True)
                T_values[p_values[k_idx]][r] = T_k
                T_obs_list.append(T_k)

            # Asymptotic combined p-value (Simes-adjusted)
            asymp_p_list = [norm.sf(T) for T in T_obs_list]
            sorted_p = np.sort(asymp_p_list)
            simes_values = [K / (i + 1) * sorted_p[i] for i in range(K)]
            asymp_p_combined = min(1.0, min(simes_values))
            asymp_pvals[r] = asymp_p_combined

            # Permutation: compute permutation statistics for each p
            Z_pooled = np.vstack([X, Y])
            # Track count of exceedances per p
            counts_greater = [0.0] * K

            for b in range(B_perm):
                perm = rng_perm.permutation(m + n)
                idx_X_perm = perm[:m]
                idx_Y_perm = perm[m:]
                X_perm = Z_pooled[idx_X_perm]
                Y_perm = Z_pooled[idx_Y_perm]

                for k_idx, calc in enumerate(calculators):
                    T_perm_k = calc.compute(X_perm, Y_perm, use_unbiased=True)
                    if T_perm_k >= T_obs_list[k_idx]:
                        counts_greater[k_idx] += 1.0

            # Permutation p-values per p, then Simes-combined
            perm_p_per_k = [(1.0 + c) / (B_perm + 1.0) for c in counts_greater]
            sorted_perm_p = np.sort(perm_p_per_k)
            simes_perm_values = [K / (i + 1) * sorted_perm_p[i] for i in range(K)]
            perm_p_combined = min(1.0, min(simes_perm_values))
            perm_pvals[r] = perm_p_combined

            # Progress
            if (r + 1) % 50 == 0 or r == 0:
                elapsed = time.time() - start_time
                power_perm = np.mean(perm_pvals[:r+1] <= alpha)
                power_asymp = np.mean(asymp_pvals[:r+1] <= alpha)
                print("  [%d/%d] elapsed=%.1fs | perm_power=%.4f | asymp_power=%.4f" %
                      (r+1, R, elapsed, power_perm, power_asymp))

        total_time = time.time() - start_time

        # Compute final power estimates
        power_perm = np.mean(perm_pvals <= alpha)
        power_asymp = np.mean(asymp_pvals <= alpha)

        # Standard errors (binomial)
        se_perm = np.sqrt(power_perm * (1 - power_perm) / R)
        se_asymp = np.sqrt(power_asymp * (1 - power_asymp) / R)

        print()
        print("=" * 70)
        print("RESULTS (Multi-p Simes Aggregation)")
        print("=" * 70)
        print("  Power (Permutation, alpha=%.2f):  %.4f +/- %.4f" %
              (alpha, power_perm, 1.96*se_perm))
        print("  Power (Asymptotic,  alpha=%.2f):  %.4f +/- %.4f" %
              (alpha, power_asymp, 1.96*se_asymp))
        for k_idx, p_val in enumerate(p_values):
            print("    Per-p mean T (p=%d): %.4f" % (p_val, np.mean(T_values[p_val])))
        print("  Total time: %.1fs (%.2fs per replicate)" %
              (total_time, total_time/R))
        print("=" * 70)

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        p_str = "_".join(str(pv) for pv in p_values)
        results_path = os.path.join(output_dir,
            "power_multip_d%d_p%s_m%d_n%d_%s.npz" %
            (d, p_str, m, n, timestamp))
        np.savez_compressed(
            results_path,
            d=d, p_values=p_values, m=m, n=n,
            kappa_X=kappa_X, kappa_Y=kappa_Y,
            alpha=alpha,
            R=R, B_perm=B_perm,
            master_seed=master_seed,
            T_values=T_values,
            perm_pvals=perm_pvals,
            asymp_pvals=asymp_pvals,
            power_perm=power_perm,
            power_asymp=power_asymp,
            se_perm=se_perm,
            se_asymp=se_asymp,
        )
        print("\nResults saved to: %s" % results_path)

        return {
            'power_perm': power_perm,
            'power_asymp': power_asymp,
            'se_perm': se_perm,
            'se_asymp': se_asymp,
            'T_values': T_values,
            'perm_pvals': perm_pvals,
            'asymp_pvals': asymp_pvals,
        }

    else:
        # Single-p mode (original behavior)
        config = SphericalTestConfig(p=p, d=d)
        calculator = OptimizedTestStatistic(config)

        print("=" * 70)
        print("DIRECTIONAL POWER SIMULATION: SPH-sqrt-p%d" % p)
        print("=" * 70)
        print("  Sphere dimension: d = %d (S^%d in R^%d)" % (d, d, d+1))
        print("  Truncation parameter: p = %d" % p)
        print("  Sample sizes: m = %d, n = %d" % (m, n))
        print("  vMF concentration: kappa_X = %.2f, kappa_Y = %.2f" % (kappa_X, kappa_Y))
        print("  kappa ratio: %.2f" % (kappa_Y/kappa_X))
        print("  Mean direction: matched (mu = [1, 0, ..., 0])")
        print("  Significance level: alpha = %.2f" % alpha)
        print("  Monte Carlo replications: R = %d" % R)
        print("  Permutation iterations: B_perm = %d" % B_perm)
        print("  Master seed: %d" % master_seed)
        print("=" * 70)

        T_values = np.zeros(R)
        asymp_pvals = np.zeros(R)
        perm_pvals = np.zeros(R)

        start_time = time.time()

        for r in range(R):
            seed_r = master_seed + r * 1000
            rng_data = np.random.default_rng(seed_r)
            rng_perm = np.random.default_rng(seed_r + 500)

            vmf_X = vonmises_fisher(mu=mu, kappa=kappa_X)
            X = vmf_X.rvs(size=m, random_state=rng_data)
            vmf_Y = vonmises_fisher(mu=mu, kappa=kappa_Y)
            Y = vmf_Y.rvs(size=n, random_state=rng_data)

            T_obs = calculator.compute(X, Y, use_unbiased=True)
            T_values[r] = T_obs

            Z_pooled = np.vstack([X, Y])
            # Precompute pooled kernel matrix once (CODE-04: batched perms)
            dot_ZZ = np.clip(Z_pooled @ Z_pooled.T, -1.0, 1.0)
            K_ZZ = config.compute_reproducing_kernel(dot_ZZ)
            
            # Null calibration: estimate null moments from B_cal permutations
            # (finite-sample correction to N(0,1) approximation)
            B_cal = min(50, B_perm)
            T_cal = np.zeros(B_cal)
            for b in range(B_cal):
                perm_cal = rng_perm.permutation(m + n)
                idx_X_cal = perm_cal[:m]
                idx_Y_cal = perm_cal[m:]
                K_XX_cal = K_ZZ[np.ix_(idx_X_cal, idx_X_cal)]
                K_YY_cal = K_ZZ[np.ix_(idx_Y_cal, idx_Y_cal)]
                K_XY_cal = K_ZZ[np.ix_(idx_X_cal, idx_Y_cal)]
                T_cal[b] = calculator.compute_from_kernels(
                    K_XX_cal, K_YY_cal, K_XY_cal, m, n, use_unbiased=True)
            
            mu_0 = np.mean(T_cal)
            sigma_0 = max(np.std(T_cal, ddof=1), 0.1)
            T_calibrated = (T_obs - mu_0) / sigma_0
            asymp_pvals[r] = norm.sf(T_calibrated)
            
            count_greater = 0
            for b in range(B_perm):
                perm = rng_perm.permutation(m + n)
                idx_X = perm[:m]
                idx_Y = perm[m:]
                K_XX_perm = K_ZZ[np.ix_(idx_X, idx_X)]
                K_YY_perm = K_ZZ[np.ix_(idx_Y, idx_Y)]
                K_XY_perm = K_ZZ[np.ix_(idx_X, idx_Y)]
                T_perm = calculator.compute_from_kernels(
                    K_XX_perm, K_YY_perm, K_XY_perm, m, n, use_unbiased=True)
                if T_perm >= T_obs:
                    count_greater += 1
            perm_pvals[r] = (1.0 + count_greater) / (B_perm + 1.0)

            if (r + 1) % 50 == 0 or r == 0:
                elapsed = time.time() - start_time
                power_perm = np.mean(perm_pvals[:r+1] <= alpha)
                power_asymp = np.mean(asymp_pvals[:r+1] <= alpha)
                print("  [%d/%d] elapsed=%.1fs | perm_power=%.4f | asymp_power=%.4f" %
                      (r+1, R, elapsed, power_perm, power_asymp))

        total_time = time.time() - start_time
        power_perm = np.mean(perm_pvals <= alpha)
        power_asymp = np.mean(asymp_pvals <= alpha)
        se_perm = np.sqrt(power_perm * (1 - power_perm) / R)
        se_asymp = np.sqrt(power_asymp * (1 - power_asymp) / R)

        print()
        print("=" * 70)
        print("RESULTS")
        print("=" * 70)
        print("  Power (Permutation, alpha=%.2f):  %.4f +/- %.4f" %
              (alpha, power_perm, 1.96*se_perm))
        print("  Power (Asymptotic,  alpha=%.2f):  %.4f +/- %.4f" %
              (alpha, power_asymp, 1.96*se_asymp))
        print("  Mean T statistic: %.4f" % np.mean(T_values))
        print("  Std T statistic:  %.4f" % np.std(T_values, ddof=1))
        print("  Total time: %.1fs (%.2fs per replicate)" %
              (total_time, total_time/R))
        print("=" * 70)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(output_dir,
            "power_directional_d%d_p%d_m%d_n%d_%s.npz" %
            (d, p, m, n, timestamp))
        np.savez_compressed(
            results_path,
            d=d, p=p, m=m, n=n,
            kappa_X=kappa_X, kappa_Y=kappa_Y,
            alpha=alpha, R=R, B_perm=B_perm,
            master_seed=master_seed,
            T_values=T_values,
            perm_pvals=perm_pvals,
            asymp_pvals=asymp_pvals,
            power_perm=power_perm,
            power_asymp=power_asymp,
            se_perm=se_perm,
            se_asymp=se_asymp,
        )
        print("\nResults saved to: %s" % results_path)

        return {
            'power_perm': power_perm,
            'power_asymp': power_asymp,
            'se_perm': se_perm,
            'se_asymp': se_asymp,
            'T_values': T_values,
            'perm_pvals': perm_pvals,
            'asymp_pvals': asymp_pvals,
        }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Directional Power Simulation')
    parser.add_argument('--d', type=int, default=2, help='Sphere dimension')
    parser.add_argument('--p', type=int, default=2, help='Truncation parameter')
    parser.add_argument('--m', type=int, default=25, help='Sample size X')
    parser.add_argument('--n', type=int, default=25, help='Sample size Y')
    parser.add_argument('--kappa_X', type=float, default=5.0, help='vMF concentration X')
    parser.add_argument('--kappa_Y', type=float, default=8.75, help='vMF concentration Y')
    parser.add_argument('--alpha', type=float, default=0.05, help='Significance level')
    parser.add_argument('--R', type=int, default=500, help='Monte Carlo replications')
    parser.add_argument('--B_perm', type=int, default=500, help='Permutation iterations')
    parser.add_argument('--seed', type=int, default=5086, help='Master seed')
    parser.add_argument('--output_dir', type=str, default='simulation_results',
                        help='Output directory')
    parser.add_argument('--single-p', action='store_true', default=False,
                        help='Use single-p mode instead of multi-p aggregation')
    args = parser.parse_args()

    results = run_power_simulation(
        d=args.d,
        m=args.m,
        n=args.n,
        p=args.p,
        kappa_X=args.kappa_X,
        kappa_Y=args.kappa_Y,
        alpha=args.alpha,
        R=args.R,
        B_perm=args.B_perm,
        master_seed=args.seed,
        output_dir=args.output_dir,
        use_multi_p=False,
    )
