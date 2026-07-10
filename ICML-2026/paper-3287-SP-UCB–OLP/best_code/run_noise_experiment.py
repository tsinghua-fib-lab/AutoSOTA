#!/usr/bin/env python3
"""
Noise Variance Experiment

Tests how different noise levels affect the optimal alpha.

Noise: LogNormal(mean=0, sigma=noise_sigma)
- sigma=0.1: Low noise (E[noise] ≈ 1.005, CV ≈ 10%)
- sigma=0.3: Medium noise (E[noise] ≈ 1.046, CV ≈ 31%)
- sigma=0.5: High noise (E[noise] ≈ 1.133, CV ≈ 53%)
- sigma=1.0: Very high noise (E[noise] ≈ 1.649, CV ≈ 131%)

Usage:
    python run_noise_experiment.py
    python run_noise_experiment.py --noise 0.3,0.5,1.0 --alphas 0.01,0.1
"""

import argparse
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy.optimize import minimize
from typing import Dict, List, Any
import sys

sys.path.insert(0, str(Path(__file__).parent))

from data.alibaba_loader import AlibabaTraceLoader
from algorithms import get_algorithm


# =============================================================================
# CONFIGURATION
# =============================================================================

NOISE_CONFIG = {
    'T': 5000,
    'K': 3,
    'd': 2,
    'rho': 0.7,
    'noise_sigmas': [0.1, 0.3, 0.5, 1.0],
    'alpha_values': [0.0, 0.01, 0.1],
    'seeds': [42, 43, 44, 45, 46],
}


# =============================================================================
# ORACLE COMPUTATION
# =============================================================================

def compute_oracle_mc(loader: AlibabaTraceLoader, rho: float, n_restarts: int = 10) -> Dict[str, Any]:
    """Compute switching-aware oracle V^mix via Monte Carlo."""
    K, d, T = loader.K, loader.d, loader.T

    samples = {}
    for theta in range(K):
        rewards, consumptions = loader.get_all_samples(theta)
        samples[theta] = (rewards, consumptions)

    B = loader.get_budget(rho)
    b = B / T

    R_max = max(np.max(samples[theta][0]) for theta in range(K))
    A_max = max(np.max(samples[theta][1]) for theta in range(K))
    b_min = np.min(b[b > 0]) if np.any(b > 0) else 1.0
    P_max = R_max / b_min + 1.0

    def empirical_surplus(theta: int, p: np.ndarray) -> float:
        rewards, consumptions = samples[theta]
        margins = rewards - consumptions @ p
        return np.mean(np.maximum(margins, 0))

    def objective_V_mix(p: np.ndarray) -> float:
        linear_term = np.dot(p, b)
        surpluses = [empirical_surplus(theta, p) for theta in range(K)]
        return linear_term + max(surpluses)

    best_val = np.inf
    best_p = np.zeros(d)

    for restart in range(n_restarts):
        p_init = np.zeros(d) if restart == 0 else np.random.uniform(0, P_max / 2, d)
        result = minimize(objective_V_mix, p_init, method='L-BFGS-B',
                         bounds=[(0, P_max)] * d, options={'maxiter': 200, 'ftol': 1e-8})
        if result.fun < best_val:
            best_val = result.fun
            best_p = result.x.copy()

    surpluses = np.array([empirical_surplus(theta, best_p) for theta in range(K)])
    max_surplus = np.max(surpluses)
    w_star = (surpluses >= max_surplus - 1e-8).astype(float)
    w_star = w_star / w_star.sum() if w_star.sum() > 0 else np.ones(K) / K

    return {
        'V_mix': best_val, 'w_star': w_star, 'p_star': best_p,
        'R_max': R_max, 'A_max': A_max, 'P_max': P_max,
    }


# =============================================================================
# SINGLE RUN
# =============================================================================

def run_single(loader, algorithm_name, rho, seed, oracle_values, alpha=0.1) -> Dict[str, Any]:
    """Run a single experiment."""
    K, d, T = loader.K, loader.d, loader.T
    B = loader.get_budget(rho)
    V_mix = oracle_values['V_mix']

    alg_config = {
        'R_max': oracle_values['R_max'],
        'A_max': oracle_values['A_max'],
        'warm_start': True,
    }

    if algorithm_name == 'SP-UCB-OLP':
        alg_config['alpha'] = alpha
    elif algorithm_name == 'Oracle':
        alg_config['w_star'] = oracle_values['w_star']
        alg_config['p_star'] = oracle_values['p_star']

    algorithm = get_algorithm(algorithm_name, K, d, T, B, alg_config)
    np.random.seed(seed + 1000)  # Different seed for algorithm randomness

    start_time = time.time()
    for t in range(T):
        theta, w, p = algorithm.select_config(t)
        r, a = loader.get_arrival(theta, t)
        accept = algorithm.decide_admission(t, theta, r, a, p)
        algorithm.update(t, theta, r, a, accept)
    elapsed_time = time.time() - start_time

    stats = algorithm.get_statistics()
    total_reward = stats['total_reward']
    competitive_ratio = total_reward / (T * V_mix) if V_mix > 0 else 0.0

    return {
        'algorithm': algorithm_name,
        'alpha': alpha,
        'rho': rho,
        'seed': seed,
        'noise_sigma': loader.noise_sigma,
        'total_reward': total_reward,
        'regret': T * V_mix - total_reward,
        'competitive_ratio': competitive_ratio,
        'acceptance_rate': stats['acceptance_rate'],
        'V_mix': V_mix,
        'T': T,
        'elapsed_time': elapsed_time,
    }


# =============================================================================
# MAIN
# =============================================================================

def run_noise_experiment(config: Dict, data_path: Path, results_dir: Path, verbose: bool = True):
    """Run noise variance experiment."""
    T = config['T']
    rho = config['rho']
    noise_sigmas = config['noise_sigmas']
    alpha_values = config['alpha_values']
    seeds = config['seeds']

    total_runs = len(noise_sigmas) * len(alpha_values) * len(seeds) + len(noise_sigmas) * len(seeds)  # +Oracle

    if verbose:
        print("=" * 70)
        print("NOISE VARIANCE EXPERIMENT")
        print("=" * 70)
        print(f"T={T}, rho={rho}")
        print(f"Noise sigmas: {noise_sigmas}")
        print(f"Alpha values: {alpha_values}")
        print(f"Seeds: {seeds}")
        print(f"Total runs: {total_runs}")
        print("=" * 70)

    results = []
    run_count = 0
    overall_start = time.time()

    for noise_sigma in noise_sigmas:
        if verbose:
            print(f"\n{'='*70}")
            print(f"NOISE SIGMA = {noise_sigma}")
            print(f"{'='*70}")

        for seed in seeds:
            # Create loader with specific noise level
            loader = AlibabaTraceLoader(
                data_path=data_path,
                T=T,
                seed=seed,
                noise_sigma=noise_sigma,
            )

            # Compute oracle
            oracle_values = compute_oracle_mc(loader, rho)
            if verbose and seed == seeds[0]:
                print(f"  V_mix = {oracle_values['V_mix']:.2f}")

            # Run SP-UCB-OLP with different alpha values
            for alpha in alpha_values:
                run_count += 1
                result = run_single(loader, 'SP-UCB-OLP', rho, seed, oracle_values, alpha=alpha)
                results.append(result)
                if verbose:
                    print(f"[{run_count}/{total_runs}] sigma={noise_sigma}, alpha={alpha}, seed={seed} -> CR={result['competitive_ratio']:.4f}")

            # Run Oracle for comparison
            run_count += 1
            result = run_single(loader, 'Oracle', rho, seed, oracle_values)
            result['alpha'] = None
            results.append(result)
            if verbose:
                print(f"[{run_count}/{total_runs}] sigma={noise_sigma}, Oracle, seed={seed} -> CR={result['competitive_ratio']:.4f}")

    total_time = time.time() - overall_start

    if verbose:
        print("\n" + "=" * 70)
        print(f"Completed {total_runs} runs in {total_time:.1f}s")
        print("=" * 70)

    # Save and summarize
    save_and_summarize(results, results_dir, verbose)

    return results


def save_and_summarize(results: List[Dict], results_dir: Path, verbose: bool = True):
    """Save results and print summary."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save CSV
    df = pd.DataFrame(results)
    csv_path = results_dir / f"noise_experiment_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    if verbose:
        print(f"Saved: {csv_path}")

    # Summary by noise_sigma and alpha
    if verbose:
        print("\n" + "=" * 70)
        print("SUMMARY: Mean Competitive Ratio")
        print("=" * 70)

        sp_results = df[df['algorithm'] == 'SP-UCB-OLP']
        oracle_results = df[df['algorithm'] == 'Oracle']

        # Create pivot table
        noise_sigmas = sorted(df['noise_sigma'].unique())
        alpha_values = sorted(sp_results['alpha'].unique())

        print(f"\n{'Noise σ':<10}", end="")
        for alpha in alpha_values:
            print(f"{'α='+str(alpha):<12}", end="")
        print(f"{'Oracle':<12}")
        print("-" * (10 + 12 * (len(alpha_values) + 1)))

        for sigma in noise_sigmas:
            print(f"{sigma:<10.1f}", end="")
            for alpha in alpha_values:
                subset = sp_results[(sp_results['noise_sigma'] == sigma) & (sp_results['alpha'] == alpha)]
                mean_cr = subset['competitive_ratio'].mean()
                print(f"{mean_cr:<12.4f}", end="")
            oracle_subset = oracle_results[oracle_results['noise_sigma'] == sigma]
            oracle_cr = oracle_subset['competitive_ratio'].mean()
            print(f"{oracle_cr:<12.4f}")

        # Best alpha for each noise level
        print("\n" + "=" * 70)
        print("BEST ALPHA BY NOISE LEVEL")
        print("=" * 70)
        for sigma in noise_sigmas:
            best_alpha = None
            best_cr = 0
            for alpha in alpha_values:
                subset = sp_results[(sp_results['noise_sigma'] == sigma) & (sp_results['alpha'] == alpha)]
                mean_cr = subset['competitive_ratio'].mean()
                if mean_cr > best_cr:
                    best_cr = mean_cr
                    best_alpha = alpha
            print(f"  σ={sigma}: Best α={best_alpha} (CR={best_cr:.4f})")


def main():
    parser = argparse.ArgumentParser(description='Run noise variance experiment')
    parser.add_argument('--data', type=str, default='./data/raw/batch_task.csv')
    parser.add_argument('--results', type=str, default='./results')
    parser.add_argument('--T', type=int, default=5000)
    parser.add_argument('--seeds', type=str, default='42,43,44,45,46')
    parser.add_argument('--noise', type=str, default='0.1,0.3,0.5,1.0',
                       help='Comma-separated noise sigma values')
    parser.add_argument('--alphas', type=str, default='0.0,0.01,0.1',
                       help='Comma-separated alpha values')
    parser.add_argument('--quiet', action='store_true')

    args = parser.parse_args()

    config = NOISE_CONFIG.copy()
    config['T'] = args.T
    config['seeds'] = [int(x) for x in args.seeds.split(',')]
    config['noise_sigmas'] = [float(x) for x in args.noise.split(',')]
    config['alpha_values'] = [float(x) for x in args.alphas.split(',')]

    data_path = Path(args.data)
    results_dir = Path(args.results)
    results_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        print(f"ERROR: Data file not found: {data_path}")
        sys.exit(1)

    run_noise_experiment(config, data_path, results_dir, verbose=not args.quiet)
    print("\nDone!")


if __name__ == '__main__':
    main()
