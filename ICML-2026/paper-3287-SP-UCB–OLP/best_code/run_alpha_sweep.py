#!/usr/bin/env python3
"""
Alpha Sweep Experiment

Tests different exploration rates (alpha) for SP-UCB-OLP with fixed rho=0.7.

Alpha values tested:
- 0.0 (pure greedy, same as Greedy baseline)
- 0.01 (minimal exploration)
- 0.05 (low exploration)
- 0.1 (moderate exploration, default)
- 0.5 (high exploration)
- 1.0 (very high exploration)

Usage:
    python run_alpha_sweep.py
    python run_alpha_sweep.py --T 5000 --seeds 42,43,44,45,46
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

ALPHA_SWEEP_CONFIG = {
    'T': 10000,
    'K': 3,
    'd': 2,
    'rho': 0.7,  # Fixed rho
    'alpha_values': [0.0, 0.01, 0.05, 0.1, 0.5, 1.0],
    'seeds': [42, 43, 44, 45, 46],
    'include_baselines': True,  # Also run Random and Oracle for comparison
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
    np.random.seed(seed)

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
        'total_reward': total_reward,
        'regret': T * V_mix - total_reward,
        'competitive_ratio': competitive_ratio,
        'acceptance_rate': stats['acceptance_rate'],
        'total_accepts': stats['total_accepts'],
        'V_mix': V_mix,
        'T': T, 'K': K, 'd': d,
        'elapsed_time': elapsed_time,
    }


# =============================================================================
# MAIN
# =============================================================================

def run_alpha_sweep(config: Dict, data_path: Path, results_dir: Path, verbose: bool = True):
    """Run alpha sweep experiment."""
    T = config['T']
    rho = config['rho']
    alpha_values = config['alpha_values']
    seeds = config['seeds']
    include_baselines = config.get('include_baselines', True)

    # Count total runs
    n_alpha_runs = len(alpha_values) * len(seeds)
    n_baseline_runs = 2 * len(seeds) if include_baselines else 0  # Random + Oracle
    total_runs = n_alpha_runs + n_baseline_runs

    if verbose:
        print("=" * 70)
        print("ALPHA SWEEP EXPERIMENT")
        print("=" * 70)
        print(f"T={T}, rho={rho} (fixed)")
        print(f"Alpha values: {alpha_values}")
        print(f"Seeds: {seeds}")
        print(f"Include baselines: {include_baselines}")
        print(f"Total runs: {total_runs}")
        print("=" * 70)

    results = []
    run_count = 0
    overall_start = time.time()

    for seed in seeds:
        # Create loader for this seed
        loader = AlibabaTraceLoader(data_path=data_path, T=T, seed=seed)

        # Compute oracle once per seed
        if verbose:
            print(f"\nSeed={seed}: Computing oracle...")
        oracle_values = compute_oracle_mc(loader, rho)
        if verbose:
            print(f"  V_mix = {oracle_values['V_mix']:.2f}")

        # Run SP-UCB-OLP with different alpha values
        for alpha in alpha_values:
            run_count += 1
            if verbose:
                print(f"[{run_count}/{total_runs}] SP-UCB-OLP alpha={alpha}, seed={seed}", end=" ")

            result = run_single(loader, 'SP-UCB-OLP', rho, seed, oracle_values, alpha=alpha)
            results.append(result)

            if verbose:
                print(f"-> CR={result['competitive_ratio']:.4f}")

        # Run baselines
        if include_baselines:
            for alg in ['Random', 'Oracle']:
                run_count += 1
                if verbose:
                    print(f"[{run_count}/{total_runs}] {alg}, seed={seed}", end=" ")

                result = run_single(loader, alg, rho, seed, oracle_values)
                result['alpha'] = None  # No alpha for baselines
                results.append(result)

                if verbose:
                    print(f"-> CR={result['competitive_ratio']:.4f}")

    total_time = time.time() - overall_start

    if verbose:
        print("\n" + "=" * 70)
        print(f"Completed {total_runs} runs in {total_time:.1f}s")
        print("=" * 70)

    # Save results
    save_results(results, results_dir, verbose)

    return results


def save_results(results: List[Dict], results_dir: Path, verbose: bool = True):
    """Save results to CSV and JSON."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save JSON
    json_path = results_dir / f"alpha_sweep_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump({'timestamp': timestamp, 'results': results}, f, indent=2)
    if verbose:
        print(f"Saved: {json_path}")

    # Save CSV
    df = pd.DataFrame(results)
    csv_path = results_dir / f"alpha_sweep_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    if verbose:
        print(f"Saved: {csv_path}")

    # Print summary
    if verbose:
        print("\n" + "=" * 70)
        print("SUMMARY: Mean Competitive Ratio by Alpha")
        print("=" * 70)

        # SP-UCB-OLP results
        sp_results = df[df['algorithm'] == 'SP-UCB-OLP']
        summary = sp_results.groupby('alpha')['competitive_ratio'].agg(['mean', 'std']).round(4)
        print("\nSP-UCB-OLP:")
        for alpha, row in summary.iterrows():
            print(f"  alpha={alpha:5.2f}: CR = {row['mean']:.4f} ± {row['std']:.4f}")

        # Baselines
        baselines = df[df['algorithm'].isin(['Random', 'Oracle'])]
        if len(baselines) > 0:
            print("\nBaselines:")
            for alg in ['Oracle', 'Random']:
                subset = baselines[baselines['algorithm'] == alg]
                if len(subset) > 0:
                    mean = subset['competitive_ratio'].mean()
                    std = subset['competitive_ratio'].std()
                    print(f"  {alg:10s}: CR = {mean:.4f} ± {std:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Run alpha sweep experiment')
    parser.add_argument('--data', type=str, default='./data/raw/batch_task.csv')
    parser.add_argument('--results', type=str, default='./results')
    parser.add_argument('--T', type=int, default=None)
    parser.add_argument('--seeds', type=str, default=None)
    parser.add_argument('--alphas', type=str, default=None,
                       help='Comma-separated alpha values (e.g., "0.0,0.1,0.5,1.0")')
    parser.add_argument('--quiet', action='store_true')

    args = parser.parse_args()

    config = ALPHA_SWEEP_CONFIG.copy()
    if args.T:
        config['T'] = args.T
    if args.seeds:
        config['seeds'] = [int(x) for x in args.seeds.split(',')]
    if args.alphas:
        config['alpha_values'] = [float(x) for x in args.alphas.split(',')]

    data_path = Path(args.data)
    results_dir = Path(args.results)
    results_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        print(f"ERROR: Data file not found: {data_path}")
        print("Run download_data.py first.")
        sys.exit(1)

    run_alpha_sweep(config, data_path, results_dir, verbose=not args.quiet)
    print("\nDone!")


if __name__ == '__main__':
    main()
