#!/usr/bin/env python3
"""
Alibaba Experiment Runner for SP-UCB-OLP

Runs experiments on real Alibaba Cluster Trace data with:
- Original temporal order (no shuffling)
- New reward formula: r = duration × (α·cpu + β·mem) × noise
- K=3 regimes (8-bit, 4-bit, batching)

Algorithms tested:
- SP-UCB-OLP (α=0.1)
- Greedy (α=0)
- Random
- Oracle

Usage:
    # Run smoke test
    python run_experiment.py --smoke

    # Run full experiment
    python run_experiment.py

    # Run with specific config
    python run_experiment.py --rho 0.7 --seeds 42,43,44
"""

import argparse
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Any
import sys

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from data.alibaba_loader import AlibabaTraceLoader
from algorithms import get_algorithm


# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

FULL_CONFIG = {
    'T': 10000,
    'K': 3,
    'd': 2,
    'algorithms': ['SP-UCB-OLP', 'Greedy', 'Random', 'Oracle'],
    'rho_values': [0.5, 0.7, 1.0, 1.2],
    'seeds': [42, 43, 44, 45, 46],
    'alpha': 0.1,  # Exploration rate for SP-UCB-OLP
}

SMOKE_CONFIG = {
    'T': 1000,
    'K': 3,
    'd': 2,
    'algorithms': ['SP-UCB-OLP', 'Greedy', 'Random', 'Oracle'],
    'rho_values': [0.7, 1.0],
    'seeds': [42],
    'alpha': 0.1,
}


# =============================================================================
# MONTE CARLO ORACLE COMPUTATION
# =============================================================================

def compute_oracle_mc(
    loader: AlibabaTraceLoader,
    rho: float,
    n_restarts: int = 10
) -> Dict[str, Any]:
    """
    Compute switching-aware oracle V^mix via Monte Carlo.

    V^mix(b) = min_p { <p, b> + max_θ g_θ(p) }

    where g_θ(p) = E[(r - <p, a>)_+] is estimated from samples.

    Parameters
    ----------
    loader : AlibabaTraceLoader
        Data loader with samples
    rho : float
        Budget scaling parameter
    n_restarts : int
        Number of optimization restarts

    Returns
    -------
    dict
        Oracle values including V_mix, w_star, p_star
    """
    K, d, T = loader.K, loader.d, loader.T

    # Collect all samples
    samples = {}
    for theta in range(K):
        rewards, consumptions = loader.get_all_samples(theta)
        samples[theta] = (rewards, consumptions)

    # Per-period budget
    B = loader.get_budget(rho)
    b = B / T

    # Bounds for optimization
    R_max = max(np.max(samples[theta][0]) for theta in range(K))
    A_max = max(np.max(samples[theta][1]) for theta in range(K))
    b_min = np.min(b[b > 0]) if np.any(b > 0) else 1.0
    P_max = R_max / b_min + 1.0

    def empirical_surplus(theta: int, p: np.ndarray) -> float:
        """g_θ(p) = (1/n) Σ (r - <p, a>)_+"""
        rewards, consumptions = samples[theta]
        margins = rewards - consumptions @ p
        return np.mean(np.maximum(margins, 0))

    def objective_V_mix(p: np.ndarray) -> float:
        """min_p { <p, b> + max_θ g_θ(p) }"""
        linear_term = np.dot(p, b)
        surpluses = [empirical_surplus(theta, p) for theta in range(K)]
        envelope = max(surpluses)
        return linear_term + envelope

    # Multi-start optimization
    best_val = np.inf
    best_p = np.zeros(d)

    for restart in range(n_restarts):
        if restart == 0:
            p_init = np.zeros(d)
        else:
            p_init = np.random.uniform(0, P_max / 2, d)

        result = minimize(
            objective_V_mix,
            p_init,
            method='L-BFGS-B',
            bounds=[(0, P_max)] * d,
            options={'maxiter': 200, 'ftol': 1e-8}
        )

        if result.fun < best_val:
            best_val = result.fun
            best_p = result.x.copy()

    # Compute optimal mixture w* (configs achieving envelope at p_star)
    surpluses = np.array([empirical_surplus(theta, best_p) for theta in range(K)])
    max_surplus = np.max(surpluses)
    w_star = (surpluses >= max_surplus - 1e-8).astype(float)
    if w_star.sum() > 0:
        w_star /= w_star.sum()
    else:
        w_star = np.ones(K) / K

    return {
        'V_mix': best_val,
        'w_star': w_star,
        'p_star': best_p,
        'R_max': R_max,
        'A_max': A_max,
        'P_max': P_max,
    }


# =============================================================================
# SINGLE RUN FUNCTION
# =============================================================================

def run_single_experiment(
    loader: AlibabaTraceLoader,
    algorithm_name: str,
    rho: float,
    seed: int,
    oracle_values: Dict[str, Any],
    alpha: float = 0.1,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Run a single experiment.

    Parameters
    ----------
    loader : AlibabaTraceLoader
        Data loader
    algorithm_name : str
        Algorithm name
    rho : float
        Budget scaling
    seed : int
        Random seed
    oracle_values : dict
        Pre-computed oracle values
    alpha : float
        Exploration rate for SP-UCB-OLP
    verbose : bool
        Print progress

    Returns
    -------
    dict
        Experiment results
    """
    K, d, T = loader.K, loader.d, loader.T
    B = loader.get_budget(rho)
    V_mix = oracle_values['V_mix']

    # Setup algorithm config
    alg_config = {
        'R_max': oracle_values['R_max'],
        'A_max': oracle_values['A_max'],
        'warm_start': True,
    }

    if algorithm_name == 'SP-UCB-OLP':
        alg_config['alpha'] = alpha
    elif algorithm_name == 'Greedy':
        alg_config['alpha'] = 0.0
    elif algorithm_name == 'Oracle':
        alg_config['w_star'] = oracle_values['w_star']
        alg_config['p_star'] = oracle_values['p_star']

    # Create algorithm
    algorithm = get_algorithm(algorithm_name, K, d, T, B, alg_config)

    # Reset random seed
    np.random.seed(seed)

    # Run simulation
    start_time = time.time()

    for t in range(T):
        theta, w, p = algorithm.select_config(t)
        r, a = loader.get_arrival(theta, t)
        accept = algorithm.decide_admission(t, theta, r, a, p)
        algorithm.update(t, theta, r, a, accept)

        if verbose and (t + 1) % 1000 == 0:
            stats = algorithm.get_statistics()
            print(f"    t={t+1}: R={stats['total_reward']:.1f}, accepts={stats['total_accepts']}")

    elapsed_time = time.time() - start_time

    # Collect results
    stats = algorithm.get_statistics()
    total_reward = stats['total_reward']

    # Compute metrics
    regret = T * V_mix - total_reward
    competitive_ratio = total_reward / (T * V_mix) if V_mix > 0 else 0.0

    return {
        'algorithm': algorithm_name,
        'rho': rho,
        'seed': seed,
        'total_reward': total_reward,
        'regret': regret,
        'competitive_ratio': competitive_ratio,
        'acceptance_rate': stats['acceptance_rate'],
        'total_accepts': stats['total_accepts'],
        'V_mix': V_mix,
        'T_V_mix': T * V_mix,
        'K': K,
        'd': d,
        'T': T,
        'elapsed_time': elapsed_time,
        'budget_utilization': list(stats['budget_utilization']),
        'N_theta': stats['N_theta'],
    }


# =============================================================================
# MAIN EXPERIMENT RUNNER
# =============================================================================

def run_experiments(config: Dict, data_path: Path, results_dir: Path, verbose: bool = True):
    """Run all experiments according to config."""
    T = config['T']
    algorithms = config['algorithms']
    rho_values = config['rho_values']
    seeds = config['seeds']
    alpha = config.get('alpha', 0.1)

    total_runs = len(algorithms) * len(rho_values) * len(seeds)

    if verbose:
        print("=" * 70)
        print("ALIBABA EXPERIMENT RUNNER")
        print("=" * 70)
        print(f"T={T}, K={config['K']}, d={config['d']}")
        print(f"Algorithms: {algorithms}")
        print(f"Rho values: {rho_values}")
        print(f"Seeds: {seeds}")
        print(f"Total runs: {total_runs}")
        print(f"Data path: {data_path}")
        print(f"Results dir: {results_dir}")
        print("=" * 70)

    results = []
    run_count = 0
    overall_start = time.time()

    for seed in seeds:
        # Create loader for this seed (applies noise)
        loader = AlibabaTraceLoader(
            data_path=data_path,
            T=T,
            seed=seed,
        )

        for rho in rho_values:
            # Compute oracle for this (seed, rho) pair
            if verbose:
                print(f"\nComputing oracle: seed={seed}, rho={rho}...")
            oracle_values = compute_oracle_mc(loader, rho)

            if verbose:
                print(f"  V_mix = {oracle_values['V_mix']:.2f}")
                print(f"  w_star = {oracle_values['w_star']}")

            for alg_name in algorithms:
                run_count += 1

                if verbose:
                    print(f"\n[{run_count}/{total_runs}] {alg_name}, rho={rho}, seed={seed}")

                try:
                    result = run_single_experiment(
                        loader=loader,
                        algorithm_name=alg_name,
                        rho=rho,
                        seed=seed,
                        oracle_values=oracle_values,
                        alpha=alpha,
                        verbose=False,
                    )
                    results.append(result)

                    if verbose:
                        print(f"  CR={result['competitive_ratio']:.4f}, "
                              f"R={result['total_reward']:.1f}, "
                              f"time={result['elapsed_time']:.1f}s")

                except Exception as e:
                    print(f"  ERROR: {e}")
                    import traceback
                    traceback.print_exc()
                    results.append({
                        'algorithm': alg_name,
                        'rho': rho,
                        'seed': seed,
                        'error': str(e),
                    })

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
    # Filter out errors
    valid_results = [r for r in results if 'error' not in r]

    if not valid_results:
        print("No valid results to save!")
        return

    # Save raw results as JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = results_dir / f"alibaba_results_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'total_results': len(valid_results),
            'results': valid_results,
        }, f, indent=2)

    if verbose:
        print(f"Saved JSON: {json_path}")

    # Save as CSV
    df = pd.DataFrame(valid_results)
    csv_path = results_dir / f"alibaba_results_{timestamp}.csv"
    df.to_csv(csv_path, index=False)

    if verbose:
        print(f"Saved CSV: {csv_path}")

    # Create summary
    summary = df.groupby(['algorithm', 'rho']).agg({
        'competitive_ratio': ['mean', 'std', 'min', 'max'],
        'regret': ['mean', 'std'],
        'total_reward': ['mean', 'std'],
        'acceptance_rate': ['mean'],
        'elapsed_time': ['mean'],
    }).round(4)

    summary_path = results_dir / f"alibaba_summary_{timestamp}.csv"
    summary.to_csv(summary_path)

    if verbose:
        print(f"Saved summary: {summary_path}")
        print("\nSummary - Mean Competitive Ratio:")
        pivot = df.pivot_table(
            values='competitive_ratio',
            index='rho',
            columns='algorithm',
            aggfunc='mean'
        ).round(4)
        print(pivot.to_string())


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Run Alibaba SP-UCB-OLP experiments')
    parser.add_argument('--smoke', action='store_true', help='Run quick smoke test')
    parser.add_argument('--data', type=str, default='./data/raw/batch_task.csv',
                       help='Path to data CSV file')
    parser.add_argument('--results', type=str, default='./results',
                       help='Results directory')
    parser.add_argument('--rho', type=str, default=None,
                       help='Comma-separated rho values (e.g., "0.5,0.7,1.0")')
    parser.add_argument('--seeds', type=str, default=None,
                       help='Comma-separated seeds (e.g., "42,43,44")')
    parser.add_argument('--T', type=int, default=None, help='Time horizon')
    parser.add_argument('--quiet', action='store_true', help='Suppress output')

    args = parser.parse_args()

    # Select config
    config = SMOKE_CONFIG.copy() if args.smoke else FULL_CONFIG.copy()

    # Override with command line args
    if args.rho:
        config['rho_values'] = [float(x) for x in args.rho.split(',')]
    if args.seeds:
        config['seeds'] = [int(x) for x in args.seeds.split(',')]
    if args.T:
        config['T'] = args.T

    # Setup paths
    data_path = Path(args.data)
    results_dir = Path(args.results)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Check data exists
    if not data_path.exists():
        print(f"ERROR: Data file not found: {data_path}")
        print("Run download_data.py first to get the data.")
        sys.exit(1)

    # Run experiments
    verbose = not args.quiet
    results = run_experiments(config, data_path, results_dir, verbose)

    if verbose:
        print("\nDone!")


if __name__ == '__main__':
    main()
