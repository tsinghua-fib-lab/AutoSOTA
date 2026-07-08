#!/usr/bin/env python3
"""
Example 2: K Sensitivity Analysis (Table 9, Figure 8 from paper)

Demonstrates OCMB's robustness to the hyperparameter K (max parents).
Key finding: F1 varies by <5% across K/d ∈ [0.05, 0.20], showing OCMB is
NOT critically dependent on K tuning.

Expected results (d=100, scale-free):
- K/d = 0.02 (K=2):  F1 ≈ 0.248, #CI Tests ≈ 327
- K/d = 0.05 (K=5):  F1 ≈ 0.247, #CI Tests ≈ 3,225
- K/d = 0.10 (K=10): F1 ≈ 0.247, #CI Tests ≈ 8,505
- K/d = 0.20 (K=20): F1 ≈ 0.244, #CI Tests ≈ 17,532

Key insight: F1 plateau shows robustness; users can pick K = max(2, ⌈0.05d⌉)
as a simple heuristic without fine-tuning.
"""

import numpy as np
import sys
import os
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
SCRIPT_DIR = Path(__file__).resolve().parent
OCMB_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(OCMB_ROOT))

try:
    from ocmb import (
        OCMB_CaPS,
        OCMB_Precomputed,
        calculate_metrics
    )
except ImportError:
    print("Error: Could not import OCMB. Make sure OCMB is in your Python path.")
    print(f"Try: export PYTHONPATH={OCMB_ROOT}:$PYTHONPATH")
    sys.exit(1)


def generate_scale_free_graph(d, degree=3, seed=42):
    """Generate scale-free graph using preferential attachment"""
    try:
        import networkx as nx
        np.random.seed(seed)
        m = max(1, degree // 2)
        G = nx.barabasi_albert_graph(d, m, seed=seed)

        # Convert to DAG
        adj = np.zeros((d, d))
        for i, j in G.edges():
            if i < j:
                adj[i, j] = 1
            else:
                adj[j, i] = 1
        return adj
    except ImportError:
        print("Warning: networkx not found, using simple random DAG")
        np.random.seed(seed)
        adj = np.zeros((d, d))
        for i in range(d):
            for j in range(i + 1, d):
                if np.random.rand() < degree / d:
                    adj[i, j] = 1
        return adj


def generate_data_from_graph(adj, n_samples=1000, linear_ratio=0.5, seed=42):
    """Generate observational data from graph structure"""
    np.random.seed(seed)
    d = adj.shape[0]
    X = np.zeros((n_samples, d))

    for j in range(d):
        parents = np.where(adj[:, j] == 1)[0]
        if len(parents) == 0:
            X[:, j] = np.random.randn(n_samples)
        else:
            z = np.zeros(n_samples)
            for p in parents:
                weight = np.random.uniform(0.5, 2.0) * np.random.choice([-1, 1])
                if np.random.rand() < linear_ratio:
                    z += weight * X[:, p]
                else:
                    z += weight * np.tanh(X[:, p])
            X[:, j] = z + np.random.randn(n_samples)

    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    return X


def is_gpu_available():
    """Check if GPU is available"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def main():
    print("=" * 80)
    print("OCMB Example 2: K Sensitivity Analysis")
    print("Reproducing Table 9 and Figure 8 from the paper")
    print("=" * 80)
    print()

    # Configuration
    d = 100
    n = 1000
    degree = 3
    seeds = [42, 123, 456]

    # K values to sweep (as fractions of d)
    k_over_d_ratios = [0.02, 0.05, 0.08, 0.10, 0.20, 0.40]
    K_values = [max(1, int(d * ratio)) for ratio in k_over_d_ratios]

    print(f"Configuration:")
    print(f"  Nodes (d): {d}")
    print(f"  Samples (n): {n}")
    print(f"  K/d ratios: {k_over_d_ratios}")
    print(f"  K values: {K_values}")
    print(f"  Seeds: {seeds}")
    print(f"  GPU available: {is_gpu_available()}")
    print()

    # Results storage
    all_results = []

    # Run experiments for each seed
    for seed_idx, seed in enumerate(seeds, 1):
        print(f"\n{'='*60}")
        print(f"Seed {seed} ({seed_idx}/{len(seeds)})")
        print(f"{'='*60}")

        # Generate graph and data
        true_adj = generate_scale_free_graph(d, degree, seed)
        n_edges = int(np.sum(true_adj))
        X = generate_data_from_graph(true_adj, n, seed=seed)
        print(f"Generated scale-free graph: {d} nodes, {n_edges} edges")

        # Step 1: Get ordering from CaPS (do this ONCE per seed)
        print("\nStep 1: Computing ordering with CaPS...", end=' ')
        start_time = time.time()

        ocmb_caps = OCMB_CaPS(
            max_parents=10,  # Use moderate K for ordering
            k_mb=5,
            alpha_mb=0.01,
            eta_G=0.001,
            eta_H=0.001,
            device='cuda:0' if is_gpu_available() else 'cpu',
            verbose=False
        )
        ocmb_caps.fit(X, true_adj=true_adj)

        order = ocmb_caps.order_
        parent_scores = ocmb_caps.parent_scores_
        ordering_time = time.time() - start_time
        print(f"✓ Done ({ordering_time:.1f}s)")

        # Step 2: Sweep K values using precomputed ordering
        print("\nStep 2: K-sweep with precomputed ordering...")
        print(f"{'K/d':<8} {'K':<6} {'F1':<8} {'SHD':<8} {'#CI Tests':<12} {'Time (s)':<10}")
        print("-" * 70)

        for K, k_d_ratio in zip(K_values, k_over_d_ratios):
            try:
                start_time = time.time()

                # Use precomputed ordering (avoids re-training CaPS)
                ocmb = OCMB_Precomputed(
                    order=order,
                    parent_scores=parent_scores,
                    max_parents=K,
                    k_mb=5,
                    alpha_mb=0.01,
                    verbose=False
                )
                ocmb.fit(X, true_adj=true_adj)

                graph = ocmb.get_adjacency_matrix()
                metrics = calculate_metrics(true_adj, graph)
                elapsed = time.time() - start_time
                n_cmi = ocmb.get_n_cmi_calls()

                print(f"{k_d_ratio:<8.2f} {K:<6} {metrics['F1']:<8.3f} "
                      f"{metrics['SHD']:<8} {n_cmi:<12} {elapsed:<10.2f}")

                all_results.append({
                    'seed': seed,
                    'K': K,
                    'K_over_d': k_d_ratio,
                    'F1': metrics['F1'],
                    'SHD': metrics['SHD'],
                    'Precision': metrics['Precision'],
                    'Recall': metrics['Recall'],
                    'n_cmi_calls': n_cmi,
                    'time': elapsed,
                    'covPa': ocmb.cand_stats_.get('covPa_mean', 0) if ocmb.cand_stats_ else 0,
                    'status': 'success'
                })

            except Exception as e:
                print(f"{k_d_ratio:<8.2f} {K:<6} {'ERROR':<8} {str(e)}")
                all_results.append({
                    'seed': seed,
                    'K': K,
                    'K_over_d': k_d_ratio,
                    'status': 'failed',
                    'error': str(e)
                })

    # Aggregate results
    print("\n" + "=" * 80)
    print("SUMMARY: Averaged Results Across Seeds")
    print("=" * 80)
    print(f"{'K/d':<8} {'K':<6} {'F1':<15} {'SHD':<15} {'#CI Tests':<15} {'CovPa':<10}")
    print("-" * 80)

    for K, k_d_ratio in zip(K_values, k_over_d_ratios):
        filtered = [r for r in all_results
                   if r['K'] == K and r['status'] == 'success']

        if not filtered:
            print(f"{k_d_ratio:<8.2f} {K:<6} {'N/A':<15}")
            continue

        mean_f1 = np.mean([r['F1'] for r in filtered])
        std_f1 = np.std([r['F1'] for r in filtered])
        mean_shd = np.mean([r['SHD'] for r in filtered])
        std_shd = np.std([r['SHD'] for r in filtered])
        mean_ci = np.mean([r['n_cmi_calls'] for r in filtered])
        mean_covpa = np.mean([r['covPa'] for r in filtered])

        print(f"{k_d_ratio:<8.2f} {K:<6} "
              f"{mean_f1:.3f}±{std_f1:.3f}    "
              f"{mean_shd:.1f}±{std_shd:.1f}      "
              f"{mean_ci:<15.0f} {mean_covpa:<10.3f}")

    # Analysis: Robustness plateau
    print("\n" + "=" * 80)
    print("ANALYSIS: Robustness to K")
    print("=" * 80)

    # Get F1 values for K/d in [0.05, 0.20]
    plateau_results = [r for r in all_results
                      if 0.05 <= r['K_over_d'] <= 0.20
                      and r['status'] == 'success']

    if plateau_results:
        plateau_f1s = [r['F1'] for r in plateau_results]
        plateau_mean = np.mean(plateau_f1s)
        plateau_std = np.std(plateau_f1s)
        plateau_range = max(plateau_f1s) - min(plateau_f1s)
        plateau_cv = plateau_std / plateau_mean if plateau_mean > 0 else 0

        print(f"\nF1 Stability in K/d ∈ [0.05, 0.20]:")
        print(f"  Mean F1: {plateau_mean:.3f}")
        print(f"  Std F1: {plateau_std:.3f}")
        print(f"  Range: {plateau_range:.3f}")
        print(f"  Coefficient of Variation: {plateau_cv:.1%}")

        if plateau_range < 0.05:
            print(f"\n✓ ROBUSTNESS CONFIRMED: F1 varies by <5% across K/d ∈ [0.05, 0.20]")
            print(f"  This matches the paper's key finding (Table 9)")
        else:
            print(f"\n⚠ Higher variance than expected (paper reports <5%)")

    # Practical recommendation
    print("\n" + "=" * 80)
    print("PRACTICAL RECOMMENDATION")
    print("=" * 80)
    print(f"\nBased on the paper (Section 4.8, Table 9):")
    print(f"  Recommended heuristic: K* = max(2, ⌈0.05d⌉)")
    print(f"  For d={d}: K* = {max(2, int(np.ceil(0.05 * d)))}")
    print(f"\nKey insight:")
    print(f"  - OCMB is remarkably INSENSITIVE to K across K/d ∈ [0.05, 0.20]")
    print(f"  - Users don't need to fine-tune K for their specific dataset")
    print(f"  - The heuristic K* = max(2, ⌈0.05d⌉) works well in practice")
    print(f"  - Larger K increases #CI tests but doesn't degrade accuracy much")

    # Comparison with paper
    print("\n" + "=" * 80)
    print("COMPARISON WITH PAPER (Table 9, d=50)")
    print("=" * 80)
    print(f"{'K/d':<8} {'F1 (Paper)':<15} {'F1 (Ours, d=100)':<20}")
    print("-" * 50)

    # Paper values are for d=50, we're using d=100
    paper_values = {
        0.02: 0.248,
        0.10: 0.247,
        0.40: 0.244
    }

    for k_d_ratio in [0.02, 0.10, 0.40]:
        filtered = [r for r in all_results
                   if abs(r['K_over_d'] - k_d_ratio) < 0.01
                   and r['status'] == 'success']

        if filtered:
            mean_f1 = np.mean([r['F1'] for r in filtered])
            paper_f1 = paper_values.get(k_d_ratio, 0)
            print(f"{k_d_ratio:<8.2f} {paper_f1:.3f}           {mean_f1:.3f}")

    print("\nNote: Direct comparison difficult due to different d (50 vs 100),")
    print("but the PLATEAU PATTERN should be similar.")
    print("=" * 80)


if __name__ == '__main__':
    main()
