#!/usr/bin/env python3
"""
Example 4: Ordering Error Sensitivity (Figure 10 from paper)

Demonstrates OCMB's robustness to ordering corruption.
Key finding: Graceful degradation under ordering errors.

Expected results (d=100, K=1, scale-free):
- 0% error:  F1 ≈ 0.84, Parent Recall ≈ 1.0, SHD ≈ 27
- 10% error: F1 ≈ 0.66, Parent Recall ≈ 0.80, SHD ≈ 325
- 20% error: F1 ≈ 0.60, Parent Recall ≈ 0.75, SHD ≈ 425
- 30% error: F1 ≈ 0.58, Parent Recall ≈ 0.70, SHD ≈ 460
- 50% error: F1 ≈ 0.56, Parent Recall ≈ 0.65, SHD ≈ 695

Key insights:
- At 50% error (random-like ordering), F1 only drops by ~33%
- Parent Recall stays at 0.65 even with 50% corruption
- Demonstrates ordering constraints provide value even when imperfect
- Much better than catastrophic failure
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
        calculate_metrics,
        get_topological_order_from_dag
    )
except ImportError:
    print("Error: Could not import OCMB. Make sure OCMB is in your Python path.")
    print(f"Try: export PYTHONPATH={OCMB_ROOT}:$PYTHONPATH")
    sys.exit(1)


def generate_scale_free_graph(d, degree=3, seed=42):
    """Generate scale-free graph"""
    try:
        import networkx as nx
        np.random.seed(seed)
        m = max(1, degree // 2)
        G = nx.barabasi_albert_graph(d, m, seed=seed)

        adj = np.zeros((d, d))
        for i, j in G.edges():
            if i < j:
                adj[i, j] = 1
            else:
                adj[j, i] = 1
        return adj
    except ImportError:
        print("Warning: networkx not found")
        np.random.seed(seed)
        adj = np.zeros((d, d))
        for i in range(d):
            for j in range(i + 1, d):
                if np.random.rand() < degree / d:
                    adj[i, j] = 1
        return adj


def generate_data_from_graph(adj, n_samples=1000, seed=42):
    """Generate data from graph"""
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
                if np.random.rand() < 0.5:
                    z += weight * X[:, p]
                else:
                    z += weight * np.tanh(X[:, p])
            X[:, j] = z + np.random.randn(n_samples)

    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    return X


def inject_ordering_errors(true_order, error_rate, seed=42):
    """
    Inject ordering errors by swapping pairs.

    Args:
        true_order: True topological ordering
        error_rate: Fraction of positions to corrupt (0 to 1)
        seed: Random seed

    Returns:
        Corrupted ordering
    """
    np.random.seed(seed)
    order = true_order.copy()
    d = len(order)

    # Number of swaps to make
    n_swaps = int(error_rate * d)

    for _ in range(n_swaps):
        # Pick two random positions and swap
        i, j = np.random.choice(d, size=2, replace=False)
        order[i], order[j] = order[j], order[i]

    return order


def compute_ordering_divergence(true_order, pred_order):
    """
    Compute ordering divergence (Kendall tau distance normalized).

    Returns value in [0, 1] where 0 = perfect, 1 = reversed
    """
    d = len(true_order)

    # Create position maps
    true_pos = {node: i for i, node in enumerate(true_order)}
    pred_pos = {node: i for i, node in enumerate(pred_order)}

    # Count inversions
    inversions = 0
    for i in range(d):
        for j in range(i + 1, d):
            # Check if i < j in true_order but j < i in pred_order
            node_i, node_j = true_order[i], true_order[j]
            if pred_pos[node_i] > pred_pos[node_j]:
                inversions += 1

    # Normalize by maximum possible inversions
    max_inversions = d * (d - 1) / 2
    divergence = inversions / max_inversions if max_inversions > 0 else 0

    return divergence


def compute_parent_recall(true_adj, candidate_sets):
    """
    Compute Parent Recall (coverage of true parents in candidate sets).

    Args:
        true_adj: True adjacency matrix
        candidate_sets: List of candidate sets for each node

    Returns:
        Parent recall (fraction of true parents captured)
    """
    d = true_adj.shape[0]
    total_true_parents = 0
    captured_parents = 0

    for j in range(d):
        true_parents = set(np.where(true_adj[:, j] == 1)[0])
        total_true_parents += len(true_parents)

        if j < len(candidate_sets) and candidate_sets[j] is not None:
            cand_set = set(candidate_sets[j])
            captured_parents += len(true_parents & cand_set)

    return captured_parents / total_true_parents if total_true_parents > 0 else 0


def is_gpu_available():
    """Check if GPU is available"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def main():
    print("=" * 80)
    print("OCMB Example 4: Ordering Error Sensitivity")
    print("Reproducing Figure 10 from the paper")
    print("=" * 80)
    print()

    # Configuration
    d = 100
    n = 1000
    degree = 3
    K_values = [1, 5, 10]  # Test with different K values
    error_rates = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]
    seeds = [42, 142, 242]

    print(f"Configuration:")
    print(f"  Nodes (d): {d}")
    print(f"  Samples (n): {n}")
    print(f"  K values: {K_values}")
    print(f"  Error rates: {[f'{e:.0%}' for e in error_rates]}")
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

        # Get true topological ordering
        true_order = get_topological_order_from_dag(true_adj)

        print(f"Generated scale-free graph: {d} nodes, {n_edges} edges")
        print(f"True topological order: {true_order[:10]}... (showing first 10)")

        # Get learned ordering from CaPS (once per seed)
        print("\nStep 1: Computing learned ordering with CaPS...")
        ocmb_caps = OCMB_CaPS(
            max_parents=10,
            k_mb=5,
            alpha_mb=0.01,
            eta_G=0.001,
            eta_H=0.001,
            device='cuda:0' if is_gpu_available() else 'cpu',
            verbose=False
        )
        ocmb_caps.fit(X, true_adj=true_adj)
        learned_order = ocmb_caps.order_
        parent_scores = ocmb_caps.parent_scores_

        learned_div = compute_ordering_divergence(true_order, learned_order)
        print(f"Learned ordering divergence: {learned_div:.3f}")

        # Test each K value
        for K in K_values:
            print(f"\n--- K={K} ---")
            print(f"{'Error Rate':<12} {'Actual Div':<12} {'F1':<10} {'SHD':<10} {'CovPa':<10}")
            print("-" * 60)

            for error_rate in error_rates:
                try:
                    # Inject errors into learned ordering
                    if error_rate == 0.0:
                        corrupted_order = learned_order.copy()
                    else:
                        corrupted_order = inject_ordering_errors(
                            learned_order, error_rate, seed=seed + int(error_rate * 100)
                        )

                    # Measure actual divergence
                    actual_div = compute_ordering_divergence(true_order, corrupted_order)

                    # Run OCMB with corrupted ordering
                    ocmb = OCMB_Precomputed(
                        order=corrupted_order,
                        parent_scores=parent_scores,  # Keep same scores
                        max_parents=K,
                        k_mb=5,
                        alpha_mb=0.01,
                        verbose=False
                    )
                    ocmb.fit(X, true_adj=true_adj)

                    graph = ocmb.get_adjacency_matrix()
                    metrics = calculate_metrics(true_adj, graph)

                    # Compute parent recall
                    covpa = ocmb.cand_stats_.get('covPa_mean', 0) if ocmb.cand_stats_ else 0

                    print(f"{error_rate:<12.0%} {actual_div:<12.3f} "
                          f"{metrics['F1']:<10.3f} {metrics['SHD']:<10} {covpa:<10.3f}")

                    all_results.append({
                        'seed': seed,
                        'K': K,
                        'error_rate': error_rate,
                        'actual_div': actual_div,
                        'F1': metrics['F1'],
                        'Precision': metrics['Precision'],
                        'Recall': metrics['Recall'],
                        'SHD': metrics['SHD'],
                        'covPa': covpa,
                        'status': 'success'
                    })

                except Exception as e:
                    print(f"{error_rate:<12.0%} {'ERROR':<12} {str(e)}")
                    all_results.append({
                        'seed': seed,
                        'K': K,
                        'error_rate': error_rate,
                        'status': 'failed',
                        'error': str(e)
                    })

    # Aggregate results
    print("\n" + "=" * 80)
    print("SUMMARY: Averaged Results Across Seeds")
    print("=" * 80)

    for K in K_values:
        print(f"\nK={K}")
        print("-" * 70)
        print(f"{'Error Rate':<12} {'F1':<15} {'SHD':<15} {'CovPa':<15} {'Drop%':<10}")
        print("-" * 70)

        baseline_f1 = None

        for error_rate in error_rates:
            filtered = [r for r in all_results
                       if r['K'] == K and r['error_rate'] == error_rate
                       and r['status'] == 'success']

            if not filtered:
                print(f"{error_rate:<12.0%} {'N/A':<15}")
                continue

            mean_f1 = np.mean([r['F1'] for r in filtered])
            std_f1 = np.std([r['F1'] for r in filtered])
            mean_shd = np.mean([r['SHD'] for r in filtered])
            mean_covpa = np.mean([r['covPa'] for r in filtered])

            if baseline_f1 is None:
                baseline_f1 = mean_f1
                drop_pct = 0
            else:
                drop_pct = (baseline_f1 - mean_f1) / baseline_f1 * 100 if baseline_f1 > 0 else 0

            print(f"{error_rate:<12.0%} {mean_f1:.3f}±{std_f1:.3f}   "
                  f"{mean_shd:<15.1f} {mean_covpa:<15.3f} {drop_pct:<10.1f}%")

    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS: Graceful Degradation")
    print("=" * 80)

    # Focus on K=1 (paper's main result)
    k1_results = [r for r in all_results if r['K'] == 1 and r['status'] == 'success']

    if k1_results:
        # Get 0% and 50% error results
        baseline = [r for r in k1_results if r['error_rate'] == 0.0]
        worst = [r for r in k1_results if r['error_rate'] == 0.5]

        if baseline and worst:
            baseline_f1 = np.mean([r['F1'] for r in baseline])
            worst_f1 = np.mean([r['F1'] for r in worst])
            worst_covpa = np.mean([r['covPa'] for r in worst])

            f1_drop = (baseline_f1 - worst_f1) / baseline_f1 * 100

            print(f"\nK=1 results:")
            print(f"  Baseline (0% error): F1 = {baseline_f1:.3f}")
            print(f"  Worst case (50% error): F1 = {worst_f1:.3f}, CovPa = {worst_covpa:.3f}")
            print(f"  F1 drop: {f1_drop:.1f}%")

            if f1_drop < 40:
                print(f"\n✓ GRACEFUL DEGRADATION: F1 only drops {f1_drop:.0f}% at 50% error")
                print(f"  Even with random-like ordering, CovPa stays at {worst_covpa:.2f}")
                print(f"  This confirms ordering constraints help even when imperfect")
            else:
                print(f"\n⚠ Higher degradation than expected (paper shows ~33% drop)")

    # Comparison with paper
    print("\n" + "=" * 80)
    print("COMPARISON WITH PAPER (Figure 10, d=100, K=1)")
    print("=" * 80)
    print(f"{'Error Rate':<12} {'F1 (Paper)':<15} {'F1 (Ours)':<15} {'Status':<10}")
    print("-" * 60)

    paper_values = {
        0.0: 0.84,
        0.5: 0.56
    }

    for error_rate in [0.0, 0.5]:
        filtered = [r for r in all_results
                   if r['K'] == 1 and abs(r['error_rate'] - error_rate) < 0.01
                   and r['status'] == 'success']

        if filtered:
            mean_f1 = np.mean([r['F1'] for r in filtered])
            paper_f1 = paper_values.get(error_rate, 0)
            diff = abs(mean_f1 - paper_f1)
            status = "✓ Close" if diff < 0.15 else "⚠ Different"
            print(f"{error_rate:<12.0%} {paper_f1:.3f}           {mean_f1:.3f}           {status:<10}")

    print("\n" + "=" * 80)
    print("KEY FINDINGS:")
    print("  1. F1 degrades gracefully as ordering errors increase")
    print("  2. At 50% error (random-like), F1 only drops ~33%")
    print("  3. Parent Recall stays reasonable even with corruption")
    print("  4. Smaller K (=1) is more sensitive but still robust")
    print("  5. Larger K (=10) provides more robustness to ordering errors")
    print("\nPractical implication:")
    print("  OCMB doesn't require perfect ordering to be useful.")
    print("  Even approximate orderings provide computational benefits.")
    print("=" * 80)


if __name__ == '__main__':
    main()
