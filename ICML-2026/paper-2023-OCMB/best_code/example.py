#!/usr/bin/env python3
"""
OCMB Example Script

This script demonstrates how to use the OCMB library for causal discovery
with different backbones (CaPS and SciNO).
"""

import numpy as np
import sys
import os

# Add OCMB to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from OCMB import (
    run_ocmb,
    OCMB_CaPS,
    OCMB_SciNO,
    OCMB_Random,
    OCMB_Oracle,
    calculate_metrics,
    get_topological_order_from_dag
)


def generate_synthetic_data(n_samples=500, n_nodes=5, seed=42):
    """
    Generate synthetic data with known causal structure.

    Structure: 0 -> 1 -> 3 <- 2 <- 0
                          3 -> 4
    """
    np.random.seed(seed)

    X = np.zeros((n_samples, n_nodes))
    noise = np.random.randn(n_samples, n_nodes) * 0.5

    # Generate data according to causal structure
    X[:, 0] = noise[:, 0]
    X[:, 1] = 0.8 * X[:, 0] + noise[:, 1]
    X[:, 2] = 0.6 * X[:, 0] + noise[:, 2]
    X[:, 3] = 0.5 * X[:, 1] + 0.5 * X[:, 2] + noise[:, 3]
    X[:, 4] = 0.7 * X[:, 3] + noise[:, 4]

    # True adjacency matrix
    true_graph = np.array([
        [0, 1, 1, 0, 0],  # 0 -> 1, 0 -> 2
        [0, 0, 0, 1, 0],  # 1 -> 3
        [0, 0, 0, 1, 0],  # 2 -> 3
        [0, 0, 0, 0, 1],  # 3 -> 4
        [0, 0, 0, 0, 0]
    ])

    return X, true_graph


def example_1_quick_start():
    """Example 1: Quick start with run_ocmb function"""
    print("=" * 80)
    print("Example 1: Quick Start with run_ocmb")
    print("=" * 80)

    # Generate data
    X, true_graph = generate_synthetic_data(n_samples=500, n_nodes=5)
    print(f"Generated data: {X.shape[0]} samples, {X.shape[1]} nodes")
    print(f"True edges: {int(np.sum(true_graph))}")

    # Run OCMB with CaPS backbone
    print("\nRunning OCMB with CaPS backbone...")
    graph, ocmb = run_ocmb(
        X,
        backbone='caps',
        max_parents=10,
        k_mb=5,
        alpha_mb=0.01,
        eta_G=0.001,
        eta_H=0.001,
        device='cuda:0',
        verbose=True
    )

    # Evaluate results
    metrics = calculate_metrics(true_graph, graph)
    print("\n" + "-" * 40)
    print("Results:")
    print("-" * 40)
    print(f"SHD: {metrics['SHD']}")
    print(f"Precision: {metrics['Precision']:.3f}")
    print(f"Recall: {metrics['Recall']:.3f}")
    print(f"F1: {metrics['F1']:.3f}")
    print(f"TP/FP/FN: {metrics['TP']}/{metrics['FP']}/{metrics['FN']}")
    print(f"\nTopological order: {ocmb.order_}")
    print(f"CMI calls: {ocmb.get_n_cmi_calls()}")
    print(f"Total time: {ocmb.get_timings()['total']:.2f}s")
    print()


def example_2_class_interface():
    """Example 2: Using class interface with detailed configuration"""
    print("=" * 80)
    print("Example 2: Class Interface with Detailed Configuration")
    print("=" * 80)

    # Generate data
    X, true_graph = generate_synthetic_data(n_samples=500, n_nodes=5)

    # Create OCMB instance with detailed configuration
    print("\nConfiguring OCMB-CaPS...")
    ocmb = OCMB_CaPS(
        # OCMB parameters
        max_parents=10,
        k_mb=5,
        alpha_mb=0.01,
        symmetry='delete',
        use_spouse_closure=True,
        score_threshold=0.0,
        # CaPS parameters
        eta_G=0.001,
        eta_H=0.001,
        dispersion='mean',
        device='cuda:0',
        # General
        verbose=True
    )

    # Fit the model
    print("\nFitting OCMB-CaPS...")
    ocmb.fit(X, true_adj=true_graph)

    # Get results
    graph = ocmb.get_adjacency_matrix()
    timings = ocmb.get_timings()

    # Evaluate
    metrics = calculate_metrics(true_graph, graph)

    print("\n" + "-" * 40)
    print("Results:")
    print("-" * 40)
    print(f"SHD: {metrics['SHD']}, F1: {metrics['F1']:.3f}")
    print(f"Ordering divergence: {ocmb.ordering_divergence_:.3f}")
    print(f"Kendall tau: {ocmb.ordering_kendall_tau_:.3f}")

    # Candidate coverage statistics
    if ocmb.cand_stats_:
        print(f"\nCandidate Coverage:")
        print(f"  Parent coverage: {ocmb.cand_stats_['covPa_mean']:.3f}")
        print(f"  MB coverage: {ocmb.cand_stats_['covMB_mean']:.3f}")
        print(f"  Avg CandPa size: {ocmb.cand_stats_['avg_CandPa_size']:.1f}")

    print(f"\nTiming Breakdown:")
    print(f"  Ordering: {timings['ordering']:.2f}s")
    print(f"  Candidates: {timings['candidates']:.2f}s")
    print(f"  MB learning: {timings['mb']:.2f}s")
    print(f"  Orientation: {timings['orientation']:.2f}s")
    print(f"  Total: {timings['total']:.2f}s")
    print()


def example_3_compare_backbones():
    """Example 3: Compare different backbones"""
    print("=" * 80)
    print("Example 3: Compare Different Backbones")
    print("=" * 80)

    # Generate data
    X, true_graph = generate_synthetic_data(n_samples=500, n_nodes=5)
    print(f"Data: {X.shape[0]} samples, {X.shape[1]} nodes")
    print(f"True edges: {int(np.sum(true_graph))}\n")

    # Test different backbones
    backbones = ['oracle', 'random', 'caps']
    results = {}

    for backbone in backbones:
        print(f"\nTesting OCMB-{backbone.upper()}...")
        print("-" * 40)

        try:
            graph, ocmb = run_ocmb(
                X,
                backbone=backbone,
                true_adj=true_graph,  # Required for oracle
                max_parents=10,
                k_mb=5,
                alpha_mb=0.01,
                verbose=False
            )

            metrics = calculate_metrics(true_graph, graph)
            results[backbone] = {
                'SHD': metrics['SHD'],
                'F1': metrics['F1'],
                'Precision': metrics['Precision'],
                'Recall': metrics['Recall'],
                'time': ocmb.get_timings()['total'],
                'cmi_calls': ocmb.get_n_cmi_calls(),
                'order_div': ocmb.ordering_divergence_
            }

            print(f"  SHD: {metrics['SHD']}, F1: {metrics['F1']:.3f}")
            print(f"  Precision: {metrics['Precision']:.3f}, Recall: {metrics['Recall']:.3f}")
            print(f"  Time: {ocmb.get_timings()['total']:.2f}s")
            if ocmb.ordering_divergence_ is not None:
                print(f"  Ordering divergence: {ocmb.ordering_divergence_:.3f}")

        except Exception as e:
            print(f"  Error: {e}")
            results[backbone] = None

    # Summary comparison
    print("\n" + "=" * 40)
    print("Summary Comparison")
    print("=" * 40)
    print(f"{'Backbone':<12} {'SHD':>6} {'F1':>8} {'Time(s)':>10} {'CMI Calls':>10}")
    print("-" * 40)
    for backbone, res in results.items():
        if res:
            print(f"{backbone:<12} {res['SHD']:>6} {res['F1']:>8.3f} "
                  f"{res['time']:>10.2f} {res['cmi_calls']:>10}")
    print()


def example_4_parameter_tuning():
    """Example 4: Parameter tuning (K sweep)"""
    print("=" * 80)
    print("Example 4: Parameter Tuning (K-sweep)")
    print("=" * 80)

    # Generate data
    X, true_graph = generate_synthetic_data(n_samples=500, n_nodes=5)
    print(f"Data: {X.shape[0]} samples, {X.shape[1]} nodes\n")

    # First, get ordering from CaPS (do this once)
    print("Step 1: Getting ordering from CaPS...")
    ocmb_caps = OCMB_CaPS(verbose=False)
    ocmb_caps.fit(X)
    order = ocmb_caps.order_
    parent_scores = ocmb_caps.parent_scores_
    print(f"  Ordering obtained: {order}")

    # Now sweep over K values using precomputed ordering
    print("\nStep 2: K-sweep with precomputed ordering...")
    from OCMB import OCMB_Precomputed

    K_values = [3, 5, 7, 10]
    results = []

    for K in K_values:
        ocmb = OCMB_Precomputed(
            order=order,
            parent_scores=parent_scores,
            max_parents=K,
            k_mb=5,
            alpha_mb=0.01,
            verbose=False
        )
        ocmb.fit(X, true_adj=true_graph)

        graph = ocmb.get_adjacency_matrix()
        metrics = calculate_metrics(true_graph, graph)

        results.append({
            'K': K,
            'SHD': metrics['SHD'],
            'F1': metrics['F1'],
            'covPa': ocmb.cand_stats_['covPa_mean'] if ocmb.cand_stats_ else 0,
            'time': ocmb.get_timings()['total']
        })

    # Print results
    print("\n" + "-" * 50)
    print(f"{'K':>5} {'SHD':>6} {'F1':>8} {'CovPa':>10} {'Time(s)':>10}")
    print("-" * 50)
    for r in results:
        print(f"{r['K']:>5} {r['SHD']:>6} {r['F1']:>8.3f} "
              f"{r['covPa']:>10.3f} {r['time']:>10.2f}")
    print()


def main():
    """Run all examples"""
    print("\n" + "=" * 80)
    print("OCMB Examples")
    print("=" * 80 + "\n")

    try:
        # Example 1: Quick start
        example_1_quick_start()
        input("Press Enter to continue to Example 2...")

        # Example 2: Detailed class interface
        example_2_class_interface()
        input("Press Enter to continue to Example 3...")

        # Example 3: Compare backbones
        example_3_compare_backbones()
        input("Press Enter to continue to Example 4...")

        # Example 4: Parameter tuning
        example_4_parameter_tuning()

    except KeyboardInterrupt:
        print("\n\nExamples interrupted by user.")
    except Exception as e:
        print(f"\n\nError running examples: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("Examples completed!")
    print("=" * 80)


if __name__ == '__main__':
    main()
