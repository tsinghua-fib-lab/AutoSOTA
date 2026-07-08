#!/usr/bin/env python3
"""
Example 1: Synthetic Graphs Experiment (Table 2 from paper)

Reproduces the main comparison on scale-free and Erdős-Rényi graphs.
This demonstrates OCMB's performance compared to PC and IAMB on synthetic data.

Expected results (d=100, n=1000):
- OCMB(caps): F1 ≈ 0.56, SHD ≈ 32, #CI Tests ≈ 1,980
- IAMB: F1 ≈ 0.48, SHD ≈ 49, #CI Tests ≈ 5,699
- PC: F1 ≈ 0.66, SHD ≈ 16, #CI Tests ≈ 2,072 (on ER graphs)

Note: PC performs better on ER graphs (homogeneous) but worse on scale-free.
OCMB is more robust across graph types.
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
        OCMB_Random,
        calculate_metrics
    )
except ImportError:
    print("Error: Could not import OCMB. Make sure OCMB is in your Python path.")
    print(f"Try: export PYTHONPATH={OCMB_ROOT}:$PYTHONPATH")
    sys.exit(1)

# Try to import synthetic data generator
try:
    sys.path.insert(0, str(OCMB_ROOT.parent / 'src'))
    from datasets.synthetic_generator import generate_synthetic_data
    HAS_GENERATOR = True
except ImportError:
    HAS_GENERATOR = False
    print("Warning: Could not import synthetic_generator. Using fallback generator.")


def generate_erdos_renyi_graph(d, degree=3, seed=42):
    """Generate Erdős-Rényi graph"""
    np.random.seed(seed)
    p = degree / (d - 1)  # Connection probability
    adj = np.zeros((d, d))

    for i in range(d):
        for j in range(i + 1, d):
            if np.random.rand() < p:
                adj[i, j] = 1

    return adj


def generate_scale_free_graph(d, degree=3, seed=42):
    """Generate scale-free graph using preferential attachment"""
    try:
        import networkx as nx
        np.random.seed(seed)

        # Use Barabási-Albert model
        m = max(1, degree // 2)
        G = nx.barabasi_albert_graph(d, m, seed=seed)

        # Convert to DAG by keeping only edges i -> j where i < j
        adj = np.zeros((d, d))
        for i, j in G.edges():
            if i < j:
                adj[i, j] = 1
            else:
                adj[j, i] = 1

        return adj
    except ImportError:
        print("Warning: networkx not found, using ER graph instead")
        return generate_erdos_renyi_graph(d, degree, seed)


def generate_data_from_graph(adj, n_samples=1000, linear_ratio=0.5, noise_type='gauss', seed=42):
    """Generate observational data from graph structure"""
    np.random.seed(seed)
    d = adj.shape[0]
    X = np.zeros((n_samples, d))

    # Generate in topological order
    for j in range(d):
        parents = np.where(adj[:, j] == 1)[0]

        if len(parents) == 0:
            # Root node
            X[:, j] = np.random.randn(n_samples)
        else:
            # Non-root node
            z = np.zeros(n_samples)
            for p in parents:
                weight = np.random.uniform(0.5, 2.0) * np.random.choice([-1, 1])
                if np.random.rand() < linear_ratio:
                    # Linear
                    z += weight * X[:, p]
                else:
                    # Nonlinear
                    z += weight * np.tanh(X[:, p])

            # Add noise
            if noise_type == 'gauss':
                noise = np.random.randn(n_samples)
            elif noise_type == 'laplace':
                noise = np.random.laplace(0, 1, n_samples)
            else:  # uniform
                noise = np.random.uniform(-np.sqrt(3), np.sqrt(3), n_samples)

            X[:, j] = z + noise

    # Standardize
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    return X


def run_experiment(X, true_adj, method='ocmb-caps', max_parents=10, verbose=False):
    """Run a single experiment with given method"""
    result = {
        'method': method,
        'status': 'pending'
    }

    try:
        start_time = time.time()

        if method == 'ocmb-caps':
            ocmb = OCMB_CaPS(
                max_parents=max_parents,
                k_mb=5,
                alpha_mb=0.01,
                eta_G=0.001,
                eta_H=0.001,
                device='cuda:0' if is_gpu_available() else 'cpu',
                verbose=verbose
            )
            ocmb.fit(X, true_adj=true_adj)
            graph = ocmb.get_adjacency_matrix()
            n_cmi_calls = ocmb.get_n_cmi_calls()

        elif method == 'ocmb-random':
            ocmb = OCMB_Random(
                max_parents=max_parents,
                k_mb=5,
                alpha_mb=0.01,
                verbose=verbose
            )
            ocmb.fit(X, true_adj=true_adj)
            graph = ocmb.get_adjacency_matrix()
            n_cmi_calls = ocmb.get_n_cmi_calls()

        else:
            raise ValueError(f"Unknown method: {method}")

        elapsed = time.time() - start_time
        metrics = calculate_metrics(true_adj, graph)

        result.update({
            'status': 'success',
            'F1': metrics['F1'],
            'Precision': metrics['Precision'],
            'Recall': metrics['Recall'],
            'SHD': metrics['SHD'],
            'TP': metrics['TP'],
            'FP': metrics['FP'],
            'FN': metrics['FN'],
            'n_cmi_calls': n_cmi_calls,
            'time': elapsed
        })

    except Exception as e:
        result['status'] = 'failed'
        result['error'] = str(e)
        if verbose:
            import traceback
            traceback.print_exc()

    return result


def is_gpu_available():
    """Check if GPU is available"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def main():
    print("=" * 80)
    print("OCMB Example 1: Synthetic Graphs Experiment")
    print("Reproducing Table 2 from the paper")
    print("=" * 80)
    print()

    # Configuration
    d = 100  # Number of nodes
    n = 1000  # Number of samples
    degree = 3  # Average degree
    seeds = [42, 123, 456]  # Multiple seeds for robustness
    graph_types = ['scale-free', 'erdos-renyi']
    methods = ['ocmb-caps', 'ocmb-random']

    print(f"Configuration:")
    print(f"  Nodes (d): {d}")
    print(f"  Samples (n): {n}")
    print(f"  Average degree: {degree}")
    print(f"  Seeds: {seeds}")
    print(f"  Graph types: {graph_types}")
    print(f"  Methods: {methods}")
    print(f"  GPU available: {is_gpu_available()}")
    print()

    # Results storage
    all_results = []

    # Run experiments
    for graph_type in graph_types:
        print(f"\n{'='*60}")
        print(f"Graph Type: {graph_type.upper()}")
        print(f"{'='*60}")

        for seed_idx, seed in enumerate(seeds, 1):
            print(f"\n--- Seed {seed} ({seed_idx}/{len(seeds)}) ---")

            # Generate graph
            if graph_type == 'scale-free':
                true_adj = generate_scale_free_graph(d, degree, seed)
            else:
                true_adj = generate_erdos_renyi_graph(d, degree, seed)

            n_edges = int(np.sum(true_adj))
            print(f"Generated {graph_type} graph: {d} nodes, {n_edges} edges")

            # Generate data
            X = generate_data_from_graph(true_adj, n, linear_ratio=0.5, seed=seed)
            print(f"Generated data: {X.shape}")

            # Run each method
            for method in methods:
                print(f"\n  Running {method}...", end=' ')
                result = run_experiment(X, true_adj, method=method, verbose=False)

                if result['status'] == 'success':
                    print(f"✓ F1={result['F1']:.3f}, SHD={result['SHD']}, "
                          f"CI={result['n_cmi_calls']}, Time={result['time']:.1f}s")
                else:
                    print(f"✗ {result.get('error', 'Unknown error')}")

                result['graph_type'] = graph_type
                result['seed'] = seed
                result['d'] = d
                result['n'] = n
                all_results.append(result)

    # Aggregate results
    print("\n" + "=" * 80)
    print("SUMMARY: Averaged Results Across Seeds")
    print("=" * 80)

    for graph_type in graph_types:
        print(f"\n{graph_type.upper()}")
        print("-" * 80)
        print(f"{'Method':<20} {'F1':<10} {'SHD':<10} {'#CI Tests':<12} {'Time (s)':<10}")
        print("-" * 80)

        for method in methods:
            # Filter results
            filtered = [r for r in all_results
                       if r['graph_type'] == graph_type
                       and r['method'] == method
                       and r['status'] == 'success']

            if not filtered:
                print(f"{method:<20} {'N/A':<10} {'N/A':<10} {'N/A':<12} {'N/A':<10}")
                continue

            # Compute means
            mean_f1 = np.mean([r['F1'] for r in filtered])
            mean_shd = np.mean([r['SHD'] for r in filtered])
            mean_ci = np.mean([r['n_cmi_calls'] for r in filtered])
            mean_time = np.mean([r['time'] for r in filtered])
            std_f1 = np.std([r['F1'] for r in filtered])
            std_shd = np.std([r['SHD'] for r in filtered])

            print(f"{method:<20} {mean_f1:.3f}±{std_f1:.3f}  "
                  f"{mean_shd:.1f}±{std_shd:.1f}  "
                  f"{mean_ci:<12.0f} {mean_time:<10.1f}")

    # Expected values comparison
    print("\n" + "=" * 80)
    print("COMPARISON WITH PAPER (Table 2, d=100, n=1000, Scale-Free)")
    print("=" * 80)
    print(f"{'Method':<20} {'F1 (Paper)':<15} {'F1 (Ours)':<15} {'Status':<10}")
    print("-" * 80)

    expected = {
        'ocmb-caps': 0.558,
        'ocmb-random': 0.244,  # CaPS-alone from Table 13
    }

    for method in methods:
        filtered = [r for r in all_results
                   if r['graph_type'] == 'scale-free'
                   and r['method'] == method
                   and r['status'] == 'success']

        if filtered:
            mean_f1 = np.mean([r['F1'] for r in filtered])
            expected_f1 = expected.get(method, 0.0)
            diff = abs(mean_f1 - expected_f1)
            status = "✓ Close" if diff < 0.1 else "⚠ Different"
            print(f"{method:<20} {expected_f1:.3f}           {mean_f1:.3f}           {status:<10}")

    print("\n" + "=" * 80)
    print("Notes:")
    print("  - OCMB-CaPS uses learned ordering from CaPS neural network")
    print("  - OCMB-Random uses random ordering (sanity check baseline)")
    print("  - Expected: OCMB-CaPS should significantly outperform OCMB-Random")
    print("  - On scale-free graphs, OCMB should get F1 ≈ 0.55-0.65")
    print("  - Some variance is expected due to random seeds and data generation")
    print("=" * 80)


if __name__ == '__main__':
    main()
