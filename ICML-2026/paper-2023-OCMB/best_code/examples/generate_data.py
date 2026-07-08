#!/usr/bin/env python3
"""
Data Generation Utility

Generates synthetic datasets for OCMB experiments.
This utility helps users create their own test datasets with known ground truth.
"""

import numpy as np
import sys
import os
from pathlib import Path
import argparse
import json

SCRIPT_DIR = Path(__file__).resolve().parent
OCMB_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(OCMB_ROOT))


def generate_erdos_renyi_graph(d, degree=3, seed=42):
    """Generate Erdős-Rényi DAG"""
    np.random.seed(seed)
    p = degree / (d - 1)
    adj = np.zeros((d, d))
    for i in range(d):
        for j in range(i + 1, d):
            if np.random.rand() < p:
                adj[i, j] = 1
    return adj


def generate_scale_free_graph(d, degree=3, seed=42):
    """Generate scale-free DAG using preferential attachment"""
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
        print("Warning: networkx not available, using ER graph instead")
        return generate_erdos_renyi_graph(d, degree, seed)


def generate_data_from_graph(adj, n_samples=1000, linear_ratio=0.5,
                             noise_type='gauss', seed=42):
    """
    Generate observational data from DAG structure

    Args:
        adj: Adjacency matrix (d x d)
        n_samples: Number of samples
        linear_ratio: Proportion of linear relationships (0-1)
        noise_type: 'gauss', 'laplace', or 'uniform'
        seed: Random seed

    Returns:
        X: Data matrix (n_samples x d)
    """
    np.random.seed(seed)
    d = adj.shape[0]
    X = np.zeros((n_samples, d))

    # Generate in topological order
    for j in range(d):
        parents = np.where(adj[:, j] == 1)[0]

        if len(parents) == 0:
            # Root node
            if noise_type == 'gauss':
                X[:, j] = np.random.randn(n_samples)
            elif noise_type == 'laplace':
                X[:, j] = np.random.laplace(0, 1, n_samples)
            else:  # uniform
                X[:, j] = np.random.uniform(-np.sqrt(3), np.sqrt(3), n_samples)
        else:
            # Non-root node
            z = np.zeros(n_samples)
            for p in parents:
                weight = np.random.uniform(0.5, 2.0) * np.random.choice([-1, 1])

                if np.random.rand() < linear_ratio:
                    # Linear mechanism
                    z += weight * X[:, p]
                else:
                    # Nonlinear mechanism (tanh)
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


def save_dataset(X, adj, output_dir, name, metadata=None):
    """Save dataset to disk"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save data
    np.save(output_dir / f"{name}_data.npy", X)
    np.save(output_dir / f"{name}_adj.npy", adj)

    # Save metadata
    if metadata is None:
        metadata = {}

    metadata.update({
        'name': name,
        'n_samples': X.shape[0],
        'n_nodes': X.shape[1],
        'n_edges': int(np.sum(adj))
    })

    with open(output_dir / f"{name}_meta.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved dataset to {output_dir}/")
    print(f"  - {name}_data.npy: {X.shape}")
    print(f"  - {name}_adj.npy: {adj.shape}")
    print(f"  - {name}_meta.json")


def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic datasets for OCMB experiments'
    )
    parser.add_argument('--graph-type', type=str, default='scale-free',
                       choices=['scale-free', 'erdos-renyi', 'er'],
                       help='Graph type')
    parser.add_argument('-d', '--nodes', type=int, default=100,
                       help='Number of nodes')
    parser.add_argument('-n', '--samples', type=int, default=1000,
                       help='Number of samples')
    parser.add_argument('--degree', type=int, default=3,
                       help='Average degree')
    parser.add_argument('--linear-ratio', type=float, default=0.5,
                       help='Proportion of linear relationships (0-1)')
    parser.add_argument('--noise-type', type=str, default='gauss',
                       choices=['gauss', 'laplace', 'uniform'],
                       help='Noise distribution')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--output-dir', type=str, default='./data',
                       help='Output directory')
    parser.add_argument('--name', type=str, default=None,
                       help='Dataset name (auto-generated if not provided)')

    args = parser.parse_args()

    print("=" * 60)
    print("OCMB Data Generator")
    print("=" * 60)
    print(f"Graph type: {args.graph_type}")
    print(f"Nodes: {args.nodes}")
    print(f"Samples: {args.samples}")
    print(f"Average degree: {args.degree}")
    print(f"Linear ratio: {args.linear_ratio}")
    print(f"Noise type: {args.noise_type}")
    print(f"Seed: {args.seed}")
    print()

    # Generate graph
    print("Generating graph...", end=' ')
    if args.graph_type in ['scale-free', 'sf']:
        adj = generate_scale_free_graph(args.nodes, args.degree, args.seed)
        graph_type = 'scale-free'
    else:
        adj = generate_erdos_renyi_graph(args.nodes, args.degree, args.seed)
        graph_type = 'erdos-renyi'

    n_edges = int(np.sum(adj))
    print(f"✓ ({n_edges} edges)")

    # Generate data
    print("Generating data...", end=' ')
    X = generate_data_from_graph(
        adj,
        n_samples=args.samples,
        linear_ratio=args.linear_ratio,
        noise_type=args.noise_type,
        seed=args.seed
    )
    print(f"✓ ({X.shape})")

    # Auto-generate name if not provided
    if args.name is None:
        name = f"{graph_type}_d{args.nodes}_n{args.samples}_seed{args.seed}"
    else:
        name = args.name

    # Save dataset
    metadata = {
        'graph_type': graph_type,
        'degree': args.degree,
        'linear_ratio': args.linear_ratio,
        'noise_type': args.noise_type,
        'seed': args.seed
    }

    save_dataset(X, adj, args.output_dir, name, metadata)

    print("\n" + "=" * 60)
    print("Dataset generation complete!")
    print("\nTo use this dataset in Python:")
    print(f"  import numpy as np")
    print(f"  X = np.load('{args.output_dir}/{name}_data.npy')")
    print(f"  true_adj = np.load('{args.output_dir}/{name}_adj.npy')")
    print("=" * 60)


if __name__ == '__main__':
    main()
