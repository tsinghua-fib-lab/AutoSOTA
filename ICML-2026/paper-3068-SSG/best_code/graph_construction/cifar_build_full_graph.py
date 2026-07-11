"""
CIFAR100 full-graph construction script without class-wise clustering.
Builds a complete graph directly from embeddings of all samples.

Optimized version:
- Computes edge scores directly without constructing the full adjacency matrix.
- Substantially reduces memory usage and serialization time.

Usage:
    python cifar_build_full_graph.py \
        --input /path/to/train_embeddings.pt \
        --output /path/to/full_graph.pkl \
        --batch-size 5000
"""

import os
import pickle
import numpy as np
import torch
import argparse
import time
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Any


def format_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def load_embeddings(embedding_path: str, verbose: bool = True):
    """
    Load a CIFAR100 embedding file.

    Returns:
        indices: np.ndarray [N]
        labels: np.ndarray [N]
        embeddings: np.ndarray [N, D]
    """
    if verbose:
        print(f"Loading embeddings from {embedding_path}...")

    data = torch.load(embedding_path, map_location='cpu', weights_only=False)

    indices = data['indices'].numpy() if isinstance(data['indices'], torch.Tensor) else data['indices']
    labels = data['labels'].numpy() if isinstance(data['labels'], torch.Tensor) else data['labels']
    embeddings = data['embeddings'].numpy() if isinstance(data['embeddings'], torch.Tensor) else data['embeddings']

    if verbose:
        print(f"Loaded {len(indices)} samples, embedding dim: {embeddings.shape[1]}")
        print(f"Number of classes: {len(np.unique(labels))}")

    return indices, labels, embeddings


def compute_edge_scores_from_embeddings(
    embeddings: np.ndarray,
    indices: np.ndarray,
    batch_size: int = 5000,
    verbose: bool = True
) -> Dict[int, float]:
    """
    Compute each node's average edge-weight score directly from embeddings.
    Avoid building the full adjacency matrix; compute in batches to save memory.

    Args:
        embeddings: [N, D] feature matrix
        indices: [N] sample indices
        batch_size: number of samples per batch
        verbose: whether to show progress

    Returns:
        scores: Dict[idx -> avg_edge_weight]
    """
    N = embeddings.shape[0]

    if verbose:
        print(f"Computing edge scores for {N} samples...")
        print(f"Method: batch computation without full adjacency matrix")

    t0 = time.perf_counter()

    # Normalize features.
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    X = embeddings / norms

    # Accumulate edge-weight sums and edge counts for each node.
    edge_weight_sum = np.zeros(N, dtype=np.float64)
    edge_count = np.zeros(N, dtype=np.int64)

    num_batches = (N + batch_size - 1) // batch_size
    iterator = range(num_batches)
    if verbose:
        iterator = tqdm(iterator, desc="Computing edge scores")

    for batch_idx in iterator:
        start_i = batch_idx * batch_size
        end_i = min((batch_idx + 1) * batch_size, N)

        # Compute distances between the current batch and all samples.
        batch_sim = X[start_i:end_i] @ X.T  # [batch, N]
        np.clip(batch_sim, -1.0, 1.0, out=batch_sim)
        batch_dist = 1.0 - batch_sim  # cosine distance

        # Accumulate edge-weight sums and edge counts for each node.
        for local_i, global_i in enumerate(range(start_i, end_i)):
            # Compute only the upper-triangular entries (global_i < j).
            for j in range(global_i + 1, N):
                weight = float(batch_dist[local_i, j])
                edge_weight_sum[global_i] += weight
                edge_count[global_i] += 1
                # Because the graph is symmetric, also account for this edge at node j.
                edge_weight_sum[j] += weight
                edge_count[j] += 1

    # Compute average edge weights.
    scores = {}
    for i in range(N):
        if edge_count[i] > 0:
            scores[int(indices[i])] = float(edge_weight_sum[i] / edge_count[i])
        else:
            scores[int(indices[i])] = 0.0

    t1 = time.perf_counter()
    if verbose:
        print(f"Edge scores computed in {format_duration(t1 - t0)}")

    return scores


def build_full_graph_optimized(
    indices: np.ndarray,
    labels: np.ndarray,
    embeddings: np.ndarray,
    batch_size: int = 5000,
    save_edges: bool = False,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Build the full graph with an optimized implementation that does not store the full adjacency matrix.

    Args:
        indices: sample indices [N]
        labels: sample labels [N]
        embeddings: feature matrix [N, D]
        batch_size: batch size for computation
        save_edges: whether to save the edge list; usually unnecessary
        verbose: whether to show progress

    Returns:
        graph_data: graph-data dictionary containing nodes, edge_scores, num_nodes, and related fields
    """
    N = len(indices)
    D = embeddings.shape[1]

    if verbose:
        print(f"Building optimized full graph for {N} samples...")
        print(f"NOT storing full adjacency matrix - saving memory and time")

    t_total = time.perf_counter()

    # 1. Normalize features.
    t0 = time.perf_counter()
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    X = embeddings / norms

    feature_mean = X.mean(axis=0)
    within_var = float(((X - feature_mean) ** 2).mean())

    if verbose:
        t1 = time.perf_counter()
        print(f"Step 1: Normalization done in {format_duration(t1 - t0)}")

    # 2. Build the node list.
    t0 = time.perf_counter()
    nodes = []
    for i in range(N):
        nodes.append({
            "idx": int(indices[i]),
            "label": int(labels[i]),
        })

    if verbose:
        t1 = time.perf_counter()
        print(f"Step 2: Built {N} nodes in {format_duration(t1 - t0)}")

    # 3. Compute edge weights by accumulation without materializing the full matrix.
    t0 = time.perf_counter()
    edge_weight_sum = np.zeros(N, dtype=np.float64)
    edge_count = np.zeros(N, dtype=np.int64)

    # If edge-list saving is requested.
    edges = [] if save_edges else None
    all_weights = []

    num_batches = (N + batch_size - 1) // batch_size
    iterator = range(num_batches)
    if verbose:
        iterator = tqdm(iterator, desc="Step 3: Computing edge weights")

    for batch_idx in iterator:
        start_i = batch_idx * batch_size
        end_i = min((batch_idx + 1) * batch_size, N)

        # Compute distances between the current batch and all samples.
        batch_sim = X[start_i:end_i] @ X.T  # [batch, N]
        np.clip(batch_sim, -1.0, 1.0, out=batch_sim)
        batch_dist = 1.0 - batch_sim

        # Accumulate edge-weight sums and edge counts for each node.
        for local_i, global_i in enumerate(range(start_i, end_i)):
            for j in range(global_i + 1, N):  # process only the upper triangle
                weight = float(batch_dist[local_i, j])

                # Accumulate edge weights.
                edge_weight_sum[global_i] += weight
                edge_count[global_i] += 1
                edge_weight_sum[j] += weight
                edge_count[j] += 1

                # Optionally save the edge list.
                if save_edges:
                    edges.append((global_i, j, weight))
                    all_weights.append(weight)

    if verbose:
        t1 = time.perf_counter()
        print(f"Step 3: Edge weights computed in {format_duration(t1 - t0)}")
        total_edges = N * (N - 1) // 2
        print(f"Total edges (in full graph): {total_edges}")

    # 4. Compute average edge weights for each node
    t0 = time.perf_counter()
    edge_scores = {}
    for i in range(N):
        if edge_count[i] > 0:
            edge_scores[int(indices[i])] = float(edge_weight_sum[i] / edge_count[i])
        else:
            edge_scores[int(indices[i])] = 0.0

    if verbose:
        t1 = time.perf_counter()
        print(f"Step 4: Edge scores computed in {format_duration(t1 - t0)}")

    # 5. Build the return dictionary without adj_matrix.
    graph_data = {
        "nodes": nodes,
        "num_nodes": N,
        "within_cluster_feature_variance": within_var,
        "edge_scores": edge_scores,  # precomputed edge scores
    }

    # Optionally save the edge list; usually unnecessary and increases file size.
    if save_edges:
        graph_data["edges"] = edges
        graph_data["cosine_distance_matrix_upper_triangle"] = np.array(all_weights, dtype=np.float32)

    # Note: adj_matrix is not stored.

    t_end = time.perf_counter()
    if verbose:
        print(f"\nTotal graph building time: {format_duration(t_end - t_total)}")

    return graph_data


def main(
    input_path: str,
    output_path: str,
    batch_size: int = 5000,
    save_edges: bool = False,
):
    """
    Main function.

    Args:
        input_path: embedding .pt file path
        output_path: output graph.pkl file path
        batch_size: batch size for computation
        save_edges: whether to save the edge list; increases file size
    """
    print("=" * 60)
    print("CIFAR100 Full Graph Building (Optimized - No Adj Matrix)")
    print("=" * 60)

    t_total = time.perf_counter()

    # 1. Load embeddings.
    t0 = time.perf_counter()
    indices, labels, embeddings = load_embeddings(input_path)
    t1 = time.perf_counter()
    print(f"[Time] Loading embeddings: {format_duration(t1 - t0)}")

    # 2. Build the full graph with the optimized implementation.
    t0 = time.perf_counter()
    graph_data = build_full_graph_optimized(
        indices=indices,
        labels=labels,
        embeddings=embeddings,
        batch_size=batch_size,
        save_edges=save_edges,
        verbose=True
    )
    t1 = time.perf_counter()
    print(f"[Time] Building graph: {format_duration(t1 - t0)}")

    # 3. Save graph data.
    t0 = time.perf_counter()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving graph to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(graph_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    t1 = time.perf_counter()
    print(f"[Time] Saving graph: {format_duration(t1 - t0)}")

    # Print summary statistics.
    print("\n===== Summary =====")
    print(f"Total nodes: {graph_data['num_nodes']}")
    print(f"Edge scores computed: {len(graph_data['edge_scores'])}")
    print(f"Output file: {output_path}")

    # Check file size.
    file_size = output_path.stat().st_size / 1024**2  # MB
    print(f"File size: {file_size:.2f} MB")

    t_end = time.perf_counter()
    print(f"\n[Time] TOTAL: {format_duration(t_end - t_total)}")
    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIFAR100 full-graph construction (optimized; no adjacency matrix stored)")
    parser.add_argument("--input", type=str, required=True, help="embedding .pt file path")
    parser.add_argument("--output", type=str, required=True, help="output graph.pkl file path")
    parser.add_argument("--batch-size", type=int, default=5000, help="batch size for computation")
    parser.add_argument("--save-edges", action="store_true", help="Save the edge list; increases file size")

    args = parser.parse_args()

    main(
        input_path=args.input,
        output_path=args.output,
        batch_size=args.batch_size,
        save_edges=args.save_edges,
    )