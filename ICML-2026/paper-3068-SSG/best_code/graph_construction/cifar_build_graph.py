"""
CIFAR100 graph-construction script.
Builds fully connected graphs from clustering results, using indices rather than filenames.

Input: clustering-result directory (class_{id}/clusters/cluster_*.parquet).
Output: graph files (class_{id}/cluster_*.graph.pkl).

Usage:
    python cifar_build_graph.py \
        --input /path/to/train_cluster/ \
        --output /path/to/train_graph/ \
        --workers 4
"""

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import csr_matrix
import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from collections import defaultdict
from typing import List, Dict, Any


def format_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def build_complete_graph_with_cosine_weights(
    df_cluster: pd.DataFrame,
    feature_col: str = "feature"
) -> Dict[str, Any]:
    """
    Build a fully connected graph for one cluster.

    Args:
        df_cluster: DataFrame containing all samples in the cluster
        feature_col: feature column name

    Returns:
        graph-data dictionary containing nodes, edges, adj_matrix, and related fields
    """
    N = len(df_cluster)
    if N == 0:
        raise ValueError("Empty cluster")

    # Extract features.
    features = np.stack(df_cluster[feature_col].values).astype(np.float32, copy=False)

    # Normalize features.
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    X = features / norms

    # Compute the cosine-similarity matrix.
    cos_sim = X @ X.T
    np.clip(cos_sim, -1.0, 1.0, out=cos_sim)
    cos_dist = 1.0 - cos_sim

    # Build the node list using indices as identifiers.
    nodes = []
    for idx, row in enumerate(df_cluster.itertuples(index=False)):
        nodes.append({
            "idx": int(getattr(row, "index")),  # original CIFAR100 dataset index
            "label": int(getattr(row, "label")),
        })

    # Build the edge list from the upper triangle.
    iu, ju = np.triu_indices(N, k=1)
    w = cos_dist[iu, ju].astype(np.float32, copy=False)
    edges = list(zip(iu.tolist(), ju.tolist(), w.tolist()))

    # Build the adjacency matrix.
    row = np.concatenate([iu, ju])
    col = np.concatenate([ju, iu])
    data = np.concatenate([w, w])
    adj_matrix = csr_matrix((data, (row, col)), shape=(N, N))

    # Statistics.
    feature_mean = features.mean(axis=0)
    within_var = float(((features - feature_mean) ** 2).mean())

    return {
        "nodes": nodes,
        "edges": edges,
        "adj_matrix": adj_matrix,
        "num_nodes": N,
        "within_cluster_feature_variance": within_var,
        "cosine_distance_matrix_upper_triangle": w,
    }


def load_cluster_data(class_dir: Path) -> Dict[int, pd.DataFrame]:
    """
    Load all cluster data for one class.
    """
    clusters_dir = class_dir / "clusters"
    if not clusters_dir.exists():
        return {}

    parquet_files = sorted(clusters_dir.glob("cluster_*.parquet"))
    clusters = {}

    for file in parquet_files:
        try:
            stem = file.stem
            cluster_id = int(stem.split("_")[1])
            df = pd.read_parquet(file)

            # Convert feature values from lists back to numpy arrays.
            if 'feature' in df.columns:
                df['feature'] = df['feature'].apply(lambda x: np.array(x, dtype=np.float32) if isinstance(x, list) else x)

            clusters.setdefault(cluster_id, []).append(df)
        except Exception as e:
            print(f"Warning: Failed to read {file}: {e}")

    # Merge multiple files for the same cluster.
    for cid in list(clusters.keys()):
        clusters[cid] = pd.concat(clusters[cid], ignore_index=True)

    return clusters


def save_graph(graph_data: Dict, save_path: Path):
    """Save graph data as a pickle file."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(graph_data, f, protocol=pickle.HIGHEST_PROTOCOL)


def _process_one_cluster(args):
    """Worker process: build and save a graph."""
    class_id, cluster_id, df_cluster, feature_col, output_root = args
    t0 = time.perf_counter()

    graph_data = build_complete_graph_with_cosine_weights(df_cluster, feature_col=feature_col)

    save_dir = Path(output_root) / f"class_{class_id}"
    save_path = save_dir / f"cluster_{cluster_id:04d}.graph.pkl"
    save_graph(graph_data, save_path)

    t1 = time.perf_counter()
    return class_id, cluster_id, graph_data["num_nodes"], (t1 - t0)


def main(
    input_clustering_root: str,
    output_graph_root: str,
    feature_col: str = "feature",
    max_workers: int = None,
    print_slowest: int = 10
):
    """
    Main function for building graphs from clustering results.

    Args:
        input_clustering_root: clustering-result root directory
        output_graph_root: graph output root directory
        feature_col: feature column name
        max_workers: maximum number of worker processes
        print_slowest: print the slowest N clusters
    """
    print("=" * 60)
    print("CIFAR100 Graph Building Task")
    print("=" * 60)

    input_root = Path(input_clustering_root)
    output_root = Path(output_graph_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # Scan all class directories.
    class_dirs = [d for d in input_root.iterdir() if d.is_dir() and d.name.startswith("class_")]
    if not class_dirs:
        raise FileNotFoundError(f"No subdirectories starting with 'class_' were found: {input_root}")

    # Build the task list.
    tasks = []
    class_task_count = {}
    scan_t0 = time.perf_counter()

    for class_dir in tqdm(sorted(class_dirs), desc="Loading cluster data"):
        class_id = int(class_dir.name.split("_")[1])
        clusters = load_cluster_data(class_dir)
        if not clusters:
            continue
        class_task_count[class_id] = len(clusters)
        for cluster_id, df_cluster in clusters.items():
            tasks.append((class_id, cluster_id, df_cluster, feature_col, str(output_root)))

    scan_t1 = time.perf_counter()
    if not tasks:
        print("No cluster tasks found.")
        return

    if max_workers is None:
        max_workers = min(4, os.cpu_count() or 1)

    print(f"Scan done: classes={len(class_task_count)}, clusters={len(tasks)} "
          f"(scan time {format_duration(scan_t1 - scan_t0)}), workers={max_workers}")

    # Execute in parallel.
    run_t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_process_one_cluster, t) for t in tasks]

        pbar = tqdm(total=len(futures), desc="Building graphs", dynamic_ncols=True)
        for fut in as_completed(futures):
            class_id, cluster_id, n, elapsed = fut.result()
            pbar.update(1)
            pbar.set_postfix_str(f"last=class{class_id} c{cluster_id:04d} N={n}")
        pbar.close()
    run_t1 = time.perf_counter()

    print("\n===== Summary =====")
    print(f"Total time: {format_duration(run_t1 - run_t0)}")
    print(f"Total clusters: {len(tasks)}")
    print(f"Avg time/cluster: {(run_t1 - run_t0) / len(tasks):.3f}s")
    print(f"\nDone! Graphs saved to: {output_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIFAR100 graph-construction script.")
    parser.add_argument("--input", type=str, required=True, help="clustering-result root directory")
    parser.add_argument("--output", type=str, required=True, help="graph output root directory")
    parser.add_argument("--feature-col", type=str, default="feature", help="feature column name")
    parser.add_argument("--workers", type=int, default=None, help="number of worker processes")
    parser.add_argument("--print-slowest", type=int, default=10, help="print the slowest N clusters")

    args = parser.parse_args()

    t0 = time.perf_counter()
    main(
        input_clustering_root=args.input,
        output_graph_root=args.output,
        feature_col=args.feature_col,
        max_workers=args.workers,
        print_slowest=args.print_slowest,
    )
    t1 = time.perf_counter()
    print(f"\nProgram total time: {format_duration(t1 - t0)}")