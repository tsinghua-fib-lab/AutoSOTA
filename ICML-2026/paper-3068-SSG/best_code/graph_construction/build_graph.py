import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import argparse
import time



def build_complete_graph_with_cosine_weights(df_cluster: pd.DataFrame, feature_col: str = 'feature'):
    """
    Build a fully connected graph from a cluster DataFrame.

    Args:
        df_cluster: DataFrame containing samples from the same cluster; must include 'loss', 'image_name', and feature_col
        feature_col: feature column name; defaults to 'feature'

    Returns:
        graph_dict: contains nodes, edges, adjacency matrix, and related metadata
    """
    N = len(df_cluster)
    if N == 0:
        raise ValueError("Empty cluster")

    # Extract and stack features.
    try:
        features = np.stack(df_cluster[feature_col].values)  # shape: (N, D)
    except Exception as e:
        print(f"Error: Failed to stack features: {e}")
        raise

    # Compute the cosine-similarity matrix.
    cos_sim = cosine_similarity(features)  # (N, N)
    cos_dist = 1 - cos_sim  # cosine distance used as edge weight

    # Build the node list.
    nodes = []
    for idx, (_, row) in enumerate(df_cluster.iterrows()):
        nodes.append({
            'idx': idx,
            'image_name': row.get('image_name', f'unknown_{idx}'),
            'loss': float(row['loss']),
            'true_class': row.get('true_class'),
            'pred_class': row.get('pred_class'),
            'correct': bool(row.get('correct')),
        })

    # Build the edge list from the upper triangle to avoid duplicates.
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            weight = float(cos_dist[i, j])
            edges.append((i, j, weight))

    # Sparse adjacency matrix.
    row = [e[0] for e in edges]
    col = [e[1] for e in edges]
    data = [e[2] for e in edges]
    adj_matrix = csr_matrix((data + data, (row + col, col + row)), shape=(N, N))  # symmetrized

    # Statistics.
    avg_loss = df_cluster['loss'].mean()
    feature_mean = features.mean(axis=0)
    within_var = ((features - feature_mean) ** 2).mean()

    return {
        'nodes': nodes,
        'edges': edges,
        'adj_matrix': adj_matrix,
        'num_nodes': N,
        'avg_loss': avg_loss,
        'within_cluster_feature_variance': float(within_var),
        'cosine_distance_matrix_upper_triangle': cos_dist[np.triu_indices(N, k=1)],  # optional statistic
    }


def load_cluster_data(class_dir: Path):
    """
    Load all cluster parquet files under one class directory.
    Returns: dict[cluster_id, DataFrame]
    """
    clusters_dir = class_dir / "clusters"
    if not clusters_dir.exists():
        return {}

    parquet_files = sorted(clusters_dir.glob("cluster_*.parquet"))
    clusters = {}

    for file in parquet_files:
        try:
            # Parse cluster ID.
            stem = file.stem  # e.g., "cluster_0003_00000"
            cluster_id = int(stem.split('_')[1])  # the cluster label is the component after the first underscore
            df = pd.read_parquet(file)

            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(df)

        except Exception as e:
            print(f"Warning: Failed to read {file}: {e}")

    # Merge chunks for each cluster.
    for cid in clusters:
        clusters[cid] = pd.concat(clusters[cid], ignore_index=True)

    return clusters


def save_graph(graph_data, save_path: Path):
    """Save graph data safely."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(graph_data, f)
    print(f"Graph saved: {save_path} (nodes: {graph_data['num_nodes']})")


def main(input_clustering_root: str, output_graph_root: str, feature_col: str = 'feature'):
    input_root = Path(input_clustering_root)
    output_root = Path(output_graph_root)
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Scanning clustering-result directory: {input_root}")

    class_dirs = [d for d in input_root.iterdir() if d.is_dir() and d.name.startswith("class_")]

    if not class_dirs:
        raise FileNotFoundError(f"No subdirectories starting with 'class_' were found: {input_root}")

    for class_dir in sorted(class_dirs):
        class_id = class_dir.name.split("_")[1]
        print(f"\nProcessing class: {class_id}")

        # Step 1: Load all cluster data for this class.
        clusters = load_cluster_data(class_dir)
        if not clusters:
            print(f"   Warning: No cluster data was loaded.")
            continue

        # Step 2: Build a graph for each cluster.
        for cluster_id, df_cluster in clusters.items():
            print(f"   Building graph for cluster {cluster_id} (samples: {len(df_cluster)})")
            try:
                graph_data = build_complete_graph_with_cosine_weights(df_cluster, feature_col=feature_col)

                # Save path: output_root/class_{id}/cluster_{id}.graph.pkl
                save_dir = output_root / f"class_{class_id}"
                save_path = save_dir / f"cluster_{cluster_id:04d}.graph.pkl"

                save_graph(graph_data, save_path)

            except Exception as e:
                print(f"   Error: Building graph for cluster {cluster_id} failed: {e}")
                continue

    print(f"\nAll graphs have been built. Results saved to: {output_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build fully connected graphs from clustering results")
    parser.add_argument("--input", type=str, required=True, help="clustering-result root directory containing class_* folders")
    parser.add_argument("--output", type=str, required=True, help="graph output root directory")
    parser.add_argument("--feature_col", type=str, default="feature", help="feature column name (default: 'feature')")

    args = parser.parse_args()
    t0 = time.perf_counter()
    main(
        input_clustering_root=args.input,
        output_graph_root=args.output,
        feature_col=args.feature_col
    )
    t1 = time.perf_counter()
    print(f"Total inference time: {format_duration(t1 - t0)}")