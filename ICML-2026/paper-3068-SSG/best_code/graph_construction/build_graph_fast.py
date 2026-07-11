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


def format_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def build_complete_graph_with_cosine_weights_fast(df_cluster: pd.DataFrame, feature_col: str = "feature"):
    N = len(df_cluster)
    if N == 0:
        raise ValueError("Empty cluster")

    features = np.stack(df_cluster[feature_col].values).astype(np.float32, copy=False)

    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    X = features / norms

    cos_sim = X @ X.T
    np.clip(cos_sim, -1.0, 1.0, out=cos_sim)
    cos_dist = 1.0 - cos_sim

    nodes = []
    for idx, row in enumerate(df_cluster.itertuples(index=False)):
        nodes.append({
            "idx": idx,
            "image_name": getattr(row, "image_name", f"unknown_{idx}"),
            "loss": float(getattr(row, "loss")),
            "true_class": getattr(row, "true_class", None),
            "pred_class": getattr(row, "pred_class", None),
            "correct": bool(getattr(row, "correct", False)),
        })

    iu, ju = np.triu_indices(N, k=1)
    w = cos_dist[iu, ju].astype(np.float32, copy=False)

    edges = list(zip(iu.tolist(), ju.tolist(), w.tolist()))

    row = np.concatenate([iu, ju])
    col = np.concatenate([ju, iu])
    data = np.concatenate([w, w])
    adj_matrix = csr_matrix((data, (row, col)), shape=(N, N))

    avg_loss = float(df_cluster["loss"].mean())
    feature_mean = features.mean(axis=0)
    within_var = float(((features - feature_mean) ** 2).mean())

    return {
        "nodes": nodes,
        "edges": edges,
        "adj_matrix": adj_matrix,
        "num_nodes": N,
        "avg_loss": avg_loss,
        "within_cluster_feature_variance": within_var,
        "cosine_distance_matrix_upper_triangle": w,
    }


def load_cluster_data(class_dir: Path):
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
            clusters.setdefault(cluster_id, []).append(df)
        except Exception as e:
            print(f"Warning: Failed to read {file}: {e}")

    for cid in list(clusters.keys()):
        clusters[cid] = pd.concat(clusters[cid], ignore_index=True)

    return clusters


def save_graph(graph_data, save_path: Path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(graph_data, f, protocol=pickle.HIGHEST_PROTOCOL)


def _process_one_cluster(args):
    """Worker process: build and save a graph; must be defined at module level."""
    class_id, cluster_id, df_cluster, feature_col, output_root = args
    t0 = time.perf_counter()

    graph_data = build_complete_graph_with_cosine_weights_fast(df_cluster, feature_col=feature_col)

    save_dir = Path(output_root) / f"class_{class_id}"
    save_path = save_dir / f"cluster_{cluster_id:04d}.graph.pkl"
    save_graph(graph_data, save_path)

    t1 = time.perf_counter()
    return class_id, cluster_id, graph_data["num_nodes"], (t1 - t0)


def main(input_clustering_root: str, output_graph_root: str, feature_col: str = "feature",
         max_workers: int = None, print_slowest: int = 10):
    input_root = Path(input_clustering_root)
    output_root = Path(output_graph_root)
    output_root.mkdir(parents=True, exist_ok=True)

    class_dirs = [d for d in input_root.iterdir() if d.is_dir() and d.name.startswith("class_")]
    if not class_dirs:
        raise FileNotFoundError(f"No subdirectories starting with 'class_' were found: {input_root}")

    # First scan and build the task list; task granularity is one cluster.
    tasks = []
    class_task_count = {}
    scan_t0 = time.perf_counter()

    for class_dir in tqdm(sorted(class_dirs), desc="loading cluster data"):
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

    # Statistics containers.
    per_class_time = defaultdict(float)
    per_class_done = defaultdict(int)
    per_class_total = class_task_count
    per_task_times = []  # (elapsed, class_id, cluster_id, N)

    # Execute in parallel with a global progress bar.
    run_t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_process_one_cluster, t) for t in tasks]

        pbar = tqdm(total=len(futures), desc="Building graphs (clusters)", dynamic_ncols=True)
        for fut in as_completed(futures):
            class_id, cluster_id, n, elapsed = fut.result()
            pbar.update(1)

            per_class_time[class_id] += elapsed
            per_class_done[class_id] += 1
            per_task_times.append((elapsed, class_id, cluster_id, n))

            # Show per-class completion status for the most recently finished class in the progress-bar postfix.
            pbar.set_postfix_str(
                f"last=class{class_id} c{cluster_id:04d} N={n} "
                f"class_progress={per_class_done[class_id]}/{per_class_total[class_id]}"
            )
        pbar.close()
    run_t1 = time.perf_counter()

    total_elapsed = run_t1 - run_t0
    avg_per_cluster = total_elapsed / len(tasks)

    print("\n===== Summary =====")
    print(f"Total time: {format_duration(total_elapsed)}")
    print(f"Total clusters: {len(tasks)}")
    print(f"Avg time/cluster: {avg_per_cluster:.3f}s")

    # Per-class summary sorted by elapsed time.
    print("\nTop classes by accumulated time:")
    for class_id, tsec in sorted(per_class_time.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  class_{class_id}: {format_duration(tsec)}  "
              f"clusters={per_class_done[class_id]}")

    # Slowest clusters.
    per_task_times.sort(reverse=True, key=lambda x: x[0])
    if print_slowest and print_slowest > 0:
        print(f"\nSlowest {min(print_slowest, len(per_task_times))} clusters:")
        for elapsed, class_id, cluster_id, n in per_task_times[:print_slowest]:
            print(f"  class_{class_id} cluster_{cluster_id:04d} N={n}  {elapsed:.3f}s")

    print(f"\nDone! Graphs saved to: {output_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build fully connected graphs from clustering results with acceleration, progress, and timing")
    parser.add_argument("--input", type=str, required=True, help="clustering-result root directory containing class_* folders")
    parser.add_argument("--output", type=str, required=True, help="graph output root directory")
    parser.add_argument("--feature_col", type=str, default="feature", help="feature column name (default: feature)")
    parser.add_argument("--workers", type=int, default=None, help="number of worker processes (default: min(4, cpu_count))")
    parser.add_argument("--print_slowest", type=int, default=10, help="print the slowest N clusters (default: 10)")

    args = parser.parse_args()

    t0 = time.perf_counter()
    main(args.input, args.output, feature_col=args.feature_col,
         max_workers=args.workers, print_slowest=args.print_slowest)
    t1 = time.perf_counter()
    print(f"\nProgram total time: {format_duration(t1 - t0)}")
