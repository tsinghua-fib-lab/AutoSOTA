import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import faiss
from tqdm import tqdm
import time

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Tuple
from typing import List, Optional
from multiprocessing import Pool, current_process

# ========================================
# Utility functions for reading all .pkl files in a directory.
# ========================================

def process_one_class(
    class_id: int,
    df_class: pd.DataFrame,
    feature_column: str,
    n_clusters_per_class: int,
    max_samples_per_class: int,
    kmeans_iter: int,
    use_gpu: bool,
    output_base_dir: str,
    max_rows_per_file: int,
    input_pkl_dir: str,
):
    """
    Worker process: cluster one class and save the results.
    Note: df_class is passed through pickle; pass only necessary columns and keep it reasonably small.
    """
    try:
        N = len(df_class)
        if max_samples_per_class and N > max_samples_per_class:
            df_class = df_class.sample(n=max_samples_per_class, random_state=42)
            N = len(df_class)

        if N < n_clusters_per_class:
            return (class_id, False, f"skip: N={N} < k={n_clusters_per_class}")

        features = np.stack(df_class[feature_column].values)
        if features.ndim != 2:
            return (class_id, False, f"bad feature shape: {features.shape}")

        centroids, labels = faiss_kmeans(
            data=features,
            k=n_clusters_per_class,
            niter=kmeans_iter,
            gpu=use_gpu,
            seed=42
        )

        save_per_class_clustering_result(
            df_class=df_class,
            class_id=class_id,
            labels=labels,
            centroids=centroids,
            output_base_dir=output_base_dir,
            max_rows_per_file=max_rows_per_file,
            extra_metadata={
                "source": input_pkl_dir,
                "feature_column": feature_column,
                "n_clusters": n_clusters_per_class,
                "total_samples": len(df_class),
                "timestamp": pd.Timestamp.now().isoformat()
            }
        )
        return (class_id, True, "ok")

    except Exception as e:
        return (class_id, False, f"error: {e}")

def read_all_pkl_files(folder_path: str, columns: List[str] = None, verbose: bool = True) -> pd.DataFrame:
    """
    Read all .pkl files in a directory and merge them into one DataFrame.

    Args:
        folder_path: directory containing .pkl files
        columns: optional columns to retain to reduce memory usage
        verbose: whether to print information

    Returns:
        pd.DataFrame
    """
    folder_path = Path(folder_path)
    if not folder_path.exists():
        raise FileNotFoundError(f"Path does not exist: {folder_path}")

    pkl_files = sorted([f for f in folder_path.glob("*.pkl") if f.is_file()])
    if not pkl_files:
        raise FileNotFoundError(f"No .pkl files found: {folder_path}")

    dfs = []
    for file in tqdm(pkl_files, desc="Loading PKL files"):
        try:
            with open(file, 'rb') as f:
                data = pickle.load(f)
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, pd.DataFrame):
                df = data
            else:
                # Try to construct a DataFrame from a dictionary.
                try:
                    df = pd.DataFrame(data)
                except Exception:
                    continue

            if columns:
                df = df[[c for c in columns if c in df.columns]]
            dfs.append(df)
        except Exception as e:
            print(f"Error: Failed to read {file}: {e}")

    df_all = pd.concat(dfs, ignore_index=True)
    if verbose:
        print(f"Loaded {len(df_all)} rows from {len(pkl_files)} files")
    return df_all


# # Worker function for reading one file; must be defined at module level.
# def _load_one_pkl_process(file_str: str, columns=None):
#     file = Path(file_str)
#     with open(file, "rb") as f:
#         data = pickle.load(f)

#     if isinstance(data, list):
#         df = pd.DataFrame(data)
#     elif isinstance(data, pd.DataFrame):
#         df = data
#     else:
#         return None

#     if columns:
#         keep = [c for c in columns if c in df.columns]
#         df = df[keep]
#     return df


# def read_all_pkl_files(folder_path, columns=None, verbose=True, max_workers=None):
#     """
#     Read all .pkl files in a directory with multiprocessing and merge them into one DataFrame.
#     """
#     folder_path = Path(folder_path)
#     if not folder_path.exists():
#         raise FileNotFoundError(f"Path does not exist: {folder_path}")

#     pkl_files = sorted([f for f in folder_path.glob("*.pkl") if f.is_file()])
#     if not pkl_files:
#         raise FileNotFoundError(f"No .pkl files found: {folder_path}")

#     # Default worker count: keep it modest to avoid overloading shared storage or memory.
#     if max_workers is None:
#         max_workers = min(8, len(pkl_files))  # can also be changed to os.cpu_count()
#     else:
#         max_workers = min(max_workers, len(pkl_files))

#     dfs = []
#     errors = 0

#     with ProcessPoolExecutor(max_workers=max_workers) as ex:
#         futures = {
#             ex.submit(_load_one_pkl_process, str(f), columns): f
#             for f in pkl_files
#         }

#         for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Loading PKL files (mp x{max_workers})"):
#             f = futures[fut]
#             try:
#                 df = fut.result()
#                 if df is not None:
#                     dfs.append(df)
#             except Exception as e:
#                 errors += 1
#                 print(f"Error: Failed to read {f}: {e}")

#     if not dfs:
#         raise RuntimeError("All PKL files failed to load; dfs is empty.")

#     df_all = pd.concat(dfs, ignore_index=True)
#     if verbose:
#         print(f"Loaded {len(df_all)} rows from {len(pkl_files)} files (failed {errors}), workers={max_workers}")
#     return df_all


# ========================================
# Faiss clustering function used within each class.
# ========================================
def faiss_kmeans(data, k, niter=20, gpu=False, seed=42):
    """
    Run KMeans clustering on the input data.
    """
    data = data.astype(np.float32)
    n_samples, dim = data.shape

    kmeans = faiss.Kmeans(
        d=dim,
        k=k,
        niter=niter,
        nredo=1,
        spherical=False,
        verbose=True,
        gpu=gpu,
        seed=seed
    )
    kmeans.train(data)

    centroids = kmeans.centroids.copy()

    index = faiss.IndexFlatL2(dim)
    if gpu:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    index.add(centroids)
    _, labels = index.search(data, 1)
    labels = labels.reshape(-1)

    return centroids, labels


# ========================================
# Save per-class clustering results
# ========================================
def save_per_class_clustering_result(
    df_class: pd.DataFrame,
    class_id: int,
    labels: np.ndarray,
    centroids: np.ndarray,
    output_base_dir: str,
    max_rows_per_file: int = 100_000,
    extra_metadata: dict = None
):
    """
    Save clustering results for one class:
        output_base_dir/
          `-- class_{id}/
               |-- centroids.npy
               |-- cluster_mapping.json
               `-- clusters/
                    |-- cluster_0000_00000.parquet
                    `-- ...
    """
    output_base_dir = Path(output_base_dir)
    class_dir = output_base_dir / f"class_{class_id}"
    class_dir.mkdir(parents=True, exist_ok=True)

    clusters_dir = class_dir / "clusters"
    clusters_dir.mkdir(exist_ok=True)

    # Add cluster labels.
    df_with_label = df_class.reset_index(drop=True).copy()
    df_with_label["cluster_label"] = labels

    # Save cluster data in chunks.
    grouped = df_with_label.groupby("cluster_label")
    print(f"Class {class_id}: {len(grouped)} subclusters")

    for label, group in grouped:
        chunk_size = max_rows_per_file
        num_chunks = (len(group) // chunk_size) + 1
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(group))
            chunk = group.iloc[start_idx:end_idx].reset_index(drop=True)
            filename = f"cluster_{label:04d}_{i:05d}.parquet"
            chunk.to_parquet(clusters_dir / filename, index=False)

    # Save centroids.
    np.save(class_dir / "centroids.npy", centroids)

    # Save mapping metadata.
    unique, counts = np.unique(labels, return_counts=True)
    mapping = {
        "class_id": int(class_id),
        "num_clusters": len(unique),
        "centroid_shape": centroids.shape,
        "cluster_distribution": dict(zip(unique.tolist(), counts.tolist())),
        "metadata": extra_metadata or {}
    }
    with open(class_dir / "cluster_mapping.json", 'w', encoding='utf-8') as f:
        import json
        json.dump(mapping, f, indent=2, ensure_ascii=False)

    print(f"Class {class_id} clustering results saved to: {class_dir}")


# ========================================
# Main function.
# ========================================
def main(
    input_pkl_dir: str,
    output_base_dir: str,
    feature_column: str = "feature",
    target_classes: list = None,  # set this to process only selected classes
    n_clusters_per_class: int = 10,  # number of clusters per class
    max_samples_per_class: int = None,  # optional cap on samples per class
    kmeans_iter: int = 20,
    use_gpu: bool = False,
    max_rows_per_file: int = 100_000
):
    print("Starting class-wise clustering task...")

    # 1. Load all PKL files.
    df_all = read_all_pkl_files(
        folder_path=input_pkl_dir,
        columns=['image_name', 'true_class', 'pred_class', 'correct', 'loss', 'logits', 'feature'],
        verbose=True,
    )

    # 2. Filter target classes.
    if target_classes is not None:
        df_all = df_all[df_all['true_class'].isin(target_classes)]
        print(f"Rows after filtering: {len(df_all)}")

    # 3. Group by true_class.
    grouped = df_all.groupby('true_class')
    print(f"Found {len(grouped)} classes")

    # 4. Cluster each class in parallel.
    groups = list(grouped)  # [(class_id, df_class), ...]
    print(f"Preparing to process {len(groups)} classes")

    # Keep only required columns to reduce inter-process pickle overhead.
    needed_cols = ['image_name', 'true_class', 'pred_class', 'correct', 'loss', 'logits', feature_column]
    groups = [(cid, dfc[needed_cols].copy()) for cid, dfc in groups]

    # Choose workers based on CPU count and task count; too many workers can increase memory pressure.
    max_workers = min(os.cpu_count() or 1, len(groups), 8)
    print(f"Process workers = {max_workers}")

    futures = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for class_id, df_class in tqdm(groups, desc="submitting jobs"):
            futures.append(ex.submit(
                process_one_class,
                class_id,
                df_class,
                feature_column,
                n_clusters_per_class,
                max_samples_per_class,
                kmeans_iter,
                use_gpu,
                output_base_dir,
                max_rows_per_file,
                input_pkl_dir,
            ))

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing classes (parallel)"):
            class_id, ok, msg = fut.result()
            if not ok:
                print(f"Error: class {class_id}: {msg}")
            else:
                print(f"class {class_id}: {msg}")

def format_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

# ========================================
# Example usage.
# ========================================
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="directory containing pkl files from inference")
    ap.add_argument("--output", required=True, help="output directory for cluster results")
    cli_args = ap.parse_args()

    input_pkl_dir = cli_args.input
    output_base_dir = cli_args.output

    t0 = time.perf_counter()
    main(
        input_pkl_dir=input_pkl_dir,
        output_base_dir=output_base_dir,
        feature_column="feature",           # field name that stores fc inputs in the PKL files
        target_classes=None,                # set to [0, 1, 2] to process only the first three classes; None means all classes
        n_clusters_per_class=25,            # cluster each class into 25 subclusters; adjustable
        max_samples_per_class=5000,         # use at most 5000 samples per class to avoid excessive memory usage
        kmeans_iter=25,
        use_gpu=False,                       # enable GPU acceleration when available
        max_rows_per_file=10000             # at most 10k rows per parquet file
    )
    t1 = time.perf_counter()
    print(f"Total inference time: {format_duration(t1 - t0)}")
