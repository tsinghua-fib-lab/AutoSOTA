"""
CIFAR100 clustering script.
Runs KMeans clustering on embedding files using CIFAR100 index format.

Input: a torch.save .pt file containing indices, labels, and embeddings.
Output: class_{id}/clusters/cluster_*.parquet files.

Usage:
    python cifar_faiss_cluster.py \
        --input /path/to/train_embeddings.pt \
        --output /path/to/output_cluster_dir \
        --n-clusters 10 \
        --feature-dim 512
"""

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import faiss
from tqdm import tqdm
import time
import argparse
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional


def format_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def load_embeddings(embedding_path: str, verbose: bool = True) -> pd.DataFrame:
    """
    Load a CIFAR100 embedding file and convert it to a DataFrame.

    Args:
        embedding_path: .pt file path
        verbose: whether to print information

    Returns:
        pd.DataFrame containing index, label, and feature columns
    """
    if verbose:
        print(f"Loading embeddings from {embedding_path}...")

    data = torch.load(embedding_path, map_location='cpu')

    indices = data['indices'].numpy() if isinstance(data['indices'], torch.Tensor) else data['indices']
    labels = data['labels'].numpy() if isinstance(data['labels'], torch.Tensor) else data['labels']
    embeddings = data['embeddings'].numpy() if isinstance(data['embeddings'], torch.Tensor) else data['embeddings']

    if verbose:
        print(f"Loaded {len(indices)} samples, embedding dim: {embeddings.shape[1]}")
        print(f"Number of classes: {len(np.unique(labels))}")

    # Store as lists for downstream processing.
    df = pd.DataFrame({
        'index': indices,
        'label': labels,
        'feature': list(embeddings)  # one numpy array per row
    })

    return df


def faiss_kmeans(data: np.ndarray, k: int, niter: int = 20, gpu: bool = False, seed: int = 42):
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


def process_one_class(
    class_id: int,
    df_class: pd.DataFrame,
    n_clusters: int,
    kmeans_iter: int,
    use_gpu: bool,
    output_base_dir: str,
    max_rows_per_file: int,
):
    """
    Cluster one class and save the results.
    """
    try:
        N = len(df_class)

        if N < n_clusters:
            return (class_id, False, f"skip: N={N} < k={n_clusters}")

        # Extract features.
        features = np.stack(df_class['feature'].values).astype(np.float32)

        # KMeans clustering.
        centroids, labels = faiss_kmeans(
            data=features,
            k=n_clusters,
            niter=kmeans_iter,
            gpu=use_gpu,
            seed=42
        )

        # Save results.
        output_base_dir = Path(output_base_dir)
        class_dir = output_base_dir / f"class_{class_id}"
        class_dir.mkdir(parents=True, exist_ok=True)
        clusters_dir = class_dir / "clusters"
        clusters_dir.mkdir(exist_ok=True)

        # Add cluster labels.
        df_with_label = df_class.reset_index(drop=True).copy()
        df_with_label["cluster_label"] = labels

        # Save each cluster in chunks.
        grouped = df_with_label.groupby("cluster_label")

        for label, group in grouped:
            chunk_size = max_rows_per_file
            num_chunks = (len(group) // chunk_size) + 1
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, len(group))
                chunk = group.iloc[start_idx:end_idx].reset_index(drop=True)

                # Convert feature values to lists for serialization.
                chunk_to_save = chunk.copy()
                chunk_to_save['feature'] = chunk_to_save['feature'].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

                filename = f"cluster_{label:04d}_{i:05d}.parquet"
                chunk_to_save.to_parquet(clusters_dir / filename, index=False)

        # Save centroids.
        np.save(class_dir / "centroids.npy", centroids)

        # Save mapping metadata.
        unique, counts = np.unique(labels, return_counts=True)
        mapping = {
            "class_id": int(class_id),
            "num_clusters": len(unique),
            "centroid_shape": centroids.shape,
            "cluster_distribution": dict(zip(unique.tolist(), counts.tolist())),
        }
        with open(class_dir / "cluster_mapping.json", 'w', encoding='utf-8') as f:
            import json
            json.dump(mapping, f, indent=2, ensure_ascii=False)

        return (class_id, True, f"ok, {len(grouped)} clusters created")

    except Exception as e:
        import traceback
        return (class_id, False, f"error: {e}\n{traceback.format_exc()}")


def main(
    input_path: str,
    output_base_dir: str,
    n_clusters_per_class: int = 10,
    kmeans_iter: int = 20,
    use_gpu: bool = False,
    max_rows_per_file: int = 10000,
    target_classes: Optional[List[int]] = None,
):
    """
    Main function for clustering CIFAR100 embeddings.

    Args:
        input_path: embedding .pt file path
        output_base_dir: output directory
        n_clusters_per_class: number of clusters per class
        kmeans_iter: number of KMeans iterations
        use_gpu: whether to use GPU
        max_rows_per_file: maximum rows per parquet file
        target_classes: only process specified classes; None means all classes
    """
    print("=" * 60)
    print("CIFAR100 Clustering Task")
    print("=" * 60)

    # 1. Load embeddings.
    df_all = load_embeddings(input_path, verbose=True)

    # 2. Filter target classes if specified.
    if target_classes is not None:
        df_all = df_all[df_all['label'].isin(target_classes)]
        print(f"Rows after filtering: {len(df_all)}")

    # 3. Group by label.
    grouped = df_all.groupby('label')
    print(f"Found {len(grouped)} classes")

    # 4. Process each class in parallel.
    groups = list(grouped)  # [(label, df_class), ...]

    # Keep only required columns.
    needed_cols = ['index', 'label', 'feature']
    groups = [(label, dfc[needed_cols].copy()) for label, dfc in groups]

    max_workers = min(os.cpu_count() or 1, len(groups), 32)
    print(f"Using {max_workers} worker processes")

    futures = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for class_id, df_class in tqdm(groups, desc="Submitting jobs"):
            futures.append(ex.submit(
                process_one_class,
                class_id,
                df_class,
                n_clusters_per_class,
                kmeans_iter,
                use_gpu,
                output_base_dir,
                max_rows_per_file,
            ))

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing classes"):
            class_id, ok, msg = fut.result()
            if not ok:
                print(f"Error: class {class_id}: {msg}")
            else:
                print(f"class {class_id}: {msg}")

    print(f"\nClustering complete. Results saved to: {output_base_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIFAR100 clustering script.")
    parser.add_argument("--input", type=str, required=True, help="embedding .pt file path")
    parser.add_argument("--output", type=str, required=True, help="output directory")
    parser.add_argument("--n-clusters", type=int, default=10, help="number of clusters per class")
    parser.add_argument("--kmeans-iter", type=int, default=20, help="number of KMeans iterations")
    parser.add_argument("--gpu", action="store_true", help="whether to use GPU")
    parser.add_argument("--max-rows", type=int, default=10000, help="maximum rows per parquet file")
    parser.add_argument("--target-classes", type=int, nargs='+', default=None, help="only process specified classes")

    args = parser.parse_args()

    t0 = time.perf_counter()
    main(
        input_path=args.input,
        output_base_dir=args.output,
        n_clusters_per_class=args.n_clusters,
        kmeans_iter=args.kmeans_iter,
        use_gpu=args.gpu,
        max_rows_per_file=args.max_rows,
        target_classes=args.target_classes,
    )
    t1 = time.perf_counter()
    print(f"Total time: {format_duration(t1 - t0)}")