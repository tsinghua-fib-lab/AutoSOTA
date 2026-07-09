#!/usr/bin/env python3
"""
Compute aspect ratio for all datasets.
Aspect ratio = max pairwise distance / min pairwise distance (among non-zero distances)
"""

import numpy as np
from pathlib import Path
import time

# Dataset paths
DATASETS = {
    "mnist": "datasets/mnist.txt",
    "fmnist": "datasets/fmnist.txt",
    "cifar10": "datasets/cifar10.txt",
    "cifar100": "datasets/cifar100.txt",
    "mnist_clip": "datasets/mnist_clip.txt",
    "fmnist_clip": "datasets/fmnist_clip.txt",
    "cifar10_clip": "datasets/cifar10_clip.txt",
    "cifar100_clip": "datasets/cifar100_clip.txt",
    "reddit": "datasets/reddit.txt",
    "har": "datasets/har.txt",
    "susy": "datasets/susy.txt",
    "stackexchange": "datasets/stackexchange.txt",
}


def load_dataset(path: str, max_points: int = None) -> np.ndarray:
    """Load dataset from text file."""
    data = np.loadtxt(path)
    if max_points and len(data) > max_points:
        indices = np.random.choice(len(data), max_points, replace=False)
        data = data[indices]
    return data


def compute_aspect_ratio_sampled(data: np.ndarray, n_samples: int = 20000) -> dict:
    """
    Compute aspect ratio using sampling for efficiency.
    Uses random sampling of point pairs for large datasets.
    """
    n = len(data)

    # Sample points if dataset is large
    if n > n_samples:
        indices = np.random.choice(n, n_samples, replace=False)
        sample = data[indices]
    else:
        sample = data

    n_sample = len(sample)

    # Compute pairwise distances for sampled points
    # Use broadcasting for efficiency
    min_dist = float('inf')
    max_dist = 0.0

    # Compute in batches to avoid memory issues
    batch_size = 500
    for i in range(0, n_sample, batch_size):
        batch_end = min(i + batch_size, n_sample)
        batch = sample[i:batch_end]

        # Compute distances from this batch to all points
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a.b
        batch_norms = np.sum(batch**2, axis=1, keepdims=True)
        all_norms = np.sum(sample**2, axis=1, keepdims=True).T

        dists_sq = batch_norms + all_norms - 2 * np.dot(batch, sample.T)
        dists_sq = np.maximum(dists_sq, 0)  # Handle numerical errors

        # Mask diagonal (self-distances) and lower triangle (duplicates)
        for j in range(batch_end - i):
            global_idx = i + j
            dists_sq[j, :global_idx+1] = np.nan

        # Find min and max
        valid_dists_sq = dists_sq[~np.isnan(dists_sq)]
        if len(valid_dists_sq) > 0:
            # Filter out very small distances (numerical zeros)
            nonzero_mask = valid_dists_sq > 1e-10
            if np.any(nonzero_mask):
                min_dist = min(min_dist, np.sqrt(np.min(valid_dists_sq[nonzero_mask])))
            max_dist = max(max_dist, np.sqrt(np.max(valid_dists_sq)))

    aspect_ratio = max_dist / min_dist if min_dist > 0 else float('inf')

    return {
        'min_dist': min_dist,
        'max_dist': max_dist,
        'aspect_ratio': aspect_ratio,
        'n_samples': n_sample,
    }


def compute_diameter_and_spread(data: np.ndarray, n_samples: int = 20000) -> dict:
    """
    Compute:
    - Diameter: max pairwise distance
    - Spread: distance from centroid to furthest point
    - Min nearest neighbor distance (approximate)
    """
    n, d = data.shape

    # Sample points if dataset is large
    if n > n_samples:
        indices = np.random.choice(n, n_samples, replace=False)
        sample = data[indices]
    else:
        sample = data

    # Compute centroid
    centroid = np.mean(sample, axis=0)

    # Distances from centroid
    dists_from_centroid = np.sqrt(np.sum((sample - centroid)**2, axis=1))
    max_dist_from_centroid = np.max(dists_from_centroid)
    min_dist_from_centroid = np.min(dists_from_centroid[dists_from_centroid > 1e-10])

    # Compute pairwise distance stats using sampling
    stats = compute_aspect_ratio_sampled(data, n_samples)

    return {
        'n': n,
        'd': d,
        'diameter': stats['max_dist'],
        'min_pairwise_dist': stats['min_dist'],
        'aspect_ratio': stats['aspect_ratio'],
        'spread': max_dist_from_centroid,
        'min_centroid_dist': min_dist_from_centroid,
        'centroid_aspect': max_dist_from_centroid / min_dist_from_centroid if min_dist_from_centroid > 0 else float('inf'),
    }


def main():
    print("=" * 100)
    print("DATASET ASPECT RATIO ANALYSIS")
    print("=" * 100)
    print(f"{'Dataset':<20} {'n':>10} {'d':>6} {'Diameter':>12} {'Min Dist':>12} {'Aspect Ratio':>15}")
    print("-" * 100)

    results = []

    for name, path in DATASETS.items():
        full_path = Path(path)
        if not full_path.exists():
            print(f"{name:<20} {'(not found)':<10}")
            continue

        print(f"Processing {name}...", end=" ", flush=True)
        start = time.time()

        try:
            data = load_dataset(path)
            stats = compute_diameter_and_spread(data)
            elapsed = time.time() - start

            print(f"\r{name:<20} {stats['n']:>10,} {stats['d']:>6} {stats['diameter']:>12.2f} {stats['min_pairwise_dist']:>12.6f} {stats['aspect_ratio']:>15.2f}")

            results.append({
                'dataset': name,
                **stats
            })
        except Exception as e:
            print(f"\r{name:<20} Error: {e}")

    print("=" * 100)

    # Print sorted by aspect ratio
    print("\n" + "=" * 80)
    print("DATASETS SORTED BY ASPECT RATIO")
    print("=" * 80)
    results.sort(key=lambda x: x['aspect_ratio'])

    for r in results:
        print(f"{r['dataset']:<20} Aspect Ratio: {r['aspect_ratio']:>12.2f}  (Diameter: {r['diameter']:.2f}, Min: {r['min_pairwise_dist']:.6f})")

    # Save to CSV
    import csv
    output_path = Path("results/aspect_ratios.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['dataset', 'n', 'd', 'diameter', 'min_pairwise_dist', 'aspect_ratio', 'spread', 'min_centroid_dist', 'centroid_aspect'])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    np.random.seed(42)
    main()
