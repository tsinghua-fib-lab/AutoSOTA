#!/usr/bin/env python3
"""
Scaling Laws Experiment for k-means Clustering

This script validates the quantization-theoretic scaling laws from the paper:
"Faster k-means Seeding Under The Manifold Hypothesis"

For each dataset, we measure:
- β_k = cost(X, μ(X)) / cost(X, C_k)  [should scale as k^ε]
- η_k = max||c_i - c_j|| / min||c_i - c_j||  [should scale as k^(ε/2)]

where ε = 2/d is the quantization exponent and d is the intrinsic dimension.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Set threading for FAISS/OpenBLAS before importing
os.environ['OMP_NUM_THREADS'] = str(multiprocessing.cpu_count())
os.environ['OPENBLAS_NUM_THREADS'] = str(multiprocessing.cpu_count())
os.environ['MKL_NUM_THREADS'] = str(multiprocessing.cpu_count())

try:
    import faiss
    HAS_FAISS = True
    # Set FAISS to use all threads
    faiss.omp_set_num_threads(multiprocessing.cpu_count())
except ImportError:
    HAS_FAISS = False
    from sklearn.cluster import KMeans

# ============================================================
# Dataset Loading
# ============================================================

def load_dataset(path: str) -> np.ndarray:
    """Load dataset from various formats."""
    path = Path(path)

    if path.suffix == '.npy':
        data = np.load(path)
    elif path.suffix == '.npz':
        npz = np.load(path)
        # Try common keys
        for key in ['data', 'X', 'arr_0', 'features']:
            if key in npz:
                data = npz[key]
                break
        else:
            # Use first array
            data = npz[list(npz.keys())[0]]
    elif path.suffix == '.txt':
        data = np.loadtxt(path, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    # Flatten if needed (e.g., images stored as (n, h, w) or (n, h, w, c))
    if data.ndim > 2:
        data = data.reshape(data.shape[0], -1)

    # Ensure float32 for FAISS compatibility and C-contiguous
    data = np.ascontiguousarray(data.astype(np.float32))

    return data


def download_dataset(name: str, cache_dir: str = "datasets") -> np.ndarray:
    """Download and cache standard datasets."""
    cache_path = Path(cache_dir)
    cache_path.mkdir(exist_ok=True)

    npy_path = cache_path / f"{name}.npy"

    if npy_path.exists():
        print(f"Loading cached {name} from {npy_path}")
        return np.load(npy_path).astype(np.float32)

    print(f"Downloading {name}...")

    if name == "mnist":
        from sklearn.datasets import fetch_openml
        mnist = fetch_openml('mnist_784', version=1, as_frame=False)
        data = mnist.data.astype(np.float32)

    elif name == "fashion_mnist":
        from sklearn.datasets import fetch_openml
        fmnist = fetch_openml('Fashion-MNIST', version=1, as_frame=False)
        data = fmnist.data.astype(np.float32)

    elif name == "cifar10":
        try:
            import torchvision.datasets as datasets
            import torchvision.transforms as transforms
            cifar = datasets.CIFAR10(root=str(cache_path), train=True, download=True)
            data = cifar.data.reshape(cifar.data.shape[0], -1).astype(np.float32)
        except ImportError:
            raise ImportError("Install torchvision to download CIFAR-10")

    elif name == "cifar100":
        try:
            import torchvision.datasets as datasets
            cifar = datasets.CIFAR100(root=str(cache_path), train=True, download=True)
            data = cifar.data.reshape(cifar.data.shape[0], -1).astype(np.float32)
        except ImportError:
            raise ImportError("Install torchvision to download CIFAR-100")

    else:
        raise ValueError(f"Unknown dataset: {name}")

    # Cache the processed data
    np.save(npy_path, data)
    print(f"Cached {name} to {npy_path}")

    return data


# ============================================================
# K-means Clustering
# ============================================================

def kmeans_faiss(X: np.ndarray, k: int, n_iter: int = 100, seed: int = None) -> tuple:
    """
    Run k-means using FAISS.

    Returns:
        centers: (k, d) array of cluster centers
        labels: (n,) array of cluster assignments
        cost: total k-means cost (sum of squared distances)
    """
    n, d = X.shape

    # Create k-means object
    kmeans = faiss.Kmeans(
        d=d,
        k=k,
        niter=n_iter,
        verbose=False,
        seed=seed if seed is not None else np.random.randint(0, 2**31),
        gpu=False  # Set to True if GPU available
    )

    # Train
    kmeans.train(X)

    # Get assignments and distances
    distances, labels = kmeans.index.search(X, 1)

    centers = kmeans.centroids
    cost = distances.sum()

    return centers, labels.flatten(), cost


def kmeans_sklearn(X: np.ndarray, k: int, n_iter: int = 100, seed: int = None) -> tuple:
    """
    Fallback k-means using sklearn.

    Returns:
        centers: (k, d) array of cluster centers
        labels: (n,) array of cluster assignments
        cost: total k-means cost (sum of squared distances)
    """
    kmeans = KMeans(
        n_clusters=k,
        max_iter=n_iter,
        n_init=1,
        random_state=seed,
        algorithm='lloyd'
    )

    labels = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_.astype(np.float32)

    # Compute cost
    cost = 0.0
    for i in range(len(X)):
        cost += np.sum((X[i] - centers[labels[i]]) ** 2)

    return centers, labels, cost


def run_kmeans(X: np.ndarray, k: int, n_iter: int = 100, seed: int = None) -> tuple:
    """Run k-means using best available backend."""
    if HAS_FAISS:
        return kmeans_faiss(X, k, n_iter, seed)
    else:
        return kmeans_sklearn(X, k, n_iter, seed)


# ============================================================
# Metric Computation
# ============================================================

def compute_opt1(X: np.ndarray) -> float:
    """Compute cost(X, μ(X)) - cost to single centroid."""
    centroid = X.mean(axis=0)
    return np.sum((X - centroid) ** 2)


def compute_beta(opt1: float, cost_k: float) -> float:
    """Compute β_k = opt1 / cost_k."""
    return opt1 / cost_k if cost_k > 0 else float('inf')


def compute_eta(centers: np.ndarray) -> float:
    """
    Compute η = max||c_i - c_j|| / min||c_i - c_j||

    This is the aspect ratio of the center set.
    """
    k = len(centers)
    if k < 2:
        return 1.0

    # Compute pairwise distances
    distances = []
    for i in range(k):
        for j in range(i + 1, k):
            dist = np.linalg.norm(centers[i] - centers[j])
            distances.append(dist)

    distances = np.array(distances)

    # Filter out zero distances (shouldn't happen but just in case)
    distances = distances[distances > 1e-10]

    if len(distances) == 0:
        return 1.0

    return distances.max() / distances.min()


# ============================================================
# Main Experiment
# ============================================================

def _run_single_k(args):
    """Helper function to run k-means for a single k value (for parallel execution)."""
    X, k, n_runs, n_iter, opt1 = args

    # Re-import faiss in subprocess if needed
    try:
        import faiss
        faiss.omp_set_num_threads(max(1, multiprocessing.cpu_count() // 4))  # Share cores
    except ImportError:
        pass

    betas = []
    etas = []

    for run in range(n_runs):
        seed = run * 1000 + k
        try:
            centers, labels, cost_k = run_kmeans(X, k, n_iter, seed)
            beta = compute_beta(opt1, cost_k)
            eta = compute_eta(centers)
            betas.append(beta)
            etas.append(eta)
        except Exception as e:
            continue

    if len(betas) > 0:
        return {
            'k': k,
            'beta_mean': np.mean(betas),
            'beta_std': np.std(betas),
            'eta_mean': np.mean(etas),
            'eta_std': np.std(etas),
            'n_successful_runs': len(betas)
        }
    return None


def run_experiment(
    X: np.ndarray,
    k_values: list,
    n_runs: int = 10,
    n_iter: int = 100,
    verbose: bool = True,
    parallel: bool = False
) -> pd.DataFrame:
    """
    Run the scaling law experiment.

    Args:
        X: Dataset (n, d) array
        k_values: List of k values to test
        n_runs: Number of runs per k value
        n_iter: Number of k-means iterations
        verbose: Print progress
        parallel: Run different k values in parallel (useful for multi-core systems)

    Returns:
        DataFrame with columns: k, beta_mean, beta_std, eta_mean, eta_std
    """
    n, d = X.shape
    print(f"Dataset shape: n={n}, d={d}")

    # Compute opt1 once
    opt1 = compute_opt1(X)
    print(f"opt1 (variance): {opt1:.4e}")

    # Filter valid k values
    valid_k = [k for k in k_values if k <= n]
    if len(valid_k) < len(k_values):
        print(f"Skipping k values larger than n={n}")

    results = []

    if parallel and len(valid_k) > 1:
        # Parallel execution across k values
        n_workers = min(len(valid_k), max(1, multiprocessing.cpu_count() // 8))
        print(f"Running in parallel with {n_workers} workers")

        args_list = [(X, k, n_runs, n_iter, opt1) for k in valid_k]

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_run_single_k, args): args[1] for args in args_list}

            for future in tqdm(as_completed(futures), total=len(futures), desc="k values"):
                result = future.result()
                if result is not None:
                    results.append(result)
    else:
        # Sequential execution (FAISS uses all cores internally)
        iterator = tqdm(valid_k, desc="k values") if verbose else valid_k

        for k in iterator:
            betas = []
            etas = []

            for run in range(n_runs):
                seed = run * 1000 + k  # Reproducible but different seeds

                try:
                    centers, labels, cost_k = run_kmeans(X, k, n_iter, seed)

                    beta = compute_beta(opt1, cost_k)
                    eta = compute_eta(centers)

                    betas.append(beta)
                    etas.append(eta)

                except Exception as e:
                    print(f"Error at k={k}, run={run}: {e}")
                    continue

            if len(betas) > 0:
                results.append({
                    'k': k,
                    'beta_mean': np.mean(betas),
                    'beta_std': np.std(betas),
                    'eta_mean': np.mean(etas),
                    'eta_std': np.std(etas),
                    'n_successful_runs': len(betas)
                })

    # Sort by k
    results = sorted(results, key=lambda x: x['k'])
    return pd.DataFrame(results)


def fit_power_law(k_values: np.ndarray, y_values: np.ndarray) -> tuple:
    """
    Fit y = a * k^b in log-log space.

    Returns:
        slope (b), intercept (log a), r_squared
    """
    log_k = np.log(k_values)
    log_y = np.log(y_values)

    # Linear regression in log space
    A = np.vstack([log_k, np.ones(len(log_k))]).T
    slope, intercept = np.linalg.lstsq(A, log_y, rcond=None)[0]

    # Compute R²
    y_pred = slope * log_k + intercept
    ss_res = np.sum((log_y - y_pred) ** 2)
    ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    return slope, intercept, r_squared


def plot_scaling_laws(
    df: pd.DataFrame,
    dataset_name: str,
    output_dir: str,
    show: bool = False
):
    """Generate log-log plots of β_k and η_k vs k."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    k_values = df['k'].values

    # Plot β_k
    ax = axes[0]
    beta_mean = df['beta_mean'].values
    beta_std = df['beta_std'].values

    ax.errorbar(k_values, beta_mean, yerr=beta_std, fmt='o', capsize=3,
                color='tab:blue', label='Data')

    # Fit power law
    slope, intercept, r2 = fit_power_law(k_values, beta_mean)
    k_fit = np.linspace(k_values.min(), k_values.max(), 100)
    y_fit = np.exp(intercept) * k_fit ** slope
    ax.plot(k_fit, y_fit, '--', color='tab:red',
            label=f'Fit: slope={slope:.3f}, R²={r2:.3f}')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('k (number of clusters)')
    ax.set_ylabel('β_k = opt₁/opt_k')
    ax.set_title(f'{dataset_name}: β_k scaling\n(ε ≈ {slope:.3f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot η_k
    ax = axes[1]
    eta_mean = df['eta_mean'].values
    eta_std = df['eta_std'].values

    ax.errorbar(k_values, eta_mean, yerr=eta_std, fmt='o', capsize=3,
                color='tab:green', label='Data')

    # Fit power law
    slope_eta, intercept_eta, r2_eta = fit_power_law(k_values, eta_mean)
    y_fit_eta = np.exp(intercept_eta) * k_fit ** slope_eta
    ax.plot(k_fit, y_fit_eta, '--', color='tab:red',
            label=f'Fit: slope={slope_eta:.3f}, R²={r2_eta:.3f}')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('k (number of clusters)')
    ax.set_ylabel('η_k = max/min center distance')
    ax.set_title(f'{dataset_name}: η_k scaling\n(ε/2 ≈ {slope_eta:.3f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    output_path = Path(output_dir) / f"{dataset_name}_scaling.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return slope, r2, slope_eta, r2_eta


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Validate k-means scaling laws",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--dataset', '-d',
        type=str,
        default=None,
        help='Path to dataset file (.npy, .npz, or .txt)'
    )

    parser.add_argument(
        '--download',
        type=str,
        choices=['mnist', 'fashion_mnist', 'cifar10', 'cifar100'],
        default=None,
        help='Download a standard dataset'
    )

    parser.add_argument(
        '--name',
        type=str,
        default=None,
        help='Dataset name (for output files). Inferred from path if not provided.'
    )

    parser.add_argument(
        '--k-values',
        type=int,
        nargs='+',
        default=[5, 10, 50, 100, 250, 500, 750, 1000],
        help='List of k values to test'
    )

    parser.add_argument(
        '--n-runs',
        type=int,
        default=10,
        help='Number of runs per k value'
    )

    parser.add_argument(
        '--n-iter',
        type=int,
        default=100,
        help='Number of k-means iterations'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/scaling_laws',
        help='Output directory for results'
    )

    parser.add_argument(
        '--show-plots',
        action='store_true',
        help='Display plots interactively'
    )

    parser.add_argument(
        '--normalize',
        action='store_true',
        help='L2 normalize the data before clustering'
    )

    parser.add_argument(
        '--subsample',
        type=int,
        default=None,
        help='Subsample dataset to this many points'
    )

    args = parser.parse_args()

    # Load dataset
    if args.download:
        X = download_dataset(args.download, cache_dir="datasets")
        dataset_name = args.name or args.download
    elif args.dataset:
        X = load_dataset(args.dataset)
        dataset_name = args.name or Path(args.dataset).stem
    else:
        parser.error("Must specify --dataset or --download")

    print(f"Loaded dataset: {dataset_name}")
    print(f"Shape: {X.shape}")

    # Preprocessing
    if args.subsample and args.subsample < len(X):
        np.random.seed(42)
        indices = np.random.choice(len(X), args.subsample, replace=False)
        X = X[indices]
        print(f"Subsampled to {len(X)} points")

    if args.normalize:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)  # Avoid division by zero
        X = X / norms
        print("L2 normalized data")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Print backend info
    if HAS_FAISS:
        print("Using FAISS backend")
    else:
        print("FAISS not available, using sklearn backend (slower)")

    # Run experiment
    print(f"\nRunning experiment with k ∈ {args.k_values}")
    print(f"Runs per k: {args.n_runs}")
    print("-" * 50)

    df = run_experiment(
        X=X,
        k_values=args.k_values,
        n_runs=args.n_runs,
        n_iter=args.n_iter,
        verbose=True
    )

    # Save results
    csv_path = output_dir / f"{dataset_name}_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved results to {csv_path}")

    # Print summary
    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)
    print(df.to_string(index=False))

    # Fit and report power laws
    k_values = df['k'].values

    slope_beta, _, r2_beta = fit_power_law(k_values, df['beta_mean'].values)
    slope_eta, _, r2_eta = fit_power_law(k_values, df['eta_mean'].values)

    print("\n" + "-" * 50)
    print("POWER LAW FITS")
    print("-" * 50)
    print(f"β_k ~ k^ε where ε = {slope_beta:.4f} (R² = {r2_beta:.4f})")
    print(f"η_k ~ k^(ε/2) where ε/2 = {slope_eta:.4f} (R² = {r2_eta:.4f})")
    print(f"\nEstimated intrinsic dimension from β: d ≈ {2/slope_beta:.1f}")
    print(f"Estimated intrinsic dimension from η: d ≈ {1/slope_eta:.1f}")

    # Generate plots
    plot_scaling_laws(df, dataset_name, str(output_dir), show=args.show_plots)

    # Save summary
    summary = {
        'dataset': dataset_name,
        'n_points': len(X),
        'ambient_dim': X.shape[1],
        'epsilon_beta': slope_beta,
        'r2_beta': r2_beta,
        'epsilon_eta': slope_eta * 2,  # Convert from ε/2 to ε
        'r2_eta': r2_eta,
        'estimated_intrinsic_dim_beta': 2 / slope_beta,
        'estimated_intrinsic_dim_eta': 1 / slope_eta,
        'timestamp': datetime.now().isoformat()
    }

    summary_path = output_dir / f"{dataset_name}_summary.csv"
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
