#!/usr/bin/env python3
"""
Run benchmarks for all algorithms on specified datasets.

Usage:
    python scripts/run/run_benchmarks.py <dataset> [options]
    python scripts/run/run_benchmarks.py mnist --algorithms qkmeans,kmeanspp --k-values 10,50,100
    python scripts/run/run_benchmarks.py --all  # Run all datasets

Examples:
    python scripts/run/run_benchmarks.py mnist
    python scripts/run/run_benchmarks.py cifar10_clip --runs 10 --k-values 50,100,200
    python scripts/run/run_benchmarks.py --all --algorithms qkmeans,afkmc2
"""

import subprocess
import json
import argparse
import sys
from pathlib import Path
from datetime import datetime

# Paths
ROOT_DIR = Path(__file__).parent.parent.parent
DATASETS_DIR = ROOT_DIR / "datasets"
CONFIGS_DIR = ROOT_DIR / "configs" / "generated"
RESULTS_DIR = ROOT_DIR / "results"
BIN_DIR = ROOT_DIR / "bin"

# Available algorithms
ALGORITHMS = [
    "kmeanspp", "afkmc2", "prone", "pronecoreset",
    "fastcoreset", "rejectionlsh", "qkmeans"
]

# Available datasets
DATASETS = [
    "mnist", "fmnist", "cifar10", "cifar100",
    "mnist_clip", "fmnist_clip", "cifar10_clip", "cifar100_clip",
    "har", "susy", "reddit", "stackexchange"
]

# Default parameters
DEFAULT_K_VALUES = [10, 50, 100, 200, 500]
DEFAULT_M_VALUES = [100]
DEFAULT_EF_VALUES = [50]
DEFAULT_ALPHA_VALUES = [0.01]
DEFAULT_RUNS = 5


def create_config(dataset: str, algorithm: str, k_values: list,
                  num_runs: int, m_values: list, ef_values: list,
                  alpha_values: list) -> Path:
    """Create a JSON config file for the benchmark."""
    config = {
        "name": dataset,
        "data_path": str(DATASETS_DIR / f"{dataset}.txt"),
        "k_values": k_values,
        "num_runs": num_runs,
        "m_values": m_values,
        "ef_values": ef_values,
        "alpha_values": alpha_values,
        "output_csv": str(RESULTS_DIR / f"{algorithm}_{dataset}.csv")
    }

    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    config_path = CONFIGS_DIR / f"{algorithm}_{dataset}.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    return config_path


def run_single_algorithm(algorithm: str, config_path: Path,
                         timeout: int = 3600) -> bool:
    """Run a single algorithm benchmark."""
    binary = BIN_DIR / "run_single"
    if not binary.exists():
        print(f"  ERROR: Binary not found: {binary}")
        print("  Run 'make' first to build binaries")
        return False

    try:
        result = subprocess.run(
            [str(binary), algorithm, str(config_path)],
            cwd=str(ROOT_DIR),
            capture_output=True,
            text=True,
            timeout=timeout
        )
        if result.returncode != 0:
            print(f"  ERROR: {result.stderr}")
            return False
        print(result.stdout)
        return True
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after {timeout}s")
        return False
    except Exception as e:
        print(f"  EXCEPTION: {e}")
        return False


def run_comparison(config_path: Path, timeout: int = 7200) -> bool:
    """Run all-algorithm comparison benchmark."""
    binary = BIN_DIR / "run_comparison"
    if not binary.exists():
        print(f"  ERROR: Binary not found: {binary}")
        return False

    try:
        result = subprocess.run(
            [str(binary), str(config_path)],
            cwd=str(ROOT_DIR),
            capture_output=True,
            text=True,
            timeout=timeout
        )
        if result.returncode != 0:
            print(f"  ERROR: {result.stderr}")
            return False
        print(result.stdout)
        return True
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after {timeout}s")
        return False
    except Exception as e:
        print(f"  EXCEPTION: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run seeding algorithm benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run/run_benchmarks.py mnist
  python scripts/run/run_benchmarks.py cifar10_clip --algorithms qkmeans,kmeanspp
  python scripts/run/run_benchmarks.py --all --runs 10
        """
    )
    parser.add_argument("dataset", nargs="?", help="Dataset name")
    parser.add_argument("--all", action="store_true",
                        help="Run on all available datasets")
    parser.add_argument("--algorithms", type=str, default=None,
                        help=f"Comma-separated algorithms (default: all). Options: {','.join(ALGORITHMS)}")
    parser.add_argument("--comparison", action="store_true",
                        help="Run all-algorithm comparison instead of individual")
    parser.add_argument("--k-values", type=str, default=None,
                        help=f"Comma-separated k values (default: {DEFAULT_K_VALUES})")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS,
                        help=f"Number of runs per config (default: {DEFAULT_RUNS})")
    parser.add_argument("--m-values", type=str, default=None,
                        help="Comma-separated m values for AFKMC2/QKMEANS")
    parser.add_argument("--ef-values", type=str, default=None,
                        help="Comma-separated ef values for QKMEANS")
    parser.add_argument("--alpha-values", type=str, default=None,
                        help="Comma-separated alpha values for coreset methods")
    parser.add_argument("--timeout", type=int, default=3600,
                        help="Timeout per algorithm in seconds (default: 3600)")

    args = parser.parse_args()

    # Determine datasets to run
    if args.all:
        datasets = DATASETS
    elif args.dataset:
        datasets = [args.dataset]
    else:
        parser.print_help()
        sys.exit(1)

    # Parse algorithm list
    if args.algorithms:
        algorithms = [a.strip() for a in args.algorithms.split(",")]
        for algo in algorithms:
            if algo not in ALGORITHMS:
                print(f"ERROR: Unknown algorithm '{algo}'")
                print(f"Available: {', '.join(ALGORITHMS)}")
                sys.exit(1)
    else:
        algorithms = ALGORITHMS

    # Parse hyperparameters
    k_values = [int(k) for k in args.k_values.split(",")] if args.k_values else DEFAULT_K_VALUES
    m_values = [int(m) for m in args.m_values.split(",")] if args.m_values else DEFAULT_M_VALUES
    ef_values = [int(e) for e in args.ef_values.split(",")] if args.ef_values else DEFAULT_EF_VALUES
    alpha_values = [float(a) for a in args.alpha_values.split(",")] if args.alpha_values else DEFAULT_ALPHA_VALUES

    # Check datasets exist
    for dataset in datasets:
        txt_path = DATASETS_DIR / f"{dataset}.txt"
        if not txt_path.exists():
            print(f"ERROR: Dataset not found: {txt_path}")
            sys.exit(1)

    # Create results directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Print summary
    print("=" * 60)
    print("QKMEANS Benchmark Runner")
    print("=" * 60)
    print(f"Datasets:   {', '.join(datasets)}")
    print(f"Algorithms: {', '.join(algorithms)}")
    print(f"k values:   {k_values}")
    print(f"Runs:       {args.runs}")
    print(f"Timeout:    {args.timeout}s")
    print("=" * 60)

    # Run benchmarks
    start_time = datetime.now()
    results = {}

    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset}")
        print("=" * 60)

        if args.comparison:
            # Run all-algorithm comparison
            config_path = create_config(
                dataset, "comparison", k_values, args.runs,
                m_values, ef_values, alpha_values
            )
            success = run_comparison(config_path, args.timeout * len(algorithms))
            results[(dataset, "comparison")] = success
        else:
            # Run individual algorithms
            for algo in algorithms:
                print(f"\n--- {algo} ---")
                config_path = create_config(
                    dataset, algo, k_values, args.runs,
                    m_values, ef_values, alpha_values
                )
                success = run_single_algorithm(algo, config_path, args.timeout)
                results[(dataset, algo)] = success

    # Print summary
    elapsed = datetime.now() - start_time
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    n_success = sum(results.values())
    n_total = len(results)
    print(f"Completed: {n_success}/{n_total}")
    print(f"Time:      {elapsed}")

    if n_success < n_total:
        print("\nFailed:")
        for (dataset, algo), success in results.items():
            if not success:
                print(f"  - {dataset}/{algo}")

    print(f"\nResults saved to: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
