#!/usr/bin/env python3
"""
Generate config files from template.

Usage:
    python configs/generate_configs.py                    # Generate all
    python configs/generate_configs.py --dataset mnist    # Single dataset
    python configs/generate_configs.py --algorithm qkmeans --dataset mnist

This script generates JSON config files in configs/generated/ by filling
in the template with dataset and algorithm names.
"""

import argparse
import json
from pathlib import Path
from string import Template

# Paths
ROOT_DIR = Path(__file__).parent.parent
TEMPLATE_PATH = Path(__file__).parent / "template.json"
OUTPUT_DIR = Path(__file__).parent / "generated"

# Available options
ALGORITHMS = [
    "kmeanspp", "afkmc2", "prone", "pronecoreset",
    "fastcoreset", "rejectionlsh", "qkmeans", "qkmeans_anns",
    "comparison", "sweep"
]

DATASETS = [
    "mnist", "fmnist", "cifar10", "cifar100",
    "mnist_clip", "fmnist_clip", "cifar10_clip", "cifar100_clip",
    "har", "susy", "reddit", "stackexchange"
]

# Default hyperparameters per algorithm
ALGORITHM_DEFAULTS = {
    "kmeanspp": {"m_values": [], "ef_values": [], "alpha_values": []},
    "afkmc2": {"m_values": [50, 100, 200], "ef_values": [], "alpha_values": []},
    "prone": {"m_values": [], "ef_values": [], "alpha_values": []},
    "pronecoreset": {"m_values": [], "ef_values": [], "alpha_values": [0.005, 0.01, 0.02]},
    "fastcoreset": {"m_values": [], "ef_values": [], "alpha_values": [0.005, 0.01, 0.02]},
    "rejectionlsh": {"m_values": [], "ef_values": [], "alpha_values": []},
    "qkmeans": {"m_values": [100], "ef_values": [25, 50, 100], "alpha_values": []},
    "qkmeans_anns": {"m_values": [100], "ef_values": [25, 50, 100], "alpha_values": []},
    "comparison": {"m_values": [100], "ef_values": [50], "alpha_values": [0.01]},
    "sweep": {
        "m_values": [50, 100, 200, 500],
        "ef_values": [10, 25, 50, 100],
        "alpha_values": [0.005, 0.01, 0.02, 0.05]
    },
}

# K values per dataset (can customize based on dataset size)
DATASET_K_VALUES = {
    "mnist": [10, 50, 100, 200, 500, 750, 1000],
    "fmnist": [10, 50, 100, 200, 500, 750, 1000],
    "cifar10": [10, 50, 100, 200, 500],
    "cifar100": [10, 50, 100, 200, 500, 750, 1000],
    "mnist_clip": [10, 50, 100, 200, 500],
    "fmnist_clip": [10, 50, 100, 200, 500],
    "cifar10_clip": [10, 50, 100, 200, 500],
    "cifar100_clip": [10, 50, 100, 200, 500],
    "har": [10, 50, 100, 200],
    "susy": [10, 50, 100, 200, 500],
    "reddit": [10, 50, 100, 200, 500],
    "stackexchange": [10, 50, 100, 200, 500],
}


def generate_config(algorithm: str, dataset: str, num_runs: int = 5) -> dict:
    """Generate a config dictionary for given algorithm and dataset."""
    k_values = DATASET_K_VALUES.get(dataset, [10, 50, 100, 200, 500])
    defaults = ALGORITHM_DEFAULTS.get(algorithm, {})

    config = {
        "name": dataset,
        "data_path": f"datasets/{dataset}.txt",
        "k_values": k_values,
        "num_runs": num_runs,
        "output_csv": f"results/{algorithm}_{dataset}.csv"
    }

    # Add algorithm-specific parameters
    if defaults.get("m_values"):
        config["m_values"] = defaults["m_values"]
    if defaults.get("ef_values"):
        config["ef_values"] = defaults["ef_values"]
    if defaults.get("alpha_values"):
        config["alpha_values"] = defaults["alpha_values"]

    # Add labels path if it might exist
    config["labels_path"] = f"datasets/{dataset}_labels.txt"

    return config


def main():
    parser = argparse.ArgumentParser(
        description="Generate config files from template",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--algorithm", "-a", type=str, default=None,
                        help=f"Algorithm (default: all). Options: {', '.join(ALGORITHMS)}")
    parser.add_argument("--dataset", "-d", type=str, default=None,
                        help=f"Dataset (default: all). Options: {', '.join(DATASETS)}")
    parser.add_argument("--runs", type=int, default=5,
                        help="Number of runs per config (default: 5)")
    parser.add_argument("--clean", action="store_true",
                        help="Remove existing generated configs first")
    parser.add_argument("--list", action="store_true",
                        help="List available algorithms and datasets")

    args = parser.parse_args()

    if args.list:
        print("Algorithms:", ", ".join(ALGORITHMS))
        print("Datasets:", ", ".join(DATASETS))
        return

    # Determine what to generate
    algorithms = [args.algorithm] if args.algorithm else ALGORITHMS
    datasets = [args.dataset] if args.dataset else DATASETS

    # Validate
    for algo in algorithms:
        if algo not in ALGORITHMS:
            print(f"ERROR: Unknown algorithm '{algo}'")
            print(f"Available: {', '.join(ALGORITHMS)}")
            return

    for ds in datasets:
        if ds not in DATASETS:
            print(f"ERROR: Unknown dataset '{ds}'")
            print(f"Available: {', '.join(DATASETS)}")
            return

    # Clean if requested
    if args.clean and OUTPUT_DIR.exists():
        import shutil
        shutil.rmtree(OUTPUT_DIR)
        print(f"Cleaned: {OUTPUT_DIR}")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Generate configs
    count = 0
    for algo in algorithms:
        for ds in datasets:
            config = generate_config(algo, ds, args.runs)
            output_path = OUTPUT_DIR / f"{algo}_{ds}.json"

            with open(output_path, 'w') as f:
                json.dump(config, f, indent=2)

            count += 1

    print(f"Generated {count} config files in {OUTPUT_DIR}/")

    # Print example
    if count > 0:
        example = list(OUTPUT_DIR.glob("*.json"))[0]
        print(f"\nExample ({example.name}):")
        with open(example) as f:
            print(f.read())


if __name__ == "__main__":
    main()
