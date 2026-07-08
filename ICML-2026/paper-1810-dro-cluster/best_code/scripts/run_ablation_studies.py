"""
FILE: run_ablation_studies.py
INPUT: CLI arguments selecting the ablation, trial count, seed range, and output directory.
OUTPUT: CSV files under results/ablation_results by default.
POS: Command-line reproduction entry point for ICML ablation studies.
NOTE: Update this header and the folder's _MANIFEST.md explicitly if logic changes.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from dro_subspace.experiments import (
    DEFAULT_MAX_OUTER_ITERATIONS,
    DEFAULT_N_EXPERIMENTS,
    DEFAULT_SEED_START,
    run_ablation_suite,
)

DEFAULT_OUTPUT_DIR = Path("results") / "ablation_results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DRO subspace clustering ablation studies.")
    parser.add_argument("--ablation", choices=["rho", "K", "alpha", "all"], default="all")
    parser.add_argument("--n-experiments", type=int, default=DEFAULT_N_EXPERIMENTS)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    parser.add_argument("--n-simulations", type=int, default=1000)
    parser.add_argument("--max-outer-iterations", type=int, default=DEFAULT_MAX_OUTER_ITERATIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    results = run_ablation_suite(
        ablation=args.ablation,
        n_experiments=args.n_experiments,
        seed_start=args.seed_start,
        output_dir=args.output_dir,
        n_simulations=args.n_simulations,
        max_outer_iterations=args.max_outer_iterations,
    )
    for name, frame in results.items():
        logging.info("%s ablation rows: %s", name, len(frame))


if __name__ == "__main__":
    main()
