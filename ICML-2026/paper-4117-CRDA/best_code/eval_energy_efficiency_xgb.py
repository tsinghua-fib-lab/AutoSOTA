#!/usr/bin/env python3
"""Evaluation script: Energy Efficiency + XGBoost CRDA experiment.

Reproduces the MSE metric from CRDA paper (Table 1, Table 12).
Target: Energy Efficiency (n=765), XGBoostRegressor, 15 seeds.
"""

import sys, os, json
sys.path.insert(0, "/repo")

def parse_existing_results():
    """Parse previously-saved experiment results and print the key metrics."""
    import glob
    # Find the most recent results.csv in EnergyEfficiency experiments
    result_files = sorted(glob.glob("/repo/experiments/EnergyEfficiency/*/results.csv"))
    if not result_files:
        print("ERROR: No results found. Run the experiment first.")
        sys.exit(1)

    import pandas as pd
    latest_file = result_files[-1]
    results = pd.read_csv(latest_file)

    print(f"Reading results from: {latest_file}")
    print("=" * 80)

    for _, row in results.iterrows():
        print(f"RESULT: {row['metric']} = {row['mean']:.6f} +/- {row['std']:.6f}")

    baseline = results[results["metric"] == "mse"]
    augmented = results[results["metric"] == "aug_mse"]
    if not baseline.empty and not augmented.empty:
        print(f"\nFINAL: baseline_mse={baseline['mean'].values[0]:.6f}")
        print(f"FINAL: crda_mse={augmented['mean'].values[0]:.6f}")

def run_experiment():
    """Run the full CRDA experiment for Energy Efficiency with XGBoost."""
    import time
    from src.utils.config import Config
    from src.utils.logger import Logger
    from src.experiment import Experiment

    config = Config(
        baseline="xgboost",
        dataset_path="/repo/data/EnergyEfficiency.csv",
        results_dir="/repo/experiments/EnergyEfficiency",
        sample_sizes=[765],
        save_params=True,
        hyperparam_tune=True,
        ignore_filter=True,
        n_crda_iterations=1,
    )

    print("=" * 80)
    print("CRDA EVALUATION: Energy Efficiency, XGBoost, sample_size=765")
    print(f"  test_size={config.test_size}, num_seeds={config.num_seeds}")
    print(f"  hyperparam_tune={config.hyperparam_tune}")
    print("=" * 80)

    logger = Logger(log_to_file=False, log_to_console=True)
    experiment = Experiment(config, logger)
    t0 = time.perf_counter()
    results = experiment.run()
    elapsed = time.perf_counter() - t0
    print(f"\nElapsed: {elapsed:.1f}s")

    if results is not None:
        for _, row in results.iterrows():
            print(f"RESULT: {row['metric']} = {row['mean']:.6f} +/- {row['std']:.6f}")
        baseline = results[results["metric"] == "mse"]
        augmented = results[results["metric"] == "aug_mse"]
        if not baseline.empty and not augmented.empty:
            print(f"\nFINAL: baseline_mse={baseline['mean'].values[0]:.6f}")
            print(f"FINAL: crda_mse={augmented['mean'].values[0]:.6f}")
    else:
        print("ERROR: No results produced")
        sys.exit(1)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--parse-only", action="store_true",
                       help="Parse existing results instead of re-running")
    parser.add_argument("--results-dir", type=str, default="/repo/experiments/EnergyEfficiency",
                       help="Directory to save/read results")
    args = parser.parse_args()

    if args.parse_only:
        parse_existing_results()
    else:
        run_experiment()
