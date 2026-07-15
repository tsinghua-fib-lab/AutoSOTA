#!/usr/bin/env python3
"""Run experiments across multiple config files with full reproducibility."""

import argparse
import sys
import os
from pathlib import Path
from typing import List
import json
from multiprocessing import Pool, cpu_count
from functools import partial

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import Config, load_config
from src.agents import generate_true_labels, generate_item_ambiguity, compute_ground_truth_scores
from src.annotators import AnnotatorPool
from src.labeling import generate_observed_labels
from src.aggregation import aggregate_all_methods
from src.metrics import evaluate_all_methods
from src.utils import set_seed, create_output_dir


def run_single_repetition(config: Config, seed: int, compute_stability: bool = False) -> dict:
    """
    Run a single repetition and return all data.
    
    Returns dict with:
        - seed
        - z (true labels)
        - d (item ambiguity)
        - y (observed labels)
        - annotators
        - true_scores
        - results (mv_scores, ds_scores, pec_scores, confusion_matrices, etc.)
        - metrics
    """
    rng = set_seed(seed)
    
    # Generate data
    z = generate_true_labels(
        config.n_agents, config.n_items, config.agent_qualities,
        config.partial_correct_prob, rng
    )
    true_scores = compute_ground_truth_scores(z, config.credit_mapping)
    
    _, d = generate_item_ambiguity(
        config.n_agents, config.n_items, config.hard_item_prob,
        tuple(config.easy_beta_params), tuple(config.hard_beta_params), rng
    )
    
    annotator_pool = AnnotatorPool(
        config.n_annotators, config.annotator_distribution, rng
    )
    
    y, annotators, _ = generate_observed_labels(
        z, d, annotator_pool, config.labels_per_item, rng
    )
    
    # Aggregate
    results = aggregate_all_methods(
        y, annotators, config.n_annotators, config.credit_mapping
    )
    
    # Metrics
    n_stab = config.n_stability_subsamples if compute_stability else 0
    metrics = evaluate_all_methods(
        results, true_scores, y, annotators, config.n_annotators,
        config.credit_mapping, config.subsample_labels, n_stab, rng
    )
    
    return {
        "seed": seed,
        "z": z,
        "d": d,
        "y": y,
        "annotators": annotators,
        "true_scores": true_scores,
        "mv_scores": results["mv_scores"],
        "ds_scores": results["ds_scores"],
        "pec_scores": results["pec_scores"],
        "mv_labels": results["mv_labels"],
        "ds_labels": results["ds_labels"],
        "confusion_matrices": results["confusion_matrices"],
        "posteriors": results["ds_gamma"],
        "metrics": metrics,
    }


def _run_single_rep_wrapper(args):
    """Wrapper for parallel execution."""
    config_dict, seed, compute_stability = args
    # Reconstruct config from dict (can't pickle Config directly)
    config = Config(**config_dict)
    return run_single_repetition(config, seed, compute_stability)


def run_repetitions(
    config: Config,
    n_repetitions: int,
    base_seed: int = 0,
    verbose: bool = True,
    save_raw: bool = True,
    compute_stability: bool = False,
    n_workers: int = None
) -> dict:
    """
    Run multiple repetitions (optionally in parallel).
    
    Returns dict with:
        - config
        - all_metrics (list of metrics per repetition)
        - aggregated_metrics (mean ± std)
        - raw_data (list of full data per repetition, if save_raw=True)
    """
    if n_workers is None:
        n_workers = 1
    
    seeds = [base_seed + rep for rep in range(n_repetitions)]
    config_dict = config.to_dict()
    
    if n_workers > 1:
        # Parallel execution
        args_list = [(config_dict, seed, compute_stability) for seed in seeds]
        
        if verbose:
            print(f"  Running {n_repetitions} repetitions on {n_workers} workers...")
        
        with Pool(n_workers) as pool:
            if verbose:
                results = list(tqdm(
                    pool.imap(_run_single_rep_wrapper, args_list),
                    total=n_repetitions,
                    desc="Repetitions",
                    leave=False
                ))
            else:
                results = pool.map(_run_single_rep_wrapper, args_list)
    else:
        # Sequential execution
        results = []
        iterator = seeds
        if verbose:
            iterator = tqdm(iterator, desc="Repetitions", leave=False)
        
        for seed in iterator:
            rep_data = run_single_repetition(config, seed, compute_stability)
            results.append(rep_data)
    
    all_metrics = [r["metrics"] for r in results]
    raw_data = results if save_raw else None
    
    # Aggregate metrics
    aggregated = aggregate_metrics(all_metrics)
    
    return {
        "config": config.to_dict(),
        "all_metrics": all_metrics,
        "aggregated_metrics": aggregated,
        "raw_data": raw_data,
    }


def aggregate_metrics(metrics_list: List[dict]) -> dict:
    """Aggregate metrics across repetitions (mean ± std)."""
    methods = ["mv", "ds", "pec"]
    metric_names = ["mse", "kendall_tau", "ranking_accuracy"]
    
    aggregated = {}
    
    for method in methods:
        aggregated[method] = {}
        for metric in metric_names:
            values = [m[method][metric] for m in metrics_list]
            aggregated[method][f"{metric}_mean"] = float(np.mean(values))
            aggregated[method][f"{metric}_std"] = float(np.std(values))
            aggregated[method][f"{metric}_se"] = float(np.std(values) / np.sqrt(len(values)))
        
        # Stability if available
        if "stability_mean" in metrics_list[0][method]:
            values = [m[method]["stability_mean"] for m in metrics_list]
            aggregated[method]["stability_mean"] = float(np.mean(values))
            aggregated[method]["stability_std"] = float(np.std(values))
            aggregated[method]["stability_se"] = float(np.std(values) / np.sqrt(len(values)))
    
    return aggregated


def detect_varying_param(configs: List[dict]) -> tuple:
    """
    Detect which parameter varies across configs.
    
    Returns (param_name, values) or (None, None) if multiple or none vary.
    """
    if len(configs) < 2:
        return None, None
    
    # Parameters to check
    params_to_check = [
        ("hard_item_prob", lambda c: c["hard_item_prob"]),
        ("labels_per_item", lambda c: c["labels_per_item"]),
        ("strict_fraction", lambda c: c["annotator_distribution"]["strict"] / c["n_annotators"]),
        ("lenient_fraction", lambda c: c["annotator_distribution"]["lenient"] / c["n_annotators"]),
        ("agent_gap_type", lambda c: c.get("agent_gap_type", "wide")),
    ]
    
    varying_params = []
    
    for param_name, extractor in params_to_check:
        try:
            values = [extractor(c) for c in configs]
            if len(set(values)) > 1:
                varying_params.append((param_name, values))
        except (KeyError, TypeError):
            continue
    
    if len(varying_params) == 1:
        return varying_params[0]
    
    return None, None


def run_configs(
    config_paths: List[str],
    n_repetitions: int = 100,
    base_seed: int = 0,
    save_raw: bool = True,
    compute_stability: bool = False,
    n_workers: int = 1,
    verbose: bool = True
) -> dict:
    """
    Run experiments for multiple config files.
    
    Returns dict in format compatible with generate_plots.py
    """
    all_results = []
    all_configs = []
    
    for config_path in config_paths:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Running: {config_path}")
            print(f"{'='*60}")
        
        config = load_config(config_path)
        result = run_repetitions(
            config, n_repetitions,
            base_seed=base_seed,
            verbose=verbose,
            save_raw=save_raw,
            compute_stability=compute_stability,
            n_workers=n_workers
        )
        result["config_path"] = config_path
        
        all_results.append(result)
        all_configs.append(result["config"])
    
    # Detect varying parameter for plotting
    param_name, param_values = detect_varying_param(all_configs)
    
    # Build output in plot-friendly format
    output = {
        "param": param_name,
        "values": param_values,
        "configs": [r["config_path"] for r in all_results],
        "metrics": [r["aggregated_metrics"] for r in all_results],
        "all_metrics": [r["all_metrics"] for r in all_results],
        "full_configs": all_configs,
    }
    
    if save_raw:
        output["raw_data"] = [r["raw_data"] for r in all_results]
    
    return output


def print_summary(results: dict):
    """Print summary table of results."""
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    if results["param"]:
        print(f"Varying parameter: {results['param']}")
    print(f"{'='*80}")
    
    print(f"\n{'Config':<30} | {'Method':<20} | {'MSE':>12} | {'Kendall τ':>12}")
    print("-" * 80)
    
    for i, config_path in enumerate(results["configs"]):
        config_name = Path(config_path).stem
        m = results["metrics"][i]
        
        for method, name in [("mv", "Majority Vote"), ("ds", "Dawid-Skene"), ("pec", "Post. Exp. Credit")]:
            mse = f"{m[method]['mse_mean']:.4f}±{m[method]['mse_se']:.4f}"
            tau = f"{m[method]['kendall_tau_mean']:.3f}±{m[method]['kendall_tau_se']:.3f}"
            
            config_str = config_name if method == "mv" else ""
            print(f"{config_str:<30} | {name:<20} | {mse:>12} | {tau:>12}")
        print("-" * 80)


def save_results(results: dict, output_dir: Path, save_raw: bool = True):
    """Save all results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save plot-friendly JSON (without raw numpy arrays)
    plot_data = {
        "param": results["param"],
        "values": results["values"],
        "configs": results["configs"],
        "metrics": results["metrics"],
        "full_configs": results["full_configs"],
    }
    with open(output_dir / "results.json", 'w') as f:
        json.dump(plot_data, f, indent=2)
    
    # Save all metrics per repetition as CSV
    rows = []
    for i, config_path in enumerate(results["configs"]):
        config_name = Path(config_path).stem
        for rep, metrics in enumerate(results["all_metrics"][i]):
            for method in ["mv", "ds", "pec"]:
                row = {
                    "config": config_name,
                    "repetition": rep,
                    "method": method,
                    "mse": metrics[method]["mse"],
                    "kendall_tau": metrics[method]["kendall_tau"],
                    "ranking_accuracy": metrics[method]["ranking_accuracy"],
                }
                if results["param"]:
                    row[results["param"]] = results["values"][i]
                rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "all_metrics.csv", index=False)
    
    # Save aggregated metrics
    agg_rows = []
    for i, config_path in enumerate(results["configs"]):
        config_name = Path(config_path).stem
        m = results["metrics"][i]
        for method in ["mv", "ds", "pec"]:
            row = {
                "config": config_name,
                "method": method,
            }
            if results["param"]:
                row[results["param"]] = results["values"][i]
            row.update(m[method])
            agg_rows.append(row)
    
    df_agg = pd.DataFrame(agg_rows)
    df_agg.to_csv(output_dir / "metrics.csv", index=False)
    
    # Save raw data if requested (numpy format)
    if save_raw and "raw_data" in results:
        raw_dir = output_dir / "raw"
        raw_dir.mkdir(exist_ok=True)
        
        for i, config_path in enumerate(results["configs"]):
            config_name = Path(config_path).stem
            config_dir = raw_dir / config_name
            config_dir.mkdir(exist_ok=True)
            
            for rep_data in results["raw_data"][i]:
                seed = rep_data["seed"]
                np.savez_compressed(
                    config_dir / f"seed_{seed:04d}.npz",
                    seed=seed,
                    z=rep_data["z"],
                    d=rep_data["d"],
                    y=rep_data["y"],
                    annotators=rep_data["annotators"],
                    true_scores=rep_data["true_scores"],
                    mv_scores=rep_data["mv_scores"],
                    ds_scores=rep_data["ds_scores"],
                    pec_scores=rep_data["pec_scores"],
                    mv_labels=rep_data["mv_labels"],
                    ds_labels=rep_data["ds_labels"],
                    confusion_matrices=rep_data["confusion_matrices"],
                    posteriors=rep_data["posteriors"],
                )
    
    print(f"\nResults saved to {output_dir}")
    print(f"  - results.json (for plotting)")
    print(f"  - metrics.csv (aggregated)")
    print(f"  - all_metrics.csv (per repetition)")
    if save_raw:
        print(f"  - raw/ (full data per repetition)")


def main():
    parser = argparse.ArgumentParser(
        description="Run experiments across multiple config files"
    )
    parser.add_argument(
        "configs", type=str, nargs="+",
        help="Paths to config YAML files"
    )
    parser.add_argument(
        "--n-repetitions", type=int, default=100,
        help="Number of repetitions per config"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Output directory"
    )
    parser.add_argument(
        "--no-raw", action="store_true",
        help="Don't save raw data (saves disk space)"
    )
    parser.add_argument(
        "--compute-stability", action="store_true",
        help="Compute stability metrics (slower)"
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help=f"Number of parallel workers (default: 1, max: {cpu_count()})"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output"
    )
    
    args = parser.parse_args()
    
    # Run experiments
    n_workers = min(args.workers, cpu_count())
    if n_workers > 1:
        print(f"Using {n_workers} parallel workers")
    
    results = run_configs(
        args.configs,
        n_repetitions=args.n_repetitions,
        save_raw=not args.no_raw,
        compute_stability=args.compute_stability,
        n_workers=n_workers,
        verbose=not args.quiet
    )
    
    # Print summary
    print_summary(results)
    
    # Save results
    output_dir = create_output_dir(args.output_dir, "comparison", timestamp=True)
    save_results(results, output_dir, save_raw=not args.no_raw)


if __name__ == "__main__":
    main()