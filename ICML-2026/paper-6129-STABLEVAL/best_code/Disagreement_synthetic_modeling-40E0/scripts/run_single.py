#!/usr/bin/env python3
"""Run a single experiment with the synthetic disagreement study."""

import argparse
import sys
import os
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from src.config import Config, load_config
from src.agents import generate_true_labels, generate_item_ambiguity, compute_ground_truth_scores
from src.annotators import AnnotatorPool
from src.labeling import generate_observed_labels
from src.aggregation import aggregate_all_methods
from src.metrics import evaluate_all_methods
from src.utils import (
    set_seed, create_output_dir, save_results, print_single_run_summary
)


def run_single_experiment(
    config: Config,
    seed: int,
    verbose: bool = True
) -> dict:
    """
    Run a single experiment.
    
    Args:
        config: Configuration object
        seed: Random seed
        verbose: Whether to print progress
    
    Returns:
        Dictionary with results and metrics
    """
    rng = set_seed(seed)
    
    if verbose:
        print(f"Running experiment with seed={seed}")
        print(f"  Agents: {config.n_agents}, Items: {config.n_items}")
        print(f"  Annotators: {config.n_annotators}, Labels/item: {config.labels_per_item}")
    
    # Step 1: Generate true labels
    if verbose:
        print("  Generating true labels...")
    z = generate_true_labels(
        config.n_agents,
        config.n_items,
        config.agent_qualities,
        config.partial_correct_prob,
        rng
    )
    
    # Compute ground truth scores
    true_scores = compute_ground_truth_scores(z, config.credit_mapping)
    
    # Step 2: Generate item ambiguity
    if verbose:
        print("  Generating item ambiguity...")
    h, d = generate_item_ambiguity(
        config.n_agents,
        config.n_items,
        config.hard_item_prob,
        tuple(config.easy_beta_params),
        tuple(config.hard_beta_params),
        rng
    )
    
    # Step 3: Create annotator pool
    if verbose:
        print("  Creating annotator pool...")
    annotator_pool = AnnotatorPool(
        config.n_annotators,
        config.annotator_distribution,
        rng
    )
    
    # Step 4: Generate observed labels
    if verbose:
        print("  Generating observed labels...")
    y, annotators, _ = generate_observed_labels(
        z, d, annotator_pool, config.labels_per_item, rng
    )
    
    # Step 5: Fit aggregation methods
    if verbose:
        print("  Fitting aggregation methods...")
    results = aggregate_all_methods(
        y, annotators, config.n_annotators, config.credit_mapping
    )
    
    # Step 6: Compute metrics
    if verbose:
        print("  Computing metrics...")
    metrics = evaluate_all_methods(
        results,
        true_scores,
        y,
        annotators,
        config.n_annotators,
        config.credit_mapping,
        config.subsample_labels,
        config.n_stability_subsamples,
        rng
    )
    
    return {
        "z": z,
        "d": d,
        "h": h,
        "y": y,
        "annotators": annotators,
        "true_scores": true_scores,
        "results": results,
        "metrics": metrics,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run single synthetic disagreement experiment"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Output directory"
    )
    parser.add_argument(
        "--name", type=str, default="single_run",
        help="Experiment name"
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Don't save results"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress output"
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    config.output_dir = args.output_dir
    
    # Run experiment
    experiment = run_single_experiment(
        config, args.seed, verbose=not args.quiet
    )
    
    # Print summary
    if not args.quiet:
        print_single_run_summary(
            experiment["metrics"],
            experiment["true_scores"],
            experiment["results"]
        )
    
    # Save results
    if not args.no_save:
        output_dir = create_output_dir(
            args.output_dir,
            f"{args.name}_seed{args.seed}",
            timestamp=True
        )
        
        save_results(
            output_dir,
            config.to_dict(),
            experiment["z"],
            experiment["d"],
            experiment["y"],
            experiment["annotators"],
            experiment["true_scores"],
            experiment["results"],
            experiment["metrics"],
            args.seed,
            save_raw=config.save_raw_data,
            save_confusion=config.save_confusion_matrices,
            save_posteriors=config.save_posteriors
        )
        
        print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()