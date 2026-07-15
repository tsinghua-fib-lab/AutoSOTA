"""Utility functions for the synthetic study."""

import numpy as np
import os
from pathlib import Path
from typing import Optional, Dict, Any
import json
from datetime import datetime


def set_seed(seed: int) -> np.random.Generator:
    """
    Set random seed and return generator.
    
    Args:
        seed: Random seed
    
    Returns:
        NumPy random generator
    """
    return np.random.default_rng(seed)


def create_output_dir(
    base_dir: str,
    experiment_name: str,
    timestamp: bool = True
) -> Path:
    """
    Create output directory for experiment.
    
    Args:
        base_dir: Base results directory
        experiment_name: Name of experiment
        timestamp: Whether to add timestamp
    
    Returns:
        Path to created directory
    """
    if timestamp:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name = f"{experiment_name}_{ts}"
    else:
        dir_name = experiment_name
    
    path = Path(base_dir) / dir_name
    path.mkdir(parents=True, exist_ok=True)
    
    return path


def save_results(
    output_dir: Path,
    config: dict,
    z: np.ndarray,
    d: np.ndarray,
    y: np.ndarray,
    annotators: np.ndarray,
    true_scores: np.ndarray,
    results: dict,
    metrics: dict,
    seed: int,
    save_raw: bool = True,
    save_confusion: bool = True,
    save_posteriors: bool = True
):
    """
    Save all experiment results.
    
    Args:
        output_dir: Output directory
        config: Configuration dictionary
        z: True labels
        d: Item ambiguity
        y: Observed labels
        annotators: Annotator IDs
        true_scores: Ground truth scores
        results: Aggregation results
        metrics: Evaluation metrics
        seed: Random seed used
        save_raw: Whether to save raw data
        save_confusion: Whether to save confusion matrices
        save_posteriors: Whether to save posteriors
    """
    output_dir = Path(output_dir)
    
    # Save config
    config_with_seed = {**config, "seed": seed}
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config_with_seed, f, indent=2)
    
    # Save metrics
    with open(output_dir / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save scores
    np.savez(
        output_dir / "scores.npz",
        true_scores=true_scores,
        mv_scores=results["mv_scores"],
        ds_scores=results["ds_scores"],
        pec_scores=results["pec_scores"]
    )
    
    # Save raw data
    if save_raw:
        np.savez(
            output_dir / "raw_data.npz",
            z=z,
            d=d,
            y=y,
            annotators=annotators,
            mv_labels=results["mv_labels"],
            ds_labels=results["ds_labels"]
        )
    
    # Save confusion matrices
    if save_confusion:
        np.savez(
            output_dir / "confusion_matrices.npz",
            estimated=results["confusion_matrices"],
            class_prior=results["class_prior"]
        )
    
    # Save posteriors
    if save_posteriors:
        np.savez_compressed(
            output_dir / "posteriors.npz",
            gamma=results["ds_gamma"]
        )


def load_results(output_dir: Path) -> dict:
    """
    Load saved experiment results.
    
    Args:
        output_dir: Directory containing results
    
    Returns:
        Dictionary with all loaded data
    """
    output_dir = Path(output_dir)
    
    results = {}
    
    # Load config
    with open(output_dir / "config.json", 'r') as f:
        results["config"] = json.load(f)
    
    # Load metrics
    with open(output_dir / "metrics.json", 'r') as f:
        results["metrics"] = json.load(f)
    
    # Load scores
    scores = np.load(output_dir / "scores.npz")
    results["true_scores"] = scores["true_scores"]
    results["mv_scores"] = scores["mv_scores"]
    results["ds_scores"] = scores["ds_scores"]
    results["pec_scores"] = scores["pec_scores"]
    
    # Load raw data if exists
    raw_path = output_dir / "raw_data.npz"
    if raw_path.exists():
        raw = np.load(raw_path)
        results["z"] = raw["z"]
        results["d"] = raw["d"]
        results["y"] = raw["y"]
        results["annotators"] = raw["annotators"]
    
    # Load confusion matrices if exists
    conf_path = output_dir / "confusion_matrices.npz"
    if conf_path.exists():
        conf = np.load(conf_path)
        results["confusion_matrices"] = conf["estimated"]
        results["class_prior"] = conf["class_prior"]
    
    # Load posteriors if exists
    post_path = output_dir / "posteriors.npz"
    if post_path.exists():
        post = np.load(post_path)
        results["gamma"] = post["gamma"]
    
    return results


def format_metrics_table(metrics_list: list, param_name: str, param_values: list) -> str:
    """
    Format metrics as a table string.
    
    Args:
        metrics_list: List of metric dictionaries
        param_name: Name of varied parameter
        param_values: Values of varied parameter
    
    Returns:
        Formatted table string
    """
    methods = ["mv", "ds", "pec"]
    method_names = {
        "mv": "Majority Vote",
        "ds": "Dawid-Skene",
        "pec": "Post. Exp. Credit"
    }
    
    lines = []
    
    # Header
    header = f"| {param_name:>15} |"
    for method in methods:
        header += f" {method_names[method]:^18} MSE |"
        header += f" {method_names[method]:^18} τ |"
    lines.append(header)
    lines.append("|" + "-" * (len(header) - 2) + "|")
    
    # Rows
    for val, metrics in zip(param_values, metrics_list):
        row = f"| {str(val):>15} |"
        for method in methods:
            mse = metrics[method]["mse"]
            tau = metrics[method]["kendall_tau"]
            row += f" {mse:^18.4f} |"
            row += f" {tau:^18.3f} |"
        lines.append(row)
    
    return "\n".join(lines)


def print_single_run_summary(metrics: dict, true_scores: np.ndarray, results: dict):
    """Print summary of a single run."""
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    
    print("\nGround Truth Scores:")
    for i, score in enumerate(true_scores):
        print(f"  Agent {i}: {score:.4f}")
    
    print("\n" + "-" * 60)
    print("Method Comparison:")
    print("-" * 60)
    
    methods = [
        ("Majority Vote", "mv", results["mv_scores"]),
        ("Dawid-Skene", "ds", results["ds_scores"]),
        ("Posterior Exp. Credit", "pec", results["pec_scores"]),
    ]
    
    print(f"\n{'Method':<25} {'MSE':>10} {'Kendall τ':>12} {'Rank Acc':>10}")
    print("-" * 60)
    
    for name, key, scores in methods:
        m = metrics[key]
        print(f"{name:<25} {m['mse']:>10.6f} {m['kendall_tau']:>12.3f} {m['ranking_accuracy']:>10.3f}")
    
    print("\n" + "=" * 60)
