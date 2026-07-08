"""I/O utilities for saving and loading experiment results."""

import os
import json
import pickle
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import torch


def save_experiment_results(
    results: Union[Dict[str, Any], List[Dict[str, Any]]],
    experiment_name: str,
    artifacts_dir: str = "artifacts",
    save_pickle: bool = True,
    save_json: bool = True,
    optimal_value: Optional[float] = None,
    verbose: bool = True
) -> Dict[str, str]:
    """Save experiment results to artifacts directory.
    
    Args:
        results: Single result dict or dict of multiple results from experiments
        experiment_name: Name of the experiment (used for folder name)
        artifacts_dir: Base directory for artifacts
        save_pickle: Whether to save pickle file (preserves torch tensors)
        save_json: Whether to save JSON summary (human-readable)
        optimal_value: Known optimal value for regret calculation
        verbose: Whether to print save locations
        
    Returns:
        Dictionary with paths to saved files
    """
    # Create experiment-specific directory
    exp_dir = Path(artifacts_dir) / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    saved_paths = {}
    
    # Handle single result dict vs multiple results
    if isinstance(results, dict) and 'X' in results and 'Y_observed' in results:
        # Single result - wrap in dict
        results = {'default': results}
    elif isinstance(results, list):
        # List of results - convert to dict with numeric keys
        results = {f'run_{i}': r for i, r in enumerate(results)}
    
    # Save pickle file (full data with torch tensors)
    if save_pickle:
        pickle_path = exp_dir / "results.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(results, f)
        saved_paths['pickle'] = str(pickle_path)
        if verbose:
            print(f"Results saved to: {pickle_path}")
    
    # Save JSON summary (human-readable)
    if save_json:
        summary_data = {}
        
        for name, result in results.items():
            summary = extract_result_summary(result, optimal_value)
            summary_data[name] = summary
        
        json_path = exp_dir / "summary.json"
        with open(json_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
        saved_paths['json'] = str(json_path)
        if verbose:
            print(f"Summary saved to: {json_path}")
    
    saved_paths['directory'] = str(exp_dir)
    return saved_paths


def extract_result_summary(result: Dict[str, Any], optimal_value: Optional[float] = None) -> Dict[str, Any]:
    """Extract key metrics from a result dictionary.
    
    Args:
        result: Result dictionary from ExperimentRunner
        optimal_value: Known optimal value for regret calculation
        
    Returns:
        Dictionary with key metrics (JSON-serializable)
    """
    summary = {}
    
    # Basic metrics
    if 'best_observed_value' in result:
        summary['best_observed_value'] = float(result['best_observed_value'])
    
    if 'best_true_value' in result:
        summary['best_true_value'] = float(result['best_true_value'])
    
    # Parameters
    if 'best_observed_params' in result:
        params = result['best_observed_params']
        if isinstance(params, dict):
            summary['best_observed_params'] = {k: float(v) if isinstance(v, (torch.Tensor, float, int)) else str(v) 
                                              for k, v in params.items()}
    
    if 'best_true_params' in result:
        params = result['best_true_params']
        if isinstance(params, dict):
            summary['best_true_params'] = {k: float(v) if isinstance(v, (torch.Tensor, float, int)) else str(v)
                                          for k, v in params.items()}
    
    # Iteration count
    if 'all_results' in result:
        summary['n_iterations'] = len(result['all_results'])
    elif 'X' in result:
        summary['n_iterations'] = len(result['X'])
    
    # Corruption statistics
    if 'corruption_levels' in result:
        corruption = result['corruption_levels']
        if isinstance(corruption, torch.Tensor):
            summary['corruption_events'] = int((corruption != 0).sum().item())
            summary['total_corruption'] = float(corruption.abs().sum().item())
            summary['max_corruption'] = float(corruption.abs().max().item())
    
    # Regret calculation
    if optimal_value is not None:
        if 'best_true_value' in result:
            summary['simple_regret'] = float(optimal_value - result['best_true_value'])
        
        # Cumulative regret
        if 'Y_true' in result:
            Y_true = result['Y_true']
            if isinstance(Y_true, torch.Tensor):
                instant_regret = optimal_value - Y_true
                summary['cumulative_regret'] = float(instant_regret.sum().item())
                summary['mean_regret'] = float(instant_regret.mean().item())
    
    # Noise statistics
    if 'all_results' in result:
        noise_values = [r.noise for r in result['all_results'] if hasattr(r, 'noise')]
        if noise_values:
            summary['mean_noise'] = float(sum(noise_values) / len(noise_values))
            summary['max_noise'] = float(max(abs(n) for n in noise_values))
    
    return summary


def load_experiment_results(
    experiment_name: str,
    artifacts_dir: str = "artifacts",
    load_pickle: bool = True
) -> Dict[str, Any]:
    """Load experiment results from artifacts directory.
    
    Args:
        experiment_name: Name of the experiment
        artifacts_dir: Base directory for artifacts
        load_pickle: Whether to load from pickle (otherwise JSON)
        
    Returns:
        Dictionary with loaded results
    """
    exp_dir = Path(artifacts_dir) / experiment_name
    
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")
    
    if load_pickle:
        pickle_path = exp_dir / "results.pkl"
        if not pickle_path.exists():
            raise FileNotFoundError(f"Pickle file not found: {pickle_path}")
        
        with open(pickle_path, 'rb') as f:
            return pickle.load(f)
    else:
        json_path = exp_dir / "summary.json"
        if not json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_path}")
        
        with open(json_path, 'r') as f:
            return json.load(f)


def save_comparison_table(
    results_dict: Dict[str, Dict[str, Any]],
    experiment_name: str,
    artifacts_dir: str = "artifacts",
    optimal_value: Optional[float] = None,
    metrics: Optional[List[str]] = None
) -> str:
    """Save a comparison table of multiple experiments.
    
    Args:
        results_dict: Dictionary mapping experiment names to results
        experiment_name: Name for the comparison
        artifacts_dir: Base directory for artifacts
        optimal_value: Known optimal value
        metrics: List of metrics to include (default: all available)
        
    Returns:
        Path to saved CSV file
    """
    import csv
    
    exp_dir = Path(artifacts_dir) / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract summaries
    summaries = {}
    for name, result in results_dict.items():
        summaries[name] = extract_result_summary(result, optimal_value)
    
    # Determine metrics to include
    if metrics is None:
        # Use all metrics from first result
        metrics = list(next(iter(summaries.values())).keys())
    
    # Write CSV
    csv_path = exp_dir / "comparison.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow(['Experiment'] + metrics)
        
        # Data rows
        for name, summary in summaries.items():
            row = [name]
            for metric in metrics:
                value = summary.get(metric, '')
                if isinstance(value, float):
                    row.append(f"{value:.4f}")
                else:
                    row.append(str(value))
            writer.writerow(row)
    
    print(f"Comparison table saved to: {csv_path}")
    return str(csv_path)