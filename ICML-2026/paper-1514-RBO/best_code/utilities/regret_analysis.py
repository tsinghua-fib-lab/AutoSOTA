"""Regret analysis and performance metrics utilities for Bayesian Optimization.

This module provides a centralized location for all regret computation and
performance analysis functions, working directly with EvaluationResult objects.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from bo_framework.base.evaluator import EvaluationResult


@dataclass
class PerformanceMetrics:
    """Container for comprehensive BO performance metrics."""

    # Best points and values
    best_observed_value: float
    best_observed_index: int
    best_observed_params: Dict[str, Any]

    best_true_value: float
    best_true_index: int
    best_true_params: Dict[str, Any]

    # Regret metrics
    simple_regret: np.ndarray  # Shape: (n_iterations,)
    instantaneous_regret: np.ndarray  # Shape: (n_iterations,)
    cumulative_regret: np.ndarray  # Shape: (n_iterations,)

    # Corruption statistics
    n_corrupted: int
    total_corruption: float
    corruption_indices: List[int]

    # Final regret values
    final_simple_regret: float = None
    final_cumulative_regret: float = None

    def __post_init__(self):
        """Compute final regret values."""
        if self.simple_regret is not None and len(self.simple_regret) > 0:
            self.final_simple_regret = self.simple_regret[-1]
        if self.cumulative_regret is not None and len(self.cumulative_regret) > 0:
            self.final_cumulative_regret = self.cumulative_regret[-1]


def extract_true_values(results: List[EvaluationResult]) -> torch.Tensor:
    """Extract true function values from evaluation results.

    Args:
        results: List of EvaluationResult objects from BO iterations

    Returns:
        Tensor of true function values
    """
    return torch.tensor([r.y_true for r in results], dtype=torch.double)


def extract_observed_values(results: List[EvaluationResult]) -> torch.Tensor:
    """Extract observed values (including corruption) from evaluation results.

    Args:
        results: List of EvaluationResult objects from BO iterations

    Returns:
        Tensor of observed values (what BO actually sees)
    """
    return torch.tensor([r.y_observed for r in results], dtype=torch.double)


def extract_noisy_values(results: List[EvaluationResult]) -> torch.Tensor:
    """Extract noisy values (before corruption) from evaluation results.

    Args:
        results: List of EvaluationResult objects from BO iterations

    Returns:
        Tensor of values with observation noise but before corruption
    """
    return torch.tensor([r.y_noisy for r in results], dtype=torch.double)


def extract_corruption_levels(results: List[EvaluationResult]) -> torch.Tensor:
    """Extract corruption levels from evaluation results.

    Args:
        results: List of EvaluationResult objects from BO iterations

    Returns:
        Tensor of corruption amounts at each point
    """
    return torch.tensor([r.corruption for r in results], dtype=torch.double)


def extract_noise_levels(results: List[EvaluationResult]) -> torch.Tensor:
    """Extract noise levels from evaluation results.

    Args:
        results: List of EvaluationResult objects from BO iterations

    Returns:
        Tensor of noise amounts at each point
    """
    return torch.tensor([r.noise for r in results], dtype=torch.double)


def extract_parameters(results: List[EvaluationResult]) -> List[Dict[str, Any]]:
    """Extract parameter dictionaries from evaluation results.

    Args:
        results: List of EvaluationResult objects from BO iterations

    Returns:
        List of parameter dictionaries
    """
    return [r.params for r in results]


def extract_trajectories(results: List[EvaluationResult]) -> Dict[str, torch.Tensor]:
    """Extract all value trajectories from evaluation results.

    Convenience function that calls all individual extraction functions.

    Args:
        results: List of EvaluationResult objects from BO iterations

    Returns:
        Dictionary containing:
            - Y_true: True function values
            - Y_observed: Values observed by BO (including corruption)
            - Y_noisy: Values with observation noise (before corruption)
            - corruption_levels: Amount of corruption at each point
            - noise_levels: Amount of noise at each point
    """
    return {
        "Y_true": extract_true_values(results),
        "Y_observed": extract_observed_values(results),
        "Y_noisy": extract_noisy_values(results),
        "corruption_levels": extract_corruption_levels(results),
        "noise_levels": extract_noise_levels(results),
    }


def compute_best_so_far(results: List[EvaluationResult]) -> torch.Tensor:
    """Compute best value found so far at each iteration.

    Args:
        results: List of EvaluationResult objects

    Returns:
        Tensor of best values found up to each iteration (cumulative maximum)
    """
    Y_true = extract_true_values(results)
    return torch.cummax(Y_true, dim=0)[0]


def compute_regret(
    results: List[EvaluationResult], optimal_value: float, regret_type: str = "simple"
) -> np.ndarray:
    """Compute regret from evaluation results.

    Args:
        results: List of EvaluationResult objects
        optimal_value: Known optimal value of the objective
        regret_type: Type of regret to compute:
            - 'simple': Best so far vs optimal (for minimization: min_so_far - optimal)
            - 'instantaneous': Regret at each iteration
            - 'cumulative': Cumulative sum of instantaneous regret

    Returns:
        Numpy array of regret values at each iteration
    """
    # Extract true values using the dedicated function
    Y_true = extract_true_values(results)

    if regret_type == "simple":
        # Simple regret: difference between best found so far and optimal
        # For maximization: optimal - best_so_far
        best_so_far = compute_best_so_far(results)
        regret = optimal_value - best_so_far.numpy()

    elif regret_type == "instantaneous":
        # Instantaneous regret at each point
        regret = optimal_value - Y_true.numpy()

    elif regret_type == "cumulative":
        # Cumulative regret over time
        instant_regret = optimal_value - Y_true.numpy()
        regret = np.cumsum(instant_regret)

    else:
        raise ValueError(
            f"Unknown regret_type: {regret_type}. "
            f"Choose from 'simple', 'instantaneous', 'cumulative'"
        )

    return regret


def find_best_points(
    results: List[EvaluationResult],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Find best points according to observed and true values.

    Args:
        results: List of EvaluationResult objects
        search_space: Optional SearchSpace object for decoding parameters

    Returns:
        Tuple of (best_observed_info, best_true_info) dictionaries containing:
            - value: Best value found
            - index: Index in results list
            - params: Parameter dictionary
            - result: The EvaluationResult object
    """
    if not results:
        raise ValueError("Results list is empty")

    # Extract values using dedicated functions
    Y_observed = extract_observed_values(results)
    Y_true = extract_true_values(results)

    # Find best indices (assuming maximization)
    best_obs_idx = Y_observed.argmax().item()
    best_true_idx = Y_true.argmax().item()

    # Compile results
    best_observed_info = {
        "value": Y_observed[best_obs_idx].item(),
        "index": best_obs_idx,
        "params": results[best_obs_idx].x,  # Use 'x' instead of 'params'
        "result": results[best_obs_idx],
    }

    best_true_info = {
        "value": Y_true[best_true_idx].item(),
        "index": best_true_idx,
        "params": results[best_true_idx].x,  # Use 'x' instead of 'params'
        "result": results[best_true_idx],
    }

    return best_observed_info, best_true_info


def compute_corruption_statistics(results: List[EvaluationResult]) -> Dict[str, Any]:
    """Compute statistics about corruption in the results.

    Args:
        results: List of EvaluationResult objects

    Returns:
        Dictionary with corruption statistics:
            - n_corrupted: Number of corrupted points
            - total_corruption: Sum of absolute corruption values
            - corruption_indices: List of indices where corruption occurred
            - max_corruption: Maximum corruption magnitude
            - mean_corruption: Mean corruption (excluding non-corrupted)
    """
    corruption_levels = extract_corruption_levels(results)

    # Find corrupted points
    corrupted_mask = corruption_levels != 0
    corruption_indices = torch.where(corrupted_mask)[0].tolist()

    stats = {
        "n_corrupted": corrupted_mask.sum().item(),
        "total_corruption": corruption_levels.abs().sum().item(),
        "corruption_indices": corruption_indices,
        "max_corruption": corruption_levels.abs().max().item()
        if len(corruption_levels) > 0
        else 0.0,
        "mean_corruption": 0.0,
    }

    # Compute mean corruption for corrupted points only
    if stats["n_corrupted"] > 0:
        stats["mean_corruption"] = corruption_levels[corrupted_mask].abs().mean().item()

    return stats


def compute_performance_metrics(
    results: List[EvaluationResult],
    optimal_value: Optional[float] = None,
) -> PerformanceMetrics:
    """Compute comprehensive performance metrics from BO results.

    This is the main entry point for performance analysis, combining all
    individual metric computations into a single PerformanceMetrics object.

    Args:
        results: List of EvaluationResult objects from BO
        optimal_value: Known optimal value (for regret computation)
        search_space: Optional SearchSpace for parameter decoding

    Returns:
        PerformanceMetrics object with all computed metrics
    """
    if not results:
        raise ValueError("Results list is empty")

    # Find best points
    best_obs_info, best_true_info = find_best_points(results)

    # Compute regret if optimal value is known
    simple_regret = None
    instantaneous_regret = None
    cumulative_regret = None

    if optimal_value is not None:
        simple_regret = compute_regret(results, optimal_value, "simple")
        instantaneous_regret = compute_regret(results, optimal_value, "instantaneous")
        cumulative_regret = compute_regret(results, optimal_value, "cumulative")

    # Compute corruption statistics
    corruption_stats = compute_corruption_statistics(results)

    # Create metrics object
    metrics = PerformanceMetrics(
        best_observed_value=best_obs_info["value"],
        best_observed_index=best_obs_info["index"],
        best_observed_params=best_obs_info["params"],
        best_true_value=best_true_info["value"],
        best_true_index=best_true_info["index"],
        best_true_params=best_true_info["params"],
        simple_regret=simple_regret,
        instantaneous_regret=instantaneous_regret,
        cumulative_regret=cumulative_regret,
        n_corrupted=corruption_stats["n_corrupted"],
        total_corruption=corruption_stats["total_corruption"],
        corruption_indices=corruption_stats["corruption_indices"],
    )

    return metrics


def print_performance_summary(
    metrics: PerformanceMetrics, experiment_name: Optional[str] = None
) -> None:
    """Print a formatted summary of performance metrics.

    Args:
        metrics: PerformanceMetrics object
        experiment_name: Optional name for the experiment
    """
    if experiment_name:
        print(f"\n{'=' * 80}")
        print(f"Performance Summary: {experiment_name}")
        print("=" * 80)
    else:
        print(f"\n{'=' * 80}")
        print("Performance Summary")
        print("=" * 80)

    # Best values
    print("\nBest Values Found:")
    print(f"  Best observed value (BO perspective): {metrics.best_observed_value:.6f}")
    print(f"  Best observed params: {metrics.best_observed_params}")
    print(f"  Best true value (actual best): {metrics.best_true_value:.6f}")
    print(f"  Best true params: {metrics.best_true_params}")

    # Regret summary
    if metrics.final_simple_regret is not None:
        print("\nRegret Analysis:")
        print(f"  Final simple regret: {metrics.final_simple_regret:.6f}")
        print(f"  Final cumulative regret: {metrics.final_cumulative_regret:.6f}")

    # Corruption statistics
    if metrics.n_corrupted > 0:
        print("\nCorruption Statistics:")
        print(f"  Points corrupted: {metrics.n_corrupted}")
        print(f"  Total corruption magnitude: {metrics.total_corruption:.4f}")
        print(f"  Corruption at iterations: {metrics.corruption_indices}")

    print("=" * 80)


def compare_experiments(
    results_dict: Dict[str, List[EvaluationResult]],
    optimal_value: Optional[float] = None,
) -> Dict[str, PerformanceMetrics]:
    """Compare multiple experiments and return their metrics.

    Args:
        results_dict: Dictionary mapping experiment names to result lists
        optimal_value: Known optimal value

    Returns:
        Dictionary mapping experiment names to PerformanceMetrics
    """
    metrics_dict = {}

    for name, results in results_dict.items():
        metrics = compute_performance_metrics(
            results,
            optimal_value,
        )
        metrics_dict[name] = metrics

    return metrics_dict


def print_comparison_table(
    metrics_dict: Dict[str, PerformanceMetrics],
    show_regret: bool = True,
    show_corruption: bool = True,
) -> None:
    """Print a comparison table of multiple experiments.

    Args:
        metrics_dict: Dictionary mapping experiment names to PerformanceMetrics
        show_regret: Whether to show regret columns
        show_corruption: Whether to show corruption statistics
    """
    print("\n" + "=" * 140)
    print("EXPERIMENT COMPARISON")
    print("=" * 140)

    # Build header
    headers = ["Experiment", "Best Observed", "Best True"]
    if show_regret:
        headers.extend(["Simple Regret", "Cumul. Regret"])
    if show_corruption:
        headers.append("# Corrupted")

    # Print header
    # Increased width to 30 to handle large regret values with std
    header_format = "{:<20}" + "{:<30}" * (len(headers) - 1)
    print(header_format.format(*headers))
    print("-" * 140)

    # Print rows
    for name, metrics in metrics_dict.items():
        row = [
            name[:20],  # Truncate long names
            f"{metrics.best_observed_value:.4f}",
            f"{metrics.best_true_value:.4f}",
        ]

        if show_regret:
            if metrics.final_simple_regret is not None:
                row.extend(
                    [
                        f"{metrics.final_simple_regret:.4f}",
                        f"{metrics.final_cumulative_regret:.4f}",
                    ]
                )
            else:
                row.extend(["N/A", "N/A"])

        if show_corruption:
            row.append(str(metrics.n_corrupted))

        print(header_format.format(*row))

    print("=" * 140)


def compare_experiments_multiseed(
    results_dict: Dict[str, List[List[EvaluationResult]]],
    optimal_value: Optional[float] = None,
) -> Dict[str, Dict[str, Any]]:
    """Compare multiple experiments across multiple seeds and return aggregated metrics.

    Args:
        results_dict: Dictionary mapping experiment names to list of seed results
                     (each seed result is a List[EvaluationResult])
        optimal_value: Known optimal value

    Returns:
        Dictionary mapping experiment names to aggregated statistics including:
            - mean and std for each metric
            - individual seed metrics
    """
    aggregated_metrics = {}

    for name, all_seeds_results in results_dict.items():
        # Compute metrics for each seed
        seed_metrics = []
        for seed_results in all_seeds_results:
            metrics = compute_performance_metrics(
                seed_results,
                optimal_value,
            )
            seed_metrics.append(metrics)

        # Aggregate across seeds
        n_seeds = len(seed_metrics)

        # Extract values for aggregation
        best_observed_values = [m.best_observed_value for m in seed_metrics]
        best_true_values = [m.best_true_value for m in seed_metrics]
        final_simple_regrets = [
            m.final_simple_regret
            for m in seed_metrics
            if m.final_simple_regret is not None
        ]
        final_cumulative_regrets = [
            m.final_cumulative_regret
            for m in seed_metrics
            if m.final_cumulative_regret is not None
        ]
        n_corrupted_list = [m.n_corrupted for m in seed_metrics]

        # Compute statistics
        aggregated_metrics[name] = {
            "n_seeds": n_seeds,
            "best_observed_mean": np.mean(best_observed_values),
            "best_observed_std": np.std(best_observed_values),
            "best_true_mean": np.mean(best_true_values),
            "best_true_std": np.std(best_true_values),
            "final_simple_regret_mean": np.mean(final_simple_regrets)
            if final_simple_regrets
            else None,
            "final_simple_regret_std": np.std(final_simple_regrets)
            if final_simple_regrets
            else None,
            "final_cumulative_regret_mean": np.mean(final_cumulative_regrets)
            if final_cumulative_regrets
            else None,
            "final_cumulative_regret_std": np.std(final_cumulative_regrets)
            if final_cumulative_regrets
            else None,
            "n_corrupted_mean": np.mean(n_corrupted_list),
            "n_corrupted_std": np.std(n_corrupted_list),
            "seed_metrics": seed_metrics,  # Keep individual seed metrics for reference
        }

    return aggregated_metrics


def print_comparison_table_multiseed(
    aggregated_metrics: Dict[str, Dict[str, Any]],
    show_regret: bool = True,
    show_corruption: bool = True,
    show_std: bool = True,
) -> None:
    """Print a comparison table of multiple experiments with multi-seed statistics.

    Args:
        aggregated_metrics: Dictionary from compare_experiments_multiseed
        show_regret: Whether to show regret columns
        show_corruption: Whether to show corruption statistics
        show_std: Whether to show standard deviations
    """
    print("\n" + "=" * 160)
    print("EXPERIMENT COMPARISON (AVERAGED ACROSS SEEDS)")
    print("=" * 160)

    # Build header
    headers = ["Experiment", "Seeds", "Best Observed", "Best True"]
    if show_regret:
        headers.extend(["Simple Regret", "Cumul. Regret"])
    if show_corruption:
        headers.append("# Corrupted")

    # Determine format based on whether we show std
    # Increased width to 30 to handle large regret values with std
    if show_std:
        header_format = "{:<20}{:<8}" + "{:<30}" * (len(headers) - 2)
    else:
        header_format = "{:<20}{:<8}" + "{:<20}" * (len(headers) - 2)

    # Print header
    print(header_format.format(*headers))
    print("-" * 160)

    # Sort by average cumulative regret if available, otherwise by simple regret
    sorted_items = sorted(
        aggregated_metrics.items(),
        key=lambda x: (
            x[1].get("final_cumulative_regret_mean", float("inf"))
            if x[1].get("final_cumulative_regret_mean") is not None
            else float("inf")
        ),
    )

    # Print rows
    for name, metrics in sorted_items:
        row = [name[:20], str(metrics["n_seeds"])]

        # Format values with mean ± std
        if show_std:
            row.append(
                f"{metrics['best_observed_mean']:.4f}±{metrics['best_observed_std']:.4f}"
            )
            row.append(
                f"{metrics['best_true_mean']:.4f}±{metrics['best_true_std']:.4f}"
            )
        else:
            row.append(f"{metrics['best_observed_mean']:.4f}")
            row.append(f"{metrics['best_true_mean']:.4f}")

        if show_regret:
            if metrics["final_simple_regret_mean"] is not None:
                if show_std:
                    row.extend(
                        [
                            f"{metrics['final_simple_regret_mean']:.4f}±{metrics['final_simple_regret_std']:.4f}",
                            f"{metrics['final_cumulative_regret_mean']:.4f}±{metrics['final_cumulative_regret_std']:.4f}",
                        ]
                    )
                else:
                    row.extend(
                        [
                            f"{metrics['final_simple_regret_mean']:.4f}",
                            f"{metrics['final_cumulative_regret_mean']:.4f}",
                        ]
                    )
            else:
                row.extend(["N/A", "N/A"])

        if show_corruption:
            if show_std:
                row.append(
                    f"{metrics['n_corrupted_mean']:.1f}±{metrics['n_corrupted_std']:.1f}"
                )
            else:
                row.append(f"{metrics['n_corrupted_mean']:.1f}")

        print(header_format.format(*row))

    print("=" * 160)
