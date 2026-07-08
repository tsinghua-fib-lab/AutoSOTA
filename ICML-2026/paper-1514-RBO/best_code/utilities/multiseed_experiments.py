"""Utilities for running and analyzing multi-seed experiments."""

import os
import json
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, List
from bo_framework import ExperimentRunner
from bo_framework.base.acquisition import UCBAcquisition
from bo_framework.wrappers.noisy import NoisyEvaluator
from bo_framework.wrappers.corrupted import CorruptedEvaluator
from bo_framework.corruption.adversarial import AdversarialCorruptor
from utilities.plotting import PlotConfig
from utilities.regret_analysis import compute_regret


def set_global_seed(seed: int):
    """Set global seeds for PyTorch, NumPy, and Python random for reproducibility.

    This ensures that BoTorch's optimize_acqf and other unseeded operations
    are deterministic while allowing components to use their own seeds.

    Args:
        seed: Random seed to set globally
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Additional PyTorch settings for full reproducibility
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def run_model_across_seeds(
    model_name: str,
    model_factory,
    model_kwargs: Dict[str, Any],
    scheduler,
    clean_evaluator,
    optimal_point: torch.Tensor,
    search_space,
    n_seeds: int,
    n_iterations: int,
    n_initial: int,
    adversarial_budget: int,
    high_corruption_value: float,
    low_corruption_value: float,
    corruptor_factory: Optional[callable] = None,
    noise_std: float = 1.0,
) -> List[Dict[str, Any]]:
    """Run a single model across multiple seeds and collect results.

    Args:
        model_name: Name of the model for logging
        model_factory: Factory function to create the model
        model_kwargs: Keyword arguments for model creation
        scheduler: Beta scheduler instance
        clean_evaluator: Base evaluator (without noise/corruption)
        optimal_point: Known optimal point for adversarial corruptor
        search_space: Search space definition
        n_seeds: Number of random seeds to run
        n_iterations: Number of BO iterations per seed
        n_initial: Number of initial points per seed
        adversarial_budget: Budget for adversarial corruption
        high_corruption_value: High corruption value
        low_corruption_value: Low corruption value
        corruptor_factory: Optional factory function to create corruptor
        noise_std: Standard deviation of observation noise

    Returns:
        List of experiment results, one per seed
    """
    all_results = []

    for seed in range(n_seeds):
        print(f"\nRunning {model_name} with seed {seed + 1}/{n_seeds}...")

        # Set global seeds for this iteration to ensure reproducibility
        # This affects BoTorch's optimize_acqf and other unseeded operations
        set_global_seed(seed)

        # Create fresh evaluator for each seed
        noisy_evaluator = NoisyEvaluator(
            clean_evaluator, noise_std=noise_std, seed=seed
        )

        # Use custom corruptor factory if provided, otherwise use default adversarial
        if corruptor_factory:
            corruptor = corruptor_factory(
                optimal_point,
                adversarial_budget,
                high_corruption_value,
                low_corruption_value,
            )
        else:
            corruptor = AdversarialCorruptor(
                optimal_point=optimal_point,
                budget=adversarial_budget,
                near_threshold=0.1,
                far_threshold=0.4,
                high_value=high_corruption_value,
                low_value=low_corruption_value,
            )

        corrupted_evaluator = CorruptedEvaluator(
            base_evaluator=noisy_evaluator, corruptor=corruptor, n_initial=n_initial
        )

        # Create runner for this seed
        runner = ExperimentRunner(search_space, corrupted_evaluator)

        # Run experiment
        results = runner.run(
            n_iterations=n_iterations,
            n_initial=n_initial,
            model_factory=model_factory,
            acquisition_factory=UCBAcquisition.create,
            model_kwargs=model_kwargs,
            beta_scheduler=scheduler,
            seed=seed,
            verbose=(seed == 0),  # Only verbose for first seed
        )

        all_results.append(results)

    return all_results


def aggregate_results_across_seeds(
    all_results: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Aggregate results from multiple seeds into mean/std statistics.

    Args:
        all_results: List of results dictionaries, one per seed

    Returns:
        Dictionary with aggregated results structure for plotting and analysis
    """
    if not all_results:
        return None

    # For the comparison analysis, we need to pass the nested structure
    # but for individual analysis, we use the first seed
    aggregated = {
        "all_results": [
            results["all_results"] for results in all_results
        ],  # Nested: list of lists
        "all_results_flat": all_results[0][
            "all_results"
        ],  # First seed for single analysis
        "final_model": all_results[0][
            "final_model"
        ],  # Use first seed's final model for plotting
        "mean_model": all_results[0]["final_model"],  # Alias for consistency
    }

    return aggregated


def save_individual_seed_results(
    model_name: str,
    seed: int,
    results: Dict[str, Any],
    optimal_value: float,
    artifacts_dir: str,
) -> str:
    """Save individual seed results as separate JSON files.

    Args:
        model_name: Name of the model
        seed: Seed number
        results: Results dictionary for this seed
        optimal_value: Known optimal value
        artifacts_dir: Directory to save artifacts

    Returns:
        Path to saved seed file
    """
    # Create seed-specific directory
    seed_dir = os.path.join(artifacts_dir, "individual_seeds")
    os.makedirs(seed_dir, exist_ok=True)

    # Extract key metrics
    eval_results = results["all_results"]
    simple_regret = compute_regret(eval_results, optimal_value, regret_type="simple")
    cumulative_regret = compute_regret(
        eval_results, optimal_value, regret_type="cumulative"
    )
    instantaneous_regret = compute_regret(
        eval_results, optimal_value, regret_type="instantaneous"
    )

    # Get observed values and parameters
    observed_values = [r.y_observed for r in eval_results]
    true_values = [r.y_true for r in eval_results]
    # Handle x as dictionary with parameter names
    X_values = [list(r.x.values()) for r in eval_results]

    seed_data = {
        "model_name": model_name,
        "seed": seed,
        "n_iterations": len(eval_results),
        "optimal_value": optimal_value,
        "X_values": X_values,
        "observed_values": observed_values,
        "true_values": true_values,
        "simple_regret": simple_regret.tolist(),
        "cumulative_regret": cumulative_regret.tolist(),
        "instantaneous_regret": instantaneous_regret.tolist(),
        "final_simple_regret": float(simple_regret[-1]),
        "final_cumulative_regret": float(cumulative_regret[-1]),
        "best_observed_value": float(max(observed_values)),
        "best_true_value": float(max(true_values)),
        # Add corruption information if available
        "corruption_info": {
            "n_corrupted": sum(
                1
                for r in eval_results
                if hasattr(r, "corruption") and r.corruption != 0
            ),
            "corruption_values": [getattr(r, "corruption", 0.0) for r in eval_results],
            "total_corruption": sum(
                abs(getattr(r, "corruption", 0.0)) for r in eval_results
            ),
        },
    }

    # Save individual seed file
    seed_filename = (
        f"{model_name.lower().replace(' ', '_').replace('-', '_')}_seed_{seed}.json"
    )
    seed_path = os.path.join(seed_dir, seed_filename)

    with open(seed_path, "w") as f:
        json.dump(seed_data, f, indent=2)

    return seed_path


def save_multiseed_summary(
    results_dict: Dict[str, Dict[str, Any]], optimal_value: float, artifacts_dir: str
) -> str:
    """Save aggregated multi-seed statistics.

    Args:
        results_dict: Dictionary mapping model names to aggregated results
        optimal_value: Known optimal value
        artifacts_dir: Directory to save artifacts

    Returns:
        Path to summary file
    """
    summary_stats = {}

    for model_name, results in results_dict.items():
        all_seeds_results = results["all_results"]

        # Compute statistics across seeds
        final_simple_regrets = []
        final_cumulative_regrets = []
        final_best_values = []

        for seed_results in all_seeds_results:
            simple_regret = compute_regret(
                seed_results, optimal_value, regret_type="simple"
            )
            cumulative_regret = compute_regret(
                seed_results, optimal_value, regret_type="cumulative"
            )

            final_simple_regrets.append(simple_regret[-1])  # Final simple regret value
            final_cumulative_regrets.append(
                cumulative_regret[-1]
            )  # Final cumulative regret value

            # Get final best value
            best_values = [r.y_observed for r in seed_results]
            final_best_values.append(max(best_values))

        # Convert to numpy for statistics
        final_simple_regrets = np.array(final_simple_regrets)
        final_cumulative_regrets = np.array(final_cumulative_regrets)
        final_best_values = np.array(final_best_values)

        summary_stats[model_name] = {
            "n_seeds": len(all_seeds_results),
            "n_iterations": len(all_seeds_results[0]),
            "final_simple_regret_mean": float(np.mean(final_simple_regrets)),
            "final_simple_regret_std": float(np.std(final_simple_regrets)),
            "final_simple_regret_min": float(np.min(final_simple_regrets)),
            "final_simple_regret_max": float(np.max(final_simple_regrets)),
            "final_cumulative_regret_mean": float(np.mean(final_cumulative_regrets)),
            "final_cumulative_regret_std": float(np.std(final_cumulative_regrets)),
            "final_cumulative_regret_min": float(np.min(final_cumulative_regrets)),
            "final_cumulative_regret_max": float(np.max(final_cumulative_regrets)),
            "best_value_mean": float(np.mean(final_best_values)),
            "best_value_std": float(np.std(final_best_values)),
            "best_value_min": float(np.min(final_best_values)),
            "best_value_max": float(np.max(final_best_values)),
        }

    # Save aggregated summary
    summary_path = os.path.join(artifacts_dir, "multiseed_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary_stats, f, indent=2)

    print(f"Multi-seed summary statistics saved to: {summary_path}")
    return summary_path


def plot_regret_comparison_multiseed(
    results_dict: Dict[str, Dict[str, Any]],
    optimal_value: float,
    n_seeds: int,
    save_path: Optional[str] = None,
    config: Optional[PlotConfig] = None,
    colors: Optional[Dict[str, str]] = None,
):
    """Create regret comparison plots with shaded areas for multi-seed data.

    Args:
        results_dict: Dictionary mapping model names to their aggregated results
            (where results['all_results'] contains list of results from all seeds)
        optimal_value: Known optimal value of the objective
        n_seeds: Number of seeds used in experiment
        save_path: Optional base path to save plots
        config: Plot configuration
        colors: Optional dictionary mapping model names to colors

    Returns:
        Tuple of (regret_comparison_fig, simple_regret_fig)
    """
    if config is None:
        config = PlotConfig()

    # Default colors if not provided
    if colors is None:
        default_colors = ["blue", "orange", "green", "red", "purple", "brown"]
        colors = {
            name: default_colors[i % len(default_colors)]
            for i, name in enumerate(results_dict.keys())
        }

    # Create figure with instantaneous and cumulative regret
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot instantaneous regret with shaded areas
    for name, results in results_dict.items():
        # results['all_results'] is now a list of results lists (one per seed)
        all_seeds_results = results["all_results"]

        # Compute regret for each seed
        regret_per_seed = []
        for seed_results in all_seeds_results:
            regret = compute_regret(
                seed_results, optimal_value, regret_type="instantaneous"
            )
            regret_per_seed.append(regret)

        # Convert to numpy array and compute statistics
        regret_array = np.array(regret_per_seed)
        mean_regret = np.mean(regret_array, axis=0)
        std_regret = np.std(regret_array, axis=0)

        iterations = np.arange(1, len(mean_regret) + 1)

        # Plot mean with shaded std
        ax1.plot(
            iterations,
            mean_regret,
            label=name,
            color=colors.get(name),
            linewidth=config.linewidth,
        )
        ax1.fill_between(
            iterations,
            mean_regret - std_regret,
            mean_regret + std_regret,
            alpha=config.alpha_fill,
            color=colors.get(name),
        )

    ax1.set_xlabel("Iteration", fontsize=config.fontsize)
    ax1.set_ylabel("Instantaneous Regret", fontsize=config.fontsize)
    ax1.set_title(
        "Instantaneous Regret Comparison (Multi-seed)",
        fontsize=config.fontsize + 2,
        fontweight="bold",
    )
    ax1.legend(fontsize=config.fontsize - 1)
    ax1.grid(config.grid, alpha=0.3)

    # Plot cumulative regret with shaded areas
    for name, results in results_dict.items():
        all_seeds_results = results["all_results"]

        # Compute cumulative regret for each seed
        regret_per_seed = []
        for seed_results in all_seeds_results:
            regret = compute_regret(
                seed_results, optimal_value, regret_type="cumulative"
            )
            regret_per_seed.append(regret)

        # Convert to numpy array and compute statistics
        regret_array = np.array(regret_per_seed)
        mean_regret = np.mean(regret_array, axis=0)
        std_regret = np.std(regret_array, axis=0)

        iterations = np.arange(1, len(mean_regret) + 1)

        # Plot mean with shaded std
        ax2.plot(
            iterations,
            mean_regret,
            label=name,
            color=colors.get(name),
            linewidth=config.linewidth,
        )
        ax2.fill_between(
            iterations,
            mean_regret - std_regret,
            mean_regret + std_regret,
            alpha=config.alpha_fill,
            color=colors.get(name),
        )

    ax2.set_xlabel("Iteration", fontsize=config.fontsize)
    ax2.set_ylabel("Cumulative Regret", fontsize=config.fontsize)
    ax2.set_title(
        "Cumulative Regret Comparison (Multi-seed)",
        fontsize=config.fontsize + 2,
        fontweight="bold",
    )
    ax2.legend(fontsize=config.fontsize - 1)
    ax2.grid(config.grid, alpha=0.3)

    model_names = " vs ".join(results_dict.keys())
    plt.suptitle(
        f"Regret Analysis: {model_names} ({n_seeds} seeds)",
        fontsize=config.fontsize + 4,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    # Save if path provided
    if save_path:
        base_dir = os.path.dirname(save_path) if os.path.dirname(save_path) else "."
        base_name = (
            os.path.splitext(os.path.basename(save_path))[0] if save_path else "regret"
        )

        regret_path = os.path.join(base_dir, f"{base_name}_comparison_multiseed.png")
        fig1.savefig(regret_path, dpi=300, bbox_inches="tight")
        print(f"Multi-seed regret comparison plot saved to: {regret_path}")

    # Create simple regret plot with shaded areas
    fig2, ax = plt.subplots(1, 1, figsize=(10, 6))

    for name, results in results_dict.items():
        all_seeds_results = results["all_results"]

        # Compute simple regret for each seed
        regret_per_seed = []
        for seed_results in all_seeds_results:
            regret = compute_regret(seed_results, optimal_value, regret_type="simple")
            regret_per_seed.append(regret)

        # Convert to numpy array and compute statistics
        regret_array = np.array(regret_per_seed)
        mean_regret = np.mean(regret_array, axis=0)
        std_regret = np.std(regret_array, axis=0)

        iterations = np.arange(1, len(mean_regret) + 1)

        # Plot mean with shaded std
        ax.plot(
            iterations,
            mean_regret,
            label=name,
            color=colors.get(name),
            linewidth=config.linewidth,
        )
        ax.fill_between(
            iterations,
            mean_regret - std_regret,
            mean_regret + std_regret,
            alpha=config.alpha_fill,
            color=colors.get(name),
        )

    ax.set_xlabel("Iteration", fontsize=config.fontsize)
    ax.set_ylabel("Simple Regret", fontsize=config.fontsize)
    ax.set_title(
        f"Simple Regret (Best So Far) Comparison ({n_seeds} seeds)",
        fontsize=config.fontsize + 2,
        fontweight="bold",
    )
    ax.legend(fontsize=config.fontsize - 1)
    ax.grid(config.grid, alpha=0.3)
    ax.set_ylim(bottom=0)

    # Save if path provided
    if save_path:
        simple_path = os.path.join(base_dir, f"{base_name}_simple_multiseed.png")
        fig2.savefig(simple_path, dpi=300, bbox_inches="tight")
        print(f"Multi-seed simple regret plot saved to: {simple_path}")

    return fig1, fig2
