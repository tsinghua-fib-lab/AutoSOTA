"""Reduced CIFAR-10 HPO that tunes learning rate and weight decay only."""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, List

import matplotlib.pyplot as plt
import torch

from bo_framework import Dimension, ExperimentRunner, SearchSpace
from bo_framework.base.acquisition import UCBAcquisition
from bo_framework.base.schedulers import (
    ConstantBetaScheduler,
    RCGPScheduler,
    TheoryGuidedScheduler,
)
from bo_framework.corruption.composable import (
    ComposableCorruptor,
    ConstantStrategy,
    CorruptionDecider,
    PeriodicDecider,
    TimeBudgetDecider,
)
from bo_framework.models.factory import (
    create_a2rcgp_model,
    create_diagnostic_gp_model,
    create_gp_model,
    create_rcgp_model,
    create_student_t_model,
)
from bo_framework.wrappers.corrupted import CorruptedEvaluator
from experiments.hpt_cifar.evaluator_lr_wd import (
    HPTCIFARLearningRateWeightDecayEvaluator,
)
from utilities.io import save_experiment_results
from utilities.regret_analysis import (
    compare_experiments_multiseed,
    print_comparison_table_multiseed,
)
from utilities.multiseed_experiments import (
    set_global_seed,
    aggregate_results_across_seeds,
    save_individual_seed_results,
    save_multiseed_summary,
    plot_regret_comparison_multiseed,
)
from utilities.plotting import PlotConfig


# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------
N_ITERATIONS = 130
N_INITIAL = 10
N_SEEDS = 5
SEED = 50  # Base seed
STANDARDIZE = True
FIT_STANDARD_GP = True
MAX_EPOCHS = 4

# Which models to run
RUN_RCGP = True
RUN_STUDENT = True
RUN_DIAGNOSTIC = True
RUN_A2RCGP = True

# Scheduler configuration per model (matches original script flexibility)
GP_SCHEDULER_TYPE = "theory"  # constant | theory
RCGP_SCHEDULER_TYPE = "rcgp-theory"  # constant | theory | rcgp-* variants
STUDENT_SCHEDULER_TYPE = "theory"
DIAGNOSTIC_SCHEDULER_TYPE = "theory"
A2RCGP_SCHEDULER_TYPE = "rcgp-theory"

CONSTANT_BETA = 2.0
THEORY_SCALE = 1.7
THEORY_OFFSET = 2
RCGP_SCALE = 1.0

# Corruption configuration (copied from original API)
CORRUPTION_TYPE = "time_budget"  # periodic | time_budget | budget | none
PERIODIC_INTERVAL = 5
TIME_BUDGET_ALPHA = 1 / 3
CORRUPTION_BUDGET = 5
CRASH_VALUE = -2.0

# ---------------------------------------------------------------------------
# Model configuration dictionaries
# ---------------------------------------------------------------------------
rcgp_kwargs = {
    "param_handling_dict": {
        "plateau_width": {"method": "heuristics", "value": 2.0},
        "c": {"method": "manual", "value": 1.0},
        "sigma": {"method": "fit"},
        "mean": {"method": "fit"},
    },
    "fitting_objective_type": "wloo-cv",
    "optimizer_type": "lbfgs",
    "standardize": STANDARDIZE,
    "fit_hyperparameters": True,
    "verbose": False,
}

student_t_kwargs = {
    "nu": 3.0,
    "standardize": STANDARDIZE,
    "fit_hyperparameters": True,
    "optimizer_type": "lbfgs",
}

diagnostic_kwargs = {
    "n_init": 3,
    "n_schedule": 1,
    "nu": 4.0,
    "alpha": 0.05,
    "fitting_kwargs": {"num_iterations": 200, "verbose": False},
    "model_kwargs": {
        "standardize": STANDARDIZE,
        "fit_hyperparameters": FIT_STANDARD_GP,
    },
}

a2rcgp_kwargs = {
    "inner_param_handling_dict": {
        "plateau_width": {"method": "heuristics", "value": 2.0},
        "c": {"method": "manual", "value": 1.0},
        "sigma": {"method": "fit"},
        "mean": {"method": "fit"},
    },
    "outer_param_handling_dict": {
        "plateau_width": {"method": "heuristics", "value": 1.5},
        "c": {"method": "manual", "value": 0.8},
        "sigma": {"method": "fit"},
        "mean": {"method": "fit"},
    },
    "fitting_objective_type": "wloo-cv",
    "optimizer_type": "lbfgs",
    "standardize": STANDARDIZE,
    "verbose": False,
}


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
def create_scheduler(scheduler_type: str, model_type: str = "gp"):
    if model_type in {"rcgp", "a2rcgp"}:
        if scheduler_type == "constant":
            return ConstantBetaScheduler(beta=CONSTANT_BETA)
        if scheduler_type == "theory":
            return TheoryGuidedScheduler(
                scale=THEORY_SCALE, offset=THEORY_OFFSET, min_beta=1.0
            )
        if scheduler_type == "rcgp-constant":
            return RCGPScheduler(
                scale=RCGP_SCALE,
                base_scheduler=ConstantBetaScheduler(beta=CONSTANT_BETA),
            )
        if scheduler_type == "rcgp-theory":
            return RCGPScheduler(
                scale=RCGP_SCALE,
                base_scheduler=TheoryGuidedScheduler(
                    scale=THEORY_SCALE, offset=THEORY_OFFSET, min_beta=1.0
                ),
            )
        raise ValueError(f"Unknown RCGP scheduler type: {scheduler_type}")

    # GP / Student-t / Diagnostic follow the standard options
    if scheduler_type == "constant":
        return ConstantBetaScheduler(beta=CONSTANT_BETA)
    if scheduler_type == "theory":
        return TheoryGuidedScheduler(
            scale=THEORY_SCALE, offset=THEORY_OFFSET, min_beta=1.0
        )
    raise ValueError(f"Unknown GP scheduler type: {scheduler_type}")


def create_training_corruptor() -> ComposableCorruptor | None:
    if CORRUPTION_TYPE == "none":
        return None

    if CORRUPTION_TYPE == "budget":

        class CountBudgetDecider(CorruptionDecider):
            def __init__(self, budget: int, skip_initial: bool = True) -> None:
                self.budget = budget
                self.corruptions_used = 0
                self.skip_initial = skip_initial

            def should_corrupt(self, iteration, total_iterations, is_initial, history):
                if is_initial and self.skip_initial:
                    return False
                return self.corruptions_used < self.budget

            def reset(self):
                self.corruptions_used = 0

            def update_corruption(self):
                self.corruptions_used += 1

            @property
            def info(self):
                return f"Budget: {self.corruptions_used}/{self.budget}"

        decider = CountBudgetDecider(CORRUPTION_BUDGET, skip_initial=True)
    elif CORRUPTION_TYPE == "periodic":
        decider = PeriodicDecider(
            period=PERIODIC_INTERVAL, skip_initial=True, n_initial=N_INITIAL
        )
    elif CORRUPTION_TYPE == "time_budget":
        decider = TimeBudgetDecider(
            alpha=TIME_BUDGET_ALPHA, skip_initial=True, n_initial=N_INITIAL
        )
    else:
        raise ValueError(f"Unknown corruption type: {CORRUPTION_TYPE}")

    strategy = ConstantStrategy(corruption_value=CRASH_VALUE)
    return ComposableCorruptor(decider=decider, strategy=strategy, skip_initial=True)


def timestamped_dir(base: str = "artifacts") -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(base, f"hpt_cifar_lr_wd_{stamp}")
    os.makedirs(path, exist_ok=True)
    return path


def persist_config(path: str, config: Dict) -> None:
    with open(os.path.join(path, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, default=str)


# ---------------------------------------------------------------------------
# Model factories (reuse existing factory helpers, merging default kwargs)
# ---------------------------------------------------------------------------
def gp_model_factory(X, Y, **kwargs):
    merged = {
        "standardize": STANDARDIZE,
        "fit_hyperparameters": FIT_STANDARD_GP,
        **kwargs,
    }
    return create_gp_model(X, Y, **merged)


def rcgp_model_factory(X, Y, **kwargs):
    merged = {**rcgp_kwargs, **kwargs}
    return create_rcgp_model(X, Y, **merged)


def student_t_model_factory(X, Y, **kwargs):
    merged = {**student_t_kwargs, **kwargs}
    return create_student_t_model(X, Y, **merged)


def diagnostic_model_factory(X, Y, **kwargs):
    merged = {**diagnostic_kwargs, **kwargs}
    return create_diagnostic_gp_model(X, Y, **merged)


def a2rcgp_model_factory(X, Y, **kwargs):
    merged = {**a2rcgp_kwargs, **kwargs}
    return create_a2rcgp_model(X, Y, **merged)


def run_multiseed_experiment(
    model_name: str,
    model_factory,
    scheduler_type: str,
    model_type: str,
    search_space: SearchSpace,
    n_seeds: int,
    n_iterations: int,
    n_initial: int,
    base_seed: int,
    model_kwargs: Dict = None,
) -> List[Dict]:
    """Run experiment across multiple seeds with fresh evaluator/corruptor instances.

    Args:
        model_name: Name of the model (e.g., 'GP', 'RCGP')
        model_factory: Factory function for creating model
        scheduler_type: Type of beta scheduler
        model_type: Model type for scheduler creation ('gp', 'rcgp', 'a2rcgp')
        search_space: SearchSpace object
        n_seeds: Number of seeds to run
        n_iterations: Number of BO iterations per seed
        n_initial: Number of initial random points
        base_seed: Base seed value
        model_kwargs: Additional kwargs for model factory

    Returns:
        List of result dictionaries, one per seed
    """
    if model_kwargs is None:
        model_kwargs = {}

    all_results = []

    for i in range(n_seeds):
        current_seed = base_seed + i
        print(f"\nRunning {model_name} with seed {current_seed} ({i + 1}/{n_seeds})...")

        # Set global random seed for reproducibility
        set_global_seed(current_seed)

        # Create fresh evaluator for this seed
        base_evaluator = HPTCIFARLearningRateWeightDecayEvaluator(
            max_epochs=MAX_EPOCHS, batch_size=128
        )

        # Create fresh corruptor for this seed
        corruptor = create_training_corruptor()

        # Wrap with CorruptedEvaluator if corruption is enabled
        evaluator = (
            CorruptedEvaluator(
                base_evaluator=base_evaluator, corruptor=corruptor, n_initial=n_initial
            )
            if corruptor is not None
            else base_evaluator
        )

        # Create scheduler
        scheduler = create_scheduler(scheduler_type, model_type)

        # Run experiment
        runner = ExperimentRunner(search_space, evaluator)
        results = runner.run(
            n_iterations=n_iterations,
            n_initial=n_initial,
            model_factory=model_factory,
            acquisition_factory=UCBAcquisition.create,
            beta_scheduler=scheduler,
            seed=current_seed,
            model_kwargs=model_kwargs,
            verbose=(i == 0),  # Only verbose for first seed
        )
        all_results.append(results)

    return all_results


def calculate_cumulative_regret(results: List, optimal_value: float) -> List[float]:
    if not results:
        return []

    observed = [r.y_observed for r in results]
    cumulative = []
    total = 0.0
    for value in observed:
        total += optimal_value - value
        cumulative.append(total)
    return cumulative


def calculate_simple_regret(results: List, optimal_value: float) -> List[float]:
    if not results:
        return []

    observed = [r.y_observed for r in results]
    best_so_far = []
    current = observed[0]
    for value in observed:
        current = max(current, value)
        best_so_far.append(current)
    return [optimal_value - v for v in best_so_far]


def plot_cumulative_regret_comparison(
    results_dict: Dict[str, List], optimal_value: float, save_path: str, title: str
) -> None:
    plt.figure(figsize=(10, 6))
    for name, results in results_dict.items():
        regret = calculate_cumulative_regret(results, optimal_value)
        iterations = list(range(1, len(regret) + 1))
        plt.plot(iterations, regret, label=name, marker="o")

    plt.xlabel("Iteration")
    plt.ylabel("Cumulative Regret")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_simple_regret_comparison(
    results_dict: Dict[str, List], optimal_value: float, save_path: str, title: str
) -> None:
    plt.figure(figsize=(10, 6))
    for name, results in results_dict.items():
        regret = calculate_simple_regret(results, optimal_value)
        iterations = list(range(1, len(regret) + 1))
        plt.plot(iterations, regret, label=name, marker="o")

    plt.xlabel("Iteration")
    plt.ylabel("Simple Regret")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_chosen_points(results_dict: Dict[str, List], save_dir: str) -> None:
    chosen_points = {}
    for name, results in results_dict.items():
        entries = []
        for idx, result in enumerate(results, start=1):
            entries.append(
                {
                    "iteration": idx,
                    "parameters": result.x,
                    "observed_value": result.y_observed,
                    "true_value": result.y_true,
                    "is_corrupted": result.y_observed != result.y_true,
                }
            )
        chosen_points[name] = entries

    json_path = os.path.join(save_dir, "chosen_points.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(chosen_points, handle, indent=2, default=str)

    csv_path = os.path.join(save_dir, "chosen_points_summary.csv")
    with open(csv_path, "w", encoding="utf-8") as handle:
        handle.write(
            "model,iteration,learning_rate,weight_decay,observed_value,true_value,is_corrupted\n"
        )
        for name, entries in chosen_points.items():
            for entry in entries:
                params = entry["parameters"]
                handle.write(
                    f"{name},{entry['iteration']},{params['learning_rate']:.8f},{params['weight_decay']:.8f},"
                    f"{entry['observed_value']:.6f},{entry['true_value']:.6f},{entry['is_corrupted']}\n"
                )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """Run HPT CIFAR experiment with multi-seed support."""
    artifacts_dir = timestamped_dir()
    print(f"Created experiment folder: {artifacts_dir}")

    # Define search space
    search_space = SearchSpace(
        (
            Dimension(
                name="learning_rate",
                type="continuous",
                bounds=(1e-5, 1e-1),
                log_scale=True,
                normalize=True,
            ),
            Dimension(
                name="weight_decay",
                type="continuous",
                bounds=(1e-6, 1e-2),
                log_scale=True,
                normalize=True,
            ),
        )
    )

    # Save configuration
    config = {
        "seed": SEED,
        "n_seeds": N_SEEDS,
        "n_iterations": N_ITERATIONS,
        "n_initial": N_INITIAL,
        "max_epochs": MAX_EPOCHS,
        "standardize": STANDARDIZE,
        "search_space": ["learning_rate", "weight_decay"],
        "corruption_type": CORRUPTION_TYPE,
        "scheduler_types": {
            "gp": GP_SCHEDULER_TYPE,
            "rcgp": RCGP_SCHEDULER_TYPE,
            "student": STUDENT_SCHEDULER_TYPE,
            "diagnostic": DIAGNOSTIC_SCHEDULER_TYPE,
            "a2rcgp": A2RCGP_SCHEDULER_TYPE,
        },
    }
    persist_config(artifacts_dir, config)

    print("\n" + "=" * 80)
    print("HPT CIFAR EXPERIMENT CONFIGURATION")
    print("=" * 80)
    print(f"Iterations: {N_ITERATIONS}, Initial points: {N_INITIAL}")
    print(f"Seeds: {N_SEEDS} (starting from {SEED})")
    print(f"Max epochs: {MAX_EPOCHS}")
    print(f"Corruption type: {CORRUPTION_TYPE}")
    print("=" * 80 + "\n")

    # Dictionary to store all results (list of lists)
    all_results_multiseed = {}

    # GP baseline (always run)
    print("=" * 80)
    print("Running GP Baseline")
    print("=" * 80)
    all_results_multiseed["GP"] = run_multiseed_experiment(
        "GP",
        gp_model_factory,
        GP_SCHEDULER_TYPE,
        "gp",
        search_space,
        N_SEEDS,
        N_ITERATIONS,
        N_INITIAL,
        SEED,
    )

    # RCGP (conditional)
    if RUN_RCGP:
        print("\n" + "=" * 80)
        print("Running RCGP")
        print("=" * 80)
        all_results_multiseed["RCGP"] = run_multiseed_experiment(
            "RCGP",
            rcgp_model_factory,
            RCGP_SCHEDULER_TYPE,
            "rcgp",
            search_space,
            N_SEEDS,
            N_ITERATIONS,
            N_INITIAL,
            SEED,
        )

    # Student-t (conditional)
    if RUN_STUDENT:
        print("\n" + "=" * 80)
        print("Running Student-t Process")
        print("=" * 80)
        all_results_multiseed["Student-t"] = run_multiseed_experiment(
            "Student-t",
            student_t_model_factory,
            STUDENT_SCHEDULER_TYPE,
            "gp",
            search_space,
            N_SEEDS,
            N_ITERATIONS,
            N_INITIAL,
            SEED,
        )

    # Diagnostic GP (conditional)
    if RUN_DIAGNOSTIC:
        print("\n" + "=" * 80)
        print("Running Diagnostic GP")
        print("=" * 80)
        all_results_multiseed["Diagnostic GP"] = run_multiseed_experiment(
            "Diagnostic GP",
            diagnostic_model_factory,
            DIAGNOSTIC_SCHEDULER_TYPE,
            "gp",
            search_space,
            N_SEEDS,
            N_ITERATIONS,
            N_INITIAL,
            SEED,
        )

    # A2RCGP (conditional)
    if RUN_A2RCGP:
        print("\n" + "=" * 80)
        print("Running A2RCGP")
        print("=" * 80)
        all_results_multiseed["A2RCGP"] = run_multiseed_experiment(
            "A2RCGP",
            a2rcgp_model_factory,
            A2RCGP_SCHEDULER_TYPE,
            "a2rcgp",
            search_space,
            N_SEEDS,
            N_ITERATIONS,
            N_INITIAL,
            SEED,
        )

    # Aggregate results
    aggregated_results = {}
    for model_name, results_list in all_results_multiseed.items():
        aggregated_results[model_name] = aggregate_results_across_seeds(results_list)

    # Find optimal value across all experiments (max accuracy)
    optimal_value = -float("inf")
    for model_results in all_results_multiseed.values():
        for seed_results_dict in model_results:
            seed_results_list = seed_results_dict["all_results"]
            seed_max = max(r.y_true for r in seed_results_list)
            if seed_max > optimal_value:
                optimal_value = seed_max

    print(f"\nOptimal value (max accuracy across all seeds): {optimal_value:.4f}")

    # Save results and create plots
    print("\n" + "=" * 80)
    print("SAVING RESULTS AND GENERATING PLOTS")
    print("=" * 80)

    # Save multi-seed results properly
    multiseed_results_dict = {}
    for model_name, results_list in all_results_multiseed.items():
        for i, results in enumerate(results_list):
            multiseed_results_dict[f"{model_name}_seed_{i}"] = results

    # Save all individual seed results (pickle + combined JSON)
    save_experiment_results(
        results=multiseed_results_dict,
        experiment_name="hpt_cifar_lr_wd",
        artifacts_dir=artifacts_dir,
        save_pickle=True,
        save_json=True,
        optimal_value=optimal_value,
        verbose=True,
    )

    # Save individual JSON files for each seed
    print("Saving individual seed JSON files...")
    for model_name, results_list in all_results_multiseed.items():
        for i, results in enumerate(results_list):
            seed = SEED + i
            seed_path = save_individual_seed_results(
                model_name=model_name,
                seed=seed,
                results=results,  # Pass the full results dict, not results['all_results']
                optimal_value=optimal_value,
                artifacts_dir=artifacts_dir,
            )
            if i == 0:
                print(f"  {model_name} individual seeds saved (e.g., {seed_path})")

    # Save aggregated multi-seed statistics
    save_multiseed_summary(
        results_dict=aggregated_results,
        optimal_value=optimal_value,
        artifacts_dir=artifacts_dir,
    )

    # Create regret comparison plots
    print("\n" + "=" * 80)
    print("CREATING REGRET COMPARISON PLOTS")
    print("=" * 80)

    colors = {
        "RCGP": "blue",
        "GP": "orange",
        "Student-t": "green",
        "A2RCGP": "red",
        "Diagnostic GP": "purple",
    }
    config_plot = PlotConfig(figsize=(15, 10))

    regret_fig, simple_regret_fig = plot_regret_comparison_multiseed(
        results_dict=aggregated_results,
        optimal_value=optimal_value,
        n_seeds=N_SEEDS,
        save_path=os.path.join(artifacts_dir, "regret"),
        config=config_plot,
        colors=colors,
    )

    plt.close(regret_fig)
    plt.close(simple_regret_fig)

    # Save chosen points for first seed (for backward compatibility)
    first_seed_results = {
        k: v["all_results_flat"] for k, v in aggregated_results.items()
    }
    save_chosen_points(first_seed_results, artifacts_dir)

    # Calculate and print regret metrics
    print("\n" + "=" * 80)
    print("REGRET ANALYSIS")
    print("=" * 80)

    comparison_results_dict = {
        name: results["all_results"] for name, results in aggregated_results.items()
    }

    multiseed_metrics_dict = compare_experiments_multiseed(
        results_dict=comparison_results_dict,
        optimal_value=optimal_value,
    )

    print_comparison_table_multiseed(
        multiseed_metrics_dict, show_regret=True, show_corruption=True, show_std=True
    )

    # Print final summary
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    print("Experiment completed successfully!")

    # Find best model from first seed
    best_model_name = max(
        aggregated_results.keys(),
        key=lambda name: max(
            r.y_true for r in aggregated_results[name]["all_results_flat"]
        ),
    )
    best_val = max(
        r.y_true for r in aggregated_results[best_model_name]["all_results_flat"]
    )

    print(f"Best model (seed {SEED}): {best_model_name} with accuracy {best_val:.4f}")
    print(f"Total evaluations per model per seed: {N_ITERATIONS}")
    print(f"Total seeds: {N_SEEDS}")
    print(f"Corruption type: {CORRUPTION_TYPE}")
    print(f"\nAll artifacts saved to: {artifacts_dir}/")
    print("Files saved:")
    print("  - config.json (configuration)")
    print("  - regret_comparison_multiseed.png (cumulative regret plot)")
    print("  - simple_regret_multiseed.png (simple regret plot)")
    print("  - chosen_points.json (first seed chosen points)")
    print("  - hpt_cifar_lr_wd/results.pkl (pickle format)")
    print("  - hpt_cifar_lr_wd/summary.json (JSON format)")
    print("  - multiseed_summary.json (aggregated stats)")
    print("  - individual_seeds/ (folder with per-seed JSONs)")


if __name__ == "__main__":
    main()
