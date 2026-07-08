"""Test the new clean API with Lunar Lander RL policy optimization."""

import os
import torch
import json
import matplotlib.pyplot as plt
from datetime import datetime
from bo_framework import SearchSpace, ExperimentRunner
from bo_framework.base.acquisition import UCBAcquisition
from bo_framework.base.schedulers import (
    ConstantBetaScheduler,
    TheoryGuidedScheduler,
    RCGPScheduler,
)
from bo_framework.models.factory import (
    create_gp_model,
    create_rcgp_model,
    create_student_t_model,
    create_a2rcgp_model,
    create_diagnostic_gp_model,
)
from bo_framework.corruption.composable import (
    ComposableCorruptor,
    TimeBudgetDecider,
    ConstantStrategy,
)
from bo_framework.wrappers.corrupted import CorruptedEvaluator
from experiments.lunar_lander.evaluator import LunarLanderEvaluator
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


# Experiment parameters
N_ITERATIONS = 200
N_INITIAL = 10
N_SEEDS = 5
SEED = 42  # Base seed
STANDARDIZE = True
FIT_HYPERPARAMETERS = True
USE_BOTORCH_MODEL = True
N_EPISODES = 4  # Number of episodes to average for evaluation

# Corruption configuration
# Choose: 'time_budget', 'none'
CORRUPTION_TYPE = "time_budget"

# Time budget parameters
TIME_BUDGET_ALPHA = 1 / 3  # T^alpha budget (0.5 = sqrt(T))

# Constant corruption value
CORRUPTION_VALUE = 1000.0

# Beta scheduler configuration
# For RCGP: Choose from 'constant', 'theory', 'rcgp-constant', 'rcgp-theory'
RCGP_SCHEDULER_TYPE = "theory"

# For GP/Student-t/Diagnostic: Choose from 'constant', 'theory'
GP_SCHEDULER_TYPE = "theory"

# For A2RCGP: Choose from 'constant', 'theory', 'rcgp-constant', 'rcgp-theory'
A2RCGP_SCHEDULER_TYPE = "theory"

# Beta scheduling parameters
CONSTANT_BETA = 2.0
RCGP_SCALE = 1.0  # Scale factor for RCGP adaptive term

# Theory scheduler parameters
THEORY_SCALE = 1.7  # Scale for theory-guided beta schedule
THEORY_OFFSET = 2  # Offset to handle early iterations

# Student-specific scheduler parameters
STUDENT_THEORY_SCALE = 1.0  # Lower scale for student-t models
STUDENT_THEORY_OFFSET = 1  # Lower offset for student-t models
STUDENT_MIN_BETA = 0.1  # Lower minimum beta for student-t models

# RCGP configuration
rcgp_kwargs = {
    "param_handling_dict": {
        "plateau_width": {
            "method": "heuristics"
        },  # Use heuristics for high-dim problem
        "c": {"method": "manual", "value": 1.0},
        "sigma": {"method": "fit"},  # Fit the noise parameter
        "mean": {"method": "fit"},  # Fit the mean parameter
    },
    "fitting_objective_type": "wloo-cv",  # Use weighted leave-one-out cross-validation
    "optimizer_type": "lbfgs",
    "standardize": STANDARDIZE,
    "verbose": False,
}

# A2RCGP configuration with inner and outer model parameters
a2rcgp_kwargs = {
    "inner_param_handling_dict": {
        "plateau_width": {"method": "heuristics"},  # Use heuristics for high-dim
        "c": {"method": "manual", "value": 1.0},
        "sigma": {"method": "fit"},
        "mean": {"method": "fit"},
    },
    "outer_param_handling_dict": {
        "plateau_width": {"method": "heuristics"},  # Use heuristics for high-dim
        "c": {"method": "manual", "value": 0.8},
        "sigma": {"method": "fit"},
        "mean": {"method": "fit"},
    },
    "fitting_objective_type": "wloo-cv",
    "optimizer_type": "lbfgs",
    "standardize": STANDARDIZE,
    "verbose": False,
}

# Student-t Process configuration
student_t_kwargs = {
    "nu": 3.0,  # Degrees of freedom (lower = heavier tails)
    "standardize": STANDARDIZE,
    "fit_hyperparameters": FIT_HYPERPARAMETERS,
    "optimizer_type": "lbfgs",
}

# Diagnostic GP (OD-BO) configuration
diagnostic_kwargs = {
    "n_init": 5,  # Start diagnosis after 5 points
    "n_schedule": 1,  # Run diagnosis every iteration
    "nu": 4.0,  # Student-t degrees of freedom
    "alpha": 0.05,  # Outlier threshold
    "fitting_kwargs": {"num_iterations": 200, "verbose": False},
    "model_kwargs": {
        "standardize": STANDARDIZE,
        "fit_hyperparameters": FIT_HYPERPARAMETERS,
        "use_botorch_model": USE_BOTORCH_MODEL,
    },
}


def create_timestamped_folder(base_dir="artifacts"):
    """Create a timestamped folder for experiment results.

    Args:
        base_dir: Base directory to create the timestamped folder in

    Returns:
        Path to the created timestamped folder
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"lunar_lander_experiment_{timestamp}"
    folder_path = os.path.join(base_dir, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path


def save_experiment_config(config_dict, folder_path):
    """Save experiment configuration to JSON file.

    Args:
        config_dict: Dictionary containing all experiment parameters
        folder_path: Path to the experiment folder
    """
    config_path = os.path.join(folder_path, "experiment_config.json")
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2, default=str)
    print(f"Experiment configuration saved to: {config_path}")


def create_scheduler(scheduler_type, model_type="gp"):
    """Create beta scheduler based on configuration.

    Args:
        scheduler_type: Type of scheduler ('constant', 'theory', 'rcgp-constant', 'rcgp-theory')
        model_type: Model type ('rcgp', 'gp', 'student', 'a2rcgp', 'diagnostic')

    Returns:
        BetaScheduler instance
    """
    if model_type == "rcgp" or model_type == "a2rcgp":
        if scheduler_type == "constant":
            return ConstantBetaScheduler(beta=CONSTANT_BETA)
        elif scheduler_type == "theory":
            return TheoryGuidedScheduler(
                scale=THEORY_SCALE, offset=THEORY_OFFSET, min_beta=1.0
            )
        elif scheduler_type == "rcgp-constant":
            return RCGPScheduler(
                scale=RCGP_SCALE,
                base_scheduler=ConstantBetaScheduler(beta=CONSTANT_BETA),
            )
        elif scheduler_type == "rcgp-theory":
            return RCGPScheduler(
                scale=RCGP_SCALE,
                base_scheduler=TheoryGuidedScheduler(
                    scale=THEORY_SCALE, offset=THEORY_OFFSET, min_beta=1.0
                ),
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    else:  # GP, Student-t, or Diagnostic
        if scheduler_type == "constant":
            return ConstantBetaScheduler(beta=CONSTANT_BETA)
        elif scheduler_type == "theory":
            if model_type == "student":
                # Student-t models use different parameters
                return TheoryGuidedScheduler(
                    scale=STUDENT_THEORY_SCALE,
                    offset=STUDENT_THEORY_OFFSET,
                    min_beta=STUDENT_MIN_BETA,
                )
            else:
                # Standard GP and Diagnostic use regular parameters
                return TheoryGuidedScheduler(
                    scale=THEORY_SCALE, offset=THEORY_OFFSET, min_beta=1.0
                )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def create_corruptor_factory(corruption_type: str = "time_budget"):
    """Create a factory function for the specified corruptor configuration.

    Args:
        corruption_type: Type of corruption ('time_budget', 'none')

    Returns:
        ComposableCorruptor instance or None
    """
    if corruption_type == "none":
        return None

    if corruption_type == "time_budget":
        decider = TimeBudgetDecider(
            alpha=TIME_BUDGET_ALPHA, skip_initial=True, n_initial=N_INITIAL
        )
    else:
        raise ValueError(f"Unknown corruption type: {corruption_type}")

    # Create constant strategy that returns the corruption value
    strategy = ConstantStrategy(corruption_value=CORRUPTION_VALUE)

    # Return composable corruptor
    return ComposableCorruptor(decider=decider, strategy=strategy, skip_initial=True)


def calculate_cumulative_regret(results, optimal_value):
    """Calculate cumulative regret for a set of results.

    Cumulative regret is the sum of instantaneous regrets:
    Σ(optimal_value - observed_value_i) for i = 1 to t

    Args:
        results: List of EvaluationResult objects
        optimal_value: The optimal value to compare against

    Returns:
        List of cumulative regret values
    """
    if not results:
        return []

    # Get all observed values
    observed_values = [r.y_observed for r in results]

    # Calculate instantaneous regrets
    instantaneous_regrets = [optimal_value - value for value in observed_values]

    # Calculate cumulative sum of regrets
    cumulative_regret = []
    cumulative_sum = 0
    for regret in instantaneous_regrets:
        cumulative_sum += regret
        cumulative_regret.append(cumulative_sum)

    return cumulative_regret


def calculate_simple_regret(results, optimal_value):
    """Calculate simple regret for a set of results.

    Simple regret is the regret of the best point found so far:
    optimal_value - max(observed_values[1:t])

    Args:
        results: List of EvaluationResult objects
        optimal_value: The optimal value to compare against

    Returns:
        List of simple regret values
    """
    if not results:
        return []

    # Get all observed values
    observed_values = [r.y_observed for r in results]

    # Calculate cumulative maximum
    cumulative_max = []
    current_max = observed_values[0]
    for value in observed_values:
        current_max = max(current_max, value)
        cumulative_max.append(current_max)

    # Calculate simple regret (optimal - cumulative_max)
    simple_regret = [optimal_value - max_val for max_val in cumulative_max]
    return simple_regret


def plot_regret_comparison(
    results_dict, optimal_value, save_path_base, title="Regret Comparison"
):
    """Plot both cumulative and simple regret comparisons for multiple models.

    Args:
        results_dict: Dictionary mapping model names to their results
        optimal_value: The optimal value to compare against
        save_path_base: Base path to save the plots (will add _cumulative.png and _simple.png)
        title: Plot title base
    """
    colors = ["blue", "orange", "green", "red", "purple"]
    markers = ["o", "s", "^", "D", "v"]

    # Plot 1: Cumulative Regret
    plt.figure(figsize=(12, 8))

    for i, (model_name, results) in enumerate(results_dict.items()):
        cumulative_regret = calculate_cumulative_regret(results, optimal_value)
        iterations = list(range(1, len(cumulative_regret) + 1))

        plt.plot(
            iterations,
            cumulative_regret,
            label=model_name,
            color=colors[i % len(colors)],
            marker=markers[i % len(markers)],
            linewidth=2,
            markersize=4,
            markevery=max(1, len(iterations) // 10),
        )

    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Cumulative Regret", fontsize=12)
    plt.title(f"{title}: Cumulative Regret", fontsize=14)
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    cumulative_path = save_path_base.replace(".png", "_cumulative.png")
    plt.savefig(cumulative_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Cumulative regret plot saved to: {cumulative_path}")

    # Plot 2: Simple Regret
    plt.figure(figsize=(12, 8))

    for i, (model_name, results) in enumerate(results_dict.items()):
        simple_regret = calculate_simple_regret(results, optimal_value)
        iterations = list(range(1, len(simple_regret) + 1))

        plt.plot(
            iterations,
            simple_regret,
            label=model_name,
            color=colors[i % len(colors)],
            marker=markers[i % len(markers)],
            linewidth=2,
            markersize=4,
            markevery=max(1, len(iterations) // 10),
        )

    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Simple Regret", fontsize=12)
    plt.title(f"{title}: Simple Regret", fontsize=14)
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    simple_path = save_path_base.replace(".png", "_simple.png")
    plt.savefig(simple_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Simple regret plot saved to: {simple_path}")


# Keep the old function for backward compatibility, but fix it to use cumulative regret
def plot_cumulative_regret_comparison(
    results_dict, optimal_value, save_path, title="Cumulative Regret Comparison"
):
    """Plot cumulative regret comparison for multiple models.

    Args:
        results_dict: Dictionary mapping model names to their results
        optimal_value: The optimal value to compare against
        save_path: Path to save the plot
        title: Plot title
    """
    plt.figure(figsize=(12, 8))

    colors = ["blue", "orange", "green", "red", "purple"]
    markers = ["o", "s", "^", "D", "v"]

    for i, (model_name, results) in enumerate(results_dict.items()):
        cumulative_regret = calculate_cumulative_regret(results, optimal_value)
        iterations = list(range(1, len(cumulative_regret) + 1))

        plt.plot(
            iterations,
            cumulative_regret,
            label=model_name,
            color=colors[i % len(colors)],
            marker=markers[i % len(markers)],
            linewidth=2,
            markersize=4,
            markevery=max(1, len(iterations) // 10),
        )  # Show markers at intervals

    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Cumulative Regret", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Cumulative regret plot saved to: {save_path}")


def plot_optimization_trajectories(
    results_dict, save_path, title="Optimization Trajectories"
):
    """Plot optimization trajectories for all models.

    Args:
        results_dict: Dictionary mapping model names to their results
        save_path: Path to save the plot
        title: Plot title
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (model_name, results) in enumerate(results_dict.items()):
        if idx >= len(axes):
            break

        ax = axes[idx]
        iterations = list(range(1, len(results) + 1))
        observed_values = [r.y_observed for r in results]
        true_values = [r.y_true for r in results]

        ax.plot(
            iterations,
            observed_values,
            "o-",
            label="Observed",
            linewidth=2,
            markersize=6,
        )
        ax.plot(
            iterations,
            true_values,
            "s-",
            label="True",
            linewidth=2,
            markersize=6,
            alpha=0.7,
        )
        ax.axvline(
            x=N_INITIAL, color="gray", linestyle="--", alpha=0.5, label="End of initial"
        )

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Reward")
        ax.set_title(f"{model_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Hide any unused subplots
    for idx in range(len(results_dict), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Optimization trajectories saved to: {save_path}")


def run_multiseed_experiment(
    model_name,
    model_factory,
    scheduler_type,
    model_type,
    search_space,
    n_seeds,
    n_iterations,
    n_initial,
    corruption_type,
    corruption_value,
    base_seed=0,
    model_kwargs=None,
):
    """Run experiment across multiple seeds with fresh environments.

    Args:
        model_name: Name of the model
        model_factory: Factory function for the model
        scheduler_type: Type of beta scheduler
        model_type: Type of model string for scheduler creation
        search_space: Search space
        n_seeds: Number of seeds
        n_iterations: Number of iterations
        n_initial: Number of initial points
        corruption_type: Type of corruption
        corruption_value: Value for corruption
        base_seed: Starting seed
        model_kwargs: Extra kwargs for model

    Returns:
        List of results for each seed
    """
    all_results = []
    model_kwargs = model_kwargs or {}

    for i in range(n_seeds):
        current_seed = base_seed + i
        print(f"\nRunning {model_name} with seed {current_seed} ({i + 1}/{n_seeds})...")

        # Set global random seeds
        set_global_seed(current_seed)

        # Create fresh base evaluator for this seed
        base_evaluator = LunarLanderEvaluator(
            env_name="LunarLander-v3",
            max_steps=1000,
            render=False,
            seed=current_seed,
            n_episodes=N_EPISODES,
        )

        # Create fresh corruptor
        corruptor = create_corruptor_factory(corruption_type)

        # Create fresh evaluator (wrapped if corruption is enabled)
        if corruptor is not None:
            evaluator = CorruptedEvaluator(
                base_evaluator=base_evaluator, corruptor=corruptor, n_initial=n_initial
            )
        else:
            evaluator = base_evaluator

        # Create fresh scheduler
        scheduler = create_scheduler(scheduler_type, model_type)

        # Create runner
        runner = ExperimentRunner(search_space, evaluator)

        # Run experiment
        results = runner.run(
            n_iterations=n_iterations,
            n_initial=n_initial,
            model_factory=model_factory,
            acquisition_factory=UCBAcquisition.create,
            beta_scheduler=scheduler,
            seed=current_seed,
            model_kwargs=model_kwargs,
            verbose=False,  # Only verbose for first seed
        )
        all_results.append(results)

        # Cleanup evaluator for this seed
        if hasattr(evaluator, "close"):
            evaluator.close()
        elif hasattr(evaluator, "base_evaluator") and hasattr(
            evaluator.base_evaluator, "close"
        ):
            evaluator.base_evaluator.close()

    return all_results


def main():
    """Demonstrate the new clean API with Lunar Lander policy optimization."""

    # Create timestamped folder for results
    artifacts_dir = create_timestamped_folder()
    print(f"Created experiment folder: {artifacts_dir}")

    # Create configuration dictionary
    config_dict = {
        "experiment_info": {
            "name": "Lunar Lander Experiment",
            "timestamp": datetime.now().isoformat(),
            "script": "lunar_lander.py",
        },
        "experiment_parameters": {
            "N_ITERATIONS": N_ITERATIONS,
            "N_INITIAL": N_INITIAL,
            "N_SEEDS": N_SEEDS,
            "BASE_SEED": SEED,
            "STANDARDIZE": STANDARDIZE,
            "FIT_HYPERPARAMETERS": FIT_HYPERPARAMETERS,
            "USE_BOTORCH_MODEL": USE_BOTORCH_MODEL,
            "N_EPISODES": N_EPISODES,
        },
        "corruption_config": {
            "CORRUPTION_TYPE": CORRUPTION_TYPE,
            "TIME_BUDGET_ALPHA": TIME_BUDGET_ALPHA,
            "CORRUPTION_VALUE": CORRUPTION_VALUE,
        },
        "scheduler_config": {
            "RCGP_SCHEDULER_TYPE": RCGP_SCHEDULER_TYPE,
            "GP_SCHEDULER_TYPE": GP_SCHEDULER_TYPE,
            "A2RCGP_SCHEDULER_TYPE": A2RCGP_SCHEDULER_TYPE,
            "CONSTANT_BETA": CONSTANT_BETA,
            "RCGP_SCALE": RCGP_SCALE,
            "THEORY_SCALE": THEORY_SCALE,
            "THEORY_OFFSET": THEORY_OFFSET,
            "STUDENT_THEORY_SCALE": STUDENT_THEORY_SCALE,
            "STUDENT_THEORY_OFFSET": STUDENT_THEORY_OFFSET,
            "STUDENT_MIN_BETA": STUDENT_MIN_BETA,
        },
        "model_configs": {
            "rcgp_kwargs": rcgp_kwargs,
            "a2rcgp_kwargs": a2rcgp_kwargs,
            "student_t_kwargs": student_t_kwargs,
            "diagnostic_kwargs": diagnostic_kwargs,
        },
    }

    # Save configuration to JSON file
    save_experiment_config(config_dict, artifacts_dir)

    # Print configuration
    print("\n" + "=" * 80)
    print("LUNAR LANDER EXPERIMENT CONFIGURATION")
    print("=" * 80)
    print(f"Iterations: {N_ITERATIONS}, Initial points: {N_INITIAL}")
    print(f"Seeds: {N_SEEDS} (starting from {SEED})")
    print(f"Episodes per evaluation: {N_EPISODES}")
    print(f"Corruption type: {CORRUPTION_TYPE}")
    if CORRUPTION_TYPE == "time_budget":
        print(f"  Time budget alpha: {TIME_BUDGET_ALPHA} (T^{TIME_BUDGET_ALPHA})")
    print(f"Corruption value: {CORRUPTION_VALUE}")
    print("=" * 80 + "\n")

    # Create 36D continuous search space for linear policy
    obs_dim, action_dim = 8, 4
    n_params = obs_dim * action_dim + action_dim  # 36 total parameters

    # All parameters bounded in [-2.0, 2.0] to prevent divergence
    bounds = torch.tensor([[-2.0] * n_params, [2.0] * n_params], dtype=torch.double)
    search_space = SearchSpace.from_bounds(bounds, normalize=True)

    print("Lunar Lander Search Space:")
    print(
        f"  Policy parameters: {n_params} ({obs_dim}×{action_dim} weights + {action_dim} biases)"
    )
    print("  Parameter bounds: [-2.0, 2.0] for all parameters")
    print(f"  Search space dimensions: {search_space.n_dims}")
    print()

    # Create model factories following Forrester pattern
    def gp_model_factory(X, Y, **kwargs):
        """Factory for GP models."""
        merged_kwargs = {
            "fit_hyperparameters": FIT_HYPERPARAMETERS,
            "standardize": STANDARDIZE,
            "use_botorch_model": USE_BOTORCH_MODEL,
            **kwargs,
        }
        return create_gp_model(X, Y, **merged_kwargs)

    def rcgp_model_factory(X, Y, **kwargs):
        """Factory for RCGP models."""
        merged_kwargs = {**rcgp_kwargs, **kwargs}
        return create_rcgp_model(X, Y, **merged_kwargs)

    def student_t_model_factory(X, Y, **kwargs):
        """Factory for Student-t Process models."""
        merged_kwargs = {**student_t_kwargs, **kwargs}
        return create_student_t_model(X, Y, **merged_kwargs)

    def a2rcgp_model_factory(X, Y, **kwargs):
        """Factory for A2RCGP models."""
        merged_kwargs = {**a2rcgp_kwargs, **kwargs}
        return create_a2rcgp_model(X, Y, **merged_kwargs)

    def diagnostic_gp_model_factory(X, Y, **kwargs):
        """Factory for Diagnostic GP models."""
        merged_kwargs = {**diagnostic_kwargs, **kwargs}
        return create_diagnostic_gp_model(X, Y, **merged_kwargs)

    # Dictionary to store all results (list of lists)
    all_results_multiseed = {}

    # Test 1: Standard GP
    print("=" * 80)
    print("Test 1: Lunar Lander with Standard GP")
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
        CORRUPTION_TYPE,
        CORRUPTION_VALUE,
        SEED,
        {},
    )

    # Test 2: RCGP model
    print("\n" + "=" * 80)
    print("Test 2: Lunar Lander with Robust Conjugate GP")
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
        CORRUPTION_TYPE,
        CORRUPTION_VALUE,
        SEED,
        {},
    )

    # Test 3: Student-t Process
    print("\n" + "=" * 80)
    print("Test 3: Lunar Lander with Student-t Process")
    print("=" * 80)
    all_results_multiseed["Student-t"] = run_multiseed_experiment(
        "Student-t",
        student_t_model_factory,
        GP_SCHEDULER_TYPE,
        "student",
        search_space,
        N_SEEDS,
        N_ITERATIONS,
        N_INITIAL,
        CORRUPTION_TYPE,
        CORRUPTION_VALUE,
        SEED,
        {},
    )

    # Test 4: Diagnostic GP
    print("\n" + "=" * 80)
    print("Test 4: Lunar Lander with Diagnostic GP")
    print("=" * 80)
    all_results_multiseed["DiagnosticGP"] = run_multiseed_experiment(
        "DiagnosticGP",
        diagnostic_gp_model_factory,
        GP_SCHEDULER_TYPE,
        "diagnostic",
        search_space,
        N_SEEDS,
        N_ITERATIONS,
        N_INITIAL,
        CORRUPTION_TYPE,
        CORRUPTION_VALUE,
        SEED,
        {},
    )

    # Test 5: A2RCGP
    print("\n" + "=" * 80)
    print("Test 5: Lunar Lander with A2RCGP")
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
        CORRUPTION_TYPE,
        CORRUPTION_VALUE,
        SEED,
        {},
    )

    # Aggregate results
    aggregated_results = {}
    for model_name, results_list in all_results_multiseed.items():
        aggregated_results[model_name] = aggregate_results_across_seeds(results_list)

    # Find optimal value across all experiments (max true reward)
    # We need to look at all seeds
    optimal_value = -float("inf")
    for model_results in all_results_multiseed.values():
        for seed_results_dict in model_results:
            # runner.run returns a dict with 'all_results' key containing the list of EvaluationResult objects
            seed_results_list = seed_results_dict["all_results"]
            seed_max = max(r.y_true for r in seed_results_list)
            if seed_max > optimal_value:
                optimal_value = seed_max

    print(f"\nOptimal value (max true reward across all seeds): {optimal_value:.1f}")

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
        experiment_name="lunar_lander_experiment",
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
                results=results,
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

    regret_save_path = os.path.join(artifacts_dir, "regret")
    colors = {
        "RCGP": "blue",
        "GP": "orange",
        "Student-t": "green",
        "A2RCGP": "red",
        "DiagnosticGP": "purple",
    }
    config = PlotConfig(figsize=(15, 10))

    regret_fig, simple_regret_fig = plot_regret_comparison_multiseed(
        results_dict=aggregated_results,
        optimal_value=optimal_value,
        n_seeds=N_SEEDS,
        save_path=regret_save_path,
        config=config,
        colors=colors,
    )

    plt.close(regret_fig)
    plt.close(simple_regret_fig)

    # Create optimization trajectories for the first seed (as example)
    trajectories_plot_path = os.path.join(
        artifacts_dir, "optimization_trajectories_seed0.png"
    )
    first_seed_results = {
        k: v["all_results_flat"] for k, v in aggregated_results.items()
    }
    plot_optimization_trajectories(
        first_seed_results,
        trajectories_plot_path,
        title=f"Lunar Lander: Optimization Trajectories (Seed {SEED})",
    )

    # Show optimization trajectories for best model (from first seed)
    # We use aggregated_results which contains 'all_results_flat' (seed 0 results)
    best_model_name = max(
        aggregated_results.keys(),
        key=lambda name: max(
            r.y_true for r in aggregated_results[name]["all_results_flat"]
        ),
    )
    best_model_results = aggregated_results[best_model_name]["all_results_flat"]
    best_val = max(r.y_true for r in best_model_results)

    print(f"\n{best_model_name} optimization trajectory (best model, seed {SEED}):")
    for i, result in enumerate(best_model_results):
        marker = "→" if i < N_INITIAL else "BO"
        corrupted = "*" if result.y_observed != result.y_true else " "
        print(
            f"  {marker} Episode {i + 1}: observed = {result.y_observed:.1f}, true = {result.y_true:.1f} {corrupted}"
        )

    # Print model hyperparameters (from first seed)
    print("\n" + "=" * 80)
    print("MODEL HYPERPARAMETERS (From First Seed)")
    print("=" * 80)

    for model_name, results in aggregated_results.items():
        model = results["final_model"]
        print(f"\n{model_name} Model:")

        try:
            print(
                f"  Noise std (sigma): {torch.sqrt(model.likelihood.noise).item():.4f}"
            )

            # Handle different covariance module structures
            covar = model.covar_module
            if hasattr(covar, "base_kernel"):
                # ScaleKernel case
                if hasattr(covar.base_kernel, "lengthscale"):
                    print(
                        f"  Lengthscale: {covar.base_kernel.lengthscale.mean().item():.4f}"
                    )
                if hasattr(covar, "outputscale"):
                    print(f"  Output scale: {covar.outputscale.item():.4f}")
            else:
                # Direct kernel case
                if hasattr(covar, "lengthscale"):
                    print(f"  Lengthscale: {covar.lengthscale.mean().item():.4f}")
                if hasattr(covar, "outputscale"):
                    print(f"  Output scale: {covar.outputscale.item():.4f}")

            if hasattr(model.mean_module, "constant"):
                print(f"  Mean constant: {model.mean_module.constant.item():.4f}")

            # Model-specific parameters
            if hasattr(model, "weighting_function"):
                print(f"  Plateau width: {model.weighting_function.plateau_width:.4f}")
                print(f"  C parameter: {model.weighting_function.c:.4f}")

            if hasattr(model, "nu"):
                print(f"  Degrees of freedom (nu): {model.nu.item():.2f}")

            if model_name == "DiagnosticGP":
                try:
                    diagnostic_info = model.get_diagnostic_info()
                    print(f"  Total points: {diagnostic_info['total_points']}")
                    print(f"  Outliers detected: {diagnostic_info['num_outliers']}")
                    if diagnostic_info["outlier_indices"]:
                        print(
                            f"  Outlier indices: {diagnostic_info['outlier_indices']}"
                        )

                    # Show underlying model hyperparameters
                    underlying_model = model.model
                    print("  Underlying Model:")
                    print(
                        f"    Noise std (sigma): {torch.sqrt(underlying_model.likelihood.noise).item():.4f}"
                    )

                    covar = underlying_model.covar_module
                    if hasattr(covar, "base_kernel"):
                        if hasattr(covar.base_kernel, "lengthscale"):
                            print(
                                f"    Lengthscale: {covar.base_kernel.lengthscale.mean().item():.4f}"
                            )
                        if hasattr(covar, "outputscale"):
                            print(f"    Output scale: {covar.outputscale.item():.4f}")
                    else:
                        if hasattr(covar, "lengthscale"):
                            print(
                                f"    Lengthscale: {covar.lengthscale.mean().item():.4f}"
                            )
                        if hasattr(covar, "outputscale"):
                            print(f"    Output scale: {covar.outputscale.item():.4f}")

                    if hasattr(underlying_model.mean_module, "constant"):
                        print(
                            f"    Mean constant: {underlying_model.mean_module.constant.item():.4f}"
                        )
                except Exception as e:
                    print(f"  Could not extract diagnostic info: {e}")

            if model_name == "A2RCGP":
                print("  Inner RCGP:")
                print(
                    f"    Plateau width: {model.inner_rcgp.weighting_function.plateau_width:.4f}"
                )
                print(f"    C parameter: {model.inner_rcgp.weighting_function.c:.4f}")
                print("  Outer RCGP:")
                print(
                    f"    Plateau width: {model.weighting_function.plateau_width:.4f}"
                )
                print(f"    C parameter: {model.weighting_function.c:.4f}")
                corruption_results = model.detect_corruptions()
                print(
                    f"  Corruptions detected (inner): {corruption_results['inner'].sum().item()}"
                )
                print(
                    f"  Corruptions detected (outer): {corruption_results['outer'].sum().item()}"
                )

        except Exception as e:
            print(f"  Could not extract hyperparameters: {e}")

    # Calculate and print regret metrics
    print("\n" + "=" * 80)
    print("REGRET ANALYSIS")
    print("=" * 80)

    # Use multiseed comparison for proper aggregation across all seeds
    # We need to construct the results_dict expected by compare_experiments_multiseed
    # which is {model_name: [list of results for seed 1, list of results for seed 2, ...]}
    # aggregated_results['all_results'] already has this structure
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
    print(f"Best model (seed {SEED}): {best_model_name} with reward {best_val:.1f}")
    print(f"Total evaluations per model per seed: {N_ITERATIONS}")
    print(f"Total seeds: {N_SEEDS}")
    print(f"Corruption type: {CORRUPTION_TYPE}")
    if CORRUPTION_TYPE == "time_budget":
        print(f"Time budget alpha: {TIME_BUDGET_ALPHA}")

    print(f"\nAll artifacts saved to: {artifacts_dir}/")
    print("Files saved:")
    print("  - experiment_config.json (configuration)")
    print("  - regret_comparison_multiseed.png (cumulative regret plot)")
    print("  - simple_regret_multiseed.png (simple regret plot)")
    print("  - optimization_trajectories_seed0.png (trajectory plots)")
    print("  - experiment_results.pkl (pickle format)")
    print("  - experiment_results.json (JSON format)")
    print("  - multiseed_summary.json (aggregated stats)")
    print("  - individual_seeds/ (folder with per-seed JSONs)")


if __name__ == "__main__":
    main()
