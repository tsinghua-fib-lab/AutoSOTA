"""Test the new clean API with CartPole RL policy optimization."""

import os
import torch
import json
import matplotlib.pyplot as plt
import numpy as np
import time
from datetime import datetime
from bo_framework import SearchSpace, ExperimentRunner
from bo_framework.base.acquisition import UCBAcquisition
from bo_framework.base.schedulers import ConstantBetaScheduler, TheoryGuidedScheduler, RCGPScheduler
from bo_framework.models.factory import (
    create_gp_model,
    create_rcgp_model,
    create_student_t_model,
    create_a2rcgp_model,
    create_diagnostic_gp_model
)
from bo_framework.corruption.composable import (
    ComposableCorruptor,
    TimeBudgetDecider,
    ConstantStrategy
)
from bo_framework.wrappers.corrupted import CorruptedEvaluator
from experiments.cartpole.evaluator import CartPoleEvaluator
# Note: Single-seed utilities removed for multi-seed implementation
# from utilities.io import save_experiment_results, save_comparison_table
# from utilities.regret_analysis import compare_experiments, print_comparison_table


# Experiment parameters
N_ITERATIONS = 300
N_INITIAL = 5
SEED = 42
N_SEEDS = 10  # Number of seeds to run for multi-seed experiment
NUM_EPISODES = 5  # Number of episodes to average over per evaluation
STANDARDIZE = True
FIT_HYPERPARAMETERS = True
USE_BOTORCH_MODEL = True

# Corruption configuration
# Choose: 'time_budget', 'none'
CORRUPTION_TYPE = 'time_budget'

# Time budget parameters
TIME_BUDGET_ALPHA = 1/3  # T^alpha budget (0.5 = sqrt(T))

# Constant corruption value
CORRUPTION_VALUE = 1000.0

# Beta scheduler configuration
# For RCGP: Choose from 'constant', 'theory', 'rcgp-constant', 'rcgp-theory'
RCGP_SCHEDULER_TYPE = 'theory'

# For GP/Student-t/Diagnostic: Choose from 'constant', 'theory'
GP_SCHEDULER_TYPE = 'theory'

# For A2RCGP: Choose from 'constant', 'theory', 'rcgp-constant', 'rcgp-theory'
A2RCGP_SCHEDULER_TYPE = 'theory'

# Beta scheduling parameters
CONSTANT_BETA = 2.0
RCGP_SCALE = 1.0  # Scale factor for RCGP adaptive term

# Theory scheduler parameters
THEORY_SCALE = 1.7  # Scale for theory-guided beta schedule
THEORY_OFFSET = 2   # Offset to handle early iterations

# Student-specific scheduler parameters
STUDENT_THEORY_SCALE = 1.0   # Lower scale for student-t models
STUDENT_THEORY_OFFSET = 1    # Lower offset for student-t models
STUDENT_MIN_BETA = 0.1       # Lower minimum beta for student-t models

# RCGP configuration
rcgp_kwargs = {
    "param_handling_dict": {
        "plateau_width": {"method": "heuristics"},  # Use heuristics for high-dim problem
        "c": {"method": "manual", "value": 1.0},
        "sigma": {"method": "fit"},  # Fit the noise parameter
        "mean": {"method": "fit"}  # Fit the mean parameter
    },
    "fitting_objective_type": "wloo-cv",  # Use weighted leave-one-out cross-validation
    "optimizer_type": "lbfgs",
    "standardize": STANDARDIZE,
    "verbose": False
}

# A2RCGP configuration with inner and outer model parameters
a2rcgp_kwargs = {
    "inner_param_handling_dict": {
        "plateau_width": {"method": "heuristics"},  # Use heuristics for high-dim
        "c": {"method": "manual", "value": 1.0},
        "sigma": {"method": "fit"},
        "mean": {"method": "fit"}
    },
    "outer_param_handling_dict": {
        "plateau_width": {"method": "heuristics"},  # Use heuristics for high-dim
        "c": {"method": "manual", "value": 0.8},
        "sigma": {"method": "fit"},
        "mean": {"method": "fit"}
    },
    "fitting_objective_type": "wloo-cv",
    "optimizer_type": "lbfgs",
    "standardize": STANDARDIZE,
    "verbose": False
}

# Student-t Process configuration
student_t_kwargs = {
    'nu': 3.0,  # Degrees of freedom (lower = heavier tails)
    'standardize': STANDARDIZE,
    'fit_hyperparameters': FIT_HYPERPARAMETERS,
    'optimizer_type': 'lbfgs'
}

# Diagnostic GP (OD-BO) configuration
diagnostic_kwargs = {
    "n_init": 5,  # Start diagnosis after 5 points
    "n_schedule": 1,  # Run diagnosis every iteration
    "nu": 4.0,  # Student-t degrees of freedom
    "alpha": 0.05,  # Outlier threshold
    "fitting_kwargs": {
        "num_iterations": 200,
        "verbose": False
    },
    "model_kwargs": {
        "standardize": STANDARDIZE,
        "fit_hyperparameters": FIT_HYPERPARAMETERS,
        "use_botorch_model": USE_BOTORCH_MODEL
    }
}


def create_timestamped_folder(base_dir="artifacts"):
    """Create a timestamped folder for experiment results.

    Args:
        base_dir: Base directory to create the timestamped folder in

    Returns:
        Path to the created timestamped folder
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"cartpole_experiment_{timestamp}"
    folder_path = os.path.join(base_dir, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path


def save_experiment_config(config_dict, folder_path):
    """Save experiment configuration to JSON file.

    Args:
        config_dict: Dictionary containing all experiment parameters
        folder_path: Path to the experiment folder
    """
    # Add multi-seed specific parameters
    config_dict["experiment_parameters"]["N_SEEDS"] = N_SEEDS
    config_dict["experiment_parameters"]["SEED_RANGE"] = f"{SEED} to {SEED + N_SEEDS - 1}"
    
    config_path = os.path.join(folder_path, "experiment_config.json")
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)
    print(f"Experiment configuration saved to: {config_path}")


def create_scheduler(scheduler_type, model_type='gp'):
    """Create beta scheduler based on configuration.

    Args:
        scheduler_type: Type of scheduler ('constant', 'theory', 'rcgp-constant', 'rcgp-theory')
        model_type: Model type ('rcgp', 'gp', 'student', 'a2rcgp', 'diagnostic')

    Returns:
        BetaScheduler instance
    """
    if model_type == 'rcgp' or model_type == 'a2rcgp':
        if scheduler_type == 'constant':
            return ConstantBetaScheduler(beta=CONSTANT_BETA)
        elif scheduler_type == 'theory':
            return TheoryGuidedScheduler(
                scale=THEORY_SCALE,
                offset=THEORY_OFFSET,
                min_beta=1.0
            )
        elif scheduler_type == 'rcgp-constant':
            return RCGPScheduler(
                scale=RCGP_SCALE,
                base_scheduler=ConstantBetaScheduler(beta=CONSTANT_BETA)
            )
        elif scheduler_type == 'rcgp-theory':
            return RCGPScheduler(
                scale=RCGP_SCALE,
                base_scheduler=TheoryGuidedScheduler(
                    scale=THEORY_SCALE,
                    offset=THEORY_OFFSET,
                    min_beta=1.0
                )
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    else:  # GP, Student-t, or Diagnostic
        if scheduler_type == 'constant':
            return ConstantBetaScheduler(beta=CONSTANT_BETA)
        elif scheduler_type == 'theory':
            if model_type == 'student':
                # Student-t models use different parameters
                return TheoryGuidedScheduler(
                    scale=STUDENT_THEORY_SCALE,
                    offset=STUDENT_THEORY_OFFSET,
                    min_beta=STUDENT_MIN_BETA
                )
            else:
                # Standard GP and Diagnostic use regular parameters
                return TheoryGuidedScheduler(
                    scale=THEORY_SCALE,
                    offset=THEORY_OFFSET,
                    min_beta=1.0
                )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def create_corruptor_factory(corruption_type: str = 'time_budget'):
    """Create a factory function for the specified corruptor configuration.

    Args:
        corruption_type: Type of corruption ('time_budget', 'none')

    Returns:
        ComposableCorruptor instance or None
    """
    if corruption_type == 'none':
        return None

    if corruption_type == 'time_budget':
        decider = TimeBudgetDecider(
            alpha=TIME_BUDGET_ALPHA,
            skip_initial=True,
            n_initial=N_INITIAL
        )
    else:
        raise ValueError(f"Unknown corruption type: {corruption_type}")

    # Create constant strategy that returns the corruption value
    strategy = ConstantStrategy(corruption_value=CORRUPTION_VALUE)

    # Return composable corruptor
    return ComposableCorruptor(
        decider=decider,
        strategy=strategy,
        skip_initial=True
    )


def create_fresh_evaluator(corruption_type: str = 'time_budget', seed: int = None):
    """Create a fresh evaluator instance with corruption if needed.

    This ensures each experiment run gets a completely fresh evaluator
    to avoid sharing hidden state between runs.

    Args:
        corruption_type: Type of corruption ('time_budget', 'none')
        seed: Random seed for the evaluator (defaults to global SEED)

    Returns:
        Fresh evaluator instance (wrapped or unwrapped)
    """
    if seed is None:
        seed = SEED
    
    # Create base evaluator
    base_evaluator = CartPoleEvaluator(
        env_name="CartPole-v1",
        max_steps=500,
        render=False,  # No rendering for BO optimization
        seed=seed,
        num_episodes=NUM_EPISODES
    )

    # Create corruptor if needed
    corruptor = create_corruptor_factory(corruption_type)

    # Wrap evaluator if corruption is enabled
    if corruptor is not None:
        evaluator = CorruptedEvaluator(
            base_evaluator=base_evaluator,
            corruptor=corruptor,
            n_initial=N_INITIAL
        )
    else:
        evaluator = base_evaluator

    return evaluator


def initialize_multi_seed_results():
    """Initialize data structure to store results across seeds."""
    return {
        'GP': [],
        'RCGP': [],
        'Student-t': [],
        'DiagnosticGP': [],
        'A2RCGP': []
    }


def aggregate_results_across_seeds(multi_seed_results):
    """Aggregate results across seeds for each model."""
    aggregated = {}
    
    for model_name, seed_results in multi_seed_results.items():
        # Extract metrics for each seed
        best_observed_values = [r['best_observed_value'] for r in seed_results]
        best_true_values = [r['best_true_value'] for r in seed_results]
        all_results = [r['all_results'] for r in seed_results]
        
        # Calculate statistics
        aggregated[model_name] = {
            'best_observed_mean': np.mean(best_observed_values),
            'best_observed_std': np.std(best_observed_values),
            'best_true_mean': np.mean(best_true_values),
            'best_true_std': np.std(best_true_values),
            'individual_seeds': seed_results,
            'all_results_aggregated': all_results
        }
    
    return aggregated


def calculate_regret_statistics(multi_seed_results, optimal_value):
    """Calculate regret statistics across seeds."""
    regret_stats = {}
    
    for model_name, seed_results in multi_seed_results.items():
        cumulative_regrets = []
        simple_regrets = []
        
        for seed_result in seed_results:
            results = seed_result['all_results']
            cumulative_regrets.append(calculate_cumulative_regret(results, optimal_value))
            simple_regrets.append(calculate_simple_regret(results, optimal_value))
        
        # Calculate mean and std across seeds
        regret_stats[model_name] = {
            'cumulative_regret_mean': np.mean(cumulative_regrets, axis=0),
            'cumulative_regret_std': np.std(cumulative_regrets, axis=0),
            'simple_regret_mean': np.mean(simple_regrets, axis=0),
            'simple_regret_std': np.std(simple_regrets, axis=0)
        }
    
    return regret_stats


def print_progress(seed_idx, model_name, start_time):
    """Print progress information."""
    elapsed = time.time() - start_time
    remaining_seeds = N_SEEDS - seed_idx - 1
    if seed_idx > 0:
        estimated_remaining = elapsed * remaining_seeds / seed_idx
    else:
        estimated_remaining = 0
    
    print(f"Progress: Seed {seed_idx + 1}/{N_SEEDS}, Model: {model_name}")
    print(f"Elapsed: {elapsed:.1f}s, Estimated remaining: {estimated_remaining:.1f}s")


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


def plot_regret_comparison(results_dict, optimal_value, save_path_base, title="Regret Comparison"):
    """Plot both cumulative and simple regret comparisons for multiple models.

    Args:
        results_dict: Dictionary mapping model names to their results
        optimal_value: The optimal value to compare against
        save_path_base: Base path to save the plots (will add _cumulative.png and _simple.png)
        title: Plot title base
    """
    colors = ['blue', 'orange', 'green', 'red', 'purple']
    markers = ['o', 's', '^', 'D', 'v']

    # Plot 1: Cumulative Regret
    plt.figure(figsize=(12, 8))

    for i, (model_name, results) in enumerate(results_dict.items()):
        cumulative_regret = calculate_cumulative_regret(results, optimal_value)
        iterations = list(range(1, len(cumulative_regret) + 1))

        plt.plot(iterations, cumulative_regret,
                label=model_name,
                color=colors[i % len(colors)],
                marker=markers[i % len(markers)],
                linewidth=2,
                markersize=4,
                markevery=max(1, len(iterations) // 10))

    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Cumulative Regret', fontsize=12)
    plt.title(f"{title}: Cumulative Regret", fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    cumulative_path = save_path_base.replace('.png', '_cumulative.png')
    plt.savefig(cumulative_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Cumulative regret plot saved to: {cumulative_path}")

    # Plot 2: Simple Regret
    plt.figure(figsize=(12, 8))

    for i, (model_name, results) in enumerate(results_dict.items()):
        simple_regret = calculate_simple_regret(results, optimal_value)
        iterations = list(range(1, len(simple_regret) + 1))

        plt.plot(iterations, simple_regret,
                label=model_name,
                color=colors[i % len(colors)],
                marker=markers[i % len(markers)],
                linewidth=2,
                markersize=4,
                markevery=max(1, len(iterations) // 10))

    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Simple Regret', fontsize=12)
    plt.title(f"{title}: Simple Regret", fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    simple_path = save_path_base.replace('.png', '_simple.png')
    plt.savefig(simple_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Simple regret plot saved to: {simple_path}")


def plot_regret_comparison_multi_seed(regret_stats, save_path_base, title="Regret Comparison"):
    """Plot regret comparisons with error bars for multi-seed results."""
    colors = ['blue', 'orange', 'green', 'red', 'purple']
    
    # Plot 1: Cumulative Regret with Error Bars
    plt.figure(figsize=(12, 8))
    
    for i, (model_name, stats) in enumerate(regret_stats.items()):
        mean_regret = stats['cumulative_regret_mean']
        std_regret = stats['cumulative_regret_std']
        iterations = list(range(1, len(mean_regret) + 1))
        
        plt.plot(iterations, mean_regret,
                label=model_name,
                color=colors[i % len(colors)],
                linewidth=2)
        
        # Add shaded area for 1 standard deviation
        plt.fill_between(iterations,
                        mean_regret - std_regret,
                        mean_regret + std_regret,
                        alpha=0.3,
                        color=colors[i % len(colors)])
    
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Cumulative Regret', fontsize=12)
    plt.title(f"{title}: Cumulative Regret (Mean ± 1σ across {N_SEEDS} seeds)", fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    cumulative_path = save_path_base.replace('.png', '_cumulative.png')
    plt.savefig(cumulative_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Cumulative regret plot saved to: {cumulative_path}")
    
    # Plot 2: Simple Regret with Error Bars
    plt.figure(figsize=(12, 8))
    
    for i, (model_name, stats) in enumerate(regret_stats.items()):
        mean_regret = stats['simple_regret_mean']
        std_regret = stats['simple_regret_std']
        iterations = list(range(1, len(mean_regret) + 1))
        
        plt.plot(iterations, mean_regret,
                label=model_name,
                color=colors[i % len(colors)],
                linewidth=2)
        
        # Add shaded area for 1 standard deviation
        plt.fill_between(iterations,
                        mean_regret - std_regret,
                        mean_regret + std_regret,
                        alpha=0.3,
                        color=colors[i % len(colors)])
    
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Simple Regret', fontsize=12)
    plt.title(f"{title}: Simple Regret (Mean ± 1σ across {N_SEEDS} seeds)", fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    simple_path = save_path_base.replace('.png', '_simple.png')
    plt.savefig(simple_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Simple regret plot saved to: {simple_path}")


def save_multi_seed_results(multi_seed_results, aggregated_results, artifacts_dir):
    """Save multi-seed results to files."""
    import pickle
    
    # Save individual seed results
    individual_path = os.path.join(artifacts_dir, "multi_seed_individual_results.pkl")
    with open(individual_path, 'wb') as f:
        pickle.dump(multi_seed_results, f)
    print(f"Individual seed results saved to: {individual_path}")
    
    # Save aggregated results
    aggregated_path = os.path.join(artifacts_dir, "multi_seed_aggregated_results.pkl")
    with open(aggregated_path, 'wb') as f:
        pickle.dump(aggregated_results, f)
    print(f"Aggregated results saved to: {aggregated_path}")
    
    # Save summary statistics as JSON
    summary_stats = {}
    for model_name, stats in aggregated_results.items():
        summary_stats[model_name] = {
            'best_observed_mean': float(stats['best_observed_mean']),
            'best_observed_std': float(stats['best_observed_std']),
            'best_true_mean': float(stats['best_true_mean']),
            'best_true_std': float(stats['best_true_std']),
            'n_seeds': len(stats['individual_seeds'])
        }
    
    summary_path = os.path.join(artifacts_dir, "multi_seed_summary_stats.json")
    with open(summary_path, 'w') as f:
        json.dump(summary_stats, f, indent=2)
    print(f"Summary statistics saved to: {summary_path}")


def print_multi_seed_summary(aggregated_results, optimal_value):
    """Print summary of multi-seed results."""
    print(f"\n{'='*80}")
    print("MULTI-SEED EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    
    print(f"\n{'Method':<15} {'Best Observed':<20} {'Best True':<20} {'Seeds':<8}")
    print(f"{'':<15} {'Mean ± Std':<20} {'Mean ± Std':<20} {'':<8}")
    print("-" * 75)
    
    for model_name, stats in aggregated_results.items():
        best_obs_mean = stats['best_observed_mean']
        best_obs_std = stats['best_observed_std']
        best_true_mean = stats['best_true_mean']
        best_true_std = stats['best_true_std']
        n_seeds = len(stats['individual_seeds'])
        
        print(f"{model_name:<15} {best_obs_mean:.1f} ± {best_obs_std:.1f}    {best_true_mean:.1f} ± {best_true_std:.1f}    {n_seeds:<8}")
    
    print(f"\nOptimal value (max true reward across all seeds): {optimal_value:.1f}")
    print(f"Total seeds run: {N_SEEDS}")
    print(f"Iterations per seed: {N_ITERATIONS}")


def plot_optimization_trajectories(results_dict, save_path, title="Optimization Trajectories"):
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

        ax.plot(iterations, observed_values, 'o-', label='Observed', linewidth=2, markersize=6)
        ax.plot(iterations, true_values, 's-', label='True', linewidth=2, markersize=6, alpha=0.7)
        ax.axvline(x=N_INITIAL, color='gray', linestyle='--', alpha=0.5, label='End of initial')

        ax.set_xlabel('Iteration')
        ax.set_ylabel('Reward')
        ax.set_title(f'{model_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Hide any unused subplots
    for idx in range(len(results_dict), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Optimization trajectories saved to: {save_path}")


def main():
    """Run multi-seed CartPole experiment."""

    # Create timestamped folder for results
    artifacts_dir = create_timestamped_folder()
    print(f"Created experiment folder: {artifacts_dir}")

    # Create configuration dictionary
    config_dict = {
        "experiment_info": {
            "name": "CartPole Multi-Seed Experiment",
            "timestamp": datetime.now().isoformat(),
            "script": "test_cartpole_clean_api.py"
        },
        "experiment_parameters": {
            "N_ITERATIONS": N_ITERATIONS,
            "N_INITIAL": N_INITIAL,
            "SEED": SEED,
            "NUM_EPISODES": NUM_EPISODES,
            "STANDARDIZE": STANDARDIZE,
            "FIT_HYPERPARAMETERS": FIT_HYPERPARAMETERS,
            "USE_BOTORCH_MODEL": USE_BOTORCH_MODEL
        },
        "corruption_config": {
            "CORRUPTION_TYPE": CORRUPTION_TYPE,
            "TIME_BUDGET_ALPHA": TIME_BUDGET_ALPHA,
            "CORRUPTION_VALUE": CORRUPTION_VALUE
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
            "STUDENT_MIN_BETA": STUDENT_MIN_BETA
        },
        "model_configs": {
            "rcgp_kwargs": rcgp_kwargs,
            "a2rcgp_kwargs": a2rcgp_kwargs,
            "student_t_kwargs": student_t_kwargs,
            "diagnostic_kwargs": diagnostic_kwargs
        }
    }

    # Save configuration to JSON file
    save_experiment_config(config_dict, artifacts_dir)

    # Print configuration
    print("\n" + "=" * 80)
    print("CARTPOLE MULTI-SEED EXPERIMENT CONFIGURATION")
    print("=" * 80)
    print(f"Iterations per seed: {N_ITERATIONS}, Initial points: {N_INITIAL}")
    print(f"Number of seeds: {N_SEEDS}")
    print(f"Seed range: {SEED} to {SEED + N_SEEDS - 1}")
    print(f"Episodes per evaluation: {NUM_EPISODES}")
    print(f"Corruption type: {CORRUPTION_TYPE}")
    if CORRUPTION_TYPE == 'time_budget':
        print(f"  Time budget alpha: {TIME_BUDGET_ALPHA} (T^{TIME_BUDGET_ALPHA})")
    print(f"Corruption value: {CORRUPTION_VALUE}")
    print(f"RCGP scheduler: {RCGP_SCHEDULER_TYPE}")
    print(f"GP scheduler: {GP_SCHEDULER_TYPE}")
    print(f"A2RCGP scheduler: {A2RCGP_SCHEDULER_TYPE}")
    print(f"Constant beta: {CONSTANT_BETA}, RCGP scale: {RCGP_SCALE}")
    print("=" * 80 + "\n")

    # Create 10D continuous search space for linear policy
    obs_dim, action_dim = 4, 2
    n_params = obs_dim * action_dim + action_dim  # 10 total parameters

    # All parameters bounded in [-2.0, 2.0] to prevent divergence
    bounds = torch.tensor([[-2.0] * n_params, [2.0] * n_params], dtype=torch.double)
    search_space = SearchSpace.from_bounds(bounds, normalize=True)

    print("CartPole Search Space:")
    print(f"  Policy parameters: {n_params} ({obs_dim}×{action_dim} weights + {action_dim} biases)")
    print("  Parameter bounds: [-2.0, 2.0] for all parameters")
    print(f"  Search space dimensions: {search_space.n_dims}")
    print()

    # Create model factories following Forrester pattern
    def gp_model_factory(X, Y, **kwargs):
        """Factory for GP models."""
        merged_kwargs = {
            'fit_hyperparameters': FIT_HYPERPARAMETERS,
            'standardize': STANDARDIZE,
            'use_botorch_model': USE_BOTORCH_MODEL,
            **kwargs
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

    # Create schedulers
    gp_scheduler = create_scheduler(GP_SCHEDULER_TYPE, 'gp')
    rcgp_scheduler = create_scheduler(RCGP_SCHEDULER_TYPE, 'rcgp')
    student_scheduler = create_scheduler(GP_SCHEDULER_TYPE, 'student')
    diagnostic_scheduler = create_scheduler(GP_SCHEDULER_TYPE, 'diagnostic')
    a2rcgp_scheduler = create_scheduler(A2RCGP_SCHEDULER_TYPE, 'a2rcgp')

    print(f"GP using scheduler: {gp_scheduler.__class__.__name__}")
    print(f"RCGP using scheduler: {rcgp_scheduler.__class__.__name__}")
    print(f"Student-t using scheduler: {student_scheduler.__class__.__name__}")
    print(f"Diagnostic GP using scheduler: {diagnostic_scheduler.__class__.__name__}")
    print(f"A2RCGP using scheduler: {a2rcgp_scheduler.__class__.__name__}")
    print()

    # Initialize multi-seed results storage
    multi_seed_results = initialize_multi_seed_results()
    
    # Start timing
    experiment_start_time = time.time()

    # Run experiments for each seed
    for seed_idx in range(N_SEEDS):
        current_seed = SEED + seed_idx  # Use different seed for each run
        
        print(f"\n{'='*80}")
        print(f"SEED {seed_idx + 1}/{N_SEEDS} (seed={current_seed})")
        print(f"{'='*80}")
        
        # Run all models for this seed
        seed_results = {}
        
        # Test 1: Standard GP
        print(f"\nSeed {seed_idx + 1}: Running GP...")
        print_progress(seed_idx, "GP", experiment_start_time)
        evaluator_gp = create_fresh_evaluator(CORRUPTION_TYPE, current_seed)
        runner_gp = ExperimentRunner(search_space, evaluator_gp)
        results_gp = runner_gp.run(
            n_iterations=N_ITERATIONS,
            n_initial=N_INITIAL,
            model_factory=gp_model_factory,
            acquisition_factory=UCBAcquisition.create,
            beta_scheduler=gp_scheduler,
            seed=current_seed,
            model_kwargs={},
            verbose=False  # Reduce verbosity for multi-seed
        )
        seed_results['GP'] = results_gp
        evaluator_gp.close() if hasattr(evaluator_gp, 'close') else None
        if hasattr(evaluator_gp, 'base_evaluator'):
            evaluator_gp.base_evaluator.close() if hasattr(evaluator_gp.base_evaluator, 'close') else None

        # Test 2: RCGP
        print(f"Seed {seed_idx + 1}: Running RCGP...")
        print_progress(seed_idx, "RCGP", experiment_start_time)
        evaluator_rcgp = create_fresh_evaluator(CORRUPTION_TYPE, current_seed)
        runner_rcgp = ExperimentRunner(search_space, evaluator_rcgp)
        results_rcgp = runner_rcgp.run(
            n_iterations=N_ITERATIONS,
            n_initial=N_INITIAL,
            model_factory=rcgp_model_factory,
            acquisition_factory=UCBAcquisition.create,
            beta_scheduler=rcgp_scheduler,
            seed=current_seed,
            model_kwargs={},
            verbose=False
        )
        seed_results['RCGP'] = results_rcgp
        evaluator_rcgp.close() if hasattr(evaluator_rcgp, 'close') else None
        if hasattr(evaluator_rcgp, 'base_evaluator'):
            evaluator_rcgp.base_evaluator.close() if hasattr(evaluator_rcgp.base_evaluator, 'close') else None

        # Test 3: Student-t Process
        print(f"Seed {seed_idx + 1}: Running Student-t...")
        print_progress(seed_idx, "Student-t", experiment_start_time)
        evaluator_student = create_fresh_evaluator(CORRUPTION_TYPE, current_seed)
        runner_student = ExperimentRunner(search_space, evaluator_student)
        results_student = runner_student.run(
            n_iterations=N_ITERATIONS,
            n_initial=N_INITIAL,
            model_factory=student_t_model_factory,
            acquisition_factory=UCBAcquisition.create,
            beta_scheduler=student_scheduler,
            seed=current_seed,
            model_kwargs={},
            verbose=False
        )
        seed_results['Student-t'] = results_student
        evaluator_student.close() if hasattr(evaluator_student, 'close') else None
        if hasattr(evaluator_student, 'base_evaluator'):
            evaluator_student.base_evaluator.close() if hasattr(evaluator_student.base_evaluator, 'close') else None

        # Test 4: Diagnostic GP
        print(f"Seed {seed_idx + 1}: Running Diagnostic GP...")
        print_progress(seed_idx, "Diagnostic GP", experiment_start_time)
        evaluator_diagnostic = create_fresh_evaluator(CORRUPTION_TYPE, current_seed)
        runner_diagnostic = ExperimentRunner(search_space, evaluator_diagnostic)
        results_diagnostic = runner_diagnostic.run(
            n_iterations=N_ITERATIONS,
            n_initial=N_INITIAL,
            model_factory=diagnostic_gp_model_factory,
            acquisition_factory=UCBAcquisition.create,
            beta_scheduler=diagnostic_scheduler,
            seed=current_seed,
            model_kwargs={},
            verbose=False
        )
        seed_results['DiagnosticGP'] = results_diagnostic
        evaluator_diagnostic.close() if hasattr(evaluator_diagnostic, 'close') else None
        if hasattr(evaluator_diagnostic, 'base_evaluator'):
            evaluator_diagnostic.base_evaluator.close() if hasattr(evaluator_diagnostic.base_evaluator, 'close') else None

        # Test 5: A2RCGP
        print(f"Seed {seed_idx + 1}: Running A2RCGP...")
        print_progress(seed_idx, "A2RCGP", experiment_start_time)
        evaluator_a2rcgp = create_fresh_evaluator(CORRUPTION_TYPE, current_seed)
        runner_a2rcgp = ExperimentRunner(search_space, evaluator_a2rcgp)
        results_a2rcgp = runner_a2rcgp.run(
            n_iterations=N_ITERATIONS,
            n_initial=N_INITIAL,
            model_factory=a2rcgp_model_factory,
            acquisition_factory=UCBAcquisition.create,
            beta_scheduler=a2rcgp_scheduler,
            seed=current_seed,
            model_kwargs={},
            verbose=False
        )
        seed_results['A2RCGP'] = results_a2rcgp
        evaluator_a2rcgp.close() if hasattr(evaluator_a2rcgp, 'close') else None
        if hasattr(evaluator_a2rcgp, 'base_evaluator'):
            evaluator_a2rcgp.base_evaluator.close() if hasattr(evaluator_a2rcgp.base_evaluator, 'close') else None

        # Store results for this seed
        for model_name, results in seed_results.items():
            multi_seed_results[model_name].append(results)
        
        # Print progress
        print(f"Completed seed {seed_idx + 1}/{N_SEEDS}")
        elapsed = time.time() - experiment_start_time
        print(f"Total elapsed time: {elapsed:.1f}s")

    # Aggregate results across seeds
    print(f"\n{'='*80}")
    print("AGGREGATING RESULTS ACROSS SEEDS")
    print(f"{'='*80}")
    
    aggregated_results = aggregate_results_across_seeds(multi_seed_results)
    
    # Find optimal value across all seeds and models
    optimal_value = max(
        max(r['best_true_value'] for r in seed_results)
        for seed_results in multi_seed_results.values()
    )
    
    # Calculate regret statistics
    regret_stats = calculate_regret_statistics(multi_seed_results, optimal_value)
    
    # Create plots with error bars
    print("\n" + "=" * 80)
    print("GENERATING MULTI-SEED PLOTS")
    print("=" * 80)
    
    regret_plot_path = os.path.join(artifacts_dir, "regret_comparison_multi_seed.png")
    plot_regret_comparison_multi_seed(regret_stats, regret_plot_path, title="CartPole Multi-Seed")
    
    # Save multi-seed results
    save_multi_seed_results(multi_seed_results, aggregated_results, artifacts_dir)
    
    # Print final summary
    print_multi_seed_summary(aggregated_results, optimal_value)
    
    # Print final timing
    total_time = time.time() - experiment_start_time
    print(f"\nTotal experiment time: {total_time:.1f}s")
    print(f"Average time per seed: {total_time/N_SEEDS:.1f}s")
    
    print(f"\nAll artifacts saved to: {artifacts_dir}/")
    print("Files saved:")
    print("  - experiment_config.json (configuration)")
    print("  - regret_comparison_multi_seed_cumulative.png (cumulative regret with error bars)")
    print("  - regret_comparison_multi_seed_simple.png (simple regret with error bars)")
    print("  - multi_seed_individual_results.pkl (individual seed results)")
    print("  - multi_seed_aggregated_results.pkl (aggregated results)")
    print("  - multi_seed_summary_stats.json (summary statistics)")

    print("\nMulti-seed experiment completed successfully!")
    print("The framework successfully handles 10D continuous RL policy optimization")
    print("with multiple robust models, corruption handling, and statistical analysis!")


if __name__ == "__main__":
    main()
