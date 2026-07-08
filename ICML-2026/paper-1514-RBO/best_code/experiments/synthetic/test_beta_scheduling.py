"""Test beta scheduling with different schedulers on the Forrester function."""

import os
import torch
import matplotlib.pyplot as plt
from bo_framework import SearchSpace, Dimension, ExperimentRunner
from bo_framework.base.acquisition import UCBAcquisition
from bo_framework.base.schedulers import ConstantBetaScheduler, TheoryGuidedScheduler, RCGPScheduler
from bo_framework.synthetic.evaluators import SyntheticEvaluator
from bo_framework.wrappers.noisy import NoisyEvaluator
from experiments.synthetic.functions import ForresterFunction
from bo_framework.models.factory import create_gp_model, create_rcgp_model
from utilities.plotting import plot_experiment_summary, PlotConfig
from utilities.io import save_experiment_results


N_ITERATIONS = 30
N_INITIAL = 5
NOISE_STD = 0.5


def run_experiment_with_scheduler(scheduler, scheduler_name, model_factory, model_name, seed=42):
    """Run a single experiment with a given scheduler and model."""
    
    # Create search space
    search_space = SearchSpace((
        Dimension(name="x0", type="continuous", bounds=(0.0, 1.0), normalize=True),
    ))
    
    # Create evaluator with noise
    forrester_func = ForresterFunction()
    clean_evaluator = SyntheticEvaluator(forrester_func)
    noisy_evaluator = NoisyEvaluator(clean_evaluator, noise_std=NOISE_STD, seed=seed)
    
    # Create runner
    runner = ExperimentRunner(search_space, noisy_evaluator)
    
    # Model kwargs
    if model_name == "RCGP":
        model_kwargs = {
            "param_handling_dict": {
                "plateau_width": {"method": "heuristics"},
                "c": {"method": "manual", "value": 1.0},
                "sigma": {"method": "fit"},
                "mean": {"method": "fit"}
            },
            "fitting_objective_type": "wloo-cv",
            "optimizer_type": "lbfgs",
            "verbose": False
        }
    else:
        model_kwargs = {
            'fit_hyperparameters': True,
            'standardize': True,
            'use_botorch_model': True
        }
    
    print(f"\n{'='*60}")
    print(f"Running {model_name} with {scheduler_name}")
    print(f"{'='*60}")
    
    # Run experiment
    results = runner.run(
        n_iterations=N_ITERATIONS,
        n_initial=N_INITIAL,
        model_factory=model_factory,
        acquisition_factory=UCBAcquisition.create,
        model_kwargs=model_kwargs,
        beta_scheduler=scheduler,
        seed=seed,
        verbose=True
    )
    
    return results


def plot_beta_progression(all_results, save_path="beta_progression.png"):
    """Plot how beta changes over iterations for different schedulers."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    colors = ['blue', 'orange', 'green', 'red', 'purple']
    
    # Plot beta values
    for i, (name, betas) in enumerate(all_results['betas'].items()):
        ax1.plot(range(len(betas)), betas, label=name, color=colors[i % len(colors)], linewidth=2)
    
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Beta Value')
    ax1.set_title('Beta Scheduling Progression')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot cumulative regret comparison
    optimal_value = ForresterFunction().optimal_value
    for i, (name, results) in enumerate(all_results['results'].items()):
        Y_true = results['Y_true']
        cumulative_regret = torch.cumsum(optimal_value - Y_true, dim=0)
        ax2.plot(range(len(cumulative_regret)), cumulative_regret.numpy(), 
                label=name, color=colors[i % len(colors)], linewidth=2)
    
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Cumulative Regret')
    ax2.set_title('Performance Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig


def main():
    """Compare different beta scheduling strategies."""
    
    # Create schedulers to test
    schedulers = [
        (ConstantBetaScheduler(beta=0.5), "Constant β=0.5"),
        (ConstantBetaScheduler(beta=2.0), "Constant β=2.0"),
        (ConstantBetaScheduler(beta=4.0), "Constant β=4.0"),
        (TheoryGuidedScheduler(scale=1.0), "Theory-Guided"),
        (TheoryGuidedScheduler(scale=2.0), "Theory-Guided (2x)"),
    ]
    
    # Store results
    all_results = {
        'results': {},
        'betas': {}
    }
    
    # Test with standard GP
    print("\n" + "="*80)
    print("TESTING WITH STANDARD GP")
    print("="*80)
    
    for scheduler, name in schedulers:
        results = run_experiment_with_scheduler(
            scheduler, name, create_gp_model, "Standard GP", seed=42
        )
        all_results['results'][f"GP_{name}"] = results
        
        # Extract beta values used (we need to modify the code to track these)
        # For now, let's compute them
        betas = []
        for i in range(N_ITERATIONS):
            beta = scheduler.get_beta(i, N_ITERATIONS, None)
            betas.append(beta)
        all_results['betas'][f"GP_{name}"] = betas
    
    # Test with RCGP (subset of schedulers)
    print("\n" + "="*80)
    print("TESTING WITH RCGP")
    print("="*80)
    
    rcgp_schedulers = [
        (ConstantBetaScheduler(beta=2.0), "Constant β=2.0"),
        (TheoryGuidedScheduler(scale=1.0), "Theory-Guided"),
        (RCGPScheduler(
            scale=1.0,
            base_scheduler=ConstantBetaScheduler(beta=2.0)
        ), "RCGP-Adaptive (Const base)"),
        (RCGPScheduler(
            scale=1.0,
            base_scheduler=TheoryGuidedScheduler(scale=1.0)
        ), "RCGP-Adaptive (Theory base)"),
    ]
    
    for scheduler, name in rcgp_schedulers:
        results = run_experiment_with_scheduler(
            scheduler, name, create_rcgp_model, "RCGP", seed=42
        )
        all_results['results'][f"RCGP_{name}"] = results
        
        # Extract beta values
        betas = []
        model = results['final_model']  # Use final model for adaptive scheduler
        for i in range(N_ITERATIONS):
            beta = scheduler.get_beta(i, N_ITERATIONS, model if i == N_ITERATIONS-1 else None)
            betas.append(beta)
        all_results['betas'][f"RCGP_{name}"] = betas
    
    # Create comparison plots
    print("\n" + "="*80)
    print("CREATING COMPARISON PLOTS")
    print("="*80)
    
    # Create artifacts directory
    os.makedirs("artifacts/beta_scheduling", exist_ok=True)
    
    # Plot beta progression
    fig = plot_beta_progression(all_results, "artifacts/beta_scheduling/beta_progression.png")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    optimal_value = ForresterFunction().optimal_value
    
    for name, results in all_results['results'].items():
        best_value = results['best_true_value']
        final_regret = optimal_value - best_value
        cumulative_regret = torch.sum(optimal_value - results['Y_true']).item()
        
        print(f"\n{name}:")
        print(f"  Best value found: {best_value:.6f}")
        print(f"  Final regret: {final_regret:.6f}")
        print(f"  Cumulative regret: {cumulative_regret:.2f}")
        
        # Show beta range for this scheduler
        betas = all_results['betas'][name]
        print(f"  Beta range: [{min(betas):.3f}, {max(betas):.3f}]")
    
    print("\n" + "="*80)
    print("Beta scheduling experiment complete!")
    print(f"Results saved to: artifacts/beta_scheduling/")
    print("="*80)


if __name__ == "__main__":
    main()