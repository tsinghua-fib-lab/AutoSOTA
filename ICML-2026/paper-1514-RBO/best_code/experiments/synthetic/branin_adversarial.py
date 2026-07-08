"""
Compare RCGP and standard GP on the 2D Branin function with both
observation noise and adversarial corruption.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from bo_framework import SearchSpace, Dimension, ExperimentRunner
from bo_framework.base.acquisition import UCBAcquisition
from bo_framework.synthetic.evaluators import SyntheticEvaluator
from bo_framework.wrappers.noisy import NoisyEvaluator
from bo_framework.wrappers.corrupted import CorruptedEvaluator
from experiments.synthetic.functions import BraninFunction
from bo_framework.corruption.adversarial import AdversarialCorruptor
from utilities.plotting_2d import plot_experiment_summary_2d, PlotConfig2D
from utilities.plotting_common import plot_regret_comparison
from utilities.io import save_experiment_results, save_comparison_table
from utilities.regret_analysis import compare_experiments, print_comparison_table
from bo_framework.models.factory import create_rcgp_model, create_gp_model


N_ITERATIONS = 100
N_INITIAL = 8
ADVERSARIAL_BUDGET = 4 # Following the theory from the paper, ideally choose n_corruptions <= T^{1/3}
STANDARDIZE = True
FIT_STANDARD_GP = True
CUSTOM_GP_MODEL = False


rcgp_kwargs = {
    "param_handling_dict": {
        "plateau_width": {"method": "heuristics", "value": 2.0},
        "c": {"method": "manual", "value": 1.0},
        "sigma": {"method": "fit"},
        "mean": {"method": "fit"}
    },
    "fitting_objective_type": "wloo-cv", # options 'mll', 'loo-cv' or 'wloo-cv'
    "optimizer_type": "lbfgs",
    "verbose": False,
    "standardize": STANDARDIZE
}

def main():
    """Compare RCGP against standard GP for Branin function with noise and adversarial corruption."""

    # Branin function setup
    branin_func = BraninFunction()
    
    # Branin has 3 global optima (in original space)
    optimal_points_original = torch.tensor([
        [-np.pi, 12.275],
        [np.pi, 2.275],
        [9.42478, 2.475]
    ], dtype=torch.double)
    
    optimal_value = branin_func.optimal_value

    # Create 2D search space
    search_space = SearchSpace((
        Dimension(name="x1", type="continuous", bounds=(-5.0, 10.0), normalize=True),
        Dimension(name="x2", type="continuous", bounds=(0.0, 15.0), normalize=True),
    ))

    # Create base evaluators
    clean_evaluator = SyntheticEvaluator(branin_func)

    # Normalize all optimal points for the corruptor
    optimal_points_normalized = []
    for opt_pt in optimal_points_original:
        x1_norm = (opt_pt[0] - (-5.0)) / (10.0 - (-5.0))
        x2_norm = (opt_pt[1] - 0.0) / 15.0
        optimal_points_normalized.append(torch.tensor([x1_norm, x2_norm], dtype=torch.double))

    # --- RCGP Experiment ---
    rcgp_noisy_evaluator = NoisyEvaluator(clean_evaluator, noise_std=1.0, seed=42)
    rcgp_corruptor = AdversarialCorruptor(
        optimal_points=optimal_points_normalized, 
        budget=ADVERSARIAL_BUDGET, 
        near_threshold=0.2,
        far_threshold=1.0, 
        high_value=10.0, 
        low_value=-20.0
    )
    rcgp_corrupted_evaluator = CorruptedEvaluator(
        base_evaluator=rcgp_noisy_evaluator, 
        corruptor=rcgp_corruptor, 
        n_initial=N_INITIAL
    )
    
    # create runner
    rcgp_runner = ExperimentRunner(search_space, rcgp_corrupted_evaluator)
    
    rcgp_results = rcgp_runner.run(
        n_iterations=N_ITERATIONS,
        n_initial=N_INITIAL,
        model_factory=create_rcgp_model,
        acquisition_factory=UCBAcquisition.create,
        model_kwargs=rcgp_kwargs,
        seed=42,
        verbose=True
    )

    # --- Standard GP Experiment ---
    gp_noisy_evaluator = NoisyEvaluator(clean_evaluator, noise_std=1.0, seed=42)
    gp_corruptor = AdversarialCorruptor(
        optimal_points=optimal_points_normalized, 
        budget=ADVERSARIAL_BUDGET, 
        near_threshold=0.2,
        far_threshold=1.0, 
        high_value=10.0, 
        low_value=-20.0
    )
    gp_corrupted_evaluator = CorruptedEvaluator(
        base_evaluator=gp_noisy_evaluator, 
        corruptor=gp_corruptor, 
        n_initial=N_INITIAL
    )
    gp_runner = ExperimentRunner(search_space, gp_corrupted_evaluator)
    gp_model_kwargs = {
        'fit_hyperparameters': FIT_STANDARD_GP,
        'standardize': STANDARDIZE,
        'use_botorch_model': not CUSTOM_GP_MODEL
    }
    # run experiment
    gp_results = gp_runner.run(
        n_iterations=N_ITERATIONS,
        n_initial=N_INITIAL,
        model_factory=create_gp_model,
        acquisition_factory=UCBAcquisition.create,
        model_kwargs=gp_model_kwargs,
        seed=42,
        verbose=True
    )

    # --- Analysis and Plotting ---
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    
    scenarios = [("RCGP", rcgp_results), ("GP", gp_results)]
    results_dict = {name: results for name, results in scenarios}
    
    metrics_dict = compare_experiments(
        results_dict={name: results['all_results'] for name, results in scenarios},
        optimal_value=optimal_value,
    )
    print_comparison_table(metrics_dict, show_regret=True, show_corruption=True)

    print("\n" + "=" * 80)
    print("SAVING RESULTS AND GENERATING PLOTS")
    print("=" * 80)
    
    saved_paths = save_experiment_results(
        results=results_dict, experiment_name="branin_adversarial_experiment",
        artifacts_dir="artifacts", save_pickle=True, save_json=True,
        optimal_value=optimal_value, verbose=True
    )
    save_comparison_table(
        results_dict=results_dict, experiment_name="branin_adversarial_experiment",
        artifacts_dir="artifacts", optimal_value=optimal_value
    )
    
    config = PlotConfig2D(figsize=(16, 12))
    artifacts_dir = saved_paths['directory']
    bounds = ((-5.0, 10.0), (0.0, 15.0))
    
    for name, results in scenarios:
        print(f"Creating 2D plot for: {name}")
        plot_filename = f"branin_adversarial_{name.lower().replace(' ', '_')}.png"
        plot_path = os.path.join(artifacts_dir, plot_filename)
        fig = plot_experiment_summary_2d(
            results=results, objective_func=lambda x: branin_func.evaluate(x),
            optimal_value=optimal_value, optimal_points=optimal_points_original,
            bounds=bounds, save_path=plot_path, config=config
        )
        plt.close(fig)
    
    print(f"\nAll artifacts saved to: {artifacts_dir}/")
    
    # Create regret comparison plots
    print("\n" + "=" * 80)
    print("CREATING REGRET COMPARISON PLOTS")
    print("=" * 80)
    
    regret_save_path = os.path.join(artifacts_dir, 'regret')
    colors = {'RCGP': 'blue', 'GP': 'orange'}
    
    regret_fig, simple_regret_fig = plot_regret_comparison(
        results_dict=results_dict,
        optimal_value=optimal_value,
        save_path=regret_save_path,
        config=config,
        colors=colors
    )
    
    # Close figures to free memory
    plt.close(regret_fig)
    plt.close(simple_regret_fig)
    
    # Print model hyperparameters
    print("\n" + "=" * 80)
    print("MODEL HYPERPARAMETERS")
    print("=" * 80)
    
    # RCGP model parameters
    rcgp_model = rcgp_results['final_model']
    print("\nRCGP Model:")
    print(f"  Noise std (sigma): {torch.sqrt(rcgp_model.likelihood.noise).item():.4f}")
    
    # Handle different covariance module structures
    covar = rcgp_model.covar_module
    if hasattr(covar, 'base_kernel'):
        # ScaleKernel case
        print(f"  Lengthscale: {covar.base_kernel.lengthscale.squeeze().tolist()}")
        print(f"  Output scale: {covar.outputscale.item():.4f}")
    else:
        # Direct kernel case
        if hasattr(covar, 'lengthscale'):
            print(f"  Lengthscale: {covar.lengthscale.squeeze().tolist()}")
        if hasattr(covar, 'outputscale'):
            print(f"  Output scale: {covar.outputscale.item():.4f}")
    
    if hasattr(rcgp_model.mean_module, 'constant'):
        print(f"  Mean constant: {rcgp_model.mean_module.constant.item():.4f}")
    print(f"  Plateau width: {rcgp_model.weighting_function.plateau_width:.4f}")
    print(f"  C parameter: {rcgp_model.weighting_function.c:.4f}")
    
    # GP model parameters
    gp_model = gp_results['final_model']
    print("\nGP Model:")
    print(f"  Noise std (sigma): {torch.sqrt(gp_model.likelihood.noise).item():.4f}")
    
    # Handle different covariance module structures
    covar = gp_model.covar_module
    if hasattr(covar, 'base_kernel'):
        # ScaleKernel case
        print(f"  Lengthscale: {covar.base_kernel.lengthscale.squeeze().tolist()}")
        print(f"  Output scale: {covar.outputscale.item():.4f}")
    else:
        # Direct kernel case
        if hasattr(covar, 'lengthscale'):
            print(f"  Lengthscale: {covar.lengthscale.squeeze().tolist()}")
        if hasattr(covar, 'outputscale'):
            print(f"  Output scale: {covar.outputscale.item():.4f}")
    
    if hasattr(gp_model.mean_module, 'constant'):
        print(f"  Mean constant: {gp_model.mean_module.constant.item():.4f}")

if __name__ == "__main__":
    main()
