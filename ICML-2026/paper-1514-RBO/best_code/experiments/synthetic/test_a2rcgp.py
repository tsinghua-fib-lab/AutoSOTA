"""Test A2RCGP (Adaptive Double RCGP) on Forrester function with adversarial corruption."""

import os
import torch
import matplotlib.pyplot as plt
from bo_framework import SearchSpace, Dimension, ExperimentRunner
from bo_framework.base.acquisition import UCBAcquisition
from bo_framework.base.schedulers import ConstantBetaScheduler, TheoryGuidedScheduler
from bo_framework.synthetic.evaluators import SyntheticEvaluator
from bo_framework.wrappers.noisy import NoisyEvaluator
from bo_framework.wrappers.corrupted import CorruptedEvaluator
from experiments.synthetic.functions import ForresterFunction
from bo_framework.corruption.adversarial import AdversarialCorruptor
from utilities.plotting import plot_experiment_summary, PlotConfig
from utilities.plotting_common import plot_regret_comparison
from utilities.io import save_experiment_results, save_comparison_table
from utilities.regret_analysis import compare_experiments, print_comparison_table
from bo_framework.models.factory import create_rcgp_model, create_gp_model, create_a2rcgp_model


# Experiment parameters
N_ITERATIONS = 30
N_INITIAL = 5
ADVERSARIAL_BUDGET = 3
STANDARDIZE = True
FIT_STANDARD_GP = True

# Beta scheduler configuration
GP_SCHEDULER_TYPE = 'theory'  
RCGP_SCHEDULER_TYPE = 'theory'
A2RCGP_SCHEDULER_TYPE = 'theory'
CONSTANT_BETA = 1.0


def create_scheduler(scheduler_type):
    """Create beta scheduler based on configuration."""
    if scheduler_type == 'constant':
        return ConstantBetaScheduler(beta=CONSTANT_BETA)
    elif scheduler_type == 'theory':
        return TheoryGuidedScheduler(scale=1.0)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def main():
    """Compare A2RCGP against RCGP and standard GP."""
    
    print("\n" + "=" * 80)
    print("A2RCGP TEST EXPERIMENT")
    print("=" * 80)
    print(f"Iterations: {N_ITERATIONS}, Initial points: {N_INITIAL}")
    print(f"Adversarial budget: {ADVERSARIAL_BUDGET}")
    print("=" * 80 + "\n")

    # Forrester optimal point and value
    forrester_func = ForresterFunction()
    optimal_point = torch.tensor([1.0], dtype=torch.double)
    optimal_value = forrester_func.optimal_value
    
    # Create search space
    search_space = SearchSpace((
        Dimension(name="x0", type="continuous", bounds=(0.0, 1.0), normalize=True),
    ))

    # Create base evaluator
    clean_evaluator = SyntheticEvaluator(forrester_func)
    
    # Model configurations
    # Standard GP
    gp_model_kwargs = {
        'fit_hyperparameters': FIT_STANDARD_GP,
        'standardize': STANDARDIZE
    }
    
    # RCGP
    rcgp_kwargs = {
        "param_handling_dict": {
            "plateau_width": {"method": "heuristics", "value": 2.0},
            "c": {"method": "manual", "value": 1.0},
            "sigma": {"method": "fit"},
            "mean": {"method": "fit"}
        },
        "fitting_objective_type": "wloo-cv",
        "optimizer_type": "lbfgs",
        "verbose": False
    }
    
    # A2RCGP - Inner and Outer configurations
    a2rcgp_kwargs = {
        "inner_param_handling_dict": {
            "plateau_width": {"method": "heuristics", "value": 2.0},
            "c": {"method": "manual", "value": 1.0},
            "sigma": {"method": "fit"},
            "mean": {"method": "fit"}  # Inner RCGP uses fitted constant mean
        },
        "outer_param_handling_dict": {
            "plateau_width": {"method": "heuristics", "value": 1.5},
            "c": {"method": "manual", "value": 0.8},
            "sigma": {"method": "fit"},
            "mean": {"method": "fit"}  # Outer RCGP also uses fitted mean
        },
        "fitting_objective_type": "wloo-cv",
        "optimizer_type": "lbfgs",
        "standardize": STANDARDIZE,
        "verbose": True
    }
    
    # Define models to test
    models_to_test = [
        ("GP", create_gp_model, gp_model_kwargs, GP_SCHEDULER_TYPE),
        ("RCGP", create_rcgp_model, rcgp_kwargs, RCGP_SCHEDULER_TYPE),
        ("A2RCGP", create_a2rcgp_model, a2rcgp_kwargs, A2RCGP_SCHEDULER_TYPE)
    ]
    
    results_dict = {}
    
    for model_name, model_factory, model_kwargs, scheduler_type in models_to_test:
        print(f"\n" + "=" * 60)
        print(f"RUNNING {model_name} EXPERIMENT")
        print("=" * 60)
        
        # Create fresh evaluator for each model
        noisy_evaluator = NoisyEvaluator(clean_evaluator, noise_std=1.0, seed=42)
        corruptor = AdversarialCorruptor(
            optimal_point=optimal_point,
            budget=ADVERSARIAL_BUDGET,
            near_threshold=0.1,
            far_threshold=0.4,
            high_value=25.0,
            low_value=-10.0
        )
        corrupted_evaluator = CorruptedEvaluator(
            base_evaluator=noisy_evaluator,
            corruptor=corruptor,
            n_initial=N_INITIAL
        )
        
        runner = ExperimentRunner(search_space, corrupted_evaluator)
        scheduler = create_scheduler(scheduler_type)
        
        print(f"Using scheduler: {scheduler.__class__.__name__}")
        
        # Run experiment
        results = runner.run(
            n_iterations=N_ITERATIONS,
            n_initial=N_INITIAL,
            model_factory=model_factory,
            acquisition_factory=UCBAcquisition.create,
            model_kwargs=model_kwargs,
            beta_scheduler=scheduler,
            seed=42,
            verbose=True
        )
        
        results_dict[model_name] = results
        
        print(f"{model_name} Best value: {results['best_observed_value']:.4f}")
        print(f"{model_name} Best params: {results['best_observed_params']}")

    # Compare results
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    
    # Use regret_analysis utilities to compute comprehensive metrics
    metrics_dict = compare_experiments(
        results_dict={name: results['all_results'] for name, results in results_dict.items()},
        optimal_value=optimal_value,
    )
    
    # Print detailed comparison
    print_comparison_table(
        metrics_dict,
        show_regret=True,
        show_corruption=True
    )

    # Save results and create plots
    print("\n" + "=" * 80)
    print("SAVING RESULTS AND GENERATING PLOTS")
    print("=" * 80)
    
    # Create artifacts directory
    os.makedirs("artifacts", exist_ok=True)
    
    # Save all results
    saved_paths = save_experiment_results(
        results=results_dict,
        experiment_name="a2rcgp_test",
        artifacts_dir="artifacts",
        save_pickle=True,
        save_json=True,
        optimal_value=optimal_value,
        verbose=True
    )
    
    # Save comparison table
    save_comparison_table(
        results_dict=results_dict,
        experiment_name="a2rcgp_test",
        artifacts_dir="artifacts",
        optimal_value=optimal_value
    )
    
    # Create individual plots
    config = PlotConfig(figsize=(15, 10))
    artifacts_dir = saved_paths['directory']
    
    for name, results in results_dict.items():
        print(f"Creating plot for: {name}")
        
        plot_filename = f"a2rcgp_test_{name.lower().replace(' ', '_')}.png"
        plot_path = os.path.join(artifacts_dir, plot_filename)
        
        fig = plot_experiment_summary(
            results=results,
            objective_func=lambda x: forrester_func.evaluate(x),
            optimal_value=optimal_value,
            save_path=plot_path,
            config=config
        )
        plt.close(fig)
    
    # Create regret comparison plots
    print("\n" + "=" * 80)
    print("CREATING REGRET COMPARISON PLOTS")
    print("=" * 80)
    
    regret_save_path = os.path.join(artifacts_dir, 'regret')
    colors = {'GP': 'orange', 'RCGP': 'blue', 'A2RCGP': 'red'}
    
    regret_fig, simple_regret_fig = plot_regret_comparison(
        results_dict=results_dict,
        optimal_value=optimal_value,
        save_path=regret_save_path,
        config=config,
        colors=colors
    )
    
    plt.close(regret_fig)
    plt.close(simple_regret_fig)
    
    print(f"\nAll artifacts saved to: {artifacts_dir}/")
    
    # Print model hyperparameters for A2RCGP
    if 'A2RCGP' in results_dict:
        print("\n" + "=" * 80)
        print("A2RCGP MODEL DETAILS")
        print("=" * 80)
        
        a2rcgp_model = results_dict['A2RCGP']['final_model']
        
        print(f"\nA2RCGP Model Structure:")
        print(f"  Inner RCGP - Plateau width: {a2rcgp_model.inner_rcgp.weighting_function.plateau_width:.4f}")
        print(f"  Inner RCGP - C parameter: {a2rcgp_model.inner_rcgp.weighting_function.c:.4f}")
        print(f"  Inner RCGP - Noise std: {torch.sqrt(a2rcgp_model.inner_rcgp.likelihood.noise).item():.4f}")
        
        print(f"  Outer RCGP - Plateau width: {a2rcgp_model.weighting_function.plateau_width:.4f}")
        print(f"  Outer RCGP - C parameter: {a2rcgp_model.weighting_function.c:.4f}")
        print(f"  Outer RCGP - Noise std: {torch.sqrt(a2rcgp_model.likelihood.noise).item():.4f}")
        
        print(f"\nCorruption Detection:")
        corruption_results = a2rcgp_model.detect_corruptions()
        inner_corruptions = corruption_results['inner'].sum().item()
        outer_corruptions = corruption_results['outer'].sum().item()
        print(f"  Inner RCGP detected corruptions: {inner_corruptions}")
        print(f"  Outer RCGP detected corruptions: {outer_corruptions}")


if __name__ == "__main__":
    main()