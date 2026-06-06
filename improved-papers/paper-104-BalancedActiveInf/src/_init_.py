"""
Balanced Active Inference Package

This package implements the methods described in the paper
"Balanced Active Inference" (NeurIPS 2024).

Main modules:
    - data_generation: Synthetic data generation
    - models: Predictive and uncertainty modeling
    - sampling_methods: Various sampling strategies
    - variance_estimators: Variance estimation for different designs
    - experiment: Experimental framework for evaluation
    - visualization: Plotting and result visualization
"""

__version__ = "1.0.0"
__author__ = "Your Name"

# Import key functions for convenient access
from .data_generation import generate_friedman_data, split_data
from .models import ActiveInferenceModels, train_predictive_models
from .sampling_methods import (
    uniform_poisson_sampling,
    active_poisson_sampling,
    cube_active_sampling,
    classical_simple_random_sampling,
    compute_sampling_probabilities
)
from .variance_estimators import (
    estimate_bernoulli_variance,
    estimate_cube_variance,
    compute_confidence_interval
)
from .experiment import run_simulation_experiment, single_trial
from .visualization import (
    plot_comparison_results,
    plot_single_metric,
    plot_uncertainty_distribution,
    plot_sampling_allocation,
    save_results_to_csv
)

__all__ = [
    # Data generation
    'generate_friedman_data',
    'split_data',
    
    # Models
    'ActiveInferenceModels',
    'train_predictive_models',
    
    # Sampling methods
    'uniform_poisson_sampling',
    'active_poisson_sampling',
    'cube_active_sampling',
    'classical_simple_random_sampling',
    'compute_sampling_probabilities',
    
    # Variance estimators
    'estimate_bernoulli_variance',
    'estimate_cube_variance',
    'compute_confidence_interval',
    
    # Experiment
    'run_simulation_experiment',
    'single_trial',
    
    # Visualization
    'plot_comparison_results',
    'plot_single_metric',
    'plot_uncertainty_distribution',
    'plot_sampling_allocation',
    'save_results_to_csv',
]
