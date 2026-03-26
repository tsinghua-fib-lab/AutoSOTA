"""
Experimental Framework Module

This module provides the infrastructure for running simulation experiments
to evaluate different sampling methods. It includes functions for:
    - Running single trials
    - Parallel simulation experiments
    - Computing evaluation metrics (RMSE, coverage, interval width)
"""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import norm
from typing import Tuple, List, Dict, Any

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


def single_trial(
    seed: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    uncertainty: np.ndarray,
    error_pred: np.ndarray,
    inclusion_probs: np.ndarray,
    budget: float,
    theta_true: float,
    confidence_level: float = 0.95,
    tau: float = 0.5
) -> Tuple[List[float], List[float], List[int], List[float], List[float]]:
    """
    Execute a single trial comparing multiple sampling methods.
    
    This function performs one replication of the experiment, evaluating:
        1. Uniform Poisson Sampling
        2. Active Poisson Sampling
        3. Cube Active Sampling (proposed method)
        4. Classical Simple Random Sampling
    
    Args:
        seed: Random seed for this trial
        y_true: True labels of shape (N,)
        y_pred: Predicted labels of shape (N,)
        uncertainty: Uncertainty estimates of shape (N,)
        error_pred: Predicted errors of shape (N,)
        inclusion_probs: Inclusion probabilities for active sampling of shape (N,)
        budget: Sampling budget (sampling rate)
        theta_true: True population mean
        confidence_level: Confidence level for intervals (default: 0.95)
        tau: Parameter for active sampling (default: 0.5)
    
    Returns:
        bias_squared: Squared bias for each method
        interval_widths: Confidence interval widths for each method
        coverage_indicators: Coverage indicators (1 if covered) for each method
        lower_bounds: Lower confidence bounds for each method
        upper_bounds: Upper confidence bounds for each method
    """
    np.random.seed(seed)
    N = len(y_true)
    z_quantile = norm.ppf(1.0 - (1.0 - confidence_level) / 2.0)
    
    # Storage for results
    bias_squared = []
    interval_widths = []
    coverage_indicators = []
    lower_bounds = []
    upper_bounds = []
    
    # ========================================================================
    # Method 1: Uniform Poisson Sampling
    # ========================================================================
    xi_uniform, pi_uniform = uniform_poisson_sampling(N, budget, random_state=seed)
    
    # Horvitz-Thompson estimator with predictions
    est_uniform = (y_pred + (y_true - y_pred) * xi_uniform / pi_uniform).mean()
    
    # Variance estimation
    var_uniform = estimate_bernoulli_variance(y_true, y_pred, xi_uniform, pi_uniform)
    std_uniform = np.sqrt(var_uniform)
    
    # Confidence interval
    margin_uniform = z_quantile * std_uniform
    lb_uniform = est_uniform - margin_uniform
    ub_uniform = est_uniform + margin_uniform
    
    # Store results
    bias_squared.append((est_uniform - theta_true) ** 2)
    interval_widths.append(2.0 * margin_uniform)
    coverage_indicators.append(int(lb_uniform < theta_true < ub_uniform))
    lower_bounds.append(lb_uniform)
    upper_bounds.append(ub_uniform)
    
    # ========================================================================
    # Method 2: Active Poisson Sampling
    # ========================================================================
    xi_active, pi_active = active_poisson_sampling(
        uncertainty, budget, tau, random_state=seed
    )
    
    # Horvitz-Thompson estimator
    est_active = (y_pred + (y_true - y_pred) * xi_active / pi_active).mean()
    
    # Variance estimation
    var_active = estimate_bernoulli_variance(y_true, y_pred, xi_active, pi_active)
    std_active = np.sqrt(var_active)
    
    # Confidence interval
    margin_active = z_quantile * std_active
    lb_active = est_active - margin_active
    ub_active = est_active + margin_active
    
    # Store results
    bias_squared.append((est_active - theta_true) ** 2)
    interval_widths.append(2.0 * margin_active)
    coverage_indicators.append(int(lb_active < theta_true < ub_active))
    lower_bounds.append(lb_active)
    upper_bounds.append(ub_active)
    
    # ========================================================================
    # Method 3: Cube Active Sampling (Proposed)
    # ========================================================================
    xi_cube, pi_cube = cube_active_sampling(
        error_pred.reshape(-1, 1), inclusion_probs, random_state=seed
    )
    
    # Horvitz-Thompson estimator
    est_cube = (y_pred + (y_true - y_pred) * xi_cube / pi_cube).mean()
    
    # Variance estimation (using cube variance estimator)
    residuals = y_true - y_pred
    var_cube = estimate_cube_variance(
        residuals, error_pred.reshape(-1, 1), xi_cube, pi_cube, N
    )
    std_cube = np.sqrt(var_cube)
    
    # Confidence interval
    margin_cube = z_quantile * std_cube
    lb_cube = est_cube - margin_cube
    ub_cube = est_cube + margin_cube
    
    # Store results
    bias_squared.append((est_cube - theta_true) ** 2)
    interval_widths.append(2.0 * margin_cube)
    coverage_indicators.append(int(lb_cube < theta_true < ub_cube))
    lower_bounds.append(lb_cube)
    upper_bounds.append(ub_cube)
    
    # ========================================================================
    # Method 4: Classical Simple Random Sampling (Baseline)
    # ========================================================================
    xi_classical, pi_classical = classical_simple_random_sampling(
        N, budget, random_state=seed
    )
    
    # Sample mean estimator
    selected_labels = y_true[xi_classical == 1]
    
    if len(selected_labels) > 1:
        est_classical = selected_labels.mean()
        sample_size = len(selected_labels)
        sample_var = selected_labels.var(ddof=1)
        
        # Finite population correction
        fpc = 1.0 - budget
        std_classical = np.sqrt(fpc * sample_var / sample_size)
        
        # Confidence interval
        margin_classical = z_quantile * std_classical
        lb_classical = est_classical - margin_classical
        ub_classical = est_classical + margin_classical
    else:
        # Handle edge case with very small sample
        est_classical = y_true.mean()
        lb_classical = ub_classical = est_classical
        margin_classical = 0.0
    
    # Store results
    bias_squared.append((est_classical - theta_true) ** 2)
    interval_widths.append(2.0 * margin_classical)
    coverage_indicators.append(int(lb_classical < theta_true < ub_classical))
    lower_bounds.append(lb_classical)
    upper_bounds.append(ub_classical)
    
    return (
        bias_squared,
        interval_widths,
        coverage_indicators,
        lower_bounds,
        upper_bounds
    )


def run_simulation_experiment(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    uncertainty: np.ndarray,
    error_pred: np.ndarray,
    budgets: np.ndarray,
    n_trials: int = 1000,
    confidence_level: float = 0.95,
    tau: float = 0.5,
    n_jobs: int = -1
) -> Dict[str, pd.DataFrame]:
    """
    Run comprehensive simulation experiment across multiple budgets.
    
    This function evaluates all sampling methods across different sampling
    budgets, computing key performance metrics:
        - RMSE (Root Mean Squared Error)
        - Confidence interval width
        - Coverage rate
    
    Args:
        y_true: True labels of shape (N,)
        y_pred: Predicted labels of shape (N,)
        uncertainty: Uncertainty estimates of shape (N,)
        error_pred: Predicted errors of shape (N,)
        budgets: Array of sampling budgets to evaluate
        n_trials: Number of Monte Carlo replications (default: 1000)
        confidence_level: Confidence level for intervals (default: 0.95)
        tau: Parameter for active sampling (default: 0.5)
        n_jobs: Number of parallel jobs (default: -1, use all cores)
    
    Returns:
        results: Dictionary containing DataFrames for each metric:
            - 'rmse': Root mean squared error
            - 'interval_width': Average confidence interval width
            - 'coverage': Empirical coverage rate
            - 'lower_bound': Average lower confidence bound
            - 'upper_bound': Average upper confidence bound
    
    Example:
        >>> results = run_simulation_experiment(
        ...     y_test, y_pred, uncertainty, error_pred,
        ...     budgets=np.arange(0.05, 0.3, 0.05),
        ...     n_trials=1000
        ... )
    """
    N = len(y_true)
    theta_true = y_true.mean()
    
    def evaluate_budget(budget: float) -> Tuple:
        """Evaluate all methods for a specific budget."""
        # Compute inclusion probabilities for active methods
        pi = compute_sampling_probabilities(uncertainty, budget, tau)
        pe = np.full(N, budget)
        
        # Run parallel trials
        trial_results = Parallel(n_jobs=n_jobs)(
            delayed(single_trial)(
                seed, y_true, y_pred, uncertainty, error_pred,
                pi, budget, theta_true, confidence_level, tau
            )
            for seed in range(n_trials)
        )
        
        # Aggregate results
        bias_sq = np.array([r[0] for r in trial_results])
        intervals = np.array([r[1] for r in trial_results])
        coverage = np.array([r[2] for r in trial_results])
        lb = np.array([r[3] for r in trial_results])
        ub = np.array([r[4] for r in trial_results])
        
        # Compute summary statistics
        rmse = np.sqrt(np.mean(bias_sq, axis=0))
        avg_interval = np.mean(intervals, axis=0)
        coverage_rate = np.mean(coverage, axis=0)
        avg_lb = np.mean(lb, axis=0)
        avg_ub = np.mean(ub, axis=0)
        
        return (rmse, avg_interval, coverage_rate, avg_lb, avg_ub, budget, len(trial_results))
    
    # Evaluate all budgets
    print(f"Running simulation with {n_trials} trials across {len(budgets)} budgets...")
    results_list = [evaluate_budget(b) for b in budgets]
    
    # Column names for methods
    method_names = ['uniform', 'poisson_active', 'cube_active', 'classical']
    columns = method_names + ['budget', 'n_trials']
    
    # Create DataFrames for each metric
    results = {
        'rmse': pd.DataFrame(
            [list(x[0]) + [x[5], x[6]] for x in results_list],
            columns=columns
        ),
        'interval_width': pd.DataFrame(
            [list(x[1]) + [x[5], x[6]] for x in results_list],
            columns=columns
        ),
        'coverage': pd.DataFrame(
            [list(x[2]) + [x[5], x[6]] for x in results_list],
            columns=columns
        ),
        'lower_bound': pd.DataFrame(
            [list(x[3]) + [x[5], x[6]] for x in results_list],
            columns=columns
        ),
        'upper_bound': pd.DataFrame(
            [list(x[4]) + [x[5], x[6]] for x in results_list],
            columns=columns
        )
    }
    
    print("Simulation complete!")
    return results
