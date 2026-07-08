"""Common plotting utilities for Bayesian Optimization experiments.

This module provides plotting functions that work for both 1D and 2D experiments.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass

# Import regret analysis utilities
from utilities.regret_analysis import (
    compute_regret,
    extract_observed_values,
    extract_corruption_levels,
    compute_corruption_statistics,
    compute_best_so_far
)


@dataclass
class PlotConfig:
    """Configuration for plot styling."""
    figsize: tuple = (10, 6)
    dpi: int = 100
    fontsize: int = 12
    linewidth: float = 2.0
    markersize: float = 8.0
    alpha_fill: float = 0.3
    grid: bool = True
    legend: bool = True


def plot_regret(
    ax: Axes,
    results: Union[Dict[str, Any], List[Dict[str, Any]]],
    optimal_value: float,
    regret_type: str = 'simple',
    log_scale: bool = False,
    show_best: bool = True,
    labels: Optional[List[str]] = None,
    config: Optional[PlotConfig] = None
) -> Axes:
    """Plot regret curves from BO results.
    
    Works for both 1D and 2D experiments.
    
    Args:
        ax: Matplotlib axes to plot on
        results: Single result dict or list of result dicts from ExperimentRunner
        optimal_value: Known optimal value of the objective
        regret_type: Type of regret to plot:
            - 'simple': Simple regret (best so far - optimal)
            - 'instantaneous': Instantaneous regret at each iteration
            - 'cumulative': Cumulative regret over time
        log_scale: Whether to use log scale for y-axis
        show_best: Whether to mark the best point
        labels: Labels for multiple result curves
        config: Plot configuration
        
    Returns:
        The axes object with the plot
    """
    if config is None:
        config = PlotConfig()
    
    # Ensure results is a list
    if not isinstance(results, list):
        results = [results]
    
    if labels is None:
        labels = [f"Run {i+1}" for i in range(len(results))]
    
    # Set up ylabel based on regret type
    ylabel_map = {
        'simple': 'Simple Regret',
        'instantaneous': 'Instantaneous Regret', 
        'cumulative': 'Cumulative Regret'
    }
    ylabel = ylabel_map.get(regret_type, f'{regret_type.capitalize()} Regret')
    
    for idx, result in enumerate(results):
        # Extract evaluation results from the experiment result dict
        eval_results = result.get('all_results')
        
        if eval_results is None:
            raise ValueError("Cannot find 'all_results' (EvaluationResult objects) in results. "
                           "Make sure the experiment returns the full EvaluationResult list.")
        
        # Use regret_analysis utility to compute regret
        regret = compute_regret(eval_results, optimal_value, regret_type)
        
        # Create iteration numbers
        n_iters = len(regret)
        iterations = np.arange(1, n_iters + 1)
        
        # Plot regret curve
        ax.plot(iterations, regret, linewidth=config.linewidth, 
                label=labels[idx], marker='o', markersize=config.markersize/2)
        
        # Mark best point for simple regret
        if show_best and regret_type == 'simple':
            best_iter = np.argmin(regret) + 1
            ax.scatter([best_iter], [regret[best_iter-1]], 
                      s=(config.markersize*1.5)**2, marker='*', 
                      color='red', zorder=5)
    
    ax.set_xlabel('Iteration', fontsize=config.fontsize)
    ax.set_ylabel(ylabel, fontsize=config.fontsize)
    
    if log_scale:
        ax.set_yscale('log')
        ax.set_ylabel(f'{ylabel} (log scale)', fontsize=config.fontsize)
    
    # Add horizontal line at y=0 for reference
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=1)
    
    if config.grid:
        ax.grid(True, alpha=0.3)
    
    if config.legend and len(results) > 1:
        ax.legend(fontsize=config.fontsize-2)
    
    return ax


def plot_best_value_progression(
    ax: Axes,
    results: Dict[str, Any],
    optimal_value: Optional[float] = None,
    config: Optional[PlotConfig] = None
) -> Axes:
    """Plot the progression of best value found over iterations.
    
    Works for both 1D and 2D experiments.
    
    Args:
        ax: Matplotlib axes to plot on
        results: Results dictionary from ExperimentRunner
        optimal_value: Optional known optimal value to show as reference
        config: Plot configuration
        
    Returns:
        The axes object with the plot
    """
    if config is None:
        config = PlotConfig()
    
    # Extract evaluation results from the experiment result dict
    eval_results = results.get('all_results')
    
    if eval_results is None:
        raise ValueError("Cannot find 'all_results' (EvaluationResult objects) in results. "
                       "Make sure the experiment returns the full EvaluationResult list.")
    
    # Calculate best so far using regret_analysis utility
    best_so_far = compute_best_so_far(eval_results)
    
    # Plot progression
    iterations = range(1, len(best_so_far) + 1)
    ax.plot(iterations, best_so_far.numpy(),
            'b-', linewidth=config.linewidth, 
            marker='o', markersize=config.markersize/2,
            label='Best found')
    
    # Add optimal value reference if known
    if optimal_value is not None:
        ax.axhline(y=optimal_value, color='r', linestyle='--', 
                  alpha=0.5, label=f'Optimal: {optimal_value:.3f}')
    
    ax.set_xlabel('Iteration', fontsize=config.fontsize)
    ax.set_ylabel('Best Value Found', fontsize=config.fontsize)
    ax.set_title('Best Value Progression', fontsize=config.fontsize+2)
    
    if config.grid:
        ax.grid(True, alpha=0.3)
    
    if config.legend:
        ax.legend(fontsize=config.fontsize-2)
    
    return ax


def plot_convergence_analysis(
    ax: Axes,
    results: Union[Dict[str, Any], List[Dict[str, Any]]],
    optimal_value: Optional[float] = None,
    labels: Optional[List[str]] = None,
    config: Optional[PlotConfig] = None
) -> Axes:
    """Plot convergence analysis showing distance to optimum over time.
    
    Works for both 1D and 2D experiments.
    
    Args:
        ax: Matplotlib axes to plot on
        results: Single result dict or list of result dicts
        optimal_value: Known optimal value
        labels: Labels for multiple curves
        config: Plot configuration
        
    Returns:
        The axes object with the plot
    """
    if config is None:
        config = PlotConfig()
    
    if optimal_value is None:
        ax.text(0.5, 0.5, 'Optimal value not provided',
                ha='center', va='center', transform=ax.transAxes)
        return ax
    
    # Ensure results is a list
    if not isinstance(results, list):
        results = [results]
    
    if labels is None:
        labels = [f"Run {i+1}" for i in range(len(results))]
    
    for idx, result in enumerate(results):
        # Extract evaluation results from the experiment result dict
        eval_results = result.get('all_results')
        
        if eval_results is None:
            raise ValueError("Cannot find 'all_results' (EvaluationResult objects) in results. "
                           "Make sure the experiment returns the full EvaluationResult list.")
        
        # Calculate best so far using regret_analysis utility
        best_so_far = compute_best_so_far(eval_results).numpy()
        
        # Calculate distance to optimum (in function space)
        distance_to_opt = np.abs(optimal_value - best_so_far)
        
        iterations = range(1, len(distance_to_opt) + 1)
        ax.semilogy(iterations, distance_to_opt + 1e-10,  # Add small constant for log scale
                   linewidth=config.linewidth, 
                   label=labels[idx], 
                   marker='o', 
                   markersize=config.markersize/2)
    
    ax.set_xlabel('Iteration', fontsize=config.fontsize)
    ax.set_ylabel('Distance to Optimum (log scale)', fontsize=config.fontsize)
    ax.set_title('Convergence Analysis', fontsize=config.fontsize+2)
    
    if config.grid:
        ax.grid(True, alpha=0.3, which='both')
    
    if config.legend and len(results) > 1:
        ax.legend(fontsize=config.fontsize-2)
    
    return ax


def plot_corruption_analysis(
    ax: Axes,
    results: Dict[str, Any],
    config: Optional[PlotConfig] = None
) -> Axes:
    """Plot analysis of corruption in the experiment.
    
    Works for both 1D and 2D experiments.
    
    Args:
        ax: Matplotlib axes to plot on
        results: Results dictionary from ExperimentRunner
        config: Plot configuration
        
    Returns:
        The axes object with the plot
    """
    if config is None:
        config = PlotConfig()
    
    # Extract evaluation results from the experiment result dict
    eval_results = results.get('all_results')
    
    if eval_results is None:
        ax.text(0.5, 0.5, 'No evaluation results available',
                ha='center', va='center', transform=ax.transAxes)
        return ax
    
    # Extract corruption levels using regret_analysis utility
    corruption = extract_corruption_levels(eval_results)
    
    if len(corruption) == 0:
        ax.text(0.5, 0.5, 'No corruption data available',
                ha='center', va='center', transform=ax.transAxes)
        return ax
    
    iterations = range(1, len(corruption) + 1)
    
    # Plot corruption levels
    ax.bar(iterations, corruption.numpy(), 
           color=['red' if c != 0 else 'blue' for c in corruption],
           alpha=0.6)
    
    # Use regret_analysis utility for statistics
    corruption_stats = compute_corruption_statistics(eval_results)
    
    ax.set_xlabel('Iteration', fontsize=config.fontsize)
    ax.set_ylabel('Corruption Level', fontsize=config.fontsize)
    ax.set_title(f'Corruption Analysis (Total: {corruption_stats["n_corrupted"]} points, '
                f'Magnitude: {corruption_stats["total_corruption"]:.1f})',
                fontsize=config.fontsize+2)
    
    if config.grid:
        ax.grid(True, alpha=0.3, axis='y')
    
    return ax


def create_figure_with_subplots(
    n_rows: int = 1,
    n_cols: int = 1,
    figsize: Optional[tuple] = None,
    **kwargs
) -> tuple:
    """Create a figure with subplots configured for our plotting functions.
    
    Args:
        n_rows: Number of subplot rows
        n_cols: Number of subplot columns
        figsize: Figure size (width, height)
        **kwargs: Additional arguments for plt.subplots
        
    Returns:
        Tuple of (figure, axes) where axes is single Axes or array of Axes
    """
    if figsize is None:
        figsize = (6 * n_cols, 4 * n_rows)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, **kwargs)
    
    # Ensure consistent spacing
    fig.tight_layout()
    
    return fig, axes


def plot_regret_comparison(
    results_dict: Dict[str, Dict[str, Any]],
    optimal_value: float,
    save_path: Optional[str] = None,
    config: Optional[PlotConfig] = None,
    colors: Optional[Dict[str, str]] = None
) -> Tuple[Figure, Figure]:
    """Create comprehensive regret comparison plots for multiple models.
    
    Creates two figures:
    1. Side-by-side instantaneous and cumulative regret comparison
    2. Simple regret (best so far) comparison
    
    Args:
        results_dict: Dictionary mapping model names to their results
            e.g., {'RCGP': rcgp_results, 'GP': gp_results}
        optimal_value: Known optimal value of the objective
        save_path: Optional base path to save plots (will append suffixes)
        config: Plot configuration
        colors: Optional dictionary mapping model names to colors
        
    Returns:
        Tuple of (regret_comparison_fig, simple_regret_fig)
    """
    if config is None:
        config = PlotConfig()
    
    # Default colors if not provided
    if colors is None:
        default_colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
        colors = {name: default_colors[i % len(default_colors)] 
                 for i, name in enumerate(results_dict.keys())}
    
    # Create figure with instantaneous and cumulative regret
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot instantaneous regret
    for name, results in results_dict.items():
        instantaneous_regret = compute_regret(
            results['all_results'], 
            optimal_value, 
            regret_type='instantaneous'
        )
        iterations = np.arange(1, len(instantaneous_regret) + 1)
        ax1.plot(iterations, instantaneous_regret, 
                label=name, color=colors.get(name), 
                linewidth=config.linewidth)
    
    ax1.set_xlabel('Iteration', fontsize=config.fontsize)
    ax1.set_ylabel('Instantaneous Regret', fontsize=config.fontsize)
    ax1.set_title('Instantaneous Regret Comparison', 
                 fontsize=config.fontsize + 2, fontweight='bold')
    ax1.legend(fontsize=config.fontsize - 1)
    ax1.grid(config.grid, alpha=0.3)
    ax1.set_xlim(1, iterations[-1] if len(iterations) > 0 else 1)
    
    # Plot cumulative regret
    for name, results in results_dict.items():
        cumulative_regret = compute_regret(
            results['all_results'], 
            optimal_value, 
            regret_type='cumulative'
        )
        iterations = np.arange(1, len(cumulative_regret) + 1)
        ax2.plot(iterations, cumulative_regret, 
                label=name, color=colors.get(name), 
                linewidth=config.linewidth)
    
    ax2.set_xlabel('Iteration', fontsize=config.fontsize)
    ax2.set_ylabel('Cumulative Regret', fontsize=config.fontsize)
    ax2.set_title('Cumulative Regret Comparison', 
                 fontsize=config.fontsize + 2, fontweight='bold')
    ax2.legend(fontsize=config.fontsize - 1)
    ax2.grid(config.grid, alpha=0.3)
    ax2.set_xlim(1, iterations[-1] if len(iterations) > 0 else 1)
    
    model_names = ' vs '.join(results_dict.keys())
    plt.suptitle(f'Regret Analysis: {model_names}', 
                fontsize=config.fontsize + 4, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        import os
        base_dir = os.path.dirname(save_path) if os.path.dirname(save_path) else '.'
        base_name = os.path.splitext(os.path.basename(save_path))[0] if save_path else 'regret'
        
        regret_path = os.path.join(base_dir, f'{base_name}_comparison.png')
        fig1.savefig(regret_path, dpi=300, bbox_inches='tight')
        print(f"Regret comparison plot saved to: {regret_path}")
    
    # Create simple regret plot
    fig2, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    for name, results in results_dict.items():
        simple_regret = compute_regret(
            results['all_results'], 
            optimal_value, 
            regret_type='simple'
        )
        iterations = np.arange(1, len(simple_regret) + 1)
        ax.plot(iterations, simple_regret, 
               label=name, color=colors.get(name), 
               linewidth=config.linewidth)
    
    ax.set_xlabel('Iteration', fontsize=config.fontsize)
    ax.set_ylabel('Simple Regret', fontsize=config.fontsize)
    ax.set_title('Simple Regret (Best So Far) Comparison', 
                fontsize=config.fontsize + 2, fontweight='bold')
    ax.legend(fontsize=config.fontsize - 1)
    ax.grid(config.grid, alpha=0.3)
    ax.set_xlim(1, iterations[-1] if len(iterations) > 0 else 1)
    ax.set_ylim(bottom=0)
    
    # Save if path provided
    if save_path:
        simple_path = os.path.join(base_dir, f'{base_name}_simple.png')
        fig2.savefig(simple_path, dpi=300, bbox_inches='tight')
        print(f"Simple regret plot saved to: {simple_path}")
    
    return fig1, fig2