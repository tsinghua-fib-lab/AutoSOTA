"""Plotting utilities for 1D Bayesian Optimization experiments.

This module provides composable plotting functions that work with both 
standalone figures and subfigures.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing import Dict, Any, Optional, List, Tuple, Union, Callable

# Import common plotting utilities
from utilities.plotting_common import (
    PlotConfig,
    plot_regret,
    plot_best_value_progression,
    plot_convergence_analysis,
    plot_corruption_analysis,
    create_figure_with_subplots
)


def plot_objective_1d(
    ax: Axes,
    objective_func: Optional[Callable] = None,
    bounds: Tuple[float, float] = (0.0, 1.0),
    n_points: int = 200,
    observations: Optional[Dict[str, torch.Tensor]] = None,
    show_corruption: bool = True,
    label: str = "True function",
    config: Optional[PlotConfig] = None
) -> Axes:
    """Plot 1D objective function with optional observations.
    
    Args:
        ax: Matplotlib axes to plot on
        objective_func: Callable that takes tensor and returns tensor
        bounds: Domain bounds (min, max)
        n_points: Number of points for function evaluation
        observations: Dict with keys:
            - 'X': Input locations (n x 1 tensor)
            - 'Y_true': True values (n tensor)
            - 'Y_observed': Observed values (n tensor)
            - 'Y_noisy': Optional noisy values (n tensor)
            - 'corruption_levels': Optional corruption amounts (n tensor)
        show_corruption: Whether to highlight corrupted points
        label: Label for the true function
        config: Plot configuration
        
    Returns:
        The axes object with the plot
    """
    if config is None:
        config = PlotConfig()
    
    # Plot true function if provided
    if objective_func is not None:
        x_plot = torch.linspace(bounds[0], bounds[1], n_points).unsqueeze(-1)
        y_plot = torch.zeros(n_points)
        for i, x in enumerate(x_plot):
            y_plot[i] = objective_func(x)
        
        ax.plot(x_plot.squeeze(), y_plot, 
                'k-', linewidth=config.linewidth, label=label, alpha=0.7)
    
    # Plot observations if provided
    if observations is not None:
        X = observations['X'].squeeze()
        
        # Plot true values
        if 'Y_true' in observations:
            ax.scatter(X, observations['Y_true'], 
                      c='green', s=config.markersize**2, 
                      marker='o', label='True values', zorder=3, alpha=0.6)
        
        # Plot observed values (what BO sees)
        if 'Y_observed' in observations:
            ax.scatter(X, observations['Y_observed'], 
                      c='blue', s=config.markersize**2, 
                      marker='x', label='Observed values', zorder=4)
            
            # Connect true and observed with lines for corrupted points
            if show_corruption and 'corruption_levels' in observations:
                corruption = observations['corruption_levels']
                corrupted_mask = (corruption != 0)
                
                if corrupted_mask.any():
                    for i in range(len(X)):
                        if corrupted_mask[i]:
                            ax.plot([X[i], X[i]], 
                                   [observations['Y_true'][i], observations['Y_observed'][i]],
                                   'r--', alpha=0.5, linewidth=1)
                    
                    # Highlight corrupted points
                    ax.scatter(X[corrupted_mask], observations['Y_observed'][corrupted_mask],
                              c='red', s=(config.markersize*1.5)**2, 
                              marker='D', label='Corrupted', zorder=5)
    
    ax.set_xlabel('x', fontsize=config.fontsize)
    ax.set_ylabel('f(x)', fontsize=config.fontsize)
    ax.set_xlim(bounds)
    
    if config.grid:
        ax.grid(True, alpha=0.3)
    
    if config.legend:
        ax.legend(fontsize=config.fontsize-2)
    
    return ax


def plot_posterior_1d(
    ax: Axes,
    model,
    bounds: Tuple[float, float] = (0.0, 1.0),
    n_points: int = 200,
    n_std: float = 2.0,
    observations: Optional[Dict[str, torch.Tensor]] = None,
    acquisition_func: Optional[Callable] = None,
    next_point: Optional[torch.Tensor] = None,
    show_samples: bool = False,
    n_samples: int = 5,
    config: Optional[PlotConfig] = None
) -> Axes:
    """Plot GP posterior mean and confidence intervals.
    
    Args:
        ax: Matplotlib axes to plot on
        model: GP model with posterior() method
        bounds: Domain bounds (min, max)
        n_points: Number of points for posterior evaluation
        n_std: Number of standard deviations for confidence interval
        observations: Optional dict with 'X' and 'Y_observed' for training points
        acquisition_func: Optional acquisition function to plot (on secondary axis)
        next_point: Optional next point selected by acquisition
        show_samples: Whether to show samples from posterior
        n_samples: Number of posterior samples to show
        config: Plot configuration
        
    Returns:
        The axes object with the plot
    """
    if config is None:
        config = PlotConfig()
    
    # Prepare test points
    x_plot = torch.linspace(bounds[0], bounds[1], n_points).unsqueeze(-1).double()
    
    # Get posterior
    model.eval()
    with torch.no_grad():
        posterior = model.posterior(x_plot)
        mean = posterior.mean.squeeze()
        variance = posterior.variance.squeeze()
        std = variance.sqrt()
    
    x_plot_np = x_plot.squeeze().numpy()
    mean_np = mean.numpy()
    std_np = std.numpy()
    
    # Plot posterior mean
    ax.plot(x_plot_np, mean_np, 'b-', linewidth=config.linewidth, label='Posterior mean')
    
    # Plot confidence intervals
    ax.fill_between(x_plot_np,
                    mean_np - n_std * std_np,
                    mean_np + n_std * std_np,
                    alpha=config.alpha_fill, color='blue',
                    label=f'±{n_std}σ confidence')
    
    # Plot posterior samples
    if show_samples:
        samples = posterior.sample(sample_shape=torch.Size([n_samples])).squeeze()
        for i, sample in enumerate(samples):
            ax.plot(x_plot_np, sample.numpy(), 'b-', alpha=0.2, linewidth=1)
    
    # Plot training observations
    if observations is not None:
        X_train = observations['X'].squeeze().numpy()
        Y_train = observations['Y_observed'].squeeze().numpy()
        ax.scatter(X_train, Y_train, c='black', s=config.markersize**2, 
                  marker='o', label='Observations', zorder=5)
    
    # Plot acquisition function on secondary axis
    if acquisition_func is not None:
        ax2 = ax.twinx()
        with torch.no_grad():
            acq_values = acquisition_func(x_plot).squeeze()
        ax2.plot(x_plot_np, acq_values.numpy(), 'g--', 
                linewidth=config.linewidth-0.5, alpha=0.7, label='Acquisition')
        ax2.set_ylabel('Acquisition value', fontsize=config.fontsize, color='g')
        ax2.tick_params(axis='y', labelcolor='g')
        
        # Mark next point
        if next_point is not None:
            next_x = next_point.squeeze().item()
            next_acq = acquisition_func(next_point.unsqueeze(0)).item()
            ax2.scatter([next_x], [next_acq], c='red', s=(config.markersize*1.5)**2,
                       marker='*', label='Next point', zorder=6)
    
    ax.set_xlabel('x', fontsize=config.fontsize)
    ax.set_ylabel('f(x)', fontsize=config.fontsize)
    ax.set_xlim(bounds)
    
    if config.grid:
        ax.grid(True, alpha=0.3)
    
    if config.legend:
        ax.legend(loc='upper left', fontsize=config.fontsize-2)
        if acquisition_func is not None:
            ax2.legend(loc='upper right', fontsize=config.fontsize-2)
    
    return ax




def plot_bo_iteration_1d(
    ax: Axes,
    iteration: int,
    model,
    search_space,
    observations: Dict[str, torch.Tensor],
    objective_func: Optional[Callable] = None,
    acquisition_func: Optional[Callable] = None,
    next_point: Optional[torch.Tensor] = None,
    config: Optional[PlotConfig] = None
) -> Axes:
    """Plot a single BO iteration showing objective, posterior, and acquisition.
    
    This is a convenience function that combines objective and posterior plots.
    
    Args:
        ax: Matplotlib axes to plot on
        iteration: Current iteration number
        model: GP model
        search_space: Search space object
        observations: Current observations dict
        objective_func: True objective function
        acquisition_func: Acquisition function
        next_point: Next selected point
        config: Plot configuration
        
    Returns:
        The axes object with the plot
    """
    if config is None:
        config = PlotConfig()
    
    bounds = (search_space.bounds[0, 0].item(), search_space.bounds[1, 0].item())
    
    # Plot objective function and observations
    plot_objective_1d(ax, objective_func, bounds, 
                     observations=observations, 
                     show_corruption=True,
                     config=config)
    
    # Overlay posterior (without observations since already plotted)
    plot_posterior_1d(ax, model, bounds,
                     acquisition_func=acquisition_func,
                     next_point=next_point,
                     config=PlotConfig(legend=False, grid=False))  # Avoid duplicate grid/legend
    
    ax.set_title(f'BO Iteration {iteration}', fontsize=config.fontsize+2)
    
    return ax




def plot_experiment_summary(
    results: Dict[str, Any],
    objective_func: Optional[Callable] = None,
    optimal_value: Optional[float] = None,
    save_path: Optional[str] = None,
    config: Optional[PlotConfig] = None
) -> Figure:
    """Create a comprehensive summary plot for a 1D BO experiment.
    
    Args:
        results: Results dictionary from ExperimentRunner
        objective_func: True objective function
        optimal_value: Known optimal value
        save_path: Optional path to save the figure
        config: Plot configuration
        
    Returns:
        The matplotlib figure
    """
    if config is None:
        config = PlotConfig(figsize=(15, 10))
    
    # Create 2x2 subplot grid
    fig, axes = create_figure_with_subplots(2, 2, figsize=config.figsize)
    
    # Prepare observations dict
    observations = {
        'X': results['X'],
        'Y_true': results.get('Y_true', results['Y_observed']),
        'Y_observed': results['Y_observed'],
        'corruption_levels': results.get('corruption_levels', torch.zeros_like(results['Y_observed']))
    }
    
    # Get bounds from first dimension
    bounds = (results['X'].min().item(), results['X'].max().item())
    
    # 1. Objective function with all observations
    plot_objective_1d(axes[0, 0], objective_func, bounds, 
                     observations=observations, config=config)
    axes[0, 0].set_title('Objective Function and Observations', fontsize=config.fontsize+2)
    
    # 2. Final posterior
    if 'final_model' in results:
        plot_posterior_1d(axes[0, 1], results['final_model'], bounds,
                         observations=observations, config=config)
        axes[0, 1].set_title('Final Posterior Distribution', fontsize=config.fontsize+2)
    else:
        axes[0, 1].text(0.5, 0.5, 'No model available',
                       ha='center', va='center', transform=axes[0, 1].transAxes)
    
    # 3. Instantaneous regret (using common function)
    if optimal_value is not None:
        plot_regret(axes[1, 0], results, optimal_value, 
                   regret_type='instantaneous', config=config)
        axes[1, 0].set_title('Instantaneous Regret', fontsize=config.fontsize+2)
    else:
        axes[1, 0].text(0.5, 0.5, 'Optimal value unknown',
                       ha='center', va='center', transform=axes[1, 0].transAxes)
    
    # 4. Cumulative regret (using common function)
    if optimal_value is not None:
        plot_regret(axes[1, 1], results, optimal_value, 
                   regret_type='cumulative', config=config)
        axes[1, 1].set_title('Cumulative Regret', fontsize=config.fontsize+2)
    else:
        axes[1, 1].text(0.5, 0.5, 'Optimal value unknown',
                       ha='center', va='center', transform=axes[1, 1].transAxes)
    
    fig.suptitle('Bayesian Optimization Experiment Summary', fontsize=config.fontsize+4)
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=config.dpi, bbox_inches='tight')
    
    return fig