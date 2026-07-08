"""Plotting utilities for 2D Bayesian Optimization experiments.

This module provides plotting functions for 2D objective functions and BO results.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from dataclasses import dataclass

# Import common plotting utilities
from utilities.plotting_common import (
    PlotConfig,
    plot_regret,
    plot_best_value_progression,
    plot_convergence_analysis,
    plot_corruption_analysis,
    create_figure_with_subplots
)


@dataclass
class PlotConfig2D(PlotConfig):
    """Configuration for 2D plot styling."""
    figsize: Tuple[float, float] = (12, 10)
    n_contour_levels: int = 20
    colormap: str = 'viridis'
    surface_alpha: float = 0.7


def plot_objective_2d_heatmap(
    ax: Axes,
    objective_func: Callable,
    bounds: Tuple[Tuple[float, float], Tuple[float, float]],
    n_points: int = 100,
    observations: Optional[Dict[str, torch.Tensor]] = None,
    show_corruption: bool = True,
    show_colorbar: bool = True,
    config: Optional[PlotConfig2D] = None
) -> Axes:
    """Plot 2D objective function as heatmap with optional observations.
    
    Args:
        ax: Matplotlib axes to plot on
        objective_func: Callable that takes tensor (n x 2) and returns tensor (n,)
        bounds: Domain bounds ((x1_min, x1_max), (x2_min, x2_max))
        n_points: Number of points per dimension for grid evaluation
        observations: Dict with keys:
            - 'X': Input locations (n x 2 tensor)
            - 'Y_true': True values (n tensor)
            - 'Y_observed': Observed values (n tensor)
            - 'corruption_levels': Optional corruption amounts (n tensor)
        show_corruption: Whether to highlight corrupted points
        show_colorbar: Whether to show colorbar
        config: Plot configuration
        
    Returns:
        The axes object with the plot
    """
    if config is None:
        config = PlotConfig2D()
    
    # Create grid
    x1 = torch.linspace(bounds[0][0], bounds[0][1], n_points)
    x2 = torch.linspace(bounds[1][0], bounds[1][1], n_points)
    X1, X2 = torch.meshgrid(x1, x2, indexing='xy')
    
    # Evaluate function on grid
    grid_points = torch.stack([X1.flatten(), X2.flatten()], dim=1)
    Z = torch.zeros(n_points * n_points)
    
    # Evaluate in batches for efficiency
    batch_size = 100
    for i in range(0, len(grid_points), batch_size):
        batch = grid_points[i:i+batch_size]
        Z[i:i+batch_size] = objective_func(batch).squeeze()
    
    Z = Z.reshape(n_points, n_points)
    
    # Create heatmap using imshow
    extent = [bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1]]
    im = ax.imshow(Z.numpy().T, extent=extent, origin='lower', 
                   cmap=config.colormap, aspect='auto', interpolation='bilinear')
    
    if show_colorbar:
        plt.colorbar(im, ax=ax, label='f(x)')
    
    # Plot observations if provided
    if observations is not None:
        X = observations['X']
        
        # Plot observed points
        if 'Y_observed' in observations:
            # Use white markers with black edges for visibility
            ax.scatter(X[:, 0], X[:, 1], 
                      c='white',
                      s=config.markersize**2,
                      edgecolors='black',
                      linewidths=1.5,
                      zorder=5,
                      label='Observations')
            
            # Highlight corrupted points
            if show_corruption and 'corruption_levels' in observations:
                corruption = observations['corruption_levels']
                corrupted_mask = (corruption != 0)
                
                if corrupted_mask.any():
                    ax.scatter(X[corrupted_mask, 0], X[corrupted_mask, 1],
                             s=(config.markersize*1.5)**2,
                             marker='D', 
                             facecolors='none',
                             edgecolors='red',
                             linewidths=2,
                             label='Corrupted',
                             zorder=6)
    
    ax.set_xlabel('x₁', fontsize=config.fontsize)
    ax.set_ylabel('x₂', fontsize=config.fontsize)
    ax.set_xlim(bounds[0])
    ax.set_ylim(bounds[1])
    
    if config.grid:
        ax.grid(True, alpha=0.3, color='white', linewidth=0.5)
    
    if config.legend and observations is not None:
        ax.legend(fontsize=config.fontsize-2, loc='best')
    
    return ax


def plot_objective_2d_surface(
    ax: Axes,
    objective_func: Callable,
    bounds: Tuple[Tuple[float, float], Tuple[float, float]],
    n_points: int = 50,
    observations: Optional[Dict[str, torch.Tensor]] = None,
    show_corruption: bool = True,
    config: Optional[PlotConfig2D] = None
) -> Axes:
    """Plot 2D objective function as 3D surface with optional observations.
    
    Args:
        ax: Matplotlib 3D axes to plot on (must be created with projection='3d')
        objective_func: Callable that takes tensor (n x 2) and returns tensor (n,)
        bounds: Domain bounds ((x1_min, x1_max), (x2_min, x2_max))
        n_points: Number of points per dimension for grid evaluation
        observations: Dict with keys:
            - 'X': Input locations (n x 2 tensor)
            - 'Y_true': True values (n tensor)
            - 'Y_observed': Observed values (n tensor)
            - 'corruption_levels': Optional corruption amounts (n tensor)
        show_corruption: Whether to highlight corrupted points
        config: Plot configuration
        
    Returns:
        The axes object with the plot
    """
    if config is None:
        config = PlotConfig2D()
    
    # Create grid
    x1 = torch.linspace(bounds[0][0], bounds[0][1], n_points)
    x2 = torch.linspace(bounds[1][0], bounds[1][1], n_points)
    X1, X2 = torch.meshgrid(x1, x2, indexing='xy')
    
    # Evaluate function on grid
    grid_points = torch.stack([X1.flatten(), X2.flatten()], dim=1)
    Z = torch.zeros(n_points * n_points)
    
    # Evaluate in batches
    batch_size = 100
    for i in range(0, len(grid_points), batch_size):
        batch = grid_points[i:i+batch_size]
        Z[i:i+batch_size] = objective_func(batch).squeeze()
    
    Z = Z.reshape(n_points, n_points)
    
    # Create surface plot
    surf = ax.plot_surface(X1.numpy(), X2.numpy(), Z.numpy(),
                          cmap=config.colormap,
                          alpha=config.surface_alpha,
                          linewidth=0,
                          antialiased=True)
    
    # Plot observations if provided
    if observations is not None:
        X = observations['X']
        
        # Plot true values
        if 'Y_true' in observations:
            ax.scatter(X[:, 0], X[:, 1], observations['Y_true'],
                      c='green', s=config.markersize**2,
                      marker='o', label='True values', zorder=5)
        
        # Plot observed values
        if 'Y_observed' in observations:
            ax.scatter(X[:, 0], X[:, 1], observations['Y_observed'],
                      c='blue', s=config.markersize**2,
                      marker='x', label='Observed values', zorder=6)
            
            # Connect true and observed for corrupted points
            if show_corruption and 'corruption_levels' in observations and 'Y_true' in observations:
                corruption = observations['corruption_levels']
                corrupted_mask = (corruption != 0)
                
                if corrupted_mask.any():
                    for i in range(len(X)):
                        if corrupted_mask[i]:
                            ax.plot([X[i, 0], X[i, 0]], 
                                   [X[i, 1], X[i, 1]],
                                   [observations['Y_true'][i], observations['Y_observed'][i]],
                                   'r--', alpha=0.5, linewidth=1)
    
    ax.set_xlabel('x₁', fontsize=config.fontsize)
    ax.set_ylabel('x₂', fontsize=config.fontsize)
    ax.set_zlabel('f(x)', fontsize=config.fontsize)
    ax.set_xlim(bounds[0])
    ax.set_ylim(bounds[1])
    
    if config.legend and observations is not None:
        ax.legend(fontsize=config.fontsize-2)
    
    return ax


def plot_posterior_2d_mean(
    ax: Axes,
    model,
    bounds: Tuple[Tuple[float, float], Tuple[float, float]],
    n_points: int = 50,
    observations: Optional[Dict[str, torch.Tensor]] = None,
    show_std: bool = False,
    config: Optional[PlotConfig2D] = None
) -> Axes:
    """Plot GP posterior mean (and optionally std) as heatmap.
    
    Args:
        ax: Matplotlib axes to plot on
        model: GP model with posterior() method
        bounds: Domain bounds ((x1_min, x1_max), (x2_min, x2_max))
        n_points: Number of points per dimension for grid evaluation
        observations: Optional dict with 'X' and 'Y_observed' for training points
        show_std: Whether to show standard deviation in a separate subplot
        config: Plot configuration
        
    Returns:
        The axes object with the plot
    """
    if config is None:
        config = PlotConfig2D()
    
    # Create grid
    x1 = torch.linspace(bounds[0][0], bounds[0][1], n_points)
    x2 = torch.linspace(bounds[1][0], bounds[1][1], n_points)
    X1, X2 = torch.meshgrid(x1, x2, indexing='xy')
    
    # Prepare test points
    grid_points = torch.stack([X1.flatten(), X2.flatten()], dim=1).double()
    
    # Get posterior
    model.eval()
    with torch.no_grad():
        posterior = model.posterior(grid_points)
        mean = posterior.mean.squeeze().reshape(n_points, n_points)
        if show_std:
            std = posterior.variance.squeeze().sqrt().reshape(n_points, n_points)
    
    # Plot posterior mean as heatmap
    extent = [bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1]]
    im = ax.imshow(mean.numpy().T, extent=extent, origin='lower',
                   cmap=config.colormap, aspect='auto', interpolation='bilinear')
    
    plt.colorbar(im, ax=ax, label='Posterior Mean')
    
    # Plot training observations
    if observations is not None:
        X_train = observations['X']
        ax.scatter(X_train[:, 0], X_train[:, 1],
                  c='white', s=config.markersize**2,
                  marker='o', edgecolors='black',
                  linewidths=1.5, label='Observations', zorder=5)
    
    ax.set_xlabel('x₁', fontsize=config.fontsize)
    ax.set_ylabel('x₂', fontsize=config.fontsize)
    ax.set_xlim(bounds[0])
    ax.set_ylim(bounds[1])
    ax.set_title('GP Posterior Mean', fontsize=config.fontsize+2)
    
    if config.grid:
        ax.grid(True, alpha=0.3, color='white', linewidth=0.5)
    
    if config.legend and observations is not None:
        ax.legend(fontsize=config.fontsize-2, loc='best')
    
    return ax


def plot_acquisition_2d(
    ax: Axes,
    acquisition_func: Callable,
    bounds: Tuple[Tuple[float, float], Tuple[float, float]],
    n_points: int = 50,
    next_point: Optional[torch.Tensor] = None,
    observations: Optional[Dict[str, torch.Tensor]] = None,
    config: Optional[PlotConfig2D] = None
) -> Axes:
    """Plot 2D acquisition function as heatmap.
    
    Args:
        ax: Matplotlib axes to plot on
        acquisition_func: Acquisition function callable
        bounds: Domain bounds ((x1_min, x1_max), (x2_min, x2_max))
        n_points: Number of points per dimension for grid evaluation
        next_point: Optional next point selected by acquisition (n x 2 tensor)
        observations: Optional dict with current observations
        config: Plot configuration
        
    Returns:
        The axes object with the plot
    """
    if config is None:
        config = PlotConfig2D()
    
    # Create grid
    x1 = torch.linspace(bounds[0][0], bounds[0][1], n_points)
    x2 = torch.linspace(bounds[1][0], bounds[1][1], n_points)
    X1, X2 = torch.meshgrid(x1, x2, indexing='xy')
    
    # Prepare test points
    grid_points = torch.stack([X1.flatten(), X2.flatten()], dim=1).double()
    
    # Evaluate acquisition function
    with torch.no_grad():
        acq_values = acquisition_func(grid_points.unsqueeze(1)).squeeze()
    
    acq_values = acq_values.reshape(n_points, n_points)
    
    # Plot acquisition function as heatmap
    extent = [bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1]]
    im = ax.imshow(acq_values.numpy().T, extent=extent, origin='lower',
                   cmap='RdYlBu_r', aspect='auto', interpolation='bilinear')
    
    plt.colorbar(im, ax=ax, label='Acquisition Value')
    
    # Plot current observations
    if observations is not None:
        X_train = observations['X']
        ax.scatter(X_train[:, 0], X_train[:, 1],
                  c='black', s=config.markersize**2,
                  marker='o', label='Current points', zorder=5)
    
    # Mark next point
    if next_point is not None:
        ax.scatter(next_point[0, 0], next_point[0, 1],
                  c='red', s=(config.markersize*2)**2,
                  marker='*', label='Next point', zorder=6)
    
    ax.set_xlabel('x₁', fontsize=config.fontsize)
    ax.set_ylabel('x₂', fontsize=config.fontsize)
    ax.set_xlim(bounds[0])
    ax.set_ylim(bounds[1])
    ax.set_title('Acquisition Function', fontsize=config.fontsize+2)
    
    if config.grid:
        ax.grid(True, alpha=0.3, color='white', linewidth=0.5)
    
    if config.legend:
        ax.legend(fontsize=config.fontsize-2, loc='best')
    
    return ax


def plot_optimization_path_2d(
    ax: Axes,
    results: Dict[str, Any],
    objective_func: Optional[Callable] = None,
    bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
    show_order: bool = True,
    config: Optional[PlotConfig2D] = None
) -> Axes:
    """Plot the optimization path on a 2D heatmap.
    
    Args:
        ax: Matplotlib axes to plot on
        results: Results dictionary from ExperimentRunner
        objective_func: Optional objective function to plot as background
        bounds: Domain bounds (if None, inferred from data)
        show_order: Whether to show evaluation order with numbers/colors
        config: Plot configuration
        
    Returns:
        The axes object with the plot
    """
    if config is None:
        config = PlotConfig2D()
    
    # Extract points
    X = results['X']
    n_points = len(X)
    
    # Infer bounds if not provided
    if bounds is None:
        x1_min, x1_max = X[:, 0].min().item(), X[:, 0].max().item()
        x2_min, x2_max = X[:, 1].min().item(), X[:, 1].max().item()
        # Add some padding
        pad1 = 0.1 * (x1_max - x1_min)
        pad2 = 0.1 * (x2_max - x2_min)
        bounds = ((x1_min - pad1, x1_max + pad1), (x2_min - pad2, x2_max + pad2))
    
    # Plot objective function as background if provided
    if objective_func is not None:
        plot_objective_2d_heatmap(ax, objective_func, bounds, 
                                 show_colorbar=False, config=config)
    
    # Plot optimization path
    if show_order:
        # Color points by iteration order
        colors = plt.cm.plasma(np.linspace(0, 1, n_points))
        for i in range(n_points):
            ax.scatter(X[i, 0], X[i, 1], c=[colors[i]], 
                      s=config.markersize**2, zorder=5,
                      edgecolors='white', linewidths=1)
            if i < 20:  # Only label first 20 points to avoid clutter
                ax.annotate(str(i+1), (X[i, 0], X[i, 1]),
                          xytext=(3, 3), textcoords='offset points',
                          fontsize=8, color='white',
                          bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.5))
        
        # Connect points with lines
        ax.plot(X[:, 0], X[:, 1], 'w-', alpha=0.3, linewidth=0.5)
    else:
        # Just plot all points
        ax.scatter(X[:, 0], X[:, 1], c='white', 
                  s=config.markersize**2, zorder=5,
                  edgecolors='black', linewidths=1)
    
    # Mark best point
    best_idx = results['Y_observed'].argmax()
    ax.scatter(X[best_idx, 0], X[best_idx, 1],
              c='red', s=(config.markersize*2)**2,
              marker='*', label='Best found', zorder=6)
    
    ax.set_xlabel('x₁', fontsize=config.fontsize)
    ax.set_ylabel('x₂', fontsize=config.fontsize)
    ax.set_xlim(bounds[0])
    ax.set_ylim(bounds[1])
    ax.set_title('Optimization Path', fontsize=config.fontsize+2)
    
    if config.grid:
        ax.grid(True, alpha=0.3, color='white', linewidth=0.5)
    
    if config.legend:
        ax.legend(fontsize=config.fontsize-2, loc='best')
    
    return ax


def plot_experiment_summary_2d(
    results: Dict[str, Any],
    objective_func: Optional[Callable] = None,
    optimal_value: Optional[float] = None,
    optimal_point: Optional[torch.Tensor] = None,
    optimal_points: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
    bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
    save_path: Optional[str] = None,
    config: Optional[PlotConfig2D] = None
) -> Figure:
    """Create a comprehensive summary plot for a 2D BO experiment.
    
    Args:
        results: Results dictionary from ExperimentRunner
        objective_func: True objective function
        optimal_value: Known optimal value
        optimal_point: Known optimal point (2D tensor) - for backward compatibility
        optimal_points: Multiple optimal points as tensor (n_optima, 2) or list of 2D tensors
        bounds: Domain bounds
        save_path: Optional path to save the figure
        config: Plot configuration
        
    Returns:
        The matplotlib figure
    """
    if config is None:
        config = PlotConfig2D(figsize=(16, 12))
    
    # Create 2x3 subplot grid
    fig = plt.figure(figsize=config.figsize)
    
    # Prepare observations dict
    observations = {
        'X': results['X'],
        'Y_true': results.get('Y_true', results['Y_observed']),
        'Y_observed': results['Y_observed'],
        'corruption_levels': results.get('corruption_levels', torch.zeros_like(results['Y_observed']))
    }
    
    # Infer bounds if not provided
    if bounds is None:
        X = results['X']
        x1_min, x1_max = X[:, 0].min().item(), X[:, 0].max().item()
        x2_min, x2_max = X[:, 1].min().item(), X[:, 1].max().item()
        pad1 = 0.1 * (x1_max - x1_min)
        pad2 = 0.1 * (x2_max - x2_min)
        bounds = ((x1_min - pad1, x1_max + pad1), (x2_min - pad2, x2_max + pad2))
    
    # 1. Objective function with observations (heatmap)
    ax1 = plt.subplot(2, 3, 1)
    if objective_func is not None:
        plot_objective_2d_heatmap(ax1, objective_func, bounds, 
                                 observations=observations, config=config)
    ax1.set_title('Objective Function', fontsize=config.fontsize+2)
    
    # Mark optimal point(s) if known
    points_to_plot = []
    if optimal_points is not None:
        # Handle multiple optimal points
        if isinstance(optimal_points, torch.Tensor):
            if optimal_points.dim() == 1:
                points_to_plot = [optimal_points]
            else:
                points_to_plot = [optimal_points[i] for i in range(optimal_points.shape[0])]
        else:
            points_to_plot = optimal_points
    elif optimal_point is not None:
        # Handle single optimal point (backward compatibility)
        points_to_plot = [optimal_point]
    
    # Plot all optimal points
    for i, opt_pt in enumerate(points_to_plot):
        label = 'True optima' if i == 0 and len(points_to_plot) > 1 else ('True optimum' if i == 0 else None)
        ax1.scatter(opt_pt[0], opt_pt[1],
                   c='yellow', s=(config.markersize*2)**2,
                   marker='*', edgecolors='black',
                   linewidths=1, label=label, zorder=7)
    
    if points_to_plot:
        ax1.legend(fontsize=config.fontsize-2)
    
    # 2. 3D surface plot
    ax2 = plt.subplot(2, 3, 2, projection='3d')
    if objective_func is not None:
        plot_objective_2d_surface(ax2, objective_func, bounds,
                                 observations=observations, config=config)
    ax2.set_title('3D Surface', fontsize=config.fontsize+2)
    
    # 3. Optimization path
    ax3 = plt.subplot(2, 3, 3)
    plot_optimization_path_2d(ax3, results, objective_func, bounds, config=config)
    ax3.set_title('Optimization Path', fontsize=config.fontsize+2)
    
    # 4. Posterior mean (if model available)
    ax4 = plt.subplot(2, 3, 4)
    if 'final_model' in results:
        plot_posterior_2d_mean(ax4, results['final_model'], bounds,
                              observations=observations, config=config)
    else:
        ax4.text(0.5, 0.5, 'No model available', 
                ha='center', va='center', transform=ax4.transAxes)
    
    # 5. Cumulative regret (using common function)
    ax5 = plt.subplot(2, 3, 5)
    if optimal_value is not None:
        plot_regret(ax5, results, optimal_value, regret_type='cumulative', config=config)
        ax5.set_title('Cumulative Regret', fontsize=config.fontsize+2)
    else:
        ax5.text(0.5, 0.5, 'Optimal value unknown',
                ha='center', va='center', transform=ax5.transAxes)
    
    # 6. Instantaneous regret (using common function)
    ax6 = plt.subplot(2, 3, 6)
    if optimal_value is not None:
        plot_regret(ax6, results, optimal_value, regret_type='instantaneous', config=config)
        ax6.set_title('Instantaneous Regret', fontsize=config.fontsize+2)
    else:
        ax6.text(0.5, 0.5, 'Optimal value unknown',
                ha='center', va='center', transform=ax6.transAxes)
    
    fig.suptitle('2D Bayesian Optimization Experiment Summary', 
                fontsize=config.fontsize+4)
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=config.dpi, bbox_inches='tight')
    
    return fig