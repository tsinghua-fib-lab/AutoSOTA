"""
Visualization Module

This module provides functions for creating publication-quality plots
to visualize experimental results, including:
    - RMSE comparison plots
    - Confidence interval width plots
    - Coverage rate plots
    - Combined multi-panel figures
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional, List


def plot_comparison_results(
    results: Dict[str, pd.DataFrame],
    output_path: Optional[str] = None,
    figsize: tuple = (15, 5),
    font_size: int = 18,
    dpi: int = 300
) -> None:
    """
    Create a three-panel comparison plot of all sampling methods.
    
    Generates publication-quality figures showing:
        (a) Root Mean Squared Error (RMSE)
        (b) Confidence Interval Width
        (c) Coverage Rate
    
    Args:
        results: Dictionary containing DataFrames with keys:
            - 'rmse': RMSE results
            - 'interval_width': Interval width results
            - 'coverage': Coverage rate results
        output_path: Path to save the figure (optional)
        figsize: Figure size as (width, height) (default: (15, 5))
        font_size: Font size for labels and titles (default: 18)
        dpi: Resolution for saved figure (default: 300)
    
    Example:
        >>> plot_comparison_results(results, 'figures/comparison.pdf')
    """
    # Extract DataFrames
    rmse_df = results['rmse']
    interval_df = results['interval_width']
    coverage_df = results['coverage']
    
    # Method names and colors
    methods = ['classical', 'uniform', 'poisson_active', 'cube_active']
    colors = {
        'uniform': 'red',
        'poisson_active': 'green',
        'cube_active': 'blue',
        'classical': 'orange'
    }
    
    # Create figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Convert budget to sample size
    N = 5000  # Assumed population size, adjust if needed
    sample_sizes = rmse_df['budget'] * N
    
    # Helper function for plotting
    def plot_metric(ax, data, ylabel, ylim_zero=False, log_scale=False):
        for method in methods:
            label = 'cube_active (ours)' if method == 'cube_active' else method
            ax.plot(
                sample_sizes,
                data[method],
                label=label,
                color=colors[method],
                linewidth=2
            )
        
        ax.set_xlabel("Sample Size", fontsize=font_size)
        ax.set_ylabel(ylabel, fontsize=font_size)
        ax.tick_params(axis='both', labelsize=font_size)
        ax.grid(True, linestyle='--', linewidth=1, alpha=0.7)
        
        if log_scale:
            ax.set_yscale('log')
        if ylim_zero:
            ax.set_ylim(bottom=0)
    
    # Panel (a): RMSE
    plot_metric(axes[0], rmse_df, "RMSE", log_scale=False)
    axes[0].set_title("(a)", fontsize=font_size)
    
    # Panel (b): Interval Width
    plot_metric(axes[1], interval_df, "Interval Width", log_scale=False)
    axes[1].set_title("(b)", fontsize=font_size)
    
    # Panel (c): Coverage Rate
    plot_metric(axes[2], coverage_df, "Coverage Rate", ylim_zero=True)
    axes[2].set_ylim(0, 1)
    axes[2].set_yticks([0, 0.5, 0.8, 0.9, 1.0])
    axes[2].axhline(y=0.95, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    axes[2].set_title("(c)", fontsize=font_size)
    
    # Remove individual legends
    for ax in axes:
        if ax.get_legend():
            ax.legend_.remove()
    
    # Create shared legend at the bottom
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.08),
        ncol=4,
        frameon=False,
        fontsize=font_size
    )
    
    plt.tight_layout()
    
    # Save figure if path provided
    if output_path:
        plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=dpi)
        print(f"Figure saved to {output_path}")
    
    plt.show()


def plot_single_metric(
    data: pd.DataFrame,
    metric_name: str,
    ylabel: str,
    output_path: Optional[str] = None,
    figsize: tuple = (8, 6),
    font_size: int = 14
) -> None:
    """
    Plot a single metric comparison.
    
    Args:
        data: DataFrame containing the metric for all methods
        metric_name: Name of the metric (for title)
        ylabel: Label for y-axis
        output_path: Path to save the figure (optional)
        figsize: Figure size (default: (8, 6))
        font_size: Font size for labels (default: 14)
    
    Example:
        >>> plot_single_metric(rmse_df, 'RMSE', 'Root Mean Squared Error')
    """
    methods = ['classical', 'uniform', 'poisson_active', 'cube_active']
    colors = {
        'uniform': 'red',
        'poisson_active': 'green',
        'cube_active': 'blue',
        'classical': 'orange'
    }
    
    fig, ax = plt.subplots(figsize=figsize)
    
    N = 5000
    sample_sizes = data['budget'] * N
    
    for method in methods:
        label = 'cube_active (ours)' if method == 'cube_active' else method
        ax.plot(
            sample_sizes,
            data[method],
            label=label,
            color=colors[method],
            linewidth=2,
            marker='o',
            markersize=4
        )
    
    ax.set_xlabel("Sample Size", fontsize=font_size)
    ax.set_ylabel(ylabel, fontsize=font_size)
    ax.set_title(f"{metric_name} Comparison", fontsize=font_size + 2)
    ax.legend(fontsize=font_size - 2)
    ax.grid(True, linestyle='--', linewidth=0.8, alpha=0.7)
    ax.tick_params(axis='both', labelsize=font_size - 2)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Figure saved to {output_path}")
    
    plt.show()


def plot_uncertainty_distribution(
    uncertainty: np.ndarray,
    bins: int = 50,
    figsize: tuple = (10, 6),
    output_path: Optional[str] = None
) -> None:
    """
    Plot the distribution of uncertainty estimates.
    
    Args:
        uncertainty: Array of uncertainty estimates
        bins: Number of histogram bins (default: 50)
        figsize: Figure size (default: (10, 6))
        output_path: Path to save the figure (optional)
    
    Example:
        >>> plot_uncertainty_distribution(uncertainty)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.hist(uncertainty, bins=bins, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(uncertainty.mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {uncertainty.mean():.4f}')
    ax.axvline(np.median(uncertainty), color='green', linestyle='--', linewidth=2,
               label=f'Median: {np.median(uncertainty):.4f}')
    
    ax.set_xlabel('Uncertainty', fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    ax.set_title('Distribution of Uncertainty Estimates', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Figure saved to {output_path}")
    
    plt.show()


def plot_sampling_allocation(
    uncertainty: np.ndarray,
    inclusion_probs: np.ndarray,
    figsize: tuple = (10, 6),
    output_path: Optional[str] = None
) -> None:
    """
    Visualize the relationship between uncertainty and sampling probability.
    
    Args:
        uncertainty: Array of uncertainty estimates
        inclusion_probs: Array of inclusion probabilities
        figsize: Figure size (default: (10, 6))
        output_path: Path to save the figure (optional)
    
    Example:
        >>> plot_sampling_allocation(uncertainty, inclusion_probs)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create scatter plot with transparency
    scatter = ax.scatter(
        uncertainty,
        inclusion_probs,
        alpha=0.5,
        c=inclusion_probs,
        cmap='viridis',
        s=10
    )
    
    ax.set_xlabel('Uncertainty', fontsize=14)
    ax.set_ylabel('Inclusion Probability', fontsize=14)
    ax.set_title('Active Sampling Allocation', fontsize=16)
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Inclusion Probability', fontsize=12)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Figure saved to {output_path}")
    
    plt.show()


def save_results_to_csv(
    results: Dict[str, pd.DataFrame],
    output_dir: str = 'results'
) -> None:
    """
    Save all experimental results to CSV files.
    
    Args:
        results: Dictionary of result DataFrames
        output_dir: Directory to save CSV files (default: 'results')
    
    Example:
        >>> save_results_to_csv(results, 'results')
    """
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each DataFrame
    for metric_name, df in results.items():
        filename = f"{metric_name}.csv"
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"Saved {metric_name} results to {filepath}")
