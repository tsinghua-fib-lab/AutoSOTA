"""
Visualization Module for Disagreement-Aware Evaluation.

This module provides functions to create various plots and visualizations:
- Agent score comparisons
- Annotator confusion matrices
- Item ambiguity distributions
- Ranking stability analysis
- Bootstrap confidence intervals
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def plot_agent_scores_comparison(
    comparison_df: pd.DataFrame,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a grouped bar chart comparing agent scores across methods.
    
    Args:
        comparison_df: DataFrame with agent_id and score columns for each method
        figsize: Figure size
        save_path: Path to save the figure
        
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    agents = comparison_df['agent_id'].values
    x = np.arange(len(agents))
    width = 0.25
    
    bars1 = ax.bar(x - width, comparison_df['score_mv'], width, label='Majority Vote', color='#2ecc71')
    bars2 = ax.bar(x, comparison_df['score_ds'], width, label='Dawid-Skene (Hard)', color='#3498db')
    bars3 = ax.bar(x + width, comparison_df['score_pec'], width, label='Posterior Expected Credit', color='#9b59b6')
    
    ax.set_xlabel('Agent', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Agent Scores: Comparison Across Methods', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(agents, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.05)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_score_scatter(
    comparison_df: pd.DataFrame,
    method1: str = 'score_mv',
    method2: str = 'score_pec',
    labels: Tuple[str, str] = ('Majority Vote', 'Posterior Expected Credit'),
    figsize: Tuple[int, int] = (8, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a scatter plot comparing scores between two methods.
    
    Args:
        comparison_df: DataFrame with agent scores
        method1, method2: Column names for the two methods
        labels: Axis labels
        figsize: Figure size
        save_path: Path to save
        
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.scatter(comparison_df[method1], comparison_df[method2], 
               s=100, alpha=0.7, c='#3498db', edgecolors='white', linewidth=2)
    
    # Add diagonal line
    lims = [0, 1]
    ax.plot(lims, lims, 'k--', alpha=0.5, label='y = x')
    
    # Add agent labels
    for idx, row in comparison_df.iterrows():
        ax.annotate(row['agent_id'], 
                   (row[method1], row[method2]),
                   xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax.set_xlabel(labels[0], fontsize=12)
    ax.set_ylabel(labels[1], fontsize=12)
    ax.set_title(f'{labels[1]} vs {labels[0]}', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_aspect('equal')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_annotator_confusion_matrices(
    confusion_matrices: Dict[str, np.ndarray],
    class_names: Optional[List[str]] = None,
    figsize_per_plot: Tuple[int, int] = (4, 4),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot confusion matrices for multiple annotators.
    
    Args:
        confusion_matrices: Dict mapping annotator_id to confusion matrix
        class_names: Names for each class
        figsize_per_plot: Size of each subplot
        save_path: Path to save
        
    Returns:
        matplotlib Figure
    """
    n_annotators = len(confusion_matrices)
    n_cols = min(4, n_annotators)
    n_rows = (n_annotators + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, 
                             figsize=(figsize_per_plot[0] * n_cols, figsize_per_plot[1] * n_rows))
    axes = np.atleast_2d(axes)
    
    if class_names is None:
        n_classes = list(confusion_matrices.values())[0].shape[0]
        class_names = [f'Class {i}' for i in range(n_classes)]
    
    for idx, (annotator_id, cm) in enumerate(confusion_matrices.items()):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        im = ax.imshow(cm, cmap='Blues', vmin=0, vmax=1)
        
        # Add text annotations
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                text = f'{cm[i, j]:.2f}'
                color = 'white' if cm[i, j] > 0.5 else 'black'
                ax.text(j, i, text, ha='center', va='center', color=color, fontsize=10)
        
        ax.set_xlabel('Observed Label')
        ax.set_ylabel('True Label')
        ax.set_title(f'{annotator_id}', fontsize=10)
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
    
    # Hide empty subplots
    for idx in range(n_annotators, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    fig.suptitle('Annotator Confusion Matrices', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_annotator_quality(
    quality_df: pd.DataFrame,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot annotator quality metrics.
    
    Args:
        quality_df: DataFrame with annotator quality metrics
        figsize: Figure size
        save_path: Path to save
        
    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Sort by accuracy
    quality_df = quality_df.sort_values('accuracy', ascending=True)
    
    # Accuracy bar chart
    ax = axes[0]
    bars = ax.barh(quality_df['annotator_id'], quality_df['accuracy'], color='#3498db')
    ax.set_xlabel('Accuracy', fontsize=12)
    ax.set_title('Annotator Accuracy', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Random baseline')
    ax.legend()
    
    # Leniency vs Strictness scatter
    ax = axes[1]
    ax.scatter(quality_df['leniency'], quality_df['strictness'], 
               s=100, alpha=0.7, c='#9b59b6')
    
    for idx, row in quality_df.iterrows():
        ax.annotate(row['annotator_id'],
                   (row['leniency'], row['strictness']),
                   xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax.set_xlabel('Leniency', fontsize=12)
    ax.set_ylabel('Strictness', fontsize=12)
    ax.set_title('Annotator Bias', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_ambiguity_distribution(
    ambiguity_df: pd.DataFrame,
    figsize: Tuple[int, int] = (10, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot the distribution of item ambiguity scores.
    
    Args:
        ambiguity_df: DataFrame with item ambiguity scores
        figsize: Figure size
        save_path: Path to save
        
    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Histogram
    ax = axes[0]
    ax.hist(ambiguity_df['ambiguity'], bins=30, edgecolor='white', color='#2ecc71', alpha=0.7)
    ax.set_xlabel('Ambiguity Score', fontsize=12)
    ax.set_ylabel('Number of Items', fontsize=12)
    ax.set_title('Distribution of Item Ambiguity', fontsize=12, fontweight='bold')
    
    # Mean and median lines
    mean_amb = ambiguity_df['ambiguity'].mean()
    median_amb = ambiguity_df['ambiguity'].median()
    ax.axvline(mean_amb, color='red', linestyle='--', label=f'Mean: {mean_amb:.3f}')
    ax.axvline(median_amb, color='blue', linestyle='--', label=f'Median: {median_amb:.3f}')
    ax.legend()
    
    # Confidence vs Ambiguity
    ax = axes[1]
    ax.scatter(ambiguity_df['confidence'], ambiguity_df['ambiguity'], 
               alpha=0.5, c='#3498db', s=20)
    ax.set_xlabel('Confidence (max probability)', fontsize=12)
    ax.set_ylabel('Ambiguity (1 - max probability)', fontsize=12)
    ax.set_title('Confidence vs Ambiguity', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_bootstrap_confidence_intervals(
    bootstrap_results: Dict[str, pd.DataFrame],
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot bootstrap confidence intervals for agent scores.
    
    Args:
        bootstrap_results: Dict mapping method name to bootstrap results DataFrame
        figsize: Figure size
        save_path: Path to save
        
    Returns:
        matplotlib Figure
    """
    n_methods = len(bootstrap_results)
    fig, axes = plt.subplots(1, n_methods, figsize=figsize, sharey=True)
    
    if n_methods == 1:
        axes = [axes]
    
    colors = ['#2ecc71', '#3498db', '#9b59b6']
    
    for idx, (method_name, df) in enumerate(bootstrap_results.items()):
        ax = axes[idx]
        df = df.sort_values('score', ascending=True)
        
        y_pos = np.arange(len(df))
        
        # Plot confidence intervals
        ax.barh(y_pos, df['score'], color=colors[idx % len(colors)], alpha=0.7, label='Score')
        
        # Calculate error bars safely (ensure non-negative)
        err_lower = np.maximum(0, df['score'].values - df['ci_lower'].values)
        err_upper = np.maximum(0, df['ci_upper'].values - df['score'].values)
        
        ax.errorbar(df['score'], y_pos,
                   xerr=[err_lower, err_upper],
                   fmt='none', ecolor='black', capsize=3, capthick=1)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df['agent_id'])
        ax.set_xlabel('Score', fontsize=12)
        ax.set_title(f'{method_name}', fontsize=12, fontweight='bold')
        ax.set_xlim(0, 1.05)
    
    fig.suptitle('Agent Scores with 95% Confidence Intervals', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_ranking_stability(
    stability_df: pd.DataFrame,
    figsize: Tuple[int, int] = (10, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot ranking stability comparison across methods.
    
    Args:
        stability_df: DataFrame with stability metrics
        figsize: Figure size
        save_path: Path to save
        
    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    x = np.arange(len(stability_df))
    colors = ['#2ecc71', '#3498db', '#9b59b6']
    
    # Mean rank standard deviation
    ax = axes[0]
    bars = ax.bar(x, stability_df['mean_rank_std'], color=colors[:len(x)])
    ax.set_ylabel('Mean Rank Std Dev', fontsize=12)
    ax.set_title('Ranking Stability\n(lower is better)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(stability_df['method'], rotation=45, ha='right')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=10)
    
    # Mean rank range
    ax = axes[1]
    bars = ax.bar(x, stability_df['mean_rank_range'], color=colors[:len(x)])
    ax.set_ylabel('Mean Rank Range', fontsize=12)
    ax.set_title('Ranking Variability\n(lower is better)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(stability_df['method'], rotation=45, ha='right')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_score_differences(
    comparison_df: pd.DataFrame,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot the differences in scores between methods.
    
    Args:
        comparison_df: DataFrame with score differences
        figsize: Figure size
        save_path: Path to save
        
    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Sort by PEC score
    df = comparison_df.sort_values('score_pec', ascending=False)
    
    # DS vs MV difference
    ax = axes[0]
    colors = ['#2ecc71' if x >= 0 else '#e74c3c' for x in df['diff_ds_mv']]
    ax.barh(df['agent_id'], df['diff_ds_mv'], color=colors, alpha=0.7)
    ax.axvline(0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Score Difference', fontsize=12)
    ax.set_title('Dawid-Skene - Majority Vote', fontsize=12, fontweight='bold')
    
    # PEC vs MV difference
    ax = axes[1]
    colors = ['#2ecc71' if x >= 0 else '#e74c3c' for x in df['diff_pec_mv']]
    ax.barh(df['agent_id'], df['diff_pec_mv'], color=colors, alpha=0.7)
    ax.axvline(0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Score Difference', fontsize=12)
    ax.set_title('PEC - Majority Vote', fontsize=12, fontweight='bold')
    
    fig.suptitle('Score Changes from Majority Vote Baseline', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def create_all_plots(
    results,  # ScoringResults object
    comparison_df: pd.DataFrame,
    stability_df: Optional[pd.DataFrame] = None,
    output_dir: str = 'plots',
    prefix: str = ''
) -> None:
    """
    Create all standard plots and save them to a directory.
    
    Args:
        results: ScoringResults object
        comparison_df: Comparison table
        stability_df: Stability analysis results
        output_dir: Directory to save plots
        prefix: Prefix for filenames
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Creating plots...")
    
    # 1. Agent scores comparison
    plot_agent_scores_comparison(
        comparison_df,
        save_path=str(output_dir / f'{prefix}agent_scores_comparison.png')
    )
    
    # 2. Score scatter plots
    plot_score_scatter(
        comparison_df,
        method1='score_mv', method2='score_pec',
        labels=('Majority Vote', 'Posterior Expected Credit'),
        save_path=str(output_dir / f'{prefix}scatter_mv_vs_pec.png')
    )
    
    plot_score_scatter(
        comparison_df,
        method1='score_mv', method2='score_ds',
        labels=('Majority Vote', 'Dawid-Skene (Hard)'),
        save_path=str(output_dir / f'{prefix}scatter_mv_vs_ds.png')
    )
    
    # 3. Score differences
    plot_score_differences(
        comparison_df,
        save_path=str(output_dir / f'{prefix}score_differences.png')
    )
    
    # 4. Annotator quality
    if results.annotator_quality is not None:
        plot_annotator_quality(
            results.annotator_quality,
            save_path=str(output_dir / f'{prefix}annotator_quality.png')
        )
    
    # 5. Item ambiguity
    if results.item_ambiguity is not None:
        plot_ambiguity_distribution(
            results.item_ambiguity,
            save_path=str(output_dir / f'{prefix}item_ambiguity.png')
        )
    
    # 6. Ranking stability
    if stability_df is not None:
        plot_ranking_stability(
            stability_df,
            save_path=str(output_dir / f'{prefix}ranking_stability.png')
        )
    
    # 7. Bootstrap confidence intervals (if available)
    if results.mv_bootstrap is not None:
        bootstrap_results = {
            'Majority Vote': results.mv_bootstrap,
            'Dawid-Skene (Hard)': results.ds_bootstrap,
            'Posterior Expected Credit': results.pec_bootstrap
        }
        plot_bootstrap_confidence_intervals(
            bootstrap_results,
            save_path=str(output_dir / f'{prefix}bootstrap_ci.png')
        )
    
    print(f"All plots saved to {output_dir}/")
    plt.close('all')


if __name__ == "__main__":
    print("Visualization module loaded. Import and use the plotting functions.")
