# -*- coding: utf-8 -*-
"""
scChord Visualization Module

This module provides visualization functions for evaluation results,
referenced from ComputePCC&CMD&RMSE.ipynb and ComputeRC&RU.ipynb.
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from scipy.stats import pearsonr

# Set matplotlib backend and default parameters
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams['font.serif'] = ['Arial']

import warnings
warnings.filterwarnings("ignore")


def plot_pcc_boxplot(
    pcc_values: np.ndarray,
    title: str = "PCC",
    xlabel: str = "Pearson Correlation Coefficient",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (6, 4),
    color: str = '#7DC9C4',
    xlim: Tuple[float, float] = (-1.1, 1.1)
):
    """
    Plot PCC boxplot.
    
    Args:
        pcc_values: Array of PCC values
        title: Plot title
        xlabel: X-axis label
        save_path: Path to save the figure (optional)
        figsize: Figure size
        color: Boxplot color
        xlim: X-axis limits
    """
    plt.figure(figsize=figsize, dpi=100)
    
    # Filter NaN values
    pcc_clean = pcc_values[~np.isnan(pcc_values)]
    
    ax = sns.boxplot(
        x=pcc_clean,
        orient="h",
        linewidth=1,
        width=0.5,
        color=color,
        fliersize=2,
        flierprops={'marker': 'o', 'markerfacecolor': 'gray', 'alpha': 0.5}
    )
    
    ax.set_xlabel(xlabel)
    ax.set_xlim(xlim)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add statistics
    mean_val = np.nanmean(pcc_values)
    median_val = np.nanmedian(pcc_values)
    ax.axvline(x=mean_val, color='red', linestyle='--', linewidth=1, alpha=0.7, label=f'Mean: {mean_val:.4f}')
    ax.axvline(x=median_val, color='blue', linestyle=':', linewidth=1, alpha=0.7, label=f'Median: {median_val:.4f}')
    ax.legend(loc='upper left', fontsize=8)
    
    plt.title(title, fontdict={'size': 12})
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Saved: {save_path}")
    
    plt.close()


def plot_metric_bar(
    value: float,
    title: str = "Metric",
    xlabel: str = "Value",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (6, 2),
    color: str = '#7DC9C4',
    xlim: Optional[Tuple[float, float]] = None
):
    """
    Plot single metric bar chart.
    """
    plt.figure(figsize=figsize, dpi=100)
    
    ax = plt.barh(['scChord'], [value], color=color, height=0.5)
    plt.xlabel(xlabel)
    plt.title(title, fontdict={'size': 12})
    
    # Display value on bar
    plt.text(value + 0.01, 0, f'{value:.4f}', va='center', fontsize=10)
    
    if xlim:
        plt.xlim(xlim)
    
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Saved: {save_path}")
    
    plt.close()


def plot_evaluation_summary(
    pcc_protein: np.ndarray,
    pcc_cell: np.ndarray,
    cmd_cell: float,
    cmd_protein: float,
    rmse: float,
    save_path: Optional[str] = None,
    title_prefix: str = "scChord"
):
    """
    Plot comprehensive evaluation results (5 subplots).
    
    Args:
        pcc_protein: Protein PCC [M]
        pcc_cell: Cell PCC [N]
        cmd_cell: Cell-cell CMD
        cmd_protein: Protein-protein CMD
        rmse: RMSE value
        save_path: Path to save the figure (optional)
        title_prefix: Title prefix
    """
    fig, axes = plt.subplots(1, 5, figsize=(20, 3), dpi=100)
    
    color = '#7DC9C4'  # scChord color
    
    # 1. PCC Protein boxplot
    ax1 = axes[0]
    pcc_prot_clean = pcc_protein[~np.isnan(pcc_protein)]
    sns.boxplot(
        x=pcc_prot_clean, orient="h", linewidth=1, width=0.5,
        color=color, fliersize=2, ax=ax1,
        flierprops={'marker': 'o', 'markerfacecolor': 'gray', 'alpha': 0.5}
    )
    ax1.set_xlabel("protein-protein PCC")
    ax1.set_xlim(-1.1, 1.1)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    mean_p = np.nanmean(pcc_protein)
    median_p = np.nanmedian(pcc_protein)
    ax1.set_title(f"Mean: {mean_p:.4f}, Median: {median_p:.4f}", fontsize=9)
    
    # 2. PCC Cell boxplot
    ax2 = axes[1]
    pcc_cell_clean = pcc_cell[~np.isnan(pcc_cell)]
    sns.boxplot(
        x=pcc_cell_clean, orient="h", linewidth=1, width=0.5,
        color=color, fliersize=2, ax=ax2,
        flierprops={'marker': 'o', 'markerfacecolor': 'gray', 'alpha': 0.5}
    )
    ax2.set_xlabel("cell-cell PCC")
    ax2.set_xlim(-1.1, 1.1)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    mean_c = np.nanmean(pcc_cell)
    median_c = np.nanmedian(pcc_cell)
    ax2.set_title(f"Mean: {mean_c:.4f}, Median: {median_c:.4f}", fontsize=9)
    
    # 3. CMD Cell bar
    ax3 = axes[2]
    ax3.barh(['scChord'], [cmd_cell], color=color, height=0.5)
    ax3.set_xlabel("cell-cell CMD")
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.set_xlim(0, max(0.5, cmd_cell * 1.2))
    ax3.text(cmd_cell + 0.01, 0, f'{cmd_cell:.4f}', va='center', fontsize=9)
    
    # 4. CMD Protein bar
    ax4 = axes[3]
    ax4.barh(['scChord'], [cmd_protein], color=color, height=0.5)
    ax4.set_xlabel("protein-protein CMD")
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.set_xlim(0, max(0.5, cmd_protein * 1.2))
    ax4.text(cmd_protein + 0.01, 0, f'{cmd_protein:.4f}', va='center', fontsize=9)
    
    # 5. RMSE bar
    ax5 = axes[4]
    ax5.barh(['scChord'], [rmse], color=color, height=0.5)
    ax5.set_xlabel("RMSE")
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    ax5.set_xlim(0, max(2.0, rmse * 1.2))
    ax5.text(rmse + 0.02, 0, f'{rmse:.4f}', va='center', fontsize=9)
    
    plt.suptitle(title_prefix, fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Saved: {save_path}")
    
    plt.close()


def plot_pcc_distribution(
    pcc_protein: np.ndarray,
    pcc_cell: np.ndarray,
    protein_names: Optional[List[str]] = None,
    save_dir: Optional[str] = None,
    title_prefix: str = "scChord"
):
    """
    Plot PCC distribution histograms and detailed statistics.
    
    Args:
        pcc_protein: Protein PCC [M]
        pcc_cell: Cell PCC [N]
        protein_names: List of protein names (optional)
        save_dir: Directory to save figures (optional)
        title_prefix: Title prefix
    """
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. PCC distribution histograms
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=100)
    
    # Protein PCC distribution
    ax1 = axes[0]
    pcc_prot_clean = pcc_protein[~np.isnan(pcc_protein)]
    ax1.hist(pcc_prot_clean, bins=30, color='#7DC9C4', edgecolor='white', alpha=0.8)
    ax1.axvline(x=np.mean(pcc_prot_clean), color='red', linestyle='--', linewidth=1.5, label=f'Mean: {np.mean(pcc_prot_clean):.4f}')
    ax1.axvline(x=np.median(pcc_prot_clean), color='blue', linestyle=':', linewidth=1.5, label=f'Median: {np.median(pcc_prot_clean):.4f}')
    ax1.set_xlabel("Protein-protein PCC")
    ax1.set_ylabel("Count")
    ax1.set_title("Distribution of Protein PCC")
    ax1.legend()
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Cell PCC distribution
    ax2 = axes[1]
    pcc_cell_clean = pcc_cell[~np.isnan(pcc_cell)]
    ax2.hist(pcc_cell_clean, bins=50, color='#E7A365', edgecolor='white', alpha=0.8)
    ax2.axvline(x=np.mean(pcc_cell_clean), color='red', linestyle='--', linewidth=1.5, label=f'Mean: {np.mean(pcc_cell_clean):.4f}')
    ax2.axvline(x=np.median(pcc_cell_clean), color='blue', linestyle=':', linewidth=1.5, label=f'Median: {np.median(pcc_cell_clean):.4f}')
    ax2.set_xlabel("Cell-cell PCC")
    ax2.set_ylabel("Count")
    ax2.set_title("Distribution of Cell PCC")
    ax2.legend()
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.suptitle(f"{title_prefix} - PCC Distribution", fontsize=12, y=1.02)
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(save_dir / "pcc_distribution.png", bbox_inches="tight", dpi=150)
        print(f"Saved: {save_dir / 'pcc_distribution.png'}")
    
    plt.close()
    
    # 2. PCC bar chart per protein
    if protein_names is not None and len(protein_names) == len(pcc_protein):
        fig, ax = plt.subplots(figsize=(12, max(6, len(protein_names) * 0.25)), dpi=100)
        
        # Sort by PCC
        sorted_idx = np.argsort(pcc_protein)[::-1]
        sorted_pcc = pcc_protein[sorted_idx]
        sorted_names = [protein_names[i] for i in sorted_idx]
        
        colors = ['#7DC9C4' if p >= 0 else '#EE9185' for p in sorted_pcc]
        
        y_pos = np.arange(len(sorted_names))
        ax.barh(y_pos, sorted_pcc, color=colors, height=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_names, fontsize=8)
        ax.set_xlabel("Protein-protein PCC")
        ax.set_title(f"{title_prefix} - PCC per Protein")
        ax.axvline(x=0, color='black', linewidth=0.5)
        ax.set_xlim(-1, 1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.invert_yaxis()
        
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(save_dir / "pcc_per_protein.png", bbox_inches="tight", dpi=150)
            print(f"Saved: {save_dir / 'pcc_per_protein.png'}")
        
        plt.close()


def save_evaluation_results(
    results: Dict,
    save_dir: str,
    protein_names: Optional[List[str]] = None,
    title_prefix: str = "scChord"
):
    """
    Save all evaluation results and visualization figures.
    
    Args:
        results: Evaluation results dictionary containing:
            - pcc_protein: np.ndarray [M]
            - pcc_cell: np.ndarray [N]
            - pcc_protein_mean: float
            - pcc_cell_mean: float
            - cmd_cell: float
            - cmd_protein: float
            - rmse: float
        save_dir: Directory to save results
        protein_names: List of protein names (optional)
        title_prefix: Title prefix
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract results
    pcc_protein = results['pcc_protein']
    pcc_cell = results['pcc_cell']
    cmd_cell = results['cmd_cell']
    cmd_protein = results['cmd_protein']
    rmse = results['rmse']
    
    # 1. Comprehensive evaluation summary plot
    plot_evaluation_summary(
        pcc_protein, pcc_cell, cmd_cell, cmd_protein, rmse,
        save_path=save_dir / "evaluation_summary.png",
        title_prefix=title_prefix
    )
    
    # 2. PCC distribution plots
    plot_pcc_distribution(
        pcc_protein, pcc_cell,
        protein_names=protein_names,
        save_dir=save_dir,
        title_prefix=title_prefix
    )
    
    # 3. Individual PCC boxplots
    plot_pcc_boxplot(
        pcc_protein,
        title="Protein-protein PCC",
        xlabel="PCC",
        save_path=save_dir / "pcc_protein_boxplot.png"
    )
    
    plot_pcc_boxplot(
        pcc_cell,
        title="Cell-cell PCC",
        xlabel="PCC",
        save_path=save_dir / "pcc_cell_boxplot.png",
        color='#E7A365'
    )
    
    # 4. Save statistics to CSV
    stats_df = pd.DataFrame({
        'Metric': ['PCC_protein_mean', 'PCC_protein_median', 'PCC_protein_std',
                   'PCC_cell_mean', 'PCC_cell_median', 'PCC_cell_std',
                   'CMD_cell', 'CMD_protein', 'RMSE'],
        'Value': [
            np.nanmean(pcc_protein), np.nanmedian(pcc_protein), np.nanstd(pcc_protein),
            np.nanmean(pcc_cell), np.nanmedian(pcc_cell), np.nanstd(pcc_cell),
            cmd_cell, cmd_protein, rmse
        ]
    })
    stats_df.to_csv(save_dir / "evaluation_stats.csv", index=False)
    print(f"Saved: {save_dir / 'evaluation_stats.csv'}")
    
    # 5. Save PCC per protein
    if protein_names is not None:
        pcc_protein_df = pd.DataFrame({
            'Protein': protein_names,
            'PCC': pcc_protein
        })
        pcc_protein_df = pcc_protein_df.sort_values('PCC', ascending=False)
        pcc_protein_df.to_csv(save_dir / "pcc_per_protein.csv", index=False)
        print(f"Saved: {save_dir / 'pcc_per_protein.csv'}")
    
    print(f"\nAll evaluation results saved to {save_dir}")

