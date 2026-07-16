"""
Paper-Quality Visualization for Lambda-Effort Analysis in FKPP1D Control.
Reads pre-computed CSV data and generates publication-ready plots.
Matching the style from visualization_updates branch.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.ticker import ScalarFormatter

# --- Path Setup ---
FIGURES_DIR = Path(__file__).parent / "figures" / "conjecture"
CSV_PATH = FIGURES_DIR / "conjecture_data_windowed.csv"

# --- Paper Style ---
def setup_paper_style():
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 24,
        "axes.labelsize": 24,
        "axes.titlesize": 24,
        "legend.fontsize": 24,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.5,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })

# --- Plotting ---
def plot_lambda_effort_paper(df):
    """Generate paper-quality plots for lambda-effort analysis."""
    setup_paper_style()
    
    # Colors matching the reference image (7 lambda values)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    
    # Sort lambda values for consistent ordering
    lambda_vals = sorted(df['lambda'].unique())
    
    # --- Plot 1: Tracking MSE ---
    fig1, ax1 = plt.subplots(figsize=(7, 4.5))
    for i, l in enumerate(lambda_vals):
        sub = df[df['lambda'] == l].sort_values('n_agents')
        ax1.semilogy(sub['n_agents'], sub['mse'], marker='s', markersize=5, 
                     label=f'$\\lambda_u = {l}$', color=colors[i % len(colors)], linewidth=1.5)
    
    ax1.set_title("Zero-Shot Scalability: Tracking MSE", fontsize=13, fontweight='bold')
    ax1.set_xlabel("Number of Agents ($M$)", fontsize=11)
    ax1.set_ylabel("Final $L^2$ Error", fontsize=11)
    ax1.set_xscale('log')
    ax1.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), framealpha=0.95, edgecolor='none')
    ax1.grid(True, which="both", ls="-", alpha=0.2)
    # Complete border
    for spine in ax1.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
    fig1.subplots_adjust(right=0.75)  # Make room for legend
    fig1.savefig(FIGURES_DIR / "paper_scaling_mse.pdf", bbox_inches='tight')
    fig1.savefig(FIGURES_DIR / "paper_scaling_mse.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: paper_scaling_mse.pdf")

    # --- Plot 2: Squared Effort ---
    fig2, ax2 = plt.subplots(figsize=(9, 5))  # Wider figure
    for i, l in enumerate(lambda_vals):
        sub = df[df['lambda'] == l].sort_values('n_agents')
        ax2.loglog(sub['n_agents'], sub['total_effort_sq'], marker='s', markersize=5,
                   label=f'$\\lambda_u = {l}$', color=colors[i % len(colors)], linewidth=1.5)
    
    ax2.set_title(r"Steady-State Effort: $\sum u_i^2$", fontsize=24)
    ax2.set_xlabel("Number of Agents ($M$)", fontsize=24)
    ax2.set_ylabel(r"Mean $\sum u_i^2$", fontsize=24)
    # Fix x-axis ticks to avoid overlap
    ax2.set_xticks([15, 20, 30, 40, 50, 60])
    ax2.set_xticklabels(['15', '20', '30', '40', '50', '60'])
    ax2.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), framealpha=0.95, edgecolor='none', fontsize=24)
    ax2.grid(True, which="both", ls="-", alpha=0.2)
    # Complete border
    for spine in ax2.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
    fig2.subplots_adjust(right=0.72)  # Make room for legend
    fig2.savefig(FIGURES_DIR / "paper_scaling_effort_sq.pdf", bbox_inches='tight')
    fig2.savefig(FIGURES_DIR / "paper_scaling_effort_sq.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: paper_scaling_effort_sq.pdf")

    # --- Plot 3: Absolute Effort ---
    fig3, ax3 = plt.subplots(figsize=(7, 4.5))
    for i, l in enumerate(lambda_vals):
        sub = df[df['lambda'] == l].sort_values('n_agents')
        ax3.loglog(sub['n_agents'], sub['total_effort_abs'], marker='s', markersize=5,
                   label=f'$\\lambda_u = {l}$', color=colors[i % len(colors)], linewidth=1.5)
    
    ax3.set_title(r"Steady-State Effort: $\sum |u_i|$", fontsize=13, fontweight='bold')
    ax3.set_xlabel("Number of Agents ($M$)", fontsize=11)
    ax3.set_ylabel(r"Mean $\sum |u_i|$", fontsize=11)
    ax3.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), framealpha=0.95, edgecolor='none')
    ax3.grid(True, which="both", ls="-", alpha=0.2)
    # Complete border
    for spine in ax3.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
    fig3.subplots_adjust(right=0.75)  # Make room for legend
    fig3.savefig(FIGURES_DIR / "paper_scaling_effort_abs.pdf", bbox_inches='tight')
    fig3.savefig(FIGURES_DIR / "paper_scaling_effort_abs.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: paper_scaling_effort_abs.pdf")
    
    plt.close('all')


def main():
    print("=" * 60)
    print("FKPP1D Lambda-Effort Paper Visualization")
    print("=" * 60)
    
    # Load CSV data
    if not CSV_PATH.exists():
        print(f"Error: CSV not found at {CSV_PATH}")
        return
    
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} rows from {CSV_PATH}")
    print(f"Lambda values: {sorted(df['lambda'].unique())}")
    print(f"Agent counts: {sorted(df['n_agents'].unique())}")
    
    # Generate plots
    plot_lambda_effort_paper(df)
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
