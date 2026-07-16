"""
Paper-quality visualization for Total Quadratic Effort scaling.

This script creates a publication-ready plot matching the paper style:
- Log-log scale
- Navy blue dashed line with circles for Baseline
- Coral red dashed line with squares for Low Noise
- Vertical dashed line at training scale N=30
"""

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# --- Config ---
DATA_DIR = Path(__file__).parent / "figures/noise_experiments/effort"
OUTPUT_DIR = DATA_DIR
CSV_FILE = DATA_DIR / "effort_data_steady.csv"


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
        "lines.linewidth": 2.5,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def plot_effort_paper_style(df):
    """
    Creates a paper-quality Total Quadratic Effort plot.
    """
    setup_paper_style()
    
    fig, ax = plt.subplots(figsize=(9, 5))  # Wider figure matching paper style
    
    # Colors matching the reference image
    color_baseline = "#2c3e50"  # Dark navy blue
    color_lownoise = "#c0392b"  # Coral red
    
    # Separate data by model
    df_baseline = df[df["Model"] == "Baseline"]
    df_lownoise = df[df["Model"] == "Low Noise"]
    
    # Plot Baseline line
    ax.plot(
        df_baseline["Agents"], df_baseline["Sum_Sq"],
        color=color_baseline, linestyle='-', linewidth=2.5,
        marker='o', markersize=8, markerfacecolor=color_baseline,
        markeredgecolor=color_baseline,
        label="Baseline"
    )
    
    # Plot Low Noise line
    ax.plot(
        df_lownoise["Agents"], df_lownoise["Sum_Sq"],
        color=color_lownoise, linestyle='--', linewidth=2.5,
        marker='s', markersize=8, markerfacecolor=color_lownoise,
        markeredgecolor=color_lownoise,
        label="Low Noise"
    )
    
    # Training scale vertical line at N=30
    ax.axvline(
        x=30, color='gray', linestyle='--', linewidth=2, alpha=0.7,
        label=r"Training $N = 30$"
    )
    
    # Set log scale on both axes
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Labels with LaTeX formatting (font sizes from setup_paper_style)
    ax.set_title(r"Total Quadratic Effort ($\sum u_i^2$)", fontsize=24)
    ax.set_xlabel(r"Number of Agents ($N$)", fontsize=24)
    ax.set_ylabel(r"Mean $\sum u_i^2$ (Steady State)", fontsize=24)
    
    # Grid styling
    ax.grid(True, which="both", ls="-", alpha=0.2)
    
    # Complete border (spines)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
    
    # Legend styling - positioned outside the plot on the right
    ax.legend(
        title="Model",
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        framealpha=0.95,
        edgecolor='none',
        fontsize=24
    )
    
    # Make room for legend
    fig.subplots_adjust(right=0.75)
    
    # Save outputs
    pdf_path = OUTPUT_DIR / "effort_scaling_log_steady_paper.pdf"
    png_path = OUTPUT_DIR / "effort_scaling_log_steady_paper.png"
    
    fig.savefig(pdf_path, bbox_inches='tight')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"Saved PDF to: {pdf_path}")
    print(f"Saved PNG to: {png_path}")
    
    plt.close()


if __name__ == "__main__":
    # Load the CSV data
    if not CSV_FILE.exists():
        print(f"Error: CSV file not found at {CSV_FILE}")
        exit(1)
    
    print(f"Loading data from: {CSV_FILE}")
    df = pd.read_csv(CSV_FILE)
    print(f"Loaded {len(df)} data points")
    print(df.head())
    
    # Generate the paper-quality plot
    plot_effort_paper_style(df)
    print("Plot generation complete!")
