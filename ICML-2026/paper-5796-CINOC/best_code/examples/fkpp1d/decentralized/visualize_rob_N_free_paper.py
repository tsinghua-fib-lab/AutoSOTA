"""
Paper-Quality Visualization for Noise-Free Robustness Test in FKPP1D Control.
Reads pre-computed CSV data and generates publication-ready plots.

This tests how models trained with different noise levels perform 
in a noise-free evaluation environment (σ_u = 0.0, σ_z = 0.0).
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# --- Path Setup ---
FIGURES_DIR = Path(__file__).parent / "figures" / "noise_experiments" / "decoupled_robustness"
CSV_PATH = FIGURES_DIR / "metrics_Noise-Free.csv"

# Model display names and styling (same as visualize_robust_act_st_paper.py)
MODEL_STYLES = {
    "baseline_clean": {"label": "Baseline", "color": "#2c3e50", "linestyle": "-", "marker": "s"},
    "actuator_only_0p02": {"label": "Actuator 0.02", "color": "#c0392b", "linestyle": "--", "marker": "s"},
    "actuator_only_0p1": {"label": "Actuator 0.1", "color": "#c0392b", "linestyle": "--", "marker": "^"},
    "state_only_0p01": {"label": "Sensor 0.01", "color": "#3498db", "linestyle": "--", "marker": "^"},
    "state_only_0p05": {"label": "Sensor 0.05", "color": "#3498db", "linestyle": "-", "marker": "o"},
}

# Model order for consistent legend
MODEL_ORDER = ["baseline_clean", "actuator_only_0p02", "actuator_only_0p1", "state_only_0p01", "state_only_0p05"]


# --- Paper Style ---
def setup_paper_style():
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 24,
        "axes.labelsize": 24,
        "axes.titlesize": 20,
        "legend.fontsize": 20,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
        "axes.linewidth": 0.8,
        "lines.linewidth": 2.5,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


# --- Plotting ---
def plot_noise_free_robustness(df, output_dir):
    """Generate a paper-quality robustness plot for noise-free evaluation."""
    setup_paper_style()
    
    fig, ax = plt.subplots(figsize=(9, 5))
    
    # Plot each model
    for model_name in MODEL_ORDER:
        if model_name not in df["Model"].unique():
            continue
        
        style = MODEL_STYLES[model_name]
        sub = df[df["Model"] == model_name].sort_values("Agents")
        
        ax.semilogy(
            sub["Agents"], sub["MSE"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2.5,
            marker=style["marker"],
            markersize=8,
            markerfacecolor=style["color"],
            markeredgecolor=style["color"],
            label=style["label"]
        )
    
    # Training scale vertical line at M=30
    ax.axvline(x=30, color="gray", linestyle="--", linewidth=2, alpha=0.7, label=r"Training Scale ($M$=30)")
    
    # Title
    ax.set_title(r"Noise-Free Evaluation ($\sigma_u = 0.0, \sigma_z = 0.0$)", fontsize=20)
    
    # Axis labels
    ax.set_xlabel("Deployment Agent Count", fontsize=24)
    ax.set_ylabel("Final MSE", fontsize=24)
    
    # X-axis ticks
    agent_counts = sorted(df["Agents"].unique())
    ax.set_xticks(agent_counts)
    ax.set_xticklabels([str(a) for a in agent_counts])
    
    # Grid styling
    ax.grid(True, which="both", ls="-", alpha=0.2)
    
    # Complete border (spines)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
    
    # Legend outside the plot
    ax.legend(
        title="Policy Type",
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        framealpha=0.95,
        edgecolor='none',
        fontsize=20
    )
    
    # Make room for legend
    fig.subplots_adjust(right=0.68)
    
    # Save outputs
    pdf_path = output_dir / "paper_robustness_Noise_Free.pdf"
    png_path = output_dir / "paper_robustness_Noise_Free.png"
    
    fig.savefig(pdf_path, bbox_inches='tight')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {pdf_path.name} and {png_path.name}")
    
    plt.close()


def main():
    print("=" * 60)
    print("FKPP1D Noise-Free Robustness Paper Visualization")
    print("=" * 60)
    
    # Load CSV data
    if not CSV_PATH.exists():
        print(f"Error: CSV not found at {CSV_PATH}")
        return
    
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} rows from {CSV_PATH}")
    print(f"Models: {sorted(df['Model'].unique())}")
    print(f"Agent counts: {sorted(df['Agents'].unique())}")
    
    # Generate plot (no filtering needed, CSV only contains Noise-Free data)
    plot_noise_free_robustness(df, FIGURES_DIR)
    
    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
