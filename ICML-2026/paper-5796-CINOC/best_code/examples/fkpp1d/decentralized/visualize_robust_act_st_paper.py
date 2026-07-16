"""
Paper-Quality Visualization for Robustness Analysis in FKPP1D Control.
Reads pre-computed CSV data and generates publication-ready plots.

Plots created:
- Robustness in Actuator Low Env
- Robustness in Actuator Mid Env
- Robustness in State Low Env
- Robustness in State Mid Env
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# --- Path Setup ---
FIGURES_DIR = Path(__file__).parent / "figures" / "noise_experiments" / "decoupled_robustness"

# CSV files for the 4 plots
CSV_FILES = {
    "Actuator_Low": FIGURES_DIR / "metrics_Actuator_Low.csv",
    "State_Low": FIGURES_DIR / "metrics_State_Low.csv",
    "Actuator_Mid": FIGURES_DIR / "metrics_Actuator_Mid.csv",
    "State_Mid": FIGURES_DIR / "metrics_State_Mid.csv",
}

# Scenario display names and noise parameters
SCENARIO_INFO = {
    "Actuator_Low": {"title": "Low Actuator Noise", "sigma_u": 0.02, "sigma_z": 0.0},
    "State_Low": {"title": "Low State  Noise", "sigma_u": 0.0, "sigma_z": 0.01},
    "Actuator_Mid": {"title": "Mid Actuator Noise", "sigma_u": 0.1, "sigma_z": 0.0},
    "State_Mid": {"title": "Mid State Noise", "sigma_u": 0.0, "sigma_z": 0.05},
}

# Model display names and styling
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
        "axes.titlesize": 24,
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
def plot_robustness_paper(scenario_name, df, output_dir):
    """Generate a paper-quality robustness plot for a given scenario."""
    setup_paper_style()
    
    info = SCENARIO_INFO[scenario_name]
    
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
    
    # Title with noise parameters
    title = f"{info['title']} " + r"($\sigma_u = " + f"{info['sigma_u']}, " + r"\sigma_z = " + f"{info['sigma_z']})$"
    ax.set_title(title, fontsize=24)
    
    # Axis labels
    ax.set_xlabel("Deployment Agent Count", fontsize=24)
    ax.set_ylabel("Final MSE Error", fontsize=24)
    
    # X-axis ticks
    ax.set_xticks([20, 30, 40, 60, 80, 100])
    ax.set_xticklabels(["20", "30", "40", "60", "80", "100"])
    
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
    pdf_path = output_dir / f"paper_robustness_{scenario_name}.pdf"
    png_path = output_dir / f"paper_robustness_{scenario_name}.png"
    
    fig.savefig(pdf_path, bbox_inches='tight')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {pdf_path.name} and {png_path.name}")
    
    plt.close()


def main():
    print("=" * 60)
    print("FKPP1D Robustness Paper Visualization")
    print("=" * 60)
    
    # Process each scenario
    for scenario_name, csv_path in CSV_FILES.items():
        if not csv_path.exists():
            print(f"Warning: CSV not found at {csv_path}, skipping...")
            continue
        
        print(f"\nProcessing {scenario_name}...")
        df = pd.read_csv(csv_path)
        print(f"  Loaded {len(df)} rows")
        print(f"  Models: {sorted(df['Model'].unique())}")
        print(f"  Agent counts: {sorted(df['Agents'].unique())}")
        
        # Generate plot
        plot_robustness_paper(scenario_name, df, FIGURES_DIR)
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
