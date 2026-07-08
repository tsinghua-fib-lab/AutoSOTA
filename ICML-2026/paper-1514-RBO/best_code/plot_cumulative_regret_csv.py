"""Plot cumulative regret from CSV results (single seed)."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set seaborn style for fancier plots
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5)  # Larger text for papers

# Configuration
CSV_PATH = "/workspaces/rcgp_ucb/hpt_cifar_lr_wd.csv"

# Model name mapping - customize these to rename models in the legend
MODEL_RENAME = {
    "RCGP": "FC-RCGP",
    "GP": "GP",
    "Student-t": "Student-t",
    "A2RCGP": "A2-RCGP",
    "Diagnostic GP": "DiagnosticsGP"
}

# Order for legend display
MODEL_ORDER = ["A2RCGP", "RCGP", "Diagnostic GP", "Student-t", "GP"]

# Color palette
COLORS = {
    "RCGP": "#1f77b4",      # blue
    "GP": "#ff7f0e",         # orange
    "Student-t": "#2ca02c", # green
    "A2RCGP": "#d62728",    # red
    "Diagnostic GP": "#9467bd"  # purple
}


def load_csv_results(csv_path):
    """Load results from CSV file."""
    df = pd.read_csv(csv_path)
    return df


def extract_cumulative_regret_from_csv(df, optimal_value):
    """Extract cumulative regret trajectories for each model from CSV.

    Args:
        df: DataFrame with columns: model, iteration, true_value
        optimal_value: Optimal function value

    Returns:
        dict: {model_name: cumulative regret array}
    """
    model_data = {}

    for model_name in df['model'].unique():
        # Filter data for this model
        model_df = df[df['model'] == model_name].sort_values('iteration')

        # Get true values
        true_values = model_df['true_value'].values

        # Compute instantaneous regret at each iteration
        # We want to maximize, so regret = optimal_value - observed_value
        instant_regret = optimal_value - true_values

        # Compute cumulative regret
        cumulative_regret = np.cumsum(instant_regret)

        model_data[model_name] = cumulative_regret

    return model_data


def plot_cumulative_regret_single_seed(model_data, model_rename=None, colors=None,
                                       model_order=None, save_path=None, figsize=(10, 6)):
    """Plot cumulative regret for single seed (no error bands).

    Args:
        model_data: Dictionary of {model_name: cumulative regret array}
        model_rename: Dictionary to rename models in legend
        colors: Dictionary of colors for each model
        model_order: List specifying order of models in legend
        save_path: Path to save figure (optional)
        figsize: Figure size tuple
    """
    fig, ax = plt.subplots(figsize=figsize)

    if model_rename is None:
        model_rename = {}
    if colors is None:
        colors = {}

    # Use specified order if provided, otherwise sort alphabetically
    if model_order is not None:
        # Only include models that exist in the data
        models_to_plot = [(name, model_data[name]) for name in model_order if name in model_data]
    else:
        models_to_plot = sorted(model_data.items())

    for model_name, cumulative_regret in models_to_plot:
        # Get display name and color
        display_name = model_rename.get(model_name, model_name)
        color = colors.get(model_name, None)

        # Plot line
        iterations = np.arange(len(cumulative_regret))
        ax.plot(iterations, cumulative_regret, label=display_name,
               linewidth=2.5, color=color)

    # Styling
    ax.set_xlabel('Iteration', fontsize=16, fontweight='bold')
    ax.set_ylabel('Cumulative Regret', fontsize=16, fontweight='bold')

    # Legend with larger text (model names)
    legend = ax.legend(loc='best', frameon=True, fancybox=True, shadow=True,
                      fontsize=18, framealpha=0.95)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('gray')

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)

    # Tight layout
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Plot saved to: {save_path}")

    return fig, ax


def main():
    """Main function to load and plot results from CSV."""
    print(f"Loading results from: {CSV_PATH}")
    df = load_csv_results(CSV_PATH)
    print(f"Loaded {len(df)} rows")

    # Compute optimal value as the maximum true_value observed across all models
    optimal_value = df['true_value'].max()
    print(f"Optimal value (max true_value): {optimal_value}")

    # Show models found
    models = df['model'].unique()
    print(f"\nFound {len(models)} models:")
    for model in models:
        n_iterations = len(df[df['model'] == model])
        print(f"  {model}: {n_iterations} iterations")

    # Extract cumulative regret data
    model_data = extract_cumulative_regret_from_csv(df, optimal_value)

    # Create plot
    print("\nCreating plot...")
    output_path = Path(CSV_PATH).parent / (Path(CSV_PATH).stem + "_cumulative_regret.png")
    fig, ax = plot_cumulative_regret_single_seed(
        model_data,
        model_rename=MODEL_RENAME,
        colors=COLORS,
        model_order=MODEL_ORDER,
        save_path=output_path,
        figsize=(10, 6)
    )

    plt.show()
    print("\nDone!")


if __name__ == "__main__":
    main()
