"""Plot cumulative regret from saved experiment results."""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set seaborn style for fancier plots
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5)  # Larger text for papers

# Configuration
RESULTS_PATH = "/workspaces/rcgp_ucb/artifacts/forrester_experiment_20250928_125059/forrester_experiment/results.pkl"

# Model name mapping - customize these to rename models in the legend
MODEL_RENAME = {
    "RCGP": "FC-RCGP",
    "GP": "GP",
    "Student-t": "Student-t",
    "A2RCGP": "A2-RCGP",
    "DiagnosticGP": "DiagnosticsGP"
}

# Order for legend display
MODEL_ORDER = ["A2RCGP", "RCGP", "DiagnosticGP", "Student-t", "GP"]

# Color palette
COLORS = {
    "RCGP": "#1f77b4",      # blue
    "GP": "#ff7f0e",         # orange
    "Student-t": "#2ca02c", # green
    "A2RCGP": "#d62728",    # red
    "DiagnosticGP": "#9467bd"  # purple
}


def load_results(pkl_path):
    """Load results from pickle file."""
    with open(pkl_path, 'rb') as f:
        results = pickle.load(f)
    return results


def extract_cumulative_regret_by_model(results, optimal_value):
    """Extract cumulative regret trajectories for each model across seeds.

    Args:
        results: Dictionary of results from pickle file
        optimal_value: Optimal function value

    Returns:
        dict: {model_name: list of cumulative regret arrays (one per seed)}
    """
    model_data = {}

    for key, value in results.items():
        # Parse model name and seed from key (e.g., "RCGP_seed_0")
        parts = key.rsplit('_seed_', 1)
        if len(parts) != 2:
            continue

        model_name = parts[0]
        seed = int(parts[1])

        if model_name not in model_data:
            model_data[model_name] = []

        # Compute cumulative regret from true observed values
        Y_true = np.array(value['Y_true'])

        # Compute instantaneous regret at each iteration
        # We want to maximize, so regret = optimal_value - observed_value
        instant_regret = optimal_value - Y_true

        # Compute cumulative regret
        cumulative_regret = np.cumsum(instant_regret)

        model_data[model_name].append(cumulative_regret)

    return model_data


def plot_cumulative_regret(model_data, model_rename=None, colors=None,
                           model_order=None, save_path=None, figsize=(10, 6)):
    """Plot cumulative regret with mean and standard error bands.

    Args:
        model_data: Dictionary of {model_name: list of cumulative regret arrays}
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

    for model_name, regret_list in models_to_plot:
        # Stack all seeds to compute statistics
        regret_array = np.stack(regret_list)  # Shape: (n_seeds, n_iterations)

        # Compute mean and standard error
        mean_regret = np.mean(regret_array, axis=0)
        std_regret = np.std(regret_array, axis=0)
        se_regret = std_regret / np.sqrt(len(regret_list))

        # Get display name and color
        display_name = model_rename.get(model_name, model_name)
        color = colors.get(model_name, None)

        # Plot mean line
        iterations = np.arange(len(mean_regret))
        line = ax.plot(iterations, mean_regret, label=display_name,
                      linewidth=2.5, color=color)

        # Add standard error band
        if color is None:
            color = line[0].get_color()
        ax.fill_between(iterations,
                        mean_regret - se_regret,
                        mean_regret + se_regret,
                        alpha=0.2, color=color)

    # Styling
    ax.set_xlabel('Iteration', fontsize=16, fontweight='bold')
    ax.set_ylabel('Cumulative Regret', fontsize=16, fontweight='bold')
    # ax.set_title('Cumulative Regret Comparison', fontsize=18, fontweight='bold', pad=20)

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
    """Main function to load and plot results."""
    print(f"Loading results from: {RESULTS_PATH}")
    results = load_results(RESULTS_PATH)
    print(f"Loaded {len(results)} result entries")

    # Get optimal value from first result entry
    first_key = list(results.keys())[0]
    optimal_value = results[first_key]['best_true_value']
    print(f"Optimal value: {optimal_value}")

    # Extract cumulative regret data
    model_data = extract_cumulative_regret_by_model(results, optimal_value)
    print(f"\nFound {len(model_data)} models:")
    for model_name, regret_list in model_data.items():
        print(f"  {model_name}: {len(regret_list)} seeds")

    # Create plot
    print("\nCreating plot...")
    output_path = Path(RESULTS_PATH).parent.parent / "cumulative_regret_plot.png"
    fig, ax = plot_cumulative_regret(
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