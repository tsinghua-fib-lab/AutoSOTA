#!/usr/bin/env python3
"""
Combine and plot results from multiple experiment runs.

This script allows you to:
1. Load results from one or more pickle files
2. Combine seeds from different runs (assuming identical experiment setup)
3. Generate regret plots with mean ± standard error across all seeds

Usage:
    python combine_and_plot_results.py <path_to_results1.pkl> [<path_to_results2.pkl> ...]

Example:
    # Single file (same as reproduce_regret_plot.py)
    python combine_and_plot_results.py artifacts/run1/results.pkl

    # Multiple files (combines seeds from different runs)
    python combine_and_plot_results.py artifacts/run1/results.pkl artifacts/run2/results.pkl
"""

import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

# Import EvaluationResult directly to avoid circular imports
from bo_framework.base.evaluation_result import EvaluationResult

# =============================================================================
# Plotting configuration
# =============================================================================
# Set seaborn style for fancier plots
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5)  # Larger text for papers

# Model name mapping - customize these to rename models in the legend
MODEL_RENAME = {
    "RCGP": "FC-RCGP",
    "GP": "GP",
    "Student-t": "Student-t",
    "A2RCGP": "A2-RCGP",
    "DiagnosticGP": "DiagnosticsGP",
    "Diagnostic GP": "DiagnosticsGP",
}

# Order for legend display
MODEL_ORDER = ["A2RCGP", "RCGP", "DiagnosticGP", "Student-t", "GP"]

# Color palette
COLORS = {
    "RCGP": "#1f77b4",  # blue
    "GP": "#ff7f0e",  # orange
    "Student-t": "#2ca02c",  # green
    "A2RCGP": "#d62728",  # red
    "DiagnosticGP": "#9467bd",  # purple
    "Diagnostic GP": "#9467bd",
}


def load_results(pickle_path):
    """Load results.pkl from the given path."""
    path = Path(pickle_path)

    if not path.exists():
        raise FileNotFoundError(f"Pickle file not found: {path}")

    with open(path, "rb") as f:
        return pickle.load(f)


def compute_cumulative_regret(results_list, optimal_value):
    """Compute cumulative regret for a list of EvaluationResult objects."""
    y_true = np.array([r.y_true for r in results_list])
    instant_regret = optimal_value - y_true
    return np.cumsum(instant_regret)


def combine_results_from_files(pickle_paths):
    """
    Combine results from multiple pickle files.

    Args:
        pickle_paths: List of paths to pickle files

    Returns:
        combined_results: Dict mapping model names to list of seed results
        optimal_value: Maximum true value across all files
    """
    combined_results = defaultdict(list)
    all_seed_keys = set()

    print(f"Loading results from {len(pickle_paths)} file(s)...")

    for i, pickle_path in enumerate(pickle_paths):
        print(f"  Loading {pickle_path}...")
        results_dict = load_results(pickle_path)

        # Track all seed keys to identify models
        all_seed_keys.update(results_dict.keys())

        # Group by model name
        for key in results_dict.keys():
            # Assuming key format "{model_name}_seed_{i}"
            if "_seed_" in key:
                model_name = key.split("_seed_")[0]
                res_data = results_dict[key]

                # Extract all_results list
                if isinstance(res_data, dict) and "all_results" in res_data:
                    combined_results[model_name].append(res_data["all_results"])
                else:
                    # Fallback if structure is different
                    combined_results[model_name].append(res_data)

    # Find unique model names
    model_names = set()
    for key in all_seed_keys:
        if "_seed_" in key:
            model_name = key.split("_seed_")[0]
            model_names.add(model_name)

    print(f"Found models: {sorted(list(model_names))}")
    print(f"Total seeds per model: {len(combined_results[list(model_names)[0]])}")

    # Calculate optimal value (max true reward across all data)
    optimal_value = -float("inf")
    for model_results_list in combined_results.values():
        for seed_results in model_results_list:
            for res in seed_results:
                if res.y_true > optimal_value:
                    optimal_value = res.y_true

    print(f"Optimal value found: {optimal_value}")

    return dict(combined_results), optimal_value


def plot_combined_results(
    combined_results, optimal_value, output_path, use_stderr=True
):
    """
    Plot combined regret results with mean ± SE or std.

    Args:
        combined_results: Dict mapping model names to list of seed results
        optimal_value: Optimal value for regret calculation
        output_path: Path to save the plot
        use_stderr: If True, use standard error; if False, use standard deviation
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Determine models to plot based on MODEL_ORDER
    models_to_plot = []
    # First add models in the specified order
    for model_name in MODEL_ORDER:
        if model_name in combined_results:
            models_to_plot.append(model_name)

    # Then add any other models found but not in the order list
    for model_name in sorted(combined_results.keys()):
        if model_name not in models_to_plot:
            models_to_plot.append(model_name)

    for model in models_to_plot:
        seeds_regret = []
        for seed_run in combined_results[model]:
            # Compute cumulative regret for this seed
            regret = compute_cumulative_regret(seed_run, optimal_value)
            seeds_regret.append(regret)

        if not seeds_regret:
            continue

        # Convert to numpy array: shape (n_seeds, n_iterations)
        # Ensure all runs have same length (truncate to min length if needed)
        min_len = min(len(r) for r in seeds_regret)
        seeds_regret = np.array([r[:min_len] for r in seeds_regret])

        n_seeds = seeds_regret.shape[0]
        mean_regret = np.mean(seeds_regret, axis=0)
        std_regret = np.std(seeds_regret, axis=0)

        if use_stderr:
            error_regret = std_regret / np.sqrt(n_seeds)
            # error_label = "SE"
        else:
            error_regret = std_regret
            # error_label = "Std"

        iterations = np.arange(len(mean_regret))

        display_name = MODEL_RENAME.get(model, model)
        color = COLORS.get(model, None)

        line = ax.plot(
            iterations,
            mean_regret,
            label=display_name,
            color=color,
            linewidth=2.5,
        )

        if color is None:
            color = line[0].get_color()

        ax.fill_between(
            iterations,
            mean_regret - error_regret,
            mean_regret + error_regret,
            color=color,
            alpha=0.2,
        )

    # Styling
    ax.set_xlabel("Iteration", fontsize=16, fontweight="bold")
    ax.set_ylabel("Cumulative Regret", fontsize=16, fontweight="bold")

    # Legend with larger text
    legend = ax.legend(
        loc="best",
        frameon=True,
        fancybox=True,
        shadow=True,
        fontsize=18,
        framealpha=0.95,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("gray")

    # Grid
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.8)
    ax.set_axisbelow(True)

    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"\nPlot saved to: {output_path}")
    plt.close()


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python combine_and_plot_results.py <path_to_results1.pkl> [<path_to_results2.pkl> ...]"
        )
        print("\nExamples:")
        print("  # Single file")
        print("  python combine_and_plot_results.py artifacts/run1/results.pkl")
        print("\n  # Multiple files (combines seeds)")
        print(
            "  python combine_and_plot_results.py artifacts/run1/results.pkl artifacts/run2/results.pkl"
        )
        sys.exit(1)

    pickle_paths = sys.argv[1:]

    # Validate all paths exist
    for path in pickle_paths:
        if not Path(path).exists():
            print(f"Error: File not found: {path}")
            sys.exit(1)

    # Combine results from all files
    combined_results, optimal_value = combine_results_from_files(pickle_paths)

    # Determine output path (same directory as first file)
    first_file_dir = Path(pickle_paths[0]).parent

    # Generate plots with both SE and Std
    print("\nGenerating plots...")

    # Plot with standard error
    output_path_se = first_file_dir / "combined_regret_plot_stderr.png"
    plot_combined_results(
        combined_results, optimal_value, output_path_se, use_stderr=True
    )

    # Plot with standard deviation
    output_path_std = first_file_dir / "combined_regret_plot_std.png"
    plot_combined_results(
        combined_results, optimal_value, output_path_std, use_stderr=False
    )

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Files combined: {len(pickle_paths)}")
    print(f"Models found: {', '.join(sorted(combined_results.keys()))}")
    print(f"Seeds per model: {len(combined_results[list(combined_results.keys())[0]])}")
    print(f"Optimal value: {optimal_value:.4f}")
    print("=" * 80)

    print_regret_table(combined_results, optimal_value)


def print_regret_table(combined_results, optimal_value):
    """
    Print a table of mean last cumulative regret and standard error.

    Args:
        combined_results: Dict mapping model names to list of seed results
        optimal_value: Optimal value for regret calculation
    """
    print("\n" + "=" * 60)
    print(f"{'Model':<20} | {'Mean Last Regret':<20} | {'Std Error':<15}")
    print("-" * 60)

    for model in sorted(combined_results.keys()):
        seeds_regret = []
        for seed_run in combined_results[model]:
            # Compute cumulative regret for this seed
            regret = compute_cumulative_regret(seed_run, optimal_value)
            seeds_regret.append(regret[-1])  # Take the last value

        if not seeds_regret:
            continue

        seeds_regret = np.array(seeds_regret)
        n_seeds = len(seeds_regret)
        mean_last_regret = np.mean(seeds_regret)
        std_last_regret = np.std(seeds_regret)
        stderr_last_regret = std_last_regret / np.sqrt(n_seeds)

        print(f"{model:<20} | {mean_last_regret:<20.4f} | {stderr_last_regret:<15.4f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
