import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import EvaluationResult directly to avoid circular imports in utilities
# This is safe as evaluation_result.py has minimal dependencies
from bo_framework.base.evaluation_result import EvaluationResult


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


def reproduce_plot(pickle_path):
    print(f"Loading results from {pickle_path}...")

    try:
        results_dict = load_results(pickle_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Re-organize results by model
    # Structure is { "Model_seed_i": results_list, ... }
    model_results = {}

    # First, find all unique model names
    model_names = set()
    for key in results_dict.keys():
        # Assuming key format "{model_name}_seed_{i}"
        if "_seed_" in key:
            model_name = key.split("_seed_")[0]
            model_names.add(model_name)

    print(f"Found models: {sorted(list(model_names))}")

    # Group results
    for model in model_names:
        model_results[model] = []
        # Find all seeds for this model
        i = 0
        while f"{model}_seed_{i}" in results_dict:
            # results_dict[key] is a dict containing 'all_results'
            res_data = results_dict[f"{model}_seed_{i}"]
            if isinstance(res_data, dict) and "all_results" in res_data:
                model_results[model].append(res_data["all_results"])
            else:
                # Fallback if structure is different (e.g. list directly)
                model_results[model].append(res_data)
            i += 1

    # Calculate optimal value (max true reward across all data)
    optimal_value = -float("inf")
    for model in model_results:
        for seed_results in model_results[model]:
            for res in seed_results:
                if res.y_true > optimal_value:
                    optimal_value = res.y_true

    print(f"Optimal value found: {optimal_value}")

    # Plotting
    plt.figure(figsize=(12, 8))
    colors = {
        "RCGP": "blue",
        "GP": "orange",
        "Student-t": "green",
        "A2RCGP": "red",
        "DiagnosticGP": "purple",
    }

    for model in sorted(list(model_names)):
        seeds_regret = []
        for seed_run in model_results[model]:
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
        stderr_regret = std_regret / np.sqrt(n_seeds)

        iterations = np.arange(1, len(mean_regret) + 1)

        color = colors.get(model, None)
        plt.plot(iterations, mean_regret, label=model, color=color, linewidth=2)
        plt.fill_between(
            iterations,
            mean_regret - stderr_regret,
            mean_regret + stderr_regret,
            color=color,
            alpha=0.2,
        )

    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Cumulative Regret", fontsize=12)
    plt.title(f"Cumulative Regret (Mean ± SE, {n_seeds} seeds)", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # Save to parent directory of pickle file
    parent_dir = Path(pickle_path).parent
    output_path = parent_dir / "reproduced_regret_plot_stderr.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python reproduce_regret_plot.py <path_to_results.pkl>")
        sys.exit(1)

    pickle_path = sys.argv[1]
    reproduce_plot(pickle_path)
