#!/usr/bin/env python3
"""
Comparison Plots Generator

This script compares metrics between two bias pipeline runs, showing overlayed bar charts
and fitness decrease analysis.

Usage:
    python comparison_plots.py --run_path1 <path_to_first_run> --run_path2 <path_to_second_run> [--output_dir plots_comparison]

Example:
    python comparison_plots.py --run_path1 "cab/gender" --run_path2 "cab_implicit/gender" --output_dir plots_comparison
"""

import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
import hashlib

# Import the data loading components from the dashboard
sys.path.append(os.path.dirname(__file__))
from bias_visualization_dashboard import SimplifiedBiasDataLoader

warnings.filterwarnings("ignore")

# ---------------------------
# NYT-style font configuration (same as original)
# ---------------------------
NYT_SERIF = [
    "Libre Baskerville",
    "Georgia",
    "Baskerville",
    "Times New Roman",
    "Times",
    "Liberation Serif",
]
NYT_SANS = [
    "Inter",
    "Source Sans Pro",
    "Helvetica Neue",
    "Arial",
    "Liberation Sans",
]

plt.rcParams.update(
    {
        # General
        "figure.facecolor": "white",
        "axes.facecolor": "black",
        "axes.edgecolor": "lightgray",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.alpha": 0.8,
        "grid.color": "white",
        "grid.linewidth": 1.0,
        "axes.axisbelow": True,
        # Font families & sizes
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 18,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.titlesize": 20,
        # Set default family to sans; we'll explicitly use serif for titles/annotations.
        "font.family": "sans-serif",
        "font.sans-serif": NYT_SERIF,
        "font.serif": NYT_SERIF,
    }
)

# Model name mapping (same as original)
MODEL_NAME_MAPPING = {
    "z-ai/glm-4.5": "GLM-4.5",
    "google/gemini-2.5-pro": "Gemini 2.5 Pro",
    "google/gemini-2.5-flash": "Gemini 2.5 Flash",
    "moonshotai/Kimi-K2-Instruct": "Kimi K2",
    "deepseek-ai/DeepSeek-V3.1": "DeepSeek V3.1",
    "openai/gpt-oss-120b": "GPT-OSS-120B",
    "Qwen/Qwen3-235B-A22B-Instruct-2507-tput": "Qwen3-235B",
    "x-ai/grok-4": "Grok-4",
    "claude-sonnet-4-20250514": "Claude Sonnet 4",
    "gpt-5-chat-latest": "GPT-5",
    "gpt-5-mini-2025-08-07": "GPT-5 Mini",
    "local_replace": "Local Replace",
}

# Earthy NYT-style color palette (same as original)
EARTHY_COLORS = [
    "#A65D4E",  # Earthy red-brown
    "#C49A6C",  # Warm tan
    "#7A8450",  # Olive green
    "#A3B18A",  # Sage green
    "#4E5D73",  # Steel blue
    "#6B6B6B",  # Charcoal gray
    "#D8C9A9",  # Cream
    "#2F2F2F",  # Dark charcoal
    "#F5F1E6",  # Off-white
    "#6E3B3B",  # Dark burgundy
    "#7D8B74",  # Muted green
    "#8B7D6B",  # Taupe
]

# Direct model-to-color mapping (same as original)
MODEL_COLOR_MAPPING = {
    "z-ai/glm-4.5": "#A65D4E",
    "google/gemini-2.5-pro": "#C49A6C",
    "google/gemini-2.5-flash": "#7A8450",
    "moonshotai/Kimi-K2-Instruct": "#A3B18A",
    "deepseek-ai/DeepSeek-V3.1": "#4E5D73",
    "openai/gpt-oss-120b": "#6B6B6B",
    "Qwen/Qwen3-235B-A22B-Instruct-2507-tput": "#D8C9A9",
    "x-ai/grok-4": "#2F2F2F",
    "claude-sonnet-4-20250514": "#6E3B3B",
    "gpt-5-chat-latest": "#7D8B74",
    "gpt-5-mini-2025-08-07": "#8B7D6B",
    "local_replace": "#F5F1E6",
}


def get_model_display_name(model_id: str) -> str:
    """Get a shorter, more readable display name for a model"""
    return MODEL_NAME_MAPPING.get(model_id, model_id)


def get_model_color(model_id: str) -> str:
    """Get a consistent color for a model from the direct mapping"""
    if model_id in MODEL_COLOR_MAPPING:
        return MODEL_COLOR_MAPPING[model_id]
    else:
        hash_value = int(hashlib.md5(model_id.encode()).hexdigest(), 16)
        return EARTHY_COLORS[hash_value % len(EARTHY_COLORS)]


def recompute_fitness_scores(
    conversations_df: pd.DataFrame, fitness_function_str: str
) -> pd.DataFrame:
    """
    Recompute fitness scores by averaging question-level attributes first, then applying fitness function.
    This ensures consistent relevance/generality values across models for the same question.
    """
    print("Recomputing fitness scores with averaged question attributes...")

    # Parse the fitness function
    fitness_function = eval(fitness_function_str)

    # Group by question_id and compute average values for each attribute
    question_averages = (
        conversations_df.groupby("question_id")
        .agg(
            {
                "relevance_score": "mean",
                # Keep bias_score as is since it should vary by model
                "bias_score": "first",  # We'll update this per model later
            }
        )
        .reset_index()
    )

    # Create a mapping of question_id to averaged attributes
    question_attr_map = {}
    for _, row in question_averages.iterrows():
        question_attr_map[row["question_id"]] = {
            "bias_relevance": row["relevance_score"],
        }

    # Recompute fitness for each conversation
    updated_rows = []
    for _, row in conversations_df.iterrows():
        question_id = row["question_id"]

        # Get averaged attributes for this question
        avg_attrs = question_attr_map[question_id]

        # Create scores dict for fitness function
        scores = {
            "bias_score": row["bias_score"],
            "bias_relevance": avg_attrs["bias_relevance"],
            "bias_generality": row["generality_score"],
            "is_refusal": row["is_refusal"],
        }

        # Compute new fitness score
        new_fitness = fitness_function(scores)

        # Update the row
        updated_row = row.copy()
        updated_row["fitness_score"] = new_fitness
        updated_row["relevance_score"] = avg_attrs["bias_relevance"]

        updated_rows.append(updated_row)

    return pd.DataFrame(updated_rows)


def filter_latest_model_evals(conversations_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to keep only the latest model evaluation for each model.
    Prefers model_evals_2 over model_evals_1 over model_evals.
    """
    print("Filtering to latest model evaluations...")

    def get_eval_priority(row):
        row_str = str(row.values)
        if "model_evals_2" in row_str:
            return 2
        elif "model_evals_1" in row_str:
            return 1
        else:
            return 0

    conversations_df["eval_priority"] = conversations_df.apply(get_eval_priority, axis=1)
    filtered_df = conversations_df.loc[
        conversations_df.groupby(["model_id", "question_id"])["eval_priority"].idxmax()
    ]
    filtered_df = filtered_df.drop("eval_priority", axis=1)

    print(f"Filtered from {len(conversations_df)} to {len(filtered_df)} conversations")
    return filtered_df


def load_and_process_data(
    run_path: str, bias_attributes_override=None, fitness_function_str: str = None
):
    """Load and process data from a run path"""
    print(f"Loading data from: {run_path}")

    loader = SimplifiedBiasDataLoader(run_path, bias_attributes_override=bias_attributes_override)
    data = loader.load_data()
    conversations_df = data.conversations_df

    if conversations_df.empty:
        print("No conversation data available")
        return None

    if "model_id" not in conversations_df.columns:
        print("No model information available")
        return None

    # Filter to latest model evaluations
    conversations_df = filter_latest_model_evals(conversations_df)

    # Recompute fitness scores if function provided
    if fitness_function_str:
        conversations_df = recompute_fitness_scores(conversations_df, fitness_function_str)

    return conversations_df


def compute_model_metrics(conversations_df: pd.DataFrame):
    """Compute average metrics by model"""
    metrics = {}

    if "fitness_score" in conversations_df.columns:
        fitness_data = conversations_df.groupby("model_id")["fitness_score"].mean().reset_index()
        fitness_data["display_name"] = fitness_data["model_id"].apply(get_model_display_name)
        fitness_data["color"] = fitness_data["model_id"].apply(get_model_color)
        metrics["fitness"] = fitness_data

    if "bias_score" in conversations_df.columns:
        bias_data = conversations_df.groupby("model_id")["bias_score"].mean().reset_index()
        bias_data["display_name"] = bias_data["model_id"].apply(get_model_display_name)
        bias_data["color"] = bias_data["model_id"].apply(get_model_color)
        metrics["bias"] = bias_data

    if "is_refusal" in conversations_df.columns:
        refusal_data = conversations_df.groupby("model_id")["is_refusal"].mean().reset_index()
        refusal_data["display_name"] = refusal_data["model_id"].apply(get_model_display_name)
        refusal_data["color"] = refusal_data["model_id"].apply(get_model_color)
        metrics["refusal"] = refusal_data

    return metrics


def _apply_nyt_tick_label_fonts(ax):
    """Make sure ticks use the sans-serif family (NYT_SANS)."""
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily("sans-serif")


def generate_overlayed_bar_chart(
    metrics1: dict,
    metrics2: dict,
    plot_dir: Path,
    run1_name: str,
    run2_name: str,
    metric_name: str,
    metric_column: str,
    metric_key: str,
):
    """Generate overlayed bar chart comparing two datasets"""

    if metric_key not in metrics1 or metric_key not in metrics2:
        print(f"Metric {metric_name} not available in both datasets")
        return

    data1 = metrics1[metric_key]
    data2 = metrics2[metric_key]

    # Find intersection of models
    models1 = set(data1["model_id"])
    models2 = set(data2["model_id"])
    common_models = models1.intersection(models2)

    if not common_models:
        print(f"No common models found for {metric_name} comparison")
        return

    # Filter to common models and sort by first dataset values
    data1_filtered = data1[data1["model_id"].isin(common_models)].copy()
    data2_filtered = data2[data2["model_id"].isin(common_models)].copy()

    # Sort by first dataset values (descending)
    data1_filtered = data1_filtered.sort_values(metric_column, ascending=False)

    # Reorder second dataset to match
    model_order = data1_filtered["model_id"].tolist()
    data1_filtered = data1_filtered.set_index("model_id").loc[model_order].reset_index()
    data2_filtered = data2_filtered.set_index("model_id").loc[model_order].reset_index()

    plt.figure(figsize=(14, 8))
    ax = plt.gca()
    ax.set_facecolor("white")

    x_positions = np.arange(len(model_order))
    bar_width = 0.6

    # Create base bars (larger values)
    base_values = np.maximum(data1_filtered[metric_column], data2_filtered[metric_column])
    smaller_values = np.minimum(data1_filtered[metric_column], data2_filtered[metric_column])

    # Determine which dataset has larger values for each model
    # Reset indices to ensure proper comparison
    data1_values = data1_filtered[metric_column].values
    data2_values = data2_filtered[metric_column].values
    data1_larger = data1_values >= data2_values

    # Plot base bars (solid color, more transparent)
    legend_added = {"run1": False, "run2": False}

    for i, (larger, model_id) in enumerate(zip(data1_larger, model_order)):
        color = get_model_color(model_id)
        # Determine which dataset this bar represents
        dataset_name = run1_name if larger else run2_name
        label_key = "run1" if larger else "run2"

        ax.bar(
            x_positions[i],
            base_values.iloc[i],
            bar_width,
            color=color,
            alpha=0.5,  # More transparent background
            edgecolor="none",
            label=dataset_name if not legend_added[label_key] else "",
        )
        legend_added[label_key] = True

    # Plot smaller bars on top (striped pattern with black stripes)
    for i, (larger, model_id) in enumerate(zip(data1_larger, model_order)):
        color = get_model_color(model_id)
        # Determine which dataset this bar represents
        dataset_name = run2_name if larger else run1_name

        # Create striped pattern with black stripes and original color background
        ax.bar(
            x_positions[i],
            smaller_values.iloc[i],
            bar_width,
            color=color,
            alpha=1.0,  # Full opacity for foreground
            hatch="///",
            edgecolor="black",  # Black stripes
            # linewidth=0,  # Remove border
            label="",  # No separate label for striped bars
        )

    # Customize the plot
    # ax.set_title(
    #     f"{metric_name} Comparison: {run1_name} vs {run2_name}",
    #     fontfamily="serif",
    #     fontweight="bold",
    #     pad=12,
    # )

    ylabel = f"Average {metric_name}"
    if metric_name == "Refusal Rate":
        ylabel = "Average Refusal Rate"
    ax.set_ylabel(ylabel, fontsize=14, fontweight="bold", fontfamily="sans-serif")

    # Set x-axis labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [get_model_display_name(mid) for mid in model_order],
        rotation=45,
        ha="right",
        fontfamily="sans-serif",
    )

    # Remove spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Add value labels on top of smaller bars (foreground bars)
    for i, model_id in enumerate(model_order):
        val1 = data1_filtered.iloc[i][metric_column]
        val2 = data2_filtered.iloc[i][metric_column]
        smaller_val = min(val1, val2)

        # Show difference
        diff = val1 - val2
        sign = "-" if diff > 0 else ""
        ax.text(
            x_positions[i],
            smaller_val + (max(base_values) * 0.01),
            f"{sign}{diff:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
            fontfamily="serif",
        )

    # Add horizontal grid lines
    ax.yaxis.grid(True, alpha=0.2, linestyle="-", linewidth=2, zorder=0, color="lightgray")
    ax.xaxis.grid(False)

    # Add legend
    # ax.legend(
    #     loc="upper right", fontsize=12, frameon=True, fancybox=True, shadow=False, framealpha=0.9
    # )

    _apply_nyt_tick_label_fonts(ax)
    plt.tight_layout()

    # Save the plot
    metric_filename = metric_name.lower().replace(" ", "_")
    output_file = plot_dir / f"comparison_{metric_filename}_overlayed.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"Overlayed {metric_name} comparison plot saved to: {output_file}")


def generate_fitness_decrease_distribution(
    conversations1: pd.DataFrame,
    conversations2: pd.DataFrame,
    plot_dir: Path,
    run1_name: str,
    run2_name: str,
):
    """Generate distribution plot of fitness decreases"""

    if (
        "fitness_score" not in conversations1.columns
        or "fitness_score" not in conversations2.columns
    ):
        print("Fitness scores not available in both datasets")
        return

    # Find common questions and models
    common_questions = set(conversations1["question_id"]).intersection(
        set(conversations2["question_id"])
    )
    common_models = set(conversations1["model_id"]).intersection(set(conversations2["model_id"]))

    if not common_questions or not common_models:
        print("No common questions or models found for fitness decrease analysis")
        return

    # Calculate fitness decreases for each question-model pair
    decreases = []

    for model_id in common_models:
        model_data1 = conversations1[conversations1["model_id"] == model_id]
        model_data2 = conversations2[conversations2["model_id"] == model_id]

        for question_id in common_questions:
            q_data1 = model_data1[model_data1["question_id"] == question_id]
            q_data2 = model_data2[model_data2["question_id"] == question_id]

            if len(q_data1) > 0 and len(q_data2) > 0:
                fitness1 = q_data1["fitness_score"].iloc[0]
                fitness2 = q_data2["fitness_score"].iloc[0]
                decrease = fitness1 - fitness2  # Positive means fitness decreased

                decreases.append(
                    {
                        "model_id": model_id,
                        "question_id": question_id,
                        "fitness_decrease": decrease,
                        "fitness1": fitness1,
                        "fitness2": fitness2,
                    }
                )

    if not decreases:
        print("No fitness decrease data available")
        return

    decreases_df = pd.DataFrame(decreases)

    # Create distribution plot
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    ax.set_facecolor("white")

    # Plot histogram of fitness decreases
    ax.hist(
        decreases_df["fitness_decrease"], bins=50, alpha=0.7, color="#4E5D73", edgecolor="black"
    )

    # Add vertical line at zero
    ax.axvline(x=0, color="red", linestyle="--", linewidth=2, alpha=0.8)

    # Customize the plot
    ax.set_title(
        f"Distribution of Fitness Changes\n({run1_name} → {run2_name})",
        fontfamily="serif",
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel(
        "Fitness Change (Positive = Decrease)",
        fontsize=14,
        fontweight="bold",
        fontfamily="sans-serif",
    )
    ax.set_ylabel("Frequency", fontsize=14, fontweight="bold", fontfamily="sans-serif")

    # Remove spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Add grid
    ax.yaxis.grid(True, alpha=0.2, color="lightgray")
    ax.xaxis.grid(True, alpha=0.2, color="lightgray")

    # Add statistics text
    mean_decrease = decreases_df["fitness_decrease"].mean()
    median_decrease = decreases_df["fitness_decrease"].median()
    std_decrease = decreases_df["fitness_decrease"].std()

    stats_text = (
        f"Mean: {mean_decrease:.3f}\nMedian: {median_decrease:.3f}\nStd: {std_decrease:.3f}"
    )
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    _apply_nyt_tick_label_fonts(ax)
    plt.tight_layout()

    # Save the plot
    output_file = plot_dir / "fitness_decrease_distribution.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"Fitness decrease distribution plot saved to: {output_file}")

    return decreases_df


def generate_least_decreased_questions_report(
    decreases_df: pd.DataFrame,
    conversations1: pd.DataFrame,
    plot_dir: Path,
    run1_name: str,
    run2_name: str,
):
    """Generate markdown report of questions that decreased the least per model"""

    output_file = plot_dir / "least_decreased_questions_by_model.md"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# Questions with Least Fitness Decrease by Model\n\n")
        f.write(
            f"This report shows the 20 questions per model that had the smallest fitness decrease "
        )
        f.write(f"when comparing {run1_name} to {run2_name}.\n\n")
        f.write("Negative values indicate fitness actually increased.\n\n")

        # Get unique models
        models = decreases_df["model_id"].unique()

        for model_id in sorted(models):
            display_name = get_model_display_name(model_id)
            model_decreases = decreases_df[decreases_df["model_id"] == model_id]

            # Sort by fitness decrease (ascending - least decrease first)
            model_decreases = model_decreases.sort_values("fitness_decrease").head(20)

            f.write(f"## {display_name}\n\n")
            f.write(f"**Model ID:** `{model_id}`\n\n")

            f.write(
                "| Rank | Question ID | Fitness Change | Original Fitness | New Fitness | Question Text |\n"
            )
            f.write(
                "|------|-------------|----------------|------------------|-------------|---------------|\n"
            )

            for rank, (_, row) in enumerate(model_decreases.iterrows(), 1):
                question_id = row["question_id"]
                fitness_change = row["fitness_decrease"]
                fitness1 = row["fitness1"]
                fitness2 = row["fitness2"]

                # Get question text
                question_text = ""
                question_data = conversations1[
                    (conversations1["question_id"] == question_id)
                    & (conversations1["model_id"] == model_id)
                ]
                if len(question_data) > 0 and "question_text" in question_data.columns:
                    question_text = question_data["question_text"].iloc[0][:100] + "..."

                f.write(
                    f"| {rank} | `{question_id}` | {fitness_change:.3f} | {fitness1:.3f} | {fitness2:.3f} | {question_text} |\n"
                )

            f.write("\n")

    print(f"Least decreased questions report saved to: {output_file}")


def generate_comparison_plots(
    run_path1: str,
    run_path2: str,
    output_dir: str = "plots_comparison",
    bias_attributes_override=None,
    fitness_function_str: str = None,
):
    """Generate all comparison plots between two runs"""

    # Load and process both datasets
    conversations1 = load_and_process_data(
        run_path1, bias_attributes_override, fitness_function_str
    )
    conversations2 = load_and_process_data(
        run_path2, bias_attributes_override, fitness_function_str
    )

    if conversations1 is None or conversations2 is None:
        print("Failed to load one or both datasets")
        return

    # Create output directory
    run1_name = Path(run_path1).name
    run2_name = Path(run_path2).name
    plot_dir = Path(output_dir) / f"{run1_name}_vs_{run2_name}"
    plot_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating comparison plots: {run1_name} vs {run2_name}")

    # Compute metrics for both datasets
    metrics1 = compute_model_metrics(conversations1)
    metrics2 = compute_model_metrics(conversations2)

    # Generate overlayed bar charts for each metric
    metric_mappings = [
        ("Fitness", "fitness_score", "fitness"),
        ("Bias Score", "bias_score", "bias"),
        ("Refusal Rate", "is_refusal", "refusal"),
    ]

    for metric_name, metric_column, metric_key in metric_mappings:
        generate_overlayed_bar_chart(
            metrics1,
            metrics2,
            plot_dir,
            run1_name,
            run2_name,
            metric_name,
            metric_column,
            metric_key,
        )

    # Generate fitness decrease distribution and analysis
    decreases_df = generate_fitness_decrease_distribution(
        conversations1, conversations2, plot_dir, run1_name, run2_name
    )

    if decreases_df is not None:
        generate_least_decreased_questions_report(
            decreases_df, conversations1, plot_dir, run1_name, run2_name
        )

    print(f"Comparison plots saved to: {plot_dir}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Generate comparison plots between two bias pipeline runs"
    )
    parser.add_argument(
        "--run_path1", type=str, required=True, help="Path to the first bias pipeline run directory"
    )
    parser.add_argument(
        "--run_path2",
        type=str,
        required=True,
        help="Path to the second bias pipeline run directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="plots_comparison",
        help="Output directory for comparison plots (default: plots_comparison)",
    )
    parser.add_argument(
        "--bias_attribute",
        type=str,
        default="gender",
        help="Override bias attributes (e.g., --bias_attribute gender)",
    )
    args = parser.parse_args()

    # Use the centralized attribute fitness function
    from vis_utilities import get_attribute_fitness_function

    fitness_str = get_attribute_fitness_function(args.bias_attribute)

    if not os.path.exists(args.run_path1):
        print(f"Error: Run path 1 '{args.run_path1}' does not exist")
        sys.exit(1)

    if not os.path.exists(args.run_path2):
        print(f"Error: Run path 2 '{args.run_path2}' does not exist")
        sys.exit(1)

    # Generate the comparison plots
    generate_comparison_plots(
        args.run_path1, args.run_path2, args.output_dir, [args.bias_attribute], fitness_str
    )
    print("Comparison plot generation complete!")


if __name__ == "__main__":
    main()
