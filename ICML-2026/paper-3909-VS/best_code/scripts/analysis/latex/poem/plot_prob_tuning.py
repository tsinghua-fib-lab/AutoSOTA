# Copyright 2025 CHATS-Lab. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt


def extract_prob_tuning_data(base_dir, model_name, method_type="vs_multi"):
    """Extract probability tuning data for a specific model and method"""
    prob_values = []
    diversity_scores = []

    model_dir = Path(base_dir) / model_name / f"{model_name}_poem" / "evaluation"

    # Find all prob_tuning directories for the specified method
    prob_dirs = []
    for item in model_dir.iterdir():
        if item.is_dir() and method_type in item.name and "prob_tuning=" in item.name:
            prob_dirs.append(item)

    print(f"prob_dirs: {prob_dirs}")
    probs = []
    for prob_dir in prob_dirs:
        # Extract prob_tuning value from directory name
        prob_str = prob_dir.name.split("prob_tuning=")[1].split(")")[0]
        probs.append(float(prob_str))
        prob_value = abs(float(prob_str))

        # Read diversity results
        diversity_file = prob_dir / "diversity_results.json"
        if diversity_file.exists():
            with open(diversity_file, "r") as f:
                data = json.load(f)
                diversity_score = data["overall_metrics"]["avg_diversity"]

                prob_values.append(prob_value)
                diversity_scores.append(diversity_score)
    print(f"probs: {probs}")
    # Sort by prob_tuning value
    sorted_data = sorted(zip(prob_values, diversity_scores))
    return [x[0] for x in sorted_data], [x[1] for x in sorted_data]


def get_baseline_scores(base_dir, model_name):
    """Get baseline scores for direct and sequence methods"""
    model_dir = Path(base_dir) / model_name / f"{model_name}_poem" / "evaluation"

    baselines = {}

    # Direct baseline
    direct_file = model_dir / "direct (samples=1)" / "diversity_results.json"
    if direct_file.exists():
        with open(direct_file, "r") as f:
            data = json.load(f)
            baselines["direct"] = data["overall_metrics"]["avg_diversity"]

    # Sequence baseline
    sequence_file = model_dir / "sequence [strict] (samples=5)" / "diversity_results.json"
    if sequence_file.exists():
        with open(sequence_file, "r") as f:
            data = json.load(f)
            baselines["sequence"] = data["overall_metrics"]["avg_diversity"]

    return baselines


def plot_prob_tuning_results():
    base_dir = "poem_experiments_prob_tuning"

    # Model configurations
    models = {"GPT-4.1": "openai_gpt-4.1", "Gemini 2.5 Flash": "google_gemini-2.5-flash"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for i, (model_display_name, model_dir_name) in enumerate(models.items()):
        ax = ax1 if i == 0 else ax2

        # Extract data for both VS-Multi (vs_multi) and structure_with_prob methods
        combined_prob_vals, combined_diversity = extract_prob_tuning_data(
            base_dir, model_dir_name, "vs_multi"
        )
        structure_prob_vals, structure_diversity = extract_prob_tuning_data(
            base_dir, model_dir_name, "vs_standard"
        )

        # Plot the probability tuning curves
        ax.plot(
            combined_prob_vals,
            combined_diversity,
            "o-",
            label="Combined",
            linewidth=2,
            markersize=6,
        )
        ax.plot(
            structure_prob_vals,
            structure_diversity,
            "s-",
            label="Structure with Prob",
            linewidth=2,
            markersize=6,
        )

        # Get and plot baselines
        baselines = get_baseline_scores(base_dir, model_dir_name)

        if combined_prob_vals:  # Only plot baselines if we have data
            x_min, x_max = min(combined_prob_vals + structure_prob_vals), max(
                combined_prob_vals + structure_prob_vals
            )

            if "direct" in baselines:
                ax.axhline(
                    y=baselines["direct"],
                    color="red",
                    linestyle="--",
                    alpha=0.7,
                    label="Direct Baseline",
                )

            if "sequence" in baselines:
                ax.axhline(
                    y=baselines["sequence"],
                    color="green",
                    linestyle="--",
                    alpha=0.7,
                    label="Sequence Baseline",
                )
        ax.set_xscale("log")
        ax.set_xlabel("Probability Tuning Value")
        ax.set_ylabel("Diversity Score")
        ax.set_title(f"{model_display_name}")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Reverse x-axis
        ax.invert_xaxis()

        # Set x-axis to log scale if we have negative values (prob_tuning=-1)
        if any(val < 0 for val in combined_prob_vals + structure_prob_vals):
            # Handle negative values by using custom tick labels
            all_vals = sorted(set(combined_prob_vals + structure_prob_vals))
            ax.set_xticks(all_vals)
            ax.set_xticklabels([str(val) for val in all_vals])

    plt.tight_layout()
    plt.savefig("latex_figures/poem_prob_tuning_diversity.png", dpi=300, bbox_inches="tight")
    plt.savefig("latex_figures/poem_prob_tuning_diversity.pdf", bbox_inches="tight")
    # plt.show()

    print("Plots saved to latex_figures/poem_prob_tuning_diversity.png and .pdf")


if __name__ == "__main__":
    # Create output directory if it doesn't exist
    os.makedirs("latex_figures", exist_ok=True)

    plot_prob_tuning_results()
