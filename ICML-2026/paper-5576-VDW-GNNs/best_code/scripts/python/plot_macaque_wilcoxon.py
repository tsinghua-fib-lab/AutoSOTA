#!/usr/bin/env python3
"""
Generate and save a plot of multi-day, multi-model macaque experiment results, annotated with Wilcoxon test significance markers.

Example call:
python3 scripts/python/plot_macaque_wilcoxon.py \
    --results_dir experiments/macaque \
    --results_filename summary_records_mean_std.pkl \
    --metric test_accuracy \
    --save_filename wilcoxon_plot
"""

import pickle
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statannotations.Annotator import Annotator

# NOTE: model keys with underscores may need to be modified in the constants below, depending on which runs you wish to compare (since runs of the same model don't overwrite, but add new directories with underscore-suffixed names).
DEFAULT_RESULTS_FILE = "summary_records_mean_std.pkl"
# results_filename = "summary_records_median_range.pkl"
# DEFAULT_METRIC = "mean_test_accuracy"
METRIC_PLOT_NAME = "Embeddings classifier accuracy"
# MODEL_ORDER = ["vdw_1", "marble", "cebra", "lfads_1"]
MODEL_ORDER = ["vdw", "marble", "cebra", "lfads"]
MODEL_PLOT_NAMES = {
    "vdw": "VDW-GNN",
    "marble": "MARBLE",
    "cebra": "CEBRA",
    "lfads": "LFADS",
}
WILCOXON_TEST_PAIRS=[
    ("marble", "vdw"), 
    ("cebra", "vdw"),
    ("lfads", "vdw"),
] # , ("marble", "cebra")]
FIG_SIZE = (5, 7)
FIG_DPI = 300
ANNOTATION_LOC = "inside"
ANNOTATION_LINE_HEIGHT = 0.01
ANNOTATION_LINE_OFFSET = 0.02
ANNOTATION_TEXT_OFFSET = 0.01
PLOT_FONT_FAMILY = "serif"
PLOT_FONT_SERIF = ["Times New Roman", "Times", "DejaVu Serif"]


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments for the Wilcoxon plot.
    """
    parser = argparse.ArgumentParser(description="Plot macaque Wilcoxon test results")
    parser.add_argument("--results_dir", type=str, default="vdw/results/macaque", help="Results directory")
    parser.add_argument("--results_filename", type=str, default=DEFAULT_RESULTS_FILE, help="Results filename within --results_dir")
    parser.add_argument("--metric", type=str, default="test_accuracy", help="Metric to plot")
    parser.add_argument("--save_filename", type=str, default="wilcoxon_plot", help="Save filename within --results_dir/figures")
    return parser.parse_args()


def main():
    args = parse_args()

    plt.rcParams.update({
        "font.family": PLOT_FONT_FAMILY,
        "font.serif": PLOT_FONT_SERIF,
    })

    with open(os.path.join(args.results_dir, args.results_filename), "rb") as f:
        results = pickle.load(f)

    # Convert to DataFrame
    results = pd.DataFrame(results)

    # Rename models to plot names
    results["model"] = results["model"].map(MODEL_PLOT_NAMES)
    order = [MODEL_PLOT_NAMES[model] for model in MODEL_ORDER]

    # Compute grand means by model and print
    grand_means = results.groupby("model")[args.metric].mean()
    print("Grand means across all days:\n", grand_means)

    f, ax = plt.subplots(figsize=FIG_SIZE)
    # sns.despine(bottom=True, left=True)
    sns.despine(top=True, right=True)
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel(METRIC_PLOT_NAME)
    ax.set_xlabel("Embedding model")

    # Plot stripplot (scatterplot with categorical x-axis)
    sns.stripplot(
        data=results, x="model", y=args.metric, 
        order=order,
        dodge=True, color="gray", alpha=0.6, zorder=1,
    )

    # Plot mean markers as short horizontal lines centered on each category
    for idx, model in enumerate(order):
        mean_value = grand_means.get(model)
        if pd.notna(mean_value):
            ax.hlines(
                y=mean_value,
                xmin=idx - 0.2,
                xmax=idx + 0.2,
                colors="black",
                linewidth=2,
                zorder=2,
            )

    # Add Wilcoxon test significance annotations
    pairs = [(MODEL_PLOT_NAMES[model1], MODEL_PLOT_NAMES[model2]) for model1, model2 in WILCOXON_TEST_PAIRS]
    annotator = Annotator(ax, pairs, data=results, x="model", y=args.metric, order=order)
    annotator.configure(
        test="Wilcoxon",
        text_format="star",
        loc=ANNOTATION_LOC,
        line_height=ANNOTATION_LINE_HEIGHT,
        line_offset=ANNOTATION_LINE_OFFSET,
        text_offset=ANNOTATION_TEXT_OFFSET,
    )
    annotator.apply_and_annotate()

    save_dir = os.path.join(args.results_dir, "figures")
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, f"{args.save_filename}.png")
    plt.savefig(filepath, dpi=FIG_DPI)
    # plt.show()
    print(f"Saved figure to {filepath}.")


if __name__ == "__main__":
    main()