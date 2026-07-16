import glob
import json
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
from matplotlib.lines import Line2D
from shapiq import InteractionValues
from shapiq_benchmark.metrics import get_all_metrics

DATA_NAMES = {
    "Cancer": "Cancer ($d=30$)",
    "Crime": "Crime ($d=101$)",
    "CG60": "CG60 ($d=60$)",
    "IL60": "IL60 ($d=60$)",
    "NHANES": "NHANES ($d=79$)",
    "DistilBERT": "DistilBERT ($d=14$)",
    "ViT16": "ViT16 ($d=16$)",
    "Estate": "Estate ($d=15$)",
    "ResNet18": "ResNet18 ($d=14$)",
    "ViT9": "ViT9 ($d=9$)",
}

LABEL_MAP = {
    "IndependentLinear60": "IL60",
    "BreastCancer": "Cancer",
    "RealEstate": "Estate",
    "Corrgroups60": "CG60",
    "NHANES": "NHANES",
    "SentimentIMDB": "DistilBERT",
    "ViT4by4": "ViT16",
    "CommunitiesAndCrime": "Crime",
    "ResNet18w14Superpixel": "ResNet18",
    "ViT3by3Patches": "ViT9",
}


# Global Plotting Constants
TITLE_FONT_SIZE = 20
LABEL_FONT_SIZE = 16
TICK_FONT_SIZE = 14
LINE_THICKNESS = 3
MARKER_SIZE = 7
LIGHT_GRAY = "#d3d3d3"

# Global Color and Sort Order for Datasets
HUE_ORDER = ["DistilBERT", "Estate", "ViT16", "Cancer", "CG60", "IL60", "NHANES", "Crime"]
_palette = sns.color_palette("deep", n_colors=len(HUE_ORDER))
COLOR_MAP = {ds: _palette[i] for i, ds in enumerate(HUE_ORDER)}


def get_n_players_from_dataset(dataset_name):
    # Mapping based on verified benchmark configurations
    n_players_map = {
        "DistilBERT": 14,
        "ResNet18": 14,
        "ViT9": 9,
        "ViT16": 16,
        "Crime": 101,
        "CG60": 60,
        "IL60": 60,
        "NHANES": 79,
        "Cancer": 30,
        "Estate": 15
    }
    return n_players_map.get(dataset_name, 0)


def load_data(experiment_type="interventional"):
    """
    Loads approximation results and ground truth for the specified experiment type.
    Computes metrics (MSE) for each approximation.
    """
    print(f"Loading data for experiment: {experiment_type}")

    # Define paths
    approx_path = Path(f"approximations/{experiment_type}")
    gt_path = Path(f"ground_truth/{experiment_type}")

    # Find all approximation files matching the ablation pattern
    approx_files = glob.glob(str(approx_path / "*_sampling_*.json"))

    results = []

    for file in approx_files:
        try:
            # Parse filename
            parts = Path(file).stem.split("_")
            order = int(parts[-1])
            index = parts[-2]
            budget = int(parts[-3])
            approx_name = parts[-4]
            id_explain = parts[-5]
            config_id = parts[-6]

            game_id = "_".join(parts[:-6])

            if "OddSHAP-pairing" in approx_name:
                sampling_key = approx_name.split("OddSHAP-")[1]
                sampling = mapping[sampling_key]
            else:
                sampling = "-1"

            approx_values = InteractionValues.load(file)

            random_state = 40
            gt_filename = f"{game_id}_{random_state}_{id_explain}_{index}_{order}_exact_values.json"
            gt_file = gt_path / gt_filename

            if not gt_file.exists():
                continue

            gt_values = InteractionValues.load(gt_file)

            metrics = get_all_metrics(gt_values, approx_values)

            mse = None
            for m in metrics:
                if getattr(m, "metric_id", "") == "MSE":
                    mse = m.value
                    break

            if mse is None:
                pass

            results.append({
                "game_id": game_id,
                "id_explain": id_explain,
                "sampling": sampling,
                "budget": budget,
                "MSE": mse,
                "experiment": experiment_type,
                "approx_name": approx_name
            })

        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue

    return pd.DataFrame(results)



def plot_final_grid(df, output_pdf="plots/ablation/sampling_results_grid.pdf"):
    """
    Creates a 3x3 grid plot.
    df must have columns: Dataset, budget, MSE, symbolic_x
    """
    from matplotlib.ticker import FuncFormatter

    # Standard ordering for the grid
    hue_order = HUE_ORDER

    budgets = [5000, 10000, 20000]
    fig, axes = plt.subplots(3, 3, figsize=(18, 9), sharex=False)

    # Common settings
    palette = COLOR_MAP

    print(f"Plotting grid for {len(df)} rows.")
    print("Dataset counts in df:", df["Dataset"].value_counts())

    global_handles, global_labels = None, None

    # prepare two-level x-axis categories (group, subgroup)
    if pd.api.types.is_categorical_dtype(df["sampling"]):
        categories = list(df["sampling"].cat.categories)
    else:
        categories = list(dict.fromkeys(df["sampling"].dropna().tolist()))

    # mapping from sampling category to (Group, Sub)
    sampling_group_map = {
        "NoPairs+All": ("Non-Paired", "All"),
        "NoPairs+Odd": ("Non-Paired", "Odd"),
        "Pairs+All": ("Paired", "All"),
        "Pairs+Odd": ("Paired", "Odd"),
    }

    positions = list(range(len(categories)))

    def _apply_two_level_xticks(ax):
        # bottom labels: subgroup (All/Odd)
        sub_labels = [sampling_group_map.get(c, (c, c))[1] for c in categories]
        ax.set_xticks(positions)
        ax.set_xticklabels(sub_labels, fontsize=TICK_FONT_SIZE)
        ax.set_xlim(-0.5, len(categories) - 0.5)

        # top/group labels: place centered text above groups
        seen = []
        for i, c in enumerate(categories):
            grp = sampling_group_map.get(c, (c, ""))[0]
            if grp not in seen:
                seen.append(grp)
        # visual group background colors (subtle) and labels
        group_colors = {"Paired": "#e8f3ff", "Non-Paired": "#fff5e6"}
        for grp in seen:
            idxs = [i for i, c in enumerate(categories) if sampling_group_map.get(c, (c, ""))[0] == grp]
            if not idxs:
                continue
            mid = (min(idxs) + max(idxs)) / 2.0
            xmin = min(idxs) - 0.5
            xmax = max(idxs) + 0.5
            # draw subtle background span for the group
            color = group_colors.get(grp, None)
            if color is not None:
                ax.axvspan(xmin, xmax, color=color, alpha=0.06, zorder=0)
            # place group label above the axis (use axis transform), slightly closer to ticks
            grp_label = f"{grp} Sampling" if grp in ("Paired", "Non-Paired") else grp
            ax.text(mid, -0.14, grp_label, transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=LABEL_FONT_SIZE, fontweight='bold')

    # Row 1: Absolute MSE
    for i, budget in enumerate(budgets):
        ax = axes[0, i]
        subset = df[df["budget"] == budget]

        if subset.empty:
            print(f"WARNING: Subset empty for budget {budget}")
            ax.axis('off')
            continue

        sns.lineplot(
            data=subset, x="sampling", y="MSE", hue="Dataset", hue_order=hue_order,
            estimator="median", errorbar=("pi", 50),
            marker="o", markersize=MARKER_SIZE, linewidth=LINE_THICKNESS,
            markeredgecolor="white", palette=palette, ax=ax, legend=(i == 0), zorder=5
        )

        # apply two-level x-axis labels
        _apply_two_level_xticks(ax)

        # Set symbolic labels and ensure visibility
        # ax.set_xlim(-0.5, 4.5)

        ax.set_yscale("log")
        #ax.set_ylim(1e-7, 20)  # Avoid squashing by outliers or k=0 artifacts
        ax.set_title(f"{budget:,} Samples", fontsize=TITLE_FONT_SIZE)
        ax.set_ylabel(r"MSE (Median $\pm$ IQR Band)", fontsize=LABEL_FONT_SIZE)
        ax.set_xlabel("", fontsize=LABEL_FONT_SIZE)
        ax.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE, bottom=True, left=True)
        ax.grid(True, axis='x', color=LIGHT_GRAY, linestyle='--')
        ax.grid(True, axis='y', color=LIGHT_GRAY, linestyle='--')
        ax.set_axisbelow(True)
        ax.axhline(1, color='black', linewidth=1.5, linestyle='-', zorder=1.6)

        # Use scientific notation for all log plots
        from matplotlib.ticker import LogLocator, LogFormatterMathtext
        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
        ax.yaxis.set_major_formatter(LogFormatterMathtext())

        if i == 0:
            global_handles, global_labels = ax.get_legend_handles_labels()
            ax.get_legend().remove()

    # Row 2: Relative MSE to LeverageSHAP (k=0)
    for i, budget in enumerate(budgets):
        ax = axes[1, i]
        subset = df[df["budget"] == budget]

        if subset.empty:
            ax.axis('off')
            continue

        subset = subset.copy()

        # Compute Relative MSE
        base_mse = subset[subset["sampling"] == "Pairs+Odd"][["game_id", "id_explain", "MSE"]].rename(
            columns={"MSE": "base_MSE"})
        subset = subset.merge(base_mse, on=["game_id", "id_explain"])
        subset["Relative MSE"] = subset["MSE"] / subset["base_MSE"]

        sns.lineplot(
            data=subset, x="sampling", y="Relative MSE", hue="Dataset", hue_order=hue_order,
            estimator="median", errorbar=("pi", 50),
            marker="o", markersize=MARKER_SIZE, linewidth=LINE_THICKNESS,
            markeredgecolor="white", palette=palette, ax=ax, legend=False, zorder=5
        )
        _apply_two_level_xticks(ax)

        ax.set_yscale("log")
        #ax.set_ylim(1e-5, 2)
        ax.set_title(f"{budget:,} Samples", fontsize=TITLE_FONT_SIZE)
        ax.set_ylabel(r"MSE Ratio (Median $\pm$ IQR Band)", fontsize=LABEL_FONT_SIZE)
        ax.set_xlabel("", fontsize=LABEL_FONT_SIZE)
        ax.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE, bottom=True, left=True)
        ax.grid(True, axis='x', color=LIGHT_GRAY, linestyle='--')
        ax.grid(True, axis='y', color=LIGHT_GRAY, linestyle='--')
        ax.set_axisbelow(True)
        ax.axhline(1, color='black', linewidth=1.5, linestyle='-', zorder=1.6)

        # Use scientific notation for all log plots
        from matplotlib.ticker import LogLocator, LogFormatterMathtext
        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
        ax.yaxis.set_major_formatter(LogFormatterMathtext())

    # Row 3: Error Reduction (Factor vs k=0)
    for i, budget in enumerate(budgets):
        ax = axes[2, i]
        subset = df[df["budget"] == budget]

        if subset.empty:
            ax.axis('off')
            continue

        subset = subset.copy()

        # Compute Error Reduction
        base_mse = subset[subset["sampling"] == "Pairs+Odd"][["game_id", "id_explain", "MSE"]].rename(
            columns={"MSE": "base_MSE"})
        subset = subset.merge(base_mse, on=["game_id", "id_explain"])
        subset["Error Reduction"] = subset["base_MSE"] / subset["MSE"]

        sns.lineplot(
            data=subset, x="sampling", y="Error Reduction", hue="Dataset", hue_order=hue_order,
            estimator="median", errorbar=("pi", 50),
            marker="o", markersize=MARKER_SIZE, linewidth=LINE_THICKNESS,
            markeredgecolor="white", palette=palette, ax=ax, legend=False, zorder=5
        )
        _apply_two_level_xticks(ax)


        ax.set_yscale("log")
        #ax.set_ylim(0.5, 1e5)
        ax.set_title(f"{budget:,} Samples", fontsize=TITLE_FONT_SIZE)
        ax.set_ylabel(r"Error Reduction", fontsize=LABEL_FONT_SIZE)
        ax.set_xlabel("", fontsize=LABEL_FONT_SIZE)
        ax.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE, bottom=True, left=True)
        ax.grid(True, axis='x', color=LIGHT_GRAY, linestyle='--')
        ax.grid(True, axis='y', color=LIGHT_GRAY, linestyle='--')
        ax.set_axisbelow(True)
        ax.axhline(1, color='black', linewidth=1.5, linestyle='-', zorder=1.6)

        # Use scientific notation for all log plots
        from matplotlib.ticker import LogLocator, LogFormatterMathtext
        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
        ax.yaxis.set_major_formatter(LogFormatterMathtext())

    # Map labels to include d=N
    display_labels = [DATA_NAMES.get(l, l) for l in global_labels]

    # Common Legend at top-right
    fig.legend(global_handles, display_labels, loc='upper right', bbox_to_anchor=(0.98, 0.98),
               ncol=1, title=r"$\bf{Dataset}$", fontsize=TICK_FONT_SIZE, title_fontsize=LABEL_FONT_SIZE, frameon=False)

    plt.suptitle("Shapley MSE by Configuration", fontsize=TITLE_FONT_SIZE + 4, y=1.02)
    plt.tight_layout(rect=[0.05, 0.10, 1, 1.02])

    plt.savefig(output_pdf, bbox_inches="tight")
    print(f"Saved final grid plot to {output_pdf}")
    plt.close()


def plot_grid_cells_individually(df, output_dir="plots/sampling"):
    """
    Plots each cell of the grid as a separate PDF and saves the legend separately.
    df must have Dataset, budget, MSE, symbolic_x
    """
    from matplotlib.ticker import FuncFormatter

    # Standard ordering for individual cell plots
    hue_order = HUE_ORDER
    palette = COLOR_MAP

    budgets = [5000, 10000, 20000]

    # helper for common styling
    def style_ax(ax, title, ylabel, budget):
        ax.set_yscale("log")
        ax.set_title(title, fontsize=TITLE_FONT_SIZE)
        ax.set_ylabel(ylabel, fontsize=LABEL_FONT_SIZE)
        ax.set_xlabel("", fontsize=LABEL_FONT_SIZE)
        ax.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE, bottom=True, left=True)
        ax.grid(True, axis='x', color=LIGHT_GRAY, linestyle='--')
        ax.grid(True, axis='y', color=LIGHT_GRAY, linestyle='--')
        ax.set_axisbelow(True)
        ax.axhline(1, color='black', linewidth=1.5, linestyle='-', zorder=1.6)

        # Use scientific notation for all log plots
        from matplotlib.ticker import LogLocator, LogFormatterMathtext
        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
        ax.yaxis.set_major_formatter(LogFormatterMathtext())

    # 1. Absolute MSE
    # prepare two-level x-axis categories for individual plots
    if pd.api.types.is_categorical_dtype(df["sampling"]):
        categories_local = list(df["sampling"].cat.categories)
    else:
        categories_local = list(dict.fromkeys(df["sampling"].dropna().tolist()))

    sampling_group_map = {
        "NoPairs+All": ("Non-Paired", "All"),
        "NoPairs+Odd": ("Non-Paired", "Odd"),
        "Pairs+All": ("Paired", "All"),
        "Pairs+Odd": ("Paired", "Odd"),
    }

    positions_local = list(range(len(categories_local)))

    def _apply_two_level_xticks_local(ax):
        sub_labels = [sampling_group_map.get(c, (c, c))[1] for c in categories_local]
        ax.set_xticks(positions_local)
        ax.set_xticklabels(sub_labels, fontsize=TICK_FONT_SIZE)
        ax.set_xlim(-0.5, len(categories_local) - 0.5)
        seen = []
        for i, c in enumerate(categories_local):
            grp = sampling_group_map.get(c, (c, ""))[0]
            if grp not in seen:
                seen.append(grp)
        # visual group background colors (subtle) and labels
        group_colors = {"Paired": "#e8f3ff", "Non-Paired": "#fff5e6"}
        for grp in seen:
            idxs = [i for i, c in enumerate(categories_local) if sampling_group_map.get(c, (c, ""))[0] == grp]
            if not idxs:
                continue
            mid = (min(idxs) + max(idxs)) / 2.0
            xmin = min(idxs) - 0.5
            xmax = max(idxs) + 0.5
            color = group_colors.get(grp, None)
            if color is not None:
                ax.axvspan(xmin, xmax, color=color, alpha=0.06, zorder=0)
            grp_label = f"{grp} Sampling" if grp in ("Paired", "Non-Paired") else grp
            ax.text(mid, -0.14, grp_label, transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=LABEL_FONT_SIZE, fontweight='bold')

    for budget in budgets:
        plt.figure(figsize=(6, 4), layout="constrained")
        subset = df[df["budget"] == budget]
        if subset.empty: continue

        # Standard plot (No Legend)
        ax = sns.lineplot(
            data=subset, x="sampling", y="MSE", hue="Dataset", hue_order=hue_order,
            estimator="median", errorbar=("pi", 50),
            marker="o", markersize=MARKER_SIZE, linewidth=LINE_THICKNESS,
            markeredgecolor="white", palette=palette, legend=False, zorder=5
        )
        style_ax(ax, f"{budget:,} Samples", r"MSE (Median $\pm$ IQR Band)", budget)
        _apply_two_level_xticks_local(ax)
        outfile = f"{output_dir}/sampling_cell_absolute_{budget}.pdf"
        plt.savefig(outfile, bbox_inches="tight", pad_inches=0.2)
        plt.close()

        # w_legend plot (Exclude Estate)
        plt.figure(figsize=(6, 4), layout="constrained")
        subset_no_estate = subset[subset["Dataset"] != "Estate"]
        hue_order_no_estate = [h for h in hue_order if h != "Estate"]
        palette_no_estate = COLOR_MAP

        ax = sns.lineplot(
            data=subset_no_estate, x="sampling", y="MSE", hue="Dataset", hue_order=hue_order_no_estate,
            estimator="median", errorbar=("pi", 50),
            marker="o", markersize=MARKER_SIZE, linewidth=LINE_THICKNESS,
            markeredgecolor="white", palette=palette_no_estate, legend=True, zorder=5
        )
        style_ax(ax, "", r"MSE (Median $\pm$ IQR Band)", budget)  # No title for legend variant usually
        _apply_two_level_xticks_local(ax)
        handles, labels = ax.get_legend_handles_labels()
        display_labels = [DATA_NAMES.get(l, l) for l in labels]
        short_labels = [l.split(' ($')[0] for l in display_labels]
        plt.legend(handles, short_labels, loc='upper right', title=None, fontsize=10, frameon=True)

        outfile_legend = f"{output_dir}/sampling_cell_absolute_{budget}_w_legend.pdf"
        plt.savefig(outfile_legend, bbox_inches="tight", pad_inches=0.2)
        plt.close()
        print(f"Saved {outfile} and {outfile_legend}")

    # 2. Relative MSE
    for budget in budgets:
        plt.figure(figsize=(6, 4), layout="constrained")
        subset = df[df["budget"] == budget]
        if subset.empty: continue
        subset = subset.copy()

        # Compute Relative MSE
        base_mse = subset[subset["sampling"] == "Pairs+Odd"][["game_id", "id_explain", "MSE"]].rename(
            columns={"MSE": "base_MSE"})
        subset = subset.merge(base_mse, on=["game_id", "id_explain"])
        subset["Relative MSE"] = subset["MSE"] / subset["base_MSE"]

        # Standard plot (No Legend)
        ax = sns.lineplot(
            data=subset, x="sampling", y="Relative MSE", hue="Dataset", hue_order=hue_order,
            estimator="median", errorbar=("pi", 50),
            marker="o", markersize=MARKER_SIZE, linewidth=LINE_THICKNESS,
            markeredgecolor="white", palette=palette, legend=False, zorder=5
        )
        style_ax(ax, f"{budget:,} Samples", r"MSE Ratio (Median $\pm$ IQR Band)", budget)
        _apply_two_level_xticks_local(ax)
        outfile = f"{output_dir}/sampling_cell_relative_{budget}.pdf"
        plt.savefig(outfile, bbox_inches="tight", pad_inches=0.2)
        plt.close()

        # w_legend plot (Exclude Estate)
        plt.figure(figsize=(6, 4), layout="constrained")
        subset_no_estate = subset[subset["Dataset"] != "Estate"]
        hue_order_no_estate = [h for h in hue_order if h != "Estate"]
        palette_no_estate = COLOR_MAP

        ax = sns.lineplot(
            data=subset_no_estate, x="sampling", y="Relative MSE", hue="Dataset", hue_order=hue_order_no_estate,
            estimator="median", errorbar=("pi", 50),
            marker="o", markersize=MARKER_SIZE, linewidth=LINE_THICKNESS,
            markeredgecolor="white", palette=palette_no_estate, legend=True, zorder=5
        )
        style_ax(ax, "", r"MSE Ratio (Median $\pm$ IQR Band)", budget)
        _apply_two_level_xticks_local(ax)
        handles, labels = ax.get_legend_handles_labels()
        display_labels = [DATA_NAMES.get(l, l) for l in labels]
        short_labels = [l.split(' ($')[0] for l in display_labels]
        plt.legend(handles, short_labels, loc='upper right', title=None, fontsize=10, frameon=True)

        outfile_legend = f"{output_dir}/sampling_cell_relative_{budget}_w_legend.pdf"
        plt.savefig(outfile_legend, bbox_inches="tight", pad_inches=0.2)
        plt.close()
        print(f"Saved {outfile} and {outfile_legend}")

    # 3. Error Reduction
    for budget in budgets:
        plt.figure(figsize=(6, 4), layout="constrained")
        subset = df[df["budget"] == budget]
        if subset.empty: continue
        subset = subset.copy()

        # Compute Error Reduction
        base_mse = subset[subset["sampling"] == 10][["game_id", "id_explain", "MSE"]].rename(
            columns={"MSE": "base_MSE"})
        subset = subset.merge(base_mse, on=["game_id", "id_explain"])
        subset["Error Reduction"] = subset["base_MSE"] / subset["MSE"]

        # Standard plot (No Legend)
        ax = sns.lineplot(
            data=subset, x="sampling", y="Error Reduction", hue="Dataset", hue_order=hue_order,
            estimator="median", errorbar=("pi", 50),
            marker="o", markersize=MARKER_SIZE, linewidth=LINE_THICKNESS,
            markeredgecolor="white", palette=palette, legend=False, zorder=5
        )
        style_ax(ax, f"{budget:,} Samples", r"Error Reduction", budget)
        _apply_two_level_xticks_local(ax)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:g}x'))
        outfile = f"{output_dir}/sampling_cell_reduction_{budget}.pdf"
        plt.savefig(outfile, bbox_inches="tight", pad_inches=0.2)
        plt.close()

        # w_legend plot (Exclude Estate)
        plt.figure(figsize=(6, 4), layout="constrained")
        subset_no_estate = subset[subset["Dataset"] != "Estate"]
        hue_order_no_estate = [h for h in hue_order if h != "Estate"]
        palette_no_estate = COLOR_MAP

        ax = sns.lineplot(
            data=subset_no_estate, x="sampling", y="Error Reduction", hue="Dataset", hue_order=hue_order_no_estate,
            estimator="median", errorbar=("pi", 50),
            marker="o", markersize=MARKER_SIZE, linewidth=LINE_THICKNESS,
            markeredgecolor="white", palette=palette_no_estate, legend=True, zorder=5
        )
        style_ax(ax, "", r"Error Reduction", budget)
        _apply_two_level_xticks_local(ax)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:g}x'))
        handles, labels = ax.get_legend_handles_labels()
        display_labels = [DATA_NAMES.get(l, l) for l in labels]
        short_labels = [l.split(' ($')[0] for l in display_labels]
        plt.legend(handles, short_labels, loc='upper right', title=None, fontsize=10, frameon=True)

        outfile_legend = f"{output_dir}/sampling_cell_reduction_{budget}_w_legend.pdf"
        plt.savefig(outfile_legend, bbox_inches="tight", pad_inches=0.2)
        plt.close()
        print(f"Saved {outfile} and {outfile_legend}")

    # Save Legend Separately
    hue_order_legend = HUE_ORDER
    fig_leg = plt.figure(figsize=(10, 0.5))
    subset_leg = df[df["budget"] == 10000]
    if subset_leg.empty: subset_leg = df

    palette = COLOR_MAP
    ax = sns.lineplot(
        data=subset_leg, x="sampling", y="MSE", hue="Dataset", hue_order=hue_order_legend,
        palette=palette
    )
    handles, labels = ax.get_legend_handles_labels()
    display_labels = [DATA_NAMES.get(l, l) for l in labels]

    fig_leg.legend(handles, display_labels, loc='center',
                   ncol=4, title=r"$\bf{Dataset}$", fontsize=TICK_FONT_SIZE, title_fontsize=LABEL_FONT_SIZE,
                   frameon=False)
    ax.remove()
    plt.axis('off')

    leg_file = f"{output_dir}/ablation_legend.pdf"
    fig_leg.savefig(leg_file, bbox_inches="tight", pad_inches=0)
    plt.close(fig_leg)
    print(f"Saved {leg_file}")


if __name__ == "__main__":
    mapping = {
        "pairing0oddonly1": "NoPairs+Odd",
        "pairing1oddonly1": "Pairs+Odd",
        "pairing0oddonly0": "NoPairs+All",
        "pairing1oddonly0": "Pairs+All",
    }
    import numpy as np

    csv_int = Path("experiments/sampling_interventional_mse_results.csv")
    csv_exh = Path("experiments/sampling_exhaustive_mse_results.csv")

    # Reload from CSV if they exist, else load from JSONs
    if csv_int.exists() and csv_exh.exists():
        print("Loading data from existing CSVs.")
        df_int = pd.read_csv(csv_int)
        df_exh = pd.read_csv(csv_exh)
    else:
        print("Loading data from JSON approximation files (this may take a while)...")
        df_int = load_data("interventional")
        df_exh = load_data("exhaustive")

    df = pd.concat([df_int, df_exh], ignore_index=True)
    print(f"Total rows loaded: {len(df)}")

    # Ensure correct data types
    df["budget"] = df["budget"].astype(int)
    if "MSE" in df.columns:
        df["MSE"] = df["MSE"].astype(float)


    # Force k to be determined from approx_name
    def extract_sampling_from_name(name):
        if "OddSHAP-pairing" in str(name):
            try:
                key = str(name).split("OddSHAP-")[1]
                return mapping[key]
            except:
                return "-1"
        return "-1"


    df["sampling"] = df["approx_name"].apply(extract_sampling_from_name)
    # Desired ordering for the sampling categories (customize as needed)
    DESIRED_SAMPLING_ORDER = ["NoPairs+All", "NoPairs+Odd","Pairs+All","Pairs+Odd"]

    # Restrict to categories that actually appear in the dataframe (defensive)
    present = df["sampling"].dropna().unique().tolist()
    final_order = [c for c in DESIRED_SAMPLING_ORDER if c in present] + [c for c in present if
                                                                         c not in DESIRED_SAMPLING_ORDER]

    # Make the column categorical with the final ordered categories
    df["sampling"] = pd.Categorical(df["sampling"], categories=final_order, ordered=True)

    # Standardize Dataset Names
    def get_clean_dataset(game_id):
        gid = game_id.rsplit('_', 1)[0]
        for kw, short in LABEL_MAP.items():
            if kw.lower() in gid.lower():
                return short
        return gid


    df["Dataset"] = df["game_id"].apply(get_clean_dataset)
    df["n_players"] = df["Dataset"].apply(get_n_players_from_dataset)

    print("Dataset counts before filtering:", df["Dataset"].value_counts().to_dict())

    # Filter cases where budget > 2^n
    df = df[df.apply(lambda x: (2 ** int(x["n_players"])) >= x["budget"], axis=1)]
    print(f"Total rows after budget filter: {len(df)}")


    # Generate plots
    plot_final_grid(df, "plots/sampling/sampling_results_grid.pdf")
    plot_grid_cells_individually(df, output_dir="plots/sampling")
