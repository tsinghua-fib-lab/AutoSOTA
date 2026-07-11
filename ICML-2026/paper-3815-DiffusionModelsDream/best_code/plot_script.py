import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

# --- Utility Functions ---
def make_unique_names(basenames):
    """Generates unique names from a list of potentially duplicate strings."""
    counts = {}
    unique = []
    for b in basenames:
        c = counts.get(b, 0) + 1
        counts[b] = c
        unique.append(b if c == 1 else f"{b}__{c}")
    return unique

def select_columns(df, col_slice=None, cols=None, dicts=None):
    """Selects columns from a DataFrame by list or by slice."""
    if cols is not None:
        cols = list(cols)
        for i,d in enumerate(cols):
            if isinstance(d, dict):

                # Extract variables and function definition
                name = d["name"]
                func_str = d["function"]

                # Copy the dictionary minus 'name' and 'function' into local scope
                local_vars = {k: df.get(v).to_numpy() for k, v in d.items() if k not in ["name", "function"]}
                
                # Evaluate the lambda in this local scope
                lambda_fn = eval(func_str, {"np": np}, {})
                new_column = lambda_fn(**local_vars)

                # Add the column into the dataframe
                df["/temp_vars/"+name] = new_column

                # Drop from cols
                cols[i] = "/temp_vars/"+name
        return df[cols].copy(), cols

    if col_slice is not None:
        s1, s2 = col_slice
        cols = df.columns[s1:s2].tolist()
        return df.iloc[:, s1:s2].copy(), cols

    # Default to first 6 columns if nothing is specified
    cols = df.columns[:6].tolist()
    return df.iloc[:, :6].copy(), cols

def discover_investigations(run_dir: Path):
    """Finds all investigation subdirectories within a given run directory."""
    run_dir = Path(run_dir)
    inv_json = run_dir / "investigations.json"
    if inv_json.exists():
        spec = json.loads(inv_json.read_text())
        names = [f"inv_{inv['name']}" for inv in spec.get("investigations", []) if "name" in inv]
        return [run_dir / n for n in names if (run_dir / n).exists()]
    return sorted([p for p in run_dir.glob("inv_*") if p.is_dir()])

# --- Main Plotting Function ---

def generate_plots(run_dir, project_root, data_csv, cols, col_slice,
                   legend_config, downsample, save_png, out_png_filename, show_training_data, categorical, extra_plots, format_labels):
    """
    Generates and saves a corner plot comparing investigation results against
    a baseline dataset.
    """
    run_dir = Path(run_dir)
    project_root = Path(project_root)
    print(f"--- Processing Run Directory: {run_dir.resolve()} ---")

    data_csv_path = Path(data_csv) if Path(data_csv).is_absolute() else project_root / data_csv
    if not data_csv_path.exists():
        print(f"[Error] Data CSV not found at: {data_csv_path}")
        return

    df_full_raw = pd.read_csv(data_csv_path)
    lablist = list(df_full_raw.columns)
    df_full = pd.DataFrame(df_full_raw.to_numpy(), columns=lablist)

    legend_title = legend_config.get("title", "Dataset")

    df_data_sel, sel_cols = select_columns(df_full, col_slice=col_slice, cols=cols)

    # Rename to unique leaf names for plotting labels
    leaf_names = [c.split("/")[-1] for c in sel_cols]
    formatted_names = ([name.replace('_', ' ').title() for name in leaf_names] if not format_labels else format_labels)
    unique_leaf_names = make_unique_names(formatted_names)
    df_data_sel = pd.DataFrame(df_data_sel.to_numpy(), columns=unique_leaf_names)
    df_data_sel["dataset"] = "Training Data"

    # ---- Collect all investigations into one DataFrame ----
    inv_dirs = discover_investigations(run_dir)
    if not inv_dirs:
        print(f"[Warning] No inv_* folders found under {run_dir}. Skipping plot.")
        return

    print(f"Found {len(inv_dirs)} investigations under {run_dir}")

    if show_training_data:
        frames = [df_data_sel]  # start with the Data baseline
    else:
        frames = []  # start with an empty list
    inv_names = []

    for inv_dir in inv_dirs:
        name = inv_dir.name
        samples_csv = inv_dir / "samples.csv"
        if not samples_csv.exists():
            print(f"[skip] {name}: no samples.csv")
            continue

        df_samp_raw = pd.read_csv(samples_csv)
        dataset_label = name

        if legend_config:
            if "legend" in legend_config:
                dataset_label = legend_config["legend"][name]
            else:
                try:
                    values = [abs(df_samp_raw[key].iloc[0]) for key in legend_config['keys']] # WARNING WE TAKE THE ABS VALUE
                    dataset_label = legend_config['format_string'].format(*values)
                except Exception as e:
                    print(f"[Warning] Could not create custom label for {name}: {e}")

        df_samp_sel, _ = select_columns(df_samp_raw, col_slice=col_slice, cols=cols) #df_samp_raw[sel_cols].copy()
        df_samp_sel = pd.DataFrame(df_samp_sel.to_numpy(), columns=unique_leaf_names)
        df_samp_sel["dataset"] = dataset_label
        frames.append(df_samp_sel)
        inv_names.append(dataset_label)

    combined = pd.concat(frames, ignore_index=True)

    if downsample:
        subs = []
        for label, group in combined.groupby("dataset"):
            subs.append(group.sample(min(downsample, len(group)), random_state=0))
        combined = pd.concat(subs, ignore_index=True)

    # ---- Generate single pairplot with all investigations ----
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12

    plot_vars = [
        c for c in combined.columns
        if c != "dataset" and c not in categorical
    ]

    axis_limits = {}
    for col in plot_vars:
        min_val, max_val = df_data_sel[col].min(), df_data_sel[col].max()
        padding = (max_val - min_val) * 0.05
        axis_limits[col] = (min_val - padding, max_val + padding)

    hue_order = ["Training Data"] + inv_names
    n_inv = len(inv_names)
    inv_colors = sns.color_palette("tab10", n_inv) if n_inv > 0 else []
    inv_palette = {inv_names[i]: inv_colors[i % len(inv_colors)] for i in range(n_inv)}

    if show_training_data:
        hue_order = ["Training Data"] + inv_names
        palette = {"Training Data": "gray"}
        palette.update(inv_palette)
    else:
        hue_order = inv_names
        palette = inv_palette

    # --- Stacked bar charts for categorical variables ---
    for col in categorical:
        if col in combined.columns:
            plt.figure(figsize=(8,5))
            (
                combined
                .groupby("dataset")[col]
                .value_counts()
                .unstack(fill_value=0)
                .rename(columns=lambda x: int(x))
                .plot(kind="bar", stacked=True)
            )
            plt.xlabel(legend_title)
            plt.ylabel("Count")
            plt.tight_layout()
            out_path = run_dir / ("categorical_" + col + out_png_filename)
            plt.savefig(out_path)

    sns.set_context("talk", font_scale=1.8) # was 1.4

    g = sns.pairplot(
        combined, kind="scatter", diag_kind="hist", hue="dataset",
        hue_order=hue_order, palette=palette, corner=True, vars=plot_vars,
        plot_kws={"s": 10, "linewidth": 0}, diag_kws={'bins': 30}
    )

    # Cross-correlations: single horizontal row + legend
    if extra_plots:
        n_plots = len(extra_plots)
        n_total = n_plots + 1  # +1 for legend panel

        fig_extra, axes = plt.subplots(
            1, n_total,
            figsize=(5 * n_total, 6),
            squeeze=False
        )

        axes = axes[0]  # flatten row

        handles = [
            plt.Line2D(
                [0], [0],
                marker='o',
                linestyle='',
                markersize=10,
                color=palette[label],
                label=label
            )
            for label in hue_order
        ]

        for k, (i, j) in enumerate(extra_plots):

            if (i >= len(g.y_vars) and i != -1) or j >= len(g.x_vars):
                axes[k].axis("off")
                continue

            ax = axes[k]

            # Categorical
            if i==-1:
                col = categorical[j]
                if col in combined.columns:
                    df_counts = combined.groupby("dataset")[col].value_counts().unstack(fill_value=0)
                    
                    n_cats = len(df_counts.columns)

                    styles = [
                        {"facecolor": "lightgrey", "hatch": None},   # solid grey
                        {"facecolor": "white",     "hatch": "///"},   # solid black
                        {"facecolor": "black",     "hatch": None},  # white + hatch
                        {"facecolor": "lightgrey", "hatch": "\\"},   # grey + hatch
                        {"facecolor": "white",     "hatch": None},   # solid white
                    ]

                    ax_colors = df_counts.plot(
                        kind="bar",
                        stacked=True,
                        ax=ax,
                        legend=False,
                        color="white"   # base color overridden below
                    ).containers

                    for i, bar_container in enumerate(ax_colors):
                        style = styles[i % len(styles)]
                        for patch in bar_container:
                            patch.set_facecolor(style["facecolor"])
                            patch.set_edgecolor("black")
                            if style["hatch"] is not None:
                                patch.set_hatch(style["hatch"])

                    ax.set_xlabel(legend_title, labelpad=20)
                    ax.set_ylabel("Count")

                    xticks = np.arange(len(df_counts.index))
                    ax.set_xticks(xticks)

                    ax.set_xticklabels(['']*len(df_counts.index))  # hide text
                    for xtick, handle in zip(ax.get_xticks(), handles):
                        ax.scatter(
                            xtick,
                            -0.09,
                            color=handle.get_color(),
                            s=100,
                            clip_on=False,
                            transform=ax.get_xaxis_transform()
                        )
                    
                    c_handles = [
                        Patch(
                            facecolor=styles[i % len(styles)]["facecolor"],
                            hatch=styles[i % len(styles)]["hatch"],
                            edgecolor="black",
                            label=str(round(cat))
                        )
                        for i, cat in enumerate(df_counts.columns)
                    ]

                    ax.legend(
                        handles=c_handles,
                        title=col,
                        ncol=2,
                        loc="upper center",
                        bbox_to_anchor=(0.5, -0.25),
                        frameon=False
                    )

                    # Room for legend
                    plt.subplots_adjust(bottom=0.3)

                    continue

            x_var = g.x_vars[j]
            y_var = g.y_vars[i]

            if i == j:
                sns.histplot(
                    data=combined,
                    x=x_var,
                    hue="dataset",
                    hue_order=hue_order,
                    palette=palette,
                    bins=30,
                    stat="count",
                    element="step",
                    common_norm=False,
                    ax=ax,
                    legend=False
                )
                ax.set_ylabel("Count")
            else:
                sns.scatterplot(
                    data=combined,
                    x=x_var,
                    y=y_var,
                    hue="dataset",
                    hue_order=hue_order,
                    palette=palette,
                    s=10,
                    linewidth=0,
                    alpha=0.2,
                    ax=ax,
                    legend=False
                )
                ax.set_ylabel(y_var)

            ax.set_xlabel(x_var)

        # Legend subplot
        ax_leg = axes[-1]
        ax_leg.axis("off")

        ax_leg.legend(
            handles=handles,
            loc="center",
            frameon=False,
            title=legend_title
        )

        fig_extra.tight_layout()
        out_path = run_dir / "extra_plots_row_with_legend.png"
        fig_extra.savefig(out_path, dpi=160)
        plt.close(fig_extra)

    if show_training_data:
        for i, y_var in enumerate(g.y_vars):
            for j, x_var in enumerate(g.x_vars):
                ax = g.axes[i, j]
                if ax is None: continue
                ax.set_xlim(axis_limits[x_var])
                if i >= j:
                    ax.set_ylim(axis_limits[y_var])

    for ax in g.axes.flatten():
        if ax:
            for coll in ax.collections:
                coll.set_alpha(0.08)

    plt.setp(g._legend.get_texts(), fontsize=20, verticalalignment='center')
    g._legend.set_title(legend_title, prop={"size": 20})
    for handle in g._legend.legend_handles:
        handle.set_markersize(15)

    if save_png:
        out_path = run_dir / out_png_filename
        g.fig.savefig(out_path, dpi=160, bbox_inches="tight")
        print(f"[saved] Plot saved to {out_path.resolve()}")
    else:
        plt.show()

    plt.close(g.fig) # Close figure to free memory

def main():
    """CLI for generating investigation comparison plots.

    Parses command‑line arguments controlling:
        --run‑dir: Path to the run directory containing multiple `inv_*` investigation folders.
        --project-root: Base directory used to resolve relative CSV paths.
        --data-csv: Baseline training dataset CSV used as the reference distribution.

    Column selection:
        --cols: JSON list of column names to extract and plot.
        --col-slice: Two integers selecting a contiguous slice of columns instead of `cols`.

    Legend and labeling:
        --legend-config: JSON dict defining how investigation labels are formatted.
        --format-labels: Optional manual override for axis labels.

    Plotting behavior:
        --downsample: Maximum number of samples per dataset included in the plots.
        --show-train-data: Whether to include the baseline training data in the pairplot.
        --categorical: JSON list of columns treated as categorical, producing stacked bar charts.
        --extra-plots: List of (row, col) indices specifying additional scatter/hist panels to export.

    Output:
        -- save-png / no-save: Whether to save the figure or display it interactively.
        -- out-png: Filename for the main corner plot.

    The function ultimately produces:
        Pair plot comparing selected variables.
        Optional categorical bar charts.
        Optional extra scatter/hist panels with a shared legend.
    """

    parser = argparse.ArgumentParser(description="Generate corner plots for simulation investigations.")

    # --- File Path Arguments ---
    parser.add_argument("--run-dir", type=str, required=True, help="Path to the specific run directory containing inv_* folders.")
    parser.add_argument("--project-root", type=str, default="./output", help="Root directory of the project.")
    parser.add_argument("--data-csv", type=str, default="./training_data/train_set.csv", help="Filename or full path to the baseline training data CSV.")

    # --- Column Selection Arguments ---
    cols_default = [
        '/res_metrics/avg_drag',
        '/res_metrics/avg_l',
        '/res_metrics/comp_weights/lift_rotors', '/res_metrics/comp_weights/total',
        '/design_tree/lift_prop/tip_radius', '/design_tree/main_wings/0/root_cross_section/root_chord_percent',
        '/design_tree/main_wings/0/chord_root'
    ]
    parser.add_argument("--cols", type=json.loads, default=json.dumps(cols_default), help="JSON string of a list of column names to plot.")
    parser.add_argument("--col-slice", type=int, nargs=2, default=None, help="Alternative to --cols: provide two integers for column slicing, e.g., 72 77.")

    # --- Plotting Configuration ---
    legend_default = {
        'format_string': "Batt. Mass: {:.2f} kg",
        'keys': ['/design_tree/battery/mass']
    }
    parser.add_argument("--legend-config", type=json.loads, default=None ,#json.dumps(legend_default), 
                        help="JSON string for configuring legend names dynamically.")
    parser.add_argument("--downsample", type=int, default=3000, help="Number of samples per dataset to use for plotting. Set to 0 to disable.")

    # --- Output Arguments ---
    parser.add_argument("--save-png", dest='save_png', action='store_true', help="Save the plot as a PNG file (default).")
    parser.add_argument("--no-save", dest='save_png', action='store_false', help="Display the plot instead of saving it.")
    parser.set_defaults(save_png=True)
    parser.add_argument("--out-png", type=str, default="all_investigations_corner.png", help="Output filename for the saved plot.")
    parser.add_argument("--show-train-data", action='store_true', help="Display the training data.")
    parser.add_argument("--categorical", type=json.loads, default="[]", help="JSON string of a list of column names to make categorical.")
    parser.add_argument("--extra-plots",type=json.loads,default="[]",help="JSON list of [row, col] positions in the pairplot to save separately.")
    parser.add_argument("--format-labels",type=json.loads,default="[]",help="Manual xlabels.")

    args = parser.parse_args()

    generate_plots(
        run_dir=args.run_dir,
        project_root=args.project_root,
        data_csv=args.data_csv,
        cols=args.cols,
        col_slice=args.col_slice,
        legend_config=args.legend_config,
        downsample=args.downsample,
        save_png=args.save_png,
        out_png_filename=args.out_png,
        show_training_data = args.show_train_data,
        categorical = args.categorical,
        extra_plots=args.extra_plots,
        format_labels = args.format_labels
    )

if __name__ == "__main__":
    main()
