# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Utility and MSE comparison plotting functions."""

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

# Marker shapes for line plots (black and white friendly)
MARKERS = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h"]


def _sort_numeric(values: np.ndarray) -> np.ndarray:
    """Sort values numerically, handling both numeric and string representations.

    Args:
        values: Array of values (can be numeric or string).

    Returns:
        Sorted array in ascending numeric order.
    """

    def to_numeric(x):
        if isinstance(x, str):
            try:
                return float(x) if "." in x else int(x)
            except ValueError:
                return float("inf")
        return x

    return np.array(sorted(values, key=to_numeric))


Y_TITLES = {
    "util_diff_bestgrid": "Utility Gap",
    "util_pct_change_bestgrid": "Utility % Change (EW vs Best Grid)",
    "util_oracle_gap_bestgrid": "Oracle Gap (Oracle - EW)",
    "util_gap_closure_pct_bestgrid": "Gap Closure % (Best Grid)",
    "util_oracle_ratio_bestgrid": "EW / Oracle %",
    "util_diff_self": "Utility Improvement",
    "util_pct_change_self": "Utility % Change (EW vs EW Pre)",
    "util_oracle_gap_self": "Oracle Gap (Oracle - EW)",
    "util_gap_closure_pct_self": "Gap Closure % (EW Pre)",
    "util_oracle_ratio_self": "EW / Oracle %",
}

PARAM_LABELS = {
    "batch_size": r"$\text{CHECK}$ input size per iteration",
    "output_dim": r"m",
    "tol": r"$\varepsilon$",
    "top_k": "Top-K",
    "spread": "Spread",
}


def format_tol_label(tol_value):
    """Format tolerance value as 2^{-n}.

    Args:
        tol_value: Tolerance value (e.g., 0.0078125 for 2^-7).

    Returns:
        LaTeX formatted string like "$2^{-7}$".
    """
    import math

    # Handle string input
    if isinstance(tol_value, str):
        try:
            tol_value = float(tol_value)
        except ValueError:
            return str(tol_value)

    # Calculate log2 and round to nearest integer
    if tol_value > 0:
        log2_val = math.log2(tol_value)
        exponent = round(log2_val)

        # Verify it's actually a power of 2 (within tolerance)
        if abs(log2_val - exponent) < 0.01:
            return f"$2^{{{exponent}}}$"

    # Fallback to original value if not a clean power of 2
    return str(tol_value)


def format_param_value(param_name, value):
    """Format parameter value for display based on parameter type.

    Args:
        param_name: Name of the parameter (e.g., "tol", "batch_size").
        value: Value to format.

    Returns:
        Formatted string.
    """
    if param_name == "tol":
        return format_tol_label(value)
    return str(value)


MetricType = Literal[
    "util_diff_bestgrid",
    "util_pct_change_bestgrid",
    "util_oracle_gap_bestgrid",
    "util_gap_closure_pct_bestgrid",
    "util_oracle_ratio_bestgrid",
    "util_diff_self",
    "util_pct_change_self",
    "util_oracle_gap_self",
    "util_gap_closure_pct_self",
    "util_oracle_ratio_self",
]


def plot_line(
    summary: pl.DataFrame,
    x_param: str,
    color_param: str,
    y_metric: MetricType = "util_diff_bestgrid",
    output_path: str | None = None,
) -> plt.Figure:
    """Create line plot of metric vs sweep parameters.

    Args:
        summary: Summary DataFrame from compute_full_summary.
        x_param: Parameter for x-axis (batch_size, tol, output_dim).
        color_param: Parameter for color encoding.
        y_metric: Which metric to plot.
        output_path: If provided, save chart to this path.

    Returns:
        Matplotlib Figure object.
    """
    agg_df = summary.group_by([x_param, color_param]).agg(
        pl.col(y_metric).mean().alias(f"{y_metric}_mean"),
        pl.col(y_metric).std().alias(f"{y_metric}_std"),
    )

    pdf = agg_df.to_pandas()

    _sort_numeric(pdf[x_param].unique())
    color_vals = _sort_numeric(pdf[color_param].unique())

    fig, ax = plt.subplots(figsize=(10, 6))

    for idx, color_val in enumerate(color_vals):
        subset = pdf[pdf[color_param] == color_val]
        subset = subset.sort_values(x_param)

        ax.plot(
            subset[x_param],
            subset[f"{y_metric}_mean"],
            marker=MARKERS[idx % len(MARKERS)],
            label=format_param_value(color_param, color_val),
        )

    y_title = Y_TITLES.get(y_metric, y_metric)
    x_label = PARAM_LABELS.get(x_param, x_param)
    color_label = PARAM_LABELS.get(color_param, color_param)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_title)
    ax.legend(title=color_label, bbox_to_anchor=(1.05, 0.5), loc="center left", fontsize=8)
    ax.grid(True, alpha=0.3)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def plot_boxplot(
    summary: pl.DataFrame,
    x_param: str,
    color_param: str | None = None,
    y_metric: MetricType = "util_diff_bestgrid",
    output_path: str | None = None,
) -> plt.Figure:
    """Create box plot of metric vs sweep parameters.

    Args:
        summary: Summary DataFrame from compute_full_summary.
        x_param: Parameter for x-axis (batch_size, tol, output_dim).
        color_param: Optional parameter for color encoding.
        y_metric: Which metric to plot.
        output_path: If provided, save chart to this path.

    Returns:
        Matplotlib Figure object.
    """
    pdf = summary.to_pandas()
    y_title = Y_TITLES.get(y_metric, y_metric)

    fig, ax = plt.subplots(figsize=(10, 6))

    if color_param:
        unique_colors = _sort_numeric(pdf[color_param].unique())
        positions = []
        box_data = []
        pos_offset = 0
        x_vals = _sort_numeric(pdf[x_param].unique())

        for x_val in x_vals:
            x_sub_offset = 0
            for color_val in unique_colors:
                subset = pdf[(pdf[x_param] == x_val) & (pdf[color_param] == color_val)]
                data = subset[y_metric].dropna().values
                if len(data) > 0:
                    box_data.append(data)
                    positions.append(len(x_vals) * pos_offset + x_sub_offset + 1)
                x_sub_offset += 1
            pos_offset += len(unique_colors) + 1

        bp = ax.boxplot(box_data, positions=positions, widths=0.6, patch_artist=True)

        color_map = plt.cm.tab10.colors
        for patch, color_val in zip(bp["boxes"], unique_colors * len(x_vals)):
            patch.set_facecolor(color_map[list(unique_colors).index(str(color_val)) % len(color_map)])
            patch.set_alpha(0.7)

        ax.set_xticks([(i * (len(unique_colors) + 1) + len(unique_colors) / 2) for i in range(len(x_vals))])
        ax.set_xticklabels(x_vals)

        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, facecolor=color_map[i % len(color_map)], alpha=0.7)
            for i in range(len(unique_colors))
        ]
        color_label = PARAM_LABELS.get(color_param, color_param)
        formatted_colors = [format_param_value(color_param, c) for c in unique_colors]
        ax.legend(
            legend_elements,
            formatted_colors,
            title=color_label,
            bbox_to_anchor=(1.05, 0.5),
            loc="center left",
            fontsize=8,
        )
    else:
        x_vals = _sort_numeric(pdf[x_param].unique())
        box_data = [pdf[pdf[x_param] == x_val][y_metric].dropna().values for x_val in x_vals]

        ax.boxplot(box_data, labels=x_vals)
        x_label = PARAM_LABELS.get(x_param, x_param)
        ax.set_xlabel(x_label)

    ax.set_ylabel(y_title)
    ax.grid(True, alpha=0.3, axis="y")

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def plot_heatmap(
    summary: pl.DataFrame,
    x_param: str,
    y_param: str,
    color_metric: MetricType = "util_diff_bestgrid",
    output_path: str | None = None,
) -> plt.Figure:
    """Create heatmap of metric across two sweep parameters.

    Args:
        summary: Summary DataFrame from compute_full_summary.
        x_param: Parameter for x-axis.
        y_param: Parameter for y-axis.
        color_metric: Which metric to use for color encoding.
        output_path: If provided, save chart to this path.

    Returns:
        Matplotlib Figure object.
    """
    agg_df = summary.group_by([x_param, y_param]).agg(
        pl.col(color_metric).mean().alias(f"{color_metric}_mean"),
    )

    pdf = agg_df.to_pandas()

    pdf.pivot(index=y_param, columns=x_param, values=f"{color_metric}_mean")

    x_vals = _sort_numeric(pdf[x_param].unique())
    y_vals = _sort_numeric(pdf[y_param].unique())

    heatmap_data = np.full((len(y_vals), len(x_vals)), np.nan)
    for y_idx, y_val in enumerate(y_vals):
        for x_idx, x_val in enumerate(x_vals):
            val = pdf[(pdf[y_param] == y_val) & (pdf[x_param] == x_val)][f"{color_metric}_mean"].values
            if len(val) > 0:
                heatmap_data[y_idx, x_idx] = val[0]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(heatmap_data, cmap="viridis", aspect="auto")

    ax.set_xticks(np.arange(len(x_vals)))
    ax.set_yticks(np.arange(len(y_vals)))
    ax.set_xticklabels(x_vals)
    ax.set_yticklabels(y_vals)

    for i in range(len(y_vals)):
        for j in range(len(x_vals)):
            val = heatmap_data[i, j]
            if not np.isnan(val):
                text_color = "black" if val > np.nanmedian(heatmap_data) else "white"
                ax.text(
                    j,
                    i,
                    f"{val:.3f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=9,
                )

    color_title = Y_TITLES.get(color_metric, color_metric)
    x_label = PARAM_LABELS.get(x_param, x_param)
    y_label = PARAM_LABELS.get(y_param, y_param)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    fig.colorbar(im, ax=ax, label=color_title)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def plot_faceted_line(
    summary: pl.DataFrame,
    x_param: str,
    color_param: str,
    facet_param: str,
    y_metric: MetricType = "util_diff_bestgrid",
    n_cols: int = 2,
    subplot_width: float = 5.0,
    subplot_height: float = 3.0,
    log_x: bool = False,
) -> plt.Figure:
    """Create faceted line plot of metric vs sweep parameters.

    Args:
        summary: Summary DataFrame from compute_full_summary.
        x_param: Parameter for x-axis.
        color_param: Parameter for color encoding.
        facet_param: Parameter to facet by (creates small multiples).
        y_metric: Which metric to plot.
        n_cols: Number of columns in facet grid.
        subplot_width: Width of each subplot in inches.
        subplot_height: Height of each subplot in inches.
        log_x: If True, use log2 scale for x-axis.

    Returns:
        Matplotlib Figure object.
    """
    agg_df = summary.group_by([x_param, color_param, facet_param]).agg(
        pl.col(y_metric).mean().alias(f"{y_metric}_mean"),
    )

    pdf = agg_df.to_pandas()
    y_title = Y_TITLES.get(y_metric, y_metric)

    facet_vals = _sort_numeric(pdf[facet_param].unique())
    color_vals = _sort_numeric(pdf[color_param].unique())

    n_rows = (len(facet_vals) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(subplot_width * n_cols, subplot_height * n_rows),
        sharex=True,
        sharey=False,
    )
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    x_label = PARAM_LABELS.get(x_param, x_param)
    color_label = PARAM_LABELS.get(color_param, color_param)

    for idx, facet_val in enumerate(facet_vals):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        for color_idx, color_val in enumerate(color_vals):
            subset = pdf[(pdf[facet_param] == facet_val) & (pdf[color_param] == color_val)]
            subset = subset.sort_values(x_param)
            if len(subset) > 0:
                ax.plot(
                    subset[x_param],
                    subset[f"{y_metric}_mean"],
                    marker=MARKERS[color_idx % len(MARKERS)],
                    label=format_param_value(color_param, color_val),
                    markersize=4,
                )

        if log_x:
            ax.set_xscale("log", base=2)

        # Only show legend on first plot
        if idx == 0:
            ax.legend(
                fontsize=8,
                title=color_label,
                bbox_to_anchor=(1.05, 0.5),
                loc="center left",
            )

        ax.set_title(f"{PARAM_LABELS.get(facet_param, facet_param)} = {format_param_value(facet_param, facet_val)}")
        ax.grid(True, alpha=0.3)

    for idx in range(len(facet_vals), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis("off")

    # Add shared axis labels with proper spacing
    fig.supxlabel(x_label, fontsize=12)
    fig.supylabel(y_title, fontsize=12)

    plt.tight_layout()

    return fig


def plot_utility_comparison(
    summary: pl.DataFrame,
    x_param: str,
    facet_param: str | None = None,
    output_path: str | None = None,
    n_cols: int = 2,
    subplot_width: float = 5.0,
    subplot_height: float = 3.5,
) -> plt.Figure:
    """Create multi-line plot comparing oracle, EW, and best_grid utility.

    Shows all three utility values (oracle, ew_final, best_grid) as separate lines
    to provide full context on the scale of improvement.

    Args:
        summary: Summary DataFrame from compute_full_summary.
        x_param: Parameter for x-axis (batch_size, tol, output_dim).
        facet_param: Optional parameter to facet by.
        output_path: If provided, save chart to this path.
        n_cols: Number of columns in facet grid.
        subplot_width: Width of each subplot in inches.
        subplot_height: Height of each subplot in inches.

    Returns:
        Matplotlib Figure object.
    """
    agg_cols = [x_param]
    if facet_param:
        agg_cols.append(facet_param)

    agg_df = summary.group_by(agg_cols).agg(
        pl.col("oracle_util").mean().alias("Oracle"),
        pl.col("ew_final_util").mean().alias("EW (Equally Weighted)"),
        pl.col("ew_pre_util").mean().alias("EW Pre"),
        pl.col("best_grid_util").mean().alias("Best Grid Cell"),
    )

    pdf = agg_df.to_pandas()

    value_cols = ["Oracle", "EW (Equally Weighted)", "EW Pre", "Best Grid Cell"]
    colors = ["#2ecc71", "#3498db", "#9b59b6", "#e74c3c"]
    linestyles = ["-", "-", "--", ":"]

    if facet_param:
        facet_vals = _sort_numeric(pdf[facet_param].unique())
        n_rows = (len(facet_vals) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(subplot_width * n_cols, subplot_height * n_rows),
            sharex=True,
            sharey=False,
        )
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        x_label = PARAM_LABELS.get(x_param, x_param)
        PARAM_LABELS.get(facet_param, facet_param)

        for idx, facet_val in enumerate(facet_vals):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]

            subset = pdf[pdf[facet_param] == facet_val]
            subset = subset.sort_values(x_param)

            for line_idx, (col_name, color, ls) in enumerate(zip(value_cols, colors, linestyles)):
                if col_name in subset.columns:
                    ax.plot(
                        subset[x_param],
                        subset[col_name],
                        marker=MARKERS[line_idx % len(MARKERS)],
                        label=col_name,
                        color=color,
                        linestyle=ls,
                        markersize=5,
                    )

            # Only show legend on first plot
            if idx == 0:
                ax.legend(fontsize=8, bbox_to_anchor=(1.05, 0.5), loc="center left")

            ax.grid(True, alpha=0.3)

        for idx in range(len(facet_vals), n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis("off")

        # Add shared axis labels with proper spacing
        fig.supxlabel(x_label, fontsize=12)
        fig.supylabel("Utility", fontsize=12)

    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        pdf = pdf.sort_values(x_param)

        x_label = PARAM_LABELS.get(x_param, x_param)

        for line_idx, (col_name, color, ls) in enumerate(zip(value_cols, colors, linestyles)):
            if col_name in pdf.columns:
                ax.plot(
                    pdf[x_param],
                    pdf[col_name],
                    marker=MARKERS[line_idx % len(MARKERS)],
                    label=col_name,
                    color=color,
                    linestyle=ls,
                )

        ax.set_xlabel(x_label)
        ax.set_ylabel("Utility")
        ax.legend(bbox_to_anchor=(1.05, 0.5), loc="center left", fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def plot_mse_history(
    raw_df: pl.DataFrame,
    facet_params: list[str],
    seed: int | None = None,
    x_axis: Literal["samples", "iterations"] = "samples",
    log_x: bool = False,
    n_cols: int = 2,
    subplot_width: float = 6.0,
    subplot_height: float = 4.0,
) -> plt.Figure:
    """Create faceted MSE history plots over sample complexity or iterations.

    Args:
        raw_df: Raw metrics DataFrame from load_sweep_data (not summary).
        facet_params: List of parameters to create facets for (e.g., ["batch_size", "output_dim"]).
                     If 2 params: first goes on columns, second on rows.
        seed: If provided, show individual experiment lines for this seed.
               If None, show mean across all seeds with std band.
        x_axis: "samples" for t * batch_size (sample complexity),
                "iterations" for t (iteration count).
        log_x: If True, use log2 scale for x-axis.
        n_cols: Number of columns in facet grid.
        subplot_width: Width of each subplot in inches.
        subplot_height: Height of each subplot in inches.

    Returns:
        Matplotlib Figure object.
    """
    mse_df = raw_df.filter(pl.col("name") == "mse")

    # Ensure t and batch_size are integers to avoid fractional x-axis values
    if x_axis == "samples":
        mse_df = mse_df.with_columns(
            (pl.col("t").cast(pl.Int64) * pl.col("batch_size").cast(pl.Int64)).alias("x_value")
        )
        x_title = "Samples (t × batch_size)"
    else:
        mse_df = mse_df.with_columns(pl.col("t").cast(pl.Int64).alias("x_value"))
        x_title = "Iterations (t)"

    group_cols = facet_params + ["tol", "x_value"]

    if seed is not None:
        mse_df = mse_df.filter(pl.col("seed") == seed)
        pdf = mse_df.to_pandas()
    else:
        pdf = (
            mse_df.group_by(group_cols)
            .agg(
                pl.col("value").mean().alias("mse_mean"),
                pl.col("value").std().alias("mse_std"),
            )
            .to_pandas()
        )

    # Filter out any x_values that are between 0 and 1 (shouldn't happen but safety check)
    if len(pdf) > 0:
        min_x = pdf["x_value"].min()
        max_x = pdf["x_value"].max()
        if max_x > 0 and max_x <= 1 and min_x >= 0:
            # This indicates a problem - likely tol values being used instead
            print(f"WARNING: x_value range [{min_x}, {max_x}] is suspicious. Check data.")
        pdf = pdf[~((pdf["x_value"] > 0) & (pdf["x_value"] < 1))]

    tols = _sort_numeric(pdf["tol"].unique())
    colors = plt.cm.tab10.colors

    # Set up faceting based on number of parameters
    if len(facet_params) == 2:
        # Two parameters: first on columns, second on rows
        col_param = facet_params[0]
        row_param = facet_params[1]

        col_vals = _sort_numeric(pdf[col_param].unique())
        row_vals = _sort_numeric(pdf[row_param].unique())

        n_rows_actual = len(row_vals)
        n_cols_actual = len(col_vals)

        fig, axes = plt.subplots(
            n_rows_actual,
            n_cols_actual,
            figsize=(subplot_width * n_cols_actual, subplot_height * n_rows_actual),
            sharex=False,
            sharey=False,
        )
        if n_rows_actual == 1 and n_cols_actual == 1:
            axes = np.array([[axes]])
        elif n_rows_actual == 1:
            axes = axes.reshape(1, -1)
        elif n_cols_actual == 1:
            axes = axes.reshape(-1, 1)

        PARAM_LABELS.get(col_param, col_param)

        for row_idx, row_val in enumerate(row_vals):
            for col_idx, col_val in enumerate(col_vals):
                ax = axes[row_idx, col_idx]

                mask = (pdf[col_param] == col_val) & (pdf[row_param] == row_val)
                subset = pdf[mask].sort_values("x_value")

                for tol_idx, tol in enumerate(tols):
                    tol_subset = subset[subset["tol"] == tol]
                    if len(tol_subset) > 0:
                        if seed is not None:
                            ax.plot(
                                tol_subset["x_value"],
                                tol_subset["value"],
                                color=colors[tol_idx % len(colors)],
                                marker=MARKERS[tol_idx % len(MARKERS)],
                                markersize=3,
                                label=format_tol_label(tol),
                            )
                        else:
                            ax.plot(
                                tol_subset["x_value"],
                                tol_subset["mse_mean"],
                                color=colors[tol_idx % len(colors)],
                                marker=MARKERS[tol_idx % len(MARKERS)],
                                markersize=3,
                                label=format_tol_label(tol),
                            )
                            ax.fill_between(
                                tol_subset["x_value"],
                                tol_subset["mse_mean"] - tol_subset["mse_std"],
                                tol_subset["mse_mean"] + tol_subset["mse_std"],
                                color=colors[tol_idx % len(colors)],
                                alpha=0.2,
                            )

                if log_x:
                    ax.set_xscale("log", base=2)

                # Only show legend on first plot
                if row_idx == 0 and col_idx == 0:
                    ax.legend(
                        fontsize=8,
                        title=PARAM_LABELS["tol"],
                        bbox_to_anchor=(1.05, 0.5),
                        loc="center left",
                    )

                ax.grid(True, alpha=0.3)

        # Add shared axis labels with proper spacing
        fig.supxlabel(x_title, fontsize=12)
        fig.supylabel("MSE", fontsize=12)
    else:
        # Fallback to original layout for non-2-param cases
        unique_facet_combos = pdf[facet_params].drop_duplicates().values
        n_combos = len(unique_facet_combos)

        n_rows_actual = (n_combos + n_cols - 1) // n_cols

        fig, axes = plt.subplots(
            n_rows_actual,
            n_cols,
            figsize=(subplot_width * n_cols, subplot_height * n_rows_actual),
            sharex=False,
            sharey=False,
        )
        if n_rows_actual == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows_actual == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        for idx, combo in enumerate(unique_facet_combos):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]

            mask = True
            for i, param in enumerate(facet_params):
                mask = mask & (pdf[param] == combo[i])

            subset = pdf[mask].sort_values("x_value")

            for tol_idx, tol in enumerate(tols):
                tol_subset = subset[subset["tol"] == tol]
                if len(tol_subset) > 0:
                    if seed is not None:
                        ax.plot(
                            tol_subset["x_value"],
                            tol_subset["value"],
                            color=colors[tol_idx % len(colors)],
                            marker=MARKERS[tol_idx % len(MARKERS)],
                            markersize=3,
                            label=str(tol),
                        )
                    else:
                        ax.plot(
                            tol_subset["x_value"],
                            tol_subset["mse_mean"],
                            color=colors[tol_idx % len(colors)],
                            marker=MARKERS[tol_idx % len(MARKERS)],
                            markersize=3,
                            label=str(tol),
                        )
                        ax.fill_between(
                            tol_subset["x_value"],
                            tol_subset["mse_mean"] - tol_subset["mse_std"],
                            tol_subset["mse_mean"] + tol_subset["mse_std"],
                            color=colors[tol_idx % len(colors)],
                            alpha=0.2,
                        )

            if log_x:
                ax.set_xscale("log", base=2)

            # Only show legend on the last plot
            if idx == len(unique_facet_combos) - 1:
                ax.legend(
                    fontsize=8,
                    title=PARAM_LABELS["tol"],
                    bbox_to_anchor=(1.05, 0.5),
                    loc="center left",
                )

            ax.grid(True, alpha=0.3)

        for idx in range(n_combos, n_rows_actual * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis("off")

        # Add shared axis labels with proper spacing
        fig.supxlabel(x_title, fontsize=12)
        fig.supylabel("MSE", fontsize=12)

    plt.tight_layout()

    return fig


def plot_utility_improvement(
    summary: pl.DataFrame,
    x_param: str,
    color_param: str,
    y_metric: MetricType = "util_diff_bestgrid",
    output_path: str | None = None,
) -> plt.Figure:
    """Deprecated: Use plot_line instead."""
    return plot_line(summary, x_param, color_param, y_metric, output_path)
