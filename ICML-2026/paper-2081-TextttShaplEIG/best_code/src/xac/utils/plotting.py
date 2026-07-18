from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import matplotlib as mpl
import numpy as np
import torch
from matplotlib import pyplot as plt

logging.getLogger("matplotlib").setLevel(logging.WARNING)

PLOT_STYLE = {
    "figure.dpi": 1000,
    "savefig.dpi": 1000,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 5.5,
    "axes.linewidth": 0.6,
    "grid.linewidth": 0.3,
    "grid.alpha": 0.5,
    "axes.grid": False,
    "lines.linewidth": 1.2,
    "lines.markersize": 3.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.constrained_layout.use": True,
}

SHAPLEIG_CATEGORY_COLORS = {
    "EIG-FP": "#AA4499",
    "Random": "#88CCEE",
    "EIG-EP": "#66AADD",
    "LeverageSHAP-GP": "#4477AA",
    "KernelSHAP": "#DD8899",
    "LeverageSHAP": "#CC6677",
    "Regression MSR": "#BB5566",
    "Permutation Sampling": "#EEAABB",
    "SVARM": "#117733",
}


def _get_category_colors(categories: List[str]) -> List[str]:
    shapleig_categories = set(SHAPLEIG_CATEGORY_COLORS)

    if len(categories) == 3:
        return ["#EE6677", "#4477AA", "#228833"]

    if set(categories) <= shapleig_categories:
        return [SHAPLEIG_CATEGORY_COLORS[category] for category in categories]

    cmap = plt.get_cmap("tab10")
    return [cmap(i) for i in range(len(categories))]


def _plot_main_series(
    ax,
    x_values: np.ndarray,
    y_values: np.ndarray,
    color,
    label: str,
    width_white: float,
    width_colored: float,
    outline_alpha: float,
    line_alpha: float,
) -> None:
    ax.plot(
        x_values,
        y_values,
        label="_nolegend_",
        color="white",
        alpha=outline_alpha,
        linewidth=width_white,
    )
    ax.plot(
        x_values,
        y_values,
        label=label,
        color=color,
        alpha=line_alpha,
        linewidth=width_colored,
    )


def _add_compact_legend(ax, legend_placement_bottom: bool) -> None:
    handles, labels = ax.get_legend_handles_labels()
    deduped = {}

    for handle, label in zip(handles, labels):
        if not label or label.startswith("_") or label in deduped:
            continue
        deduped[label] = handle

    if not deduped:
        return

    legend_location = "lower right" if legend_placement_bottom else "upper right"
    leg = ax.legend(
        list(deduped.values()),
        list(deduped.keys()),
        loc=legend_location,
        frameon=True,
        fontsize=5.5,
        handlelength=1.2,
        handletextpad=0.35,
        borderpad=0.25,
        labelspacing=0.2,
        borderaxespad=0.3,
    )
    frame = leg.get_frame()
    frame.set_facecolor("white")
    frame.set_alpha(0.8)
    frame.set_edgecolor((0.8, 0.8, 0.8))
    frame.set_linewidth(0.6)


def get_plot_name(abbrev: str) -> str:
    metric_name_map = {
        "mse": "MSE",
        "mse_normalized": "Norm. MSE",
        "mae": "MAE",
        "nlpd": "NLPD",
        "nlpd_noisy": "NLPD (Noisy)",
        "ce_loss": "CE",
        "ce_loss_noisy": "CE (Noisy)",
        "accuracy": "Accuracy",
        "accuracy_noisy": "Accuracy (Noisy)",
        "acq_fun_duration": "EIG Computation Time (s)",
        "hp_fit_duration": "GP Fitting Time (s)",
    }

    dataset_name_map = {
        "Branin": "Branin (2d)",
        "StyblinskiTang_3": "Styblinski-Tang (3d)",
        "Hartmann_3": "Hartmann (3d)",
        "Hartmann_6": "Hartmann (6d)",
        "nn_torch": "MLP",
        "rf": "Random Forest",
        "xt": "Extra Trees",
        "catboost": "CatBoost",
        "lightgbm": "LightGBM",
        "knn": "K-Nearest Neighbors",
        "xgboost": "XGBoost",
    }

    acq_fun_name_map = {
        "EIG-EP": "US (GP-based)",  # r"EIG $f_Z$",
        "EIG-FP": "ShaplEIG",  # r $\ell(f_Z)$ (ours)
        "Random": "Random (GP-based)",
    }

    if abbrev in metric_name_map:
        return metric_name_map[abbrev]
    elif abbrev in dataset_name_map:
        return dataset_name_map[abbrev]
    elif abbrev in acq_fun_name_map:
        return acq_fun_name_map[abbrev]
    else:
        return abbrev


def plot_data_hists(train_X, train_Y, attr_names, run_dir):

    out_dir = Path(run_dir) / "data/attributes"
    out_dir.mkdir(parents=True, exist_ok=True)

    # helper: 1-D tensor -> numpy flat array
    to_np = lambda t: t.detach().cpu().view(-1).numpy()

    # (1) feature histograms ------------------------------------------------
    for i, name in enumerate(attr_names):
        plt.figure()
        plt.hist(to_np(train_X[:, i]), bins="auto")
        plt.title(name)
        plt.tight_layout()
        plt.savefig(out_dir / f"hist_{name}.png", dpi=500, bbox_inches="tight")
        plt.close()

    # (2) target histogram --------------------------------------------------
    plt.figure()
    plt.hist(to_np(train_Y), bins="auto")
    plt.title("target")
    plt.tight_layout()
    plt.savefig(out_dir / "hist_target.png", dpi=500, bbox_inches="tight")
    plt.close()


def plot_trajectory(
    main_data: List[torch.Tensor],  # e.g. mean trajectories
    granular_data: Optional[List[torch.Tensor]],  # e.g. indivcidual runs
    plot_title: str,
    y_label: str,
    categories: List[
        str
    ],  # Categories provided as list entries (e.g. acquisition function names)
    path: str,
    plot_std: bool = False,  # Whether to plot granular data as standard deviation error bars
    plot_individual_runs: bool = False,  # Whether to plot granular data as individual runs
    individual_run_alpha: float = 0.35,  # 0.1,  # Alpha value for individual run plotting
    y_mean_scale: bool = False,  # Whether to scale y-axis based on mean values (as opposed to individual runs)
    y_range_top: Optional[float] = None,  # Optional y-axis top limit
    y_range_bottom: Optional[float] = None,  # Optional y-axis bottom limit
    y_log_scale: bool = False,  # Whether to use log scale for y-axis
    y_maximum: Optional[
        float
    ] = None,  # Optional maximum y-axis limit (e.g. for metrics like MAE, MSE) #
    y_minimum: Optional[float] = None,  # Optional minimum y-axis limit
    legend: bool = True,  # Whether to show legend
    legend_placement_bottom: bool = True,  # Whether to place legend at bottom
    size_X0: int = 0,  # Shift x axis if required
    width_fraction_denominator: float = 3.0,  # 1.0 = full width, 2 = half width, 3 = third width
):
    AISTATS_TEXTWIDTH_IN = 6.75

    def get_figsize(n=3, gutter_frac=0.02):
        width = (AISTATS_TEXTWIDTH_IN / n) * (1.0 - gutter_frac)
        return (width, width)  # square

    mpl.rcParams.update(PLOT_STYLE)
    figsize = get_figsize(n=width_fraction_denominator)
    fig, ax = plt.subplots(figsize=figsize)

    x_grid = torch.arange(size_X0, size_X0 + main_data[0].shape[-1])
    colors = _get_category_colors(categories)

    width_white = 1.7
    width_colored = 1.5

    shapleig_index = None

    for category_idx in range(len(categories)):
        temp_main_data = main_data[category_idx].squeeze().numpy()
        temp_granular_data = (
            granular_data[category_idx].squeeze().numpy()
            if granular_data is not None
            else None
        )

        # Plot main data
        if categories[category_idx] != "EIG-FP":
            _plot_main_series(
                ax=ax,
                x_values=x_grid.numpy(),
                y_values=temp_main_data,
                label=get_plot_name(categories[category_idx]),
                color=colors[category_idx] if colors is not None else None,
                width_white=width_white,
                width_colored=width_colored,
                outline_alpha=0.8,
                line_alpha=0.9,
            )

            # Plot granular data
            if granular_data is not None:
                if plot_individual_runs:
                    # Plot individual runs
                    if temp_granular_data.ndim > 1:
                        for i in range(temp_granular_data.shape[0]):
                            ax.plot(
                                x_grid.numpy(),
                                temp_granular_data[i],
                                color=(
                                    colors[category_idx]
                                    if colors is not None
                                    else None
                                ),
                                alpha=individual_run_alpha,
                                marker="o",
                                markersize=0.5,  # smaller dots
                                linestyle="-",
                                linewidth=0.5,  # thinner line
                            )
                elif plot_std:
                    # Plot standard deviation error bars
                    yerr = temp_granular_data.std(
                        axis=0, ddof=1
                    ).squeeze() / np.sqrt(temp_granular_data.shape[0])
                    ax.errorbar(
                        x_grid.numpy(),
                        temp_main_data,
                        yerr=yerr,
                        color=colors[category_idx] if colors is not None else None,
                        marker="o",
                        markersize=0,
                        linewidth=0,
                        elinewidth=0.4,  # thinner error bar lines
                        capsize=0,  # no caps (removes the "phi"-like ends)
                        alpha=individual_run_alpha,
                    )

        else:
            shapleig_index = category_idx

    if shapleig_index is not None:
        # Ensure that ours is plotted last
        temp_main_data = main_data[shapleig_index].squeeze().numpy()
        temp_granular_data = (
            granular_data[shapleig_index].squeeze().numpy()
            if granular_data is not None
            else None
        )

        # Plot main data
        _plot_main_series(
            ax=ax,
            x_values=x_grid.numpy(),
            y_values=temp_main_data,
            label=get_plot_name(categories[shapleig_index]),
            color=colors[shapleig_index] if colors is not None else None,
            width_white=width_white,
            width_colored=width_colored,
            outline_alpha=0.7,
            line_alpha=0.85,
        )

        if temp_granular_data is not None:
            yerr = temp_granular_data.std(axis=0, ddof=1).squeeze() / np.sqrt(
                temp_granular_data.shape[0]
            )
            ax.errorbar(
                x_grid.numpy(),
                temp_main_data,
                yerr=yerr,
                color=colors[shapleig_index] if colors is not None else None,
                marker="o",
                markersize=0,
                linewidth=0,
                elinewidth=0.4,  # thinner error bar lines
                capsize=0,  # no caps (removes the "phi"-like ends)
                alpha=individual_run_alpha,
            )

    ax.set_xlabel("Evaluations")
    ax.set_ylabel(y_label)

    if y_range_top is not None or y_range_bottom is not None:
        lo, hi = ax.get_ylim()
        if y_range_bottom is not None:
            lo = y_range_bottom
        if y_range_top is not None:
            hi = y_range_top
        ax.set_ylim(bottom=lo, top=hi)

    if y_mean_scale and granular_data is not None:
        # Set y-lim based on mean values
        upper_lim = torch.concat(main_data).max()
        lower_lim = torch.concat(main_data).min()
        ax.set_ylim(bottom=lower_lim, top=upper_lim)

    if y_log_scale:
        ax.set_yscale("log")

    if y_maximum is not None:
        if y_minimum is None:
            y_minimum, hi = ax.get_ylim()

        ax.set_ylim(bottom=y_minimum, top=y_maximum)

    if plot_title is not None:
        ax.set_title(plot_title, pad=6)

    if legend:
        _add_compact_legend(ax, legend_placement_bottom)

    path.parent.mkdir(parents=True, exist_ok=True)

    path = Path(path).with_suffix(".pdf")

    fig.savefig(f"{path}", format="pdf")  # "png")
    plt.close(fig)
