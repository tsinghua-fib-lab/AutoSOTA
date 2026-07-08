from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")


def setup_plot_style(fontsize: int = 14, linewidth: float = 2.5):
    plt.rcParams.update(
        {
            "font.size": fontsize,
            "font.family": "serif",
            "axes.linewidth": 1.2,
            "lines.linewidth": linewidth,
            "grid.linewidth": 0.8,
            "legend.fontsize": 12,
            "axes.labelsize": 16,
            "axes.titlesize": 18,
        }
    )


def plot_with_confidence_interval(
    ax,
    iterations: range,
    mean_values: List[float],
    std_values: List[float],
    color: str,
    label: str,
    linestyle: str = "-",
    marker: str = "o",
    markersize: int = 3,
    markevery: int = 1,
    linewidth: float = 2.0,
    alpha: float = 0.2,
    num_samples: int = 20,
    semilogy: bool = False,
):
    mean_vals = np.array(mean_values)
    std_vals = [2.093 * s / np.sqrt(num_samples) for s in std_values]

    if semilogy:
        ax.semilogy(
            iterations,
            mean_vals,
            color=color,
            linewidth=linewidth,
            label=label,
            linestyle=linestyle,
            marker=marker,
            markersize=markersize,
            markevery=markevery,
        )
        lower_bound = [max(mean_vals[j] - std_vals[j], 1e-10) for j in iterations]
        upper_bound = [mean_vals[j] + std_vals[j] for j in iterations]
        ax.fill_between(iterations, lower_bound, upper_bound, color=color, alpha=alpha)
    else:
        ax.plot(
            iterations,
            mean_vals,
            color=color,
            linewidth=linewidth,
            label=label,
            linestyle=linestyle,
            marker=marker,
            markersize=markersize,
            markevery=markevery,
        )
        ax.fill_between(
            iterations,
            [mean_vals[j] - std_vals[j] for j in iterations],
            [mean_vals[j] + std_vals[j] for j in iterations],
            color=color,
            alpha=alpha,
        )


def style_axes(
    ax,
    xlabel: str,
    ylabel: str,
    title: str,
    fontsize: int = 20,
    title_fontsize: int = 22,
    tick_fontsize: int = 16,
    legend_fontsize: int = 14,
    legend_loc: str = "lower right",
):
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_title(title, fontsize=title_fontsize)
    ax.tick_params(labelsize=tick_fontsize)
    ax.legend(
        fontsize=legend_fontsize,
        loc=legend_loc,
        frameon=True,
        fancybox=True,
        shadow=True,
        framealpha=0.9,
        edgecolor="black",
        facecolor="white",
    )
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("white")


def save_figure(output_path: Path, dpi: int = 300):
    plt.tight_layout()
    plt.savefig(f"{output_path}.pdf", bbox_inches="tight", dpi=dpi)
    plt.savefig(f"{output_path}.png", bbox_inches="tight", dpi=dpi)
    print(f"Saved plot to {output_path}.pdf and {output_path}.png")


def group_by_parameter(data: List[Dict], param: str = "nu") -> Dict[Any, List[Dict]]:
    grouped = {}
    for item in data:
        param_val = item.get(param, 0.1)
        if param_val not in grouped:
            grouped[param_val] = []
        grouped[param_val].append(item)
    return grouped


def get_algorithm_colors() -> Dict[str, str]:
    return {
        "pg": "#1f77b4",  # Blue
        "pepg": "#ff7f0e",  # Orange
        "mdrr": "#2ca02c",  # Green
        "reg_pepg": "#9467bd",  # Purple
    }


def get_algorithm_styles() -> Dict[str, Tuple[str, str]]:
    return {
        "pg": ("-", "o"),
        "pepg": ("--", "s"),
        "mdrr": (":", "^"),
        "reg_pepg": ("-.", "d"),
    }


def get_ablation_plot_colors() -> List[str]:
    return ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


def get_ablation_plot_styles() -> Tuple[List[str], List[str]]:
    linestyles = ["-", "--", "-.", ":", "-"]
    markers = ["o", "s", "^", "v", "D"]
    return linestyles, markers
