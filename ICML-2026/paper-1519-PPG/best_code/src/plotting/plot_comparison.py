from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from src.plotting.plot_utils import (
    get_algorithm_colors,
    get_algorithm_styles,
    plot_with_confidence_interval,
    save_figure,
    style_axes,
)


def plot_algorithm_comparison(
    pg_data: List[Dict],
    pepg_data: List[Dict],
    mdrr_data: List[Dict],
    reg_pepg_data: List[Dict],
    output_dir: str,
    param: str = "nu",
):
    from src.plotting.plot_utils import group_by_parameter

    pg_by_param = group_by_parameter(pg_data, param)
    pepg_by_param = group_by_parameter(pepg_data, param)
    mdrr_by_param = group_by_parameter(mdrr_data, param)
    reg_pepg_by_param = group_by_parameter(reg_pepg_data, param)

    all_param_values = sorted(
        set(
            list(pg_by_param.keys())
            + list(pepg_by_param.keys())
            + list(mdrr_by_param.keys())
            + list(reg_pepg_by_param.keys())
        )
    )

    if not all_param_values:
        print(f"No {param} values found for comparison")
        return

    colors = get_algorithm_colors()
    styles = get_algorithm_styles()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for param_val in all_param_values:
        _plot_algorithm_data(
            ax1, ax2, pg_by_param, param_val, "pg", "RPO FS", colors["pg"], styles["pg"]
        )
        _plot_algorithm_data(
            ax1,
            ax2,
            pepg_by_param,
            param_val,
            "pepg",
            "PePG",
            colors["pepg"],
            styles["pepg"],
        )
        _plot_algorithm_data(
            ax1,
            ax2,
            mdrr_by_param,
            param_val,
            "mdrr",
            "MDRR",
            colors["mdrr"],
            styles["mdrr"],
        )
        _plot_algorithm_data(
            ax1,
            ax2,
            reg_pepg_by_param,
            param_val,
            "reg_pepg",
            "Reg PePG",
            colors["reg_pepg"],
            styles["reg_pepg"],
        )

    style_axes(
        ax1,
        "Iteration t",
        "V^π",
        "Value Function Evolution",
        fontsize=20,
        title_fontsize=22,
        tick_fontsize=16,
        legend_fontsize=14,
    )
    style_axes(
        ax2,
        "Iteration t",
        "||d_{t+1} - d_t||_2",
        "Policy Stability",
        fontsize=20,
        title_fontsize=22,
        tick_fontsize=16,
        legend_fontsize=14,
    )

    param_str = "_".join([f"{param}{pv}" for pv in all_param_values])
    output_path = Path(output_dir) / f"comparison_pg_pepg_mdrr_{param_str}"

    save_figure(output_path, dpi=300)
    print(f"Comparison plot saved to {output_path}.pdf and {output_path}.png")


def _plot_algorithm_data(
    ax1, ax2, data_by_param, param_val, algo_key, algo_label, color, style
):
    linestyle, marker = style

    if param_val not in data_by_param or not data_by_param[param_val]:
        return

    data = data_by_param[param_val][0]

    if "v_values_mean" in data and data["v_values_mean"]:
        iterations = range(len(data["v_values_mean"]))
        plot_with_confidence_interval(
            ax1,
            iterations,
            data["v_values_mean"],
            data.get("v_values_std", [0] * len(data["v_values_mean"])),
            color=color,
            label=algo_label,
            linestyle=linestyle,
            marker=marker,
            markersize=3,
            linewidth=2,
            alpha=0.2,
        )

    if "d_diff_mean" in data and data["d_diff_mean"]:
        iterations = range(len(data["d_diff_mean"]))
        plot_with_confidence_interval(
            ax2,
            iterations,
            data["d_diff_mean"],
            data.get("d_diff_std", [0] * len(data["d_diff_mean"])),
            color=color,
            label=algo_label,
            linestyle=linestyle,
            marker=marker,
            markersize=3,
            linewidth=2,
            alpha=0.2,
            semilogy=True,
        )


def plot_individual_algorithm(
    data: List[Dict], algorithm_name: str, output_dir: str, param: str = "nu"
):
    from src.plotting.plot_utils import group_by_parameter

    Path(output_dir).mkdir(exist_ok=True)

    if not data:
        print(f"No data found for {algorithm_name}")
        return

    by_param = group_by_parameter(data, param)
    param_values = sorted(by_param.keys())

    if not param_values:
        print(f"No {param} values found for {algorithm_name}")
        return

    colors = ["blue", "green", "red", "orange", "purple", "brown", "pink"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for i, param_val in enumerate(param_values):
        if param_val not in by_param:
            continue

        color = colors[i % len(colors)]
        data_item = by_param[param_val][0]

        if "v_values_mean" in data_item and data_item["v_values_mean"]:
            iterations = range(len(data_item["v_values_mean"]))
            plot_with_confidence_interval(
                ax1,
                iterations,
                data_item["v_values_mean"],
                data_item.get("v_values_std", [0] * len(data_item["v_values_mean"])),
                color=color,
                label=f"{param}={param_val}",
                marker="o",
                markersize=3,
                linewidth=2,
                alpha=0.3,
            )

        if "d_diff_mean" in data_item and data_item["d_diff_mean"]:
            iterations = range(len(data_item["d_diff_mean"]))
            plot_with_confidence_interval(
                ax2,
                iterations,
                data_item["d_diff_mean"],
                data_item.get("d_diff_std", [0] * len(data_item["d_diff_mean"])),
                color=color,
                label=f"{param}={param_val}",
                marker="o",
                markersize=3,
                linewidth=2,
                alpha=0.3,
                semilogy=True,
            )

    style_axes(
        ax1,
        "Iteration t",
        "V^π",
        f"{algorithm_name} - Value Function Evolution",
        fontsize=20,
        title_fontsize=22,
        tick_fontsize=16,
        legend_fontsize=12,
    )
    style_axes(
        ax2,
        "Iteration t",
        "||d_{t+1} - d_t||_2",
        f"{algorithm_name} - Policy Stability",
        fontsize=20,
        title_fontsize=22,
        tick_fontsize=16,
        legend_fontsize=12,
    )

    output_path = (
        Path(output_dir)
        / f"individual_{algorithm_name.lower().replace(' ', '_')}_results"
    )
    save_figure(output_path, dpi=300)
    print(f"{algorithm_name} plot saved to {output_path}.pdf and {output_path}.png")
