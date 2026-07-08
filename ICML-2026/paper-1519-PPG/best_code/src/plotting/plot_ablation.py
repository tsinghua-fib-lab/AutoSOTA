from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

from src.plotting.plot_utils import (
    get_ablation_plot_colors,
    get_ablation_plot_styles,
    plot_with_confidence_interval,
    save_figure,
    setup_plot_style,
    style_axes,
)


def plot_ablation_study(
    results: List[Dict[str, Any]],
    param_name: str,
    param_label: str,
    output_filename_prefix: str,
    output_dir: str = ".",
    title_suffix: str = "",
    ylabel: str = "Value Function $V^\\pi$",
):
    setup_plot_style()

    fig, ax = plt.subplots(figsize=(8, 6))

    colors = get_ablation_plot_colors()
    linestyles, markers = get_ablation_plot_styles()

    for i, result in enumerate(results):
        param_value = result.get(param_name)
        if param_value == 0:
            continue

        if "v_values_mean" in result and result["v_values_mean"]:
            v_mean = np.array(result["v_values_mean"])
            v_std = np.array(result["v_values_std"])
            iterations = range(len(v_mean))

            color = colors[i % len(colors)]
            linestyle = linestyles[i % len(linestyles)]
            marker = markers[i % len(markers)]

            plot_with_confidence_interval(
                ax,
                iterations,
                v_mean,
                v_std,
                color=color,
                label=f"{param_label} = {param_value}",
                linestyle=linestyle,
                marker=marker,
                markersize=4,
                markevery=10,
                linewidth=2.5,
                alpha=0.15,
            )

    ax.set_xlabel("Iteration", fontsize=16, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=16, fontweight="bold")
    title = (
        f"Effect of {title_suffix}" if title_suffix else f"{param_label} Ablation Study"
    )
    ax.set_title(title, fontsize=18, fontweight="bold", pad=20)

    legend = ax.legend(
        loc="lower right",
        frameon=True,
        fancybox=True,
        shadow=True,
        fontsize=12,
        title=title_suffix or param_label,
        title_fontsize=13,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(0.9)
    legend.get_frame().set_edgecolor("black")
    legend.get_frame().set_linewidth(1.0)

    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_dir) / f"{output_filename_prefix}_{timestamp}"

    save_figure(output_path, dpi=300)

    print(f"\nFinal Performance Summary:")
    print("-" * 50)
    for result in results:
        param_value = result.get(param_name)
        if "v_values_mean" in result and result["v_values_mean"]:
            final_value = result["v_values_mean"][-1]
            final_value_std = result["v_values_std"][-1]
            print(
                f"{param_label} = {param_value:4.2f}: Final Value = {final_value:7.4f} ± {final_value_std:.4f}"
            )

    return str(output_path) + ".png", str(output_path) + ".pdf"
