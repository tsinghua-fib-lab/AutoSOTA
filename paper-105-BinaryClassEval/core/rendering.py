"""Rendering layer for net benefit plots."""
from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory

from proto.config import PlotConfig, SubgroupComputedData


def render_net_benefit_plot(
    computed_data_list: Sequence[SubgroupComputedData],
    plot_cfg: PlotConfig
) -> plt.Axes:
    """Render the net benefit plot based on precomputed data.

    This function handles all rendering logic including:
    - Drawing the main curves
    - Drawing confidence intervals
    - Drawing training and shifted points
    - Drawing average lines
    - Drawing AUC-ROC horizontal lines
    - Setting up axes, labels, and legends

    Parameters
    ----------
    computed_data_list : Sequence[SubgroupComputedData]
        List of precomputed data for each subgroup.
    plot_cfg : PlotConfig
        Configuration for plot appearance.

    Returns
    -------
    plt.Axes
        Matplotlib Axes with the rendered plot.
    """
    # Create axes if not provided
    ax = plot_cfg.ax
    if ax is None:
        fig, ax = plt.subplots(figsize=(3, 3))
    else:
        fig = ax.figure

    # Ensure there's enough space at the top for the title
    fig.subplots_adjust(top=0.88)

    # Default style cycle
    style_cycle = plot_cfg.style_cycle or plt.rcParams["axes.prop_cycle"].by_key()["color"]
    if plot_cfg.style_cycle_offset is not None:
        style_cycle = style_cycle[plot_cfg.style_cycle_offset:] + style_cycle[:plot_cfg.style_cycle_offset]
    style_cycle = list(style_cycle)  # Convert to list for indexing

    # Store subgroup prevalences for average line calculation
    subgroup_prevalences = []
    
    # Create a blended transform for the AUC labels: x in axes coordinates, y in data coordinates
    blended_transform = blended_transform_factory(ax.transAxes, ax.transData)

    # Draw main curves, CIs, and points for each subgroup
    for idx, data in enumerate(computed_data_list):
        color = style_cycle[idx % len(style_cycle)]

        # Store prevalence for later
        subgroup_prevalences.append(data.prevalence)

        # Draw main curve
        if not plot_cfg.hide_main:
            ax.plot(data.log_odds_grid, data.nb_curve, label=data.display_label, color=color)

        # Draw calibrated curve if available
        if data.calibrated_nb_curve is not None:
            ax.plot(
                data.log_odds_grid,
                data.calibrated_nb_curve,
                label=f"{data.display_label}\n(recalibrated)",
                color=color,
                linestyle=(0, (1, 0.6)),
                linewidth=4,
                alpha=0.5
            )

        # Draw confidence intervals if available
        if data.nb_ci_lower is not None and data.nb_ci_upper is not None:
            ax.fill_between(
                data.log_odds_grid,
                data.nb_ci_lower,
                data.nb_ci_upper,
                color=color,
                alpha=plot_cfg.ci_alpha,
                linewidth=0
            )

        # Draw training point
        if data.training_point and not plot_cfg.hide_main:
            print(data.name + " training point:", data.training_point[1])
            ax.scatter(
                data.training_point[0],
                data.training_point[1],
                color=color,
                marker="o",
                edgecolor="white",
                s=100,
                zorder=3
            )

        for prev in [0.01, 0.05, 0.25, 0.75, 0.95, 0.99]:
            idx = np.searchsorted(data.log_odds_grid, np.log(prev / (1-prev)))
            match = (1-1/(1+np.exp(data.log_odds_grid[idx])))
            print(f"{data.name} @{100*prev:.0f}%[found {100*match:.0f}%]: {data.nb_curve[idx]:.3f}")
        # Draw shifted point if available
        if data.shifted_point and plot_cfg.show_diamonds:
            print(data.name + " shifted point:", data.shifted_point[1])
            ax.scatter(
                data.shifted_point[0],
                data.shifted_point[1],
                color=color,
                marker="D",
                s=80,
                edgecolor="black",
                linewidth=1,
                alpha=0.8,
                zorder=3
            )
            
        # Draw horizontal line for AUC-ROC if available
        if hasattr(data, 'auc_roc') and data.auc_roc is not None:
            # Get x-axis limits for the horizontal line
            min_logodds_display = plot_cfg.min_logodds
            max_logodds_display = plot_cfg.max_logodds

            # Draw horizontal line for AUC-ROC
            ax.plot(
                [min_logodds_display, max_logodds_display],
                [data.auc_roc, data.auc_roc],
                color=color,
                linestyle='-',
                linewidth=2,
                alpha=0.2
            )
            
            # Add "AUC" text label at the left side of the y-axis
            # Use x=0 in axes coordinates (exactly on y-axis) and y in data coordinates
            ax.text(
                -0.01,  # Slightly to the left of the y-axis in axes coordinates
                data.auc_roc,  # At the height of the AUC line in data coordinates
                "AUC",
                color=color,
                fontsize=12,  # Increased from 8 to 12
                ha='right',  # Right-aligned text
                va='center',  # Vertically centered
                alpha=0.4,
                transform=blended_transform,  # Use the blended transform
                weight='bold'  # Added bold weight
            )

    # Draw average lines if requested
    if len(computed_data_list) >= 2 and plot_cfg.show_averages:
        # Sort prevalences
        min_prev = min(subgroup_prevalences)
        max_prev = max(subgroup_prevalences)

        # Calculate log-odds for the range we're averaging over
        min_logodds_avg = np.log(min_prev / (1 - min_prev))
        max_logodds_avg = np.log(max_prev / (1 - max_prev))

        if plot_cfg.full_width_average:
            min_logodds_display = plot_cfg.min_logodds
            max_logodds_display = plot_cfg.max_logodds
        else:
            min_logodds_display = min_logodds_avg
            max_logodds_display = max_logodds_avg

        # Draw average lines for each subgroup
        for idx, data in enumerate(computed_data_list):
            color = style_cycle[idx % len(style_cycle)]

            # Find indices where prevalence is between min_prev and max_prev
            mask = (data.log_odds_grid >= min_logodds_avg) & (data.log_odds_grid <= max_logodds_avg)

            if any(mask):
                # Calculate average net benefit in this range
                avg_nb = np.mean(data.nb_curve[mask])
                print(f"Average net benefit for {data.name} in range [{min_prev:.3f}, {max_prev:.3f}]: {avg_nb:.3f}")
                print(f"Average net benefit for {data.name} in unrestricted range: {np.mean(data.nb_curve):.3f}")
                p = 1 / (1 + np.exp(-data.log_odds_grid))
                var = p * (1 - p)
                print(f"Average (Brier) net benefit for {data.name} in unrestricted range: {np.average(data.nb_curve, weights=var):.3f}")

                if data.calibrated_nb_curve is not None:
                    avg_calibrated = np.mean(data.calibrated_nb_curve[mask])
                    print(f"Average calibrated value for {data.name}: {avg_calibrated:.3f}")

                # Draw horizontal line segment
                # Draw horizontal line segment for average
                ax.plot(
                    [min_logodds_display, max_logodds_display],
                    [avg_nb, avg_nb],
                    color=color,
                    linewidth=2,
                    alpha=0.2
                )
                
                # Add arrowhead-like confidence interval bars at the ends
                bar_height = 0.005  # Height of the confidence interval bars
                # Left arrowhead/bar
                ax.plot(
                    [min_logodds_display, min_logodds_display],
                    [avg_nb - bar_height, avg_nb + bar_height],
                    linewidth=2,
                    color=color,
                    alpha=0.2,
                )
                # Right arrowhead/bar
                ax.plot(
                    [max_logodds_display, max_logodds_display],
                    [avg_nb - bar_height, avg_nb + bar_height],
                    linewidth=2,
                    color=color,
                    alpha=0.2,
                )

    # Set x-axis limits (log-odds values corresponding to 0.001 and 0.999)
    ax.set_xlim(plot_cfg.min_logodds, plot_cfg.max_logodds)

    # Set up axes formatting
    ax.set_xlabel("Positive Class Fraction")
    ax.set_ylabel("Net Benefit")

    # Set up custom x-axis ticks showing probability values
    #prob_ticks = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
    prob_ticks = [0.01, 0.1]
    logodds_ticks = [np.log(p / (1 - p)) for p in prob_ticks]
    ax.set_xticks(logodds_ticks)
    ax.set_xticklabels([f"{p:.3f}" if p < 0.01 or p > 0.99 else f"{p:.2f}" for p in prob_ticks])

    # Set y-axis limits
    ax.set_ylim(plot_cfg.min_accuracy, 1.0)

    # Add zero line
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.7)

    # Handle custom legend mapping if provided
    if plot_cfg.subgroup_legend_mapping:
        # Create custom legend with explicit ordering
        legend_elements = []

        # Create legend elements for each mapping
        for key, label in plot_cfg.subgroup_legend_mapping.items():
            # Find the subgroup with this name
            for idx, data in enumerate(computed_data_list):
                if data.name == key:
                    color = style_cycle[idx % len(style_cycle)]
                    legend_elements.append(Line2D([0], [0], color=color, lw=2, label=label))
                    break

        # Create legend with custom elements
        if legend_elements:
            ax.legend(handles=legend_elements, ncol=min(3, len(legend_elements)),
                     loc='best', columnspacing=1.0, handletextpad=0.5, framealpha=0.7)
    else:
        # Use default legend
        ax.legend(loc='best', framealpha=0.7)

    # Use tight_layout with padding to ensure title doesn't get cut off
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    return ax
