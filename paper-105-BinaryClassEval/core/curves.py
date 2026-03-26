"""High-level API for net benefit curves visualization."""
from __future__ import annotations
from typing import Sequence, Dict, Any, Optional

from matplotlib import pyplot as plt
import numpy as np

from proto.config import ComputationConfig, PlotConfig
from proto.subgroup import SubgroupResults
from stats.prevalence import default_prevalence_grid

# Import from computation module (no circular dependency)
from core.computation import prepare_data_for_visualization


def plot_net_benefit_curves(
    subgroups: Sequence[SubgroupResults],
    cost_ratio: float = 1.0,
    ax: plt.Axes = None,
    n_bootstrap: int = 0,
    compute_ci: bool = False,
    compute_calibrated: bool = False,
    train_prevalence_override: bool = False,
    normalize: bool = False,
    ci_alpha: float = 0.2,
    style_cycle: Sequence[str] = None,
    show_diamonds: bool = True,
    show_averages: bool = True,
    hide_main: bool = False,
    prevalence_grid: np.ndarray = None,
    diamond_shift_amount: float = None,
    random_seed: int = None,
    subgroup_legend_mapping: dict[str, str] = None,
    max_logodds: float = np.log(99. / 1.),
    full_width_average: bool = False,
    min_accuracy: float = 0.85,
    style_cycle_offset: int = None
) -> plt.Axes:
    """Plot net benefit curves for multiple subgroups.

    This function serves as the main entry point for the visualization API.
    It ties together the computation and rendering layers.

    Parameters
    ----------
    subgroups : Sequence[SubgroupResults]
        List of subgroup data to plot.
    cost_ratio : float, default=1.0
        Cost ratio parameter for net benefit calculation.
    ax : plt.Axes, default=None
        Axes to plot on. If None, a new figure is created.
    n_bootstrap : int, default=0
        Number of bootstrap samples for confidence intervals.
    compute_ci : bool, default=False
        Whether to compute confidence intervals.
    compute_calibrated : bool, default=False
        Whether to compute calibrated curves.
    train_prevalence_override : bool, default=False
        Whether to use empirical prevalence from y_true.
    normalize : bool, default=False
        Whether to normalize net benefit.
    ci_alpha : float, default=0.2
        Transparency for confidence intervals.
    style_cycle : Sequence[str], default=None
        Colors/styles for groups. If None, uses default style cycle.
    show_diamonds : bool, default=True
        Whether to show diamond markers.
    show_averages : bool, default=True
        Whether to show average lines.
    hide_main : bool, default=False
        Whether to hide main curves.
    prevalence_grid : np.ndarray, default=None
        Custom grid of prevalence values. If None, uses default grid.
    diamond_shift_amount : float, default=None
        Amount to shift diamond markers on log-odds scale.
    random_seed : int, default=None
        Random seed for reproducible bootstrapping.
    subgroup_legend_mapping : dict[str, str], default=None
        Mapping from subgroup names to custom legend labels.
    max_logodds : float, default=np.log(99/1)
        Maximum log-odds value to display on x-axis.
    full_width_average : bool, default=False
        Whether to extend average lines to full plot width.
    min_accuracy : float, default=0.85
        Minimum y-axis value (lower bound for plot).
    style_cycle_offset : int, default=None
        Offset to apply to style cycle for color selection.

    Returns
    -------
    plt.Axes
        Matplotlib Axes with the rendered plot.
    """
    # Use default prevalence grid if not provided
    if prevalence_grid is None:
        prevalence_grid = default_prevalence_grid()

    # Set up computation configuration
    comp_cfg = ComputationConfig(
        prevalence_grid=prevalence_grid,
        cost_ratio=cost_ratio,
        n_bootstrap=n_bootstrap,
        train_prevalence_override=train_prevalence_override,
        normalize=normalize,
        compute_ci=compute_ci,
        compute_calibrated=compute_calibrated,
        random_seed=random_seed,
        diamond_shift_amount=diamond_shift_amount
    )

    # Set up plot configuration
    plot_cfg = PlotConfig(
        ax=ax,
        ci_alpha=ci_alpha,
        style_cycle=style_cycle,
        show_diamonds=show_diamonds,
        show_averages=show_averages,
        subgroup_legend_mapping=subgroup_legend_mapping,
        hide_main=hide_main,
        max_logodds=max_logodds,
        full_width_average=full_width_average,
        min_accuracy=min_accuracy,
        style_cycle_offset=style_cycle_offset
    )

    # Compute metrics for all subgroups
    computed_data_list = prepare_data_for_visualization(subgroups, comp_cfg)
    
    # Import rendering function here to avoid circular imports
    from core.rendering import render_net_benefit_plot
    
    # Render the plot
    ax = render_net_benefit_plot(computed_data_list, plot_cfg)

    return ax