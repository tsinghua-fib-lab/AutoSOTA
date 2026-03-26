"""Configuration classes for visualization and computation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class ComputationConfig:
    """Configuration for net benefit computation.

    Attributes
    ----------
    prevalence_grid : np.ndarray
        Grid of prevalence values to compute net benefit over.
    cost_ratio : float
        Cost ratio parameter for net benefit calculation (FP cost / FN cost).
    n_bootstrap : int, default=100
        Number of bootstrap samples for confidence intervals.
    train_prevalence_override : bool, default=False
        Whether to use empirical prevalence from y_true instead of subgroup prevalence.
    normalize : bool, default=False
        Whether to normalize net benefit to [0, 1] range.
    compute_ci : bool, default=False
        Whether to compute bootstrap confidence intervals.
    compute_calibrated : bool, default=False
        Whether to compute calibrated net benefit curves.
    random_seed : int | None, default=None
        Random seed for reproducible bootstrapping.
    diamond_shift_amount : float | None, default=None
        Amount to shift diamond markers on log-odds scale for visualization.
    """

    prevalence_grid: np.ndarray  # Grid of prevalence values
    cost_ratio: float  # Cost ratio parameter
    n_bootstrap: int = 100  # Number of bootstrap samples
    train_prevalence_override: bool = False  # Whether to use subgroup prevalence or from data
    normalize: bool = False  # Whether to normalize net benefit
    compute_ci: bool = False  # Whether to compute confidence intervals
    compute_calibrated: bool = False  # Whether to compute calibrated curves
    random_seed: int | None = None  # Seed for reproducible bootstrapping
    diamond_shift_amount: float | None = None  # Amount to shift diamond markers


@dataclass
class PlotConfig:
    """Configuration for plot appearance.

    Attributes
    ----------
    ax : plt.Axes | None, default=None
        Matplotlib Axes to plot on. If None, creates new figure and axes.
    ci_alpha : float, default=0.2
        Transparency level for confidence interval bands.
    style_cycle : Sequence[str] | None, default=None
        Colors/styles for different subgroups. If None, uses matplotlib default cycle.
    show_diamonds : bool, default=True
        Whether to show diamond markers for shifted prevalence points.
    show_averages : bool, default=True
        Whether to show average net benefit lines across prevalence range.
    hide_main : bool, default=False
        Whether to hide main net benefit curves.
    subgroup_legend_mapping : dict[str, str] | None, default=None
        Custom mapping from subgroup names to legend labels.
    min_logodds : float, default=np.log(0.01 / 0.99)
        Minimum log-odds value for x-axis display.
    max_logodds : float, default=np.log(10. / 90.)
        Maximum log-odds value for x-axis display.
    full_width_average : bool, default=False
        Whether to extend average lines to full plot width.
    min_accuracy : float, default=0.85
        Minimum y-axis value (lower bound for plot).
    style_cycle_offset : int | None, default=None
        Offset to apply to style cycle for color selection.
    """

    ax: plt.Axes | None = None  # Axes to plot on
    ci_alpha: float = 0.2  # Transparency for confidence intervals
    style_cycle: Sequence[str] | None = None  # Colors/styles for groups
    show_diamonds: bool = True  # Show diamond markers
    show_averages: bool = True  # Show average lines
    hide_main: bool = False  # Hide main curves
    subgroup_legend_mapping: dict[str, str] | None = None  # Custom legend mapping
    min_logodds: float = np.log(0.01 / 0.99)
    max_logodds: float = np.log(10. / 90.)
    full_width_average: bool = False  # Show average lines over full width
    min_accuracy: float = 0.85
    style_cycle_offset: int | None = None


@dataclass(frozen=True)
class SubgroupComputedData:
    """Computed data for a subgroup.

    This dataclass holds all precomputed metrics and curves for a single subgroup,
    ready for visualization in the rendering layer.

    Attributes
    ----------
    name : str
        Original subgroup name/identifier.
    display_label : str
        Display label for plots and legends.
    prevalence : float
        Prevalence (proportion of positive class) in the subgroup.
    log_odds_grid : np.ndarray
        Precomputed log-odds grid corresponding to prevalence_grid.
    nb_curve : np.ndarray
        Net benefit curve values across prevalence_grid.
    calibrated_nb_curve : np.ndarray | None, default=None
        Calibrated net benefit curve if computed.
    nb_ci_lower : np.ndarray | None, default=None
        Lower confidence interval bound if computed.
    nb_ci_upper : np.ndarray | None, default=None
        Upper confidence interval bound if computed.
    training_point : tuple[float, float] | None, default=None
        (log-odds, net_benefit) at training prevalence.
    shifted_point : tuple[float, float] | None, default=None
        (log-odds, net_benefit) at shifted prevalence for diamond marker.
    auc_roc : float | None, default=None
        AUC-ROC value if computed.
    """

    name: str  # Original subgroup name
    display_label: str  # Display label for plots
    prevalence: float  # Subgroup prevalence
    log_odds_grid: np.ndarray  # Precomputed log-odds grid
    nb_curve: np.ndarray  # Net benefit curve values
    calibrated_nb_curve: np.ndarray | None = None  # Calibrated curve if requested
    nb_ci_lower: np.ndarray | None = None  # Lower CI bound if requested
    nb_ci_upper: np.ndarray | None = None  # Upper CI bound if requested
    training_point: tuple[float, float] | None = None  # (log-odds, NB) at training point
    shifted_point: tuple[float, float] | None = None  # (log-odds, NB) at shifted point
    auc_roc: float | None = None  # AUC-ROC value if computed