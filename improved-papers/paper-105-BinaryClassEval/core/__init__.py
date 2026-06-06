"""Core functionality for net benefit analysis."""

from core.net_benefit import net_benefit_for_prevalences, otimes
from core.computation import compute_subgroup_metrics, prepare_data_for_visualization
from core.curves import plot_net_benefit_curves
from core.rendering import render_net_benefit_plot

__all__ = [
    "net_benefit_for_prevalences",
    "otimes",
    "compute_subgroup_metrics",
    "prepare_data_for_visualization", 
    "render_net_benefit_plot",
    "plot_net_benefit_curves",
]