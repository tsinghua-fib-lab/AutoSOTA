"""Utility functions for the BO framework.

This package contains utility functions for:
- Plotting (1D objectives, posteriors, regret curves)
- Data processing
- Experiment analysis
- Results I/O (saving and loading)
"""

from .plotting import (
    plot_objective_1d,
    plot_posterior_1d,
    plot_regret,
    plot_bo_iteration_1d,
    plot_experiment_summary,
    create_figure_with_subplots,
    PlotConfig
)

from .io import (
    save_experiment_results,
    load_experiment_results,
    extract_result_summary,
    save_comparison_table
)

__all__ = [
    # Plotting
    'plot_objective_1d',
    'plot_posterior_1d', 
    'plot_regret',
    'plot_bo_iteration_1d',
    'plot_experiment_summary',
    'create_figure_with_subplots',
    'PlotConfig',
    # I/O
    'save_experiment_results',
    'load_experiment_results',
    'extract_result_summary',
    'save_comparison_table'
]