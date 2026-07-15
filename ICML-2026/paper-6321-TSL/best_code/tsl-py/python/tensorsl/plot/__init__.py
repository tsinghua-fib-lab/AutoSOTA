"""Plotting and interpretability utilities for TSL models.

All public functions are pure — they build a matplotlib figure and return
both the figure/axes and the underlying numerical arrays, so callers can
either save the figure as-is or rebuild a custom visualization (e.g. cartopy
overlays) from the returned data. Nothing is saved automatically.

Requires matplotlib. Install with `pip install tensorsl[plots]`.
"""

from .backbone import Backbone2DResult, plot_2d_backbone
from .components import (
    plot_combined_grid_tensors,
    plot_epoch_components,
    plot_grid_tensor_components,
)
from .importance import FeatureImportanceResult, plot_feature_importance
from .local import (
    LocalExplanation,
    compute_local_explanation,
    plot_local_interpretation,
)
from .pd import (
    ICEResult,
    NormalizedDiagnostics,
    PD2DLinesResult,
    PD2DResult,
    PDDifferenceResult,
    pd_difference_plot,
    plot_2d_pd,
    plot_first_order_pd,
    plot_ice,
)
from .tilt import (
    Tilt1DResult,
    Tilt2DResult,
    TiltDiagnosticsResult,
    plot_2d_tilt,
    plot_tilt_1d,
    plot_tilt_diagnostics,
)

__all__ = [
    # PD / ICE
    "plot_first_order_pd",
    "pd_difference_plot",
    "plot_2d_pd",
    "plot_ice",
    "PDDifferenceResult",
    "NormalizedDiagnostics",
    "PD2DResult",
    "PD2DLinesResult",
    "ICEResult",
    # 2D backbone
    "plot_2d_backbone",
    "Backbone2DResult",
    # Tilt (1D step curves + 2D outer-product mesh)
    "plot_tilt_1d",
    "plot_2d_tilt",
    "plot_tilt_diagnostics",
    "Tilt1DResult",
    "Tilt2DResult",
    "TiltDiagnosticsResult",
    # Local interpretation (intercept-absorbed backbone + tilt waterfall)
    "compute_local_explanation",
    "LocalExplanation",
    "plot_local_interpretation",
    # Feature importance
    "plot_feature_importance",
    "FeatureImportanceResult",
    # GridTensor component plots
    "plot_grid_tensor_components",
    "plot_combined_grid_tensors",
    "plot_epoch_components",
]
