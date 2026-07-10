"""
src package for Conformal Prediction (CP) analysis.
Provides reusable methods for:
- Data loading and preprocessing
- LAC/APS conformal prediction (weighted/unweighted)
- Metrics computation
- Visualization and plotting
"""

from .io_utils import (
    get_raw_data,
    get_logits_data,
    convert_id_to_ans,
    cal_coverage,
    cal_set_size,
)
from .cp_methods import (
    LAC_CP,
    LAC_CP_W,
    APS_CP,
    APS_CP_W,
)
from .plotting import (
    plot_cp_comparisons,
    rerun_plots_only,
)

__all__ = [
    "get_raw_data", "get_logits_data", "convert_id_to_ans", "cal_coverage", "cal_set_size",
    "LAC_CP", "LAC_CP_W", "APS_CP", "APS_CP_W",
    "plot_cp_comparisons", "rerun_plots_only",
]
