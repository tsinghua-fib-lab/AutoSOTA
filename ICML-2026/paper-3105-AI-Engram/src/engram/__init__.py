"""engram — minimal, efficient covariance-based engram extraction & editing.

Collect input covariances and extract per-layer *engram projections*
``P = W . C_target . pinv(C_total)`` for PyTorch / HuggingFace models — forward-only,
closed-form, with automatic bias absorption and answer-token masking — then edit with
``W <- W - alpha * f_l * P_l`` under a pluggable per-layer scaling ``f_l`` (default
:func:`count_ratio`, the paper's ``n / N``). Statistics are stored as mean covariances +
counts (:class:`Statistics`). Optional fused-MoE support lives in ``engram.moe``.
See https://jeakwon.github.io/ai-engram/.
"""

from __future__ import annotations

from .collectors import CovarianceCollector
from .config import EditorConfig
from .editor import EngramEditor
from .handlers import (
    Conv1DHandler,
    LayerHandler,
    LinearHandler,
    handler_for,
)
from .llm import apply_engram, edit_llm, get_engram
from .scaling import (
    EngramResult,
    LayerScaleInfo,
    ScaleFn,
    compose,
    count_ratio,
    effective_rank,
    uniform,
    weight_norm,
)
from .stats import Statistics

__all__ = [
    "EditorConfig",
    "EngramEditor",
    "edit_llm",
    "get_engram",
    "apply_engram",
    "CovarianceCollector",
    "Statistics",
    "EngramResult",
    "LayerScaleInfo",
    "ScaleFn",
    "count_ratio",
    "weight_norm",
    "effective_rank",
    "uniform",
    "compose",
    "LayerHandler",
    "LinearHandler",
    "Conv1DHandler",
    "handler_for",
]

__version__ = "0.8.0"
