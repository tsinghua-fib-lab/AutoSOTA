"""Typed containers for CPD Online.

This module contains the configuration and result dataclasses used by the
online CUSUM implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Sequence


@dataclass(frozen=True)
class OnlineCUSUMConfig:
    """Configuration parameters for the online CUSUM detector."""

    k: float = 0.5
    h: float = 10.0
    baseline_window: int = 64
    m0: Optional[float] = None
    s0: Optional[float] = None
    reset_after_alarm: bool = False
    adaptive_k: bool = False
    k_scale: float = 0.5
    k_min: float = -1.0


@dataclass
class OnlineCUSUMState:
    """Mutable state that is updated token-by-token."""

    k: float
    h: float
    baseline_window: int
    m0: Optional[float]
    s0: Optional[float]
    reset_after_alarm: bool
    W_plus: float = 0.0
    W_minus: float = 0.0
    t: int = 0
    alarm: bool = False
    T_alarm: Optional[int] = None
    baseline_ready: bool = False
    baseline_buffer: List[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        data = asdict(self)
        # baseline_buffer can be large; keep it for reproducibility but ensure
        # floats are serializable.
        data["baseline_buffer"] = list(self.baseline_buffer)
        return data


@dataclass(frozen=True)
class OnlineCUSUMEvent:
    """Per-token diagnostic emitted by the online detector."""

    t: int
    entropy: float
    baseline_mean: float
    baseline_variance: float
    baseline_median: float
    baseline_scale: float
    z_t: float
    W_plus: float
    W_minus: float
    alarm: bool


# Convenience type alias used by calibration helpers.
EntropyLike = Sequence[float]
