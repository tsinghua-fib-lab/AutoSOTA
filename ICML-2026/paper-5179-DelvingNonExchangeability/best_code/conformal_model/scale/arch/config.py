"""Configuration dataclasses for scale models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class QuantileModelConfig:
    """Minimal config required by quantile residual models."""

    input_length: int
    num_nodes: int
    quantiles: Sequence[float]
    horizon: int
    context_exog_dim: int = 0
    context_channels: int = 1
