from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Abstract base
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class BaseAcquisitionOptimizer(ABC):
    """Base class for acquisition functions."""

    @abstractmethod
    def __call__(
        self
    ) -> torch.Tensor:
        pass

@dataclass(frozen=True)
class Exhaustive(BaseAcquisitionOptimizer):
    """Exhaustive AF optimization."""

    def __call__(
        self
    ) -> torch.Tensor:
        pass

@dataclass(frozen=True)
class Subset(BaseAcquisitionOptimizer):
    """Subset AF optimization."""
    subset_size: int = 10000

    def __call__(
        self
    ) -> torch.Tensor:
        pass