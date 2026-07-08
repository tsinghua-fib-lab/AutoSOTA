"""
Abstract base class for weighting functions in RCGP.
"""

from abc import ABC, abstractmethod
from typing import Callable, Optional
import torch


def _default_center_fn(x: torch.Tensor) -> torch.Tensor:
    """Default center function that returns zeros."""
    return torch.zeros_like(x[..., :1])


class WeightingFunction(ABC):
    """Abstract base class for weighting functions in RCGP."""

    def __init__(
        self, center_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    ):
        """
        Initialize weighting function.

        Args:
            center_fn: Function that takes input x and returns center value.
                      If None, centers at zero.
        """
        self.center_fn = center_fn or _default_center_fn

    @abstractmethod
    def weight(self, x: torch.Tensor, y: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        Compute weights for observations.

        Args:
            x: Input locations [n, d]
            y: Observations [n, 1] or [n]
            sigma: Noise standard deviation (as tensor for differentiability)

        Returns:
            Weights [n]
        """
        pass

    @abstractmethod
    def gradient_log_weight(self, x: torch.Tensor, y: torch.Tensor, sigma: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute gradient of log weight for RCGP update.

        Args:
            x: Input locations [n, d]
            y: Observations [n, 1] or [n]
            sigma: Noise standard deviation (optional, for interface consistency)

        Returns:
            Gradient correction [n]
        """
        pass

    def compute_J_matrix(self, weights: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        Compute J matrix where J = diag(σ²/(2w²)).

        Args:
            weights: Weight values [n]
            sigma: Noise standard deviation (as tensor)

        Returns:
            Diagonal matrix J [n, n] where J_ii = σ²/(2w_i²)
        """
        if weights.dim() == 0:
            weights = weights.unsqueeze(0)
        diagonal_values = (sigma**2) / (2 * weights**2)
        return torch.diag(diagonal_values)

    def set_center_function(self, center_fn: Callable[[torch.Tensor], torch.Tensor]):
        """Update the center function."""
        self.center_fn = center_fn
