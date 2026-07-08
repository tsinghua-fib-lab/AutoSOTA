"""
Inverse Multi-Quadric (IMQ) weighting function.
"""

from typing import Callable, Optional
import torch

from .base import WeightingFunction


class IMQ(WeightingFunction):
    """Inverse Multi-Quadric (IMQ) weighting function."""

    def __init__(
        self,
        c: float = 1.0,
        center_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        """
        Initialize IMQ weighting function.

        Args:
            beta: beta param (defaults to 1/√2)
            c: Scale parameter c
            center_fn: Center function g(x)
        """
        super().__init__(center_fn)
        self.c_tensor = torch.tensor(c, dtype=torch.float64)

    def weight(self, x: torch.Tensor, y: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        Compute IMQ weights: β(1 + (y - g(x))²/c²)^(-1/2)
        
        Args:
            x: Input locations [n, d]
            y: Observations [n, 1] or [n]
            sigma: Optional noise std (not used for IMQ, kept for interface compatibility)
        """
        beta = sigma / torch.sqrt(torch.tensor(2.0, dtype=sigma.dtype, device=sigma.device))
        
        # Ensure y is 1D
        if y.dim() > 1:
            y = y.squeeze(-1)

        # Get center values
        centers = self.center_fn(x)
        if centers.dim() > 1:
            centers = centers.squeeze(-1)

        # Compute deviation
        deviation = y - centers

        # Use cached tensor, move to correct device
        c_tensor = self.c_tensor.to(device=y.device)
        
        # Compute IMQ weights
        weights = beta * (1 + (deviation / c_tensor) ** 2) ** (-0.5)

        return weights

    def gradient_log_weight(self, x: torch.Tensor, y: torch.Tensor, sigma: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute gradient: ∇_y log(w²) = -2(y - g(x))/(c² + (y - g(x))²)
        
        Args:
            x: Input locations [n, d]
            y: Observations [n, 1] or [n]
            sigma: Optional noise std (not used for IMQ gradient, kept for interface compatibility)
        """
        # Ensure y is 1D
        if y.dim() > 1:
            y = y.squeeze(-1)

        # Get center values
        centers = self.center_fn(x)
        if centers.dim() > 1:
            centers = centers.squeeze(-1)

        # Compute deviation
        deviation = y - centers

        # Use cached tensor, move to correct device
        c_tensor = self.c_tensor.to(device=y.device)
        
        # Compute gradient
        denominator = c_tensor**2 + deviation**2
        gradient = -2 * deviation / denominator

        return gradient
