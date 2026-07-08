"""
Plateau-IMQ weighting functions providing zero-cost robustness.
"""

from typing import Callable, Optional
import torch
import numpy as np

from .base import WeightingFunction


class PlateauIMQ(WeightingFunction):
    """
    Plateau-IMQ weighting function providing zero-cost robustness.

    This novel weighting function maintains constant weight within a plateau
    region, recovering exact GP updates for clean observations with high probability.
    """

    def __init__(
        self,
        plateau_width: float = 1.0,
        c: float = 1.0,
        center_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        """
        Initialize Plateau-IMQ weighting function.

        Args:
            plateau_width: Half-width of plateau region l (default: 1.0)
            c: Shape parameter c for IMQ tail (default: 1.0)
            center_fn: Center function g(x)
        """
        super().__init__(center_fn)
        self.plateau_width = torch.tensor(plateau_width, dtype=torch.float64)
        self.c = torch.tensor(c, dtype=torch.float64)

    def weight(self, x: torch.Tensor, y: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        Compute Plateau-IMQ weights:

        w(x,y) = {
            β                                           if |y - g(x)| ≤ l
            β * (1 + (|y - g(x)| - l)²/c²)^(-1/2)      if |y - g(x)| > l
        }
        
        where β = σ/√2 is computed dynamically from the provided sigma.
        
        Args:
            x: Input locations [n, d]
            y: Observations [n, 1] or [n]
            sigma: Noise standard deviation (as tensor for differentiability)
        """
        # Squeeze y and centers to be 1D
        y = y.squeeze(-1) if y.dim() > 1 else y
        centers = self.center_fn(x)
        centers = centers.squeeze(-1) if centers.dim() > 1 else centers

        # Compute absolute deviation
        abs_deviation = torch.abs(y - centers)

        # Use cached tensor, move to correct device
        plateau_width_tensor = self.plateau_width.to(device=y.device)
        
        # Compute excess deviation using ReLU. This is the key change.
        # This will be 0 for points inside the plateau and |y-g(x)|-l for points outside.
        excess_deviation = torch.relu(abs_deviation - plateau_width_tensor)

        # Compute beta dynamically from sigma
        beta = sigma / torch.sqrt(torch.tensor(2.0, dtype=sigma.dtype, device=sigma.device))
        
        # Use cached tensor, move to correct device
        c_tensor = self.c.to(device=y.device)
        
        # Calculate weights in a single, differentiable operation
        imq_term = (1 + (excess_deviation / c_tensor) ** 2) ** (-0.5)
        weights = beta * imq_term

        return weights

    def gradient_log_weight(self, x: torch.Tensor, y: torch.Tensor, sigma: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute gradient of log weight:

        ∇_y log(w²) = {
            0                                          if |y - g(x)| ≤ l
            -2(y - g(x) - l·sign(y - g(x))) / (c² + (|y - g(x)| - l)²)  if |y - g(x)| > l
        }
        
        Args:
            x: Input locations [n, d]
            y: Observations [n, 1] or [n]
            sigma: Not used in gradient computation but kept for interface consistency
        """
        # Squeeze y and centers to be 1D
        y = y.squeeze(-1) if y.dim() > 1 else y
        centers = self.center_fn(x)
        centers = centers.squeeze(-1) if centers.dim() > 1 else centers

        # Compute deviation
        deviation = y - centers
        abs_deviation = torch.abs(deviation)

        # Use cached tensors, move to correct device
        plateau_width_tensor = self.plateau_width.to(device=y.device)
        c_tensor = self.c.to(device=y.device)
        
        # Compute excess deviation using ReLU. This is the key change.
        excess_deviation = torch.relu(abs_deviation - plateau_width_tensor)

        # The numerator is 0 for points in the plateau
        numerator = torch.sign(deviation) * excess_deviation
        
        # The denominator is c^2 for points in the plateau
        denominator = c_tensor**2 + excess_deviation**2

        # Calculate gradient in a single, differentiable operation
        # Add a small epsilon to the denominator for numerical stability if needed
        gradient = -2 * numerator / (denominator + 1e-8)

        return gradient

    def set_plateau_width(self, width: float):
        """Update plateau width."""
        self.plateau_width = torch.tensor(width, dtype=torch.float64)

    def update_heuristics(self, plateau_width: float, c: float, center_fn: Callable[[torch.Tensor], torch.Tensor]):
        """
        Update robust heuristics based on standardized data analysis.
        
        Args:
            plateau_width: New plateau width (95th percentile of deviations)
            c: New tail shape parameter (MAD of standardized data)
            center_fn: New center function that maps x to centers
        """
        self.plateau_width = torch.tensor(plateau_width, dtype=torch.float64)
        self.c = torch.tensor(c, dtype=torch.float64)
        self.center_fn = center_fn
    
    def is_in_plateau(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Check which observations fall within the plateau region.
        
        Args:
            x: Input locations [n, d]
            y: Observations [n, 1] or [n]

        Returns:
            Boolean tensor indicating plateau membership [n]
        """
        # Squeeze y and centers to be 1D
        y = y.squeeze(-1) if y.dim() > 1 else y
        centers = self.center_fn(x)
        centers = centers.squeeze(-1) if centers.dim() > 1 else centers

        # Compute deviation
        deviation = y - centers
        abs_deviation = torch.abs(deviation)

        # Compute excess deviation using ReLU. This is the key change.
        excess_deviation = torch.relu(abs_deviation - self.plateau_width.to(device=y.device))

        # Check plateau membership
        return excess_deviation == 0


class AdaptivePlateauIMQ(PlateauIMQ):
    """
    Adaptive Plateau-IMQ that updates plateau width based on model confidence.
    """

    def __init__(
        self,
        initial_plateau_width: float = 1.0,
        c: float = 1.0,
        center_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        """
        Initialize adaptive plateau-IMQ.

        Args:
            initial_plateau_width: Initial plateau width
            c: Shape parameter c
            center_fn: Center function g(x)
        """
        super().__init__(initial_plateau_width, c, center_fn)
        self.initial_width = initial_plateau_width

    def update_plateau_width(
        self,
        model_confidence: torch.Tensor,
        beta_t: float,
        noise_std: float,
        iteration: int,
        delta: float = 0.1,
    ):
        """
        Update plateau width based on model confidence.

        Args:
            model_confidence: Model standard deviation at evaluation point
            beta_t: Current beta parameter
            noise_std: Observation noise standard deviation
            iteration: Current iteration number
            delta: Confidence parameter
        """
        # Confidence-based term
        confidence_term = np.sqrt(beta_t) * model_confidence

        # Noise term
        noise_term = noise_std * np.sqrt(2 * np.log(4 * (iteration + 1) / delta))

        # Update plateau width
        new_width = float(confidence_term + noise_term)
        self.plateau_width_tensor = torch.tensor(new_width, dtype=torch.float64)
