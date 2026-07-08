"""
Plateau-Matern 3/2 weighting function providing zero-cost robustness.

Tail decay: w ~ |y| * exp(-|y|) (exponential).
"""

from typing import Callable, Optional
import math
import torch

from .base import WeightingFunction

_SQRT3 = math.sqrt(3.0)


class PlateauMatern32(WeightingFunction):
    """
    Plateau-Matern 3/2 weighting function.

    Inside the plateau (|r| <= L): w = beta, grad = 0  (GP-equivalent).
    Outside the plateau: w = beta * (1 + sqrt(3)*z/c) * exp(-sqrt(3)*z/c),
    where z = |r| - L.
    """

    def __init__(
        self,
        plateau_width: float = 1.0,
        c: float = 1.0,
        center_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        super().__init__(center_fn)
        self.plateau_width = torch.tensor(plateau_width, dtype=torch.float64)
        self.c = torch.tensor(c, dtype=torch.float64)

    def weight(self, x: torch.Tensor, y: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        y = y.squeeze(-1) if y.dim() > 1 else y
        centers = self.center_fn(x)
        centers = centers.squeeze(-1) if centers.dim() > 1 else centers

        abs_deviation = torch.abs(y - centers)
        plateau_width_tensor = self.plateau_width.to(device=y.device)
        excess_deviation = torch.relu(abs_deviation - plateau_width_tensor)

        beta = sigma / torch.sqrt(torch.tensor(2.0, dtype=sigma.dtype, device=sigma.device))
        c_tensor = self.c.to(device=y.device)

        a = _SQRT3 * excess_deviation / c_tensor  # sqrt(3) * z / c
        matern_term = (1 + a) * torch.exp(-a)
        weights = beta * matern_term
        return weights.clamp(min=1e-30)

    def gradient_log_weight(self, x: torch.Tensor, y: torch.Tensor, sigma: Optional[torch.Tensor] = None) -> torch.Tensor:
        y = y.squeeze(-1) if y.dim() > 1 else y
        centers = self.center_fn(x)
        centers = centers.squeeze(-1) if centers.dim() > 1 else centers

        deviation = y - centers
        abs_deviation = torch.abs(deviation)

        plateau_width_tensor = self.plateau_width.to(device=y.device)
        c_tensor = self.c.to(device=y.device)

        excess_deviation = torch.relu(abs_deviation - plateau_width_tensor)

        # d/dz log(w^2) = -6z / (c^2 + sqrt(3)*c*z)
        numerator = torch.sign(deviation) * excess_deviation
        denominator = c_tensor**2 + _SQRT3 * c_tensor * excess_deviation

        gradient = -6 * numerator / (denominator + 1e-8)
        return gradient

    def set_plateau_width(self, width: float):
        self.plateau_width = torch.tensor(width, dtype=torch.float64)

    def update_heuristics(self, plateau_width: float, c: float, center_fn: Callable[[torch.Tensor], torch.Tensor]):
        self.plateau_width = torch.tensor(plateau_width, dtype=torch.float64)
        self.c = torch.tensor(c, dtype=torch.float64)
        self.center_fn = center_fn

    def is_in_plateau(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y = y.squeeze(-1) if y.dim() > 1 else y
        centers = self.center_fn(x)
        centers = centers.squeeze(-1) if centers.dim() > 1 else centers
        abs_deviation = torch.abs(y - centers)
        excess_deviation = torch.relu(abs_deviation - self.plateau_width.to(device=y.device))
        return excess_deviation == 0
