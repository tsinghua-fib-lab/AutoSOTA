"""Synthetic benchmark functions for Bayesian Optimization."""

import torch
import numpy as np
from bo_framework.synthetic.base import ObjectiveFunction


class ForresterFunction(ObjectiveFunction):
    """Forrester function - a common 1D benchmark for Bayesian optimization."""
    
    def __init__(self):
        """
        Initialize Forrester function.
        f(x) = (6x - 2)^2 * sin(12x - 4)
        """
        super().__init__(dim=1)
        # Precompute optimal values
        self._optimal_point = torch.tensor([[1.0]], dtype=torch.double)
        self._optimal_value = 15.829732  # Global maximum
    
    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate Forrester function."""
        if x.dim() == 0:
            x = x.unsqueeze(0)
        elif x.dim() > 1:
            x = x.squeeze()
        
        return ((6 * x - 2) ** 2) * torch.sin(12 * x - 4)
    
    @property
    def optimal_value(self) -> float:
        """Global maximum value."""
        return self._optimal_value
    
    @property
    def optimal_point(self) -> torch.Tensor:
        """Global maximum location."""
        return self._optimal_point


class BraninFunction(ObjectiveFunction):
    """Branin function - a common 2D benchmark for Bayesian optimization.
    
    The Branin function (also known as Branin-Hoo function) is a popular 
    test function for optimization algorithms. It has three global minima.
    
    Standard form (minimization):
    f(x1, x2) = a(x2 - b*x1^2 + c*x1 - r)^2 + s(1 - t)cos(x1) + s
    
    where typically:
    a = 1, b = 5.1/(4π^2), c = 5/π, r = 6, s = 10, t = 1/(8π)
    
    Domain: x1 ∈ [-5, 10], x2 ∈ [0, 15]
    
    We negate it for maximization.
    """
    
    def __init__(self):
        """Initialize Branin function with standard parameters."""
        super().__init__(dim=2)
        
        # Standard Branin parameters
        self.a = 1.0
        self.b = 5.1 / (4 * np.pi**2)
        self.c = 5 / np.pi
        self.r = 6.0
        self.s = 10.0
        self.t = 1 / (8 * np.pi)
        
        # Three global minima locations (in original space)
        self._minima = torch.tensor([
            [-np.pi, 12.275],
            [np.pi, 2.275],
            [9.42478, 2.475]
        ], dtype=torch.double)
        
        # Global minimum value (negated for maximization)
        self._optimal_value = -0.397887  # Negative of minimum
        
        # We'll use the first minimum as the reference optimal point
        self._optimal_point = self._minima[0].unsqueeze(0)
        
        # Standard bounds for Branin
        self.bounds = torch.tensor([[-5.0, 0.0], [10.0, 15.0]], dtype=torch.double)
    
    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate Branin function (negated for maximization).
        
        Args:
            x: Input tensor of shape (n, 2) or (2,)
            
        Returns:
            Function values (negated for maximization)
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        x1 = x[:, 0] if x.dim() > 1 else x[0]
        x2 = x[:, 1] if x.dim() > 1 else x[1]
        
        # Branin function computation
        term1 = self.a * (x2 - self.b * x1**2 + self.c * x1 - self.r)**2
        term2 = self.s * (1 - self.t) * torch.cos(x1)
        term3 = self.s
        
        # Return negative for maximization
        result = -(term1 + term2 + term3)
        
        return result.squeeze() if x.shape[0] == 1 else result
    
    @property
    def optimal_value(self) -> float:
        """Global maximum value (negative of minimum)."""
        return self._optimal_value
    
    @property
    def optimal_point(self) -> torch.Tensor:
        """One of the global maximum locations."""
        return self._optimal_point
    
    @property
    def all_optima(self) -> torch.Tensor:
        """All three global optima locations."""
        return self._minima