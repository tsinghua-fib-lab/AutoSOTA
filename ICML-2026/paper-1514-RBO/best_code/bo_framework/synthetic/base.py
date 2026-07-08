"""Base classes for synthetic objective functions."""

import torch
from typing import Optional
from abc import ABC, abstractmethod


class ObjectiveFunction(ABC):
    """Base class for objective functions with known properties."""
    
    def __init__(
        self,
        dim: int = 1,
        bounds: Optional[torch.Tensor] = None,
    ):
        """
        Initialize objective function wrapper.
        
        Args:
            dim: Input dimension
            bounds: Search space bounds [2, d]
        """
        self.dim = dim
        self.bounds = bounds if bounds is not None else self._default_bounds()
    
    def _default_bounds(self) -> torch.Tensor:
        """Default bounds [0, 1]^d."""
        return torch.tensor([[0.0] * self.dim, [1.0] * self.dim], dtype=torch.double)
    
    @abstractmethod
    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the true (noiseless) objective function.
        
        Args:
            x: Input point(s) [*, d]
            
        Returns:
            Function value(s) [*]
        """
        pass
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the true (noiseless) objective function.
        
        Args:
            x: Input point(s) [*, d]
            
        Returns:
            Function value(s) [*]
        """
        return self.evaluate(x)
    
    @property
    def optimal_value(self) -> Optional[float]:
        """Get the known optimal value (if available)."""
        return None
    
    @property
    def optimal_point(self) -> Optional[torch.Tensor]:
        """Get the known optimal point (if available)."""
        return None
    
    @property
    def has_known_optimum(self) -> bool:
        """Check if the function has a known optimum."""
        return self.optimal_value is not None and self.optimal_point is not None