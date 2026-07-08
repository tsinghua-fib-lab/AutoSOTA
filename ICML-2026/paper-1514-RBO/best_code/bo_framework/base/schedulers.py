"""Beta scheduling strategies for acquisition functions."""

from abc import ABC, abstractmethod
from typing import Optional, Any
import math
import numpy as np

from rcgp.models.robust_gp import RobustConjugateGP


class BetaScheduler(ABC):
    """Abstract base class for beta scheduling strategies.
    
    Beta schedulers control the exploration-exploitation trade-off in acquisition
    functions like UCB over the course of optimization. They can optionally use
    the current model state to make adaptive decisions.
    """
    
    @abstractmethod
    def get_beta(self, iteration: int, n_total: int, model: Optional[Any] = None) -> float:
        """Get beta value for the current iteration.
        
        Args:
            iteration: Current iteration number (0-indexed)
            n_total: Total number of iterations planned
            model: Optional GP model for state-dependent scheduling
            
        Returns:
            Beta value for the current iteration
        """
        pass


class ConstantBetaScheduler(BetaScheduler):
    """Keep beta constant throughout optimization (default behavior)."""
    
    def __init__(self, beta: float = 2.0):
        """Initialize with constant beta value.
        
        Args:
            beta: Fixed beta value to use
        """
        self.beta = beta
    
    def get_beta(self, iteration: int, n_total: int, model: Optional[Any] = None) -> float:
        """Return constant beta value.
        
        Args:
            iteration: Current iteration (ignored)
            n_total: Total iterations (ignored)
            model: Optional model (ignored)
            
        Returns:
            Constant beta value
        """
        return self.beta


class TheoryGuidedScheduler(BetaScheduler):
    """Theory-guided scheduling for UCB with sublinear regret guarantees.
    
    Implements beta scheduling based on theoretical UCB analysis where beta
    grows as O(sqrt(log(t))) to maintain sublinear regret bounds. This ensures
    sufficient exploration throughout optimization while maintaining theoretical
    convergence guarantees.
    
    The schedule is: beta_t = scale * sqrt(2 * log(t + offset))
    
    References:
        Srinivas et al. "Gaussian Process Optimization in the Bandit Setting:
        No Regret and Experimental Design" (2010)
    """
    
    def __init__(self, scale: float = 1.7, offset: float = 2, min_beta: float = 1.0):
        """Initialize theory-guided scheduler.
        
        Args:
            scale: Multiplicative scaling factor for the schedule (default 1.7 to match constant beta=2.0 initially)
            offset: Offset added to iteration to avoid log(0) and control initial beta
            min_beta: Minimum beta value to ensure some exploration
        """
        self.scale = scale
        self.offset = offset
        self.min_beta = min_beta
    
    def get_beta(self, iteration: int, n_total: int, model: Optional[Any] = None) -> float:
        """Compute theory-guided beta value.
        
        Args:
            iteration: Current iteration (0-indexed)
            n_total: Total iterations (ignored in theory-guided schedule)
            model: Optional model (ignored)
            
        Returns:
            Theory-guided beta value: scale * sqrt(2 * log(t + offset))
        """
        t = iteration + self.offset
        beta = self.scale * math.sqrt(2 * math.log(t))
        return max(beta, self.min_beta)


class RCGPScheduler(BetaScheduler):
    """RCGP scheduler for adaptive beta based on detected corruptions.
    
    Implements beta scheduling as: beta = base_beta + scale * sqrt(T_c)
    where T_c is the number of detected corruptions/outliers.
    
    This scheduler is specifically designed for RCGP models and uses the
    number of points outside the plateau to adaptively increase exploration
    when more corruptions are detected.
    """
    
    def __init__(self, scale: float = 1.0, base_scheduler: Optional[BetaScheduler] = None):
        """Initialize RCGP scheduler.
        
        Args:
            scale: Multiplicative factor for the sqrt(T_c) term
            base_scheduler: Base scheduler to adapt (defaults to ConstantBetaScheduler)
        """
        self.base_scheduler = base_scheduler or ConstantBetaScheduler(beta=2.0)
        self.scale = scale
    
    def get_beta(self, iteration: int, n_total: int, model: Optional[Any] = None) -> float:
        """Compute corruption-adaptive beta value.
        
        Args:
            iteration: Current iteration (0-indexed)
            n_total: Total number of iterations
            model: RobustConjugateGP model (must have get_n_outside_plateau method)
            
        Returns:
            Adapted beta value based on detected corruptions
        """
        # Get base beta from underlying scheduler
        base_beta = self.base_scheduler.get_beta(iteration, n_total, model)
        
        # Add corruption-dependent term if model is available
        if model is not None and hasattr(model, 'get_n_outside_plateau'):
            n_corruptions = model.get_n_outside_plateau()
            corruption_beta = self.scale * math.sqrt(n_corruptions) if n_corruptions > 0 else 0
            return base_beta + corruption_beta
        else:
            # Fallback to base beta if model not available or not RCGP
            return base_beta