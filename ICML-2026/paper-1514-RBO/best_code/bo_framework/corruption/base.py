"""Base classes for observation corruption."""

import torch
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Tuple, Union
from bo_framework.base.evaluation_result import EvaluationResult


class BaseCorruptor(ABC):
    """Abstract base class for observation corruptors.
    
    Corruptors can intercept and modify observations to mislead
    the optimization process. Supports both count-based (int) and 
    magnitude-based (float) budgets.
    """
    
    def __init__(self, budget: Union[int, float] = float('inf'), 
                 skip_initial: bool = True):
        """Initialize corruptor.
        
        Args:
            budget: Total corruption budget (can be count of corruptions or sum of magnitudes)
            skip_initial: If True, don't corrupt initial data points
        """
        self.budget = budget
        self.budget_used = 0.0 if isinstance(budget, float) else 0
        self.skip_initial = skip_initial
        self.corruption_history = []
    
    @abstractmethod
    def corrupt(self, 
                current_result: EvaluationResult,
                history: List[EvaluationResult],
                is_initial: bool = False) -> Tuple[float, float]:
        """Decide whether and how to corrupt an observation.
        
        Args:
            current_result: Current evaluation result (before corruption)
            history: List of all previous evaluation results
            is_initial: Whether this is an initial data point
            
        Returns:
            Tuple of (corrupted_value, corruption_cost)
            If no corruption, return (current_result.y_observed, 0.0)
        """
        pass
    
    @abstractmethod
    def update_budget(self, cost: Union[int, float]) -> None:
        """Update the budget after corruption.
        
        Args:
            cost: Cost of the corruption (count or magnitude)
        """
        pass
    
    @abstractmethod
    def can_corrupt(self, cost: Union[int, float]) -> bool:
        """Check if corruption is within budget.
        
        Args:
            cost: Proposed corruption cost
            
        Returns:
            Whether corruption is allowed
        """
        pass
    
    @abstractmethod
    def reset(self):
        """Reset the corruptor state."""
        pass
    
    @property
    @abstractmethod
    def budget_remaining(self) -> Union[int, float]:
        """Get remaining corruption budget."""
        pass