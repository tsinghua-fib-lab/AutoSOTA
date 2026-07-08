"""Corrupted evaluator wrapper that applies adversarial corruption."""

import torch
import numpy as np
from typing import Union, Dict, Any, List, Optional
from bo_framework.base.evaluator import BaseEvaluator
from bo_framework.base.evaluation_result import EvaluationResult
from bo_framework.corruption.base import BaseCorruptor


class CorruptedEvaluator(BaseEvaluator):
    """Evaluator wrapper that applies corruption to observations.
    
    This wrapper takes the output from a base evaluator (which may include noise)
    and applies strategic corruption to mislead the optimizer. Can distinguish
    between initial data points and BO loop observations.
    """
    
    def __init__(self,
                 base_evaluator: BaseEvaluator,
                 corruptor: BaseCorruptor,
                 n_initial: Optional[int] = None):
        """Initialize corrupted evaluator wrapper.
        
        Args:
            base_evaluator: The underlying evaluator (can be noisy or clean)
            corruptor: Corruptor instance that decides how to corrupt observations
            n_initial: Number of initial points (if None, all points treated as BO loop)
        """
        self.base_evaluator = base_evaluator
        self.corruptor = corruptor
        self.n_initial = n_initial
        
        # Track evaluation history as list of EvaluationResults
        self.history: List[EvaluationResult] = []
    
    def evaluate(self, params: Union[Dict[str, Any], torch.Tensor]) -> EvaluationResult:
        """Evaluate with corruption applied.
        
        The corruption is applied to the y_observed value from the base evaluator,
        which may already include noise. Tracks whether this is an initial point.
        
        Args:
            params: Parameters to evaluate
            
        Returns:
            EvaluationResult with corruption applied to observation
        """
        # Get result from base evaluator (may include noise)
        base_result = self.base_evaluator.evaluate(params)
        
        # Determine if this is an initial point
        is_initial = (self.n_initial is not None and 
                     len(self.history) < self.n_initial)
        
        # Apply corruption with initial flag
        y_corrupted, _ = self.corruptor.corrupt(base_result, self.history, is_initial)
        
        # Calculate additional corruption amount (what we're adding on top of existing)
        additional_corruption = y_corrupted - base_result.y_observed
        
        # Create the corrupted result
        corrupted_result = base_result.with_corruption(additional_corruption)
        
        # Update history with the corrupted result
        self.history.append(corrupted_result)
        
        # Return result with additional corruption applied
        return corrupted_result
    
    def batch_evaluate(self, params_batch: List[Union[Dict[str, Any], torch.Tensor]]) -> List[EvaluationResult]:
        """Evaluate multiple parameter sets with corruption."""
        return [self.evaluate(params) for params in params_batch]
    
    @property
    def is_deterministic(self) -> bool:
        """Corrupted evaluator is never deterministic."""
        return False
    
    def reset(self):
        """Reset evaluation history and corruptor state."""
        self.history = []
        
        if self.corruptor is not None:
            self.corruptor.reset()