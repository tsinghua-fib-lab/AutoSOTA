"""Base evaluator class for objective function evaluation."""

import torch
from abc import ABC, abstractmethod
from typing import Union, Dict, Any, List
from .evaluation_result import EvaluationResult


class BaseEvaluator(ABC):
    """Abstract base class for objective function evaluation.
    
    This class defines the interface for evaluating objective functions
    in the BO framework. Subclasses should implement the evaluate method.
    """
    
    @abstractmethod
    def evaluate(self, params: Union[Dict[str, Any], torch.Tensor]) -> EvaluationResult:
        """Evaluate objective function at given parameters.
        
        Args:
            params: Either a dictionary of parameter names to values,
                   or a tensor of parameter values
                   
        Returns:
            EvaluationResult with:
                - y_true: True function value
                - y_observed: Value seen by BO (includes noise and corruption)
                - noise: Observation noise added
                - corruption: Corruption amount applied
        """
        pass
    
    def batch_evaluate(self, params_batch: List[Union[Dict[str, Any], torch.Tensor]]) -> List[EvaluationResult]:
        """Evaluate multiple parameter sets.
        
        Default implementation evaluates sequentially.
        Subclasses can override for parallel evaluation.
        
        Args:
            params_batch: List of parameter sets
            
        Returns:
            List of EvaluationResult objects
        """
        return [self.evaluate(params) for params in params_batch]
    
    @property
    def is_deterministic(self) -> bool:
        """Whether the objective is deterministic.
        
        Default is False (assumes stochastic).
        Subclasses can override if deterministic.
        """
        return False