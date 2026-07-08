"""Noisy evaluator wrapper for adding observation noise."""

import torch
from typing import Union, Dict, Any, List, Optional
from bo_framework.base.evaluator import BaseEvaluator
from bo_framework.base.evaluation_result import EvaluationResult


class NoisyEvaluator(BaseEvaluator):
    """Wrapper that adds Gaussian noise to evaluator observations.
    
    This represents inherent measurement noise or stochasticity in the
    objective function evaluation process.
    """
    
    def __init__(self, 
                 base_evaluator: BaseEvaluator,
                 noise_std: float = 0.1,
                 seed: Optional[int] = None):
        """Initialize noisy evaluator wrapper.
        
        Args:
            base_evaluator: The underlying evaluator to wrap
            noise_std: Standard deviation of Gaussian noise to add
            seed: Random seed for reproducibility
        """
        self.base_evaluator = base_evaluator
        self.noise_std = noise_std
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)
    
    def evaluate(self, params: Union[Dict[str, Any], torch.Tensor]) -> EvaluationResult:
        """Evaluate with added Gaussian noise.
        
        Args:
            params: Parameters to evaluate
            
        Returns:
            EvaluationResult with noise added to observation
        """
        # Get result from base evaluator
        base_result = self.base_evaluator.evaluate(params)
        
        # Add Gaussian noise
        noise = torch.randn(1, generator=self.generator).item() * self.noise_std
        
        # Return new result with noise added
        return base_result.with_noise(noise)
    
    def batch_evaluate(self, params_batch: List[Union[Dict[str, Any], torch.Tensor]]) -> List[EvaluationResult]:
        """Evaluate multiple parameters with independent noise."""
        base_results = self.base_evaluator.batch_evaluate(params_batch)
        noise_values = torch.randn(len(base_results), generator=self.generator) * self.noise_std
        
        return [
            result.with_noise(noise.item())
            for result, noise in zip(base_results, noise_values)
        ]
    
    @property
    def is_deterministic(self) -> bool:
        """Noisy evaluator is always stochastic."""
        return False
    
    @property
    def true_evaluator(self) -> BaseEvaluator:
        """Access to the underlying true evaluator (for analysis)."""
        return self.base_evaluator