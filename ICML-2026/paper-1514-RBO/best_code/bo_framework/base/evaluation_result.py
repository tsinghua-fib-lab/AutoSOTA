"""Dataclass for evaluation results."""

from dataclasses import dataclass
from typing import Optional, Union, Dict, Any
import torch


@dataclass
class EvaluationResult:
    """Structured result from evaluator.
    
    Attributes:
        x: Input parameters (either dict or tensor)
        y_true: True function value (no noise, no corruption)
        y_noisy: Value with observation noise (y_noisy = y_true + noise)
        y_observed: Value seen by BO (includes noise and corruption)
        noise: Observation noise added
        corruption: Corruption applied (y_observed = y_noisy + corruption)
    """
    x: Union[Dict[str, Any], torch.Tensor]
    y_true: float
    y_noisy: float
    y_observed: float
    noise: float = 0.0
    corruption: float = 0.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary for backward compatibility."""
        return {
            'x': self.x,
            'y_true': self.y_true,
            'y_noisy': self.y_noisy,
            'y_observed': self.y_observed,
            'noise': self.noise,
            'corruption': self.corruption
        }
    
    @classmethod
    def from_true_value(cls, x: Union[Dict[str, Any], torch.Tensor], y_true: float) -> 'EvaluationResult':
        """Create result for deterministic evaluator (no noise or corruption)."""
        return cls(x=x, y_true=y_true, y_noisy=y_true, y_observed=y_true, noise=0.0, corruption=0.0)
    
    def with_noise(self, noise: float) -> 'EvaluationResult':
        """Create new result with added noise."""
        new_noise = self.noise + noise  # Accumulate noise
        new_y_noisy = self.y_true + new_noise
        new_y_observed = new_y_noisy + self.corruption  # Apply existing corruption to new noisy value
        
        return EvaluationResult(
            x=self.x,
            y_true=self.y_true,
            y_noisy=new_y_noisy,
            y_observed=new_y_observed,
            noise=new_noise,
            corruption=self.corruption
        )
    
    def with_corruption(self, corruption: float) -> 'EvaluationResult':
        """Create new result with added corruption.
        
        Note: This assumes corruption is applied after noise.
        """
        new_corruption = self.corruption + corruption  # Accumulate corruption
        new_y_observed = self.y_noisy + new_corruption  # Apply total corruption to noisy value
        
        return EvaluationResult(
            x=self.x,
            y_true=self.y_true,
            y_noisy=self.y_noisy,  # Preserve noisy value
            y_observed=new_y_observed,
            noise=self.noise,
            corruption=new_corruption
        )