"""General evaluator for synthetic objective functions."""

import torch
from typing import Union, Dict, Any
from bo_framework.base.evaluator import BaseEvaluator
from bo_framework.base.evaluation_result import EvaluationResult
from .base import ObjectiveFunction


class SyntheticEvaluator(BaseEvaluator):
    """General evaluator for any synthetic objective function."""
    
    def __init__(self, objective_function: ObjectiveFunction):
        """
        Initialize synthetic evaluator.
        
        Args:
            objective_function: Any ObjectiveFunction instance
        """
        self.function = objective_function
    
    def evaluate(self, params: Union[Dict[str, Any], torch.Tensor]) -> EvaluationResult:
        """
        Evaluate synthetic function.
        
        Args:
            params: Either dict with parameter names or tensor
            
        Returns:
            EvaluationResult for deterministic function (no noise or corruption)
        """
        if isinstance(params, dict):
            # Convert dict to tensor based on function dimensionality
            if self.function.dim == 1:
                # For 1D functions, extract single parameter
                if 'x0' in params:
                    x = torch.tensor([params['x0']], dtype=torch.double)
                else:
                    # Take first value if parameter name is different
                    x = torch.tensor([list(params.values())[0]], dtype=torch.double)
            else:
                # For multi-D functions, extract ordered parameters
                param_values = []
                for i in range(self.function.dim):
                    key = f'x{i}'
                    if key in params:
                        param_values.append(params[key])
                    else:
                        # Fallback: use ordered values
                        param_values = list(params.values())[:self.function.dim]
                        break
                x = torch.tensor(param_values, dtype=torch.double)
        else:
            x = params.clone() if isinstance(params, torch.Tensor) else torch.tensor(params)
        
        y_true = float(self.function.evaluate(x))
        
        return EvaluationResult.from_true_value(params, y_true)
    
    @property
    def is_deterministic(self):
        return True