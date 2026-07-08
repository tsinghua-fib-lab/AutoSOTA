"""Adversarial corruption strategies."""

import torch
import numpy as np
from typing import Tuple, List, Union, Optional
from .base import BaseCorruptor
from bo_framework.base.evaluation_result import EvaluationResult


class AdversarialCorruptor(BaseCorruptor):
    """Adversarial corruptor that knows the optimal point(s) and misleads the optimizer.
    
    Strategy:
    - Replace observations near any true optimum with a low value (hide them)
    - Replace observations far from all optima with a high value (promote fake optima)
    - Uses count-based budget (number of corruptions allowed)
    - Supports both single and multiple optimal points
    """
    
    def __init__(self, 
                 optimal_point: Optional[torch.Tensor] = None,
                 optimal_points: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
                 budget: int = 10,
                 near_threshold: float = 0.2,
                 far_threshold: float = 0.5,
                 high_value: float = 10.0,
                 low_value: float = -10.0,
                 skip_initial: bool = True):
        """Initialize adversarial corruptor.
        
        Args:
            optimal_point: Single optimal point location (for backward compatibility)
            optimal_points: Multiple optimal points as tensor (n_optima, dim) or list of tensors
            budget: Total number of corruptions allowed
            near_threshold: Distance threshold for "near" any optimal point
            far_threshold: Distance threshold for "far" from all optimal points
            high_value: Value to inject for fake optimum
            low_value: Value to inject near true optima
            skip_initial: If True, don't corrupt initial data points
        """
        super().__init__(budget, skip_initial)
        
        # Handle both single and multiple optimal points
        if optimal_points is not None:
            # Multiple points provided
            if isinstance(optimal_points, torch.Tensor):
                if optimal_points.dim() == 1:
                    # Single point provided as 1D tensor via optimal_points
                    self.optimal_points = [optimal_points]
                else:
                    # Multiple points as 2D tensor
                    self.optimal_points = [optimal_points[i] for i in range(optimal_points.shape[0])]
            else:
                self.optimal_points = optimal_points
        elif optimal_point is not None:
            # Single point provided (backward compatibility)
            self.optimal_points = [optimal_point]
        else:
            raise ValueError("Either optimal_point or optimal_points must be provided")
            
        self.near_threshold = near_threshold
        self.far_threshold = far_threshold
        self.high_value = high_value
        self.low_value = low_value
        self.fake_optimal_point = None
        self.fake_optimal_points = []  # Store multiple fake optima for multi-optimum case
        self.corruptions_count = 0  # Track number of corruptions
    
    def _extract_x_tensor(self, result: EvaluationResult) -> torch.Tensor:
        """Extract x as tensor from EvaluationResult."""
        if isinstance(result.x, torch.Tensor):
            return result.x
        elif isinstance(result.x, dict):
            # For single-parameter case, extract the value
            if len(result.x) == 1:
                return torch.tensor([list(result.x.values())[0]], dtype=torch.double)
            else:
                return torch.tensor(list(result.x.values()), dtype=torch.double)
        else:
            raise ValueError(f"Unexpected type for x: {type(result.x)}")
    
    def _compute_min_distance(self, x: torch.Tensor) -> float:
        """Compute minimum normalized distance to any optimal point."""
        if x.dim() > 1:
            x = x.squeeze()
            
        min_dist = float('inf')
        for opt_point in self.optimal_points:
            if opt_point.dim() > 1:
                opt = opt_point.squeeze()
            else:
                opt = opt_point
            
            # Normalized L2 distance
            dist = torch.norm(x - opt).item() / np.sqrt(len(x))
            min_dist = min(min_dist, dist)
            
        return min_dist
    
    def _is_far_from_all(self, x: torch.Tensor) -> bool:
        """Check if point is far from ALL optimal points."""
        if x.dim() > 1:
            x = x.squeeze()
            
        for opt_point in self.optimal_points:
            if opt_point.dim() > 1:
                opt = opt_point.squeeze()
            else:
                opt = opt_point
            
            # Normalized L2 distance
            dist = torch.norm(x - opt).item() / np.sqrt(len(x))
            if dist <= self.far_threshold:
                return False
                
        return True
    
    def corrupt(self, 
                current_result: EvaluationResult,
                history: List[EvaluationResult],
                is_initial: bool = False) -> Tuple[float, float]:
        """Strategically corrupt observations based on distance from optimum/optima.
        
        Strategy:
        1. If point is near any true optimum: inject low value (hide it)
        2. If point is far from all optima and we haven't selected fake optimum: select it
        3. If point is near fake optimum: inject high value (promote it)
        """
        # Skip initial points if configured to do so
        if is_initial and self.skip_initial:
            return current_result.y_observed, 0.0
        
        # No corruption if no budget
        if self.budget_remaining <= 0:
            return current_result.y_observed, 0.0
        
        # Extract current point as tensor
        x = self._extract_x_tensor(current_result)
        
        # Compute minimum distance to any optimum
        min_dist_to_optima = self._compute_min_distance(x)
        
        # Case 1: Near ANY true optimum - inject low value
        if min_dist_to_optima < self.near_threshold:
            corruption = self.low_value - current_result.y_true
            
            if self.can_corrupt(1):  # Cost is 1 corruption
                self.update_budget(1)
                y_corrupted = current_result.y_true + corruption
                self.corruption_history.append({
                    'y_true': current_result.y_true,
                    'y_corrupted': y_corrupted,
                    'corruption': corruption,
                    'cost': 1,
                    'budget_remaining': self.budget_remaining
                })
                return y_corrupted, 1
        
        # Case 2: Select fake optimum if far from ALL optima
        if self.fake_optimal_point is None and self._is_far_from_all(x):
            self.fake_optimal_point = x.clone()
            # Inject high value at this point
            corruption = self.high_value - current_result.y_true
            
            if self.can_corrupt(1):  # Cost is 1 corruption
                self.update_budget(1)
                y_corrupted = current_result.y_true + corruption
                self.corruption_history.append({
                    'y_true': current_result.y_true,
                    'y_corrupted': y_corrupted,
                    'corruption': corruption,
                    'cost': 1,
                    'budget_remaining': self.budget_remaining
                })
                return y_corrupted, 1
        
        # Case 3: Near fake optimum - inject high value
        if self.fake_optimal_point is not None:
            dist_from_fake = torch.norm(x - self.fake_optimal_point).item() / np.sqrt(len(x))
            
            if dist_from_fake < self.near_threshold:
                corruption = self.high_value - current_result.y_true
                
                if self.can_corrupt(1):  # Cost is 1 corruption
                    self.update_budget(1)
                    y_corrupted = current_result.y_true + corruption
                    self.corruption_history.append({
                        'y_true': current_result.y_true,
                        'y_corrupted': y_corrupted,
                        'corruption': corruption,
                        'cost': 1,
                        'budget_remaining': self.budget_remaining
                    })
                    return y_corrupted, 1
        
        # No corruption for other points
        return current_result.y_observed, 0.0
    
    def update_budget(self, cost: int) -> None:
        """Update the budget after corruption.
        
        Args:
            cost: Number of corruptions (should be 1)
        """
        self.corruptions_count += cost
    
    def can_corrupt(self, cost: int) -> bool:
        """Check if corruption is within budget.
        
        Args:
            cost: Number of corruptions requested (should be 1)
            
        Returns:
            Whether corruption is allowed
        """
        return self.corruptions_count + cost <= self.budget
    
    def reset(self):
        """Reset the corruptor state."""
        self.corruptions_count = 0
        self.corruption_history = []
        self.fake_optimal_point = None
        self.fake_optimal_points = []
    
    @property
    def budget_remaining(self) -> int:
        """Get remaining corruption budget."""
        return self.budget - self.corruptions_count