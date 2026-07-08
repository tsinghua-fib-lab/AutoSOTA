"""Composable corruption framework with separated decision and strategy logic."""

import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, List, Optional, Union
from bo_framework.base.evaluation_result import EvaluationResult
from .base import BaseCorruptor


class CorruptionDecider(ABC):
    """Abstract base class for deciding when to corrupt observations.

    This class handles the logic of deciding WHETHER to corrupt,
    separate from HOW to corrupt.
    """

    @abstractmethod
    def should_corrupt(self,
                       iteration: int,
                       total_iterations: int,
                       is_initial: bool,
                       history: List[EvaluationResult]) -> bool:
        """Decide whether to corrupt the current observation.

        Args:
            iteration: Current iteration number (0-based)
            total_iterations: Total number of iterations expected
            is_initial: Whether this is an initial data point
            history: List of all previous evaluation results

        Returns:
            Whether to corrupt this observation
        """
        pass

    @abstractmethod
    def reset(self):
        """Reset the decider state."""
        pass

    @property
    @abstractmethod
    def info(self) -> dict:
        """Get information about the decider state."""
        pass


class CorruptionStrategy(ABC):
    """Abstract base class for corruption strategies.

    This class handles the logic of HOW to corrupt observations,
    separate from WHEN to corrupt.
    """

    @abstractmethod
    def compute_corruption(self,
                          current_result: EvaluationResult,
                          history: List[EvaluationResult]) -> float:
        """Compute the corruption value to apply.

        Args:
            current_result: Current evaluation result (before corruption)
            history: List of all previous evaluation results

        Returns:
            Corruption value to add to the observation
        """
        pass

    @abstractmethod
    def reset(self):
        """Reset the strategy state."""
        pass


class TimeBudgetDecider(CorruptionDecider):
    """Decider that corrupts up to T^alpha observations.

    The budget grows as a power of the current iteration number,
    allowing more corruptions as optimization progresses.
    """

    def __init__(self, alpha: float = 0.5, skip_initial: bool = True, n_initial: int = 0):
        """Initialize time-dependent budget decider.

        Args:
            alpha: Power exponent for budget growth (e.g., 0.5 for sqrt(T))
            skip_initial: Whether to skip initial data points
            n_initial: Number of initial points to exclude from iteration count
        """
        self.alpha = alpha
        self.skip_initial = skip_initial
        self.n_initial = n_initial
        self.corruptions_used = 0
        self.last_budget = 0
        self.non_initial_iterations = 0

    def should_corrupt(self,
                       iteration: int,
                       total_iterations: int,
                       is_initial: bool,
                       history: List[EvaluationResult]) -> bool:
        """Decide based on T^alpha budget constraint."""
        # Skip initial points if configured
        if is_initial and self.skip_initial:
            return False

        # Count non-initial iterations
        if not is_initial:
            self.non_initial_iterations += 1

        # Calculate current budget based on non-initial iterations
        # We use max(1, ...) to ensure at least some budget
        iterations_after_initial = max(1, self.non_initial_iterations)
        current_budget = int(np.floor(iterations_after_initial ** self.alpha))

        # Check if we have budget available
        if self.corruptions_used < current_budget:
            self.corruptions_used += 1
            self.last_budget = current_budget
            return True

        return False

    def reset(self):
        """Reset the decider state."""
        self.corruptions_used = 0
        self.last_budget = 0
        self.non_initial_iterations = 0

    @property
    def info(self) -> dict:
        """Get information about the decider state."""
        return {
            'type': 'TimeBudget',
            'alpha': self.alpha,
            'corruptions_used': self.corruptions_used,
            'current_budget': self.last_budget,
            'skip_initial': self.skip_initial
        }


class PeriodicDecider(CorruptionDecider):
    """Decider that corrupts every Nth observation after initial points.

    This provides a regular pattern of corruption throughout optimization.
    """

    def __init__(self, period: int = 5, offset: int = 0, skip_initial: bool = True, n_initial: int = 0):
        """Initialize periodic decider.

        Args:
            period: Corrupt every Nth observation (counting after initial points)
            offset: Start counting from this offset
            skip_initial: Whether to skip initial data points
            n_initial: Number of initial points to exclude from iteration count
        """
        self.period = period
        self.offset = offset
        self.skip_initial = skip_initial
        self.n_initial = n_initial
        self.non_initial_count = 0

    def should_corrupt(self,
                       iteration: int,
                       total_iterations: int,
                       is_initial: bool,
                       history: List[EvaluationResult]) -> bool:
        """Decide based on periodic pattern after initial points."""
        # Skip initial points if configured
        if is_initial and self.skip_initial:
            return False

        # Count non-initial observations
        if not is_initial:
            self.non_initial_count += 1

            # Check if this observation should be corrupted
            # Using non_initial_count ensures we count from 1 after initial points
            if (self.non_initial_count + self.offset) % self.period == 0:
                return True

        return False

    def reset(self):
        """Reset the decider state."""
        self.non_initial_count = 0

    @property
    def info(self) -> dict:
        """Get information about the decider state."""
        return {
            'type': 'Periodic',
            'period': self.period,
            'offset': self.offset,
            'non_initial_count': self.non_initial_count,
            'skip_initial': self.skip_initial
        }


class AdversarialStrategy(CorruptionStrategy):
    """Strategic corruption based on distance from optimal points."""

    def __init__(self,
                 optimal_points: Union[torch.Tensor, List[torch.Tensor]],
                 near_threshold: float = 0.2,
                 far_threshold: float = 0.5,
                 high_value: float = 10.0,
                 low_value: float = -10.0):
        """Initialize adversarial strategy.

        Args:
            optimal_points: Known optimal point(s)
            near_threshold: Distance threshold for "near" any optimal point
            far_threshold: Distance threshold for "far" from all optimal points
            high_value: Value to inject for fake optimum
            low_value: Value to inject near true optima
        """
        # Handle both single and multiple optimal points
        if isinstance(optimal_points, torch.Tensor):
            if optimal_points.dim() == 1:
                self.optimal_points = [optimal_points]
            else:
                self.optimal_points = [optimal_points[i] for i in range(optimal_points.shape[0])]
        else:
            self.optimal_points = optimal_points

        self.near_threshold = near_threshold
        self.far_threshold = far_threshold
        self.high_value = high_value
        self.low_value = low_value
        self.fake_optimal_point = None

    def _extract_x_tensor(self, result: EvaluationResult) -> torch.Tensor:
        """Extract x as tensor from EvaluationResult."""
        if isinstance(result.x, torch.Tensor):
            return result.x
        elif isinstance(result.x, dict):
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

            dist = torch.norm(x - opt).item() / np.sqrt(len(x))
            min_dist = min(min_dist, dist)

        return min_dist

    def _is_far_from_all(self, x: torch.Tensor) -> bool:
        """Check if point is far from ALL optimal points."""
        return self._compute_min_distance(x) > self.far_threshold

    def compute_corruption(self,
                          current_result: EvaluationResult,
                          history: List[EvaluationResult]) -> float:
        """Compute strategic corruption value."""
        x = self._extract_x_tensor(current_result)
        min_dist = self._compute_min_distance(x)

        # Near true optimum: inject low value
        if min_dist < self.near_threshold:
            return self.low_value - current_result.y_true

        # Select fake optimum if far from all
        if self.fake_optimal_point is None and self._is_far_from_all(x):
            self.fake_optimal_point = x.clone()
            return self.high_value - current_result.y_true

        # Near fake optimum: inject high value
        if self.fake_optimal_point is not None:
            dist_from_fake = torch.norm(x - self.fake_optimal_point).item() / np.sqrt(len(x))
            if dist_from_fake < self.near_threshold:
                return self.high_value - current_result.y_true

        return 0.0

    def reset(self):
        """Reset the strategy state."""
        self.fake_optimal_point = None


class RandomStrategy(CorruptionStrategy):
    """Random corruption strategy with configurable distribution."""

    def __init__(self,
                 corruption_range: Tuple[float, float] = (-10.0, 10.0),
                 distribution: str = 'uniform',
                 seed: Optional[int] = None):
        """Initialize random strategy.

        Args:
            corruption_range: (min, max) range for corruption values
            distribution: 'uniform' or 'normal' distribution
            seed: Random seed for reproducibility
        """
        self.corruption_range = corruption_range
        self.distribution = distribution
        self.rng = np.random.RandomState(seed)

    def compute_corruption(self,
                          current_result: EvaluationResult,
                          history: List[EvaluationResult]) -> float:
        """Generate random corruption value."""
        if self.distribution == 'uniform':
            return self.rng.uniform(self.corruption_range[0], self.corruption_range[1])
        elif self.distribution == 'normal':
            mean = (self.corruption_range[0] + self.corruption_range[1]) / 2
            std = (self.corruption_range[1] - self.corruption_range[0]) / 4  # ~95% within range
            return np.clip(self.rng.normal(mean, std),
                          self.corruption_range[0],
                          self.corruption_range[1])
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")

    def reset(self):
        """Reset the strategy state (keeps the same seed)."""
        pass


class ConstantStrategy(CorruptionStrategy):
    """Simple constant corruption strategy."""

    def __init__(self, corruption_value: float = 10.0):
        """Initialize constant strategy.

        Args:
            corruption_value: Fixed corruption value to apply
        """
        self.corruption_value = corruption_value

    def compute_corruption(self,
                          current_result: EvaluationResult,
                          history: List[EvaluationResult]) -> float:
        """Return constant corruption value."""
        return self.corruption_value

    def reset(self):
        """Reset the strategy state."""
        pass


class ComposableCorruptor(BaseCorruptor):
    """Corruptor that combines a decider and a strategy.

    This allows mix-and-match of different decision logic and corruption strategies.
    """

    def __init__(self,
                 decider: CorruptionDecider,
                 strategy: CorruptionStrategy,
                 skip_initial: bool = True):
        """Initialize composable corruptor.

        Args:
            decider: Decides when to corrupt
            strategy: Decides how to corrupt
            skip_initial: Whether to skip initial points
        """
        # Use infinite budget since decision is handled by decider
        super().__init__(budget=float('inf'), skip_initial=skip_initial)
        self.decider = decider
        self.strategy = strategy
        self.iteration = 0
        self.total_iterations = None  # Will be set when we know total iterations

    def corrupt(self,
                current_result: EvaluationResult,
                history: List[EvaluationResult],
                is_initial: bool = False) -> Tuple[float, float]:
        """Apply corruption based on decider and strategy."""
        # Track iterations
        if not is_initial:
            self.iteration += 1

        # Estimate total iterations if not set
        if self.total_iterations is None and len(history) > 0:
            # Rough estimate based on typical BO experiments
            self.total_iterations = len(history) * 2

        # Check with decider
        should_corrupt = self.decider.should_corrupt(
            iteration=self.iteration,
            total_iterations=self.total_iterations or 100,
            is_initial=is_initial,
            history=history
        )

        if not should_corrupt:
            return current_result.y_observed, 0.0

        # Apply strategy
        corruption = self.strategy.compute_corruption(current_result, history)

        if corruption != 0.0:
            y_corrupted = current_result.y_true + corruption
            self.corruption_history.append({
                'iteration': self.iteration,
                'y_true': current_result.y_true,
                'y_corrupted': y_corrupted,
                'corruption': corruption,
                'decider_info': self.decider.info
            })
            return y_corrupted, 1

        return current_result.y_observed, 0.0

    def update_budget(self, cost: Union[int, float]) -> None:
        """Update budget (handled by decider)."""
        pass

    def can_corrupt(self, cost: Union[int, float]) -> bool:
        """Check if corruption is allowed (always true, decision is in decider)."""
        return True

    def reset(self):
        """Reset both decider and strategy."""
        self.decider.reset()
        self.strategy.reset()
        self.corruption_history = []
        self.iteration = 0
        self.total_iterations = None

    @property
    def budget_remaining(self) -> Union[int, float]:
        """Get remaining budget (infinite for composable)."""
        return float('inf')