"""General synthetic test function support for benchmarking optimization algorithms."""

from .base import ObjectiveFunction
from .evaluators import SyntheticEvaluator

__all__ = ['ObjectiveFunction', 'SyntheticEvaluator']