"""General evaluator wrappers for adding noise, corruption, etc."""

from .noisy import NoisyEvaluator
from .corrupted import CorruptedEvaluator

__all__ = ['NoisyEvaluator', 'CorruptedEvaluator']