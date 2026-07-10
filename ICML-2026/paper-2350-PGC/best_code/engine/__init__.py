"""Training / evaluation engine package."""

from .evaluator import evaluate_model
from .trainer import PGCTrainer

__all__ = ["PGCTrainer", "evaluate_model"]
