"""TT-Sparse: Interpretable neural networks via sparse differentiable truth tables."""

__version__ = "0.1.0"

from tt_sparse.encoder import TabularEncoder
from tt_sparse.model import TTSparseModel, train, prune
from tt_sparse.rules import RuleSet, explain, predict_rules

__all__ = [
    "TabularEncoder",
    "TTSparseModel",
    "train",
    "prune",
    "RuleSet",
    "explain",
    "predict_rules",
]
