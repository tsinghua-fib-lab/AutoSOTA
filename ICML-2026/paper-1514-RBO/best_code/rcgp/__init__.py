"""
RCGP - Robust Conjugate Gaussian Process

This package implements Robust Conjugate Gaussian Process Upper Confidence Bound (RCGP-UCB)
algorithms for Bayesian optimization with adversarial corruptions.
"""

from .weighting import WeightingFunction, IMQ, PlateauIMQ, AdaptivePlateauIMQ
from .models import RobustConjugateGP

__version__ = "0.1.0"
__author__ = "RCGP Team"

__all__ = [
    "WeightingFunction",
    "IMQ",
    "PlateauIMQ",
    "AdaptivePlateauIMQ",
    "RobustConjugateGP",
]
