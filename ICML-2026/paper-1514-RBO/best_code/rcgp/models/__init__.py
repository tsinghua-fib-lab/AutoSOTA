"""
Robust Gaussian Process models.
"""

from .robust_gp import RobustConjugateGP
from .mixed_robust_gp import MixedRobustConjugateGP
from .a2rcgp import A2RCGP

__all__ = ["RobustConjugateGP", "MixedRobustConjugateGP", "A2RCGP"]
