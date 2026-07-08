"""
Weighting functions for Robust Conjugate Gaussian Processes.
"""

from .base import WeightingFunction
from .imq import IMQ
from .plateau_imq import PlateauIMQ, AdaptivePlateauIMQ
from .plateau_cauchy import PlateauCauchy
from .plateau_matern32 import PlateauMatern32
from .plateau_rbf import PlateauRBF

__all__ = [
    "WeightingFunction",
    "IMQ",
    "PlateauIMQ",
    "AdaptivePlateauIMQ",
    "PlateauCauchy",
    "PlateauMatern32",
    "PlateauRBF",
]
