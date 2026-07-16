"""Hermite-NGP: gradient-augmented multi-resolution hash encoding for neural PDEs."""

__version__ = "1.0.0"

from hermite_ngp.encoding import HermiteHashEncoding2D, HermiteHashEncoding3D
from hermite_ngp.models import HermitePINN2D, HermitePINN3D

__all__ = [
    "HermiteHashEncoding2D",
    "HermiteHashEncoding3D",
    "HermitePINN2D",
    "HermitePINN3D",
]
