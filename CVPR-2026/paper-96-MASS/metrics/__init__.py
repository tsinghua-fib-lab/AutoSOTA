"""Metric and loss package initialization for MASS.

Importing this package registers loss functions and exposes Dice and surface
distance utilities used by training and evaluation.
"""

from . import dice
from . import losses
from . import surface_distance

# Export important functions
from .dice import calculate_dice, calculate_dice_split
from .losses import BinaryDiceLoss, BinaryCrossEntropyLoss
from .surface_distance import calculate_surface_distance, calculate_surface_dice_at_tolerance

__all__ = [
    'dice',
    'losses',
    'surface_distance',
    'calculate_dice',
    'calculate_dice_split',
    'BinaryDiceLoss',
    'BinaryCrossEntropyLoss',
    'calculate_surface_distance',
    'calculate_surface_dice_at_tolerance',
]
