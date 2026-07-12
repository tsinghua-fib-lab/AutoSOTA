"""Concrete trainer implementations: one class per compression-training regime."""

from progressive_cramming.train.trainers.base import BaseTrainer
from progressive_cramming.train.trainers.full_cramming import FullCrammingTrainer
from progressive_cramming.train.trainers.low_dim import LowDimTrainer
from progressive_cramming.train.trainers.progressive_cramming import ProgressiveCrammingTrainer

__all__ = [
    "BaseTrainer",
    "FullCrammingTrainer",
    "LowDimTrainer",
    "ProgressiveCrammingTrainer",
]
