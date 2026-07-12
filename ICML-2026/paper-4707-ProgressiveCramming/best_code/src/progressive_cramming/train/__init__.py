"""Compression-training package: trainer implementations live in `trainers/`, building blocks at the top level."""

from progressive_cramming.train.trainers import (
    BaseTrainer,
    FullCrammingTrainer,
    LowDimTrainer,
    ProgressiveCrammingTrainer,
)

__all__ = [
    "BaseTrainer",
    "FullCrammingTrainer",
    "LowDimTrainer",
    "ProgressiveCrammingTrainer",
]
