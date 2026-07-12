"""Progressive Cramming — reliable token compression into learnable memory embeddings.

Public API: the three cramming trainers plus the shared base class.
"""

from progressive_cramming.train import (
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
