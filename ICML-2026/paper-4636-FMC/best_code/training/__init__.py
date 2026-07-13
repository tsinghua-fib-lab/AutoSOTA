"""Training module for RoPE-FM.

This module provides a unified training interface following DeepMind research
code standards with:
- Registry pattern for discoverable trainers
- Configuration-driven training
- Consistent interfaces for all methods

Usage:
    from training import TrainerRegistry, get_trainer

    # Get a trainer by name
    trainer = get_trainer("fm_post_transform", config, model_path)
    posterior = trainer.train(data, device)

    # List available trainers
    print(TrainerRegistry.list_trainers())
"""

from training.base import BaseTrainer, TrainerConfig
from training.registry import TrainerRegistry, get_trainer, register_trainer

# Import trainers to trigger registration
from training.trainers import baselines, calibration, simulation

__all__ = [
    "BaseTrainer",
    "TrainerConfig",
    "TrainerRegistry",
    "get_trainer",
    "register_trainer",
]
