# -*- coding: utf-8 -*-
"""
Training utilities.

Functions:
    train_model:           Full training loop with checkpointing.
    count_spikes_epoch:    Average spike rate over a dataset epoch.
    compute_energy_proxy:  Normalized energy estimate from spike rate.

Classes:
    TeeLogger: Dual terminal + file logger.
"""

from .trainer import train_model
from .metrics import count_spikes_epoch, compute_energy_proxy
from .logging import TeeLogger

__all__ = [
    "train_model",
    "count_spikes_epoch",
    "compute_energy_proxy",
    "TeeLogger",
]
