"""
Embryoid body single-cell RNA sequencing data.

Components:
- data: Data loading and preprocessing utilities
- trainer: EBTrainer class for training
- plotting: Single-cell specific visualizations
"""

from experiments.singlecell.trainer import EBTrainer

__all__ = ["EBTrainer"]
