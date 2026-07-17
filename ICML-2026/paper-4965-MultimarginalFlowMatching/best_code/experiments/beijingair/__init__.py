"""
Beijing air quality experiments for OTP-FM.

This module provides tools for trajectory inference on Beijing PM2.5 air quality data.

Example usage:
    from experiments.beijingair import BeijingTrainer
    from experiments.beijingair.data import load_beijing_data, create_beijing_dataloaders

Components:
- data: Data loading and preprocessing utilities
- trainer: BeijingTrainer class for training
- plotting: Beijing-specific visualizations
"""

from experiments.beijingair.trainer import BeijingTrainer

__all__ = ["BeijingTrainer"]
