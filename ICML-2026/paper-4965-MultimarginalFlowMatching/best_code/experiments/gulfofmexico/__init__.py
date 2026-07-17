"""
Gulf of Mexico experiments for OTP-FM.

This module provides tools for modeling ocean currents in the Gulf of Mexico
using optimal transport-based trajectory inference.

Example usage:
    from experiments.gulfofmexico import GoMTrainer
    from experiments.gulfofmexico.data import load_gom_data, create_gom_dataloaders

Components:
- data: Data loading and preprocessing utilities
- trainer: GoMTrainer class for training
- plotting: GoM-specific visualizations
"""

from experiments.gulfofmexico.trainer import GoMTrainer

__all__ = ["GoMTrainer"]
