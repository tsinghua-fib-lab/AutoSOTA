"""
Gaussian experiments, including 1) computing exact numerical solutions; and 2) training OTP-FM on Gaussian marginals.

Example usage:
    from experiments.gaussian import GaussianTrainer, GaussianMarginalSolver
    from experiments.gaussian.data import create_gaussian_dataloaders

Components:
- data: Data utilities for creating Gaussian marginal datasets
- trainer: GaussianTrainer class for training on Gaussian data
- solver: GaussianMarginalSolver for exact solutions
- plotting: Gaussian-specific visualization utilities
"""

from experiments.gaussian.solver import GaussianMarginalSolver
from experiments.gaussian.trainer import GaussianTrainer

__all__ = [
    "GaussianTrainer",
    "GaussianMarginalSolver",
]
