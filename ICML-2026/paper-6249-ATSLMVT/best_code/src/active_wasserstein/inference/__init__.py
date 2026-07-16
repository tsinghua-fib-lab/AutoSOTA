"""Posterior inference for Hilbert-valued Gaussian processes (GPyTorch-backed)."""

from .observation import (
    NoiseInitializer,
    TangentObservation,
    TangentObservationModel,
    variance_scaled_noise_initializer,
)
from .predictive import PredictiveProcess
from .gpytorch_regression import (
    GPyTorchHilbertPredictive,
    GPyTorchHilbertRegressor,
)
from .kernels import (
    KernelSpec,
    MaternKernelSpec,
    RBFKernelSpec,
)
from .reconstruction import reconstruct_distribution_at_time, reconstruct_distributions
from active_wasserstein.utils.scaling import InputScaler

__all__ = [
    "GPyTorchHilbertPredictive",
    "GPyTorchHilbertRegressor",
    "InputScaler",
    "KernelSpec",
    "MaternKernelSpec",
    "NoiseInitializer",
    "PredictiveProcess",
    "RBFKernelSpec",
    "TangentObservation",
    "TangentObservationModel",
    "reconstruct_distribution_at_time",
    "reconstruct_distributions",
    "variance_scaled_noise_initializer",
]
