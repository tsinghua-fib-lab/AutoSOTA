"""Data rescaling infrastructure for RoPEFM.

This module provides a unified interface for data rescaling (standardization,
whitening, etc.) with proper state management, serialization, and distribution
shift detection.

Design principles:
- Single source of truth for all rescaling logic
- Stateless transformations (rescalers are immutable once fitted)
- Type safety with clear interfaces
- Explicit over implicit (no hidden rescaling)
- Fail-fast validation
- Distribution shift detection for robustness

Features:
- Multiple rescaling modes: z_score, whiten (ZCA/PCA), none
- Automatic device management (CPU/GPU)
- State dict serialization with PyTorch
- Distribution shift warnings when eval data differs from training
- Model-based rescaling (parameters stored with model)

Example usage:
    # Create and fit rescaler
    rescaler = create_rescaler("z_score")
    rescaler.fit(training_data)

    # Transform data
    data_rescaled = rescaler.transform(data)

    # Validate evaluation data distribution
    rescaler.validate_distribution(eval_data)

    # Inverse transform
    data_original = rescaler.inverse_transform(data_rescaled)

    # Save and load
    state = rescaler.state_dict()
    rescaler.load_state_dict(state)

Distribution shift detection:
    Rescalers can detect when evaluation data has significantly different
    statistics than training data, which may indicate out-of-distribution
    samples or distribution shift. This is enabled by default but can be
    disabled via rescaler.enable_validation = False.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import warnings
import torch
from torch import Tensor
import torch.nn as nn


class DistributionShiftWarning(UserWarning):
    """Warning raised when evaluation data statistics differ from training."""
    pass


class DataRescaler(ABC, nn.Module):
    """Abstract base class for data rescaling transformations.

    A DataRescaler computes statistics from training data (via fit()),
    then applies forward and inverse transformations consistently.

    All rescalers are PyTorch modules to leverage:
    - Automatic device management via buffers
    - State dict serialization
    - Integration with model saving/loading
    """

    def __init__(self, mode: str):
        """Initialize rescaler.

        Args:
            mode: Rescaling mode name (e.g., "none", "z_score", "whiten")
        """
        super().__init__()
        self.mode = mode
        # Use a registered buffer for _is_fitted so PyTorch serializes it correctly
        self.register_buffer("_is_fitted_tensor", torch.tensor(False))
        self.enable_validation = True  # Enable distribution shift detection by default

    @property
    def _is_fitted(self) -> bool:
        """Whether the rescaler has been fitted."""
        return self._is_fitted_tensor.item()

    @_is_fitted.setter
    def _is_fitted(self, value: bool):
        """Set the fitted status."""
        self._is_fitted_tensor.fill_(value)

    @abstractmethod
    def fit(self, data: Tensor) -> None:
        """Compute rescaling statistics from data.

        Args:
            data: Training data tensor of shape (n_samples, n_features)
        """
        pass

    @abstractmethod
    def transform(self, data: Tensor) -> Tensor:
        """Apply forward rescaling transformation.

        Args:
            data: Data tensor to rescale

        Returns:
            Rescaled data tensor

        Raises:
            RuntimeError: If rescaler not fitted
        """
        pass

    @abstractmethod
    def inverse_transform(self, data: Tensor) -> Tensor:
        """Apply inverse rescaling transformation.

        Args:
            data: Rescaled data tensor

        Returns:
            Data in original space

        Raises:
            RuntimeError: If rescaler not fitted
        """
        pass

    def validate_distribution(
        self,
        data: Tensor,
        std_threshold: float = 3.0
    ) -> None:
        """Check if evaluation data statistics match training data.

        Warns if evaluation data has significantly different statistics
        than the data used for fitting, which may indicate distribution shift.

        Args:
            data: Evaluation data to validate
            std_threshold: Threshold in standard deviations for triggering warning.
                         Default is 3.0 (99.7% confidence interval).

        Note:
            This is a no-op for NoOpRescaler.
            Can be disabled by setting self.enable_validation = False.
        """
        if not self.enable_validation or not self._is_fitted:
            return

        # To be implemented by subclasses
        self._validate_distribution_impl(data, std_threshold)

    def _validate_distribution_impl(self, data: Tensor, std_threshold: float) -> None:
        """Implementation of distribution validation.

        To be overridden by subclasses that need validation.
        Default implementation does nothing (for NoOpRescaler).
        """
        pass

    def get_config(self) -> Dict[str, Any]:
        """Get rescaler configuration.

        Returns:
            Dictionary with mode and parameters
        """
        return {
            "mode": self.mode,
            "is_fitted": self._is_fitted
        }

    def _check_fitted(self) -> None:
        """Raise error if rescaler not fitted.

        Also auto-detects if rescaler was loaded from state_dict
        (in case load_state_dict wasn't called for nested modules).
        """
        if not self._is_fitted:
            # Try to auto-detect if we're actually fitted but flag wasn't set
            # This happens when loaded as a submodule of another model
            self._auto_detect_fitted()

            # If still not fitted, raise error
            if not self._is_fitted:
                raise RuntimeError(
                    f"Rescaler (mode={self.mode}) must be fitted before transform. "
                    f"Call fit() first."
                )

    def _auto_detect_fitted(self) -> None:
        """Auto-detect if rescaler is fitted based on buffer values.

        This is a fallback for when rescaler is loaded as submodule and
        load_state_dict wasn't called directly.
        """
        # To be overridden by subclasses if needed
        pass


class NoOpRescaler(DataRescaler):
    """No-op rescaler that passes data through unchanged.

    Used when rescale mode is "none". This is more explicit and safer
    than having None rescalers, as it maintains consistent interfaces.
    """

    def __init__(self):
        super().__init__(mode="none")
        self._is_fitted = True  # NoOp is always "fitted"

    def fit(self, data: Tensor) -> None:
        """No-op fit (does nothing)."""
        pass

    def transform(self, data: Tensor) -> Tensor:
        """Return data unchanged."""
        return data

    def inverse_transform(self, data: Tensor) -> Tensor:
        """Return data unchanged."""
        return data


class ZScoreRescaler(DataRescaler):
    """Z-score standardization: (x - mean) / std.

    Transforms data to have mean=0 and std=1 across features.
    Uses Bessel's correction (ddof=1) for unbiased std estimation.
    """

    def __init__(self, epsilon: float = 1e-8):
        """Initialize Z-score rescaler.

        Args:
            epsilon: Small constant to prevent division by zero
        """
        super().__init__(mode="z_score")
        self.epsilon = epsilon

        # Register buffers for statistics (automatically saved in state_dict)
        # Use scalar tensors (shape []) to match fit() output
        self.register_buffer("mean", torch.tensor(0.0))
        self.register_buffer("std", torch.tensor(1.0))

    def fit(self, data: Tensor) -> None:
        """Compute mean and std from data.

        Args:
            data: Training data of shape (n_samples, n_features)
        """
        if data.numel() == 0:
            raise ValueError("Cannot fit on empty data")

        # Compute statistics across all samples and features
        mean = data.mean()
        std = data.std(unbiased=True)  # Bessel's correction

        # Ensure std is not too small
        std = torch.clamp(std, min=self.epsilon)

        # Re-register buffers so shapes/devices stay consistent with state_dict
        self.register_buffer("mean", mean.detach())
        self.register_buffer("std", std.detach())
        self._is_fitted = True

    def transform(self, data: Tensor) -> Tensor:
        """Apply z-score normalization.

        Args:
            data: Data tensor of any shape

        Returns:
            Standardized data: (data - mean) / std
        """
        self._check_fitted()
        # Move mean and std to the same device as data
        mean = self.mean.to(data.device)
        std = self.std.to(data.device)
        return (data - mean) / std

    def inverse_transform(self, data: Tensor) -> Tensor:
        """Reverse z-score normalization.

        Args:
            data: Standardized data

        Returns:
            Data in original space: data * std + mean
        """
        self._check_fitted()
        # Move mean and std to the same device as data
        mean = self.mean.to(data.device)
        std = self.std.to(data.device)
        return data * std + mean

    def _validate_distribution_impl(self, data: Tensor, std_threshold: float) -> None:
        """Check if eval data statistics match training statistics.

        Compares evaluation data's mean and std with training statistics.
        Warns if they differ by more than std_threshold standard deviations.

        Args:
            data: Evaluation data
            std_threshold: Threshold in standard deviations
        """
        if data.numel() == 0:
            return

        # Compute eval data statistics
        eval_mean = data.mean().item()
        eval_std = data.std(unbiased=True).item()

        # Get training statistics
        train_mean = self.mean.item()
        train_std = self.std.item()

        # Check if eval statistics differ significantly from training
        # Use relative difference normalized by training std
        mean_diff = abs(eval_mean - train_mean) / train_std
        std_ratio = eval_std / train_std

        # Warn if mean differs by more than threshold
        if mean_diff > std_threshold:
            warnings.warn(
                f"Distribution shift detected: Evaluation data mean ({eval_mean:.4f}) "
                f"differs from training mean ({train_mean:.4f}) by {mean_diff:.2f} "
                f"standard deviations (threshold: {std_threshold}). "
                f"This may indicate the model is being applied to out-of-distribution data.",
                DistributionShiftWarning,
                stacklevel=4
            )

        # Warn if std differs significantly (more than 50% or less than 50%)
        if std_ratio > 2.0 or std_ratio < 0.5:
            warnings.warn(
                f"Distribution shift detected: Evaluation data std ({eval_std:.4f}) "
                f"differs significantly from training std ({train_std:.4f}) "
                f"(ratio: {std_ratio:.2f}). "
                f"This may indicate the model is being applied to out-of-distribution data.",
                DistributionShiftWarning,
                stacklevel=4
            )

    def get_config(self) -> Dict[str, Any]:
        """Get configuration including statistics."""
        config = super().get_config()
        if self._is_fitted:
            config.update({
                "mean": self.mean.item(),
                "std": self.std.item(),
                "epsilon": self.epsilon
            })
        return config

    def _auto_detect_fitted(self) -> None:
        """Auto-detect if fitted by checking if stats differ from defaults."""
        # If mean or std differ from default values, assume fitted
        # Default values are mean=0.0, std=1.0
        if not torch.allclose(self.mean, torch.tensor(0.0)) or \
           not torch.allclose(self.std, torch.tensor(1.0)):
            self._is_fitted = True


class WhitenRescaler(DataRescaler):
    """Whitening transformation using ZCA or PCA whitening.

    Transforms data to have identity covariance matrix.
    Uses eigendecomposition: Σ^{-1/2} (x - μ)
    """

    def __init__(self, epsilon: float = 1e-5, method: str = "zca"):
        """Initialize whitening rescaler.

        Args:
            epsilon: Regularization for small eigenvalues
            method: "zca" (Zero-phase Component Analysis) or "pca"
        """
        super().__init__(mode="whiten")
        self.epsilon = epsilon
        self.method = method

        if method not in ["zca", "pca"]:
            raise ValueError(f"method must be 'zca' or 'pca', got {method}")

        # Register buffers
        self.register_buffer("mean", torch.zeros(1))
        self.register_buffer("whiten_matrix", torch.eye(1))
        self.register_buffer("dewhiten_matrix", torch.eye(1))

    def fit(self, data: Tensor) -> None:
        """Compute whitening transformation from data.

        Args:
            data: Training data of shape (n_samples, n_features)
        """
        if data.numel() == 0:
            raise ValueError("Cannot fit on empty data")

        # Flatten to (n_samples, -1) for covariance computation
        original_shape = data.shape
        if data.dim() > 2:
            data = data.reshape(data.shape[0], -1)
        elif data.dim() == 1:
            data = data.unsqueeze(1)

        n_samples, n_features = data.shape

        # Compute mean
        mean = data.mean(dim=0, keepdim=True)

        # Center data
        data_centered = data - mean

        # Compute covariance matrix
        cov = (data_centered.T @ data_centered) / (n_samples - 1)

        # Eigendecomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)

        # Regularize small eigenvalues
        eigenvalues = torch.clamp(eigenvalues, min=self.epsilon)

        # Compute whitening matrix: Σ^{-1/2}
        if self.method == "pca":
            # PCA whitening: rotate to principal components then scale
            whiten_matrix = eigenvectors @ torch.diag(1.0 / torch.sqrt(eigenvalues))
        else:  # zca
            # ZCA whitening: rotate back to original space
            sqrt_inv = torch.diag(1.0 / torch.sqrt(eigenvalues))
            whiten_matrix = eigenvectors @ sqrt_inv @ eigenvectors.T

        # Compute de-whitening matrix (inverse)
        if self.method == "pca":
            dewhiten_matrix = eigenvectors @ torch.diag(torch.sqrt(eigenvalues))
        else:  # zca
            sqrt_mat = torch.diag(torch.sqrt(eigenvalues))
            dewhiten_matrix = eigenvectors @ sqrt_mat @ eigenvectors.T

        # Re-register buffers so shapes/devices stay consistent with state_dict
        self.register_buffer("mean", mean.detach().squeeze(0))  # (n_features,)
        self.register_buffer("whiten_matrix", whiten_matrix.detach())
        self.register_buffer("dewhiten_matrix", dewhiten_matrix.detach())
        self._is_fitted = True

    def transform(self, data: Tensor) -> Tensor:
        """Apply whitening transformation.

        Args:
            data: Data tensor of shape (..., n_features)

        Returns:
            Whitened data
        """
        self._check_fitted()

        # Handle various shapes
        original_shape = data.shape
        if data.dim() > 2:
            data = data.reshape(-1, data.shape[-1])
        elif data.dim() == 1:
            data = data.unsqueeze(0)

        # Whiten: (x - μ) @ Σ^{-1/2}
        data_centered = data - self.mean
        data_whitened = data_centered @ self.whiten_matrix.T

        # Restore original shape
        if len(original_shape) > 2:
            data_whitened = data_whitened.reshape(original_shape)
        elif len(original_shape) == 1:
            data_whitened = data_whitened.squeeze(0)

        return data_whitened

    def inverse_transform(self, data: Tensor) -> Tensor:
        """Reverse whitening transformation.

        Args:
            data: Whitened data

        Returns:
            Data in original space
        """
        self._check_fitted()

        # Handle various shapes
        original_shape = data.shape
        if data.dim() > 2:
            data = data.reshape(-1, data.shape[-1])
        elif data.dim() == 1:
            data = data.unsqueeze(0)

        # De-whiten: x @ Σ^{1/2} + μ
        data_dewhitened = data @ self.dewhiten_matrix.T + self.mean

        # Restore original shape
        if len(original_shape) > 2:
            data_dewhitened = data_dewhitened.reshape(original_shape)
        elif len(original_shape) == 1:
            data_dewhitened = data_dewhitened.squeeze(0)

        return data_dewhitened

    def _validate_distribution_impl(self, data: Tensor, std_threshold: float) -> None:
        """Check if eval data covariance matches training covariance.

        Compares evaluation data's mean and covariance structure with training.
        Warns if they differ significantly.

        Args:
            data: Evaluation data
            std_threshold: Threshold in standard deviations
        """
        if data.numel() == 0:
            return

        # Flatten data if needed
        original_shape = data.shape
        if data.dim() > 2:
            data = data.reshape(data.shape[0], -1)
        elif data.dim() == 1:
            data = data.unsqueeze(1)

        # Compute eval data statistics
        eval_mean = data.mean(dim=0)
        data_centered = data - eval_mean
        n_samples = data.shape[0]
        eval_cov = (data_centered.T @ data_centered) / (n_samples - 1)

        # Get training mean
        train_mean = self.mean

        # Check mean difference
        mean_diff = torch.norm(eval_mean - train_mean).item()
        mean_norm = torch.norm(train_mean).item()

        if mean_norm > 0:
            relative_mean_diff = mean_diff / mean_norm
            if relative_mean_diff > 0.5:  # 50% difference
                warnings.warn(
                    f"Distribution shift detected: Evaluation data mean differs from "
                    f"training mean by {relative_mean_diff:.2%}. "
                    f"This may indicate the model is being applied to out-of-distribution data.",
                    DistributionShiftWarning,
                    stacklevel=4
                )

        # Check covariance structure by looking at Frobenius norm difference
        # Reconstruct approximate training covariance from whitening matrix
        # For whitened data, Cov = I, so original Cov ≈ (W^{-1})^T @ I @ W^{-1}
        # This is approximated by dewhiten_matrix @ dewhiten_matrix.T
        train_cov_approx = self.dewhiten_matrix @ self.dewhiten_matrix.T

        cov_diff = torch.norm(eval_cov - train_cov_approx, p='fro').item()
        cov_norm = torch.norm(train_cov_approx, p='fro').item()

        if cov_norm > 0:
            relative_cov_diff = cov_diff / cov_norm
            if relative_cov_diff > 0.5:  # 50% difference
                warnings.warn(
                    f"Distribution shift detected: Evaluation data covariance structure "
                    f"differs from training by {relative_cov_diff:.2%}. "
                    f"This may indicate the model is being applied to out-of-distribution data.",
                    DistributionShiftWarning,
                    stacklevel=4
                )

    def get_config(self) -> Dict[str, Any]:
        """Get configuration."""
        config = super().get_config()
        config.update({
            "epsilon": self.epsilon,
            "method": self.method
        })
        return config


def create_rescaler(mode: str, **kwargs) -> DataRescaler:
    """Factory function to create rescalers.

    Args:
        mode: Rescaling mode - one of {"none", "z_score", "whiten"}
        **kwargs: Additional arguments passed to rescaler constructor

    Returns:
        Initialized DataRescaler instance

    Raises:
        ValueError: If mode is not recognized

    Example:
        >>> rescaler = create_rescaler("z_score")
        >>> rescaler.fit(data)
        >>> data_rescaled = rescaler.transform(data)
    """
    mode = mode.lower() if mode else "none"

    if mode == "none":
        return NoOpRescaler()
    elif mode == "z_score":
        return ZScoreRescaler(**kwargs)
    elif mode == "whiten":
        return WhitenRescaler(**kwargs)
    else:
        raise ValueError(
            f"Unknown rescaling mode: '{mode}'. "
            f"Must be one of: 'none', 'z_score', 'whiten'"
        )


def validate_rescaler_compatibility(
    rescaler1: DataRescaler,
    rescaler2: DataRescaler,
    name1: str = "rescaler1",
    name2: str = "rescaler2"
) -> None:
    """Validate that two rescalers have compatible configurations.

    Used when combining models that should use the same rescaling.

    Args:
        rescaler1: First rescaler
        rescaler2: Second rescaler
        name1: Name for first rescaler (for error messages)
        name2: Name for second rescaler (for error messages)

    Raises:
        ValueError: If rescalers have incompatible modes
    """
    if rescaler1.mode != rescaler2.mode:
        raise ValueError(
            f"Rescaler mode mismatch: {name1} uses '{rescaler1.mode}', "
            f"{name2} uses '{rescaler2.mode}'"
        )
