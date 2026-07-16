"""Utility helpers for scaling GP inputs and outputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class InputScaler:
    """Scale inputs to [0, 1] range for better GP hyperparameter optimization.

    This scaler applies a linear transformation to map inputs from [t_min, t_max]
    to [0, 1].
    """

    t_min: float
    t_max: float

    def __post_init__(self) -> None:
        if self.t_max <= self.t_min:
            raise ValueError("t_max must be greater than t_min")

    @classmethod
    def from_data(cls, times: np.ndarray) -> "InputScaler":
        """Create a scaler from training data."""
        t_min = float(np.min(times))
        t_max = float(np.max(times))
        if t_max == t_min:
            t_max = t_min + 1.0
        return cls(t_min=t_min, t_max=t_max)

    def forward(self, t: np.ndarray | float) -> np.ndarray:
        """Scale input times to [0, 1]."""
        t_arr = np.asarray(t, dtype=float)
        return (t_arr - self.t_min) / (self.t_max - self.t_min)

    def inverse(self, t_scaled: np.ndarray | float) -> np.ndarray:
        """Transform scaled times back to original scale."""
        t_arr = np.asarray(t_scaled, dtype=float)
        return t_arr * (self.t_max - self.t_min) + self.t_min

    @property
    def scale_factor(self) -> float:
        """Return the scale factor (t_max - t_min)."""
        return self.t_max - self.t_min


@dataclass
class OutputScaler:
    """Scale outputs to unit variance and rescale predictions."""

    scales: np.ndarray

    def __post_init__(self) -> None:
        self.scales = np.asarray(self.scales, dtype=float)
        if self.scales.ndim != 1:
            raise ValueError("scales must be one-dimensional")
        if np.any(self.scales <= 0):
            raise ValueError("scales must be positive")

    @classmethod
    def from_data(cls, values: np.ndarray, min_scale: float = 1e-6) -> "OutputScaler":
        arr = np.asarray(values, dtype=float)
        if arr.ndim != 2:
            raise ValueError("values must have shape (rank, n_obs)")
        if arr.shape[1] == 0:
            raise ValueError("values must contain at least one observation")
        if arr.shape[1] == 1:
            scales = np.maximum(np.abs(arr[:, 0]), min_scale)
        else:
            scales = np.std(arr, axis=1, ddof=1)
            scales = np.maximum(scales, min_scale)
        return cls(scales=scales)

    def scale(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        if arr.ndim == 1:
            if arr.shape[0] != self.scales.shape[0]:
                raise ValueError("value shape does not match scales")
            return arr / self.scales
        if arr.shape[0] != self.scales.shape[0]:
            raise ValueError("value shape does not match scales")
        return arr / self.scales[:, None]

    def unscale(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        if arr.ndim == 1:
            if arr.shape[0] != self.scales.shape[0]:
                raise ValueError("value shape does not match scales")
            return arr * self.scales
        if arr.shape[0] != self.scales.shape[0]:
            raise ValueError("value shape does not match scales")
        return arr * self.scales[:, None]

    def scale_variance(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        if arr.ndim == 1:
            if arr.shape[0] != self.scales.shape[0]:
                raise ValueError("value shape does not match scales")
            return arr / (self.scales**2)
        if arr.shape[0] != self.scales.shape[0]:
            raise ValueError("value shape does not match scales")
        return arr / (self.scales[:, None] ** 2)

    def unscale_variance(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        if arr.ndim == 1:
            if arr.shape[0] != self.scales.shape[0]:
                raise ValueError("value shape does not match scales")
            return arr * (self.scales**2)
        if arr.shape[0] != self.scales.shape[0]:
            raise ValueError("value shape does not match scales")
        return arr * (self.scales[:, None] ** 2)
