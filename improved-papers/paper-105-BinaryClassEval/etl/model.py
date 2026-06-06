"""Data structures for subgroup classification results.

These lightweight containers are *not* intended to provide heavy-duty data
processing but merely to hold the information required by downstream metric
and visualization utilities.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence, Union

import numpy as np
from sklearn.isotonic import IsotonicRegression

ArrayLike = Union[Sequence[float], np.ndarray]

class Model(ABC):
    """Abstract base class for model implementations.

    This class defines the minimum interface that model implementations
    must provide: access to predictions for each class and training prevalence.
    Subclasses should implement the abstract properties train_0, train_1,
    and train_prevalence.
    """

    @property
    @abstractmethod
    def train_0(self) -> np.ndarray:
        """Return predictions for class 0 samples.

        Returns
        -------
        np.ndarray
            Array of predicted probabilities for negative class samples.
        """

    @property
    @abstractmethod
    def train_1(self) -> np.ndarray:
        """Return predictions for class 1 samples.

        Returns
        -------
        np.ndarray
            Array of predicted probabilities for positive class samples.
        """

    @property
    @abstractmethod
    def train_prevalence(self) -> float:
        """Return the prevalence used during training.

        Returns
        -------
        float
            Prevalence (proportion of positive class) in training data.
        """

    def get_fpr_fnr(self, thresholds: ArrayLike | None = None):
        """Compute false-positive and false-negative rates for given thresholds.

        Parameters
        ----------
        thresholds : ArrayLike, default=None
            Threshold values for classification. If None, uses 0.5.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Tuple of (fpr, fnr) arrays with false positive and false negative rates.
        """
        if thresholds is None:
            thresholds = np.array([0.5])
        if isinstance(thresholds, list):
            thresholds = np.array(thresholds)
        elif isinstance(thresholds, float):
            thresholds = np.array([thresholds])

        # Classify based on threshold(s).
        fpr = np.mean(self.train_0[:, None] > thresholds[None, :], axis=0)
        fnr = np.mean(self.train_1[:, None] < thresholds[None, :], axis=0)
        return fpr, fnr

    @staticmethod
    def calibrate(scores: np.ndarray, y_true: np.ndarray) -> IsotonicRegression:
        """Apply isotonic regression to calibrate probability scores.

        Parameters
        ----------
        scores : np.ndarray
            Uncalibrated probability predictions.
        y_true : np.ndarray
            Binary ground-truth labels.

        Returns
        -------
        IsotonicRegression
            Fitted isotonic regression model for calibration.
        """
        return IsotonicRegression(out_of_bounds="clip").fit(scores, y_true)
