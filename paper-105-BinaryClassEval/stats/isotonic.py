"""Calibration utilities for prediction scores."""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


def calibrate_scores(y_true: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Calibrate scores using isotonic regression.

    Parameters
    ----------
    y_true : np.ndarray
        Binary ground-truth labels.
    scores : np.ndarray
        Uncalibrated probability scores.

    Returns
    -------
    np.ndarray
        Calibrated probability scores.
    """
    # Create and fit isotonic regression model
    ir = IsotonicRegression(out_of_bounds='clip')
    ir.fit(scores, y_true)

    # Transform scores
    return ir.transform(scores)


def get_calibration_model(y_true: np.ndarray, scores: np.ndarray) -> IsotonicRegression:
    """Get a fitted isotonic regression model for calibration.

    Parameters
    ----------
    y_true : np.ndarray
        Binary ground-truth labels.
    scores : np.ndarray
        Uncalibrated probability scores.

    Returns
    -------
    IsotonicRegression
        Fitted isotonic regression model.
    """
    ir = IsotonicRegression(out_of_bounds='clip')
    ir.fit(scores, y_true)
    return ir
