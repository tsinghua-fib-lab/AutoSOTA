"""
Core metrics for cost-sensitive learning experiments.

This module provides standardized metric computation for both
classification and regression tasks, with support for weighted
evaluation using |Delta| (absolute delta) as example-level importance weights.

Metrics:
  Classification:
    - accuracy: Standard unweighted accuracy
    - weighted_accuracy: |Delta|-weighted accuracy
    - expected_cost: Expected misclassification cost using |Delta| weights

  Regression:
    - mae, rmse: Mean Absolute Error and Root Mean Squared Error
    - weighted_mae, weighted_rmse: |Delta|-weighted versions
    - sign_accuracy: Accuracy of predicting sign(Delta) from regression output
    - weighted_sign_accuracy: |Delta|-weighted sign accuracy

All functions expect numpy arrays and return Python floats.
"""

import numpy as np
from typing import Union, Dict


# ============================================================================
# Classification Metrics
# ============================================================================

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Standard unweighted accuracy.

    Args:
        y_true: True labels, shape (N,)
        y_pred: Predicted labels, shape (N,)

    Returns:
        Accuracy as a float in [0, 1]
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float((y_true == y_pred).mean())


def weighted_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: np.ndarray,
) -> float:
    """
    Weighted accuracy using example-level weights (typically |Delta|).

    Computes: sum(w * correct) / sum(w)
    where correct[i] = 1 if y_true[i] == y_pred[i], else 0

    Args:
        y_true: True labels, shape (N,)
        y_pred: Predicted labels, shape (N,)
        weights: Per-example weights (e.g., |Delta|), shape (N,)

    Returns:
        Weighted accuracy as a float in [0, 1]
        Returns NaN if sum(weights) <= 0 or not finite
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    weights = np.asarray(weights, dtype=np.float32)

    correct = (y_true == y_pred).astype(np.float32)
    weight_sum = float(weights.sum())

    if weight_sum <= 0.0 or not np.isfinite(weight_sum):
        return float("nan")

    return float((weights * correct).sum() / weight_sum)


def expected_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    costs: np.ndarray,
) -> float:
    """
    Expected misclassification cost.

    Computes: sum(cost[i] * incorrect[i]) / N
    where incorrect[i] = 1 if y_true[i] != y_pred[i], else 0

    This is the average cost incurred by incorrect predictions,
    weighted by the per-example cost (typically |Delta|).

    Args:
        y_true: True labels, shape (N,)
        y_pred: Predicted labels, shape (N,)
        costs: Per-example costs (e.g., |Delta|), shape (N,)

    Returns:
        Average cost of misclassifications as a float
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    costs = np.asarray(costs, dtype=np.float32)

    incorrect = (y_true != y_pred).astype(np.float32)
    return float((costs * incorrect).mean())


# ============================================================================
# Regression Metrics
# ============================================================================

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Error.

    Args:
        y_true: True values, shape (N,)
        y_pred: Predicted values, shape (N,)

    Returns:
        MAE as a float
    """
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    return float(np.abs(y_true - y_pred).mean())


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Error.

    Args:
        y_true: True values, shape (N,)
        y_pred: Predicted values, shape (N,)

    Returns:
        RMSE as a float
    """
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    return float(np.sqrt(((y_true - y_pred) ** 2).mean()))


def weighted_mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: np.ndarray,
) -> float:
    """
    Weighted Mean Absolute Error.

    Computes: sum(w * |y_true - y_pred|) / sum(w)

    Args:
        y_true: True values, shape (N,)
        y_pred: Predicted values, shape (N,)
        weights: Per-example weights (e.g., |Delta|), shape (N,)

    Returns:
        Weighted MAE as a float
        Returns NaN if sum(weights) <= 0 or not finite
    """
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)

    weight_sum = float(weights.sum())
    if weight_sum <= 0.0 or not np.isfinite(weight_sum):
        return float("nan")

    return float((weights * np.abs(y_true - y_pred)).sum() / weight_sum)


def weighted_rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: np.ndarray,
) -> float:
    """
    Weighted Root Mean Squared Error.

    Computes: sqrt(sum(w * (y_true - y_pred)^2) / sum(w))

    Args:
        y_true: True values, shape (N,)
        y_pred: Predicted values, shape (N,)
        weights: Per-example weights (e.g., |Delta|), shape (N,)

    Returns:
        Weighted RMSE as a float
        Returns NaN if sum(weights) <= 0 or not finite
    """
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)

    weight_sum = float(weights.sum())
    if weight_sum <= 0.0 or not np.isfinite(weight_sum):
        return float("nan")

    return float(np.sqrt((weights * (y_true - y_pred) ** 2).sum() / weight_sum))


def sign_accuracy(
    y_true: np.ndarray,
    delta_pred: np.ndarray,
    threshold: float = 0.0,
) -> float:
    """
    Accuracy of predicting binary label from signed delta regression.

    Converts continuous delta predictions to binary labels using threshold,
    then computes accuracy against true binary labels.

    Args:
        y_true: True binary labels (0/1), shape (N,)
        delta_pred: Predicted signed delta values, shape (N,)
        threshold: Threshold for converting delta to binary (default: 0.0)

    Returns:
        Sign accuracy as a float in [0, 1]
    """
    y_true = np.asarray(y_true)
    delta_pred = np.asarray(delta_pred, dtype=np.float32)

    y_pred = (delta_pred >= threshold).astype(int)
    return float((y_true == y_pred).mean())


def weighted_sign_accuracy(
    y_true: np.ndarray,
    delta_pred: np.ndarray,
    weights: np.ndarray,
    threshold: float = 0.0,
) -> float:
    """
    Weighted accuracy of predicting binary label from signed delta regression.

    Converts continuous delta predictions to binary labels using threshold,
    then computes weighted accuracy against true binary labels.

    Args:
        y_true: True binary labels (0/1), shape (N,)
        delta_pred: Predicted signed delta values, shape (N,)
        weights: Per-example weights (e.g., |Delta|), shape (N,)
        threshold: Threshold for converting delta to binary (default: 0.0)

    Returns:
        Weighted sign accuracy as a float in [0, 1]
        Returns NaN if sum(weights) <= 0 or not finite
    """
    y_true = np.asarray(y_true)
    delta_pred = np.asarray(delta_pred, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)

    y_pred = (delta_pred >= threshold).astype(int)
    return weighted_accuracy(y_true, y_pred, weights)


# ============================================================================
# Convenience Functions for Multiple Metrics
# ============================================================================

def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: Union[np.ndarray, None] = None,
) -> Dict[str, float]:
    """
    Compute all classification metrics at once.

    Args:
        y_true: True labels, shape (N,)
        y_pred: Predicted labels, shape (N,)
        weights: Optional per-example weights (e.g., |Delta|), shape (N,)

    Returns:
        Dictionary with keys:
            - 'accuracy': Unweighted accuracy
            - 'weighted_accuracy': |Delta|-weighted accuracy (if weights provided)
            - 'expected_cost': Expected misclassification cost (if weights provided)
    """
    metrics = {
        'accuracy': accuracy(y_true, y_pred),
    }

    if weights is not None:
        metrics['weighted_accuracy'] = weighted_accuracy(y_true, y_pred, weights)
        metrics['expected_cost'] = expected_cost(y_true, y_pred, weights)

    return metrics


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    delta_true: Union[np.ndarray, None] = None,
    weights: Union[np.ndarray, None] = None,
) -> Dict[str, float]:
    """
    Compute all regression metrics at once.

    Args:
        y_true: True binary labels (for sign accuracy), shape (N,)
        y_pred: Predicted signed delta values, shape (N,)
        delta_true: True signed delta values (for MAE/RMSE), shape (N,)
                   If None, uses y_pred for error metrics
        weights: Optional per-example weights (e.g., |Delta|), shape (N,)

    Returns:
        Dictionary with keys:
            - 'mae': Mean Absolute Error
            - 'rmse': Root Mean Squared Error
            - 'sign_accuracy': Accuracy of sign prediction
            - 'weighted_mae': Weighted MAE (if weights provided)
            - 'weighted_rmse': Weighted RMSE (if weights provided)
            - 'weighted_sign_accuracy': Weighted sign accuracy (if weights provided)
    """
    if delta_true is None:
        delta_true = y_pred

    metrics = {
        'mae': mae(delta_true, y_pred),
        'rmse': rmse(delta_true, y_pred),
        'sign_accuracy': sign_accuracy(y_true, y_pred),
    }

    if weights is not None:
        metrics['weighted_mae'] = weighted_mae(delta_true, y_pred, weights)
        metrics['weighted_rmse'] = weighted_rmse(delta_true, y_pred, weights)
        metrics['weighted_sign_accuracy'] = weighted_sign_accuracy(y_true, y_pred, weights)

    return metrics


# ============================================================================
# Diagnostic Metrics
# ============================================================================

def corr_delta_misclass(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    abs_delta: np.ndarray,
) -> float:
    """
    Pearson correlation between |Delta| and misclassification indicator.

    This diagnostic metric measures whether higher |Delta| examples
    are more likely to be misclassified, which would validate the
    cost-sensitive learning approach.

    Args:
        y_true: True labels, shape (N,)
        y_pred: Predicted labels, shape (N,)
        abs_delta: Absolute delta values, shape (N,)

    Returns:
        Pearson correlation coefficient as a float in [-1, 1]
        Returns NaN if either variable has zero standard deviation
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    abs_delta = np.asarray(abs_delta, dtype=np.float32)

    # Misclassification indicator: 1 if wrong, 0 if correct
    misclass = (y_true != y_pred).astype(np.float32)

    # Check for zero variance
    if abs_delta.std() <= 0.0 or misclass.std() <= 0.0:
        return float("nan")

    # Compute Pearson correlation
    return float(np.corrcoef(abs_delta, misclass)[0, 1])


def normalize_delta(
    delta: np.ndarray,
    target_sum: Union[float, None] = None,
) -> np.ndarray:
    """
    Normalize delta values so that sum of absolute values equals target.

    If target_sum is None, normalizes so that sum(|delta'|) = n (length).
    This ensures average absolute delta is 1.0.

    Args:
        delta: Delta values (signed), shape (N,)
        target_sum: Target sum for |delta|. If None, uses len(delta)

    Returns:
        Normalized delta values, same shape as input
        Returns all zeros if sum(|delta|) is zero or not finite
    """
    delta = np.asarray(delta, dtype=np.float32)

    if target_sum is None:
        target_sum = float(len(delta))

    abs_sum = float(np.abs(delta).sum())

    if abs_sum <= 0.0 or not np.isfinite(abs_sum):
        return np.zeros_like(delta)

    scale = target_sum / abs_sum
    return delta * scale
