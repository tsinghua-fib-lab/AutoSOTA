"""Core net benefit calculations for model evaluation."""
from __future__ import annotations

import numpy as np

# Laplace smoothing alpha for prevalence estimation (Bayesian regularization for small samples)
_LAPLACE_ALPHA = 2.0


def otimes(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute the odds-weighted combination of two probabilities.

    This is an important helper function for computing net benefit scores.

    Parameters
    ----------
    a : np.ndarray
        First probability array.
    b : np.ndarray
        Second probability array.

    Returns
    -------
    np.ndarray
        Odds-weighted combination: (a * b) / (a * b + (1 - a) * (1 - b))
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        result = (a * b) / (a * b + (1 - a) * (1 - b))
    return result


def net_benefit_for_prevalences(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    prevalence_grid: np.ndarray,
    cost_ratio: float = 1.0,
    train_prevalence: float | None = None,
    normalize: bool = True
) -> np.ndarray:
    """Compute net benefit curve across a range of prevalence values.

    This function calculates the net benefit of a model for different prevalence values,
    with proper normalization to keep values in the [0, 1] range.

    Parameters
    ----------
    y_true : np.ndarray
        Binary ground-truth labels.
    y_pred_proba : np.ndarray
        Probabilistic predictions for the positive class.
    prevalence_grid : np.ndarray
        Grid of prevalence values to compute net benefit for.
    cost_ratio : float, default=1.0
        Cost ratio parameter (FP cost / FN cost).
    train_prevalence : float, default=None
        Training prevalence. If None, computed from y_true.
    normalize : bool, default=True
        Whether to normalize net benefit. The original implementation normalizes by default.

    Returns
    -------
    np.ndarray
        Array of net benefit values corresponding to prevalence_grid.
    """
    # Apply Laplace smoothing for prevalence estimation (Bayesian regularization).
    # This improves stability for small subgroups where raw prevalence may be unreliable.
    # Formula: (positives + alpha) / (n + 2*alpha), where alpha=_LAPLACE_ALPHA
    n = len(y_true)
    pos = np.sum(y_true)
    alpha = _LAPLACE_ALPHA
    train_prevalence = (pos + alpha) / (n + 2 * alpha)

    # Make sure predictions are sorted in ascending order
    sort_idx = np.argsort(y_pred_proba)
    y_true = y_true[sort_idx]
    y_pred_proba = y_pred_proba[sort_idx]

    # Calculate basic counts
    ground_p = np.sum(y_true)
    ground_n = np.sum(1 - y_true)

    # Precompute cumulative sum (pad with 0 at start for lookups)
    cumsum = np.pad(np.cumsum(y_true), (1, 0), mode='constant', constant_values=0)

    # Calculate thresholds for each prevalence value using odds-weighted combination
    thresholds = otimes(cost_ratio, otimes(train_prevalence, 1 - prevalence_grid))

    # Get indices of where these thresholds would be inserted in sorted predictions
    idx = np.searchsorted(y_pred_proba, thresholds)

    # Calculate true/false positive/negative rates
    if ground_p > 0:
        tpr = 1 - cumsum[idx] / ground_p
    else:
        tpr = np.zeros_like(idx, dtype=float)

    tnr = (idx - cumsum[idx]) / ground_n if ground_n > 0 else np.zeros_like(idx, dtype=float)

    # Calculate final net benefit with proper normalization
    if normalize:
        return (
            tpr * otimes(1 - cost_ratio, prevalence_grid) +
            tnr * otimes(cost_ratio, 1 - prevalence_grid)
        )
    else:
        with np.errstate(divide='ignore', invalid='ignore'):
            result = (
                tpr * prevalence_grid +
                tnr * (1 - prevalence_grid) * cost_ratio / (1 - cost_ratio)
            )
        return result
