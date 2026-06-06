"""Basic metric calculations for net benefit analysis."""
from __future__ import annotations

import numpy as np
from typing import Sequence, Tuple, Optional

from core.proto import ArrayLike, Number

def calculate_brier_score(probabilities: ArrayLike, outcomes: ArrayLike) -> float:
    """Calculate the Brier score for a set of probability predictions and binary outcomes.

    Parameters
    ----------
    probabilities : ArrayLike
        Predicted probabilities for the positive class.
    outcomes : ArrayLike
        Binary ground-truth labels (0 or 1).

    Returns
    -------
    float
        Mean squared error between predictions and outcomes.
    """
    probabilities = np.asarray(probabilities)
    outcomes = np.asarray(outcomes)
    return np.mean((probabilities - outcomes) ** 2)

def calculate_auc(probabilities: ArrayLike, outcomes: ArrayLike) -> float:
    """Calculate the Area Under the ROC Curve (AUC) for binary classification.

    Parameters
    ----------
    probabilities : ArrayLike
        Predicted probabilities for the positive class.
    outcomes : ArrayLike
        Binary ground-truth labels (0 or 1).

    Returns
    -------
    float
        AUC-ROC score. Returns 0.5 for degenerate cases (no positive or negative samples).
    """
    # This is a placeholder - in a real implementation you would use sklearn or other libraries
    # But we keep it minimal for demonstration purposes
    probabilities = np.asarray(probabilities)
    outcomes = np.asarray(outcomes)
    
    # Simple AUC approximation
    pos_probs = probabilities[outcomes == 1]
    neg_probs = probabilities[outcomes == 0]
    
    if len(pos_probs) == 0 or len(neg_probs) == 0:
        return 0.5  # Default for degenerate case
    
    # Count pairwise comparisons where positive example has higher probability
    auc = 0.0
    for pos_prob in pos_probs:
        auc += np.sum(pos_prob > neg_probs)
    
    # Normalize
    return auc / (len(pos_probs) * len(neg_probs))

def calculate_calibration_error(probabilities: ArrayLike, outcomes: ArrayLike,
                                n_bins: int = 10) -> float:
    """Calculate calibration error using binning approach.

    Parameters
    ----------
    probabilities : ArrayLike
        Predicted probabilities for the positive class.
    outcomes : ArrayLike
        Binary ground-truth labels (0 or 1).
    n_bins : int, default=10
        Number of bins to use for calibration calculation.

    Returns
    -------
    float
        Weighted average of absolute differences between predicted and observed frequencies.
    """
    probabilities = np.asarray(probabilities)
    outcomes = np.asarray(outcomes)
    
    # Create bins
    bins = np.linspace(0, 1, n_bins + 1)
    binned_indices = np.digitize(probabilities, bins) - 1
    binned_indices = np.clip(binned_indices, 0, n_bins - 1)  # Ensure valid bins
    
    # Calculate calibration error
    cal_error = 0.0
    bin_counts = np.bincount(binned_indices, minlength=n_bins)
    
    for bin_idx in range(n_bins):
        bin_mask = binned_indices == bin_idx
        if np.sum(bin_mask) > 0:
            bin_probs = probabilities[bin_mask]
            bin_outcomes = outcomes[bin_mask]
            
            avg_prob = np.mean(bin_probs)
            avg_outcome = np.mean(bin_outcomes)
            
            # Weighted absolute difference
            cal_error += np.abs(avg_prob - avg_outcome) * (bin_counts[bin_idx] / len(probabilities))
    
    return cal_error
