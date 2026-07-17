"""
Tests for src/core/metrics.py

Validates all metric computations including edge cases.
"""

import numpy as np
import pytest
from core.metrics import (
    accuracy,
    weighted_accuracy,
    expected_cost,
    mae,
    rmse,
    weighted_mae,
    weighted_rmse,
    sign_accuracy,
    weighted_sign_accuracy,
    classification_metrics,
    regression_metrics,
    corr_delta_misclass,
    normalize_delta,
)


# ============================================================================
# Classification Metrics Tests
# ============================================================================

def test_accuracy_perfect():
    """Test accuracy with perfect predictions."""
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 0, 1, 1, 0, 1])
    assert np.isclose(accuracy(y_true, y_pred), 1.0)


def test_accuracy_all_wrong():
    """Test accuracy with all wrong predictions."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([1, 1, 0, 0])
    assert np.isclose(accuracy(y_true, y_pred), 0.0)


def test_accuracy_mixed():
    """Test accuracy with mixed predictions."""
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0, 0, 1])
    # Correct: indices 0, 2, 4, 5 = 4/6
    assert np.isclose(accuracy(y_true, y_pred), 4.0 / 6.0)


def test_weighted_accuracy_uniform_weights():
    """Test weighted accuracy with uniform weights matches unweighted."""
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0, 0, 1])
    weights = np.ones(6)

    unweighted = accuracy(y_true, y_pred)
    weighted = weighted_accuracy(y_true, y_pred, weights)
    assert np.isclose(unweighted, weighted)


def test_weighted_accuracy_high_weight_on_correct():
    """Test weighted accuracy gives more weight to correct predictions."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])
    # Correct at indices 0, 3 (weights 10, 10)
    # Wrong at indices 1, 2 (weights 1, 1)
    weights = np.array([10.0, 1.0, 1.0, 10.0])

    # Expected: (10*1 + 1*0 + 1*0 + 10*1) / (10 + 1 + 1 + 10) = 20/22
    expected = 20.0 / 22.0
    assert np.isclose(weighted_accuracy(y_true, y_pred, weights), expected)


def test_weighted_accuracy_zero_weights_returns_nan():
    """Test weighted accuracy returns NaN when all weights are zero."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 0])
    weights = np.zeros(4)

    result = weighted_accuracy(y_true, y_pred, weights)
    assert np.isnan(result)


def test_weighted_accuracy_negative_weights_returns_nan():
    """Test weighted accuracy returns NaN when weight sum is negative."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 0])
    weights = np.array([-1.0, -1.0, -1.0, -1.0])

    result = weighted_accuracy(y_true, y_pred, weights)
    assert np.isnan(result)


def test_expected_cost_all_correct():
    """Test expected cost is zero when all predictions are correct."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    costs = np.array([1.0, 2.0, 3.0, 4.0])

    assert np.isclose(expected_cost(y_true, y_pred, costs), 0.0)


def test_expected_cost_all_wrong():
    """Test expected cost is mean of costs when all predictions are wrong."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([1, 1, 0, 0])
    costs = np.array([1.0, 2.0, 3.0, 4.0])

    # All wrong, so cost = mean([1, 2, 3, 4]) = 2.5
    expected = 2.5
    assert np.isclose(expected_cost(y_true, y_pred, costs), expected)


def test_expected_cost_mixed():
    """Test expected cost with mixed predictions."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])
    costs = np.array([1.0, 2.0, 3.0, 4.0])

    # Wrong at indices 1, 2 with costs 2.0, 3.0
    # Expected: mean([0, 2, 3, 0]) = 5/4 = 1.25
    expected = 1.25
    assert np.isclose(expected_cost(y_true, y_pred, costs), expected)


# ============================================================================
# Regression Metrics Tests
# ============================================================================

def test_mae_perfect():
    """Test MAE with perfect predictions."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.isclose(mae(y_true, y_pred), 0.0)


def test_mae_constant_error():
    """Test MAE with constant error."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([2.0, 3.0, 4.0, 5.0])
    # All errors are 1.0
    assert np.isclose(mae(y_true, y_pred), 1.0)


def test_mae_mixed_errors():
    """Test MAE with mixed errors."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([2.0, 2.0, 5.0, 3.0])
    # Errors: |1-2|=1, |2-2|=0, |3-5|=2, |4-3|=1
    # MAE = (1 + 0 + 2 + 1) / 4 = 1.0
    assert np.isclose(mae(y_true, y_pred), 1.0)


def test_rmse_perfect():
    """Test RMSE with perfect predictions."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.isclose(rmse(y_true, y_pred), 0.0)


def test_rmse_constant_error():
    """Test RMSE with constant error."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([2.0, 3.0, 4.0, 5.0])
    # All errors are 1.0, so RMSE = sqrt(mean([1, 1, 1, 1])) = 1.0
    assert np.isclose(rmse(y_true, y_pred), 1.0)


def test_rmse_mixed_errors():
    """Test RMSE with mixed errors."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([2.0, 2.0, 5.0, 3.0])
    # Errors: 1, 0, 2, 1
    # Squared: 1, 0, 4, 1
    # MSE = (1 + 0 + 4 + 1) / 4 = 1.5
    # RMSE = sqrt(1.5) H 1.2247
    expected = np.sqrt(1.5)
    assert np.isclose(rmse(y_true, y_pred), expected)


def test_weighted_mae_uniform_weights():
    """Test weighted MAE with uniform weights matches unweighted."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([2.0, 2.0, 5.0, 3.0])
    weights = np.ones(4)

    unweighted = mae(y_true, y_pred)
    weighted = weighted_mae(y_true, y_pred, weights)
    assert np.isclose(unweighted, weighted)


def test_weighted_mae_high_weight_on_large_error():
    """Test weighted MAE gives more weight to large errors."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([2.0, 2.0, 5.0, 3.0])
    # Errors: 1, 0, 2, 1
    # Give high weight to index 2 (error=2)
    weights = np.array([1.0, 1.0, 10.0, 1.0])

    # Expected: (1*1 + 1*0 + 10*2 + 1*1) / (1 + 1 + 10 + 1) = 22/13
    expected = 22.0 / 13.0
    assert np.isclose(weighted_mae(y_true, y_pred, weights), expected)


def test_weighted_mae_zero_weights_returns_nan():
    """Test weighted MAE returns NaN when all weights are zero."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([2.0, 2.0, 5.0, 3.0])
    weights = np.zeros(4)

    result = weighted_mae(y_true, y_pred, weights)
    assert np.isnan(result)


def test_weighted_rmse_uniform_weights():
    """Test weighted RMSE with uniform weights matches unweighted."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([2.0, 2.0, 5.0, 3.0])
    weights = np.ones(4)

    unweighted = rmse(y_true, y_pred)
    weighted = weighted_rmse(y_true, y_pred, weights)
    assert np.isclose(unweighted, weighted)


def test_weighted_rmse_high_weight_on_large_error():
    """Test weighted RMSE gives more weight to large errors."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([2.0, 2.0, 5.0, 3.0])
    # Errors: 1, 0, 2, 1
    # Squared errors: 1, 0, 4, 1
    # Give high weight to index 2 (squared error=4)
    weights = np.array([1.0, 1.0, 10.0, 1.0])

    # Expected: sqrt((1*1 + 1*0 + 10*4 + 1*1) / (1 + 1 + 10 + 1)) = sqrt(42/13)
    expected = np.sqrt(42.0 / 13.0)
    assert np.isclose(weighted_rmse(y_true, y_pred, weights), expected)


def test_weighted_rmse_zero_weights_returns_nan():
    """Test weighted RMSE returns NaN when all weights are zero."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([2.0, 2.0, 5.0, 3.0])
    weights = np.zeros(4)

    result = weighted_rmse(y_true, y_pred, weights)
    assert np.isnan(result)


# ============================================================================
# Sign Accuracy Tests
# ============================================================================

def test_sign_accuracy_all_correct():
    """Test sign accuracy when all signs are correct."""
    y_true = np.array([0, 0, 1, 1])
    delta_pred = np.array([-2.0, -1.0, 1.0, 2.0])
    # y_pred from delta_pred >= 0: [0, 0, 1, 1]
    assert np.isclose(sign_accuracy(y_true, delta_pred), 1.0)


def test_sign_accuracy_all_wrong():
    """Test sign accuracy when all signs are wrong."""
    y_true = np.array([0, 0, 1, 1])
    delta_pred = np.array([2.0, 1.0, -1.0, -2.0])
    # y_pred from delta_pred >= 0: [1, 1, 0, 0]
    assert np.isclose(sign_accuracy(y_true, delta_pred), 0.0)


def test_sign_accuracy_at_threshold():
    """Test sign accuracy at threshold boundary."""
    y_true = np.array([0, 1, 0, 1])
    delta_pred = np.array([0.0, 0.0, -0.01, 0.01])
    # At threshold=0: 0.0 >= 0 is True, so y_pred = [1, 1, 0, 1]
    # Correct at indices 1, 2, 3 = 3/4
    assert np.isclose(sign_accuracy(y_true, delta_pred, threshold=0.0), 3.0 / 4.0)


def test_sign_accuracy_custom_threshold():
    """Test sign accuracy with custom threshold."""
    y_true = np.array([0, 0, 1, 1])
    delta_pred = np.array([-2.0, 0.5, 1.0, 2.0])
    # At threshold=1.0: y_pred = [0, 0, 1, 1]
    assert np.isclose(sign_accuracy(y_true, delta_pred, threshold=1.0), 1.0)


def test_weighted_sign_accuracy_uniform_weights():
    """Test weighted sign accuracy with uniform weights matches unweighted."""
    y_true = np.array([0, 0, 1, 1])
    delta_pred = np.array([-2.0, 1.0, 1.0, -2.0])
    weights = np.ones(4)

    unweighted = sign_accuracy(y_true, delta_pred)
    weighted = weighted_sign_accuracy(y_true, delta_pred, weights)
    assert np.isclose(unweighted, weighted)


def test_weighted_sign_accuracy_high_weight_on_correct():
    """Test weighted sign accuracy gives more weight to correct predictions."""
    y_true = np.array([0, 0, 1, 1])
    delta_pred = np.array([-2.0, 1.0, 1.0, -2.0])
    # y_pred from delta_pred >= 0: [0, 1, 1, 0]
    # Correct at indices 0, 2 (weights 10, 10)
    # Wrong at indices 1, 3 (weights 1, 1)
    weights = np.array([10.0, 1.0, 10.0, 1.0])

    # Expected: (10*1 + 1*0 + 10*1 + 1*0) / (10 + 1 + 10 + 1) = 20/22
    expected = 20.0 / 22.0
    assert np.isclose(weighted_sign_accuracy(y_true, delta_pred, weights), expected)


# ============================================================================
# Convenience Function Tests
# ============================================================================

def test_classification_metrics_without_weights():
    """Test classification_metrics without weights."""
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0, 0, 1])

    metrics = classification_metrics(y_true, y_pred)

    assert 'accuracy' in metrics
    assert np.isclose(metrics['accuracy'], 4.0 / 6.0)
    assert 'weighted_accuracy' not in metrics
    assert 'expected_cost' not in metrics


def test_classification_metrics_with_weights():
    """Test classification_metrics with weights."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 0])
    weights = np.array([1.0, 2.0, 3.0, 4.0])

    metrics = classification_metrics(y_true, y_pred, weights)

    assert 'accuracy' in metrics
    assert 'weighted_accuracy' in metrics
    assert 'expected_cost' in metrics

    # Verify values
    assert np.isclose(metrics['accuracy'], 0.5)
    # Correct at 0, 2 with weights 1, 3 -> (1+3)/(1+2+3+4) = 4/10
    assert np.isclose(metrics['weighted_accuracy'], 4.0 / 10.0)
    # Wrong at 1, 3 with costs 2, 4 -> mean([0, 2, 0, 4]) = 6/4 = 1.5
    assert np.isclose(metrics['expected_cost'], 1.5)


def test_regression_metrics_without_weights():
    """Test regression_metrics without weights."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([-1.0, 1.0, 1.0, -1.0])
    delta_true = np.array([-1.0, -0.5, 0.5, 1.0])

    metrics = regression_metrics(y_true, y_pred, delta_true)

    assert 'mae' in metrics
    assert 'rmse' in metrics
    assert 'sign_accuracy' in metrics
    assert 'weighted_mae' not in metrics
    assert 'weighted_rmse' not in metrics
    assert 'weighted_sign_accuracy' not in metrics


def test_regression_metrics_with_weights():
    """Test regression_metrics with weights."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([-1.0, 1.0, 1.0, -1.0])
    delta_true = np.array([-1.0, -0.5, 0.5, 1.0])
    weights = np.array([1.0, 2.0, 3.0, 4.0])

    metrics = regression_metrics(y_true, y_pred, delta_true, weights)

    assert 'mae' in metrics
    assert 'rmse' in metrics
    assert 'sign_accuracy' in metrics
    assert 'weighted_mae' in metrics
    assert 'weighted_rmse' in metrics
    assert 'weighted_sign_accuracy' in metrics


def test_regression_metrics_delta_true_defaults_to_y_pred():
    """Test regression_metrics uses y_pred when delta_true is None."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([-1.0, 1.0, 1.0, -1.0])

    metrics = regression_metrics(y_true, y_pred, delta_true=None)

    # MAE and RMSE should be zero since we're comparing y_pred to itself
    assert np.isclose(metrics['mae'], 0.0)
    assert np.isclose(metrics['rmse'], 0.0)


# ============================================================================
# Edge Cases and Type Handling
# ============================================================================

def test_metrics_handle_lists():
    """Test that metrics correctly handle list inputs."""
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 0]

    result = accuracy(y_true, y_pred)
    assert np.isclose(result, 0.5)


def test_metrics_handle_mixed_int_float():
    """Test that metrics correctly handle mixed int/float inputs."""
    y_true = np.array([0, 1, 1, 0], dtype=int)
    y_pred = np.array([0.0, 1.0, 0.0, 0.0], dtype=float)

    result = accuracy(y_true, y_pred)
    assert np.isclose(result, 0.75)


def test_weighted_metrics_handle_inf_weights():
    """Test that weighted metrics return NaN for infinite weights."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 0])
    weights = np.array([1.0, 2.0, np.inf, 4.0])

    result = weighted_accuracy(y_true, y_pred, weights)
    assert np.isnan(result)


def test_single_example():
    """Test metrics with single example."""
    y_true = np.array([1])
    y_pred = np.array([1])

    assert np.isclose(accuracy(y_true, y_pred), 1.0)


def test_empty_arrays():
    """Test metrics with empty arrays."""
    y_true = np.array([])
    y_pred = np.array([])

    # Empty mean should be NaN
    result = accuracy(y_true, y_pred)
    assert np.isnan(result)


# ============================================================================
# Diagnostic Metrics Tests
# ============================================================================

def test_corr_delta_misclass_positive_correlation():
    """Test correlation when high delta examples are misclassified more."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 1, 1, 1, 0, 0])  # Wrong at indices 2,3,6,7
    abs_delta = np.array([1.0, 2.0, 5.0, 6.0, 1.0, 2.0, 5.0, 6.0])
    # Higher delta examples (5.0, 6.0) are all misclassified
    # Lower delta examples (1.0, 2.0) are all correct
    # Should have positive correlation

    corr = corr_delta_misclass(y_true, y_pred, abs_delta)
    assert corr > 0.0


def test_corr_delta_misclass_negative_correlation():
    """Test correlation when high delta examples are misclassified less.

    Note: This is expected behavior in cost-sensitive learning.
    High |Delta| = high annotator agreement = clearer signal = easier to classify.
    So we typically observe negative correlation between |Delta| and misclassification.
    """
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([1, 1, 0, 0, 0, 0, 1, 1])  # Wrong at indices 0,1,4,5
    abs_delta = np.array([1.0, 2.0, 5.0, 6.0, 1.0, 2.0, 5.0, 6.0])
    # Lower delta examples (1.0, 2.0) are all misclassified
    # Higher delta examples (5.0, 6.0) are all correct
    # Should have negative correlation (high agreement = easier)

    corr = corr_delta_misclass(y_true, y_pred, abs_delta)
    assert corr < 0.0


def test_corr_delta_misclass_zero_variance_delta():
    """Test correlation returns NaN when delta has zero variance."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 0])
    abs_delta = np.array([2.0, 2.0, 2.0, 2.0])  # All same

    corr = corr_delta_misclass(y_true, y_pred, abs_delta)
    assert np.isnan(corr)


def test_corr_delta_misclass_all_correct():
    """Test correlation returns NaN when all predictions are correct."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    abs_delta = np.array([1.0, 2.0, 3.0, 4.0])
    # Misclass has zero variance (all zeros)

    corr = corr_delta_misclass(y_true, y_pred, abs_delta)
    assert np.isnan(corr)


def test_corr_delta_misclass_all_wrong():
    """Test correlation returns NaN when all predictions are wrong."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([1, 1, 0, 0])
    abs_delta = np.array([1.0, 2.0, 3.0, 4.0])
    # Misclass has zero variance (all ones)

    corr = corr_delta_misclass(y_true, y_pred, abs_delta)
    assert np.isnan(corr)


def test_normalize_delta_default_target():
    """Test normalize_delta with default target (sum = n)."""
    delta = np.array([1.0, -2.0, 3.0, -4.0])
    # |delta| = [1, 2, 3, 4], sum = 10
    # Should scale by 4/10 = 0.4

    normalized = normalize_delta(delta)

    # Check sum(|normalized|) = len(delta) = 4
    assert np.isclose(np.abs(normalized).sum(), 4.0)
    # Check signs preserved
    assert np.all(np.sign(normalized) == np.sign(delta))
    # Check relative magnitudes preserved
    expected = delta * 0.4
    assert np.allclose(normalized, expected)


def test_normalize_delta_custom_target():
    """Test normalize_delta with custom target sum."""
    delta = np.array([1.0, -2.0, 3.0, -4.0])
    # |delta| = [1, 2, 3, 4], sum = 10
    # Should scale by 100/10 = 10

    normalized = normalize_delta(delta, target_sum=100.0)

    # Check sum(|normalized|) = 100
    assert np.isclose(np.abs(normalized).sum(), 100.0)
    # Check values
    expected = delta * 10.0
    assert np.allclose(normalized, expected)


def test_normalize_delta_zero_sum():
    """Test normalize_delta returns zeros when sum is zero."""
    delta = np.array([0.0, 0.0, 0.0])

    normalized = normalize_delta(delta)

    assert np.allclose(normalized, 0.0)


def test_normalize_delta_preserves_signs():
    """Test normalize_delta preserves signs of all values."""
    delta = np.array([-5.0, -3.0, 0.0, 2.0, 7.0])

    normalized = normalize_delta(delta)

    # Check signs match
    assert np.all(np.sign(normalized) == np.sign(delta))
    # Zero should remain zero
    assert normalized[2] == 0.0


def test_normalize_delta_single_value():
    """Test normalize_delta with single value."""
    delta = np.array([5.0])

    normalized = normalize_delta(delta)

    # Sum should equal 1 (length of array)
    assert np.isclose(np.abs(normalized).sum(), 1.0)
    assert np.isclose(normalized[0], 1.0)
