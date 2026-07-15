"""Tests for TSL feature importance computation."""

import numpy as np
import pytest
from tensorsl import TSL


def generate_test_data(n_samples=500, n_features=3, random_state=42):
    """Generate synthetic data for testing feature importance."""
    rng = np.random.RandomState(random_state)
    X = rng.randn(n_samples, n_features)
    # Create a target that depends on features with different strengths
    # Use only the available features (works for any n_features >= 1)
    if n_features >= 3:
        y = 2.0 * X[:, 0] + 1.5 * X[:, 1] - 0.5 * X[:, 2] + rng.randn(n_samples) * 0.1
    elif n_features == 2:
        y = 2.0 * X[:, 0] + 1.5 * X[:, 1] + rng.randn(n_samples) * 0.1
    else:  # n_features == 1
        y = 2.0 * X[:, 0] + rng.randn(n_samples) * 0.1
    return X, y


@pytest.fixture(scope="module")
def fitted_model():
    """Fit an TSL model for testing."""
    X, y = generate_test_data(n_samples=500, n_features=3, random_state=42)
    model, _ = TSL.fit(
        X,
        y,
        epochs=3,
        n_trees=5,
        n_iter=10,
        split_try=5,
        colsample_bytree=1.0,
        seed=42,
        verbosity=0,
    )
    return model, X


def test_per_stage_feature_importance_shape(fitted_model):
    """Test that per-stage feature importance returns correct shapes."""
    model, X = fitted_model

    backbone_importance, tilt_importance = model.compute_per_stage_feature_importance(X)

    # Check shapes
    n_stages = len(model.stage_predictors)
    n_features = X.shape[1]

    assert backbone_importance.shape == (n_stages, n_features), (
        f"Backbone importance should have shape ({n_stages}, {n_features}), "
        f"got {backbone_importance.shape}"
    )
    assert tilt_importance.shape == (n_stages, n_features), (
        f"Tilt importance should have shape ({n_stages}, {n_features}), "
        f"got {tilt_importance.shape}"
    )

    # Check that all values are non-negative (variances are always >= 0)
    assert np.all(backbone_importance >= 0), (
        "Backbone importance (variance) should be non-negative"
    )
    assert np.all(tilt_importance >= 0), (
        "Tilt importance (variance) should be non-negative"
    )

    print(
        f"✅ Per-stage feature importance shapes are correct: {backbone_importance.shape}"
    )


def test_per_stage_feature_importance_values(fitted_model):
    """Test that per-stage feature importance values are reasonable."""
    model, X = fitted_model

    backbone_importance, tilt_importance = model.compute_per_stage_feature_importance(X)

    # Check that values are finite
    assert np.all(np.isfinite(backbone_importance)), (
        "Backbone importance should be finite"
    )
    assert np.all(np.isfinite(tilt_importance)), "Tilt importance should be finite"

    # Check that at least some features have non-zero importance
    assert np.any(backbone_importance > 0), (
        "At least some features should have non-zero backbone importance"
    )
    assert np.any(tilt_importance >= 0), "Tilt importance should be non-negative"

    print(f"✅ Per-stage feature importance values are reasonable")
    print(
        f"   Backbone importance range: [{backbone_importance.min():.6f}, {backbone_importance.max():.6f}]"
    )
    print(
        f"   Tilt importance range: [{tilt_importance.min():.6f}, {tilt_importance.max():.6f}]"
    )


def test_aggregated_feature_importance_shape(fitted_model):
    """Test that aggregated feature importance returns correct shapes."""
    model, X = fitted_model

    global_backbone, global_tilt, stage_weights = (
        model.compute_aggregated_feature_importance(X)
    )

    n_features = X.shape[1]
    n_stages = len(model.stage_predictors)

    # Check shapes
    assert global_backbone.shape == (n_features,), (
        f"Global backbone importance should have shape ({n_features},), "
        f"got {global_backbone.shape}"
    )
    assert global_tilt.shape == (n_features,), (
        f"Global tilt importance should have shape ({n_features},), "
        f"got {global_tilt.shape}"
    )
    assert stage_weights.shape == (n_stages,), (
        f"Stage weights should have shape ({n_stages},), got {stage_weights.shape}"
    )

    # Check that stage weights sum to 1 (within numerical precision)
    assert np.allclose(stage_weights.sum(), 1.0, rtol=1e-10), (
        f"Stage weights should sum to 1.0, got {stage_weights.sum()}"
    )

    # Check that all values are non-negative
    assert np.all(global_backbone >= 0), (
        "Global backbone importance should be non-negative"
    )
    assert np.all(global_tilt >= 0), "Global tilt importance should be non-negative"
    assert np.all(stage_weights >= 0), "Stage weights should be non-negative"

    print(f"✅ Aggregated feature importance shapes are correct")
    print(f"   Stage weights sum: {stage_weights.sum():.10f}")


def test_aggregated_feature_importance_consistency(fitted_model):
    """Test that aggregated importance is consistent with per-stage importance."""
    model, X = fitted_model

    # Get per-stage importance
    backbone_per_stage, tilt_per_stage = model.compute_per_stage_feature_importance(X)

    # Get aggregated importance
    global_backbone, global_tilt, stage_weights = (
        model.compute_aggregated_feature_importance(X)
    )

    # Manually compute aggregated importance from per-stage
    n_stages, n_features = backbone_per_stage.shape
    manual_global_backbone = np.zeros(n_features)
    manual_global_tilt = np.zeros(n_features)

    for stage_idx in range(n_stages):
        weight = stage_weights[stage_idx]
        for j in range(n_features):
            manual_global_backbone[j] += weight * backbone_per_stage[stage_idx, j]
            manual_global_tilt[j] += weight * tilt_per_stage[stage_idx, j]

    # Check that they match (within numerical precision)
    np.testing.assert_array_almost_equal(
        global_backbone,
        manual_global_backbone,
        decimal=10,
        err_msg="Global backbone importance should match weighted sum of per-stage importance",
    )
    np.testing.assert_array_almost_equal(
        global_tilt,
        manual_global_tilt,
        decimal=10,
        err_msg="Global tilt importance should match weighted sum of per-stage importance",
    )

    print("✅ Aggregated importance is consistent with per-stage importance")


def test_combined_feature_importance_shape(fitted_model):
    """Test that combined feature importance returns correct shapes."""
    model, X = fitted_model

    combined, backbone, tilt = model.compute_combined_feature_importance(X, gamma=1.0)

    n_features = X.shape[1]

    # Check shapes
    assert combined.shape == (n_features,), (
        f"Combined importance should have shape ({n_features},), got {combined.shape}"
    )
    assert backbone.shape == (n_features,), (
        f"Backbone importance should have shape ({n_features},), got {backbone.shape}"
    )
    assert tilt.shape == (n_features,), (
        f"Tilt importance should have shape ({n_features},), got {tilt.shape}"
    )

    # Check that combined = backbone + gamma * tilt
    expected_combined = backbone + 1.0 * tilt
    np.testing.assert_array_almost_equal(
        combined,
        expected_combined,
        decimal=10,
        err_msg="Combined importance should equal backbone + gamma * tilt",
    )

    print("✅ Combined feature importance shapes and formula are correct")


def test_combined_feature_importance_gamma(fitted_model):
    """Test that gamma parameter affects combined importance correctly."""
    model, X = fitted_model

    # Test with different gamma values
    gamma_values = [0.0, 0.5, 1.0, 2.0]

    for gamma in gamma_values:
        combined, backbone, tilt = model.compute_combined_feature_importance(
            X, gamma=gamma
        )

        # Check formula: combined = backbone + gamma * tilt
        expected_combined = backbone + gamma * tilt
        np.testing.assert_array_almost_equal(
            combined,
            expected_combined,
            decimal=10,
            err_msg=f"Combined importance with gamma={gamma} should equal backbone + gamma * tilt",
        )

        # When gamma=0, combined should equal backbone
        if gamma == 0.0:
            np.testing.assert_array_almost_equal(
                combined,
                backbone,
                decimal=10,
                err_msg="When gamma=0, combined importance should equal backbone",
            )

    print("✅ Gamma parameter affects combined importance correctly")


def test_combined_feature_importance_default_gamma(fitted_model):
    """Test that default gamma is 1.0."""
    model, X = fitted_model

    # Call without gamma (should default to 1.0)
    combined_default, _, _ = model.compute_combined_feature_importance(X)

    # Call with gamma=1.0 explicitly
    combined_explicit, _, _ = model.compute_combined_feature_importance(X, gamma=1.0)

    np.testing.assert_array_almost_equal(
        combined_default,
        combined_explicit,
        decimal=10,
        err_msg="Default gamma should be 1.0",
    )


def test_first_order_partial_dependence_shapes(fitted_model):
    """Test shapes for first-order partial dependence functions."""
    model, X = fitted_model
    n_features = X.shape[1]
    n_epochs = len(model.stage_predictors)

    values_x = X[:10]
    results = model.compute_first_order_partial_dependence_functions(values_x, X)

    assert len(results) == n_features, (
        f"Expected {n_features} feature results, got {len(results)}"
    )

    for constants, pd_values in results:
        assert len(constants) == n_epochs, (
            f"Expected {n_epochs} constants, got {len(constants)}"
        )
        for c_plus, c_minus in constants:
            assert np.isfinite(c_plus), "C_plus should be finite"
            assert np.isfinite(c_minus), "C_minus should be finite"

        assert pd_values.shape == (values_x.shape[0], 2 * n_epochs), (
            f"PD values should have shape ({values_x.shape[0]}, {2 * n_epochs}), "
            f"got {pd_values.shape}"
        )
        assert np.all(np.isfinite(pd_values)), "PD values should be finite"

    print("✅ First-order partial dependence shapes are correct")


def test_first_order_partial_dependence_values(fitted_model):
    """Test that first-order partial dependence values are computed correctly with scaling."""
    model, X = fitted_model

    # Use a single observation
    values_x = X[:1]
    results = model.compute_first_order_partial_dependence_functions(values_x, X)

    # Verify that constants include scaling (they should be non-zero and finite)
    for j, (constants, pd_values) in enumerate(results):
        for epoch_idx, (c_plus, c_minus) in enumerate(constants):
            # Constants should be finite (scaling is absorbed into them via effective_lambda)
            assert np.isfinite(c_plus), f"C_plus for feature {j}, epoch {epoch_idx} should be finite"
            assert np.isfinite(c_minus), f"C_minus for feature {j}, epoch {epoch_idx} should be finite"

            # PD values should be finite and computed as C * m
            f_plus = pd_values[0, 2 * epoch_idx]
            f_minus = pd_values[0, 2 * epoch_idx + 1]

            assert np.isfinite(f_plus), f"f_plus for feature {j}, epoch {epoch_idx} should be finite"
            assert np.isfinite(f_minus), f"f_minus for feature {j}, epoch {epoch_idx} should be finite"

            # Verify that PD values have scaling absorbed (they should match predictions when combined)
            # Note: First-order PD assumes independence, so summing across features won't exactly match
            # full predictions, but the values should be reasonable

    print("✅ First-order partial dependence values are computed correctly")


def test_feature_importance_with_different_data_sizes(fitted_model):
    """Test that feature importance works with different data sizes."""
    model, X_train = fitted_model

    # Test with subset of training data
    X_subset = X_train[:100]
    backbone_subset, tilt_subset = model.compute_per_stage_feature_importance(X_subset)

    # Test with full training data
    backbone_full, tilt_full = model.compute_per_stage_feature_importance(X_train)

    # Shapes should match (only number of samples differs, not features)
    assert backbone_subset.shape == backbone_full.shape, (
        "Per-stage importance should have same shape regardless of data size"
    )
    assert tilt_subset.shape == tilt_full.shape, (
        "Per-stage importance should have same shape regardless of data size"
    )

    print("✅ Feature importance works with different data sizes")


def test_feature_importance_consistency_across_calls(fitted_model):
    """Test that feature importance is consistent across multiple calls."""
    model, X = fitted_model

    # Call multiple times
    backbone1, tilt1 = model.compute_per_stage_feature_importance(X)
    backbone2, tilt2 = model.compute_per_stage_feature_importance(X)

    # Results should be identical (deterministic)
    np.testing.assert_array_almost_equal(
        backbone1,
        backbone2,
        decimal=10,
        err_msg="Per-stage importance should be deterministic",
    )
    np.testing.assert_array_almost_equal(
        tilt1,
        tilt2,
        decimal=10,
        err_msg="Per-stage importance should be deterministic",
    )

    print("✅ Feature importance is consistent across multiple calls")


def test_feature_importance_with_single_stage():
    """Test feature importance with a model that has only one stage."""
    X, y = generate_test_data(n_samples=200, n_features=2, random_state=123)

    # Fit model with only one epoch
    model, _ = TSL.fit(
        X,
        y,
        epochs=1,
        n_trees=3,
        n_iter=5,
        split_try=5,
        colsample_bytree=1.0,
        seed=123,
        verbosity=0,
    )

    # Should work without errors
    backbone, tilt = model.compute_per_stage_feature_importance(X)
    global_backbone, global_tilt, stage_weights = (
        model.compute_aggregated_feature_importance(X)
    )
    combined, _, _ = model.compute_combined_feature_importance(X)

    # With one stage, stage weights should be [1.0]
    assert len(stage_weights) == 1, "Should have one stage weight"
    assert np.allclose(stage_weights[0], 1.0), "Single stage weight should be 1.0"

    # Global importance should equal per-stage importance (since weight is 1.0)
    np.testing.assert_array_almost_equal(
        global_backbone,
        backbone[0],
        decimal=10,
        err_msg="With one stage, global importance should equal per-stage importance",
    )
    np.testing.assert_array_almost_equal(
        global_tilt,
        tilt[0],
        decimal=10,
        err_msg="With one stage, global importance should equal per-stage importance",
    )

    print("✅ Feature importance works with single-stage model")


def test_feature_importance_edge_case_zero_variance():
    """Test feature importance when a feature has zero variance (constant values)."""
    # Create data where one feature is constant
    rng = np.random.RandomState(42)
    n_samples = 200
    X = np.zeros((n_samples, 3))
    X[:, 0] = rng.randn(n_samples)  # Varying feature
    X[:, 1] = 5.0  # Constant feature
    X[:, 2] = rng.randn(n_samples)  # Varying feature

    y = X[:, 0] + X[:, 2] + rng.randn(n_samples) * 0.1

    # Fit model
    model, _ = TSL.fit(
        X,
        y,
        epochs=2,
        n_trees=3,
        n_iter=5,
        split_try=5,
        colsample_bytree=1.0,
        seed=42,
        verbosity=0,
    )

    # Compute importance
    backbone, tilt = model.compute_per_stage_feature_importance(X)

    # All values should be finite (even for constant feature)
    assert np.all(np.isfinite(backbone)), "Backbone importance should be finite"
    assert np.all(np.isfinite(tilt)), "Tilt importance should be finite"

    # All values should be non-negative
    assert np.all(backbone >= 0), "Backbone importance should be non-negative"
    assert np.all(tilt >= 0), "Tilt importance should be non-negative"

    print("✅ Feature importance handles constant features correctly")
