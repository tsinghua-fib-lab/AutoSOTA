"""Test marginal expectation computation."""

import numpy as np
from tensorsl import TSL


def test_marginal_expectation_equals_averaged_predictions():
    """
    Test that marginal expectation equals averaging over predictions.

    When we fix some features and marginalize over others, the marginal expectation
    should equal the average of predictions where we replace the marginalized features
    with values from the training data.
    """
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 5

    X = np.random.randn(n_samples, n_features)
    y = X[:, 0] + 2 * X[:, 1] - X[:, 2] + np.random.randn(n_samples) * 0.1

    # Fit a simple model
    model, _ = TSL.fit(
        X,
        y,
        epochs=3,
        n_trees=10,
        n_iter=5,
        split_try=5,
        colsample_bytree=0.8,
        seed=42,
        verbosity=0,
    )

    # Test: Fix first two features, marginalize over rest
    fixed_indices = [0, 1]
    fixed_values = np.array([[0.5, -0.3]])  # Single observation

    # Compute marginal expectation
    constants, marginal_exp = model.compute_partial_dependence_function(
        fixed_indices, fixed_values, X
    )

    # Extract f+ and f- separately (shape: (1, 2 * n_epochs))
    # Note: scaling is already absorbed into these values via effective_lambda
    f_plus = marginal_exp[0, ::2]  # Columns 0, 2, 4, ... (one per epoch)
    f_minus = marginal_exp[0, 1::2]  # Columns 1, 3, 5, ... (one per epoch)

    # Verify constants are returned correctly
    assert len(constants) == len(model.stage_predictors), (
        f"Expected {len(model.stage_predictors)} constants, got {len(constants)}"
    )
    for c_plus, c_minus in constants:
        assert np.isfinite(c_plus), "C_plus should be finite"
        assert np.isfinite(c_minus), "C_minus should be finite"

    # Combine f+ and f- to get marginal expectation
    # Since scaling is absorbed into lambda, we just add f+ and f- per epoch
    stage_predictors = model.stage_predictors
    n_epochs = len(stage_predictors)

    # Combine: f+ + f- (scaling already absorbed)
    combined_marginal_exp = 0.0
    for epoch_idx in range(n_epochs):
        # Final prediction per epoch = f+ + f- (scaling already in the values)
        combined_marginal_exp += f_plus[epoch_idx] + f_minus[epoch_idx]

    # Compute marginal expectation by averaging predictions
    # For each training sample, create a new sample with fixed features set
    # and marginalized features from that training sample, then predict
    marginalized_indices = [i for i in range(n_features) if i not in fixed_indices]
    averaged_predictions = []

    for train_idx in range(n_samples):
        # Create sample: fixed features from fixed_values, rest from training sample
        x_combined = np.zeros(n_features)
        x_combined[fixed_indices] = fixed_values[0]  # Set fixed features
        x_combined[marginalized_indices] = X[
            train_idx, marginalized_indices
        ]  # Set marginalized features

        # Predict on this combined sample (this sums predictions across all epochs)
        pred = model.predict(x_combined.reshape(1, -1))[0]
        averaged_predictions.append(pred)

    # Average the predictions
    mean_prediction = np.mean(averaged_predictions)

    # Verify they match (within numerical precision)
    assert np.allclose(combined_marginal_exp, mean_prediction, rtol=1e-10), (
        f"Marginal expectation from f+ and f- ({combined_marginal_exp:.10f}) should equal "
        f"averaged predictions ({mean_prediction:.10f})"
    )

    print("✅ Marginal expectation equals averaged predictions")
    print(f"   Marginal expectation (from f+ and f-): {combined_marginal_exp:.10f}")
    print(f"   Averaged predictions: {mean_prediction:.10f}")
    print(f"   Difference: {abs(combined_marginal_exp - mean_prediction):.2e}")


def test_ice_curves_average_equals_pd_function():
    """
    Test that averaging ICE curves equals the partial dependence function.

    For a single feature j, the PD function should equal the average of ICE curves
    over all observations: PD(x_j) = (1/n) * sum_i ICE_i(x_j)
    """
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 200  # Use fewer samples for ICE curves (faster)
    n_features = 5

    X = np.random.randn(n_samples, n_features)
    y = X[:, 0] + 2 * X[:, 1] - X[:, 2] + np.random.randn(n_samples) * 0.1

    # Fit a simple model
    model, _ = TSL.fit(
        X,
        y,
        epochs=3,
        n_trees=10,
        n_iter=5,
        split_try=5,
        colsample_bytree=0.8,
        seed=42,
        verbosity=0,
    )

    # Test with feature 0
    feature_index = 0
    n_epochs = len(model.stage_predictors)

    # Create a range of values for the feature
    feature_min = X[:, feature_index].min()
    feature_max = X[:, feature_index].max()
    n_range_values = 20
    x_range = np.linspace(feature_min, feature_max, n_range_values)

    # Compute ICE curves for all observations
    # Use a subset of observations for efficiency (ICE curves can be expensive)
    # IMPORTANT: Use the same subset for both ICE curves and PD function marginalization
    n_obs_for_ice = min(50, n_samples)  # Use up to 50 observations
    observations = X[:n_obs_for_ice]
    # Use the same subset as data_x for PD function to ensure they marginalize over the same set
    data_x_subset = X[:n_obs_for_ice]

    # Compute ICE curves (shape: n_obs, n_range_values, 2 * n_epochs)
    ice_values = model.compute_ice_curves(observations, feature_index, x_range, data_x_subset)

    # Verify shape
    assert ice_values.shape == (
        n_obs_for_ice,
        n_range_values,
        2 * n_epochs,
    ), f"Expected shape ({n_obs_for_ice}, {n_range_values}, {2 * n_epochs}), got {ice_values.shape}"

    # Average ICE curves over observations (shape: n_range_values, 2 * n_epochs)
    averaged_ice = np.mean(ice_values, axis=0)

    # Compute PD function for the same feature and range
    # PD function expects fixed_values to be (n_observations, n_fixed_features)
    # IMPORTANT: Use the same data_x_subset so PD marginalizes over the same observations as ICE
    fixed_indices = [feature_index]
    fixed_values = x_range.reshape(-1, 1)  # Shape: (n_range_values, 1)

    constants, pd_values = model.compute_partial_dependence_function(
        fixed_indices, fixed_values, data_x_subset
    )

    # Verify shapes
    assert pd_values.shape == (
        n_range_values,
        2 * n_epochs,
    ), f"Expected PD shape ({n_range_values}, {2 * n_epochs}), got {pd_values.shape}"

    # Compare averaged ICE curves to PD function
    # They should match for each epoch and each value in the range
    for epoch_idx in range(n_epochs):
        # Extract f+ and f- for this epoch
        ice_f_plus = averaged_ice[:, 2 * epoch_idx]
        ice_f_minus = averaged_ice[:, 2 * epoch_idx + 1]

        pd_f_plus = pd_values[:, 2 * epoch_idx]
        pd_f_minus = pd_values[:, 2 * epoch_idx + 1]

        # They should match (within numerical precision)
        assert np.allclose(
            ice_f_plus, pd_f_plus, rtol=1e-10, atol=1e-12
        ), (
            f"Epoch {epoch_idx} f+: Averaged ICE curves should equal PD function. "
            f"Max difference: {np.max(np.abs(ice_f_plus - pd_f_plus)):.2e}"
        )

        assert np.allclose(
            ice_f_minus, pd_f_minus, rtol=1e-10, atol=1e-12
        ), (
            f"Epoch {epoch_idx} f-: Averaged ICE curves should equal PD function. "
            f"Max difference: {np.max(np.abs(ice_f_minus - pd_f_minus)):.2e}"
        )

    print("✅ Averaged ICE curves equal PD function")
    print(f"   Tested {n_obs_for_ice} observations over {n_range_values} feature values")
    print(f"   All {n_epochs} epochs match within numerical precision")
    print(f"   Max difference f+: {np.max(np.abs(averaged_ice[:, ::2] - pd_values[:, ::2])):.2e}")
    print(f"   Max difference f-: {np.max(np.abs(averaged_ice[:, 1::2] - pd_values[:, 1::2])):.2e}")
