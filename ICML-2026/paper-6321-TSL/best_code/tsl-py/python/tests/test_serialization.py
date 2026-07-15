"""Tests for TSL model serialization (save/load functionality)."""

import os
import tempfile

import numpy as np
import pytest
from tensorsl import TSL
from tensorsl.sklearn import TSLRegressor


def generate_test_data(n_samples=100, n_features=2, random_state=4):
    """Generate synthetic data for testing."""
    rng = np.random.RandomState(random_state)
    X = rng.randn(n_samples, n_features)
    y = np.sum(X**2, axis=1) + rng.randn(n_samples) * 0.1
    return X, y


def test_tsl_save_and_load():
    """Test saving and loading an TSL model (binary format)."""
    X, y = generate_test_data(n_features=1)

    # Fit a model
    original_model, _ = TSL.fit(
        X, y, epochs=1, n_trees=1, n_iter=5, split_try=5, colsample_bytree=1.0, seed=2
    )

    # Get predictions from original model
    original_predictions = original_model.predict(X)

    # Save to target/model.bin (binary format preserves exact floating point values)
    model_path = "target/model.bin"

    # Ensure target directory exists
    os.makedirs("target", exist_ok=True)

    # Save the model
    original_model.save(model_path)
    print(f"Saved model to {model_path}")

    # Verify file was created
    assert os.path.exists(model_path), "Model file should be created"

    # Load the model
    loaded_model = TSL.load(model_path)

    # Get predictions from loaded model
    loaded_predictions = loaded_model.predict(X)

    # Debug: Print arrays and differences
    diff = original_predictions - loaded_predictions
    diff_abs = np.abs(diff)
    max_diff_idx = np.argmax(diff_abs)
    max_diff = diff_abs[max_diff_idx]

    print("\n=== Prediction Comparison ===")
    print(f"Total predictions: {len(original_predictions)}")
    print(f"Max absolute difference: {max_diff}")
    print(f"Max difference index: {max_diff_idx}")
    print(f"Original[{max_diff_idx}]: {original_predictions[max_diff_idx]}")
    print(f"Loaded[{max_diff_idx}]: {loaded_predictions[max_diff_idx]}")
    print(f"Difference: {diff[max_diff_idx]}")
    print(f"\nNumber of non-zero differences: {np.count_nonzero(diff_abs > 1e-10)}")
    print(f"\nOriginal predictions:\n{original_predictions}")
    print(f"\nLoaded predictions:\n{loaded_predictions}")
    print(f"\nDifferences:\n{diff}")

    # Verify predictions are identical (binary format should preserve exact values)
    np.testing.assert_array_almost_equal(
        original_predictions, loaded_predictions, decimal=15
    )


def test_tsl_save_and_load_with_different_data():
    """Test that loaded model works with different data."""
    X_train, y_train = generate_test_data(n_samples=100, random_state=42)
    X_test, y_test = generate_test_data(n_samples=50, random_state=123)

    # Fit a model
    model, _ = TSL.fit(
        X_train,
        y_train,
        epochs=2,
        n_trees=3,
        n_iter=5,
        split_try=5,
        colsample_bytree=1.0,
        seed=42,
    )

    # Get predictions on test data
    original_predictions = model.predict(X_test)

    # Save and load (binary format)
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".bin", delete=False) as f:
        temp_path = f.name

    try:
        model.save(temp_path)
        loaded_model = TSL.load(temp_path)

        # Get predictions from loaded model on test data
        loaded_predictions = loaded_model.predict(X_test)

        # Verify predictions are identical
        np.testing.assert_array_almost_equal(
            original_predictions, loaded_predictions, decimal=15
        )

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_mpfregressor_save_and_load():
    """Test saving and loading an TSLRegressor model."""
    X, y = generate_test_data()

    # Fit a model
    original_regressor = TSLRegressor(epochs=2, n_trees=3, n_iter=5, seed=42)
    original_regressor.fit(X, y)

    # Get predictions from original model
    original_predictions = original_regressor.predict(X)

    # Save to a temporary file (binary format)
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".bin", delete=False) as f:
        temp_path = f.name

    try:
        # Save the model
        original_regressor.save(temp_path)

        # Verify file was created
        assert os.path.exists(temp_path), "Model file should be created"

        # Load the model
        loaded_regressor = TSLRegressor.load(temp_path)

        # Get predictions from loaded model
        loaded_predictions = loaded_regressor.predict(X)

        # Verify predictions are identical
        np.testing.assert_array_almost_equal(
            original_predictions, loaded_predictions, decimal=15
        )

    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_tsl_load_nonexistent_file():
    """Test that loading a nonexistent file raises an error."""
    with pytest.raises(Exception):  # Should raise IOError or similar
        TSL.load("/nonexistent/path/model.bin")


def test_tsl_save_invalid_path():
    """Test that saving to an invalid path raises an error."""
    X, y = generate_test_data()
    model, _ = TSL.fit(
        X, y, epochs=1, n_trees=1, n_iter=1, split_try=1, colsample_bytree=1.0, seed=42
    )

    # Try to save to a directory that doesn't exist
    with pytest.raises(Exception):  # Should raise IOError or similar
        model.save("/nonexistent/directory/model.bin")
