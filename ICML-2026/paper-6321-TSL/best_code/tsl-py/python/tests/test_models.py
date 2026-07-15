import numpy as np
import pytest
from tensorsl import TSL, GridTensor


def gen_data(n=5000, seed=1):
    np.random.seed(seed)
    X = np.random.uniform(-5, 5, size=(n, 2))
    # y = 3*np.sin(3 * X[:,0])*np.cos(5*X[:,1]) + np.random.normal(scale=0.5, size=n)
    y = (
        np.exp(np.sin(X[:, 0]) * np.cos(X[:, 1]))
        + X[:, 0]
        + np.random.normal(scale=0.5, size=n)
    )
    return X, y


@pytest.fixture(scope="module")
def training_data():
    X, y = gen_data(seed=1)
    return X, y.ravel()


@pytest.fixture(scope="module")
def test_data():
    X_test, y_test = gen_data(seed=2)
    return X_test, y_test.ravel()


def test_grid_tensor_fit(training_data, test_data):
    X, y = training_data
    X_test, y_test = test_data

    # Train the TSL estimator
    tg, fr = GridTensor.fit(X, y, n_iter=100, split_try=15, colsample_bytree=1.0)

    print("Fit result: ", fr)
    # TSL predictions and loss
    y_pred = tg.predict(X_test)
    tg_test_loss = np.mean((y_test - y_pred) ** 2)

    # Baseline: loss of predicting the mean of y_test
    baseline = np.mean(y_test)
    mean_test_loss = np.mean((y_test - baseline) ** 2)

    # Print losses for debugging (optional)
    print(f"Tree grid test loss: {tg_test_loss}")
    print(f"Mean test loss: {mean_test_loss}")

    print(f"Tree grid scaling: {tg.scaling}")

    assert tg_test_loss < mean_test_loss, "Tree grid should beat the mean predictor"


def test_tsl_boosted_fit(training_data, test_data):
    X, y = training_data
    X_test, y_test = test_data

    # Train the TSL estimator
    tsl, fr = TSL.fit(
        X, y, epochs=3, n_trees=37, n_iter=30, split_try=16, colsample_bytree=1.0
    )

    print("Fit result: ", fr)
    # TSL predictions and loss
    y_pred = tsl.predict(X_test)
    mpf_test_loss = np.mean((y_test - y_pred) ** 2)

    # Baseline: loss of predicting the mean of y_test
    baseline = np.mean(y_test)
    mean_test_loss = np.mean((y_test - baseline) ** 2)

    # Print losses for debugging (optional)
    print(f"TSL test loss: {mpf_test_loss}")
    print(f"Mean test loss: {mean_test_loss}")

    assert mpf_test_loss < mean_test_loss, "TSL should beat the mean predictor"


def _non_contiguous_column_slice(X):
    """Return a non-contiguous view (row stride 2) holding exactly ``X``'s values."""
    n, p = X.shape
    wide = np.empty((n, 2 * p), dtype=np.float64)
    wide[:, ::2] = X
    wide[:, 1::2] = np.nan  # junk columns that get sliced away
    sliced = wide[:, ::2]
    assert not sliced.flags["C_CONTIGUOUS"]
    return sliced


def test_tsl_predict_is_layout_invariant(training_data, test_data):
    """predict() must return identical results for C-contiguous, Fortran-ordered,
    and column-sliced (non-contiguous) inputs holding the same values."""
    X, y = training_data
    X_test, _ = test_data
    X_test = np.ascontiguousarray(X_test)

    tsl, _ = TSL.fit(
        X, y, epochs=3, n_trees=37, n_iter=30, split_try=16, colsample_bytree=1.0,
        verbosity=0,
    )
    baseline = tsl.predict(X_test)

    # Fortran-ordered (column-major) copy of the same data.
    X_f = np.asfortranarray(X_test)
    assert not X_f.flags["C_CONTIGUOUS"] and X_f.flags["F_CONTIGUOUS"]
    np.testing.assert_array_almost_equal(
        tsl.predict(X_f), baseline, decimal=12,
        err_msg="predict on Fortran-ordered input must match C-contiguous",
    )

    # Genuinely non-contiguous (column-sliced) view.
    np.testing.assert_array_almost_equal(
        tsl.predict(_non_contiguous_column_slice(X_test)), baseline, decimal=12,
        err_msg="predict on column-sliced (non-contiguous) input must match",
    )

    # Row subset (advanced indexing yields a C-contiguous copy) — sanity check.
    idx = np.sort(
        np.random.default_rng(0).choice(
            X_test.shape[0], X_test.shape[0] // 2, replace=False
        )
    )
    np.testing.assert_array_almost_equal(
        tsl.predict(X_test[idx]), baseline[idx], decimal=12,
        err_msg="predict on a row subset must match the corresponding full predictions",
    )


def test_tsl_fit_is_layout_invariant(training_data):
    """fit() produces identical results regardless of input memory layout."""
    X, y = training_data
    X = np.ascontiguousarray(X)

    def fit_then_predict(X_in):
        model, _ = TSL.fit(
            X_in, y, epochs=3, n_trees=8, n_iter=25, split_try=12,
            colsample_bytree=1.0, seed=7, verbosity=0,
        )
        return model.predict(X)

    baseline = fit_then_predict(X)

    np.testing.assert_array_almost_equal(
        fit_then_predict(np.asfortranarray(X)), baseline, decimal=10,
        err_msg="fit on Fortran-ordered input must match C-contiguous",
    )
    np.testing.assert_array_almost_equal(
        fit_then_predict(_non_contiguous_column_slice(X)), baseline, decimal=10,
        err_msg="fit on column-sliced (non-contiguous) input must match",
    )
