"""Smoke tests for tensorsl.plot — verify each public function runs end-to-end
on a small fitted model and returns sensible array shapes.
"""

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensorsl import TSL  # noqa: E402
from tensorsl.plot import (  # noqa: E402
    compute_local_explanation,
    pd_difference_plot,
    plot_2d_backbone,
    plot_2d_pd,
    plot_feature_importance,
    plot_first_order_pd,
    plot_ice,
    plot_tilt_diagnostics,
)


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.RandomState(0)
    n, p = 400, 4
    X = rng.randn(n, p)
    y = 1.5 * X[:, 0] - 0.8 * X[:, 1] + 0.5 * X[:, 0] * X[:, 1] + 0.1 * rng.randn(n)
    model, _ = TSL.fit(
        X, y,
        epochs=3,
        n_trees=4,
        n_iter=8,
        split_try=4,
        colsample_bytree=1.0,
        seed=0,
        verbosity=0,
    )
    return model, X, ["a", "b", "c", "d"]


@pytest.fixture(scope="module")
def fitted_2stage():
    """Two-stage fit for the component-scale round-trip checks. Component
    normalization is defined where each stage's OLS scaling stays non-negative,
    which holds while the model is not staged past this toy signal's effective
    rank; a third stage here lands on a near-collinear system whose scaling can
    turn negative."""
    rng = np.random.RandomState(0)
    n, p = 400, 4
    X = rng.randn(n, p)
    y = 1.5 * X[:, 0] - 0.8 * X[:, 1] + 0.5 * X[:, 0] * X[:, 1] + 0.1 * rng.randn(n)
    model, _ = TSL.fit(
        X, y,
        epochs=2,
        n_trees=4,
        n_iter=8,
        split_try=4,
        colsample_bytree=1.0,
        seed=0,
        verbosity=0,
    )
    return model, X, ["a", "b", "c", "d"]


def test_plot_first_order_pd(fitted):
    model, X, names = fitted
    res = plot_first_order_pd(model, X, feature_names=names, grid_points=30)
    assert res.f_plus.shape == (4, 30, 3)
    assert res.f_minus.shape == (4, 30, 3)
    assert res.axes.shape == (3, 4)
    plt.close(res.fig)


def test_pd_difference_plot_subset(fitted):
    model, X, names = fitted
    res = pd_difference_plot(
        model, X, features=["a", 1], feature_names=names, grid_points=25, stages=[0],
    )
    assert res.feature_indices == [0, 1]
    assert res.f_plus.shape == (2, 25, 3)
    assert res.axes.shape == (1, 2)
    assert res.pd_scale == "raw"
    assert res.normalized is None
    plt.close(res.fig)


def test_pd_difference_plot_raw_unchanged(fitted):
    """`pd_scale='raw'` must match the default behavior bit-for-bit."""
    model, X, names = fitted
    r_default = pd_difference_plot(model, X, feature_names=names, grid_points=30)
    r_raw = pd_difference_plot(
        model, X, feature_names=names, grid_points=30, pd_scale="raw"
    )
    np.testing.assert_array_equal(r_default.f_plus, r_raw.f_plus)
    np.testing.assert_array_equal(r_default.f_minus, r_raw.f_minus)
    np.testing.assert_array_equal(r_default.constants, r_raw.constants)
    assert r_raw.normalized is None
    plt.close(r_default.fig)
    plt.close(r_raw.fig)


def test_pd_difference_plot_component(fitted_2stage):
    """Verify component normalization properties on a small fitted model."""
    model, X, names = fitted_2stage
    res = pd_difference_plot(
        model, X, feature_names=names, grid_points=30, pd_scale="component"
    )
    assert res.pd_scale == "component"
    assert res.normalized is not None
    diag = res.normalized
    n_features, n_grid, n_stages = res.f_plus.shape
    assert diag.m_plus.shape == (n_features, n_grid, n_stages)
    assert diag.m_minus.shape == (n_features, n_grid, n_stages)
    assert diag.backbone.shape == (n_features, n_grid, n_stages)
    assert diag.tilt.shape == (n_features, n_grid, n_stages)

    # Round-trip: m_plus * C_plus ≈ f_plus, m_minus * C_minus ≈ -f_minus.
    C_plus = res.constants[:, :, 0][:, None, :]
    C_minus = -res.constants[:, :, 1][:, None, :]
    np.testing.assert_allclose(diag.m_plus * C_plus, res.f_plus, atol=1e-10)
    np.testing.assert_allclose(diag.m_minus * C_minus, -res.f_minus, atol=1e-10)

    # sign(m_+ - m_-) == sign(d_j) where d_j is the intrinsic stage tilt.
    # Use the helper that powers the backbone overlay.
    from tensorsl.plot._common import _stage_backbone_tilt

    n_stages = res.f_plus.shape[2]
    for j, feat_idx in enumerate(res.feature_indices):
        x_grid = res.x_grids[j]
        for s in range(n_stages):
            sp = model.stage_predictors[s]
            _, d_j = _stage_backbone_tilt(sp, feat_idx, x_grid)
            diff = diag.m_plus[j, :, s] - diag.m_minus[j, :, s]
            nz = np.abs(d_j) > 1e-9
            assert np.array_equal(np.sign(diff[nz]), np.sign(d_j[nz])), (
                f"sign mismatch for feature {feat_idx}, stage {s}"
            )

    # Constants must be feature- and stage-specific, not collapsed.
    assert res.constants.shape == (n_features, n_stages, 2)
    assert np.unique(res.constants[:, :, 0]).size > 1
    plt.close(res.fig)


def test_plot_first_order_pd_component(fitted):
    model, X, names = fitted
    res = plot_first_order_pd(
        model, X, feature_names=names, grid_points=30, pd_scale="component"
    )
    assert res.pd_scale == "component"
    assert res.normalized is not None
    plt.close(res.fig)


def test_plot_2d_pd_surface(fitted):
    model, X, names = fitted
    res = plot_2d_pd(model, X, feature_x=0, feature_y=1, feature_names=names, grid_points=20)
    assert res.pd_total.shape == (20, 20)
    assert res.pd_per_stage.shape[0] == 3
    plt.close(res.fig)


def test_plot_2d_pd_lines(fitted):
    model, X, names = fitted
    res = plot_2d_pd(
        model, X, feature_x="a", feature_y="b", feature_names=names,
        grid_points=20, kind="lines", y_values=[-1.0, 0.0, 1.0],
    )
    assert res.pd_per_stage.shape == (3, 3, 20)
    # one card per stage plus the appended "Total" card
    assert res.axes.shape == (len(model.stage_predictors) + 1,)
    plt.close(res.fig)


def test_plot_2d_pd_lines_no_total(fitted):
    model, X, names = fitted
    res = plot_2d_pd(
        model, X, feature_x="a", feature_y="b", feature_names=names,
        grid_points=20, kind="lines", y_values=[-1.0, 0.0, 1.0],
        show_total=False,
    )
    # the per-stage cards alone, no "Total" card
    assert res.axes.shape == (len(model.stage_predictors),)
    plt.close(res.fig)


def test_plot_2d_backbone(fitted):
    model, X, names = fitted
    res = plot_2d_backbone(model, X, "a", "b", feature_names=names, grid_points=15)
    assert res.backbone_per_stage.shape == (3, 15, 15)
    assert res.pd_per_stage.shape == (3, 15, 15)
    assert res.X.shape == res.Y.shape == (15, 15)
    plt.close(res.fig)


def test_plot_2d_backbone_data_only(fitted):
    model, X, names = fitted
    res = plot_2d_backbone(
        model, X, "a", "b", feature_names=names, grid_points=12, return_data_only=True,
    )
    assert res.fig is None and res.axes is None
    assert res.backbone_per_stage.shape == (3, 12, 12)


def test_plot_ice(fitted):
    model, X, names = fitted
    res = plot_ice(model, X, "a", feature_names=names, n_ice=10, grid_points=20, seed=1)
    assert res.ice.shape == (10, 20)
    assert res.pd.shape == (20,)
    plt.close(res.fig)


def test_compute_local_explanation(fitted):
    model, X, _ = fitted
    expl = compute_local_explanation(model, X[0])
    assert expl.stage_contributions.shape == (3,)
    # total should match predict
    pred = float(model.predict(X[0:1])[0])
    assert np.isclose(expl.total_prediction, pred, rtol=1e-6, atol=1e-8)


def test_plot_tilt_diagnostics(fitted):
    model, X, names = fitted
    res = plot_tilt_diagnostics(
        model, X, features=["a", "b"], feature_names=names, grid_points=25,
    )
    n_f, n_grid, n_s = 2, 25, 3
    assert res.axes.shape == (n_f * n_s, 4)
    assert res.B.shape == (n_f, n_grid, n_s)
    assert res.d.shape == (n_f, n_grid, n_s)
    assert res.d_centered.shape == (n_f, n_grid, n_s)
    assert res.curves.shape == (n_f, n_grid, n_s, 4)
    assert res.feature_indices == [0, 1]
    assert res.stages == [0, 1, 2]

    # Centered tilt mean per (feature, stage) should be ~0 along the grid.
    np.testing.assert_allclose(
        res.d_centered.mean(axis=1), 0.0, atol=1e-10,
    )

    # Curve identities: column 0 = tanh(d), column 1 = B * tanh(d).
    np.testing.assert_allclose(res.curves[..., 0], np.tanh(res.d), atol=1e-12)
    np.testing.assert_allclose(
        res.curves[..., 1], res.B * np.tanh(res.d), atol=1e-12,
    )
    np.testing.assert_allclose(
        res.curves[..., 2], np.tanh(res.d_centered), atol=1e-12,
    )
    np.testing.assert_allclose(
        res.curves[..., 3], res.B * np.tanh(res.d_centered), atol=1e-12,
    )

    # Pure-tanh values are within [-1, 1].
    assert np.all(np.abs(res.curves[..., 0]) <= 1.0 + 1e-12)
    assert np.all(np.abs(res.curves[..., 2]) <= 1.0 + 1e-12)
    plt.close(res.fig)


def test_plot_tilt_diagnostics_stage1_positive_only():
    """When y is non-negative, TSL's stage-1 positive-only invariant forces
    λ_- = 0 and d_j ≡ 0. The diagnostic must reflect that exactly, with no
    log(1/eps) offset from the m_+/m_- regularization path.
    """
    rng = np.random.RandomState(1)
    n, p = 300, 3
    X = rng.randn(n, p)
    y = np.abs(0.7 * X[:, 0] + 0.4 * X[:, 1]) + 0.1 * np.abs(rng.randn(n))
    model, _ = TSL.fit(
        X, y, epochs=3, n_trees=4, n_iter=8, split_try=4,
        colsample_bytree=1.0, seed=0, verbosity=0,
    )
    res = plot_tilt_diagnostics(
        model, X, feature_names=["a", "b", "c"], grid_points=20,
    )
    # Stage 1 (index 0): exact zero tilt and zero centered tilt.
    np.testing.assert_array_equal(res.d[:, :, 0], 0.0)
    np.testing.assert_array_equal(res.d_centered[:, :, 0], 0.0)
    # All four curves in stage 1 are zero too.
    np.testing.assert_array_equal(res.curves[:, :, 0, :], 0.0)
    plt.close(res.fig)


def test_plot_tilt_diagnostics_stage_subset(fitted):
    model, X, names = fitted
    res = plot_tilt_diagnostics(
        model, X, features=["a"], feature_names=names, grid_points=20, stages=[1],
    )
    assert res.stages == [1]
    assert res.axes.shape == (1, 4)
    assert res.B.shape == (1, 20, 1)
    assert res.curves.shape == (1, 20, 1, 4)
    plt.close(res.fig)


def test_plot_feature_importance(fitted):
    model, X, names = fitted
    res = plot_feature_importance(model, X, feature_names=names, gamma=1.0)
    assert res.backbone_per_stage.shape == (3, 4)
    assert res.combined.shape == (4,)
    assert res.stage_weights.shape == (3,)
    plt.close(res.fig)
