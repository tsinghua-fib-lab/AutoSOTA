from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import softjax as sj

from . import common


# JAX config (jax_enable_x64, matmul_precision) set in common.py

SOFTNESS = common.STABILITY_SOFTNESS


# ---------------------------------------------------------------------------
# grad_replace
# ---------------------------------------------------------------------------


def test_grad_replace_scalar():
    """grad_replace uses backward branch for gradients but forward value."""

    @sj.grad_replace
    def fn(x, *, forward: bool):
        return x if forward else 2.0 * x

    val = fn(3.0)
    assert val == 3.0
    grad = jax.grad(lambda z: fn(z))(3.0)
    assert grad == 2.0


# ---------------------------------------------------------------------------
# Elementwise ST functions
# ---------------------------------------------------------------------------

ELEMENTWISE_ST_CASES = [
    {
        "name": "abs_st",
        "st_fn": sj.abs_st,
        "hard_fn": lambda x: jnp.abs(x),
        "soft_fn": lambda x, **kw: sj.abs(x, **kw),
        "args": lambda: (jnp.array([-1.0, 0.5, -0.3, 0.8]),),
        "extra_kwargs": {},
    },
    {
        "name": "relu_st",
        "st_fn": sj.relu_st,
        "hard_fn": lambda x: jax.nn.relu(x),
        "soft_fn": lambda x, **kw: sj.relu(x, **kw),
        "args": lambda: (jnp.array([-1.0, 0.5, -0.3, 0.8]),),
        "extra_kwargs": {},
    },
    {
        "name": "clip_st",
        "st_fn": sj.clip_st,
        "hard_fn": lambda x: jnp.clip(x, -0.25, 0.25),
        "soft_fn": lambda x, **kw: sj.clip(x, -0.25, 0.25, **kw),
        "args": lambda: (jnp.array([-1.0, 0.5, -0.3, 0.8]),),
        "extra_kwargs": {"a": -0.25, "b": 0.25},
    },
    {
        "name": "sign_st",
        "st_fn": sj.sign_st,
        "hard_fn": lambda x: jnp.sign(x).astype(jnp.float_),
        "soft_fn": lambda x, **kw: sj.sign(x, **kw),
        "args": lambda: (jnp.array([-1.0, 0.5, -0.3, 0.8]),),
        "extra_kwargs": {},
    },
    {
        "name": "round_st",
        "st_fn": sj.round_st,
        "hard_fn": lambda x: jnp.round(x),
        "soft_fn": lambda x, **kw: sj.round(x, **kw),
        "args": lambda: (jnp.array([-0.7, 0.3, 1.5, -1.2]),),
        "extra_kwargs": {},
    },
    {
        "name": "heaviside_st",
        "st_fn": sj.heaviside_st,
        "hard_fn": lambda x: jnp.where(
            x < 0.0, 0.0, jnp.where(x > 0.0, 1.0, 0.5)
        ).astype(jnp.float_),
        "soft_fn": lambda x, **kw: sj.heaviside(x, **kw),
        "args": lambda: (jnp.array([-1.0, 0.5, -0.3, 0.8]),),
        "extra_kwargs": {},
    },
]


@pytest.mark.parametrize("case", ELEMENTWISE_ST_CASES, ids=lambda c: c["name"])
def test_st_elementwise_forward(case):
    """Forward value of ST function must equal the hard function output."""
    args = case["args"]()
    x = args[0]

    if case["name"] == "clip_st":
        st_out = case["st_fn"](
            x,
            case["extra_kwargs"]["a"],
            case["extra_kwargs"]["b"],
            softness=SOFTNESS,
            mode="smooth",
        )
    else:
        st_out = case["st_fn"](x, softness=SOFTNESS, mode="smooth")

    expected = case["hard_fn"](x)
    np.testing.assert_allclose(np.asarray(st_out), np.asarray(expected), atol=1e-5)


@pytest.mark.parametrize("case", ELEMENTWISE_ST_CASES, ids=lambda c: c["name"])
def test_st_elementwise_gradient(case):
    """Gradient of ST function must match soft function gradient and be finite."""
    args = case["args"]()
    x = args[0]

    if case["name"] == "clip_st":

        def loss_st(z):
            return jnp.sum(
                case["st_fn"](
                    z,
                    case["extra_kwargs"]["a"],
                    case["extra_kwargs"]["b"],
                    softness=SOFTNESS,
                    mode="smooth",
                )
            )

        def loss_soft(z):
            return jnp.sum(case["soft_fn"](z, softness=SOFTNESS, mode="smooth"))
    else:

        def loss_st(z):
            return jnp.sum(case["st_fn"](z, softness=SOFTNESS, mode="smooth"))

        def loss_soft(z):
            return jnp.sum(case["soft_fn"](z, softness=SOFTNESS, mode="smooth"))

    grad_st = jax.grad(loss_st)(x)
    grad_soft = jax.grad(loss_soft)(x)
    np.testing.assert_allclose(
        np.asarray(grad_st), np.asarray(grad_soft), atol=1e-5, rtol=1e-5
    )
    common.assert_finite(grad_st, msg=case["name"])


# ---------------------------------------------------------------------------
# Comparison ST functions
# ---------------------------------------------------------------------------

COMPARISON_ST_CASES = [
    {
        "name": "greater_st",
        "st_fn": sj.greater_st,
        "hard_fn": lambda x, y: jnp.greater(x, y).astype(jnp.float_),
        "soft_fn": sj.greater,
    },
    {
        "name": "greater_equal_st",
        "st_fn": sj.greater_equal_st,
        "hard_fn": lambda x, y: jnp.greater_equal(x, y).astype(jnp.float_),
        "soft_fn": sj.greater_equal,
    },
    {
        "name": "less_st",
        "st_fn": sj.less_st,
        "hard_fn": lambda x, y: jnp.less(x, y).astype(jnp.float_),
        "soft_fn": sj.less,
    },
    {
        "name": "less_equal_st",
        "st_fn": sj.less_equal_st,
        "hard_fn": lambda x, y: jnp.less_equal(x, y).astype(jnp.float_),
        "soft_fn": sj.less_equal,
    },
    {
        "name": "equal_st",
        "st_fn": sj.equal_st,
        "hard_fn": lambda x, y: jnp.equal(x, y).astype(jnp.float_),
        "soft_fn": sj.equal,
    },
    {
        "name": "not_equal_st",
        "st_fn": sj.not_equal_st,
        "hard_fn": lambda x, y: jnp.not_equal(x, y).astype(jnp.float_),
        "soft_fn": sj.not_equal,
    },
    {
        "name": "isclose_st",
        "st_fn": sj.isclose_st,
        "hard_fn": lambda x, y: jnp.isclose(x, y).astype(jnp.float_),
        "soft_fn": sj.isclose,
    },
]


@pytest.mark.parametrize("case", COMPARISON_ST_CASES, ids=lambda c: c["name"])
def test_st_comparison_forward(case):
    """Forward of comparison ST must equal hard comparison."""
    x = jnp.array([-1.0, 0.5, 0.0, 0.8])
    y = jnp.array([0.1, 0.2, 0.0, 0.9])
    st_out = case["st_fn"](x, y, softness=SOFTNESS, mode="smooth")
    expected = case["hard_fn"](x, y)
    np.testing.assert_allclose(np.asarray(st_out), np.asarray(expected), atol=1e-5)


@pytest.mark.parametrize("case", COMPARISON_ST_CASES, ids=lambda c: c["name"])
def test_st_comparison_gradient(case):
    """Gradient of comparison ST must match soft gradient and be finite."""
    x = jnp.array([-1.0, 0.5, 0.0, 0.8])
    y = jnp.array([0.1, 0.2, 0.0, 0.9])

    def loss_st(z):
        return jnp.sum(case["st_fn"](z, y, softness=SOFTNESS, mode="smooth"))

    def loss_soft(z):
        return jnp.sum(case["soft_fn"](z, y, softness=SOFTNESS, mode="smooth"))

    grad_st = jax.grad(loss_st)(x)
    grad_soft = jax.grad(loss_soft)(x)
    np.testing.assert_allclose(
        np.asarray(grad_st), np.asarray(grad_soft), atol=1e-5, rtol=1e-5
    )
    common.assert_finite(grad_st, msg=case["name"])


# ---------------------------------------------------------------------------
# Array ST functions
# ---------------------------------------------------------------------------


def _make_vec():
    return jnp.array([0.1, 0.4, -0.2, 0.3])


def test_st_argmax_forward_and_grad():
    """argmax_st returns hard one-hot but soft gradients."""
    x = _make_vec()
    weights = jnp.arange(4.0)

    hard_forward = sj.argmax_st(x, softness=SOFTNESS, mode="smooth")
    expected_forward = jax.nn.one_hot(jnp.argmax(x), num_classes=x.shape[0])
    np.testing.assert_allclose(np.asarray(hard_forward), np.asarray(expected_forward))

    def loss_st(inp):
        dist = sj.argmax_st(inp, softness=SOFTNESS, mode="smooth")
        return jnp.dot(dist, weights)

    def loss_soft(inp):
        dist = sj.argmax(inp, softness=SOFTNESS, mode="smooth")
        return jnp.dot(dist, weights)

    grad_st = jax.grad(loss_st)(x)
    grad_soft = jax.grad(loss_soft)(x)
    np.testing.assert_allclose(
        np.asarray(grad_st), np.asarray(grad_soft), atol=1e-5, rtol=1e-5
    )
    common.assert_finite(grad_st, msg="argmax_st")


def test_st_argmin_forward_and_grad():
    """argmin_st returns hard one-hot but soft gradients."""
    x = _make_vec()
    weights = jnp.arange(4.0)

    hard_forward = sj.argmin_st(x, softness=SOFTNESS, mode="smooth")
    expected_forward = jax.nn.one_hot(jnp.argmin(x), num_classes=x.shape[0])
    np.testing.assert_allclose(np.asarray(hard_forward), np.asarray(expected_forward))

    def loss_st(inp):
        dist = sj.argmin_st(inp, softness=SOFTNESS, mode="smooth")
        return jnp.dot(dist, weights)

    def loss_soft(inp):
        dist = sj.argmin(inp, softness=SOFTNESS, mode="smooth")
        return jnp.dot(dist, weights)

    grad_st = jax.grad(loss_st)(x)
    grad_soft = jax.grad(loss_soft)(x)
    np.testing.assert_allclose(
        np.asarray(grad_st), np.asarray(grad_soft), atol=1e-5, rtol=1e-5
    )
    common.assert_finite(grad_st, msg="argmin_st")


def test_st_max_forward_and_grad():
    """max_st returns hard max but soft gradients."""
    x = _make_vec()

    hard_forward = sj.max_st(x, softness=SOFTNESS, mode="smooth")
    expected = jnp.max(x)
    np.testing.assert_allclose(
        np.asarray(hard_forward), np.asarray(expected), atol=1e-5
    )

    def loss_st(inp):
        return jnp.sum(sj.max_st(inp, softness=SOFTNESS, mode="smooth"))

    def loss_soft(inp):
        return jnp.sum(sj.max(inp, softness=SOFTNESS, mode="smooth"))

    grad_st = jax.grad(loss_st)(x)
    grad_soft = jax.grad(loss_soft)(x)
    np.testing.assert_allclose(
        np.asarray(grad_st), np.asarray(grad_soft), atol=1e-5, rtol=1e-5
    )
    common.assert_finite(grad_st, msg="max_st")


def test_st_min_forward_and_grad():
    """min_st returns hard min but soft gradients."""
    x = _make_vec()

    hard_forward = sj.min_st(x, softness=SOFTNESS, mode="smooth")
    expected = jnp.min(x)
    np.testing.assert_allclose(
        np.asarray(hard_forward), np.asarray(expected), atol=1e-5
    )

    def loss_st(inp):
        return jnp.sum(sj.min_st(inp, softness=SOFTNESS, mode="smooth"))

    def loss_soft(inp):
        return jnp.sum(sj.min(inp, softness=SOFTNESS, mode="smooth"))

    grad_st = jax.grad(loss_st)(x)
    grad_soft = jax.grad(loss_soft)(x)
    np.testing.assert_allclose(
        np.asarray(grad_st), np.asarray(grad_soft), atol=1e-5, rtol=1e-5
    )
    common.assert_finite(grad_st, msg="min_st")


def test_st_sort_forward_and_grad():
    """sort_st returns hard sorted but soft gradients."""
    x = _make_vec()

    hard_forward = sj.sort_st(x, softness=SOFTNESS, mode="smooth")
    expected = jnp.sort(x)
    np.testing.assert_allclose(
        np.asarray(hard_forward), np.asarray(expected), atol=1e-5
    )

    def loss_st(inp):
        return jnp.sum(sj.sort_st(inp, softness=SOFTNESS, mode="smooth"))

    def loss_soft(inp):
        return jnp.sum(sj.sort(inp, softness=SOFTNESS, mode="smooth"))

    grad_st = jax.grad(loss_st)(x)
    grad_soft = jax.grad(loss_soft)(x)
    np.testing.assert_allclose(
        np.asarray(grad_st), np.asarray(grad_soft), atol=1e-5, rtol=1e-5
    )
    common.assert_finite(grad_st, msg="sort_st")


def test_st_argsort_forward_and_grad():
    """argsort_st returns hard permutation but soft gradients."""
    x = _make_vec()
    weights = jnp.arange(4.0)

    hard_forward = sj.argsort_st(x, softness=SOFTNESS, mode="smooth")
    expected = jax.nn.one_hot(jnp.argsort(x), x.shape[0])
    np.testing.assert_allclose(
        np.asarray(hard_forward), np.asarray(expected), atol=1e-5
    )

    def loss_st(inp):
        perm = sj.argsort_st(inp, softness=SOFTNESS, mode="smooth")
        return jnp.sum(perm @ weights)

    def loss_soft(inp):
        perm = sj.argsort(inp, softness=SOFTNESS, mode="smooth")
        return jnp.sum(perm @ weights)

    grad_st = jax.grad(loss_st)(x)
    grad_soft = jax.grad(loss_soft)(x)
    np.testing.assert_allclose(
        np.asarray(grad_st), np.asarray(grad_soft), atol=1e-5, rtol=1e-5
    )
    common.assert_finite(grad_st, msg="argsort_st")


def test_st_rank_forward_and_grad():
    """rank_st returns hard rank but soft gradients."""
    x = _make_vec()

    hard_forward = sj.rank_st(x, softness=SOFTNESS, mode="smooth")
    expected = sj.rank(x, mode="hard")
    np.testing.assert_allclose(
        np.asarray(hard_forward), np.asarray(expected), atol=1e-5
    )

    def loss_st(inp):
        return jnp.sum(sj.rank_st(inp, softness=SOFTNESS, mode="smooth"))

    def loss_soft(inp):
        return jnp.sum(sj.rank(inp, softness=SOFTNESS, mode="smooth"))

    grad_st = jax.grad(loss_st)(x)
    grad_soft = jax.grad(loss_soft)(x)
    np.testing.assert_allclose(
        np.asarray(grad_st), np.asarray(grad_soft), atol=1e-5, rtol=1e-5
    )
    common.assert_finite(grad_st, msg="rank_st")


def test_st_median_forward_and_grad():
    """median_st returns hard median but soft gradients."""
    x = jnp.array([0.1, 0.4, -0.2, 0.3, 0.5])

    hard_forward = sj.median_st(x, softness=SOFTNESS, mode="smooth")
    expected = jnp.median(x)
    np.testing.assert_allclose(
        np.asarray(hard_forward), np.asarray(expected), atol=1e-5
    )

    def loss_st(inp):
        return jnp.sum(sj.median_st(inp, softness=SOFTNESS, mode="smooth"))

    def loss_soft(inp):
        return jnp.sum(sj.median(inp, softness=SOFTNESS, mode="smooth"))

    grad_st = jax.grad(loss_st)(x)
    grad_soft = jax.grad(loss_soft)(x)
    np.testing.assert_allclose(
        np.asarray(grad_st), np.asarray(grad_soft), atol=1e-5, rtol=1e-5
    )
    common.assert_finite(grad_st, msg="median_st")


def test_st_argmedian_forward_and_grad():
    """argmedian_st returns hard argmedian but soft gradients."""
    x = jnp.array([0.1, 0.4, -0.2, 0.3, 0.5])
    weights = jnp.arange(5.0)

    hard_forward = sj.argmedian_st(x, softness=SOFTNESS, mode="smooth")
    expected = sj.argmedian(x, mode="hard")
    np.testing.assert_allclose(
        np.asarray(hard_forward), np.asarray(expected), atol=1e-5
    )

    def loss_st(inp):
        dist = sj.argmedian_st(inp, softness=SOFTNESS, mode="smooth")
        return jnp.dot(dist, weights)

    def loss_soft(inp):
        dist = sj.argmedian(inp, softness=SOFTNESS, mode="smooth")
        return jnp.dot(dist, weights)

    grad_st = jax.grad(loss_st)(x)
    grad_soft = jax.grad(loss_soft)(x)
    np.testing.assert_allclose(
        np.asarray(grad_st), np.asarray(grad_soft), atol=1e-5, rtol=1e-5
    )
    common.assert_finite(grad_st, msg="argmedian_st")


def test_st_quantile_forward_and_grad():
    """quantile_st returns hard quantile but soft gradients."""
    x = jnp.array([0.1, 0.4, -0.2, 0.3, 0.5])
    q = 0.75

    hard_forward = sj.quantile_st(x, q, softness=SOFTNESS, mode="smooth")
    expected = jnp.quantile(x, q)
    np.testing.assert_allclose(
        np.asarray(hard_forward), np.asarray(expected), atol=1e-5
    )

    def loss_st(inp):
        return jnp.sum(sj.quantile_st(inp, q, softness=SOFTNESS, mode="smooth"))

    def loss_soft(inp):
        return jnp.sum(sj.quantile(inp, q, softness=SOFTNESS, mode="smooth"))

    grad_st = jax.grad(loss_st)(x)
    grad_soft = jax.grad(loss_soft)(x)
    np.testing.assert_allclose(
        np.asarray(grad_st), np.asarray(grad_soft), atol=1e-5, rtol=1e-5
    )
    common.assert_finite(grad_st, msg="quantile_st")


def test_st_argquantile_forward_and_grad():
    """argquantile_st returns hard argquantile but soft gradients."""
    x = jnp.array([0.1, 0.4, -0.2, 0.3, 0.5])
    q = 0.75
    weights = jnp.arange(5.0)

    hard_forward = sj.argquantile_st(x, q, softness=SOFTNESS, mode="smooth")
    expected = sj.argquantile(x, q, mode="hard")
    np.testing.assert_allclose(
        np.asarray(hard_forward), np.asarray(expected), atol=1e-5
    )

    def loss_st(inp):
        dist = sj.argquantile_st(inp, q, softness=SOFTNESS, mode="smooth")
        return jnp.dot(dist, weights)

    def loss_soft(inp):
        dist = sj.argquantile(inp, q, softness=SOFTNESS, mode="smooth")
        return jnp.dot(dist, weights)

    grad_st = jax.grad(loss_st)(x)
    grad_soft = jax.grad(loss_soft)(x)
    np.testing.assert_allclose(
        np.asarray(grad_st), np.asarray(grad_soft), atol=1e-5, rtol=1e-5
    )
    common.assert_finite(grad_st, msg="argquantile_st")


def test_st_top_k_forward_and_grad():
    """top_k_st supports tuple outputs and straight-through gradients."""
    x = jnp.array([0.5, -0.1, 0.3, 0.8])

    hard_vals, _ = sj.top_k_st(x, k=3, softness=SOFTNESS, mode="smooth")
    expected_vals, _ = sj.top_k(x, k=3, mode="hard")
    np.testing.assert_allclose(np.asarray(hard_vals), np.asarray(expected_vals))

    def loss_st(inp):
        vals, _ = sj.top_k_st(inp, k=3, softness=SOFTNESS, mode="smooth")
        return jnp.sum(vals)

    def loss_soft(inp):
        vals, _ = sj.top_k(inp, k=3, mode="smooth", softness=SOFTNESS)
        return jnp.sum(vals)

    grad_st = jax.grad(loss_st)(x)
    grad_soft = jax.grad(loss_soft)(x)
    np.testing.assert_allclose(
        np.asarray(grad_st), np.asarray(grad_soft), atol=1e-5, rtol=1e-5
    )
    common.assert_finite(grad_st, msg="top_k_st")


@pytest.mark.parametrize("method", ["fast_soft_sort", "smooth_sort", "sorting_network"])
def test_st_top_k_none_index_methods(method):
    """top_k_st handles methods that return None for soft_index in backward.

    When the soft backward returns (values, None) but the hard forward returns
    (values, one_hot_index), st() should return the hard forward values for both
    outputs and only compute gradients through the values.
    """
    x = jnp.array([0.5, -0.1, 0.3, 0.8])

    # Forward: should return hard top-k values and hard one-hot indices
    hard_vals, hard_idx = sj.top_k_st(
        x, k=2, softness=SOFTNESS, mode="smooth", method=method
    )
    expected_vals, expected_idx = sj.top_k(x, k=2, mode="hard")
    np.testing.assert_allclose(np.asarray(hard_vals), np.asarray(expected_vals))
    np.testing.assert_allclose(np.asarray(hard_idx), np.asarray(expected_idx))

    # Gradients: should match soft method gradients (only through values)
    def loss_st(inp):
        vals, _ = sj.top_k_st(
            inp, k=2, softness=SOFTNESS, mode="smooth", method=method
        )
        return jnp.sum(vals)

    def loss_soft(inp):
        vals, _ = sj.top_k(inp, k=2, mode="smooth", softness=SOFTNESS, method=method)
        return jnp.sum(vals)

    grad_st = jax.grad(loss_st)(x)
    grad_soft = jax.grad(loss_soft)(x)
    np.testing.assert_allclose(
        np.asarray(grad_st), np.asarray(grad_soft), atol=1e-5, rtol=1e-5
    )
    common.assert_finite(grad_st, msg=f"top_k_st({method})")


def test_st_percentile_forward_and_grad():
    """percentile_st returns hard percentile but soft gradients."""
    x = jnp.array([0.1, 0.4, -0.2, 0.3, 0.5])
    p = 75.0

    hard_forward = sj.percentile_st(x, p, softness=SOFTNESS, mode="smooth")
    expected = jnp.percentile(x, p)
    np.testing.assert_allclose(
        np.asarray(hard_forward), np.asarray(expected), atol=1e-5
    )

    def loss_st(inp):
        return jnp.sum(sj.percentile_st(inp, p, softness=SOFTNESS, mode="smooth"))

    def loss_soft(inp):
        return jnp.sum(sj.percentile(inp, p, softness=SOFTNESS, mode="smooth"))

    grad_st = jax.grad(loss_st)(x)
    grad_soft = jax.grad(loss_soft)(x)
    np.testing.assert_allclose(
        np.asarray(grad_st), np.asarray(grad_soft), atol=1e-5, rtol=1e-5
    )
    common.assert_finite(grad_st, msg="percentile_st")


def test_st_argpercentile_forward_and_grad():
    """argpercentile_st returns hard argpercentile but soft gradients."""
    x = jnp.array([0.1, 0.4, -0.2, 0.3, 0.5])
    p = 75.0
    weights = jnp.arange(5.0)

    hard_forward = sj.argpercentile_st(x, p, softness=SOFTNESS, mode="smooth")
    expected = sj.argpercentile(x, p, mode="hard")
    np.testing.assert_allclose(
        np.asarray(hard_forward), np.asarray(expected), atol=1e-5
    )

    def loss_st(inp):
        dist = sj.argpercentile_st(inp, p, softness=SOFTNESS, mode="smooth")
        return jnp.dot(dist, weights)

    def loss_soft(inp):
        dist = sj.argpercentile(inp, p, softness=SOFTNESS, mode="smooth")
        return jnp.dot(dist, weights)

    grad_st = jax.grad(loss_st)(x)
    grad_soft = jax.grad(loss_soft)(x)
    np.testing.assert_allclose(
        np.asarray(grad_st), np.asarray(grad_soft), atol=1e-5, rtol=1e-5
    )
    common.assert_finite(grad_st, msg="argpercentile_st")


# ---------------------------------------------------------------------------
# st() generic wrapper
# ---------------------------------------------------------------------------


def test_st_generic_wrapper_forward():
    """st() wrapper: forward uses hard mode, backward uses soft mode."""
    x = jnp.array([0.1, 0.4, -0.2, 0.3])

    max_st_via_wrapper = sj.st(sj.max)(x, softness=SOFTNESS, mode="smooth")
    expected = jnp.max(x)
    np.testing.assert_allclose(
        np.asarray(max_st_via_wrapper), np.asarray(expected), atol=1e-5
    )


def test_st_generic_wrapper_gradient():
    """st() wrapper gradient matches the soft gradient, not hard gradient."""
    x = jnp.array([0.1, 0.4, -0.2, 0.3])

    def loss_st_wrapper(z):
        return jnp.sum(sj.st(sj.max)(z, softness=SOFTNESS, mode="smooth"))

    def loss_soft(z):
        return jnp.sum(sj.max(z, softness=SOFTNESS, mode="smooth"))

    grad_st = jax.grad(loss_st_wrapper)(x)
    grad_soft = jax.grad(loss_soft)(x)
    np.testing.assert_allclose(
        np.asarray(grad_st), np.asarray(grad_soft), atol=1e-5, rtol=1e-5
    )
    common.assert_finite(grad_st, msg="st() wrapper")


def test_st_generic_wrapper_sort():
    """st() wrapper works on sort: forward is hard sorted, backward is soft."""
    x = jnp.array([0.3, 0.1, 0.4, -0.2])

    st_sorted = sj.st(sj.sort)(x, softness=SOFTNESS, mode="smooth")
    expected = jnp.sort(x)
    np.testing.assert_allclose(np.asarray(st_sorted), np.asarray(expected), atol=1e-5)

    def loss_st(z):
        return jnp.sum(sj.st(sj.sort)(z, softness=SOFTNESS, mode="smooth"))

    def loss_soft(z):
        return jnp.sum(sj.sort(z, softness=SOFTNESS, mode="smooth"))

    grad_st = jax.grad(loss_st)(x)
    grad_soft = jax.grad(loss_soft)(x)
    np.testing.assert_allclose(
        np.asarray(grad_st), np.asarray(grad_soft), atol=1e-5, rtol=1e-5
    )


# ---------------------------------------------------------------------------
# st() mode passing variants
# ---------------------------------------------------------------------------


def test_st_no_mode_param_default():
    """st() on a function without mode param defaults to smooth backward."""
    x = jnp.array([-0.5, 0.5, 1.5])

    @sj.st
    def my_relu_prod(x, y, **kwargs):
        return sj.relu(x, **kwargs) * sj.relu(y, **kwargs)

    y = jnp.array([1.0, 2.0, 0.5])
    result = my_relu_prod(x, y)
    expected = jax.nn.relu(x) * jax.nn.relu(y)
    np.testing.assert_allclose(np.asarray(result), np.asarray(expected), atol=1e-5)


def test_st_no_mode_param_gradient():
    """st() without mode param gives non-zero gradient at hard-zero points."""
    x = jnp.array(-0.5)
    y = jnp.array(2.0)

    @sj.st
    def my_relu_prod(x, y, **kwargs):
        return sj.relu(x, **kwargs) * sj.relu(y, **kwargs)

    # Forward is hard: relu(-0.5) * relu(2.0) = 0
    result = my_relu_prod(x, y)
    np.testing.assert_allclose(float(result), 0.0, atol=1e-5)

    # But gradient should be non-zero (from smooth backward)
    grad = jax.grad(my_relu_prod)(x, y)
    assert float(grad) != 0.0
    common.assert_finite(grad, msg="st() no mode param grad")


def test_st_no_mode_param_override_mode():
    """st() without mode param allows overriding mode via kwarg."""
    x = jnp.array([-0.5, 0.5, 1.5])

    @sj.st
    def my_abs(x, **kwargs):
        return sj.abs(x, **kwargs)

    # c0 mode should also work
    result_c0 = my_abs(x, mode="c0")
    expected = jnp.abs(x)
    np.testing.assert_allclose(np.asarray(result_c0), np.asarray(expected), atol=1e-5)

    # Gradient with c0 mode
    grad_c0 = jax.grad(lambda z: jnp.sum(my_abs(z, mode="c0")))(x)
    common.assert_finite(grad_c0, msg="st() no mode c0 grad")

    # Gradient with smooth mode (explicit override, same as default)
    grad_smooth = jax.grad(lambda z: jnp.sum(my_abs(z, mode="smooth")))(x)
    common.assert_finite(grad_smooth, msg="st() no mode smooth grad")


def test_st_no_mode_param_matches_explicit_mode():
    """st() without mode param gives same results as explicit mode='smooth'."""
    x = jnp.array([-0.5, 0.5, 1.5])

    @sj.st
    def abs_no_mode(x, **kwargs):
        return sj.abs(x, **kwargs)

    @sj.st
    def abs_with_mode(x, mode="smooth", **kwargs):
        return sj.abs(x, mode=mode, **kwargs)

    result_no = abs_no_mode(x, softness=SOFTNESS)
    result_with = abs_with_mode(x, softness=SOFTNESS)
    np.testing.assert_allclose(np.asarray(result_no), np.asarray(result_with))

    grad_no = jax.grad(lambda z: jnp.sum(abs_no_mode(z, softness=SOFTNESS)))(x)
    grad_with = jax.grad(lambda z: jnp.sum(abs_with_mode(z, softness=SOFTNESS)))(x)
    np.testing.assert_allclose(
        np.asarray(grad_no), np.asarray(grad_with), atol=1e-10
    )


def test_st_explicit_mode_positional():
    """st() with explicit mode param supports passing mode positionally."""
    x = jnp.array([-0.5, 0.5, 1.5])

    @sj.st
    def my_abs(x, mode="smooth", **kwargs):
        return sj.abs(x, mode=mode, **kwargs)

    # mode as kwarg
    result_kw = my_abs(x, mode="c0", softness=SOFTNESS)
    # mode as positional
    result_pos = my_abs(x, "c0", softness=SOFTNESS)
    np.testing.assert_allclose(np.asarray(result_kw), np.asarray(result_pos))

    grad_kw = jax.grad(lambda z: jnp.sum(my_abs(z, mode="c0", softness=SOFTNESS)))(x)
    grad_pos = jax.grad(lambda z: jnp.sum(my_abs(z, "c0", softness=SOFTNESS)))(x)
    np.testing.assert_allclose(
        np.asarray(grad_kw), np.asarray(grad_pos), atol=1e-10
    )


def test_st_explicit_mode_nondefault():
    """st() respects a non-default mode in the function signature."""
    x = jnp.array([-0.5, 0.5, 1.5])

    @sj.st
    def my_abs_c0(x, mode="c0", **kwargs):
        return sj.abs(x, mode=mode, **kwargs)

    # Should use c0 by default (not smooth)
    grad_default = jax.grad(lambda z: jnp.sum(my_abs_c0(z, softness=SOFTNESS)))(x)

    @sj.st
    def my_abs_smooth(x, mode="smooth", **kwargs):
        return sj.abs(x, mode=mode, **kwargs)

    grad_smooth = jax.grad(lambda z: jnp.sum(my_abs_smooth(z, softness=SOFTNESS)))(x)

    # c0 and smooth have different gradients
    assert not np.allclose(np.asarray(grad_default), np.asarray(grad_smooth), atol=1e-3)
