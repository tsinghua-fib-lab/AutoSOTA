"""Tests for JAX transform safety (jit, vmap) and higher-order derivatives."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import softjax as sj


# JAX config (jax_enable_x64, matmul_precision) set in common.py

# ---------------------------------------------------------------------------
# Valid methods per function (mirrors test_arraywise.py)
# ---------------------------------------------------------------------------

VALUE_METHODS = [
    "softsort",
    "neuralsort",
    "fast_soft_sort",
    "smooth_sort",
    "ot",
    "sorting_network",
]
ARG_METHODS = ["softsort", "neuralsort", "ot", "sorting_network"]

FUNCTION_SPECS = {
    "sort": VALUE_METHODS,
    "rank": VALUE_METHODS,
    "max": VALUE_METHODS,
    "min": VALUE_METHODS,
    "median": VALUE_METHODS,
    "quantile": VALUE_METHODS,
    "top_k": VALUE_METHODS,
    "argsort": ARG_METHODS,
    "argmax": ARG_METHODS,
    "argmin": ARG_METHODS,
    "argmedian": ARG_METHODS,
    "argquantile": ARG_METHODS,
}

MODES = ["smooth", "c0", "c1", "c2"]
ELEMENTWISE_OPS = ["relu", "abs", "sign", "round", "heaviside", "clip"]
COMPARISON_OPS = ["greater", "less", "equal", "not_equal", "isclose"]


def _skip_unsupported(method, mode):
    if method == "smooth_sort" and mode != "smooth":
        pytest.skip("smooth_sort only supports smooth mode")


# ---------------------------------------------------------------------------
# Helpers to call operators uniformly
# ---------------------------------------------------------------------------


def _call_elementwise(op, x, mode="smooth", softness=0.5):
    fn = getattr(sj, op)
    if op == "clip":
        return fn(x, 0.0, 1.0, softness=softness, mode=mode)
    return fn(x, softness=softness, mode=mode)


def _call_comparison(op, x, mode="smooth", softness=0.5):
    return getattr(sj, op)(x, 0.0, softness=softness, mode=mode)


def _weighted_sum(out):
    if out.ndim == 0:
        return out
    weights = jnp.arange(1, out.shape[-1] + 1, dtype=out.dtype)
    return jnp.sum(out * weights)


def _axiswise_loss(op, x, method, mode, softness=0.5):
    """Scalar weighted loss from an axiswise op."""
    if op == "quantile":
        out = sj.quantile(x, 0.5, axis=-1, softness=softness, mode=mode, method=method)
    elif op == "argquantile":
        out = sj.argquantile(
            x, 0.5, axis=-1, softness=softness, mode=mode, method=method
        )
    elif op == "top_k":
        out, _ = sj.top_k(x, k=2, axis=-1, softness=softness, mode=mode, method=method)
    elif op == "argmedian":
        out = sj.argmedian(x, axis=-1, softness=softness, mode=mode, method=method)
    elif op in ("argsort", "argmax", "argmin"):
        out = getattr(sj, op)(x, axis=-1, softness=softness, mode=mode, method=method)
    else:
        out = getattr(sj, op)(x, axis=-1, softness=softness, mode=mode, method=method)
    return _weighted_sum(out)


def test_top_k_gated_grad_false_returns_differentiable_soft_index():
    """gated_grad=False gates value selection, not the returned soft index."""
    x = jnp.array([0.0, 0.2, -0.1, 0.4], dtype=jnp.float64)
    weights = jnp.array([1.0, -2.0, 0.5, 3.0], dtype=jnp.float64)

    def loss(z):
        _, soft_idx = sj.top_k(
            z,
            k=2,
            softness=0.1,
            mode="smooth",
            method="neuralsort",
            gated_grad=False,
        )
        return jnp.sum(soft_idx.sum(axis=0) * weights)

    grad = jax.grad(loss)(x)
    assert jnp.all(jnp.isfinite(grad))
    assert jnp.any(jnp.abs(grad) > 1e-8)


# ---------------------------------------------------------------------------
# Build exhaustive (op, method, mode) parametrization for axiswise tests
# ---------------------------------------------------------------------------

_AXISWISE_CASES = []
for op, methods in FUNCTION_SPECS.items():
    for method in methods:
        for mode in MODES:
            # Skip known-invalid method+mode combinations
            if method == "smooth_sort" and mode != "smooth":
                continue
            _AXISWISE_CASES.append((op, method, mode))

_AXISWISE_IDS = [f"{op}_{method}_{mode}" for op, method, mode in _AXISWISE_CASES]


# ===================================================================
# JIT tests
# ===================================================================


class TestJit:
    """Verify operators produce correct results under jax.jit."""

    @pytest.mark.parametrize("op", ELEMENTWISE_OPS)
    @pytest.mark.parametrize("mode", MODES)
    def test_elementwise(self, op, mode):
        x = jnp.array([-0.8, 0.3, -0.1, 0.6], dtype=jnp.float64)
        fn = lambda x: _call_elementwise(op, x, mode=mode)
        np.testing.assert_allclose(fn(x), jax.jit(fn)(x), rtol=1e-5)

    @pytest.mark.parametrize("op", COMPARISON_OPS)
    def test_comparison(self, op):
        x = jnp.array([-0.8, 0.3, -0.1, 0.6], dtype=jnp.float64)
        fn = lambda x: _call_comparison(op, x)
        np.testing.assert_allclose(fn(x), jax.jit(fn)(x), rtol=1e-5)

    @pytest.mark.parametrize("op,method,mode", _AXISWISE_CASES, ids=_AXISWISE_IDS)
    def test_axiswise(self, op, method, mode):
        x = jnp.array([0.3, -0.5, 1.2, 0.1], dtype=jnp.float64)
        fn = lambda x: _axiswise_loss(op, x, method, mode)
        np.testing.assert_allclose(fn(x), jax.jit(fn)(x), rtol=1e-5)

    @pytest.mark.parametrize("op,method,mode", _AXISWISE_CASES, ids=_AXISWISE_IDS)
    def test_axiswise_grad(self, op, method, mode):
        x = jnp.array([0.3, -0.5, 1.2, 0.1], dtype=jnp.float64)
        fn = lambda x: _axiswise_loss(op, x, method, mode)
        g_eager = jax.grad(fn)(x)
        g_jit = jax.jit(jax.grad(fn))(x)
        assert jnp.all(jnp.isfinite(g_jit)), (
            f"jit grad NaN/Inf for {op} {method} {mode}"
        )
        np.testing.assert_allclose(g_eager, g_jit, rtol=1e-5, atol=1e-12)


# ===================================================================
# vmap tests
# ===================================================================


class TestVmap:
    """Verify operators work correctly under jax.vmap."""

    @pytest.mark.parametrize("op", ELEMENTWISE_OPS)
    @pytest.mark.parametrize("mode", MODES)
    def test_elementwise(self, op, mode):
        xs = jnp.array(
            [[-0.8, 0.3, -0.1, 0.6], [0.5, -0.2, 0.9, -0.4]], dtype=jnp.float64
        )
        fn = lambda x: _call_elementwise(op, x, mode=mode)
        vmapped = jax.vmap(fn)(xs)
        manual = jnp.stack([fn(xs[i]) for i in range(2)])
        np.testing.assert_allclose(vmapped, manual, rtol=1e-5)

    @pytest.mark.parametrize("op,method,mode", _AXISWISE_CASES, ids=_AXISWISE_IDS)
    def test_axiswise(self, op, method, mode):
        xs = jnp.array(
            [[0.3, -0.5, 1.2, 0.1], [0.8, 0.2, -0.3, 0.6]], dtype=jnp.float64
        )
        fn = lambda x: _axiswise_loss(op, x, method, mode)
        vmapped = jax.vmap(fn)(xs)
        manual = jnp.stack([fn(xs[i]) for i in range(2)])
        np.testing.assert_allclose(vmapped, manual, rtol=1e-4)

    @pytest.mark.parametrize("op,method,mode", _AXISWISE_CASES, ids=_AXISWISE_IDS)
    def test_axiswise_grad(self, op, method, mode):
        xs = jnp.array(
            [[0.3, -0.5, 1.2, 0.1], [0.8, 0.2, -0.3, 0.6]], dtype=jnp.float64
        )
        fn = lambda x: _axiswise_loss(op, x, method, mode)
        vmapped = jax.vmap(jax.grad(fn))(xs)
        manual = jnp.stack([jax.grad(fn)(xs[i]) for i in range(2)])
        assert jnp.all(jnp.isfinite(vmapped)), (
            f"vmap grad NaN/Inf for {op} {method} {mode}"
        )
        np.testing.assert_allclose(vmapped, manual, rtol=1e-4, atol=1e-12)


# ===================================================================
# jit + vmap composition tests
# ===================================================================


class TestJitVmap:
    """Verify jit(vmap(grad(...))) works correctly."""

    @pytest.mark.parametrize("op", ELEMENTWISE_OPS[:3])
    def test_elementwise(self, op):
        xs = jnp.array([[-0.8, 0.3], [0.5, -0.2]], dtype=jnp.float64)
        fn = lambda x: _call_elementwise(op, x)
        result = jax.jit(jax.vmap(fn))(xs)
        expected = jax.vmap(fn)(xs)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    @pytest.mark.parametrize(
        "method", ["softsort", "fast_soft_sort", "sorting_network"]
    )
    def test_sort_grad(self, method):
        xs = jnp.array(
            [[0.3, -0.5, 1.2, 0.1], [0.8, 0.2, -0.3, 0.6]], dtype=jnp.float64
        )
        fn = lambda x: _axiswise_loss("sort", x, method, "smooth")
        result = jax.jit(jax.vmap(jax.grad(fn)))(xs)
        expected = jax.vmap(jax.grad(fn))(xs)
        assert jnp.all(jnp.isfinite(result))
        np.testing.assert_allclose(result, expected, rtol=1e-4)


# ===================================================================
# Higher-order derivative tests
# ===================================================================


class TestHigherOrderDerivatives:
    """Verify second-order and higher derivatives.

    Elementwise operators are composed of standard smooth functions, so they
    support arbitrary-order derivatives. Axiswise operators use custom_vjp;
    we test that jax.hessian produces finite results for all combinations.
    """

    # --- Elementwise: Hessian across all ops x modes ---

    @pytest.mark.parametrize("op", ELEMENTWISE_OPS)
    @pytest.mark.parametrize("mode", MODES)
    def test_hessian_elementwise(self, op, mode):
        """jax.hessian of elementwise ops produces finite results."""
        x = jnp.array([-0.8, 0.3, -0.1, 0.6], dtype=jnp.float64)
        loss = lambda x: jnp.sum(_call_elementwise(op, x, mode=mode))
        H = jax.hessian(loss)(x)
        assert H.shape == (4, 4)
        assert jnp.all(jnp.isfinite(H)), f"Hessian NaN/Inf for {op} {mode}"

    @pytest.mark.parametrize("op", ["relu", "abs", "sign"])
    @pytest.mark.parametrize("mode", MODES)
    def test_hessian_elementwise_vs_finite_diff(self, op, mode):
        """Elementwise Hessian matches finite-difference approximation."""
        x = jnp.array([-0.8, 0.3, -0.1, 0.6], dtype=jnp.float64)
        loss = lambda x: jnp.sum(_call_elementwise(op, x, mode=mode))
        H = jax.hessian(loss)(x)

        eps = 1e-5
        n = x.shape[0]
        H_fd = np.zeros((n, n))
        for i in range(n):
            ei = jnp.zeros_like(x).at[i].set(eps)
            H_fd[i] = (jax.grad(loss)(x + ei) - jax.grad(loss)(x - ei)) / (2 * eps)

        np.testing.assert_allclose(H, H_fd, rtol=1e-3, atol=1e-6)

    # --- Elementwise: Hessian for comparison ops ---

    @pytest.mark.parametrize("op", COMPARISON_OPS)
    @pytest.mark.parametrize("mode", MODES)
    def test_hessian_comparison(self, op, mode):
        """jax.hessian of comparison ops produces finite results."""
        x = jnp.array([-0.8, 0.3, -0.1, 0.6], dtype=jnp.float64)
        loss = lambda x: jnp.sum(_call_comparison(op, x, mode=mode))
        H = jax.hessian(loss)(x)
        assert H.shape == (4, 4)
        assert jnp.all(jnp.isfinite(H)), f"Hessian NaN/Inf for {op} {mode}"

    # --- Elementwise: higher-order nested grad (scalar input) ---

    @pytest.mark.parametrize("mode", MODES)
    def test_third_derivative_sigmoid(self, mode):
        """Third derivative of sigmoidal is finite."""
        x = jnp.array(0.3, dtype=jnp.float64)
        fn = lambda z: sj.heaviside(z, softness=0.5, mode=mode)
        g3 = jax.grad(jax.grad(jax.grad(fn)))(x)
        assert jnp.isfinite(g3), f"3rd derivative not finite for {mode}"

    def test_fourth_derivative_smooth_relu(self):
        """Smooth relu (softplus) supports 4th-order derivatives."""
        x = jnp.array(0.5, dtype=jnp.float64)
        fn = lambda z: sj.relu(z, softness=0.5, mode="smooth")
        g = fn
        for order in range(1, 5):
            g = jax.grad(g)
            assert jnp.isfinite(g(x)), f"order-{order} derivative not finite"

    # --- Axiswise: Hessian across all (op, method, mode) combinations ---

    @pytest.mark.parametrize("op,method,mode", _AXISWISE_CASES, ids=_AXISWISE_IDS)
    def test_hessian_axiswise(self, op, method, mode):
        """jax.hessian produces finite results for every valid (op, method, mode)."""
        if method == "ot" and mode == "c0":
            pytest.xfail(
                "OT c0 (L2 LBFGS + ImplicitAdjoint) does not support second-order "
                "differentiation: the linear solver in the backward pass receives "
                "non-finite inputs when differentiated through a second time."
            )
        x = jnp.array([0.3, -0.5, 1.2, 0.1], dtype=jnp.float64)
        fn = lambda x: _axiswise_loss(op, x, method, mode)
        H = jax.hessian(fn)(x)
        assert H.shape == (4, 4), f"Hessian shape wrong for {op} {method} {mode}"
        assert jnp.all(jnp.isfinite(H)), f"Hessian NaN/Inf for {op} {method} {mode}"


# ===================================================================
# Selection operator transform tests
# ===================================================================


class TestSelectionTransforms:
    """Verify selection operators work under jit and vmap."""

    def test_jit_where(self):
        cond = jnp.array([0.9, 0.1, 0.8])
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([4.0, 5.0, 6.0])
        np.testing.assert_allclose(
            sj.where(cond, x, y), jax.jit(sj.where)(cond, x, y), rtol=1e-5
        )

    def test_jit_take_along_axis(self):
        x = jnp.array([0.3, 1.0, -0.5])
        soft_ind = sj.argsort(x, softness=0.5, mode="smooth")
        eager = sj.take_along_axis(x, soft_ind)
        jitted = jax.jit(lambda a, b: sj.take_along_axis(a, b))(x, soft_ind)
        np.testing.assert_allclose(eager, jitted, rtol=1e-5)

    def test_vmap_where(self):
        cond = jnp.array([[0.9, 0.1], [0.3, 0.7]])
        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        y = jnp.array([[5.0, 6.0], [7.0, 8.0]])
        vmapped = jax.vmap(sj.where)(cond, x, y)
        manual = jnp.stack([sj.where(cond[i], x[i], y[i]) for i in range(2)])
        np.testing.assert_allclose(vmapped, manual, rtol=1e-5)


# ===================================================================
# Straight-through transform tests
# ===================================================================


class TestStraightThroughTransforms:
    """Verify ST wrappers work under jit and vmap."""

    def test_jit_relu_st(self):
        x = jnp.array([-0.5, 0.3, 1.0], dtype=jnp.float64)
        np.testing.assert_allclose(sj.relu_st(x), jax.jit(sj.relu_st)(x), rtol=1e-5)

    def test_jit_grad_sort_st(self):
        fn = lambda x: jnp.sum(sj.sort_st(x))
        x = jnp.array([0.3, -0.5, 1.2, 0.1], dtype=jnp.float64)
        g_eager = jax.grad(fn)(x)
        g_jit = jax.jit(jax.grad(fn))(x)
        assert jnp.all(jnp.isfinite(g_jit))
        np.testing.assert_allclose(g_eager, g_jit, rtol=1e-5)

    def test_vmap_relu_st(self):
        xs = jnp.array([[-0.5, 0.3], [1.0, -0.2]], dtype=jnp.float64)
        vmapped = jax.vmap(sj.relu_st)(xs)
        manual = jnp.stack([sj.relu_st(xs[i]) for i in range(2)])
        np.testing.assert_allclose(vmapped, manual, rtol=1e-5)
