import jax
import jax.numpy as jnp
import numpy as np
import pytest

import softjax as sj

from . import common


# JAX config (jax_enable_x64, matmul_precision) set in common.py


def test_selection_error_paths():
    """Negative shape/axis checks."""
    x = jnp.arange(6.0).reshape(2, 3)
    bad_soft = jnp.ones((2, 3))
    with pytest.raises(Exception):
        sj.take_along_axis(x, bad_soft, axis=0)  # rank mismatch

    with pytest.raises(Exception):
        sj.take(x, jnp.ones((2, 2, 3)), axis=None)

    with pytest.raises(ValueError):
        sj.choose(jnp.ones((2,)), jnp.ones((3, 2)))

    with pytest.raises(ValueError):
        sj.dynamic_index_in_dim(x, jnp.ones((4,)), axis=1)

    with pytest.raises(Exception):
        sj.dynamic_slice_in_dim(x, jnp.ones((2,)), slice_size=5, axis=1)

    with pytest.raises(ValueError):
        sj.argmax(x, axis=5)


def test_selection_helpers_hard_paths():
    """take_along_axis / take / choose / dynamic slices hard parity."""
    x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    one_hot = jax.nn.one_hot(jnp.array([[0, 2], [1, 0]]), x.shape[1])
    out = sj.take_along_axis(x, one_hot, axis=1)
    expected = jnp.take_along_axis(x, jnp.array([[0, 2], [1, 0]]), axis=1)
    common.assert_allclose(out, expected)

    flat_idx = jax.nn.one_hot(jnp.array([0, 3]), x.size)
    taken = sj.take(x, flat_idx, axis=None)
    expected_take = jnp.take(x, jnp.array([0, 3]), axis=None)
    common.assert_allclose(taken, expected_take)

    choices = jnp.stack([jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])])
    # soft_index must have shape (..., [n]) matching choices.ndim
    mix = sj.choose(jnp.array([[0.0, 1.0], [1.0, 0.0]]), choices)
    common.assert_allclose(mix, jnp.array([3.0, 2.0]))

    idx = jax.nn.one_hot(jnp.array(1), 2)
    dyn = sj.dynamic_index_in_dim(jnp.array([[10.0, 20.0], [30.0, 40.0]]), idx, axis=0)
    common.assert_allclose(dyn, jnp.array([[30.0, 40.0]]))

    start = jax.nn.one_hot(jnp.array([0, 1]), 4)
    sliced = sj.dynamic_slice_in_dim(jnp.arange(4.0), start[0], slice_size=2, axis=0)
    common.assert_allclose(sliced, jnp.array([0.0, 1.0]))


# ---------------------------------------------------------------------------
# dynamic_slice test
# ---------------------------------------------------------------------------


def test_dynamic_slice():
    """Test sj.dynamic_slice with soft start indices."""
    x = jnp.arange(12.0).reshape(3, 4)

    # Hard one-hot start indices: start at (1, 2) with slice_sizes (2, 2)
    start_0 = jax.nn.one_hot(1, 3)  # row start = 1
    start_1 = jax.nn.one_hot(2, 4)  # col start = 2

    out = sj.dynamic_slice(x, [start_0, start_1], slice_sizes=[2, 2])
    expected = jax.lax.dynamic_slice(x, (1, 2), (2, 2))
    common.assert_allclose(out, expected)

    # Also test start at (0, 0)
    start_0_zero = jax.nn.one_hot(0, 3)
    start_1_zero = jax.nn.one_hot(0, 4)
    out2 = sj.dynamic_slice(x, [start_0_zero, start_1_zero], slice_sizes=[2, 3])
    expected2 = jax.lax.dynamic_slice(x, (0, 0), (2, 3))
    common.assert_allclose(out2, expected2)


def test_dynamic_slice_1d():
    """Test sj.dynamic_slice on a 1D array."""
    x = jnp.arange(6.0)
    start = jax.nn.one_hot(2, 6)
    out = sj.dynamic_slice(x, [start], slice_sizes=[3])
    expected = jax.lax.dynamic_slice(x, (2,), (3,))
    common.assert_allclose(out, expected)


# ---------------------------------------------------------------------------
# Parametrized take_along_axis over shapes and dtypes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", ("float32", "float64"))
@pytest.mark.parametrize(
    "shape, axis",
    [((4,), 0), ((2, 3), 0), ((2, 3), 1), ((2, 3), -1)],
)
def test_take_along_axis_parametrized(dtype, shape, axis):
    """take_along_axis with various shapes and axes."""
    x = common.make_array(shape, dtype, "jax")

    # Pick first element along the axis
    n = x.shape[axis]
    idx_int = jnp.zeros(shape, dtype=jnp.int32)  # always pick index 0
    idx_one_hot = jax.nn.one_hot(idx_int, n)

    out = sj.take_along_axis(x, idx_one_hot, axis=axis)
    expected = jnp.take_along_axis(x, idx_int, axis=axis)
    common.assert_allclose(out, expected)


@pytest.mark.parametrize("dtype", ("float32", "float64"))
@pytest.mark.parametrize("idx_val", [0, 1, 2])
def test_dynamic_index_in_dim_parametrized(dtype, idx_val):
    """dynamic_index_in_dim with various indices."""
    x = jnp.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=dtype)
    idx = jax.nn.one_hot(idx_val, x.shape[1])
    out = sj.dynamic_index_in_dim(x, idx, axis=1)
    expected = jax.lax.dynamic_index_in_dim(x, idx_val, axis=1, keepdims=True)
    common.assert_allclose(out, expected)


# ---------------------------------------------------------------------------
# Selection gradient tests
# ---------------------------------------------------------------------------


def test_take_along_axis_gradient():
    """Gradients flow through take_along_axis (soft index)."""
    x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    # soft_index must be x.ndim+1 dimensional: (2, 1, [3]) — pick 1 element per row
    logits = jnp.array([[[0.0, 5.0, 0.0]], [[5.0, 0.0, 0.0]]])  # (2, 1, 3)
    soft_idx = jax.nn.softmax(logits, axis=-1)

    def loss(arr):
        return jnp.sum(sj.take_along_axis(arr, soft_idx, axis=1))

    grad = jax.grad(loss)(x)
    common.assert_finite(grad, msg="take_along_axis gradient wrt x")


def test_take_along_axis_gradient_wrt_index():
    """Gradients flow through take_along_axis w.r.t. soft index."""
    x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    def loss(idx_logits):
        # idx_logits: (2, 1, 3) -> softmax -> soft_index
        soft_idx = jax.nn.softmax(idx_logits, axis=-1)
        return jnp.sum(sj.take_along_axis(x, soft_idx, axis=1))

    logits = jnp.array([[[0.0, 5.0, 0.0]], [[5.0, 0.0, 0.0]]])  # (2, 1, 3)
    grad = jax.grad(loss)(logits)
    common.assert_finite(grad, msg="take_along_axis gradient wrt index")


def test_dynamic_index_in_dim_gradient():
    """Gradients flow through dynamic_index_in_dim."""
    x = jnp.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])

    def loss(arr):
        idx = jax.nn.one_hot(1, arr.shape[1])
        return jnp.sum(sj.dynamic_index_in_dim(arr, idx, axis=1))

    grad = jax.grad(loss)(x)
    common.assert_finite(grad, msg="dynamic_index_in_dim gradient")


def test_choose_gradient():
    """Gradients flow through choose w.r.t. choices."""
    choices = jnp.stack([jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])])

    def loss(c):
        soft_idx = jnp.array([[0.3, 0.7], [0.8, 0.2]])
        return jnp.sum(sj.choose(soft_idx, c))

    grad = jax.grad(loss)(choices)
    common.assert_finite(grad, msg="choose gradient")


def test_dynamic_slice_in_dim_gradient():
    """Gradients flow through dynamic_slice_in_dim."""
    x = jnp.arange(6.0)

    def loss(arr):
        start = jax.nn.one_hot(1, arr.shape[0])
        return jnp.sum(sj.dynamic_slice_in_dim(arr, start, slice_size=3, axis=0))

    grad = jax.grad(loss)(x)
    common.assert_finite(grad, msg="dynamic_slice_in_dim gradient")


# ---------------------------------------------------------------------------
# Soft interpolation tests (non-one-hot indices)
# ---------------------------------------------------------------------------


def test_take_along_axis_soft_interpolation():
    """Non-one-hot soft index yields interpolated output."""
    x = jnp.array([10.0, 20.0, 30.0])  # shape (3,)
    # soft_index: ndim = x.ndim + 1 = 2, shape (1, [3])
    soft_idx = jnp.array([[1.0 / 3, 1.0 / 3, 1.0 / 3]])
    out = sj.take_along_axis(x, soft_idx, axis=0)
    expected = jnp.array([20.0])  # mean of [10, 20, 30]
    common.assert_allclose(out, expected, tol=1e-5)


def test_choose_soft_interpolation():
    """Non-one-hot soft index interpolates between choices."""
    choices = jnp.stack([jnp.array([1.0]), jnp.array([3.0])])
    soft_idx = jnp.array([[0.5, 0.5]])
    out = sj.choose(soft_idx, choices)
    expected = jnp.array([2.0])  # 0.5*1.0 + 0.5*3.0
    common.assert_allclose(out, expected, tol=1e-5)


def test_dynamic_index_in_dim_soft_interpolation():
    """Non-one-hot soft index interpolates along dimension."""
    x = jnp.array([[10.0], [30.0]])
    soft_idx = jnp.array([0.5, 0.5])  # equal weight on both rows
    out = sj.dynamic_index_in_dim(x, soft_idx, axis=0)
    expected = jnp.array([[20.0]])  # 0.5*10 + 0.5*30
    common.assert_allclose(out, expected, tol=1e-5)


# ---------------------------------------------------------------------------
# where gradient test
# ---------------------------------------------------------------------------


def test_where_gradient():
    """Gradients flow through sj.where w.r.t. both branches."""
    condition = jnp.array([0.8, 0.2, 0.5])  # soft booleans

    def loss_x(arr):
        y = jnp.zeros_like(arr)
        return jnp.sum(sj.where(condition, arr, y))

    def loss_y(arr):
        x_arr = jnp.zeros_like(arr)
        return jnp.sum(sj.where(condition, x_arr, arr))

    x = jnp.array([1.0, 2.0, 3.0])
    grad_x = jax.grad(loss_x)(x)
    grad_y = jax.grad(loss_y)(x)
    common.assert_finite(grad_x, msg="where gradient wrt x")
    common.assert_finite(grad_y, msg="where gradient wrt y")
    # gradient wrt x branch should be the condition values
    np.testing.assert_allclose(np.asarray(grad_x), np.asarray(condition), atol=1e-5)
    # gradient wrt y branch should be (1 - condition)
    np.testing.assert_allclose(
        np.asarray(grad_y), np.asarray(1.0 - condition), atol=1e-5
    )


# ---------------------------------------------------------------------------
# Gradient vs finite differences for selection ops
# ---------------------------------------------------------------------------


def test_take_along_axis_grad_vs_finite_diff():
    """take_along_axis gradient wrt x matches finite differences."""
    x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=jnp.float64)
    soft_idx = jax.nn.softmax(
        jnp.array([[[0.0, 5.0, 0.0]], [[5.0, 0.0, 0.0]]], dtype=jnp.float64), axis=-1
    )

    def loss(arr):
        return jnp.sum(sj.take_along_axis(arr, soft_idx, axis=1))

    common.assert_grad_matches_finite_diff(loss, x, msg="take_along_axis")


def test_choose_grad_vs_finite_diff():
    """choose gradient wrt choices matches finite differences."""
    choices = jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float64)
    soft_idx = jnp.array([[0.3, 0.7], [0.8, 0.2]], dtype=jnp.float64)

    def loss(c):
        return jnp.sum(sj.choose(soft_idx, c))

    common.assert_grad_matches_finite_diff(loss, choices, msg="choose")


def test_where_grad_vs_finite_diff():
    """where gradient wrt x matches finite differences."""
    condition = jnp.array([0.8, 0.2, 0.5], dtype=jnp.float64)
    y = jnp.array([10.0, 20.0, 30.0], dtype=jnp.float64)

    def loss(arr):
        return jnp.sum(sj.where(condition, arr, y))

    x = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64)
    common.assert_grad_matches_finite_diff(loss, x, msg="where")


# ---------------------------------------------------------------------------
# End-to-end differentiable pipeline test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["smooth", "c0"])
def test_end_to_end_argmax_take(mode):
    """End-to-end pipeline: argmax -> take_along_axis is differentiable."""
    x = jnp.array([[1.0, 3.0, 2.0], [5.0, 1.0, 4.0]], dtype=jnp.float64)

    def pipeline(z):
        soft_idx = sj.argmax(z, axis=-1, keepdims=True, mode=mode, softness=1.0)
        selected = sj.take_along_axis(z, soft_idx, axis=-1)
        return jnp.sum(selected)

    # Forward pass produces finite output
    out = pipeline(x)
    common.assert_finite(out, msg=f"pipeline output mode={mode}")

    # Backward pass produces finite gradients
    grad = jax.grad(pipeline)(x)
    common.assert_finite(grad, msg=f"pipeline gradient mode={mode}")

    # Gradient vs finite differences
    common.assert_grad_matches_finite_diff(pipeline, x, msg=f"pipeline mode={mode}")
