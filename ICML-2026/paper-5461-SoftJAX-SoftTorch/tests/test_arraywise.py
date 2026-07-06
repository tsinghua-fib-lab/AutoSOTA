import jax
import jax.numpy as jnp
import numpy as np
import pytest

import softjax as sj

from . import common


# JAX config (jax_enable_x64, matmul_precision) set in common.py

BACKENDS = common.BACKENDS
DEVICES = common.DEVICES
FLOAT_DTYPES = ("float32", "float64")
NEAR_HARD_SOFTNESS = common.NEAR_HARD_SOFTNESS
STABILITY_SOFTNESS = common.STABILITY_SOFTNESS

SHAPES = [(4,), (2, 3)]
AXIS = [None, -1, 0]
MODES = common.MODES_ARRAYWISE
SOFTNESSES = [NEAR_HARD_SOFTNESS, STABILITY_SOFTNESS]
KEEPDIMS = [False, True]

make_dtype = common._make_dtype
make_array = common.make_array
assert_simplex = common.assert_simplex


# ---------------------------------------------------------------------------
# Valid methods per function — no skipping needed
# ---------------------------------------------------------------------------

# Value-only methods (no soft indices): fast_soft_sort, smooth_sort, sorting_network
VALUE_METHODS = ["softsort", "neuralsort", "fast_soft_sort", "smooth_sort", "ot"]
SORT_VALUE_METHODS = VALUE_METHODS + ["sorting_network"]
ARG_METHODS = ["softsort", "neuralsort", "ot"]

FUNCTION_SPECS = {
    # (fn_name, methods, needs_arg_validation)
    "max": SORT_VALUE_METHODS,
    "min": SORT_VALUE_METHODS,
    "argmax": ARG_METHODS + ["sorting_network"],
    "argmin": ARG_METHODS + ["sorting_network"],
    "sort": SORT_VALUE_METHODS,
    "argsort": ARG_METHODS + ["sorting_network"],
    "rank": SORT_VALUE_METHODS,
    "median": SORT_VALUE_METHODS,
    "argmedian": ARG_METHODS + ["sorting_network"],
    "quantile": SORT_VALUE_METHODS,
    "argquantile": ARG_METHODS + ["sorting_network"],
    "percentile": SORT_VALUE_METHODS,
    "argpercentile": ARG_METHODS + ["sorting_network"],
}


def _skip_unsupported(method, mode):
    """Skip unsupported method+mode combinations."""
    if method == "smooth_sort" and mode not in ("smooth", "hard", "_hard"):
        pytest.skip("smooth_sort only supports smooth mode")


def _valid_axis(shape, axis):
    """Check if axis is valid for shape without skip."""
    if axis is None:
        return True
    return -len(shape) <= axis < len(shape)


def _build_fn_method_params(fn_names):
    """Build (fn_name, method) pairs for only valid combos."""
    params = []
    for fn_name in fn_names:
        for method in FUNCTION_SPECS[fn_name]:
            params.append((fn_name, method))
    return params


# ---------------------------------------------------------------------------
# max / min / argmax / argmin parametric sweep
# ---------------------------------------------------------------------------

_MAX_MIN_PARAMS = _build_fn_method_params(["max", "argmax", "min", "argmin"])


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("axis", AXIS)
@pytest.mark.parametrize("keepdims", KEEPDIMS)
@pytest.mark.parametrize(
    "fn_name, method", _MAX_MIN_PARAMS, ids=[f"{fn}-{m}" for fn, m in _MAX_MIN_PARAMS]
)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("softness", SOFTNESSES)
def test_max_min(
    backend, dtype, shape, axis, keepdims, fn_name, method, mode, softness
):
    """max/min/argmax/argmin: simplex, shape contracts, value parity near-hard."""
    _skip_unsupported(method, mode)
    if not _valid_axis(shape, axis):
        pytest.skip(f"axis {axis} invalid for shape {shape}")

    x = make_array(shape, dtype, backend)
    fn = getattr(sj, fn_name)
    ot_kwargs = common.ot_kwargs_for_method(method, softness)
    out = fn(
        x,
        axis=axis,
        keepdims=keepdims,
        softness=softness,
        mode=mode,
        method=method,
        ot_kwargs=ot_kwargs,
    )
    assert not jnp.any(jnp.isnan(out)), f"NaN in output of {fn_name}"
    assert out.dtype in {jnp.dtype("float32"), jnp.dtype("float64")}, (
        f"Unexpected dtype {out.dtype} in output of {fn_name}"
    )

    if "arg" in fn_name:
        common.assert_simplex(out, atol=common.TOLERANCE)

    if fn_name in ("argmax", "argmin"):
        jnp_fn = jnp.argmax if fn_name == "argmax" else jnp.argmin
        out_jnp = jnp_fn(x, axis=axis, keepdims=keepdims)
        assert out.shape[:-1] == out_jnp.shape, (
            f"Unexpected shape {out.shape} from {fn_name}"
        )
        if axis is not None:
            assert out.shape[-1] == x.shape[axis], (
                f"Unexpected shape {out.shape} from {fn_name}"
            )

    if softness == NEAR_HARD_SOFTNESS:
        out_hard = fn(x, axis=axis, keepdims=keepdims, mode="hard")
        if method == "ot":
            out = common.call_with_ot_retry(
                fn,
                x,
                axis=axis,
                keepdims=keepdims,
                softness=softness,
                mode=mode,
                method=method,
                _expected=out_hard,
            )
        common.assert_allclose(out, out_hard, tol=common.TOLERANCE)


# ---------------------------------------------------------------------------
# sort / argsort / rank parametric sweep
# ---------------------------------------------------------------------------

_SORT_RANK_PARAMS = _build_fn_method_params(["sort", "argsort", "rank"])


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("axis", AXIS)
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize(
    "fn_name, method",
    _SORT_RANK_PARAMS,
    ids=[f"{fn}-{m}" for fn, m in _SORT_RANK_PARAMS],
)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("softness", SOFTNESSES)
def test_sort_rank(
    backend, dtype, shape, axis, descending, fn_name, method, mode, softness
):
    """sort/argsort/rank: simplex, shape, value parity near-hard."""
    _skip_unsupported(method, mode)
    if not _valid_axis(shape, axis):
        pytest.skip(f"axis {axis} invalid for shape {shape}")

    ot_kwargs = common.ot_kwargs_for_method(method, softness)

    x = make_array(shape, dtype, backend)
    fn = getattr(sj, fn_name)
    out = fn(
        x,
        axis=axis,
        descending=descending,
        softness=softness,
        mode=mode,
        method=method,
        ot_kwargs=ot_kwargs,
    )
    assert not jnp.any(jnp.isnan(out)), f"NaN in output of {fn_name}"
    assert out.dtype in {jnp.dtype("float32"), jnp.dtype("float64")}, (
        f"Unexpected dtype {out.dtype} in output of {fn_name}"
    )

    if "arg" in fn_name:
        common.assert_simplex(out, atol=common.TOLERANCE)

    if fn_name == "argsort":
        out_jnp = jnp.argsort(x, axis=axis, descending=descending)
        assert out.shape[:-1] == out_jnp.shape, (
            f"Unexpected shape {out.shape} from argsort"
        )
        if axis is not None:
            assert out.shape[-1] == x.shape[axis], (
                f"Unexpected shape {out.shape} from argsort"
            )

    if softness == NEAR_HARD_SOFTNESS:
        out_hard = fn(x, axis=axis, descending=descending, mode="hard")
        if method == "ot":
            out = common.call_with_ot_retry(
                fn,
                x,
                axis=axis,
                descending=descending,
                softness=softness,
                mode=mode,
                method=method,
                _expected=out_hard,
            )
        common.assert_allclose(out, out_hard, tol=common.TOLERANCE)


# ---------------------------------------------------------------------------
# median / argmedian parametric sweep
# ---------------------------------------------------------------------------

_MEDIAN_PARAMS = _build_fn_method_params(["median", "argmedian"])


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("axis", AXIS)
@pytest.mark.parametrize("keepdims", KEEPDIMS)
@pytest.mark.parametrize(
    "fn_name, method", _MEDIAN_PARAMS, ids=[f"{fn}-{m}" for fn, m in _MEDIAN_PARAMS]
)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("softness", SOFTNESSES)
def test_median(backend, dtype, shape, axis, keepdims, fn_name, method, mode, softness):
    """median/argmedian: simplex, shape, value parity near-hard."""
    _skip_unsupported(method, mode)
    if not _valid_axis(shape, axis):
        pytest.skip(f"axis {axis} invalid for shape {shape}")

    ot_kwargs = common.ot_kwargs_for_method(method, softness)

    x = make_array(shape, dtype, backend)

    fn = getattr(sj, fn_name)
    out = fn(
        x,
        axis=axis,
        keepdims=keepdims,
        softness=softness,
        mode=mode,
        method=method,
        ot_kwargs=ot_kwargs,
    )
    assert not jnp.any(jnp.isnan(out)), f"NaN in output of {fn_name}"
    assert out.dtype in {jnp.dtype("float32"), jnp.dtype("float64")}, (
        f"Unexpected dtype {out.dtype} in output of {fn_name}"
    )

    if "arg" in fn_name:
        common.assert_simplex(out, atol=common.TOLERANCE)

    if softness == NEAR_HARD_SOFTNESS:
        out_hard = fn(x, axis=axis, keepdims=keepdims, mode="hard")
        if method == "ot":
            out = common.call_with_ot_retry(
                fn,
                x,
                axis=axis,
                keepdims=keepdims,
                softness=softness,
                mode=mode,
                method=method,
                _expected=out_hard,
            )
        common.assert_allclose(out, out_hard, tol=common.TOLERANCE)


# ---------------------------------------------------------------------------
# quantile / argquantile parametric sweep
# ---------------------------------------------------------------------------

_QUANTILE_PARAMS = _build_fn_method_params(["quantile", "argquantile"])


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("axis", AXIS)
@pytest.mark.parametrize("keepdims", KEEPDIMS)
@pytest.mark.parametrize(
    "fn_name, method", _QUANTILE_PARAMS, ids=[f"{fn}-{m}" for fn, m in _QUANTILE_PARAMS]
)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("softness", SOFTNESSES)
@pytest.mark.parametrize(
    "quantile_method", ["linear", "lower", "higher", "nearest", "midpoint"]
)
@pytest.mark.parametrize("q", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_quantile(
    backend,
    dtype,
    shape,
    axis,
    keepdims,
    fn_name,
    method,
    mode,
    softness,
    quantile_method,
    q,
):
    """quantile/argquantile: simplex, shape, value parity near-hard."""
    _skip_unsupported(method, mode)
    if not _valid_axis(shape, axis):
        pytest.skip(f"axis {axis} invalid for shape {shape}")

    ot_kwargs = common.ot_kwargs_for_method(method, softness)

    x = make_array(shape, dtype, backend)

    fn = getattr(sj, fn_name)
    out = fn(
        x,
        q,
        axis=axis,
        keepdims=keepdims,
        softness=softness,
        mode=mode,
        method=method,
        quantile_method=quantile_method,
        ot_kwargs=ot_kwargs,
    )
    assert not jnp.any(jnp.isnan(out)), f"NaN in output of {fn_name}"
    assert out.dtype in {jnp.dtype("float32"), jnp.dtype("float64")}, (
        f"Unexpected dtype {out.dtype} in output of {fn_name}"
    )

    if "arg" in fn_name:
        common.assert_simplex(out, atol=common.TOLERANCE)

    if softness == NEAR_HARD_SOFTNESS:
        out_hard = fn(
            x,
            q,
            axis=axis,
            keepdims=keepdims,
            mode="hard",
            quantile_method=quantile_method,
        )
        if method == "ot":
            out = common.call_with_ot_retry(
                fn,
                x,
                q,
                axis=axis,
                keepdims=keepdims,
                softness=softness,
                mode=mode,
                method=method,
                quantile_method=quantile_method,
                _expected=out_hard,
            )
        common.assert_allclose(out, out_hard, tol=common.TOLERANCE)


# ---------------------------------------------------------------------------
# percentile / argpercentile parametric sweep
# ---------------------------------------------------------------------------

_PERCENTILE_PARAMS = _build_fn_method_params(["percentile", "argpercentile"])


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("axis", AXIS)
@pytest.mark.parametrize("keepdims", KEEPDIMS)
@pytest.mark.parametrize(
    "fn_name, method",
    _PERCENTILE_PARAMS,
    ids=[f"{fn}-{m}" for fn, m in _PERCENTILE_PARAMS],
)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("softness", SOFTNESSES)
@pytest.mark.parametrize("p", [0.0, 25.0, 50.0, 75.0, 100.0])
def test_percentile(
    backend, dtype, shape, axis, keepdims, fn_name, method, mode, softness, p
):
    """percentile/argpercentile: simplex, shape, value parity near-hard."""
    _skip_unsupported(method, mode)
    if not _valid_axis(shape, axis):
        pytest.skip(f"axis {axis} invalid for shape {shape}")

    ot_kwargs = common.ot_kwargs_for_method(method, softness)

    x = make_array(shape, dtype, backend)

    fn = getattr(sj, fn_name)
    out = fn(
        x,
        p,
        axis=axis,
        keepdims=keepdims,
        softness=softness,
        mode=mode,
        method=method,
        ot_kwargs=ot_kwargs,
    )
    assert not jnp.any(jnp.isnan(out)), f"NaN in output of {fn_name}"
    assert out.dtype in {jnp.dtype("float32"), jnp.dtype("float64")}, (
        f"Unexpected dtype {out.dtype} in output of {fn_name}"
    )

    if "arg" in fn_name:
        common.assert_simplex(out, atol=common.TOLERANCE)

    if softness == NEAR_HARD_SOFTNESS:
        out_hard = fn(x, p, axis=axis, keepdims=keepdims, mode="hard")
        if method == "ot":
            out = common.call_with_ot_retry(
                fn,
                x,
                p,
                axis=axis,
                keepdims=keepdims,
                softness=softness,
                mode=mode,
                method=method,
                _expected=out_hard,
            )
        common.assert_allclose(out, out_hard, tol=common.TOLERANCE)


# ---------------------------------------------------------------------------
# top_k parametric sweep
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize(
    "axis", [-3, -2, -1, 0, 1, 2, 3]
)  # top_k doesn't support axis=None
@pytest.mark.parametrize("k", [1, 2, 4])
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("softness", SOFTNESSES)
@pytest.mark.parametrize("method", SORT_VALUE_METHODS)
def test_top_k(backend, dtype, shape, k, axis, mode, softness, method):
    """top_k: simplex, shape, value parity near-hard."""
    _skip_unsupported(method, mode)
    if not _valid_axis(shape, axis):
        pytest.skip(f"axis {axis} invalid for shape {shape}")

    if k > shape[axis]:
        pytest.skip(f"k={k} exceeds axis size {shape[axis]}")

    ot_kwargs = common.ot_kwargs_for_method(method, softness)

    x = make_array(shape, dtype, backend)
    vals, soft_idx = sj.top_k(
        x,
        k=k,
        axis=axis,
        mode=mode,
        method=method,
        softness=softness,
        ot_kwargs=ot_kwargs,
    )

    assert not jnp.any(jnp.isnan(vals)), "NaN in output of top_k"
    assert vals.dtype in {jnp.dtype("float32"), jnp.dtype("float64")}

    # fast_soft_sort does not return soft indices (only soft values)
    if soft_idx is not None:
        assert not jnp.any(jnp.isnan(soft_idx)), "NaN in output of top_k"
        assert soft_idx.dtype in {jnp.dtype("float32"), jnp.dtype("float64")}

        # soft_idx has shape (..., k, ..., [n]) with k at the axis position
        expected_leading = list(x.shape)
        expected_leading[axis] = k
        assert soft_idx.shape[:-1] == tuple(expected_leading), (
            f"Unexpected shape {soft_idx.shape} from top_k, expected leading dims {tuple(expected_leading)}"
        )
        assert soft_idx.shape[-1] == x.shape[axis], (
            f"Unexpected last dim {soft_idx.shape[-1]} from top_k, expected {x.shape[axis]}"
        )

        common.assert_simplex(soft_idx, atol=common.TOLERANCE)

    if softness == NEAR_HARD_SOFTNESS:
        hard_vals, hard_idx = sj.top_k(x, k=k, axis=axis, mode="hard")
        if method == "ot":

            def _check_top_k(result):
                v, si = result
                common.assert_allclose(v, hard_vals, tol=common.TOLERANCE)
                common.assert_allclose(si, hard_idx, tol=common.TOLERANCE)

            result = common.call_with_ot_retry(
                sj.top_k,
                x,
                k=k,
                axis=axis,
                softness=softness,
                mode=mode,
                method=method,
                _check_fn=_check_top_k,
            )
            vals, soft_idx = result
        common.assert_allclose(vals, hard_vals, tol=common.TOLERANCE)
        if soft_idx is not None:
            common.assert_allclose(soft_idx, hard_idx, tol=common.TOLERANCE)


# ---------------------------------------------------------------------------
# all / any
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("shape", SHAPES)
def test_all_any(dtype, backend, shape):
    """Soft all/any semantics, shapes, and bounds."""
    ones = make_array(shape, dtype, backend, type="ones")
    zeros = make_array(shape, dtype, backend, type="zeros")

    axes = [-1]
    if len(shape) > 1:
        axes.append(0)
    if len(shape) > 2:
        axes.append(1)

    for axis in axes:
        out_all_ones = sj.all(ones, axis=axis)
        out_all_zeros = sj.all(zeros, axis=axis)
        out_any_ones = sj.any(ones, axis=axis)
        out_any_zeros = sj.any(zeros, axis=axis)
        for out in (out_all_ones, out_all_zeros, out_any_ones, out_any_zeros):
            assert not jnp.any(jnp.isnan(out))
            assert jnp.all(out >= -common.TOLERANCE)
            assert jnp.all(out <= 1.0 + common.TOLERANCE)

    probs = make_array(shape, dtype, backend, softbool=True)
    for axis in axes:
        out_all = sj.all(probs, axis=axis)
        out_any = sj.any(probs, axis=axis)
        assert out_all.ndim == probs.ndim - 1
        assert out_any.ndim == probs.ndim - 1
        assert not jnp.any(jnp.isnan(out_all))
        assert not jnp.any(jnp.isnan(out_any))
        common.assert_softbool(out_all)
        common.assert_softbool(out_any)
        assert np.all(np.asarray(out_all) <= np.asarray(out_any) + 1e-6)


def test_all_any_jax_parity():
    """Hard-boolean inputs: sj.all/any must match jnp.all/any."""
    x_bool = jnp.array([1.0, 1.0, 1.0])
    assert float(sj.all(x_bool, axis=-1)) == pytest.approx(1.0, abs=1e-5)
    assert float(sj.any(x_bool, axis=-1)) == pytest.approx(1.0, abs=1e-5)

    x_mixed = jnp.array([1.0, 0.0, 1.0])
    assert float(sj.all(x_mixed, axis=-1)) == pytest.approx(0.0, abs=1e-5)
    assert float(sj.any(x_mixed, axis=-1)) == pytest.approx(1.0, abs=1e-5)

    x_zeros = jnp.array([0.0, 0.0, 0.0])
    assert float(sj.all(x_zeros, axis=-1)) == pytest.approx(0.0, abs=1e-5)
    assert float(sj.any(x_zeros, axis=-1)) == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# JAX parity: max / min
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(4,), (2, 3), (2, 3, 2)])
@pytest.mark.parametrize("axis", [None, -1, 0])
@pytest.mark.parametrize("keepdims", KEEPDIMS)
def test_max_jax_parity(shape, axis, keepdims):
    """sj.max(x, mode='hard') must match jnp.max(x, ...)."""
    if not _valid_axis(shape, axis):
        pytest.skip(f"axis {axis} invalid for shape {shape}")
    x = make_array(shape, "float64", "jax")
    soft_out = sj.max(x, axis=axis, keepdims=keepdims, mode="hard")
    jax_out = jnp.max(x, axis=axis, keepdims=keepdims)
    common.assert_jax_parity(soft_out, jax_out, msg="max")


@pytest.mark.parametrize("shape", [(4,), (2, 3), (2, 3, 2)])
@pytest.mark.parametrize("axis", [None, -1, 0])
@pytest.mark.parametrize("keepdims", KEEPDIMS)
def test_min_jax_parity(shape, axis, keepdims):
    """sj.min(x, mode='hard') must match jnp.min(x, ...)."""
    if not _valid_axis(shape, axis):
        pytest.skip(f"axis {axis} invalid for shape {shape}")
    x = make_array(shape, "float64", "jax")
    soft_out = sj.min(x, axis=axis, keepdims=keepdims, mode="hard")
    jax_out = jnp.min(x, axis=axis, keepdims=keepdims)
    common.assert_jax_parity(soft_out, jax_out, msg="min")


# ---------------------------------------------------------------------------
# JAX parity: argmax / argmin
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(4,), (2, 3)])
@pytest.mark.parametrize("axis", [-1, 0])
@pytest.mark.parametrize("keepdims", KEEPDIMS)
def test_argmax_jax_parity(shape, axis, keepdims):
    """sj.argmax(x, mode='hard') produces one-hot matching jnp.argmax."""
    if not _valid_axis(shape, axis):
        pytest.skip(f"axis {axis} invalid for shape {shape}")
    x = make_array(shape, "float64", "jax")
    soft_out = sj.argmax(x, axis=axis, keepdims=keepdims, mode="hard")
    jax_idx = jnp.argmax(x, axis=axis, keepdims=keepdims)
    expected = jax.nn.one_hot(jax_idx, x.shape[axis])
    common.assert_jax_parity(soft_out, expected, msg="argmax")


@pytest.mark.parametrize("shape", [(4,), (2, 3)])
@pytest.mark.parametrize("axis", [-1, 0])
@pytest.mark.parametrize("keepdims", KEEPDIMS)
def test_argmin_jax_parity(shape, axis, keepdims):
    """sj.argmin(x, mode='hard') produces one-hot matching jnp.argmin."""
    if not _valid_axis(shape, axis):
        pytest.skip(f"axis {axis} invalid for shape {shape}")
    x = make_array(shape, "float64", "jax")
    soft_out = sj.argmin(x, axis=axis, keepdims=keepdims, mode="hard")
    jax_idx = jnp.argmin(x, axis=axis, keepdims=keepdims)
    expected = jax.nn.one_hot(jax_idx, x.shape[axis])
    common.assert_jax_parity(soft_out, expected, msg="argmin")


# ---------------------------------------------------------------------------
# JAX parity: sort / argsort / rank
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(4,), (2, 3)])
@pytest.mark.parametrize("axis", [-1, 0])
@pytest.mark.parametrize("descending", [False, True])
def test_sort_jax_parity(shape, axis, descending):
    """sj.sort(x, mode='hard') must match jnp.sort."""
    if not _valid_axis(shape, axis):
        pytest.skip(f"axis {axis} invalid for shape {shape}")
    x = make_array(shape, "float64", "jax")
    soft_out = sj.sort(x, axis=axis, descending=descending, mode="hard")
    jax_out = jnp.sort(x, axis=axis, descending=descending)
    common.assert_jax_parity(soft_out, jax_out, msg="sort")


@pytest.mark.parametrize("shape", [(4,), (2, 3)])
@pytest.mark.parametrize("axis", [-1, 0])
@pytest.mark.parametrize("descending", [False, True])
def test_argsort_jax_parity(shape, axis, descending):
    """sj.argsort(x, mode='hard') produces permutation matrices matching jnp.argsort."""
    if not _valid_axis(shape, axis):
        pytest.skip(f"axis {axis} invalid for shape {shape}")
    x = make_array(shape, "float64", "jax")
    soft_out = sj.argsort(x, axis=axis, descending=descending, mode="hard")
    jax_idx = jnp.argsort(x, axis=axis, descending=descending)
    expected = jax.nn.one_hot(jax_idx, x.shape[axis])
    common.assert_jax_parity(soft_out, expected, msg="argsort")


@pytest.mark.parametrize("shape", [(4,), (2, 3)])
@pytest.mark.parametrize("axis", [-1, 0])
def test_rank_hard_shape(shape, axis):
    """sj.rank(x, mode='hard') has correct shape and produces valid rank values."""
    if not _valid_axis(shape, axis):
        pytest.skip(f"axis {axis} invalid for shape {shape}")
    x = make_array(shape, "float64", "jax")
    out = sj.rank(x, axis=axis, mode="hard")
    assert out.shape == x.shape, f"Unexpected rank shape {out.shape}"
    common.assert_finite(out, msg="rank hard")


# ---------------------------------------------------------------------------
# JAX parity: median / quantile / percentile
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(5,), (2, 3)])
@pytest.mark.parametrize("axis", [-1, 0])
@pytest.mark.parametrize("keepdims", KEEPDIMS)
def test_median_jax_parity(shape, axis, keepdims):
    """sj.median(x, mode='hard') must match jnp.median."""
    if not _valid_axis(shape, axis):
        pytest.skip(f"axis {axis} invalid for shape {shape}")
    x = make_array(shape, "float64", "jax")
    soft_out = sj.median(x, axis=axis, keepdims=keepdims, mode="hard")
    jax_out = jnp.median(x, axis=axis, keepdims=keepdims)
    common.assert_jax_parity(soft_out, jax_out, msg="median")


@pytest.mark.parametrize("shape", [(5,), (2, 3)])
@pytest.mark.parametrize("axis", [-1, 0])
@pytest.mark.parametrize("keepdims", KEEPDIMS)
@pytest.mark.parametrize("q", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_quantile_jax_parity(shape, axis, keepdims, q):
    """sj.quantile(x, q, mode='hard') must match jnp.quantile."""
    if not _valid_axis(shape, axis):
        pytest.skip(f"axis {axis} invalid for shape {shape}")
    x = make_array(shape, "float64", "jax")
    soft_out = sj.quantile(x, q, axis=axis, keepdims=keepdims, mode="hard")
    jax_out = jnp.quantile(x, q, axis=axis, keepdims=keepdims)
    common.assert_jax_parity(soft_out, jax_out, msg=f"quantile q={q}")


@pytest.mark.parametrize("shape", [(5,), (2, 3)])
@pytest.mark.parametrize("axis", [-1, 0])
@pytest.mark.parametrize("keepdims", KEEPDIMS)
@pytest.mark.parametrize("p", [0.0, 25.0, 50.0, 75.0, 100.0])
def test_percentile_jax_parity(shape, axis, keepdims, p):
    """sj.percentile(x, p, mode='hard') must match jnp.percentile."""
    if not _valid_axis(shape, axis):
        pytest.skip(f"axis {axis} invalid for shape {shape}")
    x = make_array(shape, "float64", "jax")
    soft_out = sj.percentile(x, p, axis=axis, keepdims=keepdims, mode="hard")
    jax_out = jnp.percentile(x, p, axis=axis, keepdims=keepdims)
    common.assert_jax_parity(soft_out, jax_out, msg=f"percentile p={p}")


# ---------------------------------------------------------------------------
# JAX parity: top_k
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(4,), (2, 3)])
@pytest.mark.parametrize("axis", [-1])
@pytest.mark.parametrize("k", [1, 2])
def test_top_k_jax_parity(shape, axis, k):
    """sj.top_k(x, k, mode='hard') values must match jax.lax.top_k."""
    if k > shape[axis]:
        pytest.skip(f"k={k} exceeds axis size {shape[axis]}")
    x = make_array(shape, "float64", "jax")
    soft_vals, soft_idx = sj.top_k(x, k=k, axis=axis, mode="hard")
    jax_vals, _ = jax.lax.top_k(x, k=k)
    common.assert_jax_parity(soft_vals, jax_vals, msg="top_k values")


# ---------------------------------------------------------------------------
# SoftIndex shape validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(4,), (2, 3)])
@pytest.mark.parametrize("axis", [-1, 0])
@pytest.mark.parametrize("keepdims", KEEPDIMS)
def test_argmin_shape(shape, axis, keepdims):
    """sj.argmin SoftIndex shape = jnp.argmin shape + [n]."""
    if not _valid_axis(shape, axis):
        pytest.skip(f"axis {axis} invalid for shape {shape}")
    x = make_array(shape, "float64", "jax")
    soft_out = sj.argmin(x, axis=axis, keepdims=keepdims, mode="hard")
    jax_idx = jnp.argmin(x, axis=axis, keepdims=keepdims)
    assert soft_out.shape[:-1] == jax_idx.shape
    assert soft_out.shape[-1] == x.shape[axis]


@pytest.mark.parametrize("shape", [(5,), (2, 3)])
@pytest.mark.parametrize("axis", [-1, 0])
@pytest.mark.parametrize("keepdims", KEEPDIMS)
def test_argmedian_shape(shape, axis, keepdims):
    """sj.argmedian SoftIndex shape = reduced shape + [n]."""
    if not _valid_axis(shape, axis):
        pytest.skip(f"axis {axis} invalid for shape {shape}")
    x = make_array(shape, "float64", "jax")
    soft_out = sj.argmedian(x, axis=axis, keepdims=keepdims, mode="hard")
    median_out = sj.median(x, axis=axis, keepdims=keepdims, mode="hard")
    assert soft_out.shape[:-1] == median_out.shape
    assert soft_out.shape[-1] == x.shape[axis]


@pytest.mark.parametrize("shape", [(5,), (2, 3)])
@pytest.mark.parametrize("axis", [-1, 0])
@pytest.mark.parametrize("keepdims", KEEPDIMS)
@pytest.mark.parametrize("q", [0.25, 0.75])
def test_argquantile_shape(shape, axis, keepdims, q):
    """sj.argquantile SoftIndex shape = jnp.quantile shape + [n]."""
    if not _valid_axis(shape, axis):
        pytest.skip(f"axis {axis} invalid for shape {shape}")
    x = make_array(shape, "float64", "jax")
    soft_out = sj.argquantile(x, q, axis=axis, keepdims=keepdims, mode="hard")
    jax_out = jnp.quantile(x, q, axis=axis, keepdims=keepdims)
    assert soft_out.shape[:-1] == jax_out.shape
    assert soft_out.shape[-1] == x.shape[axis]


@pytest.mark.parametrize("shape", [(5,), (2, 3)])
@pytest.mark.parametrize("axis", [-1, 0])
@pytest.mark.parametrize("keepdims", KEEPDIMS)
@pytest.mark.parametrize("p", [25.0, 75.0])
def test_argpercentile_shape(shape, axis, keepdims, p):
    """sj.argpercentile SoftIndex shape = jnp.percentile shape + [n]."""
    if not _valid_axis(shape, axis):
        pytest.skip(f"axis {axis} invalid for shape {shape}")
    x = make_array(shape, "float64", "jax")
    soft_out = sj.argpercentile(x, p, axis=axis, keepdims=keepdims, mode="hard")
    jax_out = jnp.percentile(x, p, axis=axis, keepdims=keepdims)
    assert soft_out.shape[:-1] == jax_out.shape
    assert soft_out.shape[-1] == x.shape[axis]


# ---------------------------------------------------------------------------
# Value output shape validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fn_name", ["sort", "rank"])
@pytest.mark.parametrize("shape", [(4,), (2, 3)])
@pytest.mark.parametrize("axis", [-1, 0])
def test_sort_rank_output_shape(fn_name, shape, axis):
    """sort/rank output shape must match input shape."""
    if not _valid_axis(shape, axis):
        pytest.skip(f"axis {axis} invalid for shape {shape}")
    x = make_array(shape, "float64", "jax")
    fn = getattr(sj, fn_name)
    out = fn(x, axis=axis, mode="hard")
    assert out.shape == x.shape


@pytest.mark.parametrize(
    "fn_name,jnp_fn_name",
    [("median", "median"), ("quantile", "quantile"), ("percentile", "percentile")],
)
@pytest.mark.parametrize("keepdims", KEEPDIMS)
@pytest.mark.parametrize("shape", [(5,), (2, 3)])
@pytest.mark.parametrize("axis", [-1, 0])
def test_value_reduction_output_shape(fn_name, jnp_fn_name, keepdims, shape, axis):
    """Value reduction output shape must match jnp equivalent."""
    if not _valid_axis(shape, axis):
        pytest.skip(f"axis {axis} invalid for shape {shape}")
    x = make_array(shape, "float64", "jax")
    fn = getattr(sj, fn_name)
    jnp_fn = getattr(jnp, jnp_fn_name)
    if fn_name in ("quantile", "percentile"):
        q = 0.5 if fn_name == "quantile" else 50.0
        out = fn(x, q, axis=axis, keepdims=keepdims, mode="hard")
        expected = jnp_fn(x, q, axis=axis, keepdims=keepdims)
    else:
        out = fn(x, axis=axis, keepdims=keepdims, mode="hard")
        expected = jnp_fn(x, axis=axis, keepdims=keepdims)
    assert out.shape == expected.shape


@pytest.mark.parametrize("fn_name", ["max", "min"])
@pytest.mark.parametrize("keepdims", KEEPDIMS)
def test_max_min_output_shape(fn_name, keepdims):
    """Verify output shape for max/min matches jnp equivalent."""
    x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    fn = getattr(sj, fn_name)
    jnp_fn = getattr(jnp, fn_name)
    for axis in [None, 0, 1, -1]:
        out = fn(x, axis=axis, keepdims=keepdims, mode="hard")
        expected = jnp_fn(x, axis=axis, keepdims=keepdims)
        assert out.shape == expected.shape


@pytest.mark.parametrize("shape", [(4,), (2, 3)])
@pytest.mark.parametrize("k", [1, 2])
def test_top_k_output_shapes(shape, k):
    """top_k values shape matches jax.lax.top_k; SoftIndex shape = index shape + [n]."""
    x = make_array(shape, "float64", "jax")
    if k > shape[-1]:
        pytest.skip(f"k={k} exceeds last axis size {shape[-1]}")
    soft_vals, soft_idx = sj.top_k(x, k=k, mode="hard")
    jax_vals, jax_idx = jax.lax.top_k(x, k=k)
    assert soft_vals.shape == jax_vals.shape
    assert soft_idx.shape[:-1] == jax_idx.shape
    assert soft_idx.shape[-1] == x.shape[-1]


# ---------------------------------------------------------------------------
# Gradient finiteness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fn_name", ["max", "min", "sort", "rank", "median"])
@pytest.mark.parametrize("mode", ["smooth", "c0", "c1", "c2"])
def test_arraywise_gradient_finite(fn_name, mode):
    """Gradients through soft arraywise ops must be finite."""
    x = common.gradient_input((4,), jnp.float32)
    fn = getattr(sj, fn_name)

    def loss(z):
        return jnp.sum(fn(z, axis=-1, mode=mode, softness=1.0))

    grad = jax.grad(loss)(x)
    common.assert_finite(grad, msg=f"{fn_name} mode={mode}")


@pytest.mark.parametrize("mode", ["smooth", "c0", "c1", "c2"])
def test_quantile_gradient_finite(mode):
    """Gradients through soft quantile must be finite."""
    x = common.gradient_input((5,), jnp.float32)

    def loss(z):
        return jnp.sum(sj.quantile(z, 0.5, axis=-1, mode=mode, softness=1.0))

    grad = jax.grad(loss)(x)
    common.assert_finite(grad, msg=f"quantile mode={mode}")


@pytest.mark.parametrize("mode", ["smooth", "c0", "c1", "c2"])
def test_percentile_gradient_finite(mode):
    """Gradients through soft percentile must be finite."""
    x = common.gradient_input((5,), jnp.float32)

    def loss(z):
        return jnp.sum(sj.percentile(z, 50.0, axis=-1, mode=mode, softness=1.0))

    grad = jax.grad(loss)(x)
    common.assert_finite(grad, msg=f"percentile mode={mode}")


# ---------------------------------------------------------------------------
# fast_soft_sort + smooth mode (entropic PAV)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fn_name", ["max", "min", "sort", "rank", "median"])
def test_fast_soft_sort_smooth_mode(fn_name):
    """fast_soft_sort+smooth mode works (entropic PAV)."""
    x = make_array((4,), "float64", "jax")
    fn = getattr(sj, fn_name)
    out = fn(x, axis=-1, mode="smooth", method="fast_soft_sort", softness=1.0)
    assert not jnp.any(jnp.isnan(out)), f"NaN in {fn_name} fast_soft_sort+smooth"
    common.assert_finite(out, msg=f"{fn_name} fast_soft_sort+smooth")


@pytest.mark.parametrize("fn_name", ["max", "min", "sort", "rank", "median"])
def test_fast_soft_sort_smooth_gradient_finite(fn_name):
    """Gradients through fast_soft_sort+smooth mode must be finite."""
    x = common.gradient_input((4,), jnp.float32)
    fn = getattr(sj, fn_name)

    def loss(z):
        return jnp.sum(
            fn(z, axis=-1, mode="smooth", method="fast_soft_sort", softness=1.0)
        )

    grad = jax.grad(loss)(x)
    common.assert_finite(grad, msg=f"{fn_name} fast_soft_sort+smooth gradient")


# ---------------------------------------------------------------------------
# smooth_sort method (ESP + LBFGS)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fn_name", ["max", "min", "sort", "rank", "median"])
def test_smooth_sort_mode(fn_name):
    """smooth_sort method works."""
    x = make_array((4,), "float64", "jax")
    fn = getattr(sj, fn_name)
    out = fn(x, axis=-1, mode="smooth", method="smooth_sort", softness=1.0)
    assert not jnp.any(jnp.isnan(out)), f"NaN in {fn_name} smooth_sort"
    common.assert_finite(out, msg=f"{fn_name} smooth_sort")


@pytest.mark.parametrize("fn_name", ["max", "min", "sort", "rank", "median"])
def test_smooth_sort_gradient_finite(fn_name):
    """Gradients through smooth_sort method must be finite."""
    x = common.gradient_input((4,), jnp.float32)
    fn = getattr(sj, fn_name)

    def loss(z):
        return jnp.sum(
            fn(z, axis=-1, mode="smooth", method="smooth_sort", softness=1.0)
        )

    grad = jax.grad(loss)(x)
    common.assert_finite(grad, msg=f"{fn_name} smooth_sort gradient")


# ---------------------------------------------------------------------------
# Gradient finiteness for arg-producing ops
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fn_name", ["argmax", "argmin", "argsort", "argmedian"])
@pytest.mark.parametrize("mode", ["smooth", "c0", "c1", "c2"])
def test_arg_gradient_finite(fn_name, mode):
    """Gradients through arg ops (via weighted sum) must be finite."""
    x = common.gradient_input((4,), jnp.float32)
    weights = jnp.arange(4.0)
    fn = getattr(sj, fn_name)

    def loss(z):
        out = fn(z, axis=-1, mode=mode, softness=1.0)
        return jnp.sum(out * weights)

    grad = jax.grad(loss)(x)
    common.assert_finite(grad, msg=f"{fn_name} mode={mode}")


@pytest.mark.parametrize(
    "fn_name, q", [("argquantile", 0.25), ("argquantile", 0.75), ("argpercentile", 25.0), ("argpercentile", 75.0)]
)
@pytest.mark.parametrize("mode", ["smooth", "c0", "c1", "c2"])
def test_argquantile_argpercentile_gradient_finite(fn_name, q, mode):
    """Gradients through argquantile/argpercentile (via weighted sum) must be finite."""
    x = common.gradient_input((5,), jnp.float32)
    weights = jnp.arange(5.0)
    fn = getattr(sj, fn_name)

    def loss(z):
        out = fn(z, q, axis=-1, mode=mode, softness=1.0)
        return jnp.sum(out * weights)

    grad = jax.grad(loss)(x)
    common.assert_finite(grad, msg=f"{fn_name} q={q} mode={mode}")


# ---------------------------------------------------------------------------
# Device placement tests (CPU + GPU)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fn_name", ["max", "min", "sort", "rank", "median"])
def test_device_value_ops(device, fn_name):
    """Value arraywise ops produce correct results on each device."""
    x = common.make_array((4,), "float64", "jax", device=device)
    fn = getattr(sj, fn_name)
    out_soft = fn(x, axis=-1, mode="smooth", softness=1.0)
    out_hard = fn(x, axis=-1, mode="hard")
    common.assert_finite(out_soft, msg=f"{fn_name} on {device}")
    common.assert_finite(out_hard, msg=f"{fn_name} hard on {device}")


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fn_name", ["argmax", "argmin", "argsort", "argmedian"])
def test_device_arg_ops(device, fn_name):
    """Arg arraywise ops produce correct results on each device."""
    x = common.make_array((4,), "float64", "jax", device=device)
    fn = getattr(sj, fn_name)
    out = fn(x, axis=-1, mode="smooth", softness=1.0)
    common.assert_finite(out, msg=f"{fn_name} on {device}")
    common.assert_simplex(out, atol=common.TOLERANCE)


@pytest.mark.parametrize("device", DEVICES)
def test_device_gradient(device):
    """Gradients work on each device."""
    x = common.gradient_input((4,), jnp.float32, device=device)

    def loss(z):
        return jnp.sum(sj.sort(z, axis=-1, mode="smooth", softness=1.0))

    grad = jax.grad(loss)(x)
    common.assert_finite(grad, msg=f"sort gradient on {device}")


# ---------------------------------------------------------------------------
# Gradient vs finite differences (correctness, not just finiteness)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fn_name", ["max", "min", "sort", "rank", "median"])
@pytest.mark.parametrize("mode", ["smooth", "c0"])
def test_value_grad_vs_finite_diff(fn_name, mode):
    """Analytic gradient matches finite differences for value ops."""
    x = common.gradient_input((5,), jnp.float64)
    fn = getattr(sj, fn_name)

    def loss(z):
        return jnp.sum(fn(z, axis=-1, mode=mode, softness=1.0))

    common.assert_grad_matches_finite_diff(loss, x, msg=f"{fn_name} mode={mode}")


@pytest.mark.parametrize("fn_name", ["argmax", "argmin", "argsort", "argmedian"])
@pytest.mark.parametrize("mode", ["smooth", "c0"])
def test_arg_grad_vs_finite_diff(fn_name, mode):
    """Analytic gradient matches finite differences for arg ops."""
    x = common.gradient_input((5,), jnp.float64)
    weights = jnp.arange(5.0, dtype=jnp.float64)
    fn = getattr(sj, fn_name)

    def loss(z):
        out = fn(z, axis=-1, mode=mode, softness=1.0)
        return jnp.sum(out * weights)

    common.assert_grad_matches_finite_diff(loss, x, msg=f"{fn_name} mode={mode}")


@pytest.mark.parametrize("mode", ["smooth", "c0"])
def test_quantile_grad_vs_finite_diff(mode):
    """Analytic gradient matches finite differences for quantile."""
    x = common.gradient_input((5,), jnp.float64)

    def loss(z):
        return jnp.sum(sj.quantile(z, 0.5, axis=-1, mode=mode, softness=1.0))

    common.assert_grad_matches_finite_diff(loss, x, msg=f"quantile mode={mode}")


@pytest.mark.parametrize("mode", ["smooth", "c0"])
def test_top_k_grad_vs_finite_diff(mode):
    """Analytic gradient matches finite differences for top_k."""
    x = common.gradient_input((5,), jnp.float64)

    def loss(z):
        vals, _ = sj.top_k(z, k=3, axis=-1, mode=mode, softness=1.0)
        return jnp.sum(vals)

    common.assert_grad_matches_finite_diff(loss, x, msg=f"top_k mode={mode}")


# ---------------------------------------------------------------------------
# Error path tests (issue 15)
# ---------------------------------------------------------------------------


def test_invalid_mode():
    x = jnp.array([3.0, 1.0, 2.0])
    with pytest.raises((ValueError, KeyError)):
        sj.sort(x, mode="invalid")
    with pytest.raises((ValueError, KeyError)):
        sj.argmax(x, mode="invalid")


def test_invalid_method():
    x = jnp.array([3.0, 1.0, 2.0])
    with pytest.raises(ValueError):
        sj.sort(x, method="invalid")
    with pytest.raises(ValueError):
        sj.argsort(x, method="invalid")


def test_top_k_k_validation():
    x = jnp.array([3.0, 1.0, 2.0])
    with pytest.raises(ValueError, match="k must be positive"):
        sj.top_k(x, k=0)
    with pytest.raises(ValueError, match="k must be positive"):
        sj.top_k(x, k=-1)


def test_dynamic_slice_in_dim_validation():
    x = jnp.array([3.0, 1.0, 2.0])
    soft_start = jnp.array([0.5, 0.5, 0.0])
    with pytest.raises(ValueError, match="slice_size"):
        sj.dynamic_slice_in_dim(x, soft_start_index=soft_start, slice_size=0, axis=0)
    with pytest.raises(ValueError, match="slice_size"):
        sj.dynamic_slice_in_dim(x, soft_start_index=soft_start, slice_size=4, axis=0)


def test_dynamic_slice_validation():
    x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError, match="len"):
        sj.dynamic_slice(
            x, soft_start_indices=[jnp.array([0.5, 0.5])], slice_sizes=[1, 1]
        )


# ---------------------------------------------------------------------------
# Single-element (n=1) edge cases (issue 17)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fn_name", ["max", "min", "sort", "rank", "median"])
def test_single_element_value_ops(fn_name):
    """Value ops work correctly with a single-element input."""
    x = jnp.array([42.0])
    fn = getattr(sj, fn_name)
    out = fn(x, axis=0, mode="smooth", softness=1.0)
    common.assert_finite(out, msg=f"{fn_name} n=1")


@pytest.mark.parametrize("fn_name", ["argmax", "argmin", "argsort", "argmedian"])
def test_single_element_arg_ops(fn_name):
    """Arg ops return [1.0] for single-element input."""
    x = jnp.array([42.0])
    fn = getattr(sj, fn_name)
    out = fn(x, axis=0, mode="smooth", softness=1.0)
    common.assert_finite(out, msg=f"{fn_name} n=1")
    common.assert_simplex(out, atol=1e-4)


def test_single_element_top_k():
    x = jnp.array([42.0])
    vals, idx = sj.top_k(x, k=1, axis=0, mode="smooth", softness=1.0)
    common.assert_finite(vals, msg="top_k n=1 values")
    common.assert_finite(idx, msg="top_k n=1 indices")


def test_single_element_quantile():
    x = jnp.array([42.0])
    out = sj.quantile(x, q=0.5, axis=0, mode="smooth", softness=1.0)
    common.assert_finite(out, msg="quantile n=1")


# ---------------------------------------------------------------------------
# return_log_probs parameter for SoftIndex outputs
# ---------------------------------------------------------------------------


def _assert_log_probs_match_probs(log_probs, probs, msg, tol=1e-6):
    assert not jnp.any(jnp.isnan(log_probs)), f"NaN in {msg}"
    common.assert_allclose(jnp.exp(log_probs), probs, tol=tol)


def _manual_log_prob_eps(probs, eps):
    probs = jnp.clip(probs, eps, 1.0)
    probs = probs / jnp.sum(probs, axis=-1, keepdims=True)
    return jnp.log(probs)


_LOG_PROB_INDEX_CASES = [
    (
        "argmax",
        lambda x, method, mode, **kwargs: sj.argmax(
            x, axis=-1, method=method, mode=mode, **kwargs
        ),
    ),
    (
        "argmin",
        lambda x, method, mode, **kwargs: sj.argmin(
            x, axis=-1, method=method, mode=mode, **kwargs
        ),
    ),
    (
        "argsort",
        lambda x, method, mode, **kwargs: sj.argsort(
            x, axis=-1, method=method, mode=mode, **kwargs
        ),
    ),
    (
        "argquantile",
        lambda x, method, mode, **kwargs: sj.argquantile(
            x, q=0.35, axis=-1, method=method, mode=mode, **kwargs
        ),
    ),
    (
        "argmedian",
        lambda x, method, mode, **kwargs: sj.argmedian(
            x, axis=-1, method=method, mode=mode, **kwargs
        ),
    ),
    (
        "argpercentile",
        lambda x, method, mode, **kwargs: sj.argpercentile(
            x, p=35.0, axis=-1, method=method, mode=mode, **kwargs
        ),
    ),
]


@pytest.mark.parametrize(
    "case_name, call",
    _LOG_PROB_INDEX_CASES,
    ids=[case_name for case_name, _ in _LOG_PROB_INDEX_CASES],
)
@pytest.mark.parametrize("method", ("softsort", "neuralsort", "sorting_network"))
@pytest.mark.parametrize("mode", ("hard", "smooth", "c0", "c1", "c2"))
def test_return_log_probs_softindex_outputs(case_name, call, method, mode):
    x = jnp.array([[0.2, 1.7, -0.5, 0.9]], dtype=jnp.float64)

    probs = call(x, method, mode, softness=1.0)
    log_probs = call(x, method, mode, softness=1.0, return_log_probs=True)

    _assert_log_probs_match_probs(log_probs, probs, f"{case_name}/{method}/{mode}")


@pytest.mark.parametrize(
    "case_name, call",
    [
        (
            "argmax",
            lambda x, mode, **kwargs: sj.argmax(
                x, axis=-1, method="ot", mode=mode, **kwargs
            ),
        ),
        (
            "argsort",
            lambda x, mode, **kwargs: sj.argsort(
                x, axis=-1, method="ot", mode=mode, **kwargs
            ),
        ),
        (
            "argquantile",
            lambda x, mode, **kwargs: sj.argquantile(
                x, q=0.35, axis=-1, method="ot", mode=mode, **kwargs
            ),
        ),
    ],
)
@pytest.mark.parametrize("mode", ("smooth", "c0", "c1", "c2"))
def test_return_log_probs_ot_softindex_outputs(case_name, call, mode):
    x = jnp.array([[0.2, 1.7, -0.5]], dtype=jnp.float64)

    probs = call(x, mode, softness=1.0)
    log_probs = call(x, mode, softness=1.0, return_log_probs=True)

    _assert_log_probs_match_probs(log_probs, probs, f"{case_name}/ot/{mode}", tol=1e-5)


@pytest.mark.parametrize("method", ("softsort", "neuralsort", "ot", "sorting_network"))
def test_return_log_probs_smooth_outputs_have_finite_gradients(method):
    x = jnp.array([[0.2, 1.7, -0.5, 0.9]], dtype=jnp.float64)

    def loss(z):
        log_probs = sj.argsort(
            z,
            axis=-1,
            mode="smooth",
            method=method,
            softness=1.0,
            return_log_probs=True,
        )
        weights = jnp.arange(1, log_probs.shape[-1] + 1, dtype=log_probs.dtype)
        return jnp.sum(log_probs * weights)

    log_probs = sj.argsort(
        x,
        axis=-1,
        mode="smooth",
        method=method,
        softness=1.0,
        return_log_probs=True,
    )
    grad = jax.grad(loss)(x)

    common.assert_finite(log_probs, msg=f"{method} smooth log_probs")
    common.assert_finite(grad, msg=f"{method} smooth log_probs gradient")


def test_return_log_probs_smooth_simplex_avoids_softmax_underflow():
    x = jnp.array([1000.0, 0.0, -1000.0], dtype=jnp.float32)

    probs = sj.argmax(
        x,
        axis=0,
        mode="smooth",
        method="softsort",
        softness=1.0,
        standardize=False,
    )
    log_probs = sj.argmax(
        x,
        axis=0,
        mode="smooth",
        method="softsort",
        softness=1.0,
        standardize=False,
        return_log_probs=True,
    )

    assert jnp.isneginf(jnp.log(probs)).any()
    common.assert_finite(log_probs, msg="smooth simplex log_probs")
    _assert_log_probs_match_probs(log_probs, probs, "argmax smooth softsort")


def test_return_log_probs_sparse_mode_nan_to_num_does_not_fix_gradients():
    x = jnp.array([0.2, 1.7, -0.5, 0.9], dtype=jnp.float64)
    weights = jnp.arange(1, 5, dtype=jnp.float64)

    def loss(z):
        log_probs = sj.argmax(
            z,
            axis=0,
            mode="c0",
            softness=1.0,
            return_log_probs=True,
        )
        return jnp.sum(jnp.nan_to_num(log_probs, neginf=-1e9) * weights)

    grad = jax.grad(loss)(x)

    assert not jnp.all(jnp.isfinite(grad))


@pytest.mark.parametrize(
    "case_name, call",
    _LOG_PROB_INDEX_CASES,
    ids=[case_name for case_name, _ in _LOG_PROB_INDEX_CASES],
)
def test_return_log_probs_log_prob_eps_matches_manual_floor_renormalize(
    case_name, call
):
    x = jnp.array([[0.2, 1.7, -0.5, 0.9]], dtype=jnp.float64)
    eps = 1e-3

    probs = call(x, "softsort", "c0", softness=0.01)
    log_probs = call(
        x,
        "softsort",
        "c0",
        softness=0.01,
        return_log_probs=True,
        log_prob_eps=eps,
    )
    expected = _manual_log_prob_eps(probs, eps)

    common.assert_finite(log_probs, msg=f"{case_name} log_prob_eps")
    common.assert_allclose(log_probs, expected, tol=1e-6)
    common.assert_allclose(
        jnp.sum(jnp.exp(log_probs), axis=-1),
        jnp.ones(log_probs.shape[:-1], dtype=log_probs.dtype),
        tol=1e-6,
    )


def test_return_log_probs_log_prob_eps_matches_manual_ot():
    x = jnp.array([[0.2, 1.7, -0.5]], dtype=jnp.float64)
    eps = 1e-3

    probs = sj.argsort(x, axis=-1, method="ot", mode="c1", softness=1.0)
    log_probs = sj.argsort(
        x,
        axis=-1,
        method="ot",
        mode="c1",
        softness=1.0,
        return_log_probs=True,
        log_prob_eps=eps,
    )
    expected = _manual_log_prob_eps(probs, eps)

    common.assert_finite(log_probs, msg="ot argsort log_prob_eps")
    common.assert_allclose(log_probs, expected, tol=1e-5)


@pytest.mark.parametrize("method", ("softsort", "neuralsort"))
@pytest.mark.parametrize("mode", ("c0", "c1", "c2"))
def test_return_log_probs_log_prob_eps_has_finite_sparse_gradients(method, mode):
    x = jnp.array([0.2, 1.7, -0.5, 0.9], dtype=jnp.float64)
    weights = jnp.arange(1, 5, dtype=jnp.float64)

    def loss(z):
        log_probs = sj.argmax(
            z,
            axis=0,
            method=method,
            mode=mode,
            softness=0.01,
            return_log_probs=True,
            log_prob_eps=1e-12,
        )
        return jnp.sum(log_probs * weights)

    grad = jax.grad(loss)(x)

    common.assert_finite(grad, msg=f"{method}/{mode} log_prob_eps gradient")


def test_return_log_probs_log_prob_eps_requires_return_log_probs():
    x = jnp.array([0.2, 1.7, -0.5, 0.9], dtype=jnp.float64)

    with pytest.raises(ValueError, match="return_log_probs=True"):
        sj.argmax(x, axis=0, log_prob_eps=1e-3)

    with pytest.raises(ValueError, match="return_log_probs=True"):
        sj.top_k(x, k=2, axis=0, log_prob_eps=1e-3)


@pytest.mark.parametrize("eps", (0.0, -1.0, 1.5))
def test_return_log_probs_log_prob_eps_validates_range(eps):
    x = jnp.array([0.2, 1.7, -0.5, 0.9], dtype=jnp.float64)

    with pytest.raises(ValueError, match="log_prob_eps must be in"):
        sj.argmax(x, axis=0, return_log_probs=True, log_prob_eps=eps)


@pytest.mark.parametrize("method", ("softsort", "neuralsort", "ot"))
def test_return_log_probs_log_prob_eps_top_k_indices_and_values(method):
    x = jnp.array([[0.2, 1.7, -0.5, 0.9]], dtype=jnp.float64)
    eps = 1e-3

    values, probs = sj.top_k(
        x, k=2, axis=-1, method=method, mode="c0", softness=1.0
    )
    log_values, log_probs = sj.top_k(
        x,
        k=2,
        axis=-1,
        method=method,
        mode="c0",
        softness=1.0,
        return_log_probs=True,
        log_prob_eps=eps,
    )
    expected = _manual_log_prob_eps(probs, eps)

    common.assert_allclose(log_values, values, tol=1e-6)
    common.assert_finite(log_probs, msg=f"top_k/{method} log_prob_eps")
    common.assert_allclose(log_probs, expected, tol=1e-5)


def test_return_log_probs_log_prob_eps_vector_q_matches_manual():
    x = make_array((6,), "float64", "jax")
    eps = 1e-3

    probs = sj.argquantile(
        x, _VECTOR_Q, axis=0, method="softsort", mode="c0", softness=0.01
    )
    log_probs = sj.argquantile(
        x,
        _VECTOR_Q,
        axis=0,
        method="softsort",
        mode="c0",
        softness=0.01,
        return_log_probs=True,
        log_prob_eps=eps,
    )
    expected = _manual_log_prob_eps(probs, eps)

    common.assert_finite(log_probs, msg="argquantile vector q log_prob_eps")
    common.assert_allclose(log_probs, expected, tol=1e-6)


@pytest.mark.parametrize("mode", ("c0", "c1", "c2"))
def test_sparse_mode_safe_log_probabilities_have_finite_gradients(mode):
    x = jnp.array([0.2, 1.7, -0.5, 0.9], dtype=jnp.float64)
    weights = jnp.arange(1, 5, dtype=jnp.float64)

    def loss(z):
        safe_log_probs = sj.argmax(
            z,
            axis=0,
            mode=mode,
            softness=0.01,
            return_log_probs=True,
            log_prob_eps=1e-12,
        )
        return jnp.sum(safe_log_probs * weights)

    grad = jax.grad(loss)(x)

    common.assert_finite(grad, msg=f"{mode} safe log probability gradient")


def test_return_log_probs_hard_zeros_are_negative_infinity():
    x = jnp.array([3.0, 1.0, 2.0])

    probs = sj.argsort(x, axis=0, mode="hard")
    log_probs = sj.argsort(x, axis=0, mode="hard", return_log_probs=True)

    _assert_log_probs_match_probs(log_probs, probs, "argsort hard")
    assert jnp.all(log_probs[probs == 1] == 0)
    assert jnp.all(jnp.isneginf(log_probs[probs == 0]))


@pytest.mark.parametrize("method", ("softsort", "neuralsort", "ot"))
@pytest.mark.parametrize("mode", ("smooth", "c0"))
def test_return_log_probs_top_k_indices_and_values(method, mode):
    x = jnp.array([[0.2, 1.7, -0.5, 0.9]], dtype=jnp.float64)

    values, probs = sj.top_k(x, k=2, axis=-1, method=method, mode=mode, softness=1.0)
    log_values, log_probs = sj.top_k(
        x,
        k=2,
        axis=-1,
        method=method,
        mode=mode,
        softness=1.0,
        return_log_probs=True,
    )

    common.assert_allclose(log_values, values, tol=1e-6)
    _assert_log_probs_match_probs(log_probs, probs, f"top_k/{method}/{mode}", tol=1e-5)


@pytest.mark.parametrize("method", ("softsort", "neuralsort", "ot"))
def test_return_log_probs_top_k_preserves_value_gradients(method):
    x0 = jnp.array([[0.2, 1.7, -0.5, 0.9]], dtype=jnp.float64)

    def value_and_grad(return_log_probs):
        def loss(z):
            values, _ = sj.top_k(
                z,
                k=2,
                axis=-1,
                method=method,
                mode="smooth",
                softness=1.0,
                standardize=False,
                return_log_probs=return_log_probs,
            )
            return jnp.sum(values)

        return loss(x0), jax.grad(loss)(x0)

    values, grad = value_and_grad(False)
    log_values, log_grad = value_and_grad(True)

    common.assert_allclose(log_values, values, tol=1e-6)
    common.assert_allclose(log_grad, grad, tol=1e-5)


@pytest.mark.parametrize("method", ("fast_soft_sort", "smooth_sort", "sorting_network"))
def test_return_log_probs_top_k_preserves_none_indices(method):
    x = jnp.array([[0.2, 1.7, -0.5, 0.9]], dtype=jnp.float64)

    values, idx = sj.top_k(x, k=2, axis=-1, method=method, mode="smooth", softness=1.0)
    log_values, log_idx = sj.top_k(
        x,
        k=2,
        axis=-1,
        method=method,
        mode="smooth",
        softness=1.0,
        return_log_probs=True,
    )

    assert idx is None
    assert log_idx is None
    common.assert_allclose(log_values, values, tol=1e-6)


@pytest.mark.parametrize("method", ("softsort", "neuralsort", "ot", "sorting_network"))
def test_argquantile_vector_q_return_log_probs(method):
    x = make_array((6,), "float64", "jax")
    q_vec = _VECTOR_Q

    probs = sj.argquantile(
        x, q_vec, axis=0, mode="smooth", method=method, softness=1.0
    )
    log_probs = sj.argquantile(
        x,
        q_vec,
        axis=0,
        mode="smooth",
        method=method,
        softness=1.0,
        return_log_probs=True,
    )

    _assert_log_probs_match_probs(log_probs, probs, f"argquantile vector q/{method}")


@pytest.mark.parametrize("keepdims", (False, True))
def test_return_log_probs_axis_none_shapes(keepdims):
    x = jnp.array([[0.2, 1.7], [-0.5, 0.9]], dtype=jnp.float64)

    argmax_probs = sj.argmax(x, axis=None, keepdims=keepdims)
    argmax_log_probs = sj.argmax(
        x, axis=None, keepdims=keepdims, return_log_probs=True
    )
    _assert_log_probs_match_probs(argmax_log_probs, argmax_probs, "argmax axis=None")

    argquantile_probs = sj.argquantile(x, q=0.35, axis=None, keepdims=keepdims)
    argquantile_log_probs = sj.argquantile(
        x, q=0.35, axis=None, keepdims=keepdims, return_log_probs=True
    )
    _assert_log_probs_match_probs(
        argquantile_log_probs, argquantile_probs, "argquantile axis=None"
    )
    assert argmax_log_probs.shape == argmax_probs.shape
    assert argquantile_log_probs.shape == argquantile_probs.shape


@pytest.mark.parametrize(
    "case_name, st_call, hard_call",
    [
        (
            "argmax_st",
            lambda x, mode: sj.argmax_st(
                x, axis=0, mode=mode, return_log_probs=True
            ),
            lambda x: sj.argmax(x, axis=0, mode="hard"),
        ),
        (
            "argsort_st",
            lambda x, mode: sj.argsort_st(
                x, axis=0, mode=mode, return_log_probs=True
            ),
            lambda x: sj.argsort(x, axis=0, mode="hard"),
        ),
        (
            "argquantile_st",
            lambda x, mode: sj.argquantile_st(
                x, q=0.35, axis=0, mode=mode, return_log_probs=True
            ),
            lambda x: sj.argquantile(x, q=0.35, axis=0, mode="hard"),
        ),
        (
            "argpercentile_st",
            lambda x, mode: sj.argpercentile_st(
                x, p=35.0, axis=0, mode=mode, return_log_probs=True
            ),
            lambda x: sj.argpercentile(x, p=35.0, axis=0, mode="hard"),
        ),
    ],
)
@pytest.mark.parametrize("mode", ("smooth", "c0", "c1", "c2"))
def test_return_log_probs_st_outputs_handle_negative_infinity(
    case_name, st_call, hard_call, mode
):
    x = jnp.array([0.2, 1.7, -0.5, 0.9], dtype=jnp.float64)

    log_probs = st_call(x, mode)

    assert not jnp.any(jnp.isnan(log_probs)), f"NaN in {case_name}/{mode}"
    _assert_log_probs_match_probs(log_probs, hard_call(x), f"{case_name}/{mode}")


@pytest.mark.parametrize("mode", ("smooth", "c0", "c1", "c2"))
def test_return_log_probs_top_k_st_handles_negative_infinity(mode):
    x = jnp.array([0.2, 1.7, -0.5, 0.9], dtype=jnp.float64)

    _, log_probs = sj.top_k_st(x, k=2, axis=0, mode=mode, return_log_probs=True)
    _, hard_probs = sj.top_k(x, k=2, axis=0, mode="hard")

    assert not jnp.any(jnp.isnan(log_probs)), f"NaN in top_k_st/{mode}"
    _assert_log_probs_match_probs(log_probs, hard_probs, f"top_k_st/{mode}")


# ---------------------------------------------------------------------------
# Vector q support for quantile / argquantile
# ---------------------------------------------------------------------------

_VECTOR_Q = jnp.array([0.25, 0.5, 0.75])
_ARGQUANTILE_METHODS = ["neuralsort", "softsort", "ot", "sorting_network"]
_QUANTILE_METHODS = _ARGQUANTILE_METHODS + ["fast_soft_sort", "smooth_sort"]


@pytest.mark.parametrize("shape", [(5,), (2, 4)])
@pytest.mark.parametrize("axis", [-1, 0])
@pytest.mark.parametrize("keepdims", KEEPDIMS)
@pytest.mark.parametrize("method", _ARGQUANTILE_METHODS)
def test_argquantile_vector_q_shape(shape, axis, keepdims, method):
    """argquantile with vector q: output shape matches jnp.quantile shape + [n]."""
    if not _valid_axis(shape, axis):
        pytest.skip(f"axis {axis} invalid for shape {shape}")
    x = make_array(shape, "float64", "jax")
    out = sj.argquantile(
        x, _VECTOR_Q, axis=axis, keepdims=keepdims, mode="hard", method=method,
    )
    jax_out = jnp.quantile(x, _VECTOR_Q, axis=axis, keepdims=keepdims)
    assert out.shape[:-1] == jax_out.shape, (
        f"shape mismatch: {out.shape[:-1]} vs {jax_out.shape}"
    )
    assert out.shape[-1] == x.shape[axis]


@pytest.mark.parametrize("shape", [(5,), (2, 4)])
@pytest.mark.parametrize("axis", [-1, 0])
@pytest.mark.parametrize("keepdims", KEEPDIMS)
@pytest.mark.parametrize("method", _QUANTILE_METHODS)
def test_quantile_vector_q_shape(shape, axis, keepdims, method):
    """quantile with vector q: output shape matches jnp.quantile."""
    _skip_unsupported(method, "smooth")
    if not _valid_axis(shape, axis):
        pytest.skip(f"axis {axis} invalid for shape {shape}")
    x = make_array(shape, "float64", "jax")
    out = sj.quantile(
        x, _VECTOR_Q, axis=axis, keepdims=keepdims, mode="hard", method=method,
    )
    jax_out = jnp.quantile(x, _VECTOR_Q, axis=axis, keepdims=keepdims)
    assert out.shape == jax_out.shape, f"shape mismatch: {out.shape} vs {jax_out.shape}"


@pytest.mark.parametrize("method", _ARGQUANTILE_METHODS)
def test_argquantile_vector_q_consistency(method):
    """Vector-q result matches stacked scalar-q results."""
    x = make_array((6,), "float64", "jax")
    q_vec = _VECTOR_Q
    vec_out = sj.argquantile(x, q_vec, axis=0, mode="smooth", method=method, softness=1.0)
    for i, qi in enumerate(q_vec):
        scalar_out = sj.argquantile(
            x, float(qi), axis=0, mode="smooth", method=method, softness=1.0,
        )
        np.testing.assert_allclose(
            np.array(vec_out[i]), np.array(scalar_out), atol=1e-6,
            err_msg=f"argquantile vector vs scalar mismatch at q={qi}, method={method}",
        )


@pytest.mark.parametrize("method", _QUANTILE_METHODS)
def test_quantile_vector_q_consistency(method):
    """Vector-q result matches stacked scalar-q results."""
    _skip_unsupported(method, "smooth")
    x = make_array((6,), "float64", "jax")
    q_vec = _VECTOR_Q
    vec_out = sj.quantile(x, q_vec, axis=0, mode="smooth", method=method, softness=1.0)
    for i, qi in enumerate(q_vec):
        scalar_out = sj.quantile(
            x, float(qi), axis=0, mode="smooth", method=method, softness=1.0,
        )
        np.testing.assert_allclose(
            np.array(vec_out[i]), np.array(scalar_out), atol=1e-6,
            err_msg=f"quantile vector vs scalar mismatch at q={qi}, method={method}",
        )


@pytest.mark.parametrize("method", _ARGQUANTILE_METHODS)
def test_argquantile_vector_q_simplex(method):
    """argquantile with vector q: each row is a valid SoftIndex."""
    x = make_array((6,), "float64", "jax")
    out = sj.argquantile(
        x, _VECTOR_Q, axis=0, mode="smooth", method=method, softness=1.0,
    )
    for i in range(len(_VECTOR_Q)):
        common.assert_simplex(out[i], atol=common.TOLERANCE)


def test_quantile_vector_q_monotonicity():
    """quantile values should be non-decreasing in q."""
    x = make_array((8,), "float64", "jax")
    q_fine = jnp.linspace(0.0, 1.0, 5)
    out = sj.quantile(x, q_fine, axis=0, mode="smooth", softness=1.0)
    diffs = jnp.diff(out)
    assert jnp.all(diffs >= -1e-6), "quantile values should be non-decreasing in q"


@pytest.mark.parametrize("shape", [(5,), (2, 4)])
@pytest.mark.parametrize("keepdims", KEEPDIMS)
def test_quantile_vector_q_hard_parity(shape, keepdims):
    """mode='hard' with vector q matches jnp.quantile."""
    x = make_array(shape, "float64", "jax")
    q_vec = _VECTOR_Q
    soft_out = sj.quantile(x, q_vec, axis=-1, keepdims=keepdims, mode="hard")
    jax_out = jnp.quantile(x, q_vec, axis=-1, keepdims=keepdims)
    common.assert_jax_parity(soft_out, jax_out, msg="quantile vector q hard parity")


def test_quantile_vector_q_axis_none():
    """Vector q with axis=None flattens input before computing quantiles."""
    x = jnp.array([[3.0, 1.0], [4.0, 2.0]])
    q_vec = jnp.array([0.25, 0.75])
    out = sj.quantile(x, q_vec, axis=None, mode="smooth", softness=1.0)
    assert out.shape == (2,), f"expected (2,) for axis=None, got {out.shape}"
    out_hard = sj.quantile(x, q_vec, axis=None, mode="hard")
    jax_out = jnp.quantile(x, q_vec, axis=None)
    common.assert_jax_parity(out_hard, jax_out, msg="quantile vector q axis=None")


def test_argquantile_vector_q_axis_none():
    """argquantile with vector q and axis=None flattens input."""
    x = jnp.array([[3.0, 1.0], [4.0, 2.0]])
    q_vec = jnp.array([0.25, 0.75])
    out = sj.argquantile(x, q_vec, axis=None, mode="hard")
    # jnp.quantile with axis=None flattens to (4,), so shape should be (2, [4])
    jax_out = jnp.quantile(x, q_vec, axis=None)
    assert out.shape[:-1] == jax_out.shape
    assert out.shape[-1] == x.size


def test_argquantile_vector_q_2d_rejection():
    """2D q should raise ValueError."""
    x = jnp.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="q must be scalar or 1-D"):
        sj.argquantile(x, jnp.array([[0.25, 0.75]]), axis=0)


def test_quantile_vector_q_2d_rejection():
    """2D q should raise ValueError."""
    x = jnp.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="q must be scalar or 1-D"):
        sj.quantile(x, jnp.array([[0.25, 0.75]]), axis=0)


def test_quantile_vector_q_gradient():
    """Gradients through vector-q quantile must be finite."""
    x = common.gradient_input((6,), jnp.float32)
    q_vec = jnp.array([0.25, 0.5, 0.75])

    def loss(z):
        return jnp.sum(sj.quantile(z, q_vec, axis=-1, mode="smooth", softness=1.0))

    grad = jax.grad(loss)(x)
    common.assert_finite(grad, msg="quantile vector q gradient")


def test_argquantile_vector_q_gradient():
    """Gradients through vector-q argquantile (via weighted sum) must be finite."""
    x = common.gradient_input((6,), jnp.float32)
    weights = jnp.arange(6.0)
    q_vec = jnp.array([0.25, 0.75])

    def loss(z):
        out = sj.argquantile(z, q_vec, axis=-1, mode="smooth", softness=1.0)
        return jnp.sum(out * weights)

    grad = jax.grad(loss)(x)
    common.assert_finite(grad, msg="argquantile vector q gradient")


# ---------------------------------------------------------------------------
# NaN input propagation (issue 18)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fn_name", ["sort", "max", "min", "rank"])
def test_nan_propagation(fn_name):
    """NaN inputs should propagate through soft operators."""
    x = jnp.array([1.0, float("nan"), 3.0])
    fn = getattr(sj, fn_name)
    out = fn(x, axis=0, mode="c0", softness=1.0)
    assert jnp.any(jnp.isnan(out)), f"NaN should propagate through {fn_name}"
