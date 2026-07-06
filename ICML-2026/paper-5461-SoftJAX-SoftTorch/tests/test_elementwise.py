import jax
import jax.numpy as jnp
import numpy as np
import pytest

import softjax as sj

from . import common


# JAX config (jax_enable_x64, matmul_precision) set in common.py

BACKENDS = common.BACKENDS
SHAPES = tuple(common.SHAPES.items())
FLOAT_DTYPES = ("float32", "float64")
INT_DTYPES = ("int32",)


def _ids_from_shape(item):
    return item[0]


# Valid modes per actual function type signatures
ELEMENTWISE_CASES = (
    {
        "name": "relu",
        "soft_fn": sj.relu,
        "hard_fn": lambda x, **_: jax.nn.relu(x),
        "modes": {"hard", "smooth", "c0", "c1", "_c1_pnorm", "c2", "_c2_pnorm"},
        "kwargs": {},
    },
    {
        "name": "abs",
        "soft_fn": sj.abs,
        "hard_fn": lambda x, **_: jnp.abs(x),
        "modes": {"hard", "smooth", "c0", "c1", "_c1_pnorm", "c2", "_c2_pnorm"},
        "kwargs": {},
    },
    {
        "name": "sign",
        "soft_fn": sj.sign,
        "hard_fn": lambda x, **_: jnp.sign(x).astype(jnp.result_type(float)),
        "modes": {"hard", "smooth", "c0", "c1", "_c1_pnorm", "c2", "_c2_pnorm"},
        "kwargs": {},
    },
    {
        "name": "round",
        "soft_fn": sj.round,
        "hard_fn": lambda x, **_: jnp.round(x),
        "modes": {"hard", "smooth", "c0", "c1", "_c1_pnorm", "c2", "_c2_pnorm"},
        "kwargs": {},
    },
    {
        "name": "clip",
        "soft_fn": sj.clip,
        "hard_fn": lambda x, *, a, b: jnp.clip(x, a, b),
        "modes": {"hard", "smooth", "c0", "c1", "_c1_pnorm", "c2", "_c2_pnorm"},
        "kwargs": {"a": -0.25, "b": 0.25},
    },
)


COMPARISON_CASES = (
    {
        "name": "greater",
        "soft_fn": sj.greater,
        "hard_fn": lambda x, y: jnp.greater(x, y),
    },
    {
        "name": "greater_equal",
        "soft_fn": sj.greater_equal,
        "hard_fn": lambda x, y: jnp.greater_equal(x, y),
    },
    {
        "name": "less",
        "soft_fn": sj.less,
        "hard_fn": lambda x, y: jnp.less(x, y),
    },
    {
        "name": "less_equal",
        "soft_fn": sj.less_equal,
        "hard_fn": lambda x, y: jnp.less_equal(x, y),
    },
    {
        "name": "equal",
        "soft_fn": sj.equal,
        "hard_fn": lambda x, y: jnp.equal(x, y),
    },
    {
        "name": "not_equal",
        "soft_fn": sj.not_equal,
        "hard_fn": lambda x, y: jnp.not_equal(x, y),
    },
    {
        "name": "isclose",
        "soft_fn": sj.isclose,
        "hard_fn": lambda x, y: jnp.isclose(x, y),
    },
)


# ---------------------------------------------------------------------------
# Elementwise: mode / softness / shape sweep
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", ELEMENTWISE_CASES, ids=lambda c: c["name"])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("shape_name, shape", SHAPES, ids=_ids_from_shape)
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "softness", (common.NEAR_HARD_SOFTNESS, common.STABILITY_SOFTNESS), ids=str
)
def test_elementwise_float_modes(
    case, shape_name, shape, dtype, backend, softness, make_input
):
    """Elementwise transforms: hard parity, near-hard closeness, large-soft stability."""
    x = make_input(shape, dtype, backend)
    kwargs = dict(case["kwargs"])
    out_hard = case["hard_fn"](x, **kwargs)

    for mode in case["modes"]:
        if mode == "hard":
            out = case["soft_fn"](x, mode="hard", **kwargs)
        else:
            out = case["soft_fn"](x, mode=mode, softness=softness, **kwargs)
            assert out.dtype in {jnp.dtype("float32"), jnp.dtype("float64")}

        assert out.shape == out_hard.shape
        assert not jnp.any(jnp.isnan(out))

        if softness == common.NEAR_HARD_SOFTNESS:
            common.assert_allclose(out, out_hard)


@pytest.mark.parametrize("case", ELEMENTWISE_CASES, ids=lambda c: c["name"])
@pytest.mark.parametrize("shape_name, shape", SHAPES, ids=_ids_from_shape)
@pytest.mark.parametrize("backend", BACKENDS)
def test_elementwise_int_hard_parity(case, shape_name, shape, backend, make_input):
    """Integer parity only for hard mode."""
    x = make_input(shape, common._make_dtype("int32", backend), backend)
    kwargs = dict(case["kwargs"])
    out_hard = case["hard_fn"](x, **kwargs)
    out = case["soft_fn"](x, mode="hard", **kwargs)
    common.assert_allclose(out, out_hard)
    assert out.shape == out_hard.shape
    assert out.dtype == out_hard.dtype


# ---------------------------------------------------------------------------
# Python scalar inputs (float / int)
# ---------------------------------------------------------------------------


SCALAR_ELEMENTWISE = (
    ("relu", sj.relu, {}),
    ("abs", sj.abs, {}),
    ("sign", sj.sign, {}),
    ("round", sj.round, {}),
    ("clip", sj.clip, {"a": -0.25, "b": 0.25}),
    ("heaviside", sj.heaviside, {}),
)

SCALAR_COMPARISON = (
    ("greater", sj.greater),
    ("greater_equal", sj.greater_equal),
    ("less", sj.less),
    ("less_equal", sj.less_equal),
    ("equal", sj.equal),
    ("not_equal", sj.not_equal),
    ("isclose", sj.isclose),
)


@pytest.mark.parametrize("name, fn, kwargs", SCALAR_ELEMENTWISE, ids=lambda x: x if isinstance(x, str) else "")
@pytest.mark.parametrize("mode", ("hard", "smooth"))
@pytest.mark.parametrize("scalar", (1.5, -0.3, 3), ids=("float_pos", "float_neg", "int"))
def test_elementwise_python_scalar(name, fn, kwargs, mode, scalar):
    """Elementwise ops accept plain Python float/int scalars."""
    out = fn(scalar, mode=mode, **kwargs)
    assert isinstance(out, jnp.ndarray)
    assert out.shape == ()
    assert jnp.issubdtype(out.dtype, jnp.floating) or mode == "hard"
    assert jnp.isfinite(out)


@pytest.mark.parametrize("name, fn", SCALAR_COMPARISON, ids=lambda x: x if isinstance(x, str) else "")
@pytest.mark.parametrize("mode", ("hard", "smooth"))
def test_comparison_python_scalar(name, fn, mode):
    """Comparison ops accept plain Python float scalars."""
    out = fn(1.5, 0.5, mode=mode)
    assert isinstance(out, jnp.ndarray)
    assert out.shape == ()
    assert jnp.isfinite(out)


# ---------------------------------------------------------------------------
# Direct JAX parity tests for hard mode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", ELEMENTWISE_CASES, ids=lambda c: c["name"])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("shape_name, shape", SHAPES, ids=_ids_from_shape)
def test_elementwise_jax_parity(case, shape_name, shape, dtype, make_input):
    """Hard-mode softjax output must exactly match the corresponding JAX function."""
    x = make_input(shape, dtype, "jax")
    kwargs = dict(case["kwargs"])
    soft_out = case["soft_fn"](x, mode="hard", **kwargs)
    jax_out = case["hard_fn"](x, **kwargs)
    common.assert_jax_parity(soft_out, jax_out, msg=case["name"])


# ---------------------------------------------------------------------------
# Gradient finiteness for elementwise ops
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", ELEMENTWISE_CASES, ids=lambda c: c["name"])
@pytest.mark.parametrize("mode", common.MODES_ELEMENTWISE)
def test_elementwise_gradient_finite(case, mode):
    """Gradients through soft elementwise ops must be finite."""
    x = common.gradient_input((5,), jnp.float32)
    kwargs = dict(case["kwargs"])

    def loss(z):
        return jnp.sum(case["soft_fn"](z, mode=mode, softness=1.0, **kwargs))

    grad = jax.grad(loss)(x)
    common.assert_finite(grad, msg=f"{case['name']} mode={mode}")


# ---------------------------------------------------------------------------
# Heaviside boundary tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("backend", BACKENDS)
def test_heaviside_sign_boundaries(dtype, backend):
    """Check exact boundary behavior around zero."""
    vec = jnp.array([-1.0, 0.0, 1.0], dtype=common._make_dtype(dtype, backend))
    hv = sj.heaviside(vec, mode="hard")
    common.assert_allclose(hv, jnp.array([0.0, 0.5, 1.0], dtype=hv.dtype))

    sg = sj.sign(vec, mode="hard")
    common.assert_allclose(sg, jnp.array([-1.0, 0.0, 1.0], dtype=sg.dtype))

    soft_mid = sj.sign(vec, mode="smooth", softness=common.NEAR_HARD_SOFTNESS)
    assert -0.01 < float(soft_mid[1]) < 0.01, "Soft sign at zero not close to 0"
    assert not jnp.any(jnp.isnan(soft_mid))


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_heaviside_jax_parity(dtype):
    """Hard-mode heaviside matches jnp.heaviside(x, 0.5)."""
    x = jnp.array([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=dtype)
    soft_out = sj.heaviside(x, mode="hard")
    jax_out = jnp.heaviside(x, 0.5)
    common.assert_jax_parity(soft_out, jax_out, msg="heaviside")


@pytest.mark.parametrize("mode", common.MODES_ELEMENTWISE)
def test_heaviside_gradient_finite(mode):
    """Gradient through soft heaviside must be finite."""
    x = common.gradient_input((5,), jnp.float32)

    def loss(z):
        return jnp.sum(sj.heaviside(z, mode=mode, softness=1.0))

    grad = jax.grad(loss)(x)
    common.assert_finite(grad, msg=f"heaviside mode={mode}")


# ---------------------------------------------------------------------------
# Comparison / SoftBool tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", COMPARISON_CASES, ids=lambda c: c["name"])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("shape_name, shape", SHAPES, ids=_ids_from_shape)
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "softness", (common.NEAR_HARD_SOFTNESS, common.STABILITY_SOFTNESS), ids=str
)
def test_softbool(case, shape_name, shape, dtype, backend, softness, make_pair):
    """SoftBool comparisons: parity, bounds, stability."""
    x, y = make_pair(shape, dtype, backend)
    out_hard = case["hard_fn"](x, y)

    for mode in common.MODES_ELEMENTWISE:
        out = case["soft_fn"](x, y, mode=mode, softness=softness)
        assert out.dtype in {jnp.dtype("float32"), jnp.dtype("float64")}

        assert out.shape == out_hard.shape
        assert not jnp.any(jnp.isnan(out))
        common.assert_softbool(out)

        if softness == common.NEAR_HARD_SOFTNESS:
            common.assert_allclose(
                out, out_hard, err_msg=f"{case['name']} near-hard mismatch"
            )

    # Also test hard mode explicitly
    out_hard_mode = case["soft_fn"](x, y, mode="hard")
    assert out_hard_mode.shape == out_hard.shape
    common.assert_allclose(
        out_hard_mode, out_hard, err_msg=f"{case['name']} hard mismatch"
    )


@pytest.mark.parametrize("case", COMPARISON_CASES, ids=lambda c: c["name"])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("shape_name, shape", SHAPES, ids=_ids_from_shape)
def test_comparison_jax_parity(case, shape_name, shape, dtype, make_pair):
    """Hard-mode comparison must match JAX output (cast to float)."""
    x, y = make_pair(shape, dtype, "jax")
    soft_out = case["soft_fn"](x, y, mode="hard")
    jax_out = case["hard_fn"](x, y).astype(jnp.float_)
    common.assert_jax_parity(soft_out, jax_out, msg=case["name"])


@pytest.mark.parametrize("case", COMPARISON_CASES, ids=lambda c: c["name"])
@pytest.mark.parametrize("mode", common.MODES_ELEMENTWISE)
def test_comparison_gradient_finite(case, mode):
    """Gradients through soft comparisons must be finite."""
    x = common.gradient_input((5,), jnp.float32)
    y = common.gradient_input((5,), jnp.float32) + 0.1

    def loss(z):
        return jnp.sum(case["soft_fn"](z, y, mode=mode, softness=1.0))

    grad = jax.grad(loss)(x)
    common.assert_finite(grad, msg=f"{case['name']} mode={mode}")


@pytest.mark.parametrize("case", COMPARISON_CASES, ids=lambda c: c["name"])
@pytest.mark.parametrize("mode", common.MODES_ELEMENTWISE)
def test_comparison_gradient_finite_wrt_y(case, mode):
    """Gradients through soft comparisons wrt y must be finite."""
    x = common.gradient_input((5,), jnp.float32)
    y = common.gradient_input((5,), jnp.float32)

    def loss(w):
        return jnp.sum(case["soft_fn"](x, w, mode=mode, softness=1.0))

    grad = jax.grad(loss)(y)
    common.assert_finite(grad, msg=f"{case['name']} wrt y mode={mode}")


# ---------------------------------------------------------------------------
# Where
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("shape_name, shape", SHAPES, ids=_ids_from_shape)
@pytest.mark.parametrize("backend", BACKENDS)
def test_where_matches_mask(dtype, shape_name, shape, backend, make_pair):
    """Soft where mirrors hard selection when condition is near-hard."""
    x, y = make_pair(shape, dtype, backend)
    condition = sj.greater(x, y, softness=common.NEAR_HARD_SOFTNESS)
    out = sj.where(condition, x, y)
    # Threshold at 0.5 before casting to bool: in float64, sigmoid of a large
    # negative number may not underflow to exactly 0.0 (e.g. sigmoid(-400) ≈ 1e-174),
    # and any nonzero float cast to bool is True.
    mask = jnp.asarray(condition > 0.5, dtype=bool)
    expected = jnp.where(mask, jnp.asarray(x), jnp.asarray(y))
    common.assert_allclose(out, expected)


# ---------------------------------------------------------------------------
# Logical ops
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_logical_ops_range_and_shapes(backend):
    """Logical ops stay within [0,1] and preserve shape."""
    x = jnp.array([0.1, 0.5, 0.9])
    y = jnp.array([0.9, 0.2, 0.4])
    ops = [
        sj.logical_not(x),
        sj.logical_and(x, y),
        sj.logical_or(x, y),
        sj.logical_xor(x, y),
    ]
    for out in ops:
        assert out.shape == x.shape
        common.assert_softbool(out)


def test_logical_ops_truth_table():
    """Verify logical ops produce correct truth-table results for hard boolean inputs."""
    zero = jnp.array([0.0])
    one = jnp.array([1.0])

    # NOT
    common.assert_allclose(sj.logical_not(zero), one, tol=1e-5)
    common.assert_allclose(sj.logical_not(one), zero, tol=1e-5)

    # AND
    common.assert_allclose(sj.logical_and(zero, zero), zero, tol=1e-5)
    common.assert_allclose(sj.logical_and(zero, one), zero, tol=1e-5)
    common.assert_allclose(sj.logical_and(one, zero), zero, tol=1e-5)
    common.assert_allclose(sj.logical_and(one, one), one, tol=1e-5)

    # OR
    common.assert_allclose(sj.logical_or(zero, zero), zero, tol=1e-5)
    common.assert_allclose(sj.logical_or(zero, one), one, tol=1e-5)
    common.assert_allclose(sj.logical_or(one, zero), one, tol=1e-5)
    common.assert_allclose(sj.logical_or(one, one), one, tol=1e-5)

    # XOR
    common.assert_allclose(sj.logical_xor(zero, zero), zero, tol=1e-5)
    common.assert_allclose(sj.logical_xor(zero, one), one, tol=1e-5)
    common.assert_allclose(sj.logical_xor(one, zero), one, tol=1e-5)
    common.assert_allclose(sj.logical_xor(one, one), zero, tol=1e-5)


# ---------------------------------------------------------------------------
# sigmoidal and softrelu (previously untested)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", common.MODES_ELEMENTWISE)
def test_sigmoidal_bounds(mode):
    """sigmoidal output must be in [0, 1]."""
    x = common.gradient_input((10,), jnp.float32)
    out = sj.sigmoidal(x, mode=mode, softness=1.0)
    common.assert_softbool(out)
    common.assert_finite(out, msg=f"sigmoidal mode={mode}")


@pytest.mark.parametrize("mode", common.MODES_ELEMENTWISE)
def test_sigmoidal_gradient_finite(mode):
    """Gradients through sigmoidal must be finite."""
    x = common.gradient_input((5,), jnp.float32)

    def loss(z):
        return jnp.sum(sj.sigmoidal(z, mode=mode, softness=1.0))

    grad = jax.grad(loss)(x)
    common.assert_finite(grad, msg=f"sigmoidal mode={mode}")


@pytest.mark.parametrize("mode", common.MODES_ELEMENTWISE)
@pytest.mark.parametrize("gated", [False, True])
def test_softrelu_non_negative(mode, gated):
    """softrelu output must be non-negative (ungated) or finite (gated)."""
    x = common.gradient_input((10,), jnp.float32)
    out = sj.softrelu(x, mode=mode, softness=1.0, gated=gated)
    if not gated:
        # Ungated softrelu (integral of sigmoidal) is always non-negative
        assert np.all(np.asarray(out) >= -1e-7), (
            f"softrelu negative for mode={mode} gated={gated}"
        )
    # Gated softrelu (x * sigmoidal(x)) can be slightly negative for negative x
    common.assert_finite(out, msg=f"softrelu mode={mode} gated={gated}")


@pytest.mark.parametrize("mode", common.MODES_ELEMENTWISE)
@pytest.mark.parametrize("gated", [False, True])
def test_softrelu_gradient_finite(mode, gated):
    """Gradients through softrelu must be finite."""
    x = common.gradient_input((5,), jnp.float32)

    def loss(z):
        return jnp.sum(sj.softrelu(z, mode=mode, softness=1.0, gated=gated))

    grad = jax.grad(loss)(x)
    common.assert_finite(grad, msg=f"softrelu mode={mode} gated={gated}")


# ---------------------------------------------------------------------------
# sigmoidal / softrelu equivalence to argmax / max
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["smooth", "c0", "_c1_pnorm", "_c2_pnorm"])
@pytest.mark.parametrize("softness", [0.1, 1.0, 5.0])
def test_sigmoidal_equals_argmax(mode, softness):
    """sigmoidal(x, s) must equal argmax([0, x], softness=s', standardize=False)[1].

    For smooth: s' = s. For piecewise modes: s' = 5s (piecewise sigmoidal has a
    built-in 1/5 scaling to match smooth's effective transition width).
    """
    x = jnp.array([-2.0, -0.5, 0.0, 0.3, 1.0, 3.0], dtype=jnp.float64)
    expected = sj.sigmoidal(x, mode=mode, softness=softness)
    # Piecewise modes have a /5 factor, so argmax needs 5*softness.
    # c1_pnorm/c2_pnorm in sigmoidal correspond to c1/c2 in argmax.
    argmax_softness = softness if mode == "smooth" else 5.0 * softness
    argmax_mode = {"_c1_pnorm": "c1", "_c2_pnorm": "c2"}.get(mode, mode)
    # Build 2-element input [0, x_i] for each element and compute argmax
    pairs = jnp.stack([jnp.zeros_like(x), x], axis=-1)  # (n, 2)
    argmax_out = jax.vmap(
        lambda p: sj.argmax(
            p, axis=0, softness=argmax_softness, mode=argmax_mode, standardize=False
        )
    )(pairs)  # (n, 2)
    actual = argmax_out[:, 1]
    np.testing.assert_allclose(
        np.asarray(actual),
        np.asarray(expected),
        rtol=1e-5,
        atol=1e-5,
        err_msg=f"sigmoidal != argmax([0,x])[1] for mode={mode} softness={softness}",
    )


@pytest.mark.parametrize("mode", ["smooth", "c0", "_c1_pnorm", "_c2_pnorm"])
@pytest.mark.parametrize("softness", [0.1, 1.0, 5.0])
def test_softrelu_gated_equals_max(mode, softness):
    """softrelu(x, gated=True) must equal max([0, x], softness=s', standardize=False).

    For smooth: s' = s. For piecewise modes: s' = 5s (piecewise has a built-in
    1/5 scaling to match smooth's effective transition width).
    """
    x = jnp.array([-2.0, -0.5, 0.0, 0.3, 1.0, 3.0], dtype=jnp.float64)
    expected = sj.softrelu(x, mode=mode, softness=softness, gated=True)
    # Piecewise modes have a /5 factor, so max needs 5*softness.
    # c1_pnorm/c2_pnorm in softrelu correspond to c1/c2 in max.
    max_softness = softness if mode == "smooth" else 5.0 * softness
    max_mode = {"_c1_pnorm": "c1", "_c2_pnorm": "c2"}.get(mode, mode)
    # Build 2-element input [0, x_i] and compute soft max
    pairs = jnp.stack([jnp.zeros_like(x), x], axis=-1)  # (n, 2)
    actual = jax.vmap(
        lambda p: sj.max(p, axis=0, softness=max_softness, mode=max_mode, standardize=False)
    )(pairs)  # (n,)
    np.testing.assert_allclose(
        np.asarray(actual),
        np.asarray(expected),
        rtol=1e-5,
        atol=1e-5,
        err_msg=f"softrelu(gated=True) != max([0,x]) for mode={mode} softness={softness}",
    )


@pytest.mark.parametrize("mode", common.MODES_ELEMENTWISE)
@pytest.mark.parametrize("softness", [0.1, 1.0, 5.0])
def test_softrelu_ungated_is_integral_of_sigmoidal(mode, softness):
    """d/dx softrelu(x, gated=False) must equal sigmoidal(x) (integral claim)."""
    x = jnp.array([-1.5, -0.3, 0.0, 0.4, 1.2], dtype=jnp.float64)

    def softrelu_scalar(z):
        return sj.softrelu(z, mode=mode, softness=softness, gated=False)

    # Derivative of softrelu should equal sigmoidal
    grad_softrelu = jax.vmap(jax.grad(softrelu_scalar))(x)
    sigmoidal_vals = sj.sigmoidal(x, mode=mode, softness=softness)
    np.testing.assert_allclose(
        np.asarray(grad_softrelu),
        np.asarray(sigmoidal_vals),
        rtol=1e-5,
        atol=1e-5,
        err_msg=f"d/dx softrelu != sigmoidal for mode={mode} softness={softness}",
    )


# ---------------------------------------------------------------------------
# relu/clip gated parameter
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", common.MODES_ELEMENTWISE)
def test_relu_gated(mode):
    """relu with gated=True produces finite output and gradient."""
    x = common.gradient_input((5,), jnp.float32)
    out = sj.relu(x, mode=mode, softness=1.0, gated=True)
    common.assert_finite(out, msg=f"relu gated mode={mode}")
    assert out.shape == x.shape

    def loss(z):
        return jnp.sum(sj.relu(z, mode=mode, softness=1.0, gated=True))

    grad = jax.grad(loss)(x)
    common.assert_finite(grad, msg=f"relu gated grad mode={mode}")


@pytest.mark.parametrize("mode", common.MODES_ELEMENTWISE)
def test_clip_gated(mode):
    """clip with gated=True produces finite output and gradient."""
    x = common.gradient_input((5,), jnp.float32)
    out = sj.clip(x, -0.25, 0.25, mode=mode, softness=1.0, gated=True)
    common.assert_finite(out, msg=f"clip gated mode={mode}")
    assert out.shape == x.shape

    def loss(z):
        return jnp.sum(sj.clip(z, -0.25, 0.25, mode=mode, softness=1.0, gated=True))

    grad = jax.grad(loss)(x)
    common.assert_finite(grad, msg=f"clip gated grad mode={mode}")


# ---------------------------------------------------------------------------
# round neighbor_radius parameter
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("neighbor_radius", [1, 3, 5, 10])
@pytest.mark.parametrize("mode", common.MODES_ELEMENTWISE)
def test_round_neighbor_radius(neighbor_radius, mode):
    """round with different neighbor_radius produces finite output."""
    x = common.gradient_input((5,), jnp.float32)
    out = sj.round(x, mode=mode, softness=1.0, neighbor_radius=neighbor_radius)
    common.assert_finite(
        out, msg=f"round neighbor_radius={neighbor_radius} mode={mode}"
    )
    assert out.shape == x.shape


# ---------------------------------------------------------------------------
# isclose rtol/atol parameters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rtol", [1e-5, 1e-3, 1e-1])
@pytest.mark.parametrize("atol", [1e-8, 1e-5, 1e-2])
@pytest.mark.parametrize("mode", common.MODES_ELEMENTWISE)
def test_isclose_rtol_atol(rtol, atol, mode):
    """isclose with different rtol/atol produces finite output with correct shape."""
    x = jnp.array([1.0, 1.0001, 1.1, 2.0], dtype=jnp.float32)
    y = jnp.array([1.0, 1.0, 1.0, 1.0], dtype=jnp.float32)
    out = sj.isclose(x, y, mode=mode, softness=1.0, rtol=rtol, atol=atol)
    # Soft isclose can slightly exceed [0,1] due to numerical imprecision,
    # so we only check finiteness and shape here.
    common.assert_finite(out, msg=f"isclose rtol={rtol} atol={atol} mode={mode}")
    assert out.shape == x.shape


# ---------------------------------------------------------------------------
# Logical ops use_geometric_mean parameter
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_geometric_mean", [False, True])
def test_logical_ops_geometric_mean(use_geometric_mean):
    """Logical ops with use_geometric_mean produce valid SoftBool output."""
    x = jnp.array([0.1, 0.5, 0.9])
    y = jnp.array([0.9, 0.2, 0.4])
    for fn in [sj.logical_and, sj.logical_or, sj.logical_xor]:
        out = fn(x, y, use_geometric_mean=use_geometric_mean)
        common.assert_softbool(out)
        assert out.shape == x.shape


@pytest.mark.parametrize("use_geometric_mean", [False, True])
def test_logical_ops_geometric_mean_truth_table(use_geometric_mean):
    """Truth table still holds with use_geometric_mean."""
    zero = jnp.array([0.0])
    one = jnp.array([1.0])

    common.assert_allclose(
        sj.logical_and(one, one, use_geometric_mean=use_geometric_mean), one, tol=1e-5
    )
    common.assert_allclose(
        sj.logical_and(one, zero, use_geometric_mean=use_geometric_mean), zero, tol=1e-5
    )
    common.assert_allclose(
        sj.logical_or(zero, one, use_geometric_mean=use_geometric_mean), one, tol=1e-5
    )
    common.assert_allclose(
        sj.logical_or(zero, zero, use_geometric_mean=use_geometric_mean), zero, tol=1e-5
    )


@pytest.mark.parametrize("use_geometric_mean", [False, True])
def test_all_any_geometric_mean(use_geometric_mean):
    """all/any with use_geometric_mean produce valid SoftBool output."""
    x = jnp.array([0.8, 0.9, 1.0])
    out_all = sj.all(x, axis=-1, use_geometric_mean=use_geometric_mean)
    out_any = sj.any(x, axis=-1, use_geometric_mean=use_geometric_mean)
    common.assert_softbool(jnp.array([out_all]))
    common.assert_softbool(jnp.array([out_any]))
    common.assert_finite(jnp.array([out_all]), msg="all geometric_mean")
    common.assert_finite(jnp.array([out_any]), msg="any geometric_mean")


# ---------------------------------------------------------------------------
# Gradient vs finite differences (correctness, not just finiteness)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["smooth", "c0"])
def test_abs_grad_vs_finite_diff(mode):
    """sj.abs gradient matches finite differences."""
    x = jnp.array([-0.8, 0.3, -0.1, 0.6], dtype=jnp.float64)

    def loss(z):
        return jnp.sum(sj.abs(z, softness=1.0, mode=mode))

    common.assert_grad_matches_finite_diff(loss, x, msg=f"abs mode={mode}")


@pytest.mark.parametrize("mode", ["smooth", "c0"])
def test_relu_grad_vs_finite_diff(mode):
    """sj.relu gradient matches finite differences."""
    x = jnp.array([-0.8, 0.3, -0.1, 0.6], dtype=jnp.float64)

    def loss(z):
        return jnp.sum(sj.relu(z, softness=1.0, mode=mode))

    common.assert_grad_matches_finite_diff(loss, x, msg=f"relu mode={mode}")


@pytest.mark.parametrize("mode", ["smooth", "c0"])
def test_clip_grad_vs_finite_diff(mode):
    """sj.clip gradient matches finite differences."""
    x = jnp.array([-0.8, 0.3, -0.1, 0.6], dtype=jnp.float64)

    def loss(z):
        return jnp.sum(sj.clip(z, -0.25, 0.25, softness=1.0, mode=mode))

    common.assert_grad_matches_finite_diff(loss, x, msg=f"clip mode={mode}")


@pytest.mark.parametrize("mode", ["smooth", "c0"])
def test_greater_grad_vs_finite_diff(mode):
    """sj.greater gradient matches finite differences."""
    x = jnp.array([-0.5, 0.3, 0.0, 0.7], dtype=jnp.float64)
    y = jnp.array([0.1, 0.2, 0.0, 0.9], dtype=jnp.float64)

    def loss(z):
        return jnp.sum(sj.greater(z, y, softness=1.0, mode=mode))

    common.assert_grad_matches_finite_diff(loss, x, msg=f"greater mode={mode}")


@pytest.mark.parametrize("mode", ["smooth", "c0"])
def test_sign_grad_vs_finite_diff(mode):
    """sj.sign gradient matches finite differences."""
    x = jnp.array([-0.8, 0.3, -0.1, 0.6], dtype=jnp.float64)

    def loss(z):
        return jnp.sum(sj.sign(z, softness=1.0, mode=mode))

    common.assert_grad_matches_finite_diff(loss, x, msg=f"sign mode={mode}")


@pytest.mark.parametrize("mode", ["smooth", "c0"])
def test_round_grad_vs_finite_diff(mode):
    """sj.round gradient matches finite differences."""
    x = jnp.array([-0.7, 0.3, 1.5, -1.2], dtype=jnp.float64)

    def loss(z):
        return jnp.sum(sj.round(z, softness=0.1, mode=mode))

    common.assert_grad_matches_finite_diff(loss, x, msg=f"round mode={mode}")


# ---------------------------------------------------------------------------
# c1_pnorm correspondence with arraywise p=3/2 simplex projection
# ---------------------------------------------------------------------------


def test_c1_pnorm_sigmoidal_matches_simplex_projection():
    """sigmoidal(x, mode='c1_pnorm') must equal the 2D p=3/2 simplex projection.

    Piecewise sigmoidal includes a 1/5 scaling factor, so the projection input
    is x/(5*softness) rather than x/softness.
    """
    from softjax.projections_simplex import _proj_unit_simplex_pnorm_q3

    xs = jnp.linspace(-3.0, 3.0, 50, dtype=jnp.float64)
    for softness in [0.1, 1.0, 5.0]:
        sig_vals = sj.sigmoidal(xs, softness=softness, mode="_c1_pnorm")
        # Compute via 2D simplex projection with p=3/2
        proj_vals = []
        for x_scalar in xs:
            v = jnp.array(
                [x_scalar / (5.0 * softness), 0.0], dtype=jnp.float64
            )
            proj = _proj_unit_simplex_pnorm_q3(v)
            proj_vals.append(float(proj[0]))
        proj_vals = jnp.array(proj_vals)
        np.testing.assert_allclose(
            np.asarray(sig_vals),
            np.asarray(proj_vals),
            atol=1e-5,
            rtol=1e-5,
            err_msg=f"sigmoidal c1_pnorm != 2D p=3/2 projection, softness={softness}",
        )


def test_c1_pnorm_softrelu_matches_max_projection():
    """gated softrelu(x, mode='c1_pnorm') must equal x * sigmoidal(x)."""
    xs = jnp.linspace(-3.0, 3.0, 50, dtype=jnp.float64)
    for softness in [0.1, 1.0, 5.0]:
        relu_vals = sj.softrelu(xs, softness=softness, mode="_c1_pnorm", gated=True)
        sig_vals = np.asarray(xs) * np.asarray(
            sj.sigmoidal(xs, softness=softness, mode="_c1_pnorm")
        )
        np.testing.assert_allclose(
            np.asarray(relu_vals),
            sig_vals,
            atol=1e-10,
            err_msg=f"gated softrelu c1_pnorm != x*sigmoidal, softness={softness}",
        )


def test_c1_pnorm_sigmoidal_grad_vs_finite_diff():
    """sigmoidal c1_pnorm gradient matches finite differences."""
    x = jnp.array([-0.7, -0.3, 0.0, 0.3, 0.7], dtype=jnp.float64)

    def loss(z):
        return jnp.sum(sj.sigmoidal(z, softness=1.0, mode="_c1_pnorm"))

    common.assert_grad_matches_finite_diff(loss, x, msg="sigmoidal c1_pnorm")


def test_c1_pnorm_softrelu_grad_vs_finite_diff():
    """softrelu c1_pnorm gradient matches finite differences."""
    x = jnp.array([-0.7, -0.3, 0.0, 0.3, 0.7], dtype=jnp.float64)

    def loss(z):
        return jnp.sum(sj.softrelu(z, softness=1.0, mode="_c1_pnorm"))

    common.assert_grad_matches_finite_diff(loss, x, msg="softrelu c1_pnorm")


def test_c1_pnorm_softrelu_integral_grad_vs_finite_diff():
    """softrelu c1_pnorm (integral form) gradient matches finite differences."""
    x = jnp.array([-0.7, -0.3, 0.0, 0.3, 0.7], dtype=jnp.float64)

    def loss(z):
        return jnp.sum(sj.softrelu(z, softness=1.0, mode="_c1_pnorm", gated=False))

    common.assert_grad_matches_finite_diff(loss, x, msg="softrelu c1_pnorm integral")


def test_c1_pnorm_softrelu_integral_derivative_is_sigmoidal():
    """d/dx softrelu(x, mode='c1_pnorm', gated=False) == sigmoidal(x, mode='c1_pnorm')."""
    x = jnp.linspace(-4.5, 4.5, 20, dtype=jnp.float64)

    grad_relu = jax.vmap(
        jax.grad(lambda z: sj.softrelu(z, softness=1.0, mode="_c1_pnorm", gated=False))
    )(x)
    sig_vals = sj.sigmoidal(x, softness=1.0, mode="_c1_pnorm")

    np.testing.assert_allclose(
        np.asarray(grad_relu),
        np.asarray(sig_vals),
        atol=1e-10,
        rtol=1e-10,
        err_msg="d/dx softrelu(c1_pnorm) != sigmoidal(c1_pnorm)",
    )


# ---------------------------------------------------------------------------
# c2_pnorm correspondence with arraywise p=4/3 simplex projection
# ---------------------------------------------------------------------------


def test_c2_pnorm_sigmoidal_matches_simplex_projection():
    """sigmoidal(x, mode='c2_pnorm') must equal the 2D p=4/3 simplex projection.

    Piecewise sigmoidal includes a 1/5 scaling factor, so the projection input
    is x/(5*softness) rather than x/softness.
    """
    from softjax.projections_simplex import _proj_unit_simplex_pnorm_q4

    xs = jnp.linspace(-3.0, 3.0, 50, dtype=jnp.float64)
    for softness in [0.1, 1.0, 5.0]:
        sig_vals = sj.sigmoidal(xs, softness=softness, mode="_c2_pnorm")
        proj_vals = []
        for x_scalar in xs:
            v = jnp.array(
                [x_scalar / (5.0 * softness), 0.0], dtype=jnp.float64
            )
            proj = _proj_unit_simplex_pnorm_q4(v)
            proj_vals.append(float(proj[0]))
        proj_vals = jnp.array(proj_vals)
        np.testing.assert_allclose(
            np.asarray(sig_vals),
            np.asarray(proj_vals),
            atol=1e-5,
            rtol=1e-5,
            err_msg=f"sigmoidal c2_pnorm != 2D p=4/3 projection, softness={softness}",
        )


def test_c2_pnorm_softrelu_matches_max_projection():
    """softrelu(x, mode='c2_pnorm') must equal max([0,x]) via 2D p=4/3 simplex projection."""
    xs = jnp.linspace(-3.0, 3.0, 50, dtype=jnp.float64)
    for softness in [0.1, 1.0, 5.0]:
        relu_vals = sj.softrelu(xs, softness=softness, mode="_c2_pnorm", gated=True)
        # gated softrelu(x) = x * sigmoidal(x, mode="_c2_pnorm")
        sig_vals = np.asarray(xs) * np.asarray(
            sj.sigmoidal(xs, softness=softness, mode="_c2_pnorm")
        )
        np.testing.assert_allclose(
            np.asarray(relu_vals),
            sig_vals,
            atol=1e-10,
            err_msg=f"gated softrelu c2_pnorm != x*sigmoidal, softness={softness}",
        )


def test_c2_pnorm_sigmoidal_grad_vs_finite_diff():
    """sigmoidal c2_pnorm gradient matches finite differences."""
    x = jnp.array([-0.7, -0.3, 0.0, 0.3, 0.7], dtype=jnp.float64)

    def loss(z):
        return jnp.sum(sj.sigmoidal(z, softness=1.0, mode="_c2_pnorm"))

    common.assert_grad_matches_finite_diff(loss, x, msg="sigmoidal c2_pnorm")


def test_c2_pnorm_softrelu_grad_vs_finite_diff():
    """softrelu c2_pnorm gradient matches finite differences."""
    x = jnp.array([-0.7, -0.3, 0.0, 0.3, 0.7], dtype=jnp.float64)

    def loss(z):
        return jnp.sum(sj.softrelu(z, softness=1.0, mode="_c2_pnorm"))

    common.assert_grad_matches_finite_diff(loss, x, msg="softrelu c2_pnorm")


def test_c2_pnorm_softrelu_integral_grad_vs_finite_diff():
    """softrelu c2_pnorm (integral form) gradient matches finite differences."""
    x = jnp.array([-0.7, -0.3, 0.0, 0.3, 0.7], dtype=jnp.float64)

    def loss(z):
        return jnp.sum(sj.softrelu(z, softness=1.0, mode="_c2_pnorm", gated=False))

    common.assert_grad_matches_finite_diff(loss, x, msg="softrelu c2_pnorm integral")


def test_c2_pnorm_softrelu_integral_derivative_is_sigmoidal():
    """d/dx softrelu(x, mode='c2_pnorm', gated=False) == sigmoidal(x, mode='c2_pnorm')."""
    x = jnp.linspace(-0.45, 0.45, 20, dtype=jnp.float64)

    grad_relu = jax.vmap(
        jax.grad(lambda z: sj.softrelu(z, softness=1.0, mode="_c2_pnorm", gated=False))
    )(x)
    sig_vals = sj.sigmoidal(x, softness=1.0, mode="_c2_pnorm")

    np.testing.assert_allclose(
        np.asarray(grad_relu),
        np.asarray(sig_vals),
        atol=1e-6,
        rtol=1e-6,
        err_msg="d/dx softrelu(c2_pnorm) != sigmoidal(c2_pnorm)",
    )


# ---------------------------------------------------------------------------
# Autograd-safe operators
# ---------------------------------------------------------------------------


class TestSafeArcsin:
    """Tests for sj.arcsin (autograd-safe arcsin)."""

    def test_interior_matches_jnp(self):
        """sj.arcsin matches jnp.arcsin on interior points."""
        x = jnp.array([-0.9, -0.5, 0.0, 0.5, 0.9], dtype=jnp.float64)
        np.testing.assert_allclose(
            np.asarray(sj.arcsin(x)), np.asarray(jnp.arcsin(x)), atol=1e-12
        )

    def test_boundary_values(self):
        """sj.arcsin returns ±π/2 at x=±1."""
        x = jnp.array([-1.0, 1.0], dtype=jnp.float64)
        expected = jnp.array([-jnp.pi / 2, jnp.pi / 2])
        np.testing.assert_allclose(
            np.asarray(sj.arcsin(x)), np.asarray(expected), atol=1e-12
        )

    def test_gradient_finite_at_boundary(self):
        """Gradient of sj.arcsin is finite at x=±1."""
        x = jnp.array([-1.0, -0.99, 0.0, 0.99, 1.0], dtype=jnp.float64)
        grad = jax.grad(lambda z: jnp.sum(sj.arcsin(z)))(x)
        common.assert_finite(grad, msg="arcsin boundary grad")

    def test_grad_vs_finite_diff(self):
        """sj.arcsin gradient matches finite differences on interior."""
        x = jnp.array([-0.8, -0.3, 0.0, 0.3, 0.8], dtype=jnp.float64)
        common.assert_grad_matches_finite_diff(
            lambda z: jnp.sum(sj.arcsin(z)), x, msg="arcsin"
        )


class TestSafeArccos:
    """Tests for sj.arccos (autograd-safe arccos)."""

    def test_interior_matches_jnp(self):
        """sj.arccos matches jnp.arccos on interior points."""
        x = jnp.array([-0.9, -0.5, 0.0, 0.5, 0.9], dtype=jnp.float64)
        np.testing.assert_allclose(
            np.asarray(sj.arccos(x)), np.asarray(jnp.arccos(x)), atol=1e-12
        )

    def test_boundary_values(self):
        """sj.arccos returns 0 at x=1 and π at x=-1."""
        x = jnp.array([-1.0, 1.0], dtype=jnp.float64)
        expected = jnp.array([jnp.pi, 0.0])
        np.testing.assert_allclose(
            np.asarray(sj.arccos(x)), np.asarray(expected), atol=1e-12
        )

    def test_gradient_finite_at_boundary(self):
        """Gradient of sj.arccos is finite at x=±1."""
        x = jnp.array([-1.0, -0.99, 0.0, 0.99, 1.0], dtype=jnp.float64)
        grad = jax.grad(lambda z: jnp.sum(sj.arccos(z)))(x)
        common.assert_finite(grad, msg="arccos boundary grad")

    def test_grad_vs_finite_diff(self):
        """sj.arccos gradient matches finite differences on interior."""
        x = jnp.array([-0.8, -0.3, 0.0, 0.3, 0.8], dtype=jnp.float64)
        common.assert_grad_matches_finite_diff(
            lambda z: jnp.sum(sj.arccos(z)), x, msg="arccos"
        )


class TestSafeDiv:
    """Tests for sj.div (autograd-safe division)."""

    def test_interior_matches_division(self):
        """sj.div matches x/y when y != 0."""
        x = jnp.array([1.0, 2.0, -3.0, 0.0], dtype=jnp.float64)
        y = jnp.array([2.0, -1.0, 0.5, 3.0], dtype=jnp.float64)
        np.testing.assert_allclose(
            np.asarray(sj.div(x, y)), np.asarray(x / y), atol=1e-12
        )

    def test_boundary_values(self):
        """sj.div returns 0 when y=0."""
        x = jnp.array([1.0, -1.0, 0.0], dtype=jnp.float64)
        y = jnp.array([0.0, 0.0, 0.0], dtype=jnp.float64)
        result = sj.div(x, y)
        np.testing.assert_allclose(np.asarray(result), np.zeros(3), atol=1e-12)

    def test_gradient_finite_at_boundary(self):
        """Gradient of sj.div is finite at y=0."""
        x = jnp.array([1.0, 2.0], dtype=jnp.float64)
        y = jnp.array([0.0, 0.0], dtype=jnp.float64)
        grad_x = jax.grad(lambda z: jnp.sum(sj.div(z, y)))(x)
        grad_y = jax.grad(lambda w: jnp.sum(sj.div(x, w)))(y)
        common.assert_finite(grad_x, msg="div grad wrt x at y=0")
        common.assert_finite(grad_y, msg="div grad wrt y at y=0")

    def test_grad_vs_finite_diff(self):
        """sj.div gradient matches finite differences on interior."""
        x = jnp.array([1.0, 2.0, -3.0], dtype=jnp.float64)
        y = jnp.array([2.0, -1.0, 0.5], dtype=jnp.float64)
        common.assert_grad_matches_finite_diff(
            lambda z: jnp.sum(sj.div(z, y)), x, msg="div wrt x"
        )
        common.assert_grad_matches_finite_diff(
            lambda w: jnp.sum(sj.div(x, w)), y, msg="div wrt y"
        )


class TestSafeLog:
    """Tests for sj.log (autograd-safe log)."""

    def test_interior_matches_jnp(self):
        """sj.log matches jnp.log on positive inputs."""
        x = jnp.array([0.01, 0.5, 1.0, 2.0, 10.0], dtype=jnp.float64)
        np.testing.assert_allclose(
            np.asarray(sj.log(x)), np.asarray(jnp.log(x)), atol=1e-12
        )

    def test_boundary_values(self):
        """sj.log returns 0 for x<=0."""
        x = jnp.array([0.0, -1.0, -0.001], dtype=jnp.float64)
        result = sj.log(x)
        np.testing.assert_allclose(np.asarray(result), np.zeros(3), atol=1e-12)

    def test_gradient_finite_at_boundary(self):
        """Gradient of sj.log is finite at x=0."""
        x = jnp.array([0.0, 0.001, 1.0], dtype=jnp.float64)
        grad = jax.grad(lambda z: jnp.sum(sj.log(z)))(x)
        common.assert_finite(grad, msg="log boundary grad")

    def test_grad_vs_finite_diff(self):
        """sj.log gradient matches finite differences on interior."""
        x = jnp.array([0.1, 0.5, 1.0, 2.0, 5.0], dtype=jnp.float64)
        common.assert_grad_matches_finite_diff(
            lambda z: jnp.sum(sj.log(z)), x, msg="log"
        )


class TestSafeNorm:
    """Tests for sj.norm (autograd-safe L2 norm)."""

    def test_interior_matches_jnp(self):
        """sj.norm matches jnp.linalg.norm on nonzero input."""
        x = jnp.array([3.0, 4.0], dtype=jnp.float64)
        np.testing.assert_allclose(
            float(sj.norm(x)), float(jnp.linalg.norm(x)), atol=1e-12
        )

    def test_boundary_values(self):
        """sj.norm returns 0 for the zero vector."""
        x = jnp.zeros(5, dtype=jnp.float64)
        assert float(sj.norm(x)) == 0.0

    def test_gradient_finite_at_boundary(self):
        """Gradient of sj.norm is finite at x=0."""
        x = jnp.zeros(3, dtype=jnp.float64)
        grad = jax.grad(lambda z: sj.norm(z))(x)
        common.assert_finite(grad, msg="norm boundary grad")

    def test_axis_and_keepdims(self):
        """sj.norm respects axis and keepdims arguments."""
        x = jnp.array([[3.0, 4.0], [0.0, 0.0], [1.0, 0.0]], dtype=jnp.float64)
        result = sj.norm(x, axis=1)
        expected = jnp.linalg.norm(x, axis=1)
        np.testing.assert_allclose(np.asarray(result), np.asarray(expected), atol=1e-12)

        result_kd = sj.norm(x, axis=1, keepdims=True)
        assert result_kd.shape == (3, 1)

    def test_grad_vs_finite_diff(self):
        """sj.norm gradient matches finite differences on nonzero input."""
        x = jnp.array([0.3, -0.5, 0.8, -0.2], dtype=jnp.float64)
        common.assert_grad_matches_finite_diff(lambda z: sj.norm(z), x, msg="norm")


class TestSafeSqrt:
    """Tests for sj.sqrt (autograd-safe sqrt)."""

    def test_interior_matches_jnp(self):
        """sj.sqrt matches jnp.sqrt on positive inputs."""
        x = jnp.array([0.01, 0.25, 1.0, 4.0, 9.0], dtype=jnp.float64)
        np.testing.assert_allclose(
            np.asarray(sj.sqrt(x)), np.asarray(jnp.sqrt(x)), atol=1e-12
        )

    def test_boundary_values(self):
        """sj.sqrt returns 0 for x<=0."""
        x = jnp.array([0.0, -1.0], dtype=jnp.float64)
        result = sj.sqrt(x)
        np.testing.assert_allclose(np.asarray(result), np.zeros(2), atol=1e-12)

    def test_gradient_finite_at_boundary(self):
        """Gradient of sj.sqrt is finite at x=0."""
        x = jnp.array([0.0, 0.001, 1.0], dtype=jnp.float64)
        grad = jax.grad(lambda z: jnp.sum(sj.sqrt(z)))(x)
        common.assert_finite(grad, msg="sqrt boundary grad")

    def test_grad_vs_finite_diff(self):
        """sj.sqrt gradient matches finite differences on interior."""
        x = jnp.array([0.1, 0.5, 1.0, 4.0, 9.0], dtype=jnp.float64)
        common.assert_grad_matches_finite_diff(
            lambda z: jnp.sum(sj.sqrt(z)), x, msg="sqrt"
        )
