import jax
import jax.numpy as jnp
import numpy as np

import softjax as sj


jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "high")


BACKENDS = ("jax",)


# Auto-detect available JAX devices
def _available_devices():
    devs = ["cpu"]
    try:
        if jax.devices("gpu"):
            devs.append("gpu")
    except RuntimeError:
        pass
    return tuple(devs)


DEVICES = _available_devices()
SHAPES = {
    "vector": (4,),
    "matrix": (2, 3),
    "tensor": (4, 2, 3),
}

# Suite-wide constants
NEAR_HARD_SOFTNESS = 1e-3
STABILITY_SOFTNESS = 100.0
TOLERANCE = 1e-2


def _base_values(shape: tuple[int, ...], offset: float = 0.0) -> np.ndarray:
    size = int(np.prod(shape))
    data = np.linspace(-1.0, 1.0, num=size, dtype=np.float64)
    return data.reshape(shape) + offset


def _make_dtype(dtype_str, backend):
    return np.dtype(dtype_str) if backend == "numpy" else jnp.dtype(dtype_str)


def _resolve_jax_device(device: str) -> jax.Device:
    device_lower = device.lower()
    if device_lower not in {"cpu", "gpu"}:
        raise ValueError(f"Unknown device '{device}'")
    devices = jax.devices(device_lower)
    if not devices:
        raise RuntimeError(f"Requested device '{device_lower}' but none are available")
    return devices[0]


def make_array(
    shape: tuple[int, ...],
    dtype: str,
    backend: str,
    device: str | None = None,
    *,
    offset: float = 0.0,
    type: str = "",  # zeros, ones, or default
    softbool=False,
) -> np.ndarray | jnp.ndarray:
    dtype = _make_dtype(dtype, backend)

    if type == "zeros":
        values = np.zeros(shape, dtype=np.float64)
    elif type == "ones":
        values = np.ones(shape, dtype=np.float64)
    else:
        values = _base_values(shape, offset=offset)

    if softbool:
        values = sj.sigmoidal(values)

    dtype_np = np.dtype(dtype)
    if np.issubdtype(dtype_np, np.integer):
        cast_values = np.round(values).astype(dtype_np)
    else:
        cast_values = values.astype(dtype_np)

    if backend == "numpy":
        return cast_values
    if backend == "jax":
        arr = jnp.array(cast_values, dtype=dtype)
        if device is not None:
            arr = jax.device_put(arr, _resolve_jax_device(device))
        return arr
    raise ValueError(f"Unknown backend '{backend}'")


def pair_arrays(
    shape: tuple[int, ...],
    dtype: jnp.dtype,
    backend: str,
    device: str | None = None,
    *,
    delta: float = 0.2,
) -> tuple[np.ndarray | jnp.ndarray, np.ndarray | jnp.ndarray]:
    x = make_array(shape, dtype, backend, device, offset=0.0)
    y = make_array(shape, dtype, backend, device, offset=delta)
    return x, y


def assert_allclose(actual, expected, tol: float = TOLERANCE, err_msg="") -> None:
    np.testing.assert_allclose(
        np.asarray(actual), np.asarray(expected), rtol=tol, atol=tol, err_msg=err_msg
    )


def assert_softbool(values, atol: float = 1e-6) -> None:
    """Asserts that the given values are valid SoftBool outputs."""
    arr = np.asarray(values)
    assert np.all(arr >= -atol), f"SoftBool entries must be >= 0 (min={arr.min():.2e})"
    assert np.all(arr <= 1.0 + atol), (
        f"SoftBool entries must be <= 1 (max={arr.max():.2e})"
    )


def assert_simplex(values, axis: int = -1, atol: float = 1e-6) -> None:
    """Asserts that the given values form a simplex along the specified axis."""
    arr = np.asarray(values)
    assert arr.ndim >= 1, "SoftIndex outputs must have at least one dimension"
    assert np.all(arr >= -atol), "SoftIndex entries must be non-negative"
    sums = np.sum(arr, axis=axis)
    np.testing.assert_allclose(sums, 1.0, atol=atol)


def _axis_is_valid(shape, axis):
    if axis is None:
        return True
    return -len(shape) <= axis < len(shape)


def make_invalid_softindex(length: int = 3) -> jnp.ndarray:
    return jnp.ones((length,), dtype=jnp.float32) / length


def gradient_input(
    shape: tuple[int, ...], dtype: jnp.dtype = jnp.float32, device: str | None = None
) -> jnp.ndarray:
    base = _base_values(shape)
    arr = jnp.array(base, dtype=dtype)
    if device is not None:
        arr = jax.device_put(arr, _resolve_jax_device(device))
    return arr


# Mode constants
MODES_ELEMENTWISE = ("smooth", "c0", "c1", "_c1_pnorm", "c2", "_c2_pnorm")
MODES_ARRAYWISE = ("smooth", "c0", "c1", "c2", "hard", "_hard")


OT_SOLVER_TOLS = [
    {
        "lbfgs_tol": 1e-5,
        "lbfgs_max_iter": 10000,
        "sinkhorn_tol": 1e-3,
        "sinkhorn_max_iter": 10000,
    },
    {
        "lbfgs_tol": 1e-7,
        "lbfgs_max_iter": 20000,
        "sinkhorn_tol": 1e-5,
        "sinkhorn_max_iter": 20000,
    },
    {
        "lbfgs_tol": 1e-9,
        "lbfgs_max_iter": 50000,
        "sinkhorn_tol": 1e-7,
        "sinkhorn_max_iter": 50000,
    },
]


def ot_kwargs_for_method(method: str, softness: float) -> dict:
    """Return default ot_kwargs; tightened for near-hard OT to help convergence."""
    if method == "ot" and softness <= NEAR_HARD_SOFTNESS:
        return {
            "lbfgs_tol": 1e-7,
            "lbfgs_max_iter": 10000,
            "sinkhorn_tol": 1e-5,
            "sinkhorn_max_iter": 10000,
        }
    return {}


def _any_nan(out):
    """Check if any element in `out` is NaN, handling tuples of arrays."""
    if isinstance(out, tuple):
        return any(_any_nan(o) for o in out)
    if out is None:
        return False
    return bool(np.any(np.isnan(np.asarray(out))))


def call_with_ot_retry(fn, *args, method: str = "", tol: float = TOLERANCE, **kwargs):
    """Call a softjax function, retrying with tighter OT solver tolerance on failure.

    For non-OT methods, calls once and returns the result directly.
    For OT methods, tries progressively tighter solver settings if
    the result contains NaN or fails an allclose check against `expected`
    (if `expected` is provided in kwargs).
    """
    expected = kwargs.pop("_expected", None)
    check_fn = kwargs.pop("_check_fn", None)

    # Re-inject method into kwargs so fn receives it
    kwargs["method"] = method

    if method != "ot":
        return fn(*args, **kwargs)

    for solver_kwargs in OT_SOLVER_TOLS:
        kwargs["ot_kwargs"] = solver_kwargs
        out = fn(*args, **kwargs)
        if _any_nan(out):
            continue
        if expected is not None:
            try:
                assert_allclose(out, expected, tol=tol)
                return out
            except AssertionError:
                continue
        if check_fn is not None:
            try:
                check_fn(out)
                return out
            except AssertionError:
                continue
        return out

    # Return last result even if it didn't pass — let the caller's assert report the failure
    return out


def assert_finite(values, msg: str = "") -> None:
    """Asserts that all values are finite (no NaN or Inf)."""
    arr = np.asarray(values)
    assert np.all(np.isfinite(arr)), f"Non-finite values found. {msg}"


HARD_TOLERANCE = 1e-10


def assert_jax_parity(
    soft_output, jax_output, tol: float = HARD_TOLERANCE, msg: str = ""
) -> None:
    """Asserts that softjax hard-mode output matches JAX output."""
    assert_allclose(
        soft_output, jax_output, tol=tol, err_msg=f"JAX parity mismatch. {msg}"
    )


def assert_grad_matches_finite_diff(
    loss_fn,
    x: jnp.ndarray,
    eps: float = 1e-5,
    rtol: float = 1e-3,
    atol: float = 1e-4,
    msg: str = "",
) -> None:
    """Assert that jax.grad matches central finite differences.

    Uses float64 for numerical stability of finite differences.
    """
    x64 = x.astype(jnp.float64)
    analytic_grad = np.asarray(jax.grad(loss_fn)(x64))
    assert np.all(np.isfinite(analytic_grad)), f"Non-finite analytic gradient. {msg}"

    # Central finite differences
    x_np = np.asarray(x64)
    fd_grad = np.zeros_like(x_np)
    for idx in np.ndindex(x_np.shape):
        x_plus = x_np.copy()
        x_minus = x_np.copy()
        x_plus[idx] += eps
        x_minus[idx] -= eps
        fd_grad[idx] = (
            float(loss_fn(jnp.array(x_plus))) - float(loss_fn(jnp.array(x_minus)))
        ) / (2 * eps)

    np.testing.assert_allclose(
        analytic_grad,
        fd_grad,
        rtol=rtol,
        atol=atol,
        err_msg=f"Gradient vs finite diff mismatch. {msg}",
    )
