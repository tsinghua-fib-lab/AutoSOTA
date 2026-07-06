import operator
from typing import Literal

import equinox.internal as eqi
import jax
import jax.numpy as jnp
from jax import lax


def _validate_softness(softness: jnp.ndarray) -> None:
    """Raise ValueError if softness is not positive (skipped when traced)."""
    if isinstance(softness, jax.core.Tracer):
        return
    if float(softness) <= 0:
        raise ValueError(f"softness must be positive, got {softness}")


def _ensure_float(x: jnp.ndarray) -> jnp.ndarray:
    """Cast to default float dtype if not already floating point."""
    x = jnp.asarray(x)
    return x if jnp.issubdtype(x.dtype, jnp.floating) else x.astype(jnp.result_type(float))


def _canonicalize_shape(shape: int | tuple[int, ...] | None) -> tuple[int, ...] | None:
    """Canonicalize a user-provided shape without silently truncating dimensions."""
    if shape is None:
        return None
    if isinstance(shape, int):
        shape = (shape,)
    try:
        return tuple(operator.index(dim) for dim in shape)
    except TypeError as exc:
        raise TypeError(
            "Shapes must be 1D sequences of concrete values of integer type, "
            f"got {shape}."
        ) from exc


def _check_broadcast_shape(name: str, shape: tuple[int, ...], *param_shapes) -> None:
    if not param_shapes:
        return
    try:
        broadcast_shape = lax.broadcast_shapes(shape, *param_shapes)
    except ValueError as exc:
        raise ValueError(
            f"{name} parameter shapes must be broadcast-compatible with shape "
            f"argument, got parameter shapes {param_shapes} and shape argument "
            f"{shape}."
        ) from exc
    if broadcast_shape != shape:
        raise ValueError(
            f"{name} parameter shapes must be broadcast-compatible with shape "
            f"argument, and the result of broadcasting the shapes must equal "
            f"the shape argument, but got result {broadcast_shape} for shape "
            f"argument {shape}."
        )


def _standardize_and_squash(
    x: jnp.ndarray,
    axis: int = -1,
    eps: float = 1e-6,
    temperature: float = 1.0,
    return_mean_std: bool = False,
) -> jnp.ndarray:
    """
    1) standardize: (x - mean) / std  along `axis`
    2) squash: map to (0,1) (or approx [0,1]) with a smooth function
    """
    mean = jnp.mean(x, axis=axis, keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=axis, keepdims=True)
    std = jnp.sqrt(var + eps)

    z = (x - mean) / std
    z = z / temperature
    z = jax.nn.sigmoid(z)
    if return_mean_std:
        return z, mean, std
    else:
        return z


def _unsquash_and_destandardize(
    y: jnp.ndarray,
    mean: jnp.ndarray,
    std: jnp.ndarray,
    eps: float = 1e-10,
    temperature: float = 1.0,
) -> jnp.ndarray:
    """
    Inverse of _standardize_and_squash
    1) unsquash: map from (0,1) back to R with logit
    2) destandardize: y * std + mean along `axis`
    """
    safe_eps = jnp.maximum(eps, 10 * jnp.finfo(y.dtype).eps)
    y = jnp.clip(y, safe_eps, 1.0 - safe_eps)
    z = jnp.log(y / (1.0 - y))
    z = z * temperature
    x = z * std + mean
    return x


def _quantile_interpolation_params(
    q: jnp.ndarray,
    n: int,
    quantile_method: Literal["linear", "lower", "higher", "nearest", "midpoint"],
):
    # p in [0, n-1]
    p = q * (n - 1)

    if quantile_method == "linear":
        k = jnp.floor(p).astype(jnp.int32)
        a = p - k  # in [0,1)
        take_next = True
    elif quantile_method == "lower":
        k = jnp.floor(p).astype(jnp.int32)
        k = jnp.clip(k, 0, n - 1)
        a = jnp.zeros_like(p)
        take_next = False
    elif quantile_method == "higher":
        k = jnp.ceil(p).astype(jnp.int32)
        k = jnp.clip(k, 0, n - 1)
        a = jnp.zeros_like(p)
        take_next = False
    elif quantile_method == "nearest":
        # Round down on ties to mimic jax behavior
        flag = jnp.less_equal(p - jnp.floor(p), 0.5)
        k = jnp.where(flag, jnp.floor(p), jnp.ceil(p)).astype(jnp.int32)
        a = jnp.zeros_like(p)
        take_next = False
    elif quantile_method == "midpoint":
        k = jnp.floor(p).astype(jnp.int32)
        a = jnp.full_like(p, 0.5)
        # Special-case when p is an integer: midpoint should equal that exact order stat
        is_int = jnp.isclose(p, jnp.round(p))
        a = jnp.where(is_int, 0.0, a)
        take_next = True
    else:
        raise ValueError(f"Unknown quantile_method={quantile_method}")
    return k, a, take_next


def _map_in_chunks(f, xs, chunk_size):
    """Map ``f`` row-wise over axis 0 of ``xs`` using ``lax.scan`` over chunks.

    ``f`` receives a chunk of shape ``(chunk_size, *rest)`` and must return
    ``(chunk_size, *out_rest)``.  Uses ``jax.checkpoint`` for O(n) backward memory.
    """
    n = xs.shape[0]
    if chunk_size >= n:
        return f(xs)
    remainder = n % chunk_size
    if remainder:
        pad_size = chunk_size - remainder
        padding = jnp.zeros((pad_size, *xs.shape[1:]), dtype=xs.dtype)
        xs_padded = jnp.concatenate([xs, padding], axis=0)
    else:
        xs_padded = xs
    n_padded = xs_padded.shape[0]
    xs_chunked = xs_padded.reshape(n_padded // chunk_size, chunk_size, *xs.shape[1:])
    f_remat = jax.checkpoint(f)
    _, ys = jax.lax.scan(
        lambda _, chunk: (None, f_remat(chunk)), None, xs_chunked
    )
    ys = ys.reshape(n_padded, *ys.shape[2:])
    return ys[:n]


def _reduce_in_chunks(f, xs, chunk_size):
    """Apply ``f`` to chunks of ``xs`` along axis 0 and sum the results.

    ``f`` receives a chunk of shape ``(chunk_size, *rest)`` and must return a result whose shape does **not** include the chunk dimension.
    Remainder rows are zero-padded; the correction ``f(zeros)`` is subtracted.
    """
    n = xs.shape[0]
    if chunk_size >= n:
        return f(xs)
    remainder = n % chunk_size
    if remainder:
        pad_size = chunk_size - remainder
        padding = jnp.zeros((pad_size, *xs.shape[1:]), dtype=xs.dtype)
        xs_padded = jnp.concatenate([xs, padding], axis=0)
    else:
        xs_padded = xs
    n_padded = xs_padded.shape[0]
    xs_chunked = xs_padded.reshape(n_padded // chunk_size, chunk_size, *xs.shape[1:])

    out_struct = jax.eval_shape(f, xs_chunked[0])
    init = jnp.zeros(out_struct.shape, dtype=out_struct.dtype)

    f_remat = jax.checkpoint(f)

    def body(acc, chunk):
        return acc + f_remat(chunk), None

    result, _ = eqi.scan(body, init, xs_chunked, kind="checkpointed")
    # Undo contribution of zero-padding in the last chunk.
    if remainder:
        zero_pad = jnp.zeros((pad_size, *xs.shape[1:]), dtype=xs.dtype)
        result = result - f(zero_pad)
    return result



def _canonicalize_axis(axis: int | None, num_dims: int) -> int:
    if axis is None:
        raise ValueError("axis must be specified")
    if not -num_dims <= axis < num_dims:
        raise ValueError(
            f"axis {axis} is out of bounds for array of dimension {num_dims}"
        )
    if axis < 0:
        axis += num_dims
    return axis
