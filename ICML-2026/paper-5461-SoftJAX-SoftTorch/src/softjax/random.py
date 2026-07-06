import inspect
import numbers
from typing import Literal

import jax
import jax.numpy as jnp
from jax import lax, tree_util as jtu
from jaxtyping import Array, Float

from softjax.functions import (
    _argtop_k,
    _maybe_log_soft_index,
    _validate_log_prob_request,
    argmax,
    argsort,
    less,
    SoftBool,
    SoftIndex,
    take_along_axis,
)
from softjax.straight_through import _replace_value_keep_grad
from softjax.utils import (
    _canonicalize_axis,
    _canonicalize_shape,
    _check_broadcast_shape,
)


RandomMode = Literal["hard", "_hard", "smooth", "c0", "c1", "c2"]
ArgMethod = Literal["ot", "softsort", "neuralsort", "sorting_network"]
_private_hard_st_cache = {}


def _validate_noise(noise: str, expected: str) -> None:
    if noise != expected:
        raise ValueError(
            f"Invalid noise: {noise}. Only noise='{expected}' is supported."
        )


def _static_nonnegative_int(name: str, value) -> int:
    if isinstance(value, numbers.Real):
        value_float = float(value)
    else:
        value_arr = jnp.asarray(value)
        if value_arr.shape != ():
            raise NotImplementedError(
                f"softjax.random.{name} currently supports only scalar static counts "
                "in soft modes."
            )
        try:
            value_float = float(value_arr)
        except TypeError as exc:
            raise NotImplementedError(
                f"softjax.random.{name} requires a concrete scalar count in soft modes."
            ) from exc
    value_int = int(value_float)
    if value_int < 0 or value_float != value_int:
        raise ValueError(f"{name} count must be a nonnegative integer, got {value}.")
    return value_int


def _as_float_dtype(dtype, fallback) -> jnp.dtype:
    if dtype is None:
        return jnp.result_type(fallback, float)
    dtype = jnp.dtype(dtype)
    if not jnp.issubdtype(dtype, jnp.floating):
        raise ValueError(f"dtype must be a float dtype, got {dtype}.")
    return dtype


def _finite_logits_for_gumbel_perturbation(logits: Array, axis: int) -> Array:
    finite = jnp.isfinite(logits)
    finite_logits = jnp.where(finite, logits, jnp.inf)
    min_finite = jnp.min(finite_logits, axis=axis, keepdims=True)
    has_finite = jnp.any(finite, axis=axis, keepdims=True)
    min_finite = jnp.where(has_finite, min_finite, jnp.zeros((), dtype=logits.dtype))
    finfo = jnp.finfo(logits.dtype)
    base_gap = jnp.asarray(min(1_000.0, float(finfo.max) / 4.0), dtype=logits.dtype)
    max_gap = jnp.asarray(float(finfo.max) / 4.0, dtype=logits.dtype)
    gap = jnp.minimum(jnp.maximum(jnp.abs(min_finite), base_gap), max_gap)
    floor = jnp.maximum(min_finite - gap, jnp.asarray(finfo.min, dtype=logits.dtype))
    return jnp.where(finite, logits, floor)


def _cached_private_hard_st(fn):
    if fn in _private_hard_st_cache:
        return _private_hard_st_cache[fn]

    sig = inspect.signature(fn)
    mode_param = sig.parameters.get("mode")
    mode_default = mode_param.default
    mode_idx = list(sig.parameters.keys()).index("mode")

    def wrapped(*args, **kwargs):
        if len(args) > mode_idx:
            mode = args[mode_idx]
            args = args[:mode_idx] + args[mode_idx + 1 :]
        else:
            mode = kwargs.pop("mode", mode_default)
        fw_y = fn(*args, **kwargs, mode="_hard")
        bw_y = fn(*args, **kwargs, mode=mode)
        fw_leaves, fw_treedef = jtu.tree_flatten(fw_y, is_leaf=lambda x: x is None)
        bw_leaves, _ = jtu.tree_flatten(bw_y, is_leaf=lambda x: x is None)
        out_leaves = [
            _replace_value_keep_grad(f, b) for f, b in zip(fw_leaves, bw_leaves)
        ]
        return jtu.tree_unflatten(fw_treedef, out_leaves)

    _private_hard_st_cache[fn] = wrapped
    return wrapped


def _categorical_batch_shape(logits: Array, axis: int) -> tuple[int, ...]:
    return tuple(dim for i, dim in enumerate(logits.shape) if i != axis)


def _categorical_gumbel_perturbation(
    key: Array,
    logits: Float[Array, "..."],
    axis: int,
    shape: tuple[int, ...] | None,
    rng_mode: Literal["high", "low"] | None,
) -> tuple[Array, int]:
    logits = jnp.asarray(logits)
    batch_shape = _categorical_batch_shape(logits, axis)
    if shape is None:
        shape = batch_shape
    else:
        _check_broadcast_shape("categorical", shape, batch_shape)

    shape_prefix = shape[: len(shape) - len(batch_shape)]
    axis = axis - logits.ndim if axis >= 0 else axis
    logits_shape = list(shape[len(shape) - len(batch_shape) :])
    logits_shape.insert(axis % logits.ndim, logits.shape[axis])

    gumbel = jax.random.gumbel(
        key, (*shape_prefix, *logits_shape), logits.dtype, mode=rng_mode
    )  # (..., n, ...)
    logits = lax.expand_dims(
        logits, tuple(range(len(shape_prefix)))
    )  # (1..., 1..., ..., n, ...)
    logits = _finite_logits_for_gumbel_perturbation(logits, axis)
    return gumbel + logits, axis  # (..., n, ...)


def _categorical_without_replacement(
    key: Array,
    logits: Float[Array, "..."],
    axis: int,
    shape: tuple[int, ...] | None,
    rng_mode: Literal["high", "low"] | None,
    softness: float | Array,
    mode: RandomMode,
    method: ArgMethod,
    standardize: bool,
    ot_kwargs: dict | None,
    return_log_probs: bool,
    log_prob_eps: float | Array | None,
) -> SoftIndex:
    batch_shape = _categorical_batch_shape(logits, axis)
    if shape is None:
        shape = batch_shape
    else:
        _check_broadcast_shape("categorical", shape, batch_shape)
    shape_prefix = shape[: len(shape) - len(batch_shape)]
    k = 1
    for dim in shape_prefix:
        k *= dim

    num_classes = logits.shape[axis]
    if k > num_classes:
        raise ValueError(
            f"Number of samples without replacement ({k}) cannot exceed number "
            f"of categories ({num_classes})."
        )

    logits = _finite_logits_for_gumbel_perturbation(logits, axis)
    perturbed = logits + jax.random.gumbel(
        key, logits.shape, logits.dtype, mode=rng_mode
    )  # (..., n, ...)
    perturbed_last = jnp.moveaxis(perturbed, axis, -1)  # (..., ..., n)
    soft_index = _argtop_k(
        perturbed_last,
        k=k,
        axis=-1,
        softness=softness,
        mode=mode,
        method=method,
        standardize=standardize,
        ot_kwargs=ot_kwargs,
        return_log_probs=return_log_probs,
        log_prob_eps=log_prob_eps,
    )  # (..., k, [n])

    if shape_prefix:
        soft_index = jnp.moveaxis(soft_index, -2, 0)  # (k, ..., [n])
        soft_index = jnp.reshape(
            soft_index, (*shape_prefix, *batch_shape, num_classes)
        )  # (..., [n])
    else:
        soft_index = jnp.squeeze(soft_index, axis=-2)  # (..., [n])
    return soft_index


def categorical(
    key: Array,
    logits: Float[Array, "..."],
    axis: int = -1,
    shape: tuple[int, ...] | None = None,
    replace: bool = True,
    rng_mode: Literal["high", "low"] | None = None,
    softness: float | Array = 0.1,
    mode: RandomMode = "smooth",
    method: ArgMethod = "softsort",
    noise: Literal["gumbel"] = "gumbel",
    standardize: bool = False,
    ot_kwargs: dict | None = None,
    return_log_probs: bool = False,
    log_prob_eps: float | Array | None = None,
) -> SoftIndex:
    """Soft replacement for [`jax.random.categorical`](https://docs.jax.dev/en/latest/_autosummary/jax.random.categorical.html).

    **Arguments:**

    - `key`: JAX PRNG key.
    - `logits`: Unnormalized log probabilities. `softmax(logits, axis)` gives the categorical probabilities.
    - `axis`: Axis along which logits belong to the same categorical distribution.
    - `shape`: Optional output sample shape, following `jax.random.categorical`.
    - `replace`: Whether to sample with replacement. Soft modes use Gumbel-max with replacement and Gumbel top-k without replacement.
    - `rng_mode`: JAX Gumbel sampler precision mode, `"high"` or `"low"`. If `None`, JAX chooses its configured default.
    - `softness`: Softness passed to [`softjax.argmax`][].
    - `mode`: Relaxation mode. `"hard"` calls `jax.random.categorical` and returns a one-hot `SoftIndex`; `"_hard"` mirrors the Gumbel-max/top-k implementation via SoftJAX primitives; soft modes apply SoftJAX relaxations to Gumbel-perturbed logits.
    - `method`: Method passed to [`softjax.argmax`][].
    - `noise`: Perturbation distribution. Currently only `"gumbel"` is supported.
    - `standardize`: Whether to standardize perturbed logits before applying [`softjax.argmax`][]. Defaults to `False` so logits keep their probabilistic scale.
    - `ot_kwargs`: Optional keyword arguments for OT-based [`softjax.argmax`][].
    - `return_log_probs`: If True, returns log relaxed sample weights instead of probabilities. These are not categorical log-likelihoods under the original logits.
    - `log_prob_eps`: Optional probability floor used only with `return_log_probs=True`.

    **Returns:**

    A `SoftIndex` over sampled categories. In hard mode this is a one-hot encoding of the integer categories returned by `jax.random.categorical`; in soft modes it is a relaxed stochastic argmax sample.
    """
    _validate_noise(noise, "gumbel")
    _validate_log_prob_request(return_log_probs, log_prob_eps)
    shape = _canonicalize_shape(shape)
    logits = jnp.asarray(logits)
    axis = _canonicalize_axis(axis, logits.ndim)
    num_classes = logits.shape[axis]

    if mode == "hard":
        indices = jax.random.categorical(
            key, logits, axis=axis, shape=shape, replace=replace, mode=rng_mode
        )
        soft_index = jax.nn.one_hot(indices, num_classes=num_classes, axis=-1)
        return _maybe_log_soft_index(
            soft_index, return_log_probs, already_log=False, log_prob_eps=log_prob_eps
        )

    if not replace:
        return _categorical_without_replacement(
            key,
            logits,
            axis=axis,
            shape=shape,
            rng_mode=rng_mode,
            softness=softness,
            mode=mode,
            method=method,
            standardize=standardize,
            ot_kwargs=ot_kwargs,
            return_log_probs=return_log_probs,
            log_prob_eps=log_prob_eps,
        )

    perturbed, perturbed_axis = _categorical_gumbel_perturbation(
        key, logits, axis=axis, shape=shape, rng_mode=rng_mode
    )  # (..., n, ...)
    return argmax(
        perturbed,
        axis=perturbed_axis,
        softness=softness,
        mode=mode,
        method=method,
        standardize=standardize,
        ot_kwargs=ot_kwargs,
        return_log_probs=return_log_probs,
        log_prob_eps=log_prob_eps,
    )


def _choice_select(a: Array, soft_index: SoftIndex, axis: int) -> Array:
    prefix_ndim = axis
    suffix_ndim = a.ndim - axis - 1
    sample_ndim = soft_index.ndim - 1
    result = jnp.tensordot(
        a, soft_index, axes=([axis], [-1])
    )  # (prefix..., suffix..., sample...)
    permutation = (
        tuple(range(prefix_ndim))
        + tuple(
            range(prefix_ndim + suffix_ndim, prefix_ndim + suffix_ndim + sample_ndim)
        )
        + tuple(range(prefix_ndim, prefix_ndim + suffix_ndim))
    )
    return jnp.transpose(result, permutation)  # (prefix..., sample..., suffix...)


def _choice_logits(p: Array | None, n_inputs: int) -> Array:
    if p is None:
        return jnp.zeros((n_inputs,), dtype=jnp.result_type(float))

    p = jnp.asarray(p)
    if p.shape != (n_inputs,):
        raise ValueError(
            "p must be None or a 1D vector with the same size as a.shape[axis]. "
            f"p has shape {p.shape} and a.shape[axis] is {n_inputs}."
        )
    return jnp.log(p)


def choice(
    key: Array,
    a: int | Array,
    shape: tuple[int, ...] = (),
    replace: bool = True,
    p: Float[Array, "..."] | None = None,
    axis: int = 0,
    rng_mode: Literal["high", "low"] | None = None,
    softness: float | Array = 0.1,
    mode: RandomMode = "smooth",
    method: ArgMethod = "softsort",
    noise: Literal["gumbel"] = "gumbel",
    standardize: bool = False,
    ot_kwargs: dict | None = None,
) -> Array:
    """Soft replacement for [`jax.random.choice`](https://docs.jax.dev/en/latest/_autosummary/jax.random.choice.html).

    **Arguments:**

    - `key`: JAX PRNG key.
    - `a`: Integer population size or array to sample from.
    - `shape`: Output sample shape, following `jax.random.choice`.
    - `replace`: Whether to sample with replacement.
    - `p`: Optional 1D probabilities or nonnegative weights over entries along `axis`.
    - `axis`: Axis of `a` along which selection is performed.
    - `rng_mode`: Gumbel sampler precision mode used by soft modes.
    - `softness`: Softness passed to categorical/permutation relaxations.
    - `mode`: Relaxation mode. `"hard"` calls `jax.random.choice`; soft modes sample a relaxed index and return the corresponding soft value.
    - `method`: Method passed to the soft categorical/top-k relaxation.
    - `noise`: Perturbation distribution. Currently only `"gumbel"` is supported.
    - `standardize`: Whether to standardize random scores before applying SoftJAX relaxations.
    - `ot_kwargs`: Optional keyword arguments for OT-based relaxations.

    **Returns:**

    A soft sample from `a`. Soft modes use a stochastic relaxed categorical index, so the result is random and not merely the expectation under `p`.
    """
    _validate_noise(noise, "gumbel")
    shape = _canonicalize_shape(shape)
    if shape is None:
        raise TypeError("shape must be a tuple of integers, got None.")

    if mode == "hard":
        return jax.random.choice(
            key, a, shape=shape, replace=replace, p=p, axis=axis, mode=rng_mode
        )

    a = jnp.asarray(a)
    n_draws = 1
    for dim in shape:
        n_draws *= dim
    if a.ndim == 0:
        n_inputs = int(a)
        if n_draws == 0:
            return jnp.zeros(shape, dtype=a.dtype)
        if n_inputs <= 0:
            raise ValueError("a must be greater than 0 unless no samples are taken.")
        values = jnp.arange(n_inputs, dtype=a.dtype)  # (n,)
        axis = 0
    else:
        axis = _canonicalize_axis(axis, a.ndim)
        n_inputs = a.shape[axis]
        values = a  # (prefix..., n, suffix...)

    if n_draws == 0:
        result_shape = (
            shape if a.ndim == 0 else a.shape[:axis] + shape + a.shape[axis + 1 :]
        )
        return jnp.zeros(result_shape, dtype=a.dtype)
    if not replace and n_draws > n_inputs:
        raise ValueError(
            f"Cannot take a larger sample (size {n_draws}) than population "
            f"(size {n_inputs}) when 'replace=False'."
        )

    logits = _choice_logits(p, n_inputs)
    soft_index = categorical(
        key,
        logits,
        shape=shape,
        replace=replace,
        rng_mode=rng_mode,
        softness=softness,
        mode=mode,
        method=method,
        noise=noise,
        standardize=standardize,
        ot_kwargs=ot_kwargs,
    )  # (sample..., [n])
    return _choice_select(values, soft_index, axis=axis)


def _bernoulli_uniform(
    key: Array,
    p: Float[Array, "..."],
    shape: tuple[int, ...] | None,
    rng_mode: Literal["high", "low"],
) -> Array:
    p = jnp.asarray(p)
    if shape is None:
        shape = p.shape
    else:
        _check_broadcast_shape("bernoulli", shape, p.shape)

    if rng_mode == "high":
        u1, u2 = jax.random.uniform(key, (2, *shape), p.dtype)
        u2 *= 2 ** -jnp.finfo(p.dtype).nmant
        return u1 + u2
    if rng_mode == "low":
        return jax.random.uniform(key, shape, p.dtype)
    raise ValueError(f"got rng_mode={rng_mode!r}, expected 'high' or 'low'")


def bernoulli(
    key: Array,
    p: Float[Array, "..."] = 0.5,
    shape: tuple[int, ...] | None = None,
    rng_mode: Literal["high", "low"] = "low",
    softness: float | Array = 0.1,
    mode: RandomMode = "smooth",
    noise: Literal["uniform"] = "uniform",
) -> SoftBool:
    """Soft replacement for [`jax.random.bernoulli`](https://docs.jax.dev/en/latest/_autosummary/jax.random.bernoulli.html).

    **Arguments:**

    - `key`: JAX PRNG key.
    - `p`: Bernoulli probability. Must be broadcast-compatible with `shape`.
    - `shape`: Optional output sample shape, following `jax.random.bernoulli`.
    - `rng_mode`: JAX Bernoulli sampler precision mode, `"high"` or `"low"`.
    - `softness`: Softness passed to the soft threshold.
    - `mode`: Relaxation mode. `"hard"` calls `jax.random.bernoulli`; `"_hard"` mirrors JAX threshold sampling; soft modes apply a soft threshold to the same uniform draw.
    - `noise`: Threshold noise distribution. Currently only `"uniform"` is supported.

    **Returns:**

    A numeric `SoftBool`. In hard mode this is the boolean JAX sample cast to the probability dtype; in soft modes it is a relaxed uniform-threshold sample.
    """
    _validate_noise(noise, "uniform")
    shape = _canonicalize_shape(shape)
    p = jnp.asarray(p)

    if mode == "hard":
        sample = jax.random.bernoulli(key, p=p, shape=shape, mode=rng_mode)
        return sample.astype(p.dtype)

    u = _bernoulli_uniform(key, p, shape=shape, rng_mode=rng_mode)
    if mode == "_hard":
        return jnp.less(u, p).astype(p.dtype)
    return less(u, p, softness=softness, mode=mode)


def rademacher(
    key: Array,
    shape: tuple[int, ...] = (),
    dtype=None,
    softness: float | Array = 0.1,
    mode: RandomMode = "smooth",
    noise: Literal["uniform"] = "uniform",
) -> Array:
    """Soft replacement for [`jax.random.rademacher`](https://docs.jax.dev/en/latest/_autosummary/jax.random.rademacher.html).

    Soft modes implement a Rademacher sample as `2 * bernoulli(0.5) - 1`.
    """
    _validate_noise(noise, "uniform")
    shape = _canonicalize_shape(shape)
    if shape is None:
        raise TypeError("shape must be a tuple of integers, got None.")
    if mode == "hard":
        return jax.random.rademacher(key, shape=shape, dtype=dtype)

    out_dtype = jnp.result_type(float) if dtype is None else jnp.dtype(dtype)
    if not jnp.issubdtype(out_dtype, jnp.floating):
        out_dtype = jnp.result_type(float)
    probs = jnp.full(shape, 0.5, dtype=out_dtype)  # (...,)
    return (
        2.0
        * bernoulli(
            key,
            probs,
            shape=shape,
            softness=softness,
            mode=mode,
            noise=noise,
        )
        - 1.0
    )


def permutation(
    key: Array,
    x: int | Array,
    axis: int = 0,
    independent: bool = False,
    *,
    out_sharding=None,
    rng_mode: Literal["high", "low"] | None = None,
    softness: float | Array = 0.1,
    mode: RandomMode = "smooth",
    method: ArgMethod = "neuralsort",
    noise: Literal["gumbel"] = "gumbel",
    standardize: bool = False,
    ot_kwargs: dict | None = None,
) -> Array:
    """Soft replacement for [`jax.random.permutation`](https://docs.jax.dev/en/latest/_autosummary/jax.random.permutation.html).

    Soft modes sort random Gumbel scores and apply the resulting soft permutation to `x`.
    """
    _validate_noise(noise, "gumbel")
    if mode == "hard":
        return jax.random.permutation(
            key, x, axis=axis, independent=independent, out_sharding=out_sharding
        )
    if out_sharding is not None:
        raise NotImplementedError("out_sharding is supported only with mode='hard'.")

    x = jnp.asarray(x)
    if x.ndim == 0:
        axis = _canonicalize_axis(axis, 1)
        if not jnp.issubdtype(x.dtype, jnp.integer):
            raise TypeError("x must be an integer or at least 1-dimensional.")
        n_inputs = int(x)
        values = jnp.arange(n_inputs, dtype=x.dtype)  # (n,)
    else:
        values = x  # (..., n, ...)
        axis = _canonicalize_axis(axis, values.ndim)
        n_inputs = values.shape[axis]
    if n_inputs <= 0:
        return values

    score_dtype = (
        values.dtype
        if jnp.issubdtype(values.dtype, jnp.floating)
        else jnp.result_type(float)
    )
    if independent or values.ndim == 1:
        scores = jax.random.gumbel(
            key, values.shape, score_dtype, mode=rng_mode
        )  # (..., n, ...)
        soft_index = argsort(
            scores,
            axis=axis,
            descending=True,
            softness=softness,
            mode=mode,
            method=method,
            standardize=standardize,
            ot_kwargs=ot_kwargs,
        )  # (..., n, ..., [n])
    else:
        scores = jax.random.gumbel(key, (n_inputs,), score_dtype, mode=rng_mode)  # (n,)
        shared_index = argsort(
            scores,
            axis=0,
            descending=True,
            softness=softness,
            mode=mode,
            method=method,
            standardize=standardize,
            ot_kwargs=ot_kwargs,
        )  # (n, [n])
        soft_shape = [1] * values.ndim + [n_inputs]
        soft_shape[axis] = n_inputs
        soft_index = jnp.reshape(shared_index, soft_shape)  # (1..., n, 1..., [n])
    return take_along_axis(values, soft_index, axis=axis)


def binomial(
    key: Array,
    n: Float[Array, "..."],
    p: Float[Array, "..."],
    shape: tuple[int, ...] | None = None,
    dtype=None,
    softness: float | Array = 0.1,
    mode: RandomMode = "smooth",
    noise: Literal["uniform"] = "uniform",
) -> Array:
    """Soft replacement for [`jax.random.binomial`](https://docs.jax.dev/en/latest/_autosummary/jax.random.binomial.html).

    Soft modes currently support scalar static `n` and sum `n` independent relaxed Bernoulli samples.
    """
    _validate_noise(noise, "uniform")
    if mode == "hard":
        return jax.random.binomial(key, n=n, p=p, shape=shape, dtype=dtype)

    n_int = _static_nonnegative_int("binomial", n)
    shape = _canonicalize_shape(shape)
    p = jnp.asarray(p)
    out_dtype = _as_float_dtype(dtype, p)
    if shape is None:
        shape = p.shape
    else:
        _check_broadcast_shape("binomial", shape, p.shape)
    p = jnp.broadcast_to(p, shape).astype(out_dtype)  # (...,)
    if n_int == 0:
        return jnp.zeros(shape, dtype=out_dtype)

    keys = jax.random.split(key, n_int)  # (n,)
    samples = jax.vmap(
        lambda subkey: bernoulli(
            subkey,
            p,
            shape=shape,
            softness=softness,
            mode=mode,
            noise=noise,
        )
    )(keys)  # (n, ...)
    return jnp.sum(samples, axis=0).astype(out_dtype)


def multinomial(
    key: Array,
    n: Float[Array, "..."],
    p: Float[Array, "..."],
    *,
    shape: tuple[int, ...] | None = None,
    dtype=None,
    unroll: int | bool = 1,
    softness: float | Array = 0.1,
    mode: RandomMode = "smooth",
    method: ArgMethod = "softsort",
    noise: Literal["gumbel"] = "gumbel",
    standardize: bool = False,
    ot_kwargs: dict | None = None,
) -> Array:
    """Soft replacement for [`jax.random.multinomial`](https://docs.jax.dev/en/latest/_autosummary/jax.random.multinomial.html).

    Soft modes currently support scalar static `n` and sum `n` independent relaxed categorical samples.
    """
    _validate_noise(noise, "gumbel")
    if mode == "hard":
        return jax.random.multinomial(
            key, n=n, p=p, shape=shape, dtype=dtype, unroll=unroll
        )

    n_int = _static_nonnegative_int("multinomial", n)
    shape = _canonicalize_shape(shape)
    p = jnp.asarray(p)
    out_dtype = _as_float_dtype(dtype, p)
    if shape is None:
        shape = p.shape
    else:
        _check_broadcast_shape("multinomial", shape, p.shape)
    p = jnp.broadcast_to(p, shape).astype(out_dtype)  # (..., outcomes)
    if p.ndim == 0:
        raise ValueError("p must have at least one outcome axis.")
    if n_int == 0:
        return jnp.zeros(shape, dtype=out_dtype)

    keys = jax.random.split(key, n_int)  # (n,)
    logits = jnp.log(p)  # (..., outcomes)
    batch_shape = shape[:-1]
    samples = jax.vmap(
        lambda subkey: categorical(
            subkey,
            logits,
            axis=-1,
            shape=batch_shape,
            softness=softness,
            mode=mode,
            method=method,
            noise=noise,
            standardize=standardize,
            ot_kwargs=ot_kwargs,
        )
    )(keys)  # (n, ..., outcomes)
    return jnp.sum(samples, axis=0).astype(out_dtype)


def categorical_st(*args, **kwargs):
    """Straight-through version of [`softjax.random.categorical`][].

    Returns a hard one-hot categorical sample during the forward pass, while gradients are computed through the requested soft stochastic argmax relaxation.
    """
    return _cached_private_hard_st(categorical)(*args, **kwargs)


def bernoulli_st(*args, **kwargs):
    """Straight-through version of [`softjax.random.bernoulli`][].

    Returns a hard numeric Bernoulli sample during the forward pass, while gradients are computed through the requested soft threshold relaxation.
    """
    return _cached_private_hard_st(bernoulli)(*args, **kwargs)


def choice_st(*args, **kwargs):
    """Straight-through version of [`softjax.random.choice`][].

    Uses the internal `_hard` path for the forward pass so hard and soft branches share the same Gumbel perturbation.
    """
    return _cached_private_hard_st(choice)(*args, **kwargs)


def rademacher_st(*args, **kwargs):
    """Straight-through version of [`softjax.random.rademacher`][]."""
    return _cached_private_hard_st(rademacher)(*args, **kwargs)


def permutation_st(*args, **kwargs):
    """Straight-through version of [`softjax.random.permutation`][].

    Uses the internal `_hard` path for the forward pass so hard and soft branches share the same random scores.
    """
    return _cached_private_hard_st(permutation)(*args, **kwargs)


def binomial_st(*args, **kwargs):
    """Straight-through version of [`softjax.random.binomial`][].

    Uses the internal `_hard` path for the forward pass so hard and soft branches share the same Bernoulli draws.
    """
    return _cached_private_hard_st(binomial)(*args, **kwargs)


def multinomial_st(*args, **kwargs):
    """Straight-through version of [`softjax.random.multinomial`][].

    Uses the internal `_hard` path for the forward pass so hard and soft branches share the same categorical draws.
    """
    return _cached_private_hard_st(multinomial)(*args, **kwargs)


__all__ = [
    "bernoulli",
    "bernoulli_st",
    "binomial",
    "binomial_st",
    "categorical",
    "categorical_st",
    "choice",
    "choice_st",
    "multinomial",
    "multinomial_st",
    "permutation",
    "permutation_st",
    "rademacher",
    "rademacher_st",
]
