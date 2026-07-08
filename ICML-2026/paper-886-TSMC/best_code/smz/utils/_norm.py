from typing import Callable as _Callable

import jax as _jax
import jax.numpy as _jnp


Normalizer = _Callable[[_jax.Array], _jax.Array]


def unnormed(x: _jax.Array, *args, **kwargs) -> _jax.Array:
    return x


def logmax_scaled(x: _jax.Array, t: float, bound: float = 0.0) -> _jax.Array:
    """Scale x by its log(1 + maximum magnitude).

    This scale increases the magnitude of the temperature `t` based on the
    extreme values of `x`.

    In contrast to minmax, this rescaling is unbiased in the limit of
    |x_i| -> infty. Meaning if a logit of x is infinitely better than all the
    other logits, this function will not bias the solution to choose other
    highly suboptimal actions. This does not happen with e.g., `minmax` scaling
    since the logits will always be scaled to [low, high].
    """
    high = x.max()
    x = x - high

    max_magnitude = _jnp.abs(x.min())
    max_magnitude = _jnp.clip(max_magnitude, a_min=bound)

    return _jax.nn.log_softmax(x / (t * (1 + _jnp.log1p(max_magnitude))))


def minmax(x: _jax.Array, jitter: float = 1e-6) -> _jax.Array:
    normed = (x - x.min()) / (x.max() - x.min() + jitter)

    return _jax.lax.select(
        _jnp.isclose(normed, normed[0]).all(), _jnp.zeros_like(x), normed
    )


def standardize(x: _jax.Array, jitter: float = 1e-6) -> _jax.Array:
    normed = (x - x.mean()) / (x.std() + jitter)

    return _jax.lax.select(
        _jnp.isclose(normed, normed[0]).all(), _jnp.zeros_like(x), normed
    )


def logsoftmax(x: _jax.Array, t: _jax.Array) -> _jax.Array:
    return _jax.nn.log_softmax(x / t)


def logsoftmax_minmax(x: _jax.Array, t: _jax.Array) -> _jax.Array:
    return logsoftmax(minmax(x), t)


def logsoftmax_standardize(x: _jax.Array, t: _jax.Array) -> _jax.Array:
    return logsoftmax(standardize(x), t)
