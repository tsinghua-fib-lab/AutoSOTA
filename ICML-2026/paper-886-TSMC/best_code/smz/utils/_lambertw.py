"""Module for a jax-based implementation of the Lambert W_0 function

We only implement this function for the zeroth branch for real numbers.

This module also provides a heuristic approximation using a softplus function
with optimized scale and shift parameters.

See Also
--------
https://en.wikipedia.org/wiki/Lambert_W_function
"""

from functools import partial

import jax
import jax.numpy as jnp


def _real_lambertw_recursion(w: jax.Array, x: jax.Array) -> jax.Array:
    # https://en.wikipedia.org/wiki/Lambert_W_function#Numerical_evaluation
    return w / (1+w) * (1+jnp.log(x / w))


@partial(jax.custom_jvp, nondiff_argnums=(1,))
def _lambertwk0(x, max_steps=5):

    w_0 = jax.lax.select(
        x > jnp.e,
        jnp.log(x) - jnp.log(jnp.log(x)),
        x / jnp.e
    )
    w_0 = jax.lax.select(
        x > 0,
        w_0,
        jnp.e * x / (1 + jnp.e * x + jnp.sqrt(1 + jnp.e * x)) * jnp.log(
            1 + jnp.sqrt(1 + jnp.e * x))
    )

    w, _ = jax.lax.scan(
        lambda carry, _: (_real_lambertw_recursion(carry, x),)*2,
        w_0,
        xs=None, length=max_steps
    )

    w = jax.lax.select(
        jnp.isclose(x, 0.0),
        0.0,
        w
    )

    return w


@_lambertwk0.defjvp
def _lambertw_jvp(max_steps, primals, tangents):
    # Note: All branches for lambert W satisfy this JVP.
    x, = primals
    t, = tangents

    y = _lambertwk0(x, max_steps)
    dydx = 1 / (x + jnp.exp(y))

    jvp = jax.lax.select(
        jnp.isclose(x, -1/jnp.e),
        jnp.nan,
        dydx * t
    )

    return y, jvp


@jnp.vectorize
def lambertw(
        x: jax.typing.ArrayLike,
        k: int = 0, max_steps: int = 5
) -> jax.Array:
    # https://en.wikipedia.org/wiki/Lambert_W_function#Numerical_evaluation
    # Uses Iacono and Boyd's recursion formula
    # max_steps = 10 guarantees ~ < 1e-50 error for all init-points
    # I.e., with float32 more than 10 steps does not give more accuracy.

    if k != 0:
        raise NotImplementedError()

    return _lambertwk0(x, max_steps=max_steps)


def _softplus_lambertw_exp(
        x: jax.typing.ArrayLike,
        a: float = 0.11775704, b: float = 1.1418955
) -> jax.Array:
    """Optimized Softplus approximation to W_0(exp(x))."""
    return (jnp.logaddexp(jnp.log1p(a) + x, jnp.log(b)) -
            jnp.log1p(jnp.logaddexp(jnp.log(a) + x, jnp.log(b))))


def log_lambertw_exp(
        x: jax.typing.ArrayLike,
        a: float = 0.11775704, b: float = 1.1418955,
        low: float = -2.8378477
) -> jax.Array:
    """log of `lambertw_exp` and using W_0(x) \approx x for x -> 0.

    Choices for hyperparameters:
    The values of a, b were found using a gradient based optimizer using a
    l2-error between the true functions lambertw(exp(x)) and the approximation
    over a bounded input range.

    The value for low was found using a grid-search on the l2 error between
    log( lambertw(exp(x) ) and the log of our approximation. This was done
    after optimization of `a` and `b`, since `low` is non-differentiable.
    """
    sp_approx = _softplus_lambertw_exp(x, a=a, b=b)
    return jax.lax.select(
        x < low, jnp.asarray(x, dtype=sp_approx.dtype), jnp.log(sp_approx)
    )


def lambertw_exp(
        x: jax.typing.ArrayLike,
        a: float = 0.11775704, b: float = 1.1418955,
        low: float = -2.8378477
) -> jax.Array:
    """Exponential of log_lambertw_exp(...) """
    return jnp.exp(log_lambertw_exp(x, a=a, b=b, low=low))
