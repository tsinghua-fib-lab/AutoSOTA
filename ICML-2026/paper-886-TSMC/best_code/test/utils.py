import jax
import jax.numpy as jnp


def to_pmf(pi):
    """Utility to convert a discrete evaluation of PDF into a PMF."""

    def pi_as_pmf(*args, **kwargs):
        likelihoods = pi(*args, **kwargs)
        return likelihoods / likelihoods.sum()

    return pi_as_pmf


def min_max(f, shift: float = 0):

    def normed(x):
        ys = f(x)
        same = jnp.all(ys == ys[0])
        norm = (ys - ys.min()) / (ys.max() - ys.min() + 1e-8)

        return jax.lax.select(
            same,
            jnp.zeros_like(norm),
            norm
        ) + shift

    return normed
