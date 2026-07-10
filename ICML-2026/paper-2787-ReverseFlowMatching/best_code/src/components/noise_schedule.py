import functools
import jax
import jax.numpy as jnp


"""Geometric noise schedule functions."""


@functools.partial(jax.jit, static_argnames=("sigma_min", "sigma_diff"))
def noise_g(t, sigma_min, sigma_diff):
    return sigma_min * (sigma_diff**t) * ((2 * jnp.log(sigma_diff)) ** 0.5)


@functools.partial(jax.jit, static_argnames=("sigma_min", "sigma_diff"))
def noise_h(t, sigma_min, sigma_diff):
    return (sigma_min * (((sigma_diff ** (2 * t)) - 1) ** 0.5)) ** 2
