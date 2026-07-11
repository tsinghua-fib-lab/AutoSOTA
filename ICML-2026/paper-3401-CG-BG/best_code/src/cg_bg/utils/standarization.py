import jax.numpy as jnp


def standardize(x: jnp.ndarray) -> jnp.ndarray:
    com = x.mean(axis=1, keepdims=True)
    x = x - com
    std = x.std()
    x = x / std
    return x, std


def destandardize(x: jnp.ndarray, std: float) -> jnp.ndarray:
    return x * std
