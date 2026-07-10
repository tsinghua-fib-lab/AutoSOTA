import jax
import jax.numpy as jnp

import flax
import flax.linen as nn

from functools import partial
from typing import Any, Callable, Optional, Sequence, Tuple, Union, Collection

def get_timestep_embedding(timestep: float, embedding_dim: int):
    """
    Compute sinusoidal embeddings for a scalar timestep.
    """
    half_dim = embedding_dim // 2
    k = 10000
    emb = jnp.log(k) / (half_dim - 1)
    emb = jnp.exp(jnp.arange(half_dim) * -emb)
    emb = timestep * emb
    emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)])
    return emb  # shape (embedding_dim,)

class MLP(nn.Module):
    dim_hidden: Sequence[int] = (128, 128, 128)
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    out_dim: int = 1

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        z = x
        for n_hidden in self.dim_hidden:
            z = nn.Dense(n_hidden, use_bias=True)(z)
            z = self.activation(z)
        z = nn.Dense(self.out_dim, use_bias=True)(z)
        return z

class ScoreMLP(nn.Module):
    dim_hidden: Sequence[int] = (128, 128, 128)
    emb_dim_hidden: Sequence[int] = (64, 64)
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    out_dim: int = 1

    @nn.compact
    def __call__(self, x: jnp.ndarray, t: float) -> jnp.ndarray:
        # x: (d,), t: scalar

        x_emb = MLP(
            dim_hidden=self.emb_dim_hidden,
            activation=self.activation,
            out_dim=-(self.dim_hidden[0] // -2)
        )(x)  # shape: (hidden_dim//2,)

        t_emb = get_timestep_embedding(t, embedding_dim=64)
        t_emb = MLP(
            dim_hidden=self.emb_dim_hidden,
            activation=self.activation,
            out_dim=self.dim_hidden[0] // 2
        )(t_emb)  # shape: (hidden_dim//2,)

        vec = jnp.concatenate([x_emb, t_emb], axis=-1)  # shape: (dim_hidden[0],)

        out = MLP(
            dim_hidden=self.dim_hidden,
            activation=self.activation,
            out_dim=self.out_dim
        )(vec)  # shape: (out_dim,)

        return out


class PotentialMLP(nn.Module):
    dim_hidden: Sequence[int] = (128, 128, 128)
    emb_dim_hidden: Sequence[int] = (64, 64)
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    # φ should be scalar
    out_dim: int = 1

    @nn.compact
    def __call__(self, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        # x: (..., d) or (d,), t: scalar or (...,)
        x_emb = MLP(
            dim_hidden=self.emb_dim_hidden,
            activation=self.activation,
            out_dim=-(self.dim_hidden[0] // -2)
        )(x)  # (..., hidden//2)

        t_emb = get_timestep_embedding(t, embedding_dim=64)
        t_emb = MLP(
            dim_hidden=self.emb_dim_hidden,
            activation=self.activation,
            out_dim=self.dim_hidden[0] // 2
        )(t_emb)  # (..., hidden//2)

        vec = jnp.concatenate([x_emb, t_emb], axis=-1)  # (..., dim_hidden[0])

        # scalar potential
        phi = MLP(
            dim_hidden=self.dim_hidden,
            activation=self.activation,
            out_dim=self.out_dim
        )(vec)  # (..., 1)

        # return scalar
        return jnp.squeeze(phi, axis=-1)  # (...,)


class ConservativeMLP(nn.Module):
    """Conservative vector field: returns grad_x φ(x, t) where φ is produced by PotentialMLP."""
    dim_hidden: Sequence[int] = (128, 128, 128)
    emb_dim_hidden: Sequence[int] = (64, 64)
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, x: jnp.ndarray, t: float) -> jnp.ndarray:
        # x: (d,), t: scalar

        # instantiate the underlying potential network once
        pot_net = PotentialMLP(
            dim_hidden=self.dim_hidden,
            emb_dim_hidden=self.emb_dim_hidden,
            activation=self.activation,
            out_dim=1,
        )

        # single-sample φ(x, t) (must return a scalar)
        def phi_single(x1, t1):
            # ensure scalar t for single-sample path
            t1 = jnp.asarray(t1)
            return pot_net(x1, t1)  # scalar

        # grad wrt x for a single sample
        grad_phi_single = jax.grad(phi_single, argnums=0)

        return grad_phi_single(x, t)
    