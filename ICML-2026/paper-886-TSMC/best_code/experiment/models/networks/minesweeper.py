from typing import Sequence

import jax
import jax.numpy as jnp

import flax.linen as nn

from jaxtyping import PRNGKeyArray

from .. import blocks
from ._jumanji_base import CNNPolicyNetwork, CNNValueNetwork, CNNQValueNetwork


class PreProcessMinesweeper:

    def preprocess(
            self, obs: tuple[jax.Array]
    ) -> tuple[jax.Array, tuple[jax.Array]]:

        # Binary encoding of unbounded int32
        bit_code = blocks.binary_encoding(obs.num_mines)

        return jnp.expand_dims(obs.board, -1), (bit_code, )


class PolicyNetwork(PreProcessMinesweeper, CNNPolicyNetwork):
    output_size: int = 100  # Minesweeper-v0 defaults
    output_sizes: tuple[int, int] = (10, 10)

    continuous: bool = False

    def sample(
            self,
            key: PRNGKeyArray,
            shape: Sequence[int],
            dist_params: jax.Array,
    ) -> jax.Array:
        indices = jax.random.categorical(key, logits=dist_params, shape=shape)
        out = jnp.asarray(jnp.unravel_index(indices, self.output_sizes)).T
        return out

    def logprob(
            self,
            dist_params: jax.Array,
            action: jax.Array
    ) -> jax.Array:
        index = jnp.ravel_multi_index(action, self.output_sizes, mode='clip')
        return dist_params.at[index].get()

    def enumerate_atoms(self) -> jax.Array | None:
        acts = jnp.arange(self.output_size)
        atoms = jax.vmap(jnp.unravel_index, in_axes=(0, None))(
            acts, self.output_sizes
        )
        return jnp.asarray(atoms).T


class ValueNetwork(PreProcessMinesweeper, CNNValueNetwork):
    ...


class QValueNetwork(PreProcessMinesweeper, CNNQValueNetwork):
    output_size: int = 100  # Minesweeper-v0 defaults
    output_sizes: tuple[int, int] = (10, 10)

    @nn.compact
    def __call__(self, obs: jax.Array, action: jax.Array) -> jax.Array:
        action = jnp.ravel_multi_index(
            action, self.output_sizes, mode='clip'  # type: ignore
        )
        return super().__call__(obs, action)
