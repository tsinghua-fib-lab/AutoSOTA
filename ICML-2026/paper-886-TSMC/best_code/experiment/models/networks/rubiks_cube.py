from typing import Sequence, Literal
from functools import partial

import jax
import jax.numpy as jnp

import flax.linen as nn

import numpy as np

from jaxtyping import PRNGKeyArray

from .. import blocks


class PreProcessRubiksCube:

    def preprocess(
            self, obs: tuple[jax.Array]
    ) -> tuple[jax.Array, tuple[jax.Array]]:
        cube, step_count = obs

        # Binary encoding of unbounded int32
        bit_code = blocks.binary_encoding(step_count)

        # One-hot-encoding of cube with 6 faces
        # cube_code = jax.nn.one_hot(cube.reshape(6, -1), 6)

        return cube.ravel().astype(jnp.float32), (bit_code,)


class PolicyNetwork(PreProcessRubiksCube, nn.Module):
    mlp_kwargs: dict
    # (3, 3) RubiksCube-v0 defaults
    output_size: int = 18
    output_sizes: tuple[int, int, int] = (6, 1, 3)
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

    def entropy(self, dist_params: jax.Array) -> jax.Array:
        return -jnp.sum(jnp.exp(dist_params) * dist_params)

    def enumerate_atoms(self) -> jax.Array | None:
        acts = jnp.arange(self.output_size)
        atoms = jax.vmap(jnp.unravel_index, in_axes=(0, None))(
            acts, self.output_sizes
        )
        return jnp.asarray(atoms).T

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:

        embedding, metadata = self.preprocess(x)

        # Concatenate with vector data
        output = jnp.concatenate([embedding, *metadata], axis=-1)

        out_model = blocks.MLP(
            list(self.mlp_kwargs['layer_features']) + [self.output_size],
            activation=getattr(
                jax.nn, self.mlp_kwargs.get('activation', 'leaky_relu')
            ),
            use_layernorm=self.mlp_kwargs.get('use_layernorm', True),
            activate_final=False,
            name=blocks.FINETUNABLE_PREFIX
        )

        return blocks.ProjectedCategorical(out_model)(output, softmax=False)


class ValueNetwork(PreProcessRubiksCube, nn.Module):
    mlp_kwargs: dict
    value_transform: Literal['sqrt', 'log'] | None = None
    scale: float = 1.0
    distributional: bool = False

    def sample(
            self,
            key: PRNGKeyArray,
            shape: Sequence[int],
            dist_params: jax.Array,  # mean
    ) -> jax.Array:
        dist_params = dist_params.squeeze()
        if self.distributional:
            # TODO; sampling from inverse of `value_transform`
            return jax.random.normal(key, shape) * self.scale + dist_params
        return dist_params

    def logprob(
            self,
            dist_params: jax.Array,  # mean
            value: jax.Array
    ) -> jax.Array:
        if self.value_transform is None:
            y = value
            correction = 0
        else:
            y, dgdy = jax.value_and_grad(
                partial(blocks.signed_transform, option=self.value_transform)
            )(value, True)
            correction = jnp.log(jnp.abs(dgdy) + 1e-8)

        log_fx = jax.scipy.stats.norm.logpdf(
            y, dist_params.squeeze(), self.scale
        )

        return log_fx + correction

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:

        embedding, metadata = self.preprocess(x)

        # Concatenate with vector data
        output = jnp.concatenate([embedding, *metadata], axis=-1)

        out_model = blocks.MLP(
            list(self.mlp_kwargs['layer_features']) + [1],
            activation=getattr(
                jax.nn, self.mlp_kwargs.get('activation', 'leaky_relu')
            ),
            use_layernorm=self.mlp_kwargs.get('use_layernorm', True),
            activate_final=False,
            name=blocks.FINETUNABLE_PREFIX
        )

        value = out_model(output)

        if self.value_transform is None:
            return value

        return blocks.signed_transform(
            value, False, option=self.value_transform
        )


class QValueNetwork(PreProcessRubiksCube, nn.Module):
    mlp_kwargs: dict
    # (3, 3) RubiksCube-v0 defaults
    output_size: int = 18
    output_sizes: tuple[int, int, int] = (6, 1, 3)
    value_transform: Literal['sqrt', 'log'] | None = None
    scale: float = 1.0
    distributional: bool = False

    def sample(
            self,
            key: PRNGKeyArray,
            shape: Sequence[int],
            dist_params: jax.Array,  # Mean
    ) -> jax.Array:
        dist_params = dist_params.squeeze()
        if self.distributional:
            # TODO; sampling from inverse of `value_transform`
            return jax.random.normal(key, shape) * self.scale + dist_params
        return dist_params

    def logprob(
            self,
            dist_params: jax.Array,  # mean
            value: jax.Array
    ) -> jax.Array:
        if self.value_transform is None:
            y = value
            correction = 0
        else:
            y, dgdy = jax.value_and_grad(
                partial(blocks.signed_transform, option=self.value_transform)
            )(value, True)
            correction = jnp.log(jnp.abs(dgdy) + 1e-8)

        log_fx = jax.scipy.stats.norm.logpdf(
            y, dist_params.squeeze(), self.scale
        )

        return log_fx + correction

    @nn.compact
    def __call__(self, obs: jax.Array, action: jax.Array) -> jax.Array:
        action = jnp.ravel_multi_index(
            action, self.output_sizes, mode='clip'  # type: ignore
        )

        embedding, metadata = self.preprocess(obs)

        # Concatenate with vector data
        output = jnp.concatenate([embedding, *metadata], axis=-1)

        out_model = blocks.MLP(
            list(self.mlp_kwargs['layer_features']) + [self.output_size],
            activation=getattr(
                jax.nn, self.mlp_kwargs.get('activation', 'leaky_relu')
            ),
            use_layernorm=self.mlp_kwargs.get('use_layernorm', True),
            activate_final=False,
            name=blocks.FINETUNABLE_PREFIX
        )

        qs = out_model(output)
        q_val = qs.at[action].get()

        if self.value_transform is None:
            return q_val

        return blocks.signed_transform(
            q_val, False, option=self.value_transform
        )
