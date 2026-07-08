from typing import Sequence, Literal
from functools import partial

import flax.linen as nn

import jax
import jax.numpy as jnp

from jaxtyping import PRNGKeyArray

from .. import blocks


def make_convnet(
        conv_kwargs: dict,
        out_kwargs: dict,
        last_out_size: int
) -> tuple[blocks.ConvNet, blocks.MLP]:
    cnn = blocks.ConvNet(
        conv_kwargs=conv_kwargs, name=blocks.FROZEN_PREFIX + '_CNN'
    )

    activation = out_kwargs.get('activation', 'leaky_relu')
    act_fun = getattr(jax.nn, activation)

    mlp = blocks.MLP(
        layer_features=list(out_kwargs['layer_features']) + [last_out_size],
        dense_kwargs=out_kwargs.get('dense_kwargs', None),
        activation=act_fun,
        activate_final=False,
        use_layernorm=out_kwargs.get('use_layernorm', False),
        name=blocks.FINETUNABLE_PREFIX + '_MLP'
    )

    return cnn, mlp


class RequirePreProcess[Observation]:

    def preprocess(
            self, obs: Observation
    ) -> tuple[jax.Array, tuple[jax.Array, ...]]:
        ...


class CNNPolicyNetwork[Action](RequirePreProcess, nn.Module):
    conv_kwargs: dict
    out_kwargs: dict
    output_size: int

    def sample(
            self,
            key: PRNGKeyArray,
            shape: Sequence[int],
            dist_params: jax.Array,
    ) -> jax.Array:
        return jax.random.categorical(key, logits=dist_params, shape=shape)

    def logprob(
            self,
            dist_params: jax.Array,
            action: jax.Array
    ) -> jax.Array:
        return dist_params.at[action].get()

    def entropy(self, dist_params: jax.Array) -> jax.Array:
        return -jnp.sum(jnp.exp(dist_params) * dist_params)

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:

        image, metadata = self.preprocess(x)

        # Get shared embedding from image data
        cnn, out_model = make_convnet(
            self.conv_kwargs, self.out_kwargs, self.output_size
        )
        embedding = cnn(image).ravel()

        # Concatenate with vector data
        output = jnp.concatenate([embedding, *metadata], axis=-1)

        # return normalized logits
        return blocks.ProjectedCategorical(out_model)(output, softmax=False)


class CNNValueNetwork(RequirePreProcess, nn.Module):
    conv_kwargs: dict
    out_kwargs: dict
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

        image, metadata = self.preprocess(x)

        # Get shared embedding from RGB data
        cnn, out_model = make_convnet(self.conv_kwargs, self.out_kwargs, 1)
        embedding = cnn(image).ravel()

        # Concatenate with vector data
        output = jnp.concatenate([embedding, *metadata], axis=-1)

        value = out_model(output)

        if self.value_transform is None:
            return value

        return blocks.signed_transform(
            value, False, option=self.value_transform
        )


class CNNQValueNetwork(RequirePreProcess, nn.Module):
    conv_kwargs: dict
    out_kwargs: dict
    output_size: int
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

        image, metadata = self.preprocess(obs)

        # Get shared embedding from RGB data
        cnn, out_model = make_convnet(self.conv_kwargs, self.out_kwargs, self.output_size)
        embedding = cnn(image).ravel()

        # Concatenate with vector data
        output = jnp.concatenate([embedding, *metadata], axis=-1)

        qs = out_model(output)
        q_val = qs.at[action].get()

        if self.value_transform is None:
            return q_val

        return blocks.signed_transform(
            q_val, False, option=self.value_transform
        )
