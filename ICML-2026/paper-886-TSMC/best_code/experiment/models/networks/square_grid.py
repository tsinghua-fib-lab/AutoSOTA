from typing import Sequence
import flax.linen as nn

import jax
import jax.numpy as jnp

from jaxtyping import PRNGKeyArray

from .. import blocks


def make_model(
        conv_kwargs: Sequence[dict] | None,
        out_kwargs: dict,
        last_out_size: int
) -> tuple[blocks.ConvNet | None, blocks.MLP]:
    embedder = None
    if conv_kwargs is not None:
        embedder = blocks.ConvNet(
            conv_kwargs=conv_kwargs, name=blocks.FROZEN_PREFIX + '_CNN'
        )

    activation = out_kwargs.get('activation', 'leaky_relu')
    act_fun = getattr(jax.nn, activation)

    mlp = blocks.MLP(
        layer_features=out_kwargs['layer_features'] + [last_out_size],
        dense_kwargs=out_kwargs.get('dense_kwargs', None),
        activation=act_fun,
        activate_final=False,
        use_layernorm=out_kwargs.get('use_layernorm', False),
        name=blocks.FINETUNABLE_PREFIX + '_MLP'
    )

    return embedder, mlp


class PolicyNetwork[Action](nn.Module):
    output_size: int
    use_cnn: bool = False
    continuous: bool = False

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

    def enumerate_atoms(self) -> jax.Array | None:
        return jnp.arange(self.output_size)

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:

        embedder, model = make_model(
            conv_kwargs=(None if not self.use_cnn else [
                {'features': 8, 'kernel_size': (3, 3), 'strides': 2},
                {'features': 16, 'kernel_size': (3, 3), 'strides': 2},
            ]),
            out_kwargs={
                'layer_features': [128, 128],
                'use_layernorm': True,
                'activation': 'leaky_relu'
            },
            last_out_size=self.output_size
        )

        embed = x if embedder is None else embedder(x)

        # return normalized logits
        return blocks.ProjectedCategorical(model)(embed.ravel(), softmax=False)


class ValueNetwork(nn.Module):
    use_cnn: bool = False
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
            return jax.random.normal(key, shape) * self.scale + dist_params
        return dist_params

    def logprob(
            self,
            dist_params: jax.Array,  # mean
            value: jax.Array
    ) -> jax.Array:
        return jax.scipy.stats.norm.logpdf(
            value, dist_params.squeeze(), self.scale
        )

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        embedder, model = make_model(
            conv_kwargs=(None if not self.use_cnn else [
                {'features': 8, 'kernel_size': (3, 3), 'strides': 2},
                {'features': 16, 'kernel_size': (3, 3), 'strides': 2},
            ]),
            out_kwargs={
                'layer_features': [128, 128],
                'use_layernorm': True,
                'activation': 'leaky_relu'
            },
            last_out_size=1
        )

        embed = x if embedder is None else embedder(x)
        return model(embed.ravel())


class QValueNetwork(nn.Module):
    output_size: int
    use_cnn: bool = False
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
            return jax.random.normal(key, shape) * self.scale + dist_params
        return dist_params

    def logprob(
            self,
            dist_params: jax.Array,  # mean
            value: jax.Array
    ) -> jax.Array:
        return jax.scipy.stats.norm.logpdf(
            value, dist_params.squeeze(), self.scale
        )

    @nn.compact
    def __call__(self, obs: jax.Array, action: jax.Array) -> jax.Array:
        embedder, model = make_model(
            conv_kwargs=(None if not self.use_cnn else [
                {'features': 8, 'kernel_size': (3, 3), 'strides': 2},
                {'features': 16, 'kernel_size': (3, 3), 'strides': 2},
            ]),
            out_kwargs={
                'layer_features': [128, 128],
                'use_layernorm': True,
                'activation': 'leaky_relu'
            },
            last_out_size=self.output_size
        )

        embed = obs if embedder is None else embedder(obs)

        qs = model(embed.ravel())

        return qs.at[action].get()
