from typing import Sequence, Literal

import flax.linen as nn
from dataclasses import field

import jax
import jax.numpy as jnp

from jaxtyping import PRNGKeyArray

from .. import blocks


class PolicyNetwork(nn.Module):
    """Parametrizes a diagonal multivariate Gaussian policy.

    See Also
    --------
    https://en.wikipedia.org/wiki/Normal_distribution
    """
    output_size: int
    mlp_kwargs: dict
    option: Literal['clip', 'tanh', 'trunc']  # TODO: truncnorm
    bounds: tuple[jax.typing.ArrayLike, jax.typing.ArrayLike]
    bound_scale_slack: float = 4.0

    continuous: bool = True

    @property
    def output_sizes(self):
        return (self.output_size, )

    def sample(
            self,
            key: PRNGKeyArray,
            shape: Sequence[int],
            dist_params: tuple[jax.Array, jax.Array]
    ) -> jax.Array:
        mean, scale = dist_params

        samples = jax.random.normal(key, (*shape, *mean.shape)) * scale + mean

        if self.option == 'clip':
            samples = jnp.clip(samples, *self.bounds)
        elif self.option == 'tanh':
            samples = jnp.tanh(samples)
        elif self.option == 'trunc':
            pass

        return samples

    def logprob(
            self,
            dist_params: tuple[jax.Array, jax.Array],
            action: jax.Array
    ) -> jax.Array:
        mean, scale = dist_params

        if self.option == 'clip':
            return jax.scipy.stats.norm.logpdf(action, mean, scale).sum()
        elif self.option == 'tanh':

            # Change of variable rule; see Appendix C of
            # https://arxiv.org/abs/1801.01290
            u = jnp.arctanh(jnp.clip(action, -1 + 1e-6, 1 - 1e-6))

            lp = jax.scipy.stats.norm.logpdf(u, mean, scale)

            # Numerically stable log-determinant of tanh jacobian
            logdet = 2 * (jnp.log(2) - u - jax.nn.softplus(-2 * u))
            return (lp - logdet).sum()

        elif self.option == 'trunc':
            pass

    def entropy(self, dist_params: tuple[jax.Array, jax.Array]) -> jax.Array:
        mean, scale = dist_params

        c = jnp.log(2 * jnp.pi * jnp.e) / 2
        mvn_ent = c + jnp.log(scale).sum()

        if self.option == 'clip':
            return mvn_ent
        elif self.option == 'tanh':
            # Stratified/ Quantile based approximation of entropy
            qs = jnp.linspace(0.01, 0.99, 100)
            xs = jax.scipy.stats.norm.ppf(qs, 0.0, 1.0)

            xs = xs * scale[:, None] + mean[:, None]

            # Numerically stable version of `ln(1 - tanh(x)^2)`
            correction = jnp.mean(
                (2 * (jnp.log(2) - xs - jax.nn.softplus(-2 * xs))).sum(axis=0)
            )

            return mvn_ent - correction
        elif self.option == 'trunc':
            pass

    def enumerate_atoms(self) -> jax.Array | None:
        return None

    @nn.compact
    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        flat = blocks.Vectorize()(x)

        model = blocks.MLP(
            list(self.mlp_kwargs['layer_features']) + [2 * self.output_size],
            activation=getattr(
                jax.nn, self.mlp_kwargs.get('activation', 'leaky_relu')
            ),
            use_layernorm=self.mlp_kwargs.get('use_layernorm', True),
            activate_final=False,
            name=blocks.FINETUNABLE_PREFIX
        )

        dist = blocks.ProjectedMVN(model, bounds=self.bounds)

        mu, scale = dist(flat)
        mu = jnp.clip(
            mu,
            a_min=self.bound_scale_slack * self.bounds[0],
            a_max=self.bound_scale_slack * self.bounds[1]
        )

        return mu, scale


class ValueNetwork(nn.Module):
    mlp_kwargs: dict
    scale: float = 1.0
    value_transform: Literal['sqrt', 'log'] | None = None
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
        flat = blocks.Vectorize()(x)

        model = blocks.MLP(
            list(self.mlp_kwargs['layer_features']) + [1],
            activation=getattr(
                jax.nn, self.mlp_kwargs.get('activation', 'leaky_relu')
            ),
            use_layernorm=self.mlp_kwargs.get('use_layernorm', True),
            activate_final=False,
            name=blocks.FINETUNABLE_PREFIX
        )

        return model(flat)


class QValueNetwork(nn.Module):
    mlp_kwargs: dict
    scale: float = 1.0
    value_transform: Literal['sqrt', 'log'] | None = None
    distributional: bool = False

    @property
    def output_sizes(self):
        return (self.output_size, )

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
    def __call__(self, obs: jax.Array, action: jax.Array) -> jax.Array:
        flat_obs = blocks.Vectorize()(obs)
        flat_act = blocks.Vectorize()(action)

        model_obs = blocks.MLP(
            [128],
            activation=jax.nn.leaky_relu,
            use_layernorm=True,
            activate_final=True
        )
        model_action = blocks.MLP(
            [128],
            activation=jax.nn.leaky_relu,
            use_layernorm=True,
            activate_final=True
        )

        embed_obs = model_obs(flat_obs)
        embed_act = model_action(flat_act)

        combined = jnp.concatenate([embed_obs, embed_act], axis=-1)

        model = blocks.MLP(
            list(self.mlp_kwargs['layer_features']) + [1],
            activation=getattr(
                jax.nn, self.mlp_kwargs.get('activation', 'leaky_relu')
            ),
            use_layernorm=self.mlp_kwargs.get('use_layernorm', True),
            activate_final=False,
            name=blocks.FINETUNABLE_PREFIX
        )

        return model(combined)
