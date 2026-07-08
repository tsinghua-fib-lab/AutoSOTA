from typing import Callable, Sequence, Any, Literal

import flax.linen as nn

import jax
import jax.numpy as jnp


# Tags to name modules as to identify these dynamically post-init.
FROZEN_PREFIX: str = 'frozen'
FINETUNABLE_PREFIX: str = 'finetunable'


def binary_encoding(int_value: jax.Array) -> jax.Array:
    as_int = jnp.atleast_1d(int_value.astype(jnp.int32))
    bit_code = jnp.unpackbits(as_int.view('uint8'), bitorder='little')
    bit_code = bit_code.ravel().astype(jnp.float32)
    return bit_code


def signed_transform(
        x: jax.Array, inverse: bool,
        *, option: Literal['sqrt', 'log'], **kwargs
) -> jax.Array:

    if option == 'sqrt':
        # See Appendix A, Pohlen et al., 2018; https://arxiv.org/abs/1805.11593
        epsilon = kwargs.get('epsilon', 1e-2)

        forward = jnp.sign(x) * (jnp.sqrt(jnp.abs(x)+1) - 1) + epsilon * x

        num = jnp.sqrt(1 + 4 * epsilon*(jnp.abs(x) + 1 + epsilon)) - 1
        reverse = jnp.sign(x) * (jnp.square(num / (2 * epsilon)) - 1)

        return jax.lax.select(inverse, reverse, forward)
    elif option == 'log':
        forward = jnp.sign(x) * jnp.log1p(jnp.abs(x))
        reverse = jnp.sign(x) * jnp.expm1(jnp.abs(x))
        return jax.lax.select(inverse, reverse, forward)
    else:
        raise NotImplementedError(
            f'`signed_transform`: option {option} not supported. '
            f'Choose either `sqrt` or `log`.'
        )


class ProjectedMVN(nn.Module):
    """Multivariate Normal distribution parametrized by a given module"""
    projection: nn.Module
    bounds: tuple[jax.typing.ArrayLike, jax.typing.ArrayLike]

    scale_init: jax.typing.ArrayLike = 1.0  # Can also be an Array.

    @nn.compact
    def __call__(self, *args) -> tuple[jax.Array, jax.Array]:

        out = self.projection(*args)

        mean, stddev = jnp.split(out, 2)
        stddev = jnp.clip(
            jax.nn.softplus(self.scale_init + stddev),
            a_min=1e-4, a_max=5.0
        )

        return mean, stddev


class ProjectedCategorical(nn.Module):
    """Categorical distribution parametrized by a given module"""
    projection: nn.Module

    temperature: jax.typing.ArrayLike = 1.0  # Can also be an Array.

    @nn.compact
    def __call__(self, *args, softmax: bool = False) -> jax.Array:

        out = self.projection(*args)
        logits = jax.nn.log_softmax(out / self.temperature)

        return jax.lax.select(softmax, jnp.exp(logits), logits)


class Vectorize(nn.Module):
    """Helper module to flatten all inputs"""

    @nn.compact
    def __call__(self, *args) -> jax.Array:
        # Make args a list of 1D arrays
        flat = jax.tree_util.tree_map(jnp.ravel, jax.tree.leaves(args))

        return jnp.concatenate(flat, -1)


class ConvNet(nn.Module):
    """Helper module for a CNN with optional LayerNorm.

    # Example use-case:
    model = ConvNet(
        conv_kwargs=[
            {'features': 128, 'kernel_size': (3, 3), 'strides': 2},
            {'features': 64, 'kernel_size': (3, 3), 'strides': 1},
        ],
        pool_kwargs=[None, None],
        activation=jax.nn.relu
    )
    """
    conv_kwargs: Sequence[dict[str, Any]]
    activation: Callable[[jax.Array], jax.Array] = nn.leaky_relu

    use_layernorm: bool = False

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:

        layers = []
        for i in range(len(self.conv_kwargs)):
            layers.append(nn.Conv(**self.conv_kwargs[i]))

            if self.use_layernorm:
                layers.append(nn.LayerNorm())

            layers.append(self.activation)

        out = nn.Sequential(layers)(x)

        return out


class MLP(nn.Module):
    """Helper module for a MultiLayer-Perceptron with optional LayerNorm.

    # Example use-case for predicting binary output:
    model = MLP([32, 16, 1], activate_final=True, activation=jax.nn.sigmoid)

    # or
    model = MLP([32, 16, 1],
        activate_final=False, activation=jax.nn.relu, use_layernorm=True
    )
    """

    layer_features: Sequence[int]
    dense_kwargs: dict[str, Any] | None = None

    activation: Callable[[jax.Array], jax.Array] = nn.leaky_relu
    activate_final: bool = False

    use_layernorm: bool = False

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:

        if self.use_layernorm:

            layers = [
                layer for size in self.layer_features for layer in (
                    nn.Dense(size, **(self.dense_kwargs or {})),
                    nn.LayerNorm(), self.activation
                )
            ]

            if not self.activate_final:
                layers.pop()
                layers.pop()

        else:
            layers = [
                layer for size in self.layer_features for layer in (
                    nn.Dense(size, **(self.dense_kwargs or {})),
                    self.activation
                )
            ]

            if not self.activate_final:
                layers.pop()

        out = nn.Sequential(layers)(x)

        return out


class ResBlock(nn.Module):
    """Helper to create a residual layer from another module.

    Transformation is computed as:
        Resblock(x): out = activation(module(x) - x)

    Optionally also applies LayerNorm to `out`.
    """
    module: nn.Module
    activation: Callable[[jax.Array], jax.Array] = nn.leaky_relu
    norm: bool = True

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        output = self.module(x)
        out = self.activation(output - x)

        if self.norm:
            out = nn.LayerNorm()(out)

        return out


class CombineInputs(nn.Module):
    modules: Sequence[nn.Module]
    vec: Vectorize = Vectorize()

    @nn.compact
    def __call__(self, *args: Sequence[jax.Array]) -> jax.Array:
        assert len(args) == len(self.modules), \
            "Inputs do not match module-specifications"

        results = [self.vec(m(x)) for x, m in zip(args, self.modules)]
        return jnp.concatenate(results, -1)
