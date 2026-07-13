from typing import Sequence, Callable

import jax
import jax.numpy as np
from jax import Array

import flax.linen as nn

from rp_ssm.distmaps import DistMap
from rp_ssm.dists import NatParam


INITIALIZER = jax.nn.initializers.variance_scaling(
    scale=0.1, mode='fan_in', distribution='truncated_normal'
)


class RPMRecognition(nn.Module):
    network: nn.Module
    dist_map: DistMap
    kernel_init: Callable = INITIALIZER
    bias_init: Callable = jax.nn.initializers.zeros
    constant_cov: bool = (
        False  # if True, recognition covariance is constant across all data
    )

    @nn.compact
    def __call__(self, x: Array) -> NatParam:
        x = self.network(x)
        if self.constant_cov:
            mean_dim = self.dist_map.latent_dim
            cov_dim = self.dist_map.input_dim - mean_dim
            cov_flat = self.variable(
                'params', 'cov', np.zeros, (cov_dim,)
            )
            x = nn.Dense(mean_dim, kernel_init=INITIALIZER)(x)
            x = np.concatenate((x, cov_flat.value))
        else:
            x = nn.Dense(
                self.dist_map.input_dim,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init
            )(x)

        return self.dist_map(x)


class MLP(nn.Module):
    """Simple MLP with no final Dense layer."""
    features : Sequence[int]

    @nn.compact
    def __call__(self, x: Array) -> Array:
        for feat in self.features:
            x = nn.Dense(feat)(x)
            x = nn.relu(x)
        return x


class CNN(nn.Module):
    """Simple CNN with no final Dense layer."""
    cnn_features : Sequence[dict]

    @nn.compact
    def __call__(self, x: Array) -> Array:
        for feat in self.cnn_features:
            x = nn.Conv(**feat)(x)
            x = nn.relu(x)
        return x.flatten()