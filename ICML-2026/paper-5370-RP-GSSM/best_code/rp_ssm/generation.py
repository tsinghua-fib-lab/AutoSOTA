from typing import Sequence

import jax.numpy as np
import jax.random as jr
from jax import vmap, Array

import flax.linen as nn

import optax

from rp_ssm.dists import GaussianDistParam
from rp_ssm.datasets import TrainData


class GenerativeNetwork(nn.Module):
    """
    Class containing nonlinear-Gaussian
    generative model.

    Currently only supports diagonal covariance.
    """
    network: nn.Module
    data_size: int # size of a single datapoint
    constant_cov: bool
    sigmoid: bool = False # whether to pass network output through sigmoid to map it to (0,1)

    @nn.compact
    def __call__(self, x: Array) -> GaussianDistParam:
        x = self.network(x)
        if self.constant_cov:
            x = nn.Dense(self.data_size)(x)
            
            cov_flat = self.variable(
                'params', 'cov', np.zeros, (self.data_size,)
            )
            cov = cov_flat.value
            
        else:
            x = nn.Dense(2 * self.data_size)(x)
            
            cov = x[self.data_size:]

        if self.sigmoid:
            mean = nn.sigmoid(x[:self.data_size])
        else:
            mean = x[:self.data_size]
            
        return GaussianDistParam(
            mean=mean,
            cov=nn.softplus(cov)
        )


class DCNN(nn.Module):
    """
    Simple deconvolutional NN with
    no final Dense layer.
    """
    dcnn_features : Sequence[dict]
    base_features: int = 128
    base_size: int = 6

    @nn.compact
    def __call__(self, z: Array) -> Array:
        x = nn.Dense(self.base_features * self.base_size ** 2)(z)
        x = x.reshape((self.base_size, self.base_size, self.base_features))
        x = nn.relu(x)
        for feat in self.dcnn_features:
            x = nn.ConvTranspose(**feat)(x)
            x = nn.relu(x)
        return x.flatten()
        

class GenerativeModel:
    def __init__(self, generative_network: GenerativeNetwork):
        self.generative_network = generative_network

    def init(self, key, latent_dim, config):
        params = self.generative_network.init(key, np.zeros(latent_dim))
        opt = optax.adam(config.lr)
        opt_state = opt.init(params)
        return params, opt_state, opt

    def apply(self, params, data):
        return self.generative_network.apply(params, data)

    def loss(self, params, data: TrainData, key, trainer, num_samples):
        # data is of shape BxTxD

        _, posterior = trainer.apply(data)
        posterior = posterior.mean_field # BxTxK

        def _single_loss(k, x, q):
            """
            Compute loss over single datapoint.
            `x` is of shape D.
            """
            zs = q.sample(k, num_samples)
            log_probs = vmap(
                lambda z: self.generative_network.apply(params, z).log_prob(x.flatten()) # very important to flatten x!
            )(zs)
            return np.mean(log_probs)
        
        def _sequence_loss(k, x, q):
            """
            Compute loss over sequence.
            `x` is of shape TxD.
            """
            keys = jr.split(k, x.shape[0])
            return vmap(_single_loss)(keys, x, q)
        
        keys = jr.split(key, data.obs[0].shape[0])
        loss = vmap(_sequence_loss)(keys, data.obs[0], posterior)

        Z = data.obs[0].shape[0] * data.obs[0].shape[1]
        return -np.sum(loss) / Z, None