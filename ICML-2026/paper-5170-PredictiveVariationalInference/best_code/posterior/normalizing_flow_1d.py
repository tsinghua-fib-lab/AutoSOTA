import jax
import jax.scipy as jsc
import pandas as pd
from jax import random, grad, vmap
import jax.numpy as jnp
import generax
import equinox as eqx

class NormalizingFlow1D:
    def __init__(self, dim, key = None, knots = 16, ):
        self.dim = dim
        k1, k2 = random.split(key)
        transform = generax.RationalQuadraticSpline(input_shape=(1,),
                                                    K=knots,
                                                    key=k1)
        prior = generax.Gaussian(input_shape=(1,))
        self.flow = generax.NormalizingFlow(transform=transform, prior=prior)
        initials = random.normal(k2, (1000, dim))
        self.flow = self.flow.data_dependent_init(initials, key=key)
        self.gradf = eqx.filter_jit(eqx.filter_grad(self._log_posterior))

    def extract_params(self, params):
        raise NotImplementedError()

    def log_posterior(self, theta, params):
        return params.log_prob(theta)

    def _log_posterior(self, params, theta):
        return self.log_posterior(theta, params)

    def sample(self, key, params, number = 1):
        keys = random.split(key, number)
        samples, _ = eqx.filter_vmap(params.sample_and_log_prob)(keys,)
        return samples

    def posterior_parameters(self, params):
        raise NotImplementedError()

    def gen_params(self):
        return self.flow

    def get_grad(self, theta, params):
        return self.gradf(params, theta)

