import jax
import jax.scipy as jsc
from jax import random, grad, vmap
import jax.numpy as jnp
import generax
import equinox as eqx

class NormalizingFlow:
    def __init__(self, dim, key = None, n_flow_layers = 3, *args, **kwargs):
        self.dim = dim
        k1, k2 = random.split(key)
        self.flow = generax.RealNVP(
            input_shape=(dim,),
            n_flow_layers=n_flow_layers,
            working_size=16,
            hidden_size=32,
            n_blocks=4,
            #n_spline_knots=8,
            key=k1
        )
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




