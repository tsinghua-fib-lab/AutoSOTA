import jax
import jax.scipy as jsc
from jax import random, value_and_grad, jit, vmap
import jax.numpy as jnp

# Log score


class PACMVIBasic:
    def __init__(self, model, posterior, y=None, s=1, *args, **kwargs):
        self.model = model
        self.posterior = posterior
        self.y = y
        self.s = s
        self.vg_o = jit(value_and_grad(self.objective, argnums=1))

    def objective(self, key, params):
        theta_sample = self.posterior.sample(key, params, self.s)
        def ll_kl(sample):
            return self.model.log_likelihoods(sample, self.y)
        lls = vmap(ll_kl)(theta_sample)
        return jnp.sum(jsc.special.logsumexp(lls, axis=0) - jnp.log(self.s))


    def value_and_grad(self, key, params):
        return self.vg_o(key, params)

    def name(self):
        return 'PVI'