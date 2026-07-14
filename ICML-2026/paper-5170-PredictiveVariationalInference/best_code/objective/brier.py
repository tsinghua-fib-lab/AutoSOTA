import jax
import jax.scipy as jsc
from jax import random, value_and_grad, jit, vmap
import jax.numpy as jnp

# Quadratic score for binary outcome

class QuadraticBernoulli:
    def __init__(self, model, posterior, y=None, s=1, *args, **kwargs):
        self.model = model
        self.posterior = posterior
        self.y = y
        self.s = s
        self.vg_o = jit(value_and_grad(self.objective, argnums=1))

    def objective(self, key, params):
        theta_sample = self.posterior.sample(key, params, self.s)
        def lgs(sample):
            return self.model.logits(sample)
        logits = vmap(lgs)(theta_sample)
        fs = jsc.special.expit(logits)
        ave_fs = jnp.mean(fs, axis=0)
        y = self.model.data()
        return jnp.mean(2 * (ave_fs * y - jnp.square(ave_fs) + (1-ave_fs) * (1-y)) - jnp.square(1-ave_fs))


    def value_and_grad(self, key, params):
        return self.vg_o(key, params)

    def name(self):
        return 'PVI'