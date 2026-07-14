import jax
import jax.scipy as jsc
from jax import random, grad, value_and_grad, jit, vmap
import jax.numpy as jnp
from objective import VIBasic
from functools import partial

# CRPS for 1D continuous outcomes


class PVICRPS:
    def __init__(self, model, posterior, y=None,  s = 10, *args, **kwargs):
        self.model = model
        self.posterior = posterior
        self.y = y
        self.s = s
        self.vg_o = jit(self.all)
        if y is None:
            self.y = model.data()
        #self.M = self.model.M

    def objective(self, key, params):
        key1, key2 = random.split(key)
        keys1 = random.split(key1, self.s * 2)
        keys2 = random.split(key2, self.s * 2)

        def get_samples(key1, key2):
            theta = self.posterior.sample(key1, params, )
            y = self.model.sample_datapoint(key2, theta[0])
            return y

        ys = vmap(get_samples)(keys1, keys2)
        if ys.shape[1] == 1:
            ys = ys[:, 0, ...]
        y1 = jnp.swapaxes(ys[:self.s], 0, 1)
        y2 = jnp.swapaxes(ys[self.s:], 0, 1)
        y = self.y  # jnp.expand_dims(self.y, 1)
        if len(y.shape) == 1:
            y = jnp.expand_dims(self.y, 1)
        return jnp.sum(
            -jnp.mean(jnp.abs(y - y1), axis=1) / 2 - jnp.mean(jnp.abs(y - y2), axis=1) / 2 + jnp.mean(jnp.abs(y1 - y2),
                                                                                                      axis=1) / 2)

    def all(self, key, params):
        pred_obj, pred_grad = value_and_grad(self.objective, argnums=1)(key, params)
        return pred_obj, pred_grad

    def value_and_grad(self, key, params):
        return self.vg_o(key, params)