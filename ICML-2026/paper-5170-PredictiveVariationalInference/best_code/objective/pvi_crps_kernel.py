import jax.numpy as jnp
from functools import partial
from jax import vmap, random
import numpy as np

# CRPS for nD continuous outcomes (with a kernel)

class PVICRPSKernel:
    def __init__(self, model, posterior, y,  s = 10, *args, **kwargs):
        self.model = model
        self.posterior = posterior
        self.y = self.model.data()
        self.s = s

    def objective(self, key, params, data):
        key1, key2 = random.split(key)
        keys1 = random.split(key1, self.s * 2)
        keys2 = random.split(key2, self.s * 2)

        def get_samples(key1, key2):
            theta = self.posterior.sample(key1, params, )
            y = self.model.sample_datapoint(theta, key2)
            return y
        ys = vmap(get_samples)(keys1, keys2)
        if ys.shape[1] == 1:
            ys = ys[:, 0, ...]
        y1 = jnp.expand_dims(ys[:self.s], 0)
        y2 = jnp.expand_dims(ys[self.s:], 0,)
        #print(y)

        if len(data.shape) == 3:
            data = jnp.expand_dims(data, 1)
        return jnp.mean(
            -jnp.mean(jnp.linalg.norm(data - y1, axis=(2, 3)), axis=1) / 2 - jnp.mean(jnp.linalg.norm(data - y2, axis=(2, 3)), axis=1) / 2 + jnp.mean(jnp.linalg.norm(y1 - y2, axis = (2, 3)),
                                                                                                      axis=1) / 2)