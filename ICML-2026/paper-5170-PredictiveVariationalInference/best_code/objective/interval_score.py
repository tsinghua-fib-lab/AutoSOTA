import jax
import jax.scipy as jsc
from jax import random, value_and_grad, jit, vmap
import jax.numpy as jnp
import numpy as np

# Quadratic score for continuous outcome

class IntervalScore:
    def __init__(self, model, posterior, y=None, s=1, alpha=0.1, *args, **kwargs):
        self.model = model
        self.posterior = posterior
        self.y = y
        self.s = s
        self.vg_o = jit(value_and_grad(self.objective, argnums=1))
        self.alpha = alpha
        if y is None:
            self.y = model.data()

    def objective(self, key, params):
        key1, key2 = random.split(key)
        keys1 = random.split(key1, self.s )
        keys2 = random.split(key2, self.s )

        def get_samples(key1, key2):
            theta = self.posterior.sample(key1, params, )
            y = self.model.sample_datapoint(key2, theta[0])
            return y

        ys = vmap(get_samples)(keys1, keys2)
        if ys.shape[1] == 1:
            ys = ys[:, 0, ...]
        y0 = jnp.swapaxes(ys, 0, 1)
        y = self.y
        l = jnp.quantile(y0, self.alpha/2, axis=1)
        u = jnp.quantile(y0, 1-self.alpha/2, axis=1)
        #if len(y.shape) == 1:
        #    y = jnp.expand_dims(self.y, 1)
        #print(l.shape, u.shape, y.shape, np.sum((y<l)|(y>u)))
        return -jnp.sum((u-l) + 2/self.alpha * (l-y) * (y<l) + 2/self.alpha * (y-u) * (y>u))


    def value_and_grad(self, key, params):
        return self.vg_o(key, params)

    def name(self):
        return 'PVI'