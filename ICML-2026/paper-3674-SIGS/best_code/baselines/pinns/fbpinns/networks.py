"""
Defines standard neural network models

Each network class must inherit from the Network base class.
Each network class must define the NotImplemented methods.

This module is used by constants.py (and subsequently trainers.py)
"""

import jax.numpy as jnp
from jax import random


class Network:
    """Base neural network class to be inherited by different neural network classes.

    Note all methods in this class are jit compiled / used by JAX,
    so they must not include any side-effects!
    (A side-effect is any effect of a function that doesn’t appear in its output)
    This is why only static methods are defined.
    """

    # required methods

    @staticmethod
    def init_params(*args):
        """Initialise class parameters.
        Returns tuple of dicts ({k: pytree}, {k: pytree}) containing static and trainable parameters"""
        raise NotImplementedError

    @staticmethod
    def network_fn(params, x):
        """Forward model, for a SINGLE point with shape (xd,)"""
        raise NotImplementedError




class FCN(Network):
    "Fully connected network"

    @staticmethod
    def init_params(key, layer_sizes):
        keys = random.split(key, len(layer_sizes)-1)
        params = [FCN._random_layer_params(k, m, n)
                for k, m, n in zip(keys, layer_sizes[:-1], layer_sizes[1:])]
        trainable_params = {"layers": params}
        return {}, trainable_params

    @staticmethod
    def _random_layer_params(key, m, n):
        "Create a random layer parameters"

        w_key, b_key = random.split(key)
        v = jnp.sqrt(1/m)
        w = random.uniform(w_key, (n, m), minval=-v, maxval=v)
        b = random.uniform(b_key, (n,), minval=-v, maxval=v)
        return w,b

    @staticmethod
    def network_fn(params, x):
        params = params["trainable"]["network"]["subdomain"]["layers"]
        for w, b in params[:-1]:
            x = jnp.dot(w, x) + b
            x = jnp.tanh(x)
        w, b = params[-1]
        x = jnp.dot(w, x) + b
        return x
    

class AdaptiveFCN(Network):
    "Fully connected network with adaptive activations"

    @staticmethod
    def init_params(key, layer_sizes):
        keys = random.split(key, len(layer_sizes)-1)
        params = [AdaptiveFCN._random_layer_params(k, m, n)
                for k, m, n in zip(keys, layer_sizes[:-1], layer_sizes[1:])]
        trainable_params = {"layers": params}
        return {}, trainable_params

    @staticmethod
    def _random_layer_params(key, m, n):
        "Create a random layer parameters"

        w_key, b_key = random.split(key)
        v = jnp.sqrt(1/m)
        w = random.uniform(w_key, (n, m), minval=-v, maxval=v)
        b = random.uniform(b_key, (n,), minval=-v, maxval=v)
        a = jnp.ones_like(b)
        return w,b,a

    @staticmethod
    def network_fn(params, x):
        params = params["trainable"]["network"]["subdomain"]["layers"]
        for w, b, a in params[:-1]:
            x = jnp.dot(w, x) + b
            x = a*jnp.tanh(x/a)
        w, b, _ = params[-1]
        x = jnp.dot(w, x) + b
        return x

class SIREN(Network):
    "Fully connected network with sin activations"

    @staticmethod
    def init_params(key, layer_sizes):
        keys = random.split(key, len(layer_sizes)-1)
        params = [SIREN._random_layer_params(k, m, n)
                for k, m, n in zip(keys, layer_sizes[:-1], layer_sizes[1:])]
        trainable_params = {"layers": params}
        return {}, trainable_params

    @staticmethod
    def _random_layer_params(key, m, n):
        "Create a random layer parameters"

        w_key, b_key = random.split(key)
        v = jnp.sqrt(6/m)
        w = random.uniform(w_key, (n, m), minval=-v, maxval=v)
        b = random.uniform(b_key, (n,), minval=-v, maxval=v)
        return w,b

    @staticmethod
    def network_fn(params, x):
        params = params["trainable"]["network"]["subdomain"]["layers"]
        for w, b in params[:-1]:
            x = jnp.dot(w, x) + b
            x = jnp.sin(x)
        w, b = params[-1]
        x = jnp.dot(w, x) + b
        return x

class AdaptiveSIREN(Network):
    "Fully connected network with adaptive sin activations"

    @staticmethod
    def init_params(key, layer_sizes):
        keys = random.split(key, len(layer_sizes)-1)
        params = [AdaptiveSIREN._random_layer_params(k, m, n)
                for k, m, n in zip(keys, layer_sizes[:-1], layer_sizes[1:])]
        trainable_params = {"layers": params}
        return {}, trainable_params

    @staticmethod
    def _random_layer_params(key, m, n):
        "Create a random layer parameters"

        w_key, b_key = random.split(key)
        v = jnp.sqrt(6/m)
        w = random.uniform(w_key, (n, m), minval=-v, maxval=v)
        b = random.uniform(b_key, (n,), minval=-v, maxval=v)
        c,o = jnp.ones_like(b), jnp.ones_like(b)
        return w,b,c,o

    @staticmethod
    def network_fn(params, x):
        params = params["trainable"]["network"]["subdomain"]["layers"]
        for w,b,c,o in params[:-1]:
            x = jnp.dot(w, x) + b
            x = c*jnp.sin(o*x)
        w,b,_,_ = params[-1]
        x = jnp.dot(w, x) + b
        return x

class FourierFCN(FCN):
    "Fully connected network with Fourier features"

    @staticmethod
    def init_params(key, layer_sizes, mu, sd, n_features):

        # get Fourier feature parameters
        key, subkey = random.split(key)
        omega = 2*jnp.pi*(mu+sd*random.normal(subkey, (n_features, layer_sizes[0])))
        layer_sizes = [2*n_features]+list(layer_sizes)[1:]

        # get FCN parameters
        keys = random.split(key, len(layer_sizes)-1)
        params = [FCN._random_layer_params(k, m, n)
                for k, m, n in zip(keys, layer_sizes[:-1], layer_sizes[1:])]
        trainable_params = {"layers": params}
        return {"omega":omega}, trainable_params

    @staticmethod
    def network_fn(params, x):
        omega = params["static"]["network"]["subdomain"]["omega"]
        params = params["trainable"]["network"]["subdomain"]["layers"]
        x = jnp.dot(omega, x)
        x = jnp.concatenate([jnp.sin(x), jnp.cos(x)])# (2*n_features)
        for w, b in params[:-1]:
            x = jnp.dot(w, x) + b
            x = jnp.tanh(x)
        w, b = params[-1]
        x = jnp.dot(w, x) + b
        return x


def norm(mu, sd, x):
    return (x-mu)/sd

def unnorm(mu, sd, x):
    return x*sd + mu



if __name__ == "__main__":

    x = jnp.ones(2)
    key = random.PRNGKey(0)
    layer_sizes = [2,16,32,16,1]
    for NN in [FCN, AdaptiveFCN, SIREN, AdaptiveSIREN, FourierFCN]:
        network = NN
        if NN is FourierFCN:
            ps_ = network.init_params(key, layer_sizes, 0, 1, 10)
        else:
            ps_ = network.init_params(key, layer_sizes)
        params = {"static":{"network":{"subdomain":ps_[0]}},
                  "trainable":{"network":{"subdomain":ps_[1]}}}
        print(x.shape, network.network_fn(params, x).shape, NN.__name__)