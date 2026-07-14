import jax
import jax.scipy as jsc
from jax import random, grad
import jax.numpy as jnp
import numpyro.distributions as dist
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Diagonal Gaussian parametric family

class Basic:
    def __init__(self, dim):
        self.dim = dim
        self.gradq = grad(self.log_posterior, argnums=1)

    def extract_params(self, params):
        loc = params[:self.dim]
        logscale = params[self.dim:]
        return loc, logscale

    def log_posterior(self, theta, params):
        loc, logscale = self.extract_params(params)
        scale = jnp.exp(logscale)
        return jnp.sum(jsc.stats.norm.logpdf(theta, loc, scale))

    def sample(self, key, params, number = 1):
        loc, logscale = self.extract_params(params)
        scale = jnp.exp(logscale)
        return random.normal(key, shape=(number, self.dim, )) * scale + loc

    def posterior_parameters(self, params):
        loc, logscale = self.extract_params(params)
        scale = jnp.exp(logscale)
        return (loc, scale)


    def gen_params(self):
        return jnp.zeros(self.dim * 2)

    def get_grad(self, theta, params):
        return self.gradq(theta, params)



