import jax
import jax.scipy as jsc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from jax import random, grad, vmap
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro.distributions.util import vec_to_tril_matrix
import numpy as np

# Full-rank Gaussian parametric family


class BasicFullRank:
    def __init__(self, dim):
        self.dim = dim
        self.gradq = grad(self.log_posterior, argnums=1)

    def extract_params(self, params):
        loc = params[:self.dim]
        scale = params[self.dim:]
        return loc, scale

    def log_posterior(self, theta, params):
        loc, scale = self.extract_params(params)
        scale_tril = vec_to_tril_matrix(scale[self.dim:], diagonal = -1) + jnp.diag(jnp.maximum(1e-2, jnp.exp(scale[:self.dim])))

        return jnp.sum(dist.MultivariateNormal(loc, scale_tril=scale_tril).log_prob(theta))

    def sample(self, key, params, number = 1):
        loc, scale = self.extract_params(params)
        scale_tril = vec_to_tril_matrix(scale[self.dim:], diagonal = -1) + jnp.diag(jnp.maximum(1e-2, jnp.exp(scale[:self.dim])))
        return dist.MultivariateNormal(loc, scale_tril=scale_tril).sample(key, (number,))

    def posterior_parameters(self, params):
        loc, scale = self.extract_params(params)
        scale_tril = vec_to_tril_matrix(scale[self.dim:], diagonal = -1) + jnp.diag(jnp.maximum(1e-2, jnp.exp(scale[:self.dim])))
        return (loc, scale_tril)


    def gen_params(self):
        return jnp.zeros(self.dim + self.dim * (self.dim + 1) // 2)

    def get_grad(self, theta, params):
        return self.gradq(theta, params)



