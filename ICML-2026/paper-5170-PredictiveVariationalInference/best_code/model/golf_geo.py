import jax
import jax.scipy as jsc
from jax import random
import jax.numpy as jnp
from numpyro import distributions as dist
from data import get_matrix, get_matrix_test
import numpy as np

# Geometric model for partial golf data


class GolfGeo:
    gt = None
    def __init__(self, *args, **kwargs):
        x = []
        n = []
        y = []
        with open('data/golf.dat', 'r') as f:
            lines = f.readlines()
            for l in lines:
                a, b, c = l.split()
                x.append(float(a))
                n.append(int(b))
                y.append(int(c))
        self.x = np.array(x)
        self.n = np.array(n)
        self.y = np.array(y)
        self.d = 1
        self.r = (1.68/2)/12
        self.R = (4.25/2)/12


    def log_prior(self, theta):
        return jnp.sum(jsc.stats.norm.logpdf(theta, 0, 1))

    def log_likelihood(self, theta, y):
        return jnp.sum(self.log_likelihoods(theta, y))

    def log_likelihoods(self, theta, y):
        sigma = jnp.exp(theta)
        p = 2 * jsc.stats.norm.cdf( jnp.arcsin((self.R - self.r) / self.x) / sigma) - 1
        return dist.Binomial(self.n, probs=p).log_prob(self.y)

    def convert(self, theta):
        return theta

    def valid_log_likelihoods(self, theta):
        return 0

    def test_log_likelihoods(self, theta,):
        raise NotImplementedError()
    def sample_datapoint(self, key, theta):
        raise NotImplementedError()

    def data(self, key = None):
        return self.y

    def validate_crps(self, theta1, theta2, key):
        raise NotImplementedError()

    def test_crps(self, theta1, theta2, key, test_y = None):
        raise NotImplementedError()
    def likelihood_parameters(self, theta):
        raise NotImplementedError()

    def M(self, theta = None):
        return 1