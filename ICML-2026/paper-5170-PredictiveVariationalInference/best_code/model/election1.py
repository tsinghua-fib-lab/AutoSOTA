import jax
import jax.scipy as jsc
from jax import random
import jax.numpy as jnp
from numpyro import distributions as dist
from data import get_matrix, get_matrix_test, get_matrix_valid
import numpy as np

class ElectionIncomplete:
    gt = None
    def __init__(self, *args, **kwargs):
        self.shape, self.N, self.y = get_matrix(np.array([1, 2, 3]))
        _, self.N_valid, self.y_valid = get_matrix_valid(np.array([1, 2, 3]))

        _, self.N_test, self.y_test = get_matrix_test(np.array([1, 2, 3]))

        self.da = self.shape[0]
        self.db = self.shape[1]
        self.dc = self.shape[2]
        self.d = self.da + self.db + 1
        self.coeff = np.arange(self.dc) + 1

    def log_prior(self, theta):
        return jnp.sum(jsc.stats.norm.logpdf(theta, 0, 1))

    def log_likelihood(self, theta, y):
        return jnp.sum(self.log_likelihoods(theta, y))

    def log_likelihoods(self, theta, y=None):
        if y is None:
            y = self.y
        ua = jnp.reshape(theta[..., :self.da], (self.da, 1, 1))
        ub = jnp.reshape(theta[..., self.da:self.da + self.db], (1, self.db, 1))
        uc = jnp.reshape(self.coeff, (1, 1, self.dc)) * theta[-1]
        u = ua + ub + uc# + theta[-1]
        return dist.Binomial(self.N, logits=u).log_prob(y)

    def convert(self, theta):
        return theta

    def valid_log_likelihoods(self, theta,):
        ua = jnp.reshape(theta[..., :self.da], (self.da, 1, 1))
        ub = jnp.reshape(theta[..., self.da:self.da + self.db], (1, self.db, 1))
        uc = jnp.reshape(self.coeff, (1, 1, self.dc)) * theta[-1]
        u = ua + ub + uc# + theta[-1]
        return dist.Binomial(self.N_valid, logits=u).log_prob(self.y_valid)

    def test_log_likelihoods(self, theta,):
        ua = jnp.reshape(theta[..., :self.da], (self.da, 1, 1))
        ub = jnp.reshape(theta[..., self.da:self.da + self.db], (1, self.db, 1))
        uc = jnp.reshape(self.coeff, (1, 1, self.dc)) * theta[-1]
        u = ua + ub + uc# + theta[-1]
        return dist.Binomial(self.N_test, logits=u).log_prob(self.y_test)

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