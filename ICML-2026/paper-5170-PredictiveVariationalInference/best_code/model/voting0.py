import jax
import jax.scipy as jsc
from jax import random
import jax.numpy as jnp
from numpyro import distributions as dist

# The synthesized voting model with the formula: 1

class Voting0:
    da = 50
    db = 5
    dc = 5
    d = 1 #da + db# + dc
    valid_y = None
    test_y = None
    gt = None
    def __init__(self, N = 100, *args, **kwargs):
        self.N = N

    def log_prior(self, theta):
        return jnp.sum(jsc.stats.norm.logpdf(theta, 0, 1))

    def log_likelihood(self, theta, y):
        u = theta[0]
        return jnp.sum(dist.Binomial(self.N, logits=u).log_prob(y))

    def log_likelihoods(self, theta, y):
        u = theta[0]
        return jnp.sum(dist.Binomial(self.N, logits=u).log_prob(y))

    def valid_log_likelihoods(self, theta, y):
        u = theta[0]
        return jnp.sum(dist.Binomial(self.N, logits=u).log_prob(y))

    def test_log_likelihoods(self, theta, y):
        u = theta[0]
        return jnp.sum(dist.Binomial(self.N, logits=u).log_prob(y))

    def sample_datapoint(self, key, theta):
        raise NotImplementedError()

    def data(self, key = None):
        coeff = jnp.arange(self.dc) + 1
        key1, key2, key3, key4, key5, key6, key7 = random.split(key, 7)
        u1 = random.normal(key1, (self.da, 1, 1))
        u2 = random.normal(key2, (1, self.db, 1))
        u3 = jnp.reshape(coeff, (1, 1, self.dc)) * random.normal(key3, (self.da, 1, 1))
        u = u1 + u2 + u3

        self.gt = jnp.concatenate([u1.flatten(), u2.flatten(), u3.flatten(),])
        data = dist.Binomial(self.N, logits = u).sample(key5, )
        self.valid_y = dist.Binomial(self.N, logits = u).sample(key6, )
        self.test_y = dist.Binomial(self.N, logits = u).sample(key7, )
        return data

    def validate_crps(self, theta1, theta2, key):
        raise NotImplementedError()

    def test_crps(self, theta1, theta2, key, test_y = None):
        raise NotImplementedError()
    def likelihood_parameters(self, theta):
        raise NotImplementedError()

    def M(self, theta = None):
        return 1