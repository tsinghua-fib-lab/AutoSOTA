import numpy as np
import numpyro.distributions as dist
import jax.numpy as jnp
import json
from numpyro.distributions.util import vec_to_tril_matrix
from jax import random

class Earnings:
    def __init__(self):
        with open('jax_posteriordb/data/earnings.json', 'r') as file:
            data = json.load(file)

        self.N = data['N']
        self.earn = np.array(data['earn'])
        self.height = np.array(data['height'])
        self.male = np.array(data['male'])

        self.logearn = np.log(self.earn)
        self.zheight =  (self.height - np.mean(self.height)) / np.std(self.height)
        self.inter = self.zheight * self.male

        self.n = 5
        self.test_N = self.N // 5
        self.valid_N = self.N // 5
        self.train_N = int(self.N - self.valid_N - self.test_N)
        perm = np.random.RandomState(0).permutation(self.N).astype(int)
        self.id_train = perm[:self.train_N]
        self.id_valid = perm[self.train_N:self.train_N + self.valid_N]
        self.id_test = perm[self.train_N + self.valid_N:]

    def log_prior(self, theta):
        beta, logsigma = self.theta2par(theta)
        pbeta = dist.Normal().log_prob(beta)
        plogsigma = dist.Normal().log_prob(logsigma)
        return np.sum(pbeta) + plogsigma


    def theta2par(self, theta):
        beta = theta[:4]
        logsigma = theta[4]
        return beta, logsigma


    def log_likelihoods(self, theta, *args, **kwargs):
        beta, logsigma = self.theta2par(theta)
        sigma = jnp.exp(logsigma)
        mu = beta[0] + beta[1] * self.zheight[self.id_train] + beta[2] * self.male[self.id_train] + beta[3] * self.inter[self.id_train]
        return dist.Normal(mu, sigma).log_prob(self.logearn[self.id_train])

    def log_likelihood(self, theta, *args, **kwargs):
        return jnp.sum(self.log_likelihoods(theta))

    def data(self):
        return self.logearn[self.id_train]

    def sample_datapoint(self, key, theta):
        beta, logsigma = self.theta2par(theta)
        sigma = jnp.exp(logsigma)
        mu = beta[0] + beta[1] * self.zheight[self.id_train] + beta[2] * self.male[self.id_train] + beta[3] * self.inter[self.id_train]
        return dist.Normal(mu, sigma).sample(key)

    def valid_log_likelihoods(self, theta):
        beta, logsigma = self.theta2par(theta)
        sigma = jnp.exp(logsigma)
        mu = beta[0] + beta[1] * self.zheight[self.id_valid] + beta[2] * self.male[self.id_valid] + beta[3] * self.inter[self.id_valid]
        return dist.Normal(mu, sigma).log_prob(self.logearn[self.id_valid])

    def sample_valid_datapoint(self, key, theta):
        beta, logsigma = self.theta2par(theta)
        sigma = jnp.exp(logsigma)
        mu = beta[0] + beta[1] * self.zheight[self.id_valid] + beta[2] * self.male[self.id_valid] + beta[3] * self.inter[self.id_valid]
        return dist.Normal(mu, sigma).sample(key)

    def valid_data(self):
        return self.logearn[self.id_valid]

    def test_log_likelihoods(self, theta):
        beta, logsigma = self.theta2par(theta)
        sigma = jnp.exp(logsigma)
        mu = beta[0] + beta[1] * self.zheight[self.id_test] + beta[2] * self.male[self.id_test] + beta[3] * self.inter[self.id_test]
        return dist.Normal(mu, sigma).log_prob(self.logearn[self.id_test])

    def sample_test_datapoint(self, key, theta):
        beta, logsigma = self.theta2par(theta)
        sigma = jnp.exp(logsigma)
        mu = beta[0] + beta[1] * self.zheight[self.id_test] + beta[2] * self.male[self.id_test] + beta[3] * self.inter[self.id_test]
        return dist.Normal(mu, sigma).sample(key)

    def test_data(self):
        return self.logearn[self.id_test]

if __name__ == '__main__':
    cls = Radon()
    sd = random.PRNGKey(3)
    theta = random.normal(sd, (cls.n,))
    print(cls.log_prior(theta), jnp.mean(cls.log_likelihoods(theta)), jnp.mean(cls.valid_log_likelihoods(theta)), jnp.mean(cls.test_log_likelihoods(theta)))