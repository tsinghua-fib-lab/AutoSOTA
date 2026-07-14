import numpy as np
import numpyro.distributions as dist
import jax.numpy as jnp
import json
from numpyro.distributions.util import vec_to_tril_matrix
from jax import random

class GLMM:
    def __init__(self):
        with open('jax_posteriordb/data/GLMM_data.json', 'r') as file:
            data = json.load(file)

        self.nyear = data['nyear']
        self.nobs = data['nobs']
        self.nsite = data['nsite']
        self.obssite = np.array(data['obssite'])
        self.obsyear = np.array(data['obsyear'])
        self.obs = np.array(data['obs'])


        self.n = self.nsite + self.nyear + 3
        self.test_N = self.nobs // 5
        self.valid_N = self.nobs // 5
        self.train_N = int(self.nobs - self.valid_N - self.test_N)
        perm = np.random.RandomState(0).permutation(self.nobs).astype(int)

        self.site_train = self.obssite[perm[:self.train_N]]
        self.year_train = self.obsyear[perm[:self.train_N]]
        self.obs_train = self.obs[perm[:self.train_N]]

        self.site_valid = self.obssite[perm[self.train_N:self.train_N + self.valid_N]]
        self.year_valid = self.obsyear[perm[self.train_N:self.train_N + self.valid_N]]
        self.obs_valid = self.obs[perm[self.train_N:self.train_N + self.valid_N]]

        self.site_test = self.obssite[perm[self.train_N + self.valid_N:]]
        self.year_test = self.obsyear[perm[self.train_N + self.valid_N:]]
        self.obs_test = self.obs[perm[self.train_N + self.valid_N:]]

    def log_prior(self, theta):
        alpha, eps, mu, sd_alpha, sd_eps = self.theta2par(theta)
        log_p_alpha = jnp.sum(dist.Normal(0, jnp.exp(sd_alpha)).log_prob(alpha))
        log_p_eps = jnp.sum(dist.Normal(0, jnp.exp(sd_eps)).log_prob(eps))
        log_p_mu = dist.Normal(0, 10).log_prob(mu)
        log_p_sd1 = dist.Normal(1, 1).log_prob(sd_alpha)
        log_p_sd2 = dist.Normal(0, 1).log_prob(sd_eps)
        return log_p_mu + log_p_sd1 + log_p_sd2 + log_p_alpha + log_p_eps

    def theta2par(self, theta):
        alpha = theta[:self.nsite]
        eps = theta[self.nsite: self.nsite+self.nyear]
        mu = theta[self.nsite+self.nyear]
        sd_alpha = theta[self.nsite+self.nyear + 1]
        sd_eps = theta[self.nsite+self.nyear + 2]
        return alpha, eps, mu, sd_alpha, sd_eps


    def log_likelihoods(self, theta, *args, **kwargs):
        alpha, eps, mu, sd_alpha, sd_eps = self.theta2par(theta)
        return dist.Poisson(jnp.exp(mu + alpha[self.site_train] + eps[self.year_train])).log_prob(self.obs_train)
    def log_likelihood(self, theta, *args, **kwargs):
        return jnp.sum(self.log_likelihoods(theta))

    def valid_log_likelihoods(self, theta):
        alpha, eps, mu, sd_alpha, sd_eps = self.theta2par(theta)
        return dist.Poisson(jnp.exp(mu + alpha[self.site_valid] + eps[self.year_valid])).log_prob(self.obs_valid)

    def test_log_likelihoods(self, theta):
        alpha, eps, mu, sd_alpha, sd_eps = self.theta2par(theta)
        return dist.Poisson(jnp.exp(mu + alpha[self.site_test] + eps[self.year_test])).log_prob(self.obs_test)


if __name__ == '__main__':
    cls = GLMM()
    sd = random.PRNGKey(3)
    theta = random.normal(sd, (cls.n,))
    print(cls.log_prior(theta), jnp.mean(cls.log_likelihoods(theta)), jnp.mean(cls.valid_log_likelihoods(theta)), jnp.mean(cls.test_log_likelihoods(theta)))