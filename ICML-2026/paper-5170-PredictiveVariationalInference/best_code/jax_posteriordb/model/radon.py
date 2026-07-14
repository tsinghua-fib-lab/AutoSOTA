import numpy as np
import numpyro.distributions as dist
import jax.numpy as jnp
import json
from numpyro.distributions.util import vec_to_tril_matrix
from jax import random

class Radon:
    def __init__(self):
        with open('jax_posteriordb/data/radon_all.json', 'r') as file:
            data = json.load(file)

        self.N = data['N']
        self.J = data['J']
        self.county_idx = np.array(data['county_idx'])
        self.floor_measure = np.array(data['floor_measure'])
        self.log_radon = np.array(data['log_radon'])


        self.n = 2 * self.J + 5
        self.test_N = self.N // 5
        self.valid_N = self.N // 5
        self.train_N = int(self.N - self.valid_N - self.test_N)
        perm = np.random.RandomState(0).permutation(self.N).astype(int)
        self.id_train = perm[:self.train_N]
        self.id_valid = perm[self.train_N:self.train_N + self.valid_N]
        self.id_test = perm[self.train_N + self.valid_N:]

    def log_prior(self, theta):
        alpha, beta, mu_alpha, mu_beta, sigma_alpha, sigma_beta, sigma_y = self.theta2par(theta)
        logpalpha = jnp.sum(dist.Normal(mu_alpha, jnp.exp(sigma_alpha)).log_prob(alpha))
        logpbeta = jnp.sum(dist.Normal(mu_beta, jnp.exp(sigma_beta)).log_prob(beta))
        logpsigma1 = dist.Normal().log_prob(sigma_alpha)
        logpsigma2 = dist.Normal().log_prob(sigma_beta)
        logpsigma3 = dist.Normal().log_prob(sigma_y)
        logpmu1 = dist.Normal(0, 10).log_prob(mu_alpha)
        logpmu2 = dist.Normal(0, 10).log_prob(mu_beta)

        return logpalpha + logpbeta + logpsigma1 + logpsigma2 + logpsigma3 + logpmu1 + logpmu2


    def theta2par(self, theta):
        alpha = theta[:self.J]
        beta = theta[self.J:self.J * 2]
        sigma_y = theta[self.J * 2]
        sigma_alpha = theta[self.J * 2 + 1]
        sigma_beta = theta[self.J * 2 + 2]
        mu_alpha = theta[self.J * 2 + 3]
        mu_beta = theta[self.J * 2 + 4]
        return alpha, beta, mu_alpha, mu_beta, sigma_alpha, sigma_beta, sigma_y


    def log_likelihoods(self, theta, *args, **kwargs):
        alpha, beta, mu_alpha, mu_beta, sigma_alpha, sigma_beta, sigma_y = self.theta2par(theta)
        sigma_y = jnp.exp(sigma_y)
        mu = alpha[self.county_idx[self.id_train]] + self.floor_measure[self.id_train] * beta[self.county_idx[self.id_train]]
        return dist.Normal(mu, sigma_y).log_prob(self.log_radon[self.id_train])

    def log_likelihood(self, theta, *args, **kwargs):
        return jnp.sum(self.log_likelihoods(theta))

    def data(self):
        return self.log_radon[self.id_train]

    def sample_datapoint(self, key, theta):
        alpha, beta, mu_alpha, mu_beta, sigma_alpha, sigma_beta, sigma_y = self.theta2par(theta)
        sigma_y = jnp.exp(sigma_y)
        mu = alpha[self.county_idx[self.id_train]] + self.floor_measure[self.id_train] * beta[self.county_idx[self.id_train]]
        return dist.Normal(mu, sigma_y).sample(key)

    def valid_log_likelihoods(self, theta):
        alpha, beta, mu_alpha, mu_beta, sigma_alpha, sigma_beta, sigma_y = self.theta2par(theta)
        sigma_y = jnp.exp(sigma_y)
        mu = alpha[self.county_idx[self.id_valid]] + self.floor_measure[self.id_valid] * beta[self.county_idx[self.id_valid]]
        return dist.Normal(mu, sigma_y).log_prob(self.log_radon[self.id_valid])

    def sample_valid_datapoint(self, key, theta):
        alpha, beta, mu_alpha, mu_beta, sigma_alpha, sigma_beta, sigma_y = self.theta2par(theta)
        sigma_y = jnp.exp(sigma_y)
        mu = alpha[self.county_idx[self.id_valid]] + self.floor_measure[self.id_valid] * beta[self.county_idx[self.id_valid]]
        return dist.Normal(mu, sigma_y).sample(key)

    def valid_data(self):
        return self.log_radon[self.id_valid]

    def test_log_likelihoods(self, theta):
        alpha, beta, mu_alpha, mu_beta, sigma_alpha, sigma_beta, sigma_y = self.theta2par(theta)
        sigma_y = jnp.exp(sigma_y)
        mu = alpha[self.county_idx[self.id_test]] + self.floor_measure[self.id_test] * beta[self.county_idx[self.id_test]]
        return dist.Normal(mu, sigma_y).log_prob(self.log_radon[self.id_test])

    def sample_test_datapoint(self, key, theta):
        alpha, beta, mu_alpha, mu_beta, sigma_alpha, sigma_beta, sigma_y = self.theta2par(theta)
        sigma_y = jnp.exp(sigma_y)
        mu = alpha[self.county_idx[self.id_test]] + self.floor_measure[self.id_test] * beta[self.county_idx[self.id_test]]
        return dist.Normal(mu, sigma_y).sample(key)

    def test_data(self):
        return self.log_radon[self.id_test]

if __name__ == '__main__':
    cls = Radon()
    sd = random.PRNGKey(3)
    theta = random.normal(sd, (cls.n,))
    print(cls.log_prior(theta), jnp.mean(cls.log_likelihoods(theta)), jnp.mean(cls.valid_log_likelihoods(theta)), jnp.mean(cls.test_log_likelihoods(theta)))