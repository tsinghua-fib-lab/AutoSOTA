import numpy as np
import numpyro.distributions as dist
import jax.numpy as jnp
import json
from numpyro.distributions.util import vec_to_tril_matrix
from jax import random

class KidScore:
    def __init__(self):
        with open('jax_posteriordb/data/kidiq.json', 'r') as file:
            data = json.load(file)

        self.N = data['N']
        self.kid_score = np.array(data['kid_score'])
        self.mom_hs = np.array(data['mom_hs'])
        self.mom_iq = np.array(data['mom_iq'])


        self.n = 5
        self.test_N = self.N // 5
        self.valid_N = self.N // 5
        self.train_N = int(self.N - self.valid_N - self.test_N)
        perm = np.random.RandomState(0).permutation(self.N).astype(int)
        self.id_train = perm[:self.train_N]
        self.id_valid = perm[self.train_N:self.train_N + self.valid_N]
        self.id_test = perm[self.train_N + self.valid_N:]

        self.z_mom_hs = (self.mom_hs - np.mean(self.mom_hs)) / (2 * np.std(self.mom_hs))
        self.z_mom_iq = (self.mom_iq - np.mean(self.mom_iq)) / (2 * np.std(self.mom_iq))
        self.inter = self.z_mom_hs * self.z_mom_iq

    def log_prior(self, theta):
        alpha, sigma = self.theta2par(theta)
        logpalpha = jnp.sum(dist.Normal().log_prob(alpha))
        logpsigma = dist.HalfNormal().log_prob(jnp.exp(sigma)) + sigma
        return logpalpha + logpsigma

    def theta2par(self, theta):
        sigma = theta[0]
        alpha = theta[1:]
        return alpha, sigma

    def log_likelihoods(self, theta, *args, **kwargs):
        alpha, sigma = self.theta2par(theta)
        mu = alpha[0] + alpha[1] * self.z_mom_hs[self.id_train] + alpha[2] * self.z_mom_iq[self.id_train] + alpha[3] * self.inter[self.id_train]
        return dist.Normal(mu, jnp.exp(sigma)).log_prob(self.kid_score[self.id_train])

    def log_likelihood(self, theta, *args, **kwargs):
        return jnp.sum(self.log_likelihoods(theta))

    def data(self):
        return self.kid_score[self.id_train]

    def sample_datapoint(self, key, theta):
        alpha, sigma = self.theta2par(theta)
        mu = alpha[0] + alpha[1] * self.z_mom_hs[self.id_train] + alpha[2] * self.z_mom_iq[self.id_train] + alpha[3] * self.inter[self.id_train]
        return dist.Normal(mu, jnp.exp(sigma)).sample(key)

    def valid_log_likelihoods(self, theta):
        alpha, sigma = self.theta2par(theta)
        mu = alpha[0] + alpha[1] * self.z_mom_hs[self.id_valid] + alpha[2] * self.z_mom_iq[self.id_valid] + alpha[3] * self.inter[self.id_valid]
        return dist.Normal(mu, jnp.exp(sigma)).log_prob(self.kid_score[self.id_valid])

    def sample_valid_datapoint(self, key, theta):
        alpha, sigma = self.theta2par(theta)
        mu = alpha[0] + alpha[1] * self.z_mom_hs[self.id_valid] + alpha[2] * self.z_mom_iq[self.id_valid] + alpha[3] * self.inter[self.id_valid]
        return dist.Normal(mu, jnp.exp(sigma)).sample(key)

    def valid_data(self):
        return self.kid_score[self.id_valid]

    def test_log_likelihoods(self, theta):
        alpha, sigma = self.theta2par(theta)
        mu = alpha[0] + alpha[1] * self.z_mom_hs[self.id_test] + alpha[2] * self.z_mom_iq[self.id_test] + alpha[3] * self.inter[self.id_test]
        return dist.Normal(mu, jnp.exp(sigma)).log_prob(self.kid_score[self.id_test])

    def sample_test_datapoint(self, key, theta):
        alpha, sigma = self.theta2par(theta)
        mu = alpha[0] + alpha[1] * self.z_mom_hs[self.id_test] + alpha[2] * self.z_mom_iq[self.id_test] + alpha[3] * self.inter[self.id_test]
        return dist.Normal(mu, jnp.exp(sigma)).sample(key)

    def test_data(self):
        return self.kid_score[self.id_test]

if __name__ == '__main__':
    cls = KidScore()
    sd = random.PRNGKey(3)
    theta = random.normal(sd, (cls.n,))
    print(cls.log_prior(theta), jnp.mean(cls.log_likelihoods(theta)), jnp.mean(cls.valid_log_likelihoods(theta)), jnp.mean(cls.test_log_likelihoods(theta)))