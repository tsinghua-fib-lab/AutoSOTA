import numpy as np
import numpyro.distributions as dist
import jax.numpy as jnp
import json
from numpyro.distributions.util import vec_to_tril_matrix
from jax import random

class NES:
    def __init__(self):
        with open('jax_posteriordb/data/nes2000.json', 'r') as file:
            data = json.load(file)

        self.N = data['N']
        self.partyid7 = np.array(data['partyid7'])
        self.real_ideo = np.array(data['real_ideo'])
        self.race_adj = np.array(data['race_adj'])
        self.educ1 = np.array(data['educ1'])
        self.gender = np.array(data['gender'])
        self.income = np.array(data['income'])
        self.age_discrete = np.array(data['age_discrete'])
        self.age30_44 = self.age_discrete == 2
        self.age45_64 = self.age_discrete == 3
        self.age65up = self.age_discrete == 4

        self.n = 10
        self.test_N = self.N // 5
        self.valid_N = self.N // 5
        self.train_N = int(self.N - self.valid_N - self.test_N)
        perm = np.random.RandomState(0).permutation(self.N).astype(int)
        self.id_train = perm[:self.train_N]
        self.id_valid = perm[self.train_N:self.train_N + self.valid_N]
        self.id_test = perm[self.train_N + self.valid_N:]

    def log_prior(self, theta):
        beta, sigma = self.theta2par(theta)
        logpbeta = jnp.sum(dist.Normal().log_prob(beta))
        logpsigma = dist.HalfNormal().log_prob(sigma)
        return logpbeta + logpsigma


    def theta2par(self, theta):
        beta = theta[:9]
        sigma = theta[9]
        return beta, sigma


    def log_likelihoods(self, theta, *args, **kwargs):
        beta, sigma = self.theta2par(theta)
        sigma = jnp.exp(sigma)
        mu = beta[0] + beta[1] * self.real_ideo[self.id_train] + beta[2] * self.race_adj[self.id_train] + beta[3] * self.age30_44[self.id_train] + beta[4] * self.age45_64[self.id_train] + beta[5] * self.age65up[self.id_train] + beta[6] * self.educ1[self.id_train] + beta[7] * self.gender[self.id_train] + beta[8] * self.income[self.id_train]
        return dist.Normal(mu, sigma).log_prob(self.partyid7[self.id_train])

    def log_likelihood(self, theta, *args, **kwargs):
        return jnp.sum(self.log_likelihoods(theta))

    def data(self):
        return self.partyid7[self.id_train]

    def sample_datapoint(self, key, theta):
        beta, sigma = self.theta2par(theta)
        sigma = jnp.exp(sigma)
        mu = beta[0] + beta[1] * self.real_ideo[self.id_train] + beta[2] * self.race_adj[self.id_train] + beta[3] * self.age30_44[self.id_train] + beta[4] * self.age45_64[self.id_train] + beta[5] * self.age65up[self.id_train] + beta[6] * self.educ1[self.id_train] + beta[7] * self.gender[self.id_train] + beta[8] * self.income[self.id_train]
        return dist.Normal(mu, sigma).sample(key)

    def valid_log_likelihoods(self, theta):
        beta, sigma = self.theta2par(theta)
        sigma = jnp.exp(sigma)

        mu = beta[0] + beta[1] * self.real_ideo[self.id_valid] + beta[2] * self.race_adj[self.id_valid] + beta[3] * self.age30_44[self.id_valid] + beta[4] * self.age45_64[self.id_valid] + beta[5] * self.age65up[self.id_valid] + beta[6] * self.educ1[self.id_valid] + beta[7] * self.gender[self.id_valid] + beta[8] * self.income[self.id_valid]
        return dist.Normal(mu, sigma).log_prob(self.partyid7[self.id_valid])

    def sample_valid_datapoint(self, key, theta):
        beta, sigma = self.theta2par(theta)
        sigma = jnp.exp(sigma)

        mu = beta[0] + beta[1] * self.real_ideo[self.id_valid] + beta[2] * self.race_adj[self.id_valid] + beta[3] * self.age30_44[self.id_valid] + beta[4] * self.age45_64[self.id_valid] + beta[5] * self.age65up[self.id_valid] + beta[6] * self.educ1[self.id_valid] + beta[7] * self.gender[self.id_valid] + beta[8] * self.income[self.id_valid]
        return dist.Normal(mu, sigma).sample(key)

    def valid_data(self):
        return self.partyid7[self.id_valid]

    def test_log_likelihoods(self, theta):
        beta, sigma = self.theta2par(theta)
        sigma = jnp.exp(sigma)
        mu = beta[0] + beta[1] * self.real_ideo[self.id_test] + beta[2] * self.race_adj[self.id_test] + beta[3] * self.age30_44[self.id_test] + beta[4] * self.age45_64[self.id_test] + beta[5] * self.age65up[self.id_test] + beta[6] * self.educ1[self.id_test] + beta[7] * self.gender[self.id_test] + beta[8] * self.income[self.id_test]
        return dist.Normal(mu, sigma).log_prob(self.partyid7[self.id_test])

    def sample_test_datapoint(self, key, theta):
        beta, sigma = self.theta2par(theta)
        sigma = jnp.exp(sigma)
        mu = beta[0] + beta[1] * self.real_ideo[self.id_test] + beta[2] * self.race_adj[self.id_test] + beta[3] * self.age30_44[self.id_test] + beta[4] * self.age45_64[self.id_test] + beta[5] * self.age65up[self.id_test] + beta[6] * self.educ1[self.id_test] + beta[7] * self.gender[self.id_test] + beta[8] * self.income[self.id_test]
        return dist.Normal(mu, sigma).sample(key)

    def test_data(self):
        return self.partyid7[self.id_test]

if __name__ == '__main__':
    cls = NES()
    sd = random.PRNGKey(3)
    theta = random.normal(sd, (cls.n,))
    print(cls.log_prior(theta), jnp.mean(cls.log_likelihoods(theta)), jnp.mean(cls.valid_log_likelihoods(theta)), jnp.mean(cls.test_log_likelihoods(theta)))