import numpy as np
import numpyro.distributions as dist
import jax.numpy as jnp
import json
from numpyro.distributions.util import vec_to_tril_matrix
from jax import random

class Wells:
    def __init__(self):
        with open('jax_posteriordb/data/wells_data.json', 'r') as file:
            data = json.load(file)

        self.N = data['N']
        self.switched = np.array(data['switched'])
        self.dist = np.array(data['dist'])
        self.arsenic = np.array(data['arsenic'])
        self.educ = np.array(data['educ'])


        self.n = 7
        self.test_N = self.N // 5
        self.valid_N = self.N // 5
        self.train_N = int(self.N - self.valid_N - self.test_N)
        perm = np.random.RandomState(0).permutation(self.N).astype(int)
        self.id_train = perm[:self.train_N]
        self.id_valid = perm[self.train_N:self.train_N + self.valid_N]
        self.id_test = perm[self.train_N + self.valid_N:]


        self.c_dist100 = (self.dist - np.mean(self.dist)) / 100.0
        self.c_arsenic = self.arsenic - np.mean(self.arsenic)
        self.c_educ4 = (self.educ - np.mean(self.educ)) / 4.0
        self.da_inter = self.c_dist100 * self.c_arsenic
        self.de_inter = self.c_dist100 * self.c_educ4
        self.ae_inter = self.c_arsenic * self.c_educ4

    def log_prior(self, theta):
        return jnp.sum(dist.Normal().log_prob(theta))

    def theta2par(self, theta):
        alpha = theta[0]
        beta = theta[1:]
        return alpha, beta

    def log_likelihoods(self, theta, *args, **kwargs):
        alpha, beta = self.theta2par(theta)
        logits = alpha + beta[0] * self.c_dist100[self.id_train] + beta[1] * self.c_arsenic[self.id_train] + beta[2] * self.c_educ4[self.id_train] \
                + beta[3] * self.da_inter[self.id_train] + beta[4] * self.de_inter[self.id_train] + beta[5] * self.ae_inter[self.id_train]
        return dist.Bernoulli(logits=logits).log_prob(self.switched[self.id_train])

    def logits(self, theta):
        alpha, beta = self.theta2par(theta)
        logits = alpha + beta[0] * self.c_dist100[self.id_train] + beta[1] * self.c_arsenic[self.id_train] + beta[2] * \
                 self.c_educ4[self.id_train] \
                 + beta[3] * self.da_inter[self.id_train] + beta[4] * self.de_inter[self.id_train] + beta[5] * \
                 self.ae_inter[self.id_train]
        return logits
    def log_likelihood(self, theta, *args, **kwargs):
        return jnp.sum(self.log_likelihoods(theta))

    def data(self):
        return self.switched[self.id_train]

    def valid_log_likelihoods(self, theta):
        alpha, beta = self.theta2par(theta)
        logits = alpha + beta[0] * self.c_dist100[self.id_valid] + beta[1] * self.c_arsenic[self.id_valid] + beta[2] * self.c_educ4[self.id_valid] \
                + beta[3] * self.da_inter[self.id_valid] + beta[4] * self.de_inter[self.id_valid] + beta[5] * self.ae_inter[self.id_valid]
        return dist.Bernoulli(logits=logits).log_prob(self.switched[self.id_valid])

    def valid_logits(self, theta):
        alpha, beta = self.theta2par(theta)
        logits = alpha + beta[0] * self.c_dist100[self.id_valid] + beta[1] * self.c_arsenic[self.id_valid] + beta[2] * \
                 self.c_educ4[self.id_valid] \
                 + beta[3] * self.da_inter[self.id_valid] + beta[4] * self.de_inter[self.id_valid] + beta[5] * \
                 self.ae_inter[self.id_valid]
        return logits

    def valid_data(self):
        return self.switched[self.id_valid]

    def test_log_likelihoods(self, theta):
        alpha, beta = self.theta2par(theta)
        logits = alpha + beta[0] * self.c_dist100[self.id_test] + beta[1] * self.c_arsenic[self.id_test] + beta[2] * self.c_educ4[self.id_test] \
                + beta[3] * self.da_inter[self.id_test] + beta[4] * self.de_inter[self.id_test] + beta[5] * self.ae_inter[self.id_test]
        return dist.Bernoulli(logits=logits).log_prob(self.switched[self.id_test])

    def test_logits(self, theta):
        alpha, beta = self.theta2par(theta)
        logits = alpha + beta[0] * self.c_dist100[self.id_test] + beta[1] * self.c_arsenic[self.id_test] + beta[2] * \
                 self.c_educ4[self.id_test] \
                 + beta[3] * self.da_inter[self.id_test] + beta[4] * self.de_inter[self.id_test] + beta[5] * \
                 self.ae_inter[self.id_test]
        return logits

    def test_data(self):
        return self.switched[self.id_test]

if __name__ == '__main__':
    cls = Wells()
    sd = random.PRNGKey(3)
    theta = random.normal(sd, (cls.n,))
    print(cls.log_prior(theta), jnp.mean(cls.log_likelihoods(theta)), jnp.mean(cls.valid_log_likelihoods(theta)), jnp.mean(cls.test_log_likelihoods(theta)))