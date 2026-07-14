import numpy as np
import numpyro.distributions as dist
import jax.numpy as jnp
import json
from numpyro.distributions.util import vec_to_tril_matrix
from jax import random

class Election:
    def __init__(self):
        with open('jax_posteriordb/data/election88.json', 'r') as file:
            data = json.load(file)

        self.N = data['N']
        self.n_age = data['n_age']
        self.n_age_edu = data['n_age_edu']
        self.n_edu = data['n_edu']
        self.n_region_full = data['n_region_full']
        self.n_state = data['n_state']
        self.age = np.array(data['age'])
        self.age_edu = np.array(data['age_edu'])
        self.black = np.array(data['black'])
        self.edu = np.array(data['edu'])
        self.female = np.array(data['female'])
        self.region_full = np.array(data['region_full'])
        self.state = np.array(data['state'])
        self.v_prev_full = np.array(data['v_prev_full'])
        self.y = np.array(data['y'])

        self.n = self.n_age + self.n_age_edu + self.n_edu + self.n_region_full + self.n_state + 10
        self.test_N = self.N // 5
        self.valid_N = self.N // 5
        self.train_N = int(self.N - self.valid_N - self.test_N)
        perm = np.random.RandomState(0).permutation(self.N).astype(int)
        self.id_train = perm[:self.train_N]
        self.id_valid = perm[self.train_N:self.train_N + self.valid_N]
        self.id_test = perm[self.train_N + self.valid_N:]

    def log_prior(self, theta):
        a, b, c, d, e, beta, sigma_a, sigma_b, sigma_c, sigma_d, sigma_e = self.theta2par(theta)
        logs1 = dist.Normal(3, 1).log_prob(sigma_a)
        logs2 = dist.Normal(3, 1).log_prob(sigma_b)
        logs3 = dist.Normal(3, 1).log_prob(sigma_c)
        logs4 = dist.Normal(3, 1).log_prob(sigma_d)
        logs5 = dist.Normal(3, 1).log_prob(sigma_e)
        logpbeta = jnp.sum(dist.Normal(0, 100).log_prob(beta))
        logpa = jnp.sum(dist.Normal(0, jnp.exp(sigma_a)).log_prob(a))
        logpb = jnp.sum(dist.Normal(0, jnp.exp(sigma_b)).log_prob(b))
        logpc = jnp.sum(dist.Normal(0, jnp.exp(sigma_c)).log_prob(c))
        logpd = jnp.sum(dist.Normal(0, jnp.exp(sigma_d)).log_prob(d))
        logpe = jnp.sum(dist.Normal(0, jnp.exp(sigma_e)).log_prob(e))
        return logs1 + logs2 + logs3 + logs4 + logs5 + logpbeta + logpa + logpb + logpc + logpd + logpe


    def theta2par(self, theta):
        a = theta[:self.n_age]
        b = theta[self.n_age: self.n_age + self.n_age_edu]
        c = theta[self.n_age + self.n_age_edu: self.n_age + self.n_age_edu + self.n_edu]
        d = theta[self.n_age + self.n_age_edu + self.n_edu : self.n_age + self.n_age_edu + self.n_edu + self.n_region_full]
        e = theta[self.n_age + self.n_age_edu + self.n_edu + self.n_region_full:self.n_age + self.n_age_edu + self.n_edu + self.n_region_full + self.n_state]
        beta = theta[-10:-5]
        sigma_a = theta[-5]
        sigma_b = theta[-4]
        sigma_c = theta[-3]
        sigma_d = theta[-2]
        sigma_e = theta[-1]
        return a, b, c, d, e, beta, sigma_a, sigma_b, sigma_c, sigma_d, sigma_e


    def log_likelihoods(self, theta, *args, **kwargs):
        a, b, c, d, e, beta, sigma_a, sigma_b, sigma_c, sigma_d, sigma_e = self.theta2par(theta)
        logits = beta[0] + beta[1] * self.black[self.id_train] + beta[2] * self.female[self.id_train] + beta[4] * self.female[self.id_train] * self.black[self.id_train] + beta[3] * self.v_prev_full[self.id_train] + a[self.age[self.id_train]] + b[self.edu[self.id_train]] + c[self.age_edu[self.id_train]] + d[self.state[self.id_train]] + e[self.region_full[self.id_train]]
        return dist.Bernoulli(logits=logits).log_prob(self.y[self.id_train])

    def logits(self, theta):
        a, b, c, d, e, beta, sigma_a, sigma_b, sigma_c, sigma_d, sigma_e = self.theta2par(theta)
        logits = beta[0] + beta[1] * self.black[self.id_train] + beta[2] * self.female[self.id_train] + beta[4] * \
                 self.female[self.id_train] * self.black[self.id_train] + beta[3] * self.v_prev_full[self.id_train] + a[
                     self.age[self.id_train]] + b[self.edu[self.id_train]] + c[self.age_edu[self.id_train]] + d[
                     self.state[self.id_train]] + e[self.region_full[self.id_train]]
        return logits

    def log_likelihood(self, theta, *args, **kwargs):
        return jnp.sum(self.log_likelihoods(theta))

    def data(self):
        return self.y[self.id_train]

    def valid_log_likelihoods(self, theta):
        a, b, c, d, e, beta, sigma_a, sigma_b, sigma_c, sigma_d, sigma_e = self.theta2par(theta)
        logits = beta[0] + beta[1] * self.black[self.id_valid] + beta[2] * self.female[self.id_valid] + beta[4] * self.female[self.id_valid] * self.black[self.id_valid] + beta[3] * self.v_prev_full[self.id_valid] + a[self.age[self.id_valid]] + b[self.edu[self.id_valid]] + c[self.age_edu[self.id_valid]] + d[self.state[self.id_valid]] + e[self.region_full[self.id_valid]]
        return dist.Bernoulli(logits=logits).log_prob(self.y[self.id_valid])

    def valid_logits(self, theta):
        a, b, c, d, e, beta, sigma_a, sigma_b, sigma_c, sigma_d, sigma_e = self.theta2par(theta)
        logits = beta[0] + beta[1] * self.black[self.id_valid] + beta[2] * self.female[self.id_valid] + beta[4] * \
                 self.female[self.id_valid] * self.black[self.id_valid] + beta[3] * self.v_prev_full[self.id_valid] + a[
                     self.age[self.id_valid]] + b[self.edu[self.id_valid]] + c[self.age_edu[self.id_valid]] + d[
                     self.state[self.id_valid]] + e[self.region_full[self.id_valid]]
        return logits

    def valid_data(self):
        return self.y[self.id_valid]
    def test_log_likelihoods(self, theta):
        a, b, c, d, e, beta, sigma_a, sigma_b, sigma_c, sigma_d, sigma_e = self.theta2par(theta)
        logits = beta[0] + beta[1] * self.black[self.id_test] + beta[2] * self.female[self.id_test] + beta[4] * self.female[self.id_test] * self.black[self.id_test] + beta[3] * self.v_prev_full[self.id_test] + a[self.age[self.id_test]] + b[self.edu[self.id_test]] + c[self.age_edu[self.id_test]] + d[self.state[self.id_test]] + e[self.region_full[self.id_test]]
        return dist.Bernoulli(logits=logits).log_prob(self.y[self.id_test])

    def test_logits(self, theta):
        a, b, c, d, e, beta, sigma_a, sigma_b, sigma_c, sigma_d, sigma_e = self.theta2par(theta)
        logits = beta[0] + beta[1] * self.black[self.id_test] + beta[2] * self.female[self.id_test] + beta[4] * \
                 self.female[self.id_test] * self.black[self.id_test] + beta[3] * self.v_prev_full[self.id_test] + a[
                     self.age[self.id_test]] + b[self.edu[self.id_test]] + c[self.age_edu[self.id_test]] + d[
                     self.state[self.id_test]] + e[self.region_full[self.id_test]]
        return logits

    def test_data(self):
        return self.y[self.id_test]

if __name__ == '__main__':
    cls = Election()
    sd = random.PRNGKey(3)
    theta = random.normal(sd, (cls.n,))
    print(cls.log_prior(theta), jnp.mean(cls.log_likelihoods(theta)), jnp.mean(cls.valid_log_likelihoods(theta)), jnp.mean(cls.test_log_likelihoods(theta)))