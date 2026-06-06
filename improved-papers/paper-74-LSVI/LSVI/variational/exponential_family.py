from abc import ABC, abstractmethod
from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
import scipy
from scipy.stats import qmc

from variational.utils import vec, unvec


class ExponentialDistribution(ABC):
    r"""
    Abstract base class for an exponential-family distribution written in
        canonical form (using natural parameter).

    This class defines the minimal interface for probabilistic models of the form:
        p(x) \propto \exp(\eta^\top s(x)),
    where:
        - s(x) is the sufficient statistic.

    Subclasses must define:
        - `sufficient_statistic`
        - `sampling_method`
    where:
        sufficient_statistic should include by default an intercept term,
    and
        sampling_method should be a jax-compatible sampling method
    Optionally, the method `sanity` can be defined, as a way
    to check that the parameter indeed defines a proper density.
    By default, `sanity` uses a call to `sampling_method`, i.e.,
    if `smapling_method` works, then naturally the input parameter
    defines a density.
    """

    def __init__(self, dimension: int):
        self.dimension = dimension

    @staticmethod
    @abstractmethod
    def sufficient_statistic(x: jnp.ndarray):
        r"""
        Compute the sufficient statistic vector s(x).

        Parameters
        ----------
        x : jnp.ndarray
            The shape depends on the distribution.

        Returns
        -------
        jnp.ndarray
            The sufficient statistic vector. Must include an intercept / bias
            term by convention (e.g., a trailing `1`).
        """
        pass

    @abstractmethod
    def sampling_method(self, eta_or_theta: jnp.ndarray, key: jax.Array):
        pass

    def log_density(self, eta: jnp.ndarray, x: jnp.ndarray):
        return eta.T @ self.sufficient_statistic(x)

    def sanity(self, eta: jnp.ndarray):
        r"""
        Basic sanity check for natural parameters.

        The default implementation attempts to draw a sample from the
        distribution using the provided natural parameter and checks whether
        the result contains NaNs. If sampling fails (e.g., due to invalid
        parameters), the method returns `True`.

        Parameters
        ----------
        eta : jnp.ndarray
            Extended natural parameter vector.

        Returns
        -------
        bool
            True if the parameter appears invalid, False otherwise.

        Notes
        -----
        Subclasses should override this method when a more principled or
        efficient validity check exists (e.g., positive definiteness for Gaussians).
        """
        key = jax.random.PRNGKey(0)
        return jnp.isnan(self.sampling_method(eta.at[:-1].get(), key)).any()


class GenericNormalDistribution(ExponentialDistribution):
    def __init__(self, dimension: int):
        super().__init__(dimension=dimension)

    @staticmethod
    def sufficient_statistic(x: jnp.ndarray):
        r"""
        Compute the extended sufficient statistic:

            s(x) = ( x,
                     vec(xxᵀ),
                     1 )

        Parameters
        ----------
        x : jnp.ndarray
            A sample vector of shape (dimension,).

        Returns
        -------
        jnp.ndarray
            A concatenated vector containing:
            - the raw vector x,
            - the vectorized outer product xxᵀ,
            - a constant bias term 1.

        """
        return jnp.concatenate([x, vec(x[:, jnp.newaxis] @ x[:, jnp.newaxis].T), jnp.array([1.])])

    def sampling_method(self, theta: jnp.ndarray, key: jax.Array):
        mean, cov = self.get_mean_cov(theta)
        return jax.random.multivariate_normal(key, mean, cov)

    def get_mean_cov(self, theta: jnp.ndarray):
        """
        Given the natural parameter theta, returns the mean and the covariance matrix.
        """
        invcov = -2 * theta.at[self.dimension:].get()
        cov = jnp.linalg.pinv(unvec(invcov, (self.dimension, self.dimension)))
        mean = cov @ theta.at[:self.dimension].get()
        return mean, cov

    @staticmethod
    def get_theta(mean: jnp.ndarray, cov: jnp.ndarray):
        invcov = jnp.linalg.pinv(cov)
        theta = jnp.concatenate([invcov @ mean, -0.5 * vec(invcov)])
        return theta

    @staticmethod
    def get_eta(mean: jnp.ndarray, cov: jnp.ndarray):
        theta = GenericNormalDistribution.get_theta(mean, cov)
        eta = jnp.concatenate([theta, jnp.array([1.])])
        return eta

    def sanity(self, eta):
        """
        The parameter defines a proper Gaussian distribution if the supposed
        covariance matrix can be factorised via Cholesky.
        """
        _, cov = self.get_mean_cov(eta.at[:-1].get())
        return jnp.isnan(jnp.linalg.cholesky(cov)).any()


class GenericMeanFieldNormalDistribution(ExponentialDistribution):

    def __init__(self, dimension: int):
        super().__init__(dimension=dimension)

    @staticmethod
    def sufficient_statistic(x: jnp.ndarray):
        return jnp.concatenate([x, x ** 2, jnp.array([1.])])

    def sampling_method(self, theta: jnp.ndarray, key: jax.Array):
        mean, cov = self.get_mean_cov(theta)
        return mean + jnp.sqrt(cov) * jax.random.multivariate_normal(key, jnp.zeros(self.dimension),
                                                                     jnp.eye(self.dimension))

    def get_mean_cov(self, theta: jnp.ndarray):
        vec_diag_cov = 1. / (-2 * theta.at[self.dimension:].get())
        mean = vec_diag_cov * theta.at[:self.dimension].get()
        return mean, vec_diag_cov

    @staticmethod
    def get_theta(mean: jnp.ndarray, vec_diag_cov: jnp.ndarray):
        invcov = 1 / vec_diag_cov
        theta = jnp.concatenate([invcov * mean, -0.5 * invcov])
        return theta

    @staticmethod
    def get_eta(mean: jnp.ndarray, vec_diag_cov: jnp.ndarray):
        theta = GenericMeanFieldNormalDistribution.get_theta(mean, vec_diag_cov)
        eta = jnp.concatenate([theta, jnp.array([1.])])
        return eta

    def sanity(self, eta):
        mean, cov = self.get_mean_cov(eta.at[:-1].get())
        res = jnp.any(cov <= 0)
        return res


class GenericTruncatedMFNormalDistribution(ExponentialDistribution):
    def __init__(self, dimension: int, lower: jnp.ndarray, upper: jnp.ndarray):
        self.lower = lower
        self.upper = upper

        super().__init__(dimension=dimension)

    @staticmethod
    def sufficient_statistic(x: jnp.ndarray):
        return jnp.concatenate([x, x ** 2, jnp.array([1.])])

    def sampling_method(self, theta: jnp.ndarray, key: jax.Array):
        mean, cov = self.get_mean_cov(theta)
        sqcov = jnp.sqrt(cov)
        samples = jax.random.truncated_normal(key, lower=(self.lower - mean) / sqcov,
                                              upper=(self.upper - mean) / sqcov)
        samples = mean + sqcov * samples
        return samples

    def get_mean_cov(self, theta: jnp.ndarray):
        vec_diag_cov = 1. / (-2 * theta.at[self.dimension:].get())
        mean = vec_diag_cov * theta.at[:self.dimension].get()
        return mean, vec_diag_cov

    @staticmethod
    def get_theta(mean: jnp.ndarray, vec_diag_cov: jnp.ndarray):
        invcov = 1 / vec_diag_cov
        theta = jnp.concatenate([invcov * mean, -0.5 * invcov])
        return theta

    @staticmethod
    def get_eta(mean: jnp.ndarray, vec_diag_cov: jnp.ndarray):
        theta = GenericTruncatedMFNormalDistribution.get_theta(mean, vec_diag_cov)
        eta = jnp.concatenate([theta, jnp.array([1.])])
        return eta


class GenericWishartDistribution(ExponentialDistribution):
    def __init__(self, dimension: int):
        super().__init__(dimension=dimension)

    @staticmethod
    def sufficient_statistic(x: jnp.ndarray):
        return jnp.concatenate([vec(x), jnp.array([jnp.log(jnp.abs(jnp.linalg.det(x)))]), jnp.array([1.])])

    def sampling_method(self, theta: jnp.ndarray, key: jax.Array):
        degree, scale = self.get_degree_and_scale(theta)

        def body_fun(_, vals: Tuple[jnp.ndarray, jax.Array]):
            val, key = vals
            g = jax.random.multivariate_normal(key, jnp.zeros(self.dimension), scale).reshape((self.dimension, 1))
            _, key = jax.random.split(key)
            return val + g @ g.T, key

        S, _ = jax.lax.fori_loop(0, degree, body_fun, (jnp.zeros((self.dimension, self.dimension)), key))

        return S

    def get_degree_and_scale(self, theta: jnp.ndarray):
        inv_scale = - 2 * theta.at[:-1].get()
        scale = jnp.linalg.pinv(unvec(inv_scale))
        degree = 2 * theta.at[-1].get() + self.dimension + 1
        return jnp.array(degree, int), scale

    def get_theta(self, degree: int, scale):
        inv_scale = jnp.linalg.pinv(scale)
        theta = jnp.concatenate([-0.5 * vec(inv_scale), jnp.array([(degree - self.dimension - 1) / 2.])])
        return theta

    def get_eta(self, degree: int, scale):
        theta = self.get_theta(degree, scale)
        eta = jnp.concatenate([theta, jnp.array([1.])])
        return eta


class GenericBernoulliDistributionNumpy(ExponentialDistribution):
    """
    Numpy implementation of the BernoulliDistribution
    """

    def __init__(self, dimension: int):
        super().__init__(dimension=dimension)

    @staticmethod
    def sufficient_statistic(x: np.ndarray):
        return np.concatenate(
            [x, np.array([1.])])

    def sampling_method_numpy_qmc(self, theta: np.ndarray, n_samples: int):
        p = self.get_p(theta)
        samples = qmc.Sobol(d=self.dimension).random(n_samples)
        samples = samples.reshape((n_samples, self.dimension)) <= p
        return samples

    def sampling_method_numpy(self, theta: np.ndarray, n_samples: int, eps=0.):
        p = self.get_p(theta, eps)
        samples = np.random.uniform(0, 1, self.dimension * n_samples).reshape((n_samples, self.dimension)) <= p
        return samples

    @staticmethod
    def sufficient_statistic_numpy(x: np.ndarray):
        return np.concatenate([x, np.ones(x.shape[0])[:, np.newaxis]], axis=-1)

    @staticmethod
    def get_p(theta: np.ndarray, eps=0.):
        """
        eps should be taken as 1/N_samples
        """
        p = scipy.special.expit(theta)
        p[p <= eps] = eps
        p[p >= 1 - eps] = 1 - eps
        return p

    @staticmethod
    def get_theta(p: float):
        return scipy.special.logit(p)

    def get_eta(self, p: float):
        theta = self.get_theta(p)
        eta = np.concatenate([theta, np.array([1.])])
        return eta

    sufficient_statistic = sufficient_statistic_numpy
    sampling_method = sampling_method_numpy


class ExponentialDistributionFixedTheta:
    """
    Similar to the previous class, but fixed theta.
    """

    def __init__(self, theta: jnp.ndarray, dimension: int):
        self.theta = theta
        if dimension:
            self.dimension = dimension
        self.eta = jnp.concatenate([theta, jnp.array([1.])])

    @staticmethod
    @abstractmethod
    def sufficient_statistic(x: jnp.ndarray):
        pass

    @abstractmethod
    def sampling_method(self, key: jax.Array):
        pass

    def log_density(self, x: jnp.ndarray):
        return self.eta.T @ self.sufficient_statistic(x)


class NormalDistribution(ExponentialDistributionFixedTheta):
    def __init__(self, mean: jnp.ndarray, cov: jnp.ndarray):
        self.mean = mean
        self.cov = cov
        dimension = mean.shape[0]
        invcov = jnp.linalg.pinv(cov)
        theta = jnp.concatenate([invcov @ mean, -0.5 * vec(invcov)])

        super().__init__(theta, dimension)

    @staticmethod
    def sufficient_statistic(x: jnp.ndarray):
        return jnp.concatenate(
            [x, vec(x[:, jnp.newaxis] @ x[:, jnp.newaxis].T), jnp.array([1.])])

    @partial(jax.vmap, in_axes=(None, 0))
    def sampling_method(self, key: jax.Array):
        return jax.random.multivariate_normal(key, self.mean, self.cov)


class MeanFieldNormalDistribution(ExponentialDistributionFixedTheta):
    def __init__(self, mean: jnp.ndarray, vec_diag_cov: jnp.ndarray):
        cov = jnp.diag(vec_diag_cov)
        self.mean = mean
        self.cov = cov
        dimension = mean.shape[0]
        invcov = 1 / vec_diag_cov
        theta = jnp.concatenate([invcov * mean, -0.5 * invcov])

        super().__init__(theta=theta,
                         dimension=dimension)

    @staticmethod
    def sufficient_statistic(x: jnp.ndarray):
        return jnp.concatenate([x, x ** 2, jnp.array([1.])])

    @partial(jax.vmap, in_axes=(None, 0))
    def sampling_method(self, key: jax.Array):
        return jax.random.multivariate_normal(key, self.mean, self.cov)


class WishartDistribution(ExponentialDistributionFixedTheta):
    def __init__(self, degree: int, scale):
        self.degree = degree
        self.scale = scale
        dimension = scale.shape[0]
        inv_scale = jnp.linalg.pinv(scale)
        theta = jnp.concatenate([-0.5 * vec(inv_scale), jnp.array([(degree - dimension - 1) / 2.])])

        super().__init__(theta=theta, dimension=dimension)

    @staticmethod
    def sufficient_statistic(x: jnp.ndarray):
        return jnp.concatenate([vec(x), jnp.array([jnp.log(jnp.abs(jnp.linalg.det(x)))]), jnp.array([1.])])

    @partial(jax.vmap, in_axes=(None, 0))
    def sampling_method(self, key: jax.Array):
        G = jax.random.multivariate_normal(key, jnp.zeros(self.dimension), self.scale, shape=(self.degree,))
        S = G.T @ G
        return S


class BernoulliDistribution(ExponentialDistributionFixedTheta):
    def __init__(self, p: jnp.ndarray):
        self.p = p
        dimension = p.shape[0]
        theta = self.get_theta(p)

        super().__init__(theta=theta, dimension=dimension)

    @staticmethod
    def sufficient_statistic(x: jnp.ndarray):
        return jnp.concatenate(
            [x, jnp.array([1.])])

    @partial(jax.vmap, in_axes=(None, 0))
    def sampling_method(self, key: jax.Array):
        keys = jax.random.split(key, self.dimension)
        return jax.vmap(lambda key, p: jax.random.bernoulli(key, p))(keys, self.p).astype(jnp.float64)

    @staticmethod
    def get_p(theta: jnp.ndarray):
        return jax.scipy.special.expit(theta)

    @staticmethod
    def get_theta(p: jnp.ndarray):
        return jax.scipy.special.logit(p)
