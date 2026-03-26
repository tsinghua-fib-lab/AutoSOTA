from typing import Callable
from typing import Optional, Union, Tuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike


def vec(X: ArrayLike) -> Array:
    """
    Vectorization of a matrix.
    """
    return X.reshape(-1, order='F')


def unvec_triu(X: ArrayLike, size=Optional[int]) -> Array:
    """
    Invert the previous operation
    """
    if size is None:
        size = int(0.5 * (-1 + jnp.sqrt(1 + 8 * X.shape[0])) ** 0.5)
    out = jnp.zeros((size, size), dtype=X.dtype)
    out = out.at[jnp.triu_indices(size)].set(X)
    out = out + out.T - jnp.diag(jnp.diag(out))
    return out


def vec_triu(X: ArrayLike) -> Array:
    """
    Vectorization of the upper triangular part of a square matrix.
    """
    return X[jnp.triu_indices(X.shape[0])]


def unvec(vecX: ArrayLike, shape=Optional[Tuple[int, int]]) -> Array:
    """
    Invert the previous operation
    """
    if shape is None:
        shape = (int(vecX.shape[0] ** 0.5), int(vecX.shape[0] ** 0.5))
    return vecX.reshape(shape, order='F')


def SampleWishart(key: Array, scale: ArrayLike, df=Optional[int]) -> Array:
    """
    Generate a random symmetric positive definite matrix.
    From the Wishart distribution.
    """
    if len(scale.shape) == 1:
        scale = jnp.diag(scale)
    if df is None:
        df = scale.shape[0]
    dim = scale.shape[0]
    if scale.shape[0] != 1:
        A = jax.random.multivariate_normal(key, jnp.zeros(dim), scale, shape=(df,))
    else:
        A = jnp.sqrt(scale) * jax.random.normal(key, shape=(df,))
    return A.T @ A


def elbo_integrand(logpdf: Callable, tgt_log_density: Callable, x: Union[ArrayLike, float]) -> Union[Array, float]:
    """
    Integrand for the opposite of the ELBO
    """
    return logpdf(x) - tgt_log_density(x)


def gaussian_loss(OP_key: Array, theta: jnp.ndarray, gaussian, tgt_log_density: Callable, n_samples_for_loss=100,
                  vmap=True) -> jnp.ndarray:
    """
    Compute the ELBO for a Gaussian variational family.
    """
    mean, cov = gaussian.get_mean_cov(theta)
    n = cov.shape[0]
    if len(cov.shape) == 1:
        def logpdf_normal(x):
            L = cov ** 0.5
            y = (x - mean) / L
            return (-1 / 2 * jnp.einsum('...i,...i->...', y, y) - n / 2 * jnp.log(2 * jnp.pi)
                    - jnp.log(L).sum(-1))

        def sampling(key):
            return mean + cov ** 0.5 * jax.random.multivariate_normal(key, jnp.zeros(n), jnp.eye(n),
                                                                      (n_samples_for_loss,))

    else:
        def logpdf_normal(x):
            return jax.scipy.stats.multivariate_normal.logpdf(x, mean, cov)

        def sampling(key):
            return jax.random.multivariate_normal(key, mean, cov, (n_samples_for_loss,))

    def my_integrand(x):
        return elbo_integrand(logpdf_normal, tgt_log_density, x)

    samples = sampling(OP_key)
    if not vmap:
        integral = jnp.sum(my_integrand(samples)) / n_samples_for_loss
    else:
        integral = jnp.sum(jax.vmap(my_integrand)(samples)) / n_samples_for_loss
    return integral


def truncated_mf_gaussian_loss(OP_key: Array, theta: jnp.ndarray,
                               truncated_gaussian, tgt_log_density: Callable,
                               n_samples_for_loss=100):
    """
        Compute the ELBO for a truncated Gaussian variational family.
    """
    mean, cov = truncated_gaussian.get_mean_cov(theta)
    n = cov.shape[0]
    assert len(cov.shape) == 1

    @jax.vmap
    def logpdf_normal(x):
        L = cov ** 0.5
        y = (x - mean) / L
        return (-1 / 2 * jnp.einsum('...i,...i->...', y, y) - n / 2 * jnp.log(2 * jnp.pi)
                - jnp.log(L).sum(-1))

    def sampling(key):
        keys = jax.random.split(key, n_samples_for_loss)
        return jax.vmap(truncated_gaussian.sampling_method, in_axes=(None, 0))(theta, keys)

    def my_integrand(x):
        return elbo_integrand(logpdf_normal, tgt_log_density, x)

    samples = sampling(OP_key)
    integral = jnp.sum(my_integrand(samples)) / n_samples_for_loss
    return integral


def kl_gaussians(mu1: ArrayLike, cov1: ArrayLike, mu2: ArrayLike, cov2: ArrayLike) -> float:
    """
    Compute the KL divergence between two Gaussian distributions.
    """
    if len(cov1.shape) == 1:
        cov1 = jnp.diag(cov1)
    if len(cov2.shape) == 1:
        cov2 = jnp.diag(cov2)
    cov2_inv = jnp.linalg.inv(cov2)
    diff = mu2 - mu1
    return 0.5 * (jnp.trace(cov2_inv @ cov1) + diff.T @ cov2_inv @ diff - len(mu1) + jnp.log(
        jnp.linalg.det(cov2) / jnp.linalg.det(cov1)))


def get_residual(y: ArrayLike, X: ArrayLike, beta: ArrayLike) -> Array:
    return jnp.var(y - beta @ X.T)


def OLS(X: ArrayLike, y: ArrayLike) -> Array:
    """
    Ordinary Least Squares estimator
    """
    return jnp.linalg.lstsq(X, y)[0]


def OLS_pinv(X: ArrayLike, y: ArrayLike) -> Array:
    return jnp.linalg.pinv(X.T @ X) @ X.T @ y
