"""Sampling from likelihood"""

from typing import Optional
import numpy as np
import scipy as sp

from credal_dro import constants as _constants

# Fixed df used in the Student-t likelihood (override by adding STUDENT_T_DF to constants.py)
STUDENT_T_DF = float(getattr(_constants, "STUDENT_T_DF", 3.0))


def rvs_multivariate_t(
    *,
    df: float,
    loc: np.ndarray,
    shape: np.ndarray,
    size: int,
    generator: np.random.Generator,
) -> np.ndarray:
    """
    Multivariate Student-t sampler using the normal/chi-square mixture representation.

    Returns: array (size, dim)
    """
    loc = np.asarray(loc, dtype=float)
    shape = np.asarray(shape, dtype=float)
    dim = int(loc.shape[0])

    z = generator.multivariate_normal(mean=np.zeros(dim), cov=shape, size=size)
    z = np.asarray(z)
    if z.ndim == 1:
        z = z.reshape((1, dim))

    g = generator.chisquare(df, size=size) / df
    g = np.asarray(g)
    if g.ndim == 0:
        g = g.reshape((1,))

    return loc + z / np.sqrt(g)[:, None]


def sample_likelihood(
    likelihood: str,
    theta_sample: np.ndarray,
    dim: int,
    num_likelihood_samples: int,
    num_posterior_samples: int,
    generator: Optional[np.random.Generator],
    inference: str = "bayes",
) -> np.ndarray:
    """Sample from the likelihood

    Returns:
        ndarray of shape `(num_posterior_samples, num_likelihood_samples)`
    """
    if not generator:
        generator = np.random.default_rng()
    # num_posterior_samples = theta_sample.shape[0]
    xi = np.zeros([num_posterior_samples, num_likelihood_samples, dim])
    if likelihood == "exponential":
        for i in range(num_posterior_samples):
            xi[i] = sp.stats.expon.rvs(
                scale=1 / theta_sample[i],
                size=(num_likelihood_samples, dim),
                random_state=generator,
            )
    elif likelihood == "normal":
        for i in range(num_posterior_samples):
            # NOTE numpy normal takes standard deviation as scale parameter - not variance or precision!
            xi[i] = generator.normal(
                theta_sample[i, 0],
                theta_sample[i, 1],
                size=(num_likelihood_samples, dim),
            )
    elif likelihood == "multivariate_normal":
        for i in range(num_posterior_samples):
            mu = theta_sample[i,:dim]
            vec_cov = theta_sample[i,dim:]
            if inference == "bayes":
                cov = reconstruct_covariance_from_triu(vec_cov, dim)
            elif inference == "npl_mmd":
                cov = cholesky_param_to_covariance(dim, vec_cov)
            else:
                raise ValueError(f"Not a valid inference: {inference}")
            xi[i] = generator.multivariate_normal(mu, cov, size=num_likelihood_samples)
    elif likelihood == "multivariate_normal_known_cov":
        for i in range(num_posterior_samples):
            mu = theta_sample[i,:]
            cov = (5**2)*np.eye(dim)
            # vec_triu = theta_sample[i,dim:]
            # cov = reconstruct_covariance_from_triu(vec_triu, dim)
            xi[i] = generator.multivariate_normal(mu, cov, size=num_likelihood_samples).reshape((num_likelihood_samples,dim))
    elif likelihood == "multivariate_student_t":
        xi = np.zeros((num_posterior_samples, num_likelihood_samples, dim))
        for i in range(num_posterior_samples):
            mu_i = theta_sample[i, :dim]
            cov_i = reconstruct_covariance_from_triu(theta_sample[i, dim:], dim)
            xi[i, :, :] = rvs_multivariate_t(
                df=STUDENT_T_DF,
                loc=mu_i,
                shape=cov_i,
                size=num_likelihood_samples,
                generator=generator,
            )
    elif likelihood == "normal_known_var":
        for i in range(num_posterior_samples):
            xi[i] = generator.normal(
                    loc=theta_sample[i],
                    # scale=DGP_STD_TRUNCATED_NORMAL,
                    scale=5,
                    size=num_likelihood_samples,
                ).reshape((num_likelihood_samples,dim))

    elif likelihood == "uniform_0_theta":
        # x | θ ~ Uniform(0, θ), θ>0
        for i in range(num_posterior_samples):
            theta_i = float(np.squeeze(theta_sample[i]))
            if theta_i <= 0.0:
                raise ValueError(f"uniform_0_theta requires θ>0, got θ={theta_i}")
            xi[i] = generator.uniform(low=0.0, high=theta_i, size=(num_likelihood_samples, dim))

    else:
        raise NotImplementedError(
            f"Likelihood '{likelihood}' not implemented."
        )
    return xi

def reconstruct_covariance_from_triu(vec_triu: np.array, dim: int):
    """Reconstruct the covariance matrix from a upper triangular vector"""
    X = np.zeros((dim,dim))
    X[np.triu_indices(dim)] = vec_triu
    return X + X.T - np.diag(np.diag(X))

def cholesky_param_to_covariance(dim, L_flat):
    # Reshape the flat array into a lower triangular matrix
    L = np.zeros((dim, dim))
    tril_indices = np.tril_indices(dim)
    L[tril_indices] = L_flat

    # Diagonal elements of L should be strictly positive to ensure positive definiteness
    L[np.diag_indices(dim)] = np.exp(np.diag(L))

    # Return the covariance matrix
    return L @ L.T
