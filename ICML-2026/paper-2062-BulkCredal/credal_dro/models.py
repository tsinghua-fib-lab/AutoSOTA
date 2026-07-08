"""Model classes compatible for use within the npl_mmd class"""

import jax
import jax.numpy as jnp
from bayesian_dro.Bayesian_DRO_continuous import DGP_STD_TRUNCATED_NORMAL
from .constants import upper_triangular_size, DGP_NORMAL_KNOWN_VARIANCE_STD

class ExponentialModel:
    def __init__(self, m):
        self.m = m  # number of points sampled from the model at each approximation of the MMD

    def sample(self, theta, key):
        lamb = jnp.exp(theta)  # Re-parametrisation to ensure lambda > 0!
        x = (
            jax.random.exponential(key, shape=(self.m, 1)) / lamb
        )  # Exponential with parameter lambda
        return x
    
    def init_params(self, data):
        # This function return the initialisation parameters for the minimisation of the mmd
        return jnp.log((1/jnp.mean(data)))*jnp.ones(1) # Initialisation of unknown parameter, here I inistialise at MLE
    
    def parametrise(self, theta):
        return jnp.exp(theta)  # rate parameter of expoenential model is re-parametrised to ensure positivity
        
        
    
class univariate_GaussianModel:
    def __init__(self, m):
        self.m = m
    
    def sample(self, theta, key):
        mu = theta[0]
        std = jnp.exp(theta[1]) # make sure standard deviation is positive!
        x = (
            mu + std*jax.random.normal(key, shape=(self.m,1))
        )
        
        return x
    
    def init_params(self, data):
        return jnp.array([jnp.mean(data), jnp.log(jnp.std(data))]).reshape((2,))
    
    def parametrise(self, theta): 
        theta = theta.at[1].set(jnp.exp(theta[1]))  # scale parameter is reparametrised to ensure postivity - now parametrise back
        return theta 
    
class univariate_GaussianModel_known_variance:
    def __init__(self, m):
        self.m = m
    
    def sample(self, theta, key):
        mu = theta[0]
        return mu + DGP_NORMAL_KNOWN_VARIANCE_STD*jax.random.normal(key, shape=(self.m,1))        
    
    def init_params(self, data):
        return jnp.mean(data).reshape((1,))
    
    def parametrise(self, theta):
        return theta

class multivariate_GaussianModel:
    def __init__(self, m, d, known_cov):
        self.m = m
        self.d = d
        self.known_cov = known_cov
    
    def sample(self, theta, key):
        if self.known_cov == True:
            mu = theta
            sigma = DGP_STD_TRUNCATED_NORMAL
            x = (
                jax.random.multivariate_normal(key, mean = mu, cov = (sigma**2)*jnp.eye(self.d), shape=(self.m,self.d))
            )
        else:
            mu = theta[:self.d]
            vec_triu = theta[self.d:]
            Sigma = self.cholesky_param_to_covariance(vec_triu)
            x = (
                jax.random.multivariate_normal(key, mean = mu, cov = Sigma, shape=(self.m,self.d))
            )
        return x
    
    def init_params(self, data):
        if self.known_cov == True:
            return  jnp.mean(data, axis=0).reshape((self.d,))
        else:
            mu_init = jnp.mean(data, axis=0).reshape((self.d,))
            cov_matrix = jnp.cov(data, rowvar=False)
            L = jnp.linalg.cholesky(cov_matrix)
            diag_idx = jnp.diag_indices(L.shape[0])
            diag_entries = L[diag_idx]
            L = L.at[diag_idx].set(jnp.log(diag_entries))
            vec_triu_init = L[jnp.tril_indices(L.shape[0])]
            triu_size = upper_triangular_size(self.d)
            theta_init = jnp.zeros(self.d + triu_size)
            theta_init = theta_init.at[:self.d].set(mu_init)
            theta_init = theta_init.at[self.d:].set(vec_triu_init)
            return theta_init
    
    def parametrise(self, theta):
        # FIXME needs to match what likelihood takes as argument!
        return theta
    
    def reconstruct_covariance_from_triu(self, vec_triu: jnp.array):
        """Reconstruct the covariance matrix from a upper triangular vector in JAX"""
        X = jnp.zeros((self.d, self.d))
        X = X.at[jnp.triu_indices(self.d)].set(vec_triu)
        return X + X.T - jnp.diag(jnp.diag(X))
    
    def cholesky_param_to_covariance(self, L_flat):
        """
        Converts a flattened lower triangular matrix to a covariance matrix.

        L_flat: The flattened lower triangular part of the matrix.
        dim: The dimensionality of the covariance matrix.
        """
        # Reshape the flat array into a lower triangular matrix
        L = jnp.zeros((self.d, self.d))
        tril_indices = jnp.tril_indices(self.d)
        L = L.at[tril_indices].set(L_flat)

        # Diagonal elements of L should be strictly positive to ensure positive definiteness
        L = L.at[jnp.diag_indices(self.d)].set(jnp.exp(jnp.diag(L)))

        # Return the covariance matrix
        return L @ L.T

        
    