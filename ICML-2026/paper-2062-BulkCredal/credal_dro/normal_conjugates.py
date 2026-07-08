"""Closed-form expression for Bayesian conjugate models"""

from typing import Optional
import numpy as np
import scipy as sp


def default_prior_params(prior: str) -> tuple:
    """Get the default prior parameters"""
    prior_params = ()
    if prior == "gamma":
        alpha_prior, beta_prior = 1, 1
        prior_params = (alpha_prior, beta_prior)
    elif prior == "normal_gamma":
        mu_prior, kappa_prior, alpha_prior, beta_prior = 0.0, 1.0, 1.0, 1.0
        prior_params = (mu_prior, kappa_prior, alpha_prior, beta_prior)
    elif prior == "normal_known_var":
        mu_prior, var_prior = 10, 1
        prior_params = (mu_prior, var_prior)
    else:
        raise NotImplementedError(f"Prior '{prior}' not implemented.")
    return prior_params


def get_posterior_params(
    posterior: str, data: np.ndarray, prior_params: tuple
) -> tuple:
    """Given prior parameters, return the updated posterior parameters"""
    post_params = []
    if posterior == "gamma":
        alpha_prior, beta_prior = prior_params
        alpha_posterior = alpha_prior + data.shape[0]
        beta_posterior = beta_prior + np.sum(data)
        post_params = (alpha_posterior, beta_posterior)
    elif posterior == "normal_gamma":
        mu_prior, kappa_prior, alpha_prior, beta_prior = prior_params
        (
            mu_posterior,
            kappa_posterior,
            alpha_posterior,
            beta_posterior,
        ) = normal_gamma_posterior(data, mu_prior, kappa_prior, alpha_prior, beta_prior)
        post_params = (mu_posterior, kappa_posterior, alpha_posterior, beta_posterior)
    elif posterior == "normal_known_var":
        mu_prior, var_prior = prior_params
        known_var = 100
        N = data.shape[0]
        mu_posterior = known_var*mu_prior/(N*var_prior + known_var) + var_prior*np.sum(data)/(N*var_prior + known_var)
        var_posterior = var_prior*known_var/(N*var_prior + known_var)
        post_params = (mu_posterior, var_posterior)
    else:
        raise NotImplementedError(f"Posterior '{posterior}' is not implemented")
    return post_params


def derive_analytical_posterior_params(
    posterior: str, posterior_params: tuple
) -> np.ndarray:
    """When using our closed form expressions for 'Bayesian ambiguity sets',
    we derive a new distribution from the posterior parameters"""
    if posterior == "normal_gamma":
        theta = np.zeros((1, 2))
        mu_posterior, _, alpha_posterior, beta_posterior = posterior_params
        # we want an analytical form for the precision, which is alpha over beta
        theta[0] = np.array([mu_posterior, alpha_posterior / beta_posterior])
        return theta
    else:
        raise NotImplementedError(
            f"We haven't derived an analytical posterior expression for a '{posterior}' posterior"
        )


def sample_posterior(
    posterior: str,
    posterior_params: tuple,
    num_posterior_samples: int,
    generator: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Sample from the posterior"""
    if posterior == "gamma":
        alpha_post, beta_post = posterior_params
        return sp.stats.gamma.rvs(
            a=alpha_post,
            scale=1 / beta_post,
            size=num_posterior_samples,
            random_state=generator,
        )
    if posterior == "normal_gamma":
        (
            mu_posterior,
            kappa_posterior,
            alpha_posterior,
            beta_posterior,
        ) = posterior_params
        return normal_gamma_rvs(
            num_posterior_samples,
            mu_posterior,
            kappa_posterior,
            alpha_posterior,
            beta_posterior,
            generator=generator,
        )
    if posterior == "normal_known_var":
        mu_posterior, var_posterior = posterior_params
        return sp.stats.norm.rvs(loc=mu_posterior, scale=np.sqrt(var_posterior), size=num_posterior_samples, random_state=generator)
    else:
        raise NotImplementedError(f"Posterior '{posterior}' is not implemented")
    
def sample_pp(
    posterior: str,
    posterior_params: tuple,
    num_pp_samples: int,
    generator: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Sample from the posterior"""
   
    if posterior == "normal_known_var":
        mu_posterior, var_posterior = posterior_params
        known_var = 100
        return sp.stats.norm.rvs(loc=mu_posterior, scale=np.sqrt(var_posterior + known_var), size=num_pp_samples, random_state=generator)
    else:
        raise NotImplementedError(f"Posterior '{posterior}' is not implemented")
