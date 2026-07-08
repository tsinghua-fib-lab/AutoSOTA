"""Closed-form expression for Bayesian conjugate models"""

from typing import Optional
import numpy as np
import scipy as sp
from bayesian_dro.Bayesian_DRO_continuous import DGP_STD_TRUNCATED_NORMAL
from .constants import upper_triangular_size, DGP_NORMAL_KNOWN_VARIANCE_STD, PRIOR_SIGMA_SCALE
from .likelihood import reconstruct_covariance_from_triu
from . import constants as _constants
from scipy.stats import invwishart

def default_prior_params(prior: str, dim: int = 1) -> tuple:
    """Get the default prior parameters"""
    prior_params = ()
    if prior == "gamma":
        alpha_prior, beta_prior = 1, 1
        prior_params = (alpha_prior, beta_prior)
    elif prior == "normal_gamma":
        mu_prior, kappa_prior, alpha_prior, beta_prior = 0.0, 1.0, 1.0, 1.0
        prior_params = (mu_prior, kappa_prior, alpha_prior, beta_prior)
    elif prior == "normal_inverse_wishart":
        prior_params = normal_inverse_wishart_prior(dim)
    elif prior == "multivariate_normal_known_cov":
        prior_params = (np.zeros(dim), dim+2)
    elif prior == "normal_known_var":
        mu_prior, std_prior = 0.0, DGP_NORMAL_KNOWN_VARIANCE_STD
        prior_params = (mu_prior, std_prior)
    elif prior == "pareto":
        # Conjugate prior for Uniform(0, θ) likelihood (non-regular model).
        # θ ~ Pareto(m0, alpha0) with support θ >= m0.
        if dim != 1:
            raise ValueError(f"Pareto prior is implemented for dim=1, got dim={dim}")
        m0 = 1.0 
        alpha0 = 2.0
        prior_params = (m0, alpha0)
    elif prior == "student_t_niw":
        prior_params = student_t_niw_prior(dim)

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
    elif posterior == "normal_inverse_wishart":
        post_params = normal_inverse_wishart_posterior(data, *prior_params)
    elif posterior == "multivariate_normal_known_cov":
        mu_prior, kappa_prior = prior_params
        N = data.shape[0]
        xi_mean = np.mean(data, axis=0)
        kappa_posterior = kappa_prior + N
        mu_posterior = (kappa_prior * mu_prior + N * xi_mean) / kappa_posterior
        post_params = (mu_posterior, kappa_posterior)
    elif posterior == "normal_known_var":
        mu_prior, std_prior = prior_params
        data_mean = np.mean(data, axis=0)
        N = data.shape[0]
        true_var = DGP_NORMAL_KNOWN_VARIANCE_STD**2
        mu_posterior = (true_var * mu_prior + N * std_prior**2 * data_mean) / (N * std_prior**2 + true_var)
        std_posterior = np.sqrt(std_prior**2*true_var/(true_var+N*std_prior**2))
        post_params = (mu_posterior, std_posterior)
    elif posterior == "pareto":
        # Likelihood: x_i | θ ~ Uniform(0, θ), θ>0
        # Prior:      θ ~ Pareto(m0, alpha0)
        # Posterior:  θ | x ~ Pareto(m_n, alpha0+n) with m_n = max(m0, max_i x_i)
        m0, alpha0 = prior_params
        n = int(data.shape[0])
        x_max = float(np.max(data)) if n > 0 else -np.inf
        m_n = float(max(m0, x_max))
        alpha_n = float(alpha0 + n)
        post_params = (m_n, alpha_n)
    elif posterior == "student_t_niw":
        post_params = student_t_niw_posterior(data, *prior_params)

    else:
        raise NotImplementedError(f"Posterior '{posterior}' is not implemented")
    return post_params


def derive_analytical_posterior_params(
    posterior: str, posterior_params: tuple
) -> np.ndarray:
    """When using our closed form expressions for 'Bayesian ambiguity sets',
    we derive a new distribution from the posterior parameters"""
    if posterior == "gamma":
        alpha_posterior, beta_posterior = posterior_params
        return np.array([alpha_posterior / beta_posterior])
    elif posterior == "normal_gamma":
        theta = np.zeros((1, 2))
        mu_posterior, _, alpha_posterior, beta_posterior = posterior_params
        # we want an analytical form for the precision, which is alpha over beta
        # but the numpy normal dist function takes the standard deviation
        # so we take square root and inverse
        std = np.sqrt(beta_posterior / alpha_posterior)
        theta[0] = np.array([mu_posterior, std])
        return theta
    elif posterior == "normal_inverse_wishart":
        mu_post, kappa_post, _, Psi_post = posterior_params
        dim = mu_post.shape[0]
        vec_triu_size = upper_triangular_size(dim)
        theta = np.zeros((1, dim+int(vec_triu_size)))
        theta[0, :dim] = mu_post
        theta[0, dim:] = (1/(kappa_post - dim - 2)) * Psi_post[np.triu_indices(dim)]
        return theta
    elif posterior == "multivariate_normal_known_cov":
        mu_posterior, _ = posterior_params
        dim = mu_posterior.shape[0]
        return mu_posterior.reshape((1,dim))
    elif posterior == "normal_known_var":
        mu_posterior, _ = posterior_params
        dim = 1
        return mu_posterior.reshape((1,dim))
    elif posterior == "pareto":
        raise NotImplementedError(
            "Analytical posterior embedding for KL-DRO-BAS is not implemented for the non-regular Uniform(0,θ)–Pareto model."
        )

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
    if posterior == "normal_inverse_wishart":
        return normal_inverse_wishart_samples(num_posterior_samples, *posterior_params, generator=generator)
    if posterior == "inverse_wishart":
        iota_post, Psi_post = posterior_params
        dim = Psi_post.shape[0]
        iw_post = sp.stats.invwishart(iota_post, Psi_post, seed=generator)
        cov_samples = iw_post.rvs(num_posterior_samples)
        idx_triu = np.triu_indices(dim)
        triu_samples = np.zeros((int(num_posterior_samples), upper_triangular_size(dim)))
        for i, cov in enumerate(cov_samples):
            triu_samples[i] = cov[idx_triu]
        return triu_samples
    if posterior == "multivariate_normal_known_cov":
        mu_posterior, kappa_posterior = posterior_params
        dim = mu_posterior.shape[0]
        return sp.stats.multivariate_normal.rvs(mean=mu_posterior, cov=(1/kappa_posterior)*(DGP_NORMAL_KNOWN_VARIANCE_STD**2)*np.eye(dim), size=num_posterior_samples, random_state=generator)
    if posterior == "normal_known_var":
        mu_posterior, std_posterior = posterior_params
        return sp.stats.norm.rvs(loc=mu_posterior, scale=std_posterior, size=num_posterior_samples, random_state=generator)
    if posterior == "pareto":
        m_post, alpha_post = posterior_params
        # SciPy Pareto Type I: pdf = b * scale^b / x^(b+1), x >= scale
        return sp.stats.pareto.rvs(b=alpha_post, scale=m_post, size=num_posterior_samples, random_state=generator)
    if posterior == "student_t_niw":
        return student_t_niw_samples(num_posterior_samples, posterior_params, generator=generator)

    raise NotImplementedError(f"Posterior '{posterior}' is not implemented")

def get_log_partition_constant(posterior: str, posterior_params: list) -> float:
    """Given the posterior params, return the optimization constant for Bayesian DRO"""
    if posterior == "gamma":
        alpha_posterior, _ = posterior_params
        return np.log(alpha_posterior) - sp.special.digamma(alpha_posterior)
    elif posterior == "normal_gamma":
        _, kappa_posterior, alpha_posterior, _ = posterior_params
        return get_normal_gamma_constant(alpha_posterior, kappa_posterior)
    elif posterior == "normal_inverse_wishart":
        mu_post, kappa_post, _, _ = posterior_params
        dim = mu_post.shape[0]
        return get_normal_inverse_wishart_G_constant(dim, kappa_post)
    elif posterior == "multivariate_normal_known_cov":
        mu_posterior, kappa_posterior = posterior_params
        dim = mu_posterior.shape[0]
        return dim/(2*kappa_posterior)
    elif posterior == "normal_known_var":
        _, std_posterior = posterior_params
        return (std_posterior**2)/(2*DGP_NORMAL_KNOWN_VARIANCE_STD**2)
    elif posterior == "pareto":
        raise NotImplementedError(
            "get_log_partition_constant is not available for the non-regular Uniform(0,θ)–Pareto model. "
            "Use sampling-based methods (kl_pp / kl_bdro) instead of kl_dro_bas."
        )

    else:
        raise NotImplementedError(f"get_log_partition_constant not implemented for posterior {posterior}")


def get_normal_gamma_constant(alpha: int, kappa: float) -> float:
    """Get the constant for normal-gamma DRO"""
    return 0.5 * (1 / kappa + np.log(alpha) - sp.special.digamma(alpha))


def normal_gamma_posterior(
    data: np.ndarray,
    mu_prior: float,
    kappa_prior: float,
    alpha_prior: float,
    beta_prior: float,
) -> tuple[float, float, float, float]:
    """Get the closed-form expression for normal-gamma posterior

    Returns:
        Posterior parameters for alpha, beta, mu, kappa
    """
    num_observations = data.shape[0]
    data_mean = np.mean(data)
    mu_posterior = (kappa_prior * mu_prior + num_observations * data_mean) / (
        num_observations * kappa_prior
    )
    kappa_posterior = kappa_prior + num_observations
    alpha_posterior = alpha_prior + 0.5 * num_observations
    beta_posterior = (
        beta_prior
        + 0.5 * np.sum(np.square(data - data_mean))
        + (0.5 * num_observations * kappa_prior * np.square(data_mean - mu_prior))
        / (kappa_prior * num_observations)
    )
    return mu_posterior, kappa_posterior, alpha_posterior, beta_posterior


def normal_gamma_rvs(
    num_samples: int,
    mu: float,
    kappa: float,
    alpha: float,
    beta: float,
    generator: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Normal-gamma random variates

    Returns:
        mu_samples:
        standard_dev_samples:

    Notes:
        1. Sample precision `lambda` from a Gamma(`alpha`, `beta`) distribution
        2. Sample from a normal distribution with mean `mu` and standard deviation `sqrt(1/(kappa * lambda))`
    """
    samples = np.zeros((num_samples, 2))
    if not generator:
        generator = np.random.default_rng()
    precision_samples = generator.gamma(alpha, 1.0 / beta, num_samples)

    for i in range(num_samples):
        # NOTE numpy Normal distribution takes standard deviation as a parameter (hence sqrt) - not variance!
        samples[i, 0] = generator.normal(
            mu, np.sqrt(1.0 / (kappa * precision_samples[i])), 1
        )
    # NOTE we don't use kappa for the standard deviation samples: kappa only used for sampling mean above
    # see https://en.wikipedia.org/wiki/Normal-gamma_distribution#Generating_normal-gamma_random_variates
    # these are standard deviation samples! By taking inverse sqrt
    standard_devs = np.sqrt(1.0 / (precision_samples))
    samples[:, 1] = standard_devs
    return samples

def normal_inverse_wishart_prior(dim: int):
    """Default normal-inverse-Wishart prior hyperparameters for D dimensions
    
    Args:
        D: dimension

    Returns:
        mu: Prior mean. Vector with shape (D,)
        kappa: Reflects belief in prior mean (positive scalar)
        iota: Reflects belief in prior over covariance (positive scalar)
        Psi: matrix proportional to prior over covariance. Matrix with shape (D,D).
    """
    # since we derived our result via the exponential family, we set kappa = iota + D + 2
    iota = float(dim + 1)   # degrees of freedom must be greater than dim
    kappa = iota + dim + 2
    return np.zeros(dim), kappa, iota, np.identity(dim)*(PRIOR_SIGMA_SCALE**2)

def normal_inverse_wishart_posterior(data, mu_prior, kappa_prior, iota_prior, Psi_prior):
    """Normal-inverse-Wishart posterior hyperparameters for D dimensions

    Args:
        data: Observations with shape (N, D)
        mu: Prior mean. Vector with shape (D,)
        kappa: Reflects belief in prior mean (positive scalar)
        iota: Reflects belief in prior over covariance (positive scalar)
        Psi: matrix for prior over covariance. Matrix with shape (D,D).

    Returns:
        mu: Updated posterior mean. Vector with shape (D,)
        kappa: Updated belief in posterior mean (positive scalar)
        iota: Updated belief in posterior over covariance (positive scalar)
        Psi: Updated matrix for posterior over covariance. Matrix with shape (D,D).

    """
    N = data.shape[0]
    xi_mean = np.mean(data, axis=0)
    kappa_post = kappa_prior + N
    mu_post = (kappa_prior * mu_prior + N * xi_mean) / kappa_post
    iota_post = iota_prior + N
    Psi_post = Psi_prior + data.T @ data + kappa_prior * np.outer(mu_prior, mu_prior) - kappa_post * np.outer(mu_post, mu_post)
    return mu_post, kappa_post, iota_post, Psi_post

def normal_inverse_wishart_samples(num_samples: int, mu: np.ndarray, kappa: float, iota: float, Psi: np.ndarray, generator: Optional[np.random.Generator] = None) -> tuple[np.ndarray, np.ndarray]:
    """Get N samples of mean and covariance from the normal-inverse-Wishart distribution.

    Args:
        num_samples: N for short
        mu: Mean hyperparameter. Vector with shape (D,)
        kappa: Positive scalar
        iota: Positive scalar
        Psi: Matrix with shape (D,D)
        generator: Numpy random number generator

    Returns:
        mu_samples: Matrix with shape (N,D)
        cov_samples: Matrix with shape (N, D + D*(D-1)/2 + D).
            The covariance samples are stored as a vector in upper triangular format.
    """
    dim = int(mu.shape[0])
    assert dim == Psi.shape[0] and dim == Psi.shape[1]
    vec_triu_size = int(upper_triangular_size(dim))
    samples = np.zeros((num_samples, dim+vec_triu_size))

    # sample from the inverse Wishart
    iw_post = sp.stats.invwishart(iota, Psi, seed=generator)
    cov_samples = iw_post.rvs(num_samples)

    # sample from a multivariate normal given the covariance samples
    idx_triu = np.triu_indices(dim)
    for i, cov in enumerate(cov_samples):
        samples[i, :dim] = sp.stats.multivariate_normal(mu, 1/kappa * cov, seed=generator).rvs()
        samples[i, dim:] = cov[idx_triu]
    return samples

def multivariate_digamma(a: float, p: int) -> float:
    """The multivariate digamma function of dimension p

    Notes:
        [1] https://search.r-project.org/CRAN/refmans/CholWishart/html/mvdigamma.html
        [2] https://en.wikipedia.org/wiki/Multivariate_gamma_function#Derivatives
    """
    return np.sum([sp.special.digamma(a + (1-i)/2) for i in range(p)])

def get_normal_inverse_wishart_G_constant(dim: int, kappa_post: float) -> float:
    """Returns the constant G(tau, nu) for the normal-inverse-Wishart"""
    term1 = -0.5 * dim * np.log(2)
    term2 = - 0.5 * multivariate_digamma(0.5 * (kappa_post - dim - 2), dim)
    term3 = 0.5 * (dim / kappa_post)
    term4 = 0.5 * dim * np.log(kappa_post - dim - 2)
    return term1 + term2 + term3 + term4

def posterior_predictive_params(posterior: str, posterior_params: tuple) -> np.array:
    """Get the posterior predictive params"""
    if posterior == "normal_gamma":
        mu_posterior, kappa_posterior, alpha_posterior, beta_posterior = posterior_params

        # the scale is rougly equalivalent to the standard deviation
        # in particular, the case where dof = infinity is a Gaussian
        scale = np.sqrt((beta_posterior * (kappa_posterior + 1)) / (alpha_posterior * kappa_posterior))

        dof = 2 * alpha_posterior   # degrees of freedom for student t
        return np.array([[mu_posterior, scale, dof]])

    if posterior == "normal_inverse_wishart":
        mu_posterior, kappa_posterior, nu_posterior, Psi_posterior = posterior_params
        dim = mu_posterior.shape[0]
        vec_triu_size = int(upper_triangular_size(dim))
        df = nu_posterior - dim + 1     # degrees of freedom

        # store the posterior params in a vectorized vector
        pp_params = np.zeros((1, dim + vec_triu_size + 1))
        idx_triu = np.triu_indices(dim)
        shape = (kappa_posterior + 1)/(kappa_posterior*df) * Psi_posterior
        pp_params[0,:dim] = mu_posterior
        pp_params[0,dim:dim + vec_triu_size] = shape[idx_triu]
        pp_params[0,dim + vec_triu_size] = df
        return pp_params
    
    if posterior == "gamma":
        alpha_posterior, beta_posterior = posterior_params
        shape = alpha_posterior
        scale = beta_posterior
        return np.array([[shape, scale]])
    
    if posterior == "pareto":
        m_post, alpha_post = posterior_params
        return np.array([[float(m_post), float(alpha_post)]], dtype=float)

    if posterior == "student_t_niw":
        # Sampling-based posterior predictive: we pass through the (mutable) posterior dict.
        return posterior_params

    if posterior == "normal_known_var":
        mu_posterior, std_posterior = posterior_params
        return np.array([mu_posterior[0], std_posterior])
    
    if posterior == "multivariate_normal_known_cov":
        mu_posterior, kappa_posterior = posterior_params
        dim = mu_posterior.shape[0]
        pp_params = np.zeros(dim + 1)
        pp_params[:dim] = mu_posterior
        pp_params[-1] = kappa_posterior
        return pp_params


    raise NotImplementedError(f"Posterior predictive not implemented for posterior '{posterior}'.")

def sample_posterior_predictive(
    likelihood: str,
    posterior: str,
    theta_sample: np.ndarray,
    dim: int,
    num_likelihood_samples: int,
    generator: Optional[np.random.Generator] = None,
) -> np.array:
    """Sample from the posterior predictive"""
    if likelihood == "normal" and posterior == "normal_gamma":
        mu, scale, df = theta_sample[0]
        return sp.stats.t.rvs(df, loc=mu, scale=scale, size=(num_likelihood_samples, dim), random_state=generator)
    if likelihood == "multivariate_normal" and posterior == "normal_inverse_wishart":
        pp_params = theta_sample[0]
        loc = pp_params[:dim]
        vec_triu_size = int(upper_triangular_size(dim))
        vec_shape = pp_params[dim: dim + vec_triu_size]
        df = pp_params[dim + vec_triu_size]
        shape = reconstruct_covariance_from_triu(vec_shape, dim)
        samples = sp.stats.multivariate_t.rvs(loc=loc, shape=shape, df=df, size=(1, num_likelihood_samples), random_state=generator)
        return samples
    if likelihood == "exponential" and posterior == "gamma":
        shape, scale = theta_sample[0]
        return sp.stats.lomax.rvs(c=shape, scale=scale, size=(num_likelihood_samples, dim), random_state=generator)
    if likelihood == "normal_known_var" and posterior == "normal_known_var":
        mu, scale = theta_sample
        return sp.stats.norm.rvs(loc=mu, scale=np.sqrt(scale**2+DGP_NORMAL_KNOWN_VARIANCE_STD**2), size=(num_likelihood_samples, dim), random_state=generator)
    if  likelihood == "multivariate_normal_known_cov" and posterior == "multivariate_normal_known_cov":
        pp_params = theta_sample
        dim = pp_params.shape[0] - 1
        mu = pp_params[:dim]
        kappa = pp_params[-1]
        var = (1/kappa)*(DGP_NORMAL_KNOWN_VARIANCE_STD**2) +  DGP_NORMAL_KNOWN_VARIANCE_STD**2  
        return sp.stats.multivariate_normal.rvs(mean=mu, cov=var*np.eye(dim), size=num_likelihood_samples, random_state=generator)
    if likelihood == "uniform_0_theta" and posterior == "pareto":
        if dim != 1:
            raise ValueError(f"uniform_0_theta posterior predictive is implemented for dim=1, got dim={dim}")
        if generator is None:
            generator = np.random.default_rng()

        # theta_sample can arrive as (1,2), (2,), or a tuple/list
        if isinstance(theta_sample, (tuple, list)) and len(theta_sample) == 2:
            m_post, alpha_post = float(theta_sample[0]), float(theta_sample[1])
        else:
            arr = np.asarray(theta_sample, dtype=float).reshape(-1)
            if arr.size < 2:
                raise ValueError(
                    f"Expected theta_sample to contain (m_post, alpha_post), got shape {np.asarray(theta_sample).shape}"
                )
            m_post, alpha_post = float(arr[0]), float(arr[1])

        # Draw θ ~ Pareto(m_post, alpha_post), then x | θ ~ Uniform(0, θ)
        theta_draws = sp.stats.pareto.rvs(
            b=alpha_post, scale=m_post, size=num_likelihood_samples, random_state=generator
        )
        u = generator.uniform(low=0.0, high=1.0, size=(num_likelihood_samples, 1))
        return u * theta_draws.reshape(-1, 1)
    if posterior == "student_t_niw" and likelihood == "multivariate_student_t":
        return student_t_niw_posterior_predictive_samples(
            theta_posterior=theta_sample,
            dim=dim,
            num_samples=num_likelihood_samples,
            generator=generator,
        )


    raise NotImplementedError()

# ---------------------------------------------------------------------------
# Student-t likelihood with NIW prior (sampling-based posterior / predictive)
# ---------------------------------------------------------------------------

def student_t_niw_prior(dim: int):
    """Default NIW prior hyperparameters for a multivariate Student-t likelihood.

    Prior:
        Σ ~ InvWishart(iota0, Psi0)
        μ | Σ ~ Normal(mu0, Σ/kappa0)

    Likelihood:
        x | μ, Σ ~ MultivariateStudentT(df=t_df, loc=μ, scale=Σ)
    """
    t_df = float(getattr(_constants, "STUDENT_T_DF", 3.0))

    # Weak but proper defaults (adjust if you want stronger shrinkage)
    mu0 = 30.0 * np.ones(dim)
    kappa0 = 0.1
    iota0 = float(dim + 2)  # > dim + 1 gives finite mean for Σ
    Psi0 = (10.0**2) * np.eye(dim)

    return mu0, kappa0, iota0, Psi0, t_df


def student_t_niw_posterior(
    data: np.ndarray,
    mu0: np.ndarray,
    kappa0: float,
    iota0: float,
    Psi0: np.ndarray,
    t_df: float,
):
    """Return a *sampling-based* posterior representation (data + priors + cached Gibbs state)."""
    data = np.asarray(data, dtype=float)
    return {
        "data": data,
        "mu0": np.asarray(mu0, dtype=float),
        "kappa0": float(kappa0),
        "iota0": float(iota0),
        "Psi0": np.asarray(Psi0, dtype=float),
        "t_df": float(t_df),
        "_gibbs_state": None,        # cached chain state (optional warm-start)
        "_gibbs_burned": False,      # whether burn-in has been run for the cached state
    }


def _safe_cholesky(a: np.ndarray, jitter: float, max_tries: int = 6) -> np.ndarray:
    """Cholesky with exponentially increasing jitter (SPD rescue)."""
    a = 0.5 * (a + a.T)
    d = a.shape[0]
    for k in range(max_tries):
        try:
            return np.linalg.cholesky(a + (jitter * (10.0**k)) * np.eye(d))
        except np.linalg.LinAlgError:
            continue
    # last attempt
    return np.linalg.cholesky(a + (jitter * (10.0**max_tries)) * np.eye(d))


def _student_t_niw_init_state(data: np.ndarray, jitter: float) -> dict:
    n, dim = data.shape
    mu = data.mean(axis=0)

    if n >= 2:
        Sigma = np.cov(data, rowvar=False)
        Sigma = np.asarray(Sigma, dtype=float)
        if Sigma.ndim == 0:
            Sigma = np.array([[float(Sigma)]], dtype=float)
    else:
        Sigma = np.eye(dim, dtype=float)

    Sigma = 0.5 * (Sigma + Sigma.T) + jitter * np.eye(dim)
    omega = np.ones(n, dtype=float)

    return {"mu": mu, "Sigma": Sigma, "omega": omega}


def _rvs_multivariate_t(
    *,
    df: float,
    loc: np.ndarray,
    shape: np.ndarray,
    size: int,
    generator: np.random.Generator,
) -> np.ndarray:
    """Local multivariate Student-t sampler (normal/chi-square mixture)."""
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


def _student_t_niw_gibbs_step(
    *,
    state: dict,
    data: np.ndarray,
    mu0: np.ndarray,
    kappa0: float,
    iota0: float,
    Psi0: np.ndarray,
    t_df: float,
    generator: np.random.Generator,
    jitter: float,
) -> dict:
    """One Gibbs sweep for the Student-t likelihood via latent gamma weights ω_i."""
    mu = state["mu"]
    Sigma = state["Sigma"]

    n, dim = data.shape

    # ---- ω | μ, Σ, x  (Gamma conditionals)
    L = _safe_cholesky(Sigma, jitter=jitter)
    centred = data - mu
    sol = np.linalg.solve(L, centred.T)  # (dim, n)
    delta = np.sum(sol**2, axis=0)       # (n,)

    shape = 0.5 * (t_df + dim)
    rate = 0.5 * (t_df + delta)
    omega = generator.gamma(shape, 1.0 / rate)  # numpy gamma uses scale=1/rate

    # ---- NIW update with weighted sufficient statistics
    W = float(omega.sum())
    xbar = (omega[:, None] * data).sum(axis=0) / W

    xc = data - xbar
    Sw = xc.T @ (omega[:, None] * xc)

    kappa_n = kappa0 + W
    mu_n = (kappa0 * mu0 + W * xbar) / kappa_n
    iota_n = iota0 + n

    diff = (xbar - mu0).reshape(-1, 1)
    Psi_n = Psi0 + Sw + (kappa0 * W / kappa_n) * (diff @ diff.T)
    Psi_n = 0.5 * (Psi_n + Psi_n.T) + jitter * np.eye(dim)

    Sigma_new = invwishart.rvs(df=iota_n, scale=Psi_n, random_state=generator)
    Sigma_new = 0.5 * (Sigma_new + Sigma_new.T) + jitter * np.eye(dim)

    mu_new = generator.multivariate_normal(mu_n, Sigma_new / kappa_n)

    return {"mu": mu_new, "Sigma": Sigma_new, "omega": omega}


def _pack_mu_sigma_triu(mu: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    dim = int(mu.shape[0])
    triu = Sigma[np.triu_indices(dim)]
    return np.concatenate([mu, triu])


def student_t_niw_samples(
    num_samples: int,
    theta_posterior: dict,
    *,
    generator: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Draw θ=(μ,Σ) posterior samples via Gibbs.

    Output shape: (num_samples, dim + upper_triangular_size(dim)),
    consistent with the normal_inverse_wishart sampler.
    """
    if generator is None:
        generator = np.random.default_rng()

    burn_in = int(getattr(_constants, "STUDENT_T_NIW_GIBBS_BURN_IN", 200))
    thin = int(getattr(_constants, "STUDENT_T_NIW_GIBBS_THIN", 1))
    jitter = float(getattr(_constants, "STUDENT_T_NIW_GIBBS_JITTER", 1e-6))

    data = np.asarray(theta_posterior["data"], dtype=float)
    mu0 = np.asarray(theta_posterior["mu0"], dtype=float)
    kappa0 = float(theta_posterior["kappa0"])
    iota0 = float(theta_posterior["iota0"])
    Psi0 = np.asarray(theta_posterior["Psi0"], dtype=float)
    t_df = float(theta_posterior["t_df"])

    n, dim = data.shape

    state = _student_t_niw_init_state(data, jitter=jitter)

    # burn-in
    for _ in range(burn_in):
        state = _student_t_niw_gibbs_step(
            state=state,
            data=data,
            mu0=mu0,
            kappa0=kappa0,
            iota0=iota0,
            Psi0=Psi0,
            t_df=t_df,
            generator=generator,
            jitter=jitter,
        )

    out = np.zeros((num_samples, dim + upper_triangular_size(dim)), dtype=float)

    for s in range(num_samples):
        for _ in range(thin):
            state = _student_t_niw_gibbs_step(
                state=state,
                data=data,
                mu0=mu0,
                kappa0=kappa0,
                iota0=iota0,
                Psi0=Psi0,
                t_df=t_df,
                generator=generator,
                jitter=jitter,
            )
        out[s, :] = _pack_mu_sigma_triu(state["mu"], state["Sigma"])

    return out


def student_t_niw_posterior_predictive_samples(
    *,
    theta_posterior: dict,
    dim: int,
    num_samples: int,
    generator: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Sampling-based posterior predictive sampler (SAA path).

    This is what kl_pp and LV-BAS bulk-sampling use.
    It caches a Gibbs chain state inside theta_posterior to avoid re-running burn-in
    if called repeatedly (e.g. LV-BAS rejection loops).
    """
    if generator is None:
        generator = np.random.default_rng()

    burn_in = int(getattr(_constants, "STUDENT_T_NIW_GIBBS_BURN_IN", 200))
    thin = int(getattr(_constants, "STUDENT_T_NIW_GIBBS_THIN", 1))
    jitter = float(getattr(_constants, "STUDENT_T_NIW_GIBBS_JITTER", 1e-6))

    data = np.asarray(theta_posterior["data"], dtype=float)
    mu0 = np.asarray(theta_posterior["mu0"], dtype=float)
    kappa0 = float(theta_posterior["kappa0"])
    iota0 = float(theta_posterior["iota0"])
    Psi0 = np.asarray(theta_posterior["Psi0"], dtype=float)
    t_df = float(theta_posterior["t_df"])

    state = theta_posterior.get("_gibbs_state", None)
    burned = bool(theta_posterior.get("_gibbs_burned", False))

    if state is None:
        state = _student_t_niw_init_state(data, jitter=jitter)
        burned = False

    if not burned:
        for _ in range(burn_in):
            state = _student_t_niw_gibbs_step(
                state=state,
                data=data,
                mu0=mu0,
                kappa0=kappa0,
                iota0=iota0,
                Psi0=Psi0,
                t_df=t_df,
                generator=generator,
                jitter=jitter,
            )
        burned = True

    xi = np.zeros((num_samples, dim), dtype=float)

    for s in range(num_samples):
        for _ in range(thin):
            state = _student_t_niw_gibbs_step(
                state=state,
                data=data,
                mu0=mu0,
                kappa0=kappa0,
                iota0=iota0,
                Psi0=Psi0,
                t_df=t_df,
                generator=generator,
                jitter=jitter,
            )

        draw = _rvs_multivariate_t(
            df=t_df,
            loc=state["mu"],
            shape=state["Sigma"],
            size=1,
            generator=generator,
        )
        xi[s, :] = draw.reshape(-1)

    # cache warm-start state
    theta_posterior["_gibbs_state"] = state
    theta_posterior["_gibbs_burned"] = burned

    return xi

def _autocorr_1d(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Autocorrelation up to max_lag (naive O(n*max_lag), fine for diagnostics)."""
    x = np.asarray(x, dtype=float).reshape(-1)
    n = int(x.size)
    if n == 0:
        return np.asarray([], dtype=float)
    x = x - float(np.mean(x))
    var = float(np.mean(x * x))
    if var <= 0.0:
        return np.ones(max_lag + 1, dtype=float)

    max_lag = int(min(max_lag, n - 1))
    ac = np.empty(max_lag + 1, dtype=float)
    ac[0] = 1.0
    for lag in range(1, max_lag + 1):
        ac[lag] = float(np.mean(x[:-lag] * x[lag:]) / var)
    return ac


def _ess_1d(x: np.ndarray, max_lag: Optional[int] = None) -> float:
    """Geyer's initial positive sequence ESS estimate for one-dimensional trace."""
    x = np.asarray(x, dtype=float).reshape(-1)
    n = int(x.size)
    if n <= 1:
        return float(n)

    if max_lag is None:
        max_lag = min(1000, n - 1)
    max_lag = int(max_lag)

    ac = _autocorr_1d(x, max_lag=max_lag)
    if ac.size <= 1:
        return float(n)

    tau = 1.0
    # initial positive sequence on pairs (rho_{2k-1}+rho_{2k})
    for k in range(1, ac.size, 2):
        pair = float(ac[k] + (ac[k + 1] if (k + 1) < ac.size else 0.0))
        if not np.isfinite(pair) or pair <= 0.0:
            break
        tau += 2.0 * pair

    ess = n / tau
    ess = float(max(1.0, min(n, ess)))
    return ess


def _geweke_z(x: np.ndarray, first: float = 0.1, last: float = 0.5) -> float:
    """Split-chain Geweke z-score with variance adjusted by (estimated) ESS."""
    x = np.asarray(x, dtype=float).reshape(-1)
    n = int(x.size)
    if n < 20:
        return float("nan")

    n_first = max(5, int(first * n))
    n_last = max(5, int(last * n))

    x_first = x[:n_first]
    x_last = x[-n_last:]

    ess_first = _ess_1d(x_first)
    ess_last = _ess_1d(x_last)

    v_first = float(np.var(x_first, ddof=1)) / ess_first if ess_first > 0 else float("inf")
    v_last = float(np.var(x_last, ddof=1)) / ess_last if ess_last > 0 else float("inf")

    denom = float(np.sqrt(v_first + v_last))
    if denom == 0.0 or not np.isfinite(denom):
        return float("nan")

    return float((np.mean(x_first) - np.mean(x_last)) / denom)


def student_t_niw_gibbs_diagnostics(
    *,
    theta_posterior: dict,
    num_draws: int = 2000,
    burn_in: Optional[int] = None,
    thin: Optional[int] = None,
    max_lag: Optional[int] = None,
    generator: Optional[np.random.Generator] = None,
) -> dict:
    """Run a Gibbs chain and return basic convergence diagnostics.

    Returns a dict with:
      - mu_trace: (num_draws, dim)
      - logdet_sigma_trace: (num_draws,)
      - ess_mu: (dim,)
      - geweke_mu: (dim,)
      - ess_logdet_sigma: float
      - geweke_logdet_sigma: float
    """
    if generator is None:
        generator = np.random.default_rng()

    burn_in_ = int(burn_in) if burn_in is not None else int(getattr(_constants, "STUDENT_T_NIW_GIBBS_BURN_IN", 200))
    thin_ = int(thin) if thin is not None else int(getattr(_constants, "STUDENT_T_NIW_GIBBS_THIN", 1))
    jitter = float(getattr(_constants, "STUDENT_T_NIW_GIBBS_JITTER", 1e-6))

    data = np.asarray(theta_posterior["data"], dtype=float)
    mu0 = np.asarray(theta_posterior["mu0"], dtype=float)
    kappa0 = float(theta_posterior["kappa0"])
    iota0 = float(theta_posterior["iota0"])
    Psi0 = np.asarray(theta_posterior["Psi0"], dtype=float)
    t_df = float(theta_posterior["t_df"])

    n, dim = data.shape
    _ = n  # keep for clarity / debugging

    state = _student_t_niw_init_state(data, jitter=jitter)

    for _ in range(burn_in_):
        state = _student_t_niw_gibbs_step(
            state=state,
            data=data,
            mu0=mu0,
            kappa0=kappa0,
            iota0=iota0,
            Psi0=Psi0,
            t_df=t_df,
            generator=generator,
            jitter=jitter,
        )

    mu_trace = np.zeros((int(num_draws), int(dim)), dtype=float)
    logdet_sigma_trace = np.zeros((int(num_draws),), dtype=float)

    for s in range(int(num_draws)):
        for _ in range(thin_):
            state = _student_t_niw_gibbs_step(
                state=state,
                data=data,
                mu0=mu0,
                kappa0=kappa0,
                iota0=iota0,
                Psi0=Psi0,
                t_df=t_df,
                generator=generator,
                jitter=jitter,
            )
        mu_trace[s, :] = state["mu"]
        sign, logdet = np.linalg.slogdet(state["Sigma"])
        logdet_sigma_trace[s] = float(logdet) if sign > 0 else float("nan")

    if max_lag is None:
        max_lag_ = min(1000, int(num_draws) - 1)
    else:
        max_lag_ = int(max_lag)

    ess_mu = np.array([_ess_1d(mu_trace[:, j], max_lag=max_lag_) for j in range(dim)], dtype=float)
    geweke_mu = np.array([_geweke_z(mu_trace[:, j]) for j in range(dim)], dtype=float)

    valid_logdet = logdet_sigma_trace[np.isfinite(logdet_sigma_trace)]
    ess_logdet = _ess_1d(valid_logdet, max_lag=max_lag_) if valid_logdet.size > 1 else float("nan")
    geweke_logdet = _geweke_z(valid_logdet) if valid_logdet.size > 10 else float("nan")

    return {
        "burn_in": burn_in_,
        "thin": thin_,
        "mu_trace": mu_trace,
        "logdet_sigma_trace": logdet_sigma_trace,
        "ess_mu": ess_mu,
        "geweke_mu": geweke_mu,
        "ess_logdet_sigma": float(ess_logdet),
        "geweke_logdet_sigma": float(geweke_logdet),
    }


