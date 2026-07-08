"""Functions for the portfolio problem"""

from typing import Optional, Callable, Dict, Any, Tuple
import numpy as np
import cvxpy as cp

from .bayes_conjugates import (
    sample_posterior,
    posterior_predictive_params,
    sample_posterior_predictive,
)
from .lv_bulk_set import build_score, dkw_select_threshold
from .lv_dro import make_bulk_set_spec



def portfolio_objective_cvxpy(x, xi):
    """CVXPY portfolio objective"""
    return - xi @ x


def get_kl_portfolio_problem(num_stocks: int, num_cov_samples: int) -> cp.Problem:
    """Evaluate portfolio cost function with cvxpy assuming a Gaussian likelihood

    Args:
        num_stocks: Number of stocks in portfolio, i.e. dimension of random variable
        num_cov_samples: Number of covariance samples from the posterior

    Returns:
        Cvxpy problem object
    """
    # variables
    x = cp.Variable(num_stocks, name="x")

    # parameters
    epsilon_minus_constant = cp.Parameter(1, name="epsilon_minus_constant", nonneg=True)
    mu_post = cp.Parameter(num_stocks, name="mu_post")
    sqrt_cov_post_samples = [
        cp.Parameter((num_stocks, num_stocks), name=f"sqrt_cov_post_{i}")
        for i in range(num_cov_samples)
    ]

    # objective function: maximise return whilst minimising standard deviation
    portfolio_objective = cp.Minimize(
        - mu_post @ x
        + cp.sqrt(2 * epsilon_minus_constant)
        * (1.0 / float(num_cov_samples))
        * cp.sum([cp.norm(sqrt_cov_post_samples[i] @ x) for i in range(num_cov_samples)])
    )

    # constraints
    constraints = [x >= 0, cp.sum(x) == 1]

    return cp.Problem(portfolio_objective, constraints)


def bdro_portfolio_posterior_samples(
    num_posterior_samples: int,
    mu_post: np.ndarray,
    iota_post: float,
    Psi_post: np.ndarray,
    generator: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """With KL BDRO, we only need to sample the covariance using an inverse Wishart.
    We don't need to sample the mean because its available in closed form."""
    # get covariance samples from inverse Wishart
    dim = Psi_post.shape[0]
    inverse_wishart_params = (iota_post, Psi_post)
    vec_triu_cov_samples = sample_posterior(
        "inverse_wishart", inverse_wishart_params, num_posterior_samples, generator=generator
    )

    # vectorize the covariance matrix
    vec_triu_size = vec_triu_cov_samples.shape[1]

    # return the samples in vectorized format
    theta_sample = np.zeros((num_posterior_samples, dim + vec_triu_size))
    for i, vec_triu_cov in enumerate(vec_triu_cov_samples):
        theta_sample[i, :dim] = mu_post
        theta_sample[i, dim:] = vec_triu_cov
    return theta_sample


def calibrate_lv_bulk_set(
    selection_returns: np.ndarray,
    audit_returns: np.ndarray,
    score_type: str,
    gamma: float,
    delta: float,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Calibrate an LV-BAS bulk set Xi_0 for portfolio returns using DKW.

    Parameters
    ----------
    selection_returns : (n_sel, d) array_like
        Returns used to fit the score geometry (centre and scales).
    audit_returns : (n_audit, d) array_like
        Independent audit sample used in the DKW inequality.
    score_type : {"ellipsoid", "box", "directional"}
        Score type supported by :func:`lv_bulk_set.build_score`.
    gamma : float
        Target bulk shortfall parameter in (0, 1).
    delta : float
        Confidence level parameter in (0, 1) for the DKW bound.

    Returns
    -------
    xi0_spec : dict
        Bulk-set specification suitable for :mod:`mis_dro.lv_dro`.
    info : dict
        Dictionary with keys:
          - "score_fn": the score function s(xi),
          - "score_meta": score metadata (mu_c, Sigma_c or w, etc.),
          - "dkw_result": the dict returned by dkw_select_threshold.
    """
    selection_returns = np.asarray(selection_returns, dtype=float)
    audit_returns = np.asarray(audit_returns, dtype=float)

    score_fn, score_meta = build_score(selection_returns, score_type=score_type)
    audit_scores = score_fn(audit_returns)
    dkw_result = dkw_select_threshold(audit_scores, gamma=gamma, delta=delta)
    if not dkw_result["exists"]:
        raise RuntimeError(
            f"DKW calibration infeasible for gamma={gamma}, delta={delta}; "
            f"r = {dkw_result['r']:.4g}, gamma must be >= r."
        )
    t_hat = float(dkw_result["t_hat"])
    xi0_spec = make_bulk_set_spec(score_meta, t_hat)

    info: Dict[str, Any] = {
        "score_fn": score_fn,
        "score_meta": score_meta,
        "dkw_result": dkw_result,
    }
    return xi0_spec, info


def make_lv_portfolio_f_spec(dim: int) -> dict:
    """Construct an f_spec for LV-BAS portfolio loss f_x(xi) = -x^T xi.

    This matches the API expected by mis_dro.lv_dro:
      - f_spec["type"] == "linear"
      - f_spec["a_fn"](x) returns a_x
      - f_spec["b_fn"](x) returns b_x
    """
    def a_fn(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.size != dim:
            raise ValueError(f"x must have dimension {dim}, got {x.size}.")
        # LV-BAS portfolio loss f_x(ξ) = -xᵀ ξ  ⇒  a_x = -x, b_x = 0
        return -x

    def b_fn(x: np.ndarray) -> float:
        return 0.0

    return {
        "type": "linear",
        "a_fn": a_fn,
        "b_fn": b_fn,
    }



def make_gaussian_pc_sampler(mu: np.ndarray, cov: np.ndarray) -> Callable[[int, np.random.Generator], np.ndarray]:
    """Create a Gaussian posterior predictive sampler P_c for LV-BAS.

    This utility does *not* prescribe how (mu, cov) are obtained; they can
    come from the existing Bayesian conjugate machinery. It simply wraps
    them into the (n, rng) -> samples interface expected by lv_dro.
    """
    mu = np.asarray(mu, dtype=float).reshape(-1)
    cov = np.asarray(cov, dtype=float)
    dim = mu.size
    if cov.shape != (dim, dim):
        raise ValueError(f"cov must have shape ({dim}, {dim}), got {cov.shape}.")

    def pc_sampler(n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.multivariate_normal(mean=mu, cov=cov, size=n)

    return pc_sampler

def make_lv_pc_sampler_from_niw(
    theta_posterior: tuple,
    dim: int,
    likelihood: str = "multivariate_normal",
    posterior: str = "normal_inverse_wishart",
) -> Callable[[int, np.random.Generator], np.ndarray]:
    """
    Build a posterior predictive sampler P_c for LV-BAS in the portfolio setting,
    reusing the same normal-inverse-Wishart machinery as KL-BAS / KL-PP.

    Parameters
    ----------
    theta_posterior : tuple
        Posterior parameters as returned by
        bayes_conjugates.get_posterior_params("normal_inverse_wishart", data, prior).

    dim : int
        Number of assets (dimension of xi).

    likelihood : str, default "multivariate_normal"
        Likelihood string; must match the KL-BAS portfolio setting.

    posterior : str, default "normal_inverse_wishart"
        Posterior string; must be "normal_inverse_wishart".

    Returns
    -------
    pc_sampler : callable
        Function pc_sampler(n, rng) -> np.ndarray of shape (n, dim),
        drawing i.i.d. samples from the NIW posterior predictive P_c.
    """
    if posterior != "normal_inverse_wishart":
        raise ValueError(
            f"make_lv_pc_sampler_from_niw requires posterior 'normal_inverse_wishart', got {posterior!r}."
        )
    if likelihood != "multivariate_normal":
        raise ValueError(
            f"make_lv_pc_sampler_from_niw requires likelihood 'multivariate_normal', got {likelihood!r}."
        )

    # Convert NIW posterior parameters to posterior predictive parameters.
    pp_params = posterior_predictive_params(posterior, theta_posterior)
    pp_params = np.asarray(pp_params, dtype=float).reshape(1, -1)

    def pc_sampler(n: int, rng: Optional[np.random.Generator]) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        samples = sample_posterior_predictive(
            likelihood=likelihood,
            posterior=posterior,
            theta_sample=pp_params,
            dim=dim,
            num_likelihood_samples=n,
            generator=rng,
        )
        samples = np.asarray(samples, dtype=float)
        # sample_posterior_predictive returns a batch; for NIW it is (1, n, dim)
        # or (n, dim). We standardise to (n, dim).
        return samples.reshape(-1, dim)

    return pc_sampler


def solve_lv_portfolio_socp(
    mu_eff: np.ndarray,
    Sigma_bulk: np.ndarray,
    lam: float,
    *,
    verbose: bool = False,
) -> np.ndarray:
    """
    Solve     min_x  -mu_eff^T x + lam * ||Sigma_bulk^{1/2} x||_2
    subject to x >= 0, 1^T x = 1.

    Returns
    -------
    x_opt : (d,) numpy array
        Optimal long-only, fully-invested portfolio.
    """
    mu_eff = np.asarray(mu_eff, dtype=float).ravel()
    d = mu_eff.size

    Sigma_bulk = np.asarray(Sigma_bulk, dtype=float).reshape(d, d)
    # Symmetrise and add a tiny ridge for numerical robustness
    Sigma_sym = 0.5 * (Sigma_bulk + Sigma_bulk.T)
    Sigma_sym += 1e-8 * np.eye(d)

    # Cholesky factor: Sigma ≈ L L^T
    L = np.linalg.cholesky(Sigma_sym)

    x = cp.Variable(d)
    obj = -mu_eff @ x
    if lam > 0:
        obj = obj + lam * cp.norm(L @ x, 2)

    prob = cp.Problem(cp.Minimize(obj), [x >= 0, cp.sum(x) == 1])

    # Use MOSEK if available, fall back otherwise
    installed = set(cp.installed_solvers())
    if "MOSEK" in installed:
        prob.solve(solver=None, verbose=verbose)
    else:
        prob.solve(verbose=verbose)

    if x.value is None:
        raise RuntimeError(f"LV portfolio SOCP failed with status {prob.status}.")

    return np.asarray(x.value, dtype=float).ravel()

def get_lv_portfolio_problem(num_stocks: int) -> cp.Problem:
    """
    Parametrised LV-BAS portfolio SOCP:

        min_x  -mu_eff^T x + lam * ||sqrt_cov_bulk @ x||_2
        s.t.   x >= 0, 1^T x = 1.

    Parameters (all set per replication via problem.param_dict):
      - mu_eff        : (num_stocks,)
      - sqrt_cov_bulk : (num_stocks, num_stocks) Cholesky / matrix square root
      - lam           : scalar >= 0
    """
    # decision variable
    x = cp.Variable(num_stocks, name="x")

    # parameters
    mu_eff = cp.Parameter(num_stocks, name="mu_eff")
    sqrt_cov_bulk = cp.Parameter(
        (num_stocks, num_stocks), name="sqrt_cov_bulk"
    )
    lam = cp.Parameter(nonneg=True, name="lam")

    # objective
    objective = cp.Minimize(
        - mu_eff @ x + lam * cp.norm(sqrt_cov_bulk @ x, 2)
    )

    # constraints: long-only, fully invested
    constraints = [x >= 0, cp.sum(x) == 1]

    return cp.Problem(objective, constraints)
