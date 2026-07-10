"""
Simulation utils, allowing to flexibly consider different DGPs
"""
# Author: Alicia Curth
from typing import Any, Optional, Tuple

import numpy as np
from scipy.special import expit


def simulate_treatment_setup(
    n: int,
    d: int = 25,
    n_w: int = 0,
    n_c: int = 0,
    n_o: int = 0,
    n_t: int = 0,
    covariate_model: Any = None,
    covariate_model_params: Optional[dict] = None,
    propensity_model: Any = None,
    propensity_model_params: Optional[dict] = None,
    mu_0_model: Any = None,
    mu_0_model_params: Optional[dict] = None,
    mu_1_model: Any = None,
    mu_1_model_params: Optional[dict] = None,
    error_sd: float = 1,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generic function to flexibly simulate a treatment setup.

    Parameters
    ----------
    n: int
        Number of observations to generate
    d: int
        dimension of X to generate
    n_o: int
        Dimension of outcome-factor
    n_c: int
        Dimension of confounding factor
    n_t: int
        Dimension of purely predictive variables (support of tau(x)
    n_w: int
        Dimension of treatment assignment factor
    covariate_model:
        Model to generate covariates. Default: multivariate normal
    covariate_model_params: dict
        Additional parameters to pass to covariate model
    propensity_model:
        Model to generate propensity scores
    propensity_model_params:
        Additional parameters to pass to propensity model
    mu_0_model:
        Model to generate untreated outcomes
    mu_0_model_params:
        Additional parameters to pass to untreated outcome model
    mu_1_model:
        Model to generate treated outcomes.
    mu_1_model_params:
        Additional parameters to pass to treated outcome model
    error_sd: float, default 1
        Standard deviation of normal errors
    seed: int
        Seed

    Returns
    -------
        X, y, w, p, t - Covariates, observed outcomes, treatment indicators, propensities, CATE
    """
    # input checks
    n_nuisance = d - (n_c + n_o + n_w + n_t)
    if n_nuisance < 0:
        raise ValueError("Dimensions should add up to maximally d.")

    # set defaults
    if covariate_model is None:
        covariate_model = normal_covariate_model

    if covariate_model_params is None:
        covariate_model_params = {}

    if propensity_model is None:
        propensity_model = propensity_AISTATS

    if propensity_model_params is None:
        propensity_model_params = {}

    if mu_0_model is None:
        mu_0_model = mu0_AISTATS

    if mu_0_model_params is None:
        mu_0_model_params = {}

    if mu_1_model is None:
        mu_1_model = mu1_AISTATS

    if mu_1_model_params is None:
        mu_1_model_params = {}

    np.random.seed(seed)

    # generate data and outcomes
    X = covariate_model(
        n=n,
        n_nuisance=n_nuisance,
        n_c=n_c,
        n_o=n_o,
        n_w=n_w,
        n_t=n_t,
        **covariate_model_params
    )
    mu_0 = mu_0_model(X, n_c=n_c, n_o=n_o, n_w=n_w, **mu_0_model_params)
    mu_1 = mu_1_model(
        X, n_c=n_c, n_o=n_o, n_w=n_w, n_t=n_t, mu_0=mu_0, **mu_1_model_params
    )
    t = mu_1 - mu_0

    # generate treatments
    p = propensity_model(X, n_c=n_c, n_w=n_w, **propensity_model_params)
    w = np.random.binomial(1, p=p)

    # generate observables
    y = w * mu_1 + (1 - w) * mu_0 + np.random.normal(0, error_sd, n)

    return X, y, w, p, t


# normal covariate model (Adapted from Hassanpour & Greiner, 2020) -------------
def get_multivariate_normal_params(
    m: int, correlated: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    # Adapted from Hassanpour & Greiner (2020)
    if correlated:
        mu = np.zeros(m)  # np.random.normal(size=m)/10
        temp = np.random.uniform(size=(m, m))
        temp = 0.5 * (np.transpose(temp) + temp)
        sig = (np.ones((m, m)) - np.eye(m)) * temp / 10 + 0.5 * np.eye(
            m
        )  # (temp + m * np.eye(m)) / 10

    else:
        mu = np.zeros(m)
        sig = np.eye(m)

    return mu, sig


def get_set_normal_covariates(m: int, n: int, correlated: bool = False) -> np.ndarray:
    if m == 0:
        return
    mu, sig = get_multivariate_normal_params(m, correlated=correlated)
    return np.random.multivariate_normal(mean=mu, cov=sig, size=n)


def normal_covariate_model(
    n: int,
    n_nuisance: int = 25,
    n_c: int = 0,
    n_o: int = 0,
    n_w: int = 0,
    n_t: int = 0,
    correlated: bool = False,
) -> np.ndarray:
    X_stack: Tuple = ()
    for n_x in [n_w, n_c, n_o, n_t, n_nuisance]:
        if n_x > 0:
            X_stack = (*X_stack, get_set_normal_covariates(n_x, n, correlated))

    return np.hstack(X_stack)


def propensity_AISTATS(
    X: np.ndarray,
    n_c: int = 0,
    n_w: int = 0,
    xi: float = 0.5,
    nonlinear: bool = True,
    offset: Any = 0,
    target_prop: Optional[np.ndarray] = None,
) -> np.ndarray:
    if n_c + n_w == 0:
        # constant propensity
        return xi * np.ones(X.shape[0])
    else:
        coefs = np.ones(n_c + n_w)

        if nonlinear:
            z = np.dot(X[:, : (n_c + n_w)] ** 2, coefs) / (n_c + n_w)
        else:
            z = np.dot(X[:, : (n_c + n_w)], coefs) / (n_c + n_w)

        if type(offset) is float or type(offset) is int:
            prop = expit(xi * z + offset)
            if target_prop is not None:
                avg_prop = np.average(prop)
                prop = target_prop / avg_prop * prop
            return prop
        elif offset == "center":
            # center the propensity scores to median 0.5
            prop = expit(xi * (z - np.median(z)))
            if target_prop is not None:
                avg_prop = np.average(prop)
                prop = target_prop / avg_prop * prop
            return prop
        else:
            raise ValueError("Not a valid value for offset")


def propensity_constant(
    X: np.ndarray, n_c: int = 0, n_w: int = 0, xi: float = 0.5
) -> np.ndarray:
    return xi * np.ones(X.shape[0])


def mu0_AISTATS(
    X: np.ndarray, n_w: int = 0, n_c: int = 0, n_o: int = 0, scale: bool = False
) -> np.ndarray:
    if n_c + n_o == 0:
        return np.zeros((X.shape[0]))
    else:
        if not scale:
            coefs = np.ones(n_c + n_o)
        else:
            coefs = 10 * np.ones(n_c + n_o) / (n_c + n_o)
        return np.dot(X[:, n_w : (n_w + n_c + n_o)] ** 2, coefs)


def mu1_AISTATS(
    X: np.ndarray,
    n_w: int = 0,
    n_c: int = 0,
    n_o: int = 0,
    n_t: int = 0,
    mu_0: Optional[np.ndarray] = None,
    nonlinear: int = 2,
    withbase: bool = True,
    scale: bool = False,
) -> np.ndarray:
    if n_t == 0:
        return mu_0
    # use additive effect
    else:
        if scale:
            coefs = 10 * np.ones(n_t) / n_t
        else:
            coefs = np.ones(n_t)
        X_sel = X[:, (n_w + n_c + n_o) : (n_w + n_c + n_o + n_t)]
    if withbase:
        return mu_0 + np.dot(X_sel**nonlinear, coefs)
    else:
        return np.dot(X_sel**nonlinear, coefs)


# Other simulation settings not used in AISTATS paper
# uniform covariate model
def uniform_covariate_model(
    n: int,
    n_nuisance: int = 0,
    n_c: int = 0,
    n_o: int = 0,
    n_w: int = 0,
    n_t: int = 0,
    low: int = -1,
    high: int = 1,
) -> np.ndarray:
    d = n_nuisance + n_c + n_o + n_w + n_t
    return np.random.uniform(low=low, high=high, size=(n, d))


def mu1_additive(
    X: np.ndarray,
    n_w: int = 0,
    n_c: int = 0,
    n_o: int = 0,
    n_t: int = 0,
    mu_0: Optional[np.ndarray] = None,
) -> np.ndarray:
    if n_t == 0:
        return mu_0
    else:
        coefs = np.random.normal(size=n_t)
        return np.dot(X[:, (n_w + n_c + n_o) : (n_w + n_c + n_o + n_t)], coefs) / n_t


# regression surfaces from Hassanpour & Greiner
def mu0_hg(X: np.ndarray, n_w: int = 0, n_c: int = 0, n_o: int = 0) -> np.ndarray:
    if n_c + n_o == 0:
        return np.zeros((X.shape[0]))
    else:
        coefs = np.random.normal(size=n_c + n_o)
        return np.dot(X[:, n_w : (n_w + n_c + n_o)], coefs) / (n_c + n_o)


def mu1_hg(
    X: np.ndarray,
    n_w: int = 0,
    n_c: int = 0,
    n_o: int = 0,
    n_t: int = 0,
    mu_0: Optional[np.ndarray] = None,
) -> np.ndarray:
    if n_c + n_o == 0:
        return np.zeros((X.shape[0]))
    else:
        coefs = np.random.normal(size=n_c + n_o)
        return np.dot(X[:, n_w : (n_w + n_c + n_o)] ** 2, coefs) / (n_c + n_o)


def propensity_hg(
    X: np.ndarray, n_c: int = 0, n_w: int = 0, xi: Optional[float] = None
) -> np.ndarray:
    # propensity set-up used in Hassanpour & Greiner (2020)
    if n_c + n_w == 0:
        return 0.5 * np.ones(X.shape[0])
    else:
        if xi is None:
            xi = 1

        coefs = np.random.normal(size=n_c + n_w)
        z = np.dot(X[:, : (n_c + n_w)], coefs)
        return expit(xi * z)
