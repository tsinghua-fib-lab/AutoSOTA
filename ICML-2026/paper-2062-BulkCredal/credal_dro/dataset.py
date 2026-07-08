"""Data generation and dataset functions"""

import math
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

import os
import warnings

from scipy.stats import expon, gamma, norm, t, multivariate_normal, chi2
from sklearn.datasets import make_spd_matrix


from bayesian_dro.Bayesian_DRO_continuous import data_generation, DGP_STD_TRUNCATED_NORMAL
from .constants import IN_SAMPLE_TIME_WINDOW, OUT_OF_SAMPLE_TIME_WINDOW, DGP_NORMAL_KNOWN_VARIANCE_STD

from urllib.error import HTTPError, URLError

def _student_t_base_params(dim: int) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Base parameters for the multivariate Student-t DGP used in the newsvendor experiments.

    Returns
    -------
    df : float
    loc : (dim,)
    shape : (dim, dim) SPD "shape" matrix for the Student-t
    base_std : (dim,) marginal std proxy (sqrt(diag(shape)))
    """
    df = 3.0
    loc = 30.0 * np.ones(dim)

    # Correlated SPD scale via Toeplitz correlation
    rho = 0.6
    idx = np.arange(dim)
    corr = rho ** np.abs(idx[:, None] - idx[None, :])

    # Heteroscedastic scales (still SPD)
    scales = 10.0 * (1.0 + 0.1 * idx)
    shape = np.diag(scales) @ corr @ np.diag(scales)

    base_std = np.sqrt(np.maximum(np.diag(shape), 1e-12))
    return df, loc, shape, base_std


def _sample_adversarial_ellipsoid_surface(
    *,
    loc: np.ndarray,
    shape: np.ndarray,
    quantile: float,
    size: int,
    generator: np.random.Generator,
    jitter_scale: float = 0.0,
) -> np.ndarray:
    """
    Adversarial samples concentrated on an ellipsoid "surface" defined by (loc, shape).

    We use radius r = sqrt(chi2.ppf(quantile, df=dim)) and sample points:
        x = loc + r * v^T * chol(shape),
    where v is a random unit direction.

    Optional small Gaussian jitter can be added (still unbounded support).
    """
    loc = np.asarray(loc, dtype=float).reshape(-1)
    shape = np.asarray(shape, dtype=float)
    dim = int(loc.size)

    shape_sym = 0.5 * (shape + shape.T) + 1e-12 * np.eye(dim)
    L = np.linalg.cholesky(shape_sym)

    r = float(np.sqrt(chi2.ppf(quantile, df=dim)))

    Z = generator.normal(size=(size, dim))
    norms = np.linalg.norm(Z, axis=1)
    norms = np.where(norms == 0.0, 1.0, norms)
    V = Z / norms[:, None]                      # unit directions
    X = loc[None, :] + r * (V @ L.T)            # on ellipsoid "surface"

    if jitter_scale > 0.0:
        X = X + generator.normal(scale=jitter_scale, size=X.shape)

    return np.asarray(X, dtype=float)


def rvs_multivariate_t(
    *,
    df: float,
    loc: np.ndarray,
    shape: np.ndarray,
    size: int,
    generator: np.random.Generator,
) -> np.ndarray:
    """
    Draw samples from a multivariate Student-t distribution using the
    normal/chi-square mixture representation:

        X = loc + Z / sqrt(U/df),
        Z ~ N(0, shape),  U ~ ChiSquare(df)

    Returns: array of shape (size, dim)
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


def sample_dgp(
    dgp: str,
    num_observations: int,
    contamination: float = 0.0,
    dim: int = 1,
    contamination_type: Optional[str] = None,
    generator: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Sample from the DGP"""
    if not generator:
        generator = np.random.default_rng()
    if dgp == "unif_0_theta":
        """Univariate Uniform(0, θ_true) DGP.

        Notes
        -----
        We keep the `sample_dgp` signature unchanged, so θ_true is fixed here.
        This DGP is intended to be paired with:
          likelihood = "uniform_0_theta"
          posterior  = "pareto"
        """
        if dim != 1:
            raise ValueError(f"unif_0_theta is implemented for dim=1, got dim={dim}")
        theta_true = 50.0  # mean demand = 25.0, roughly aligned with existing newsvendor scales
        x = generator.uniform(low=0.0, high=theta_true, size=num_observations)
        return np.asarray(x, dtype=float).reshape((num_observations, 1))

    if dgp == "normal":
        return norm.rvs(
            loc=25,
            scale=DGP_STD_TRUNCATED_NORMAL,
            # scale=5,
            size=num_observations,
            random_state=generator,
        ).reshape((num_observations, 1))
    if dgp == "truncated_normal":
        return data_generation(
            num_observations, random_state=generator
        ).reshape((num_observations, 1))  # generate new observations
    if dgp == "contaminated_exp":
        # specify contamination level
        return data_generation_outliers(
            num_observations, contamination, random_state=generator
        ).reshape((num_observations, 1))
    if dgp == "contaminated_exp_large_outliers":
        return data_generation_outliers(
            num_observations, contamination, outlier_mean=1000.0, random_state=generator
        ).reshape((num_observations, 1))
    if dgp == "contaminated_exp_small_outliers":
        return contaminated_exp_small_outliers(num_observations, contamination, random_state=generator).reshape((num_observations, 1))
    if dgp == "exponential":
        return expon.rvs(scale=20.0, size=num_observations, random_state=generator).reshape((num_observations, 1))
    if dgp == "gamma":
        return data_generation_gamma(num_observations, a=10, random_state=generator).reshape((num_observations, 1))
    if dgp == "multivariate_normal_known_cov":
        dgp_mean = np.array([10.0, 20.0, 30.0, 35.0, 22.0])
        dgp_cov = (DGP_STD_TRUNCATED_NORMAL**2)*np.eye(dim)
        # cov_multiplier = 20.0
        # # NOTE sklearn doesn't seem to accept a Generator
        # sklearn_cov_seed = 1    # NOTE fix the seed, we always want the same covariance
        # sklearn_random_state = np.random.RandomState(seed=sklearn_cov_seed)
        # dgp_cov = cov_multiplier * make_spd_matrix(dim, random_state=sklearn_random_state)
        return multivariate_normal.rvs(dgp_mean, dgp_cov, size=num_observations, random_state=generator)
    if dgp == "multivariate_normal":
        dgp_mean = np.array([10.0, 20.0, 30.0, 35.0, 22.0])
        cov_multiplier = 20.0
        # NOTE sklearn doesn't seem to accept a Generator
        sklearn_cov_seed = 1    # NOTE fix the seed, we always want the same covariance
        sklearn_random_state = np.random.RandomState(seed=sklearn_cov_seed)
        dgp_cov = cov_multiplier * make_spd_matrix(dim, random_state=sklearn_random_state)
        return multivariate_normal.rvs(dgp_mean, dgp_cov, size=num_observations, random_state=generator)
    if dgp == "cont_multivariate_normal":
        return cont_multivariate_normal(
            num_observations, contamination, random_state=generator)
    if dgp == "portfolio_contaminated_multivariate_normal":
        return portfolio_contaminated_multivariate_normal(num_observations, contamination, random_state=generator)
    if dgp == "contaminated_normal":
        return contaminated_normal(
            num_observations, contamination, random_state=generator)
    if dgp == "student_t":
        """
        Student-t synthetic demand DGP (optionally contaminated).

        Clean model:
          xi ~ multivariate Student-t(df=3, loc=30*1, shape=SPD)

        Contamination (applied only if contamination>0 and contamination_type is not None):
          - "shift":      mean-shifted Student-t (same shape)
          - "scale":      higher-variance Student-t (shape scaled up)
          - "spike":      rare "event spikes" (tight Gaussian around a large mean shift)
          - "adversarial": mass near an ellipsoid boundary defined by (loc, shape)
        """
        if dim < 1:
            raise ValueError(f"student_t requires dim>=1, got dim={dim}")

        # ---- 1D special case (kept simple)
        if dim == 1:
            df, loc, scale = 3.0, 30.0, 10.0

            if contamination <= 0.0:
                x = t.rvs(df=df, loc=loc, scale=scale, size=num_observations, random_state=generator)
                return np.asarray(x, dtype=float).reshape((num_observations, 1))

            if contamination_type is None:
                raise ValueError("contamination_type must be set when contamination>0 for dgp='student_t'.")

            cont_size = int(np.floor(contamination * num_observations))
            n_real = int(num_observations - cont_size)

            x_real = t.rvs(df=df, loc=loc, scale=scale, size=n_real, random_state=generator)
            x_real = np.asarray(x_real, dtype=float).reshape((n_real, 1))

            if cont_size == 0:
                return x_real

            if contamination_type == "scale":
                # Larger variance
                x_out = t.rvs(df=df, loc=loc, scale=2.0 * scale, size=cont_size, random_state=generator)
            elif contamination_type == "shift":
                x_out = t.rvs(df=df, loc=loc + 3.0 * scale, scale=scale, size=cont_size, random_state=generator)
            elif contamination_type == "spike":
                # Rare "promotion/event" spikes: tight normal around a large mean
                x_out = norm.rvs(loc=loc + 6.0 * scale, scale=0.2 * scale, size=cont_size, random_state=generator)
            elif contamination_type == "adversarial":
                # Put mass near a high quantile “boundary”
                # (not truly adversarial in 1D, but matches the intent)
                r = float(abs(t.ppf(0.95, df=df)))
                x_out = loc + r * scale + norm.rvs(loc=0.0, scale=0.05 * scale, size=cont_size, random_state=generator)
            elif contamination_type == "entire_shift":
                # Full distribution-family shift: replace Student-t by a Gaussian.
                # By default this matches the clean Student-t mean and variance.
                mean_out = loc
                std_out = math.sqrt(df / (df - 2.0)) * scale
                x_out = norm.rvs(loc=mean_out, scale=std_out, size=cont_size, random_state=generator)
            else:
                raise ValueError(
                    f"Unknown contamination_type={contamination_type!r} for dgp='student_t'. "
                    "Use 'shift', 'scale', 'spike', 'adversarial', or 'entire_shift'."
                )

            x_out = np.asarray(x_out, dtype=float).reshape((cont_size, 1))
            out = np.concatenate([x_real, x_out], axis=0)
            generator.shuffle(out)
            return out

        # ---- Multivariate case
        df, loc, shape, base_std = _student_t_base_params(dim)

        # Clean sample (default behaviour)
        if contamination <= 0.0:
            x = rvs_multivariate_t(df=df, loc=loc, shape=shape, size=num_observations, generator=generator)
            return np.asarray(x, dtype=float).reshape((num_observations, dim))

        # If user asked for contamination, they must specify a type
        if contamination_type is None:
            raise ValueError(
                "contamination_type must be one of "
                "{'shift','scale','spike','adversarial','entire_shift'} "
                "when contamination>0 for dgp='student_t'."
            )

        cont_size = int(np.floor(contamination * num_observations))
        n_real = int(num_observations - cont_size)

        x_real = rvs_multivariate_t(df=df, loc=loc, shape=shape, size=n_real, generator=generator)
        x_real = np.asarray(x_real, dtype=float).reshape((n_real, dim))

        if cont_size == 0:
            return x_real

        if contamination_type == "scale":
            # Larger covariance (shape scaled up)
            shape_scale = 4.0
            x_out = rvs_multivariate_t(
                df=df,
                loc=loc,
                shape=shape_scale * shape,
                size=cont_size,
                generator=generator,
            )

        elif contamination_type == "shift":
            # Mean shift by a few marginal stds (elementwise)
            shift_mult = 3.0
            mean_out = loc + shift_mult * base_std
            x_out = rvs_multivariate_t(
                df=df,
                loc=mean_out,
                shape=shape,
                size=cont_size,
                generator=generator,
            )

        elif contamination_type == "spike":
            # Rare "event spikes": concentrated cluster at high demand
            spike_mult = 6.0
            spike_mean = loc + spike_mult * base_std
            spike_cov_scale = 0.05  # tight cluster, still unbounded
            shape_sym = 0.5 * (shape + shape.T) + 1e-12 * np.eye(dim)
            spike_cov = spike_cov_scale * shape_sym
            x_out = generator.multivariate_normal(mean=spike_mean, cov=spike_cov, size=cont_size)

        elif contamination_type == "adversarial":
            # Concentrate mass near a high-quantile ellipsoid surface for the *base* shape
            # (intended to stress "bulk" methods)
            x_out = _sample_adversarial_ellipsoid_surface(
                loc=loc,
                shape=shape,
                quantile=0.95,
                size=cont_size,
                generator=generator,
                jitter_scale=0.0,  # set e.g. 0.1*base_std.mean() if you want slight spread
            )
        elif contamination_type == "entire_shift":
            # Full distribution-family shift: replace Student-t by a Gaussian.
            # By default this matches the clean Student-t mean and covariance.
            mean_out = loc
            cov_out = (df / (df - 2.0)) * (0.5 * (shape + shape.T) + 1e-12 * np.eye(dim))
            x_out = generator.multivariate_normal(mean=mean_out, cov=cov_out, size=cont_size)

        else:
            raise ValueError(
                f"Unknown contamination_type={contamination_type!r}. "
                "Use 'shift', 'scale', 'spike', 'adversarial', or 'entire_shift'."
            )
        
        x_out = np.asarray(x_out, dtype=float).reshape((cont_size, dim))
        out = np.concatenate([x_real, x_out], axis=0)
        generator.shuffle(out)
        return out


    if dgp == "contaminated_exp_old":
        cont_size = int(np.floor(contamination * num_observations))
        n_real = num_observations - cont_size
        data = expon.rvs(scale=10, size=n_real, random_state=generator)
        outl = expon.rvs(scale=70, size=cont_size, random_state=generator)
        data = np.concatenate((data, outl), axis=0)
        generator.shuffle(data)  # shuffles the data in-place
        return data.reshape((num_observations,1))
    if dgp == "bimodal_multivariate_gaussian":
        cont_size = int(np.floor(contamination * num_observations))
        n_real = num_observations - cont_size
        data_mode1 = multivariate_normal.rvs(mean=np.array([10,20,33,22,25]), cov=5*np.eye(5), size=int(n_real/2), random_state=generator)
        data_mode2 = multivariate_normal.rvs(mean=60*np.ones(5), cov=5*np.eye(5), size=int(n_real/2), random_state=generator)
        outl = multivariate_normal.rvs(mean=90*np.ones(5), cov=5*np.eye(5), size=cont_size, random_state=generator)
        data = np.concatenate((data_mode1, data_mode2, outl), axis=0)
        generator.shuffle(data)  # shuffles the data in-place
        return data
    if dgp == "bimodal_univariate_gaussian":
        cont_size = int(np.floor(contamination * num_observations))
        n_real = num_observations - cont_size
        data_mode1 = norm.rvs(loc=10, scale=5, size=int(n_real/2), random_state=generator)
        data_mode2 = norm.rvs(loc=60, scale=5, size=int(n_real/2), random_state=generator)
        outl = norm.rvs(loc=90, scale=5, size=cont_size, random_state=generator)
        data = np.concatenate((data_mode1, data_mode2, outl), axis=0)
        generator.shuffle(data)  # shuffles the data in-place
        return data.reshape((num_observations,1))
    if dgp == "portfolio_gaussian_5d":
        """Synthetic 5D Gaussian portfolio returns (optionally contaminated).

        Contamination types (applied only if contamination>0 and contamination_type is not None):
          - "scale": same mean, covariance scaled up
          - "shift": shifted mean, same covariance
          - "adversarial": points on the ~95% Mahalanobis-radius ellipsoid of the clean DGP
        """
        if dim != 5:
            raise ValueError(f"portfolio_gaussian_5d is defined for dim=5, got dim={dim}")

        # True mean vector (weekly returns)
        #dgp_mean = np.array([0.012, 0.010, 0.008, 0.006, 0.015])
        dgp_mean = np.array([0.0024, 0.0020, 0.0016, 0.0012, 0.003])
        # Base SPD matrix, turned into a correlation, then rescaled to target vols
        sklearn_random_state = np.random.RandomState(seed=123)
        base_cov = make_spd_matrix(dim, random_state=sklearn_random_state)
        std_base = np.sqrt(np.diag(base_cov))
        corr = base_cov / np.outer(std_base, std_base)

        target_std = np.array([0.02, 0.025, 0.018, 0.017, 0.03])  # 2–3% weekly vols
        dgp_cov = corr * np.outer(target_std, target_std)

        # Clean sample (default behaviour)
        if contamination <= 0.0:
            x = multivariate_normal.rvs(mean=dgp_mean, cov=dgp_cov, size=num_observations, random_state=generator)
            return np.asarray(x, dtype=float).reshape(num_observations, dim)

        # If user asked for contamination, they must specify a type
        if contamination_type is None:
            raise ValueError("contamination_type must be one of {'scale','shift','adversarial'} when contamination>0.")

        cont_size = int(np.floor(contamination * num_observations))
        n_real = num_observations - cont_size

        x_real = multivariate_normal.rvs(mean=dgp_mean, cov=dgp_cov, size=n_real, random_state=generator)
        x_real = np.asarray(x_real, dtype=float).reshape(n_real, dim)

        if cont_size == 0:
            return x_real

        if contamination_type == "scale":
            # Larger covariance (std scaled by 2 -> cov scaled by 4)
            cov_scale = 4.0
            x_out = multivariate_normal.rvs(
                mean=dgp_mean,
                cov=cov_scale * dgp_cov,
                size=cont_size,
                random_state=generator,
            )
            x_out = np.asarray(x_out, dtype=float).reshape(cont_size, dim)

        elif contamination_type == "shift":
            # Mean shift by a few sigmas (elementwise)
            shift_mult = 3.0
            mean_out = dgp_mean + shift_mult * target_std
            x_out = multivariate_normal.rvs(
                mean=mean_out,
                cov=dgp_cov,
                size=cont_size,
                random_state=generator,
            )
            x_out = np.asarray(x_out, dtype=float).reshape(cont_size, dim)

        elif contamination_type == "adversarial":
            # Concentrate mass on the ~95% chi-square ellipsoid surface:
            # ||Sigma^{-1/2}(x-mu)|| = sqrt(chi2_{dim}(0.95))
            r = float(np.sqrt(chi2.ppf(0.95, df=dim)))
            L = np.linalg.cholesky(dgp_cov + 1e-12 * np.eye(dim))  # Σ^{1/2}
            Z = generator.normal(size=(cont_size, dim))
            norms = np.linalg.norm(Z, axis=1)
            norms = np.where(norms == 0.0, 1.0, norms)
            V = Z / norms[:, None]                       # unit directions
            x_out = dgp_mean[None, :] + r * (V @ L.T)     # on ellipsoid surface
            x_out = np.asarray(x_out, dtype=float).reshape(cont_size, dim)

        else:
            raise ValueError(f"Unknown contamination_type={contamination_type!r}. Use 'scale', 'shift', or 'adversarial'.")

        data = np.concatenate([x_real, x_out], axis=0)
        generator.shuffle(data)  # shuffle rows in-place
        return data


    raise ValueError(f"The data-generating process specified is not supported: {dgp}")


def data_generation_outliers(
    num_observations: int,
    contamination: float,
    outlier_mean: float = 100.0,
    random_state: Optional[np.random.Generator] = None,
):
    """A contaminated exponential data-generating process (DGP)

    Args:
        num_observations: Number of observations from DGP
        contamination: Ratio of contaminated observations (outliers)
        outlier_mean: Location of the mean of the Gaussian to draw contaminated samples from
        random_state: A numpy random generator, if provided

    Notes:
        Shuffles data to ensure anomalies are not grouped together
    """
    if not random_state:
        random_state = np.random.default_rng()
    cont_size = int(np.floor(contamination * num_observations))
    n_real = num_observations - cont_size
    data = expon.rvs(scale=20, size=n_real, random_state=random_state)
    outl = norm.rvs(loc=outlier_mean, scale=0.5, size=cont_size, random_state=random_state) 
    data = np.concatenate((data, outl), axis=0)
    random_state.shuffle(data)  # shuffle the data in-place
    return data

def contaminated_exp_small_outliers(
    num_observations: int,
    contamination: float,
    random_state = None,
):
    """A contaminated exponential data-generating process (DGP) with
    small outliers coming from an exponential distribution with mean 0.01.

    Args:
        num_observations: Number of observations from DGP
        contamination: Ratio of contaminated observations (outliers)
        random_state: A numpy random generator, if provided

    Notes:
        Shuffles data to ensure anomalies are not grouped together
    """
    if not random_state:
        random_state = np.random.default_rng()
    cont_size = int(np.floor(contamination * num_observations))
    n_real = num_observations - cont_size
    data = expon.rvs(scale=20, size=n_real, random_state=random_state)
    outl = expon.rvs(scale=0.01, size=cont_size, random_state=random_state) 
    data = np.concatenate((data, outl), axis=0)
    random_state.shuffle(data)  # shuffle the data in-place
    return data

def contaminated_normal(num_observations: int, contamination: float, random_state: Optional[np.random.Generator] = None):
    """A contaminated Gaussian data-generating process (DGP)

    Args:
        num_observations: Number of observations from DGP
        contamination: Ratio of contaminated observations (outliers)
        random_state: A numpy random generator, if provided

    Notes:
        Shuffles data to ensure anomalies are not grouped together
    """
    if not random_state:
        random_state = np.random.default_rng()
    cont_size = int(np.floor(contamination * num_observations))
    n_real = num_observations - cont_size
    data = norm.rvs(loc=25, scale=DGP_NORMAL_KNOWN_VARIANCE_STD, size=n_real, random_state=random_state) 
    outl = norm.rvs(loc=75, scale=DGP_NORMAL_KNOWN_VARIANCE_STD, size=cont_size, random_state=random_state) 
    data = np.concatenate((data, outl), axis=0)
    random_state.shuffle(data)  # shuffles the data in-place
    return data
    
def cont_multivariate_normal(num_observations: int, contamination: float, random_state: Optional[np.random.Generator] = None):
    
    if not random_state:
        random_state = np.random.default_rng()
    cont_size = int(np.floor(contamination * num_observations))
    n_real = num_observations - cont_size
    dgp_mean = np.array([10.0, 20.0, 30.0, 35.0, 22.0]) #10
    dgp_mean_outl = dgp_mean + 30
    dgp_cov = (DGP_STD_TRUNCATED_NORMAL**2)*np.eye(5)
    data = multivariate_normal.rvs(dgp_mean, dgp_cov, size=n_real, random_state=random_state)
    outl = multivariate_normal.rvs(dgp_mean_outl, dgp_cov, size=cont_size, random_state=random_state)
    data = np.concatenate((data, outl), axis=0)
    random_state.shuffle(data)  # shuffles the data in-place
    return data

def portfolio_contaminated_multivariate_normal(num_observations: int, contamination: float, random_state: Optional[np.random.Generator] = None):
    """Portfolio contamination dataset"""
    if not random_state:
        random_state = np.random.default_rng()
    cont_size = int(np.floor(contamination * num_observations))
    n_real = num_observations - cont_size
    dgp_mean = np.array([2.5, 0.5, -1.0, -3.0, 3.5])
    # dgp_mean = np.zeros(5)
    dgp_mean_outl = dgp_mean + np.array([2.5, 50.0, 50.0, 50.0, 3.5])
    dgp_cov = np.diag(np.array([5.0, 10.0, 15.0, 20.0, 30.0]))
    # dgp_cov = np.diag(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    data = multivariate_normal.rvs(dgp_mean, dgp_cov, size=n_real, random_state=random_state)
    outl = multivariate_normal.rvs(dgp_mean_outl, dgp_cov, size=cont_size, random_state=random_state)
    data = np.concatenate((data, outl), axis=0)
    random_state.shuffle(data)  # shuffles the data in-place
    return data

def data_generation_gamma(num_observations: int, a: float, random_state: Optional[np.random.Generator] = None):
    """A Gamma data-generating process (DGP)

    Args:
        num_observations: Number of observations from DGP
        a: shape parameter
        random_state: A numpy random generator, if provided
    """
    if not random_state:
        random_state = np.random.default_rng()

    gamma_samples = gamma.rvs(a, size=num_observations)

    return gamma_samples

def portfolio_dataset(
    dgp: str,
    time_window_id: int,
    mmc2_dir: Path,
    in_sample_time_window: int = IN_SAMPLE_TIME_WINDOW,
    out_of_sample_time_window: int = OUT_OF_SAMPLE_TIME_WINDOW,
) -> tuple[np.ndarray, np.ndarray]:
    """Gets the training and test datasets for the porfolio problem.

    Args:
        dgp: Options include 'DowJones', 'DowJones-crash', 'FF49Industries', 'FTSE100', 'NASDAQ100', 'NASDAQComp', 'SP500'
        time_window_id: The ID of the time window.
        mmc2_dir: Path to the directory downloaded from 'data-in-brief' webpage below

    Returns:
        training_data: numpy array of shape (NUM_TRADING_DAYS_IN_YEAR, NUM_STOCKS)
        test_data: numpy array of shape (NUM_TRADING_DAYS_IN_QUARTER, NUM_STOCKS)

    Notes:
        Download data from https://www.data-in-brief.com/article/S2352-3409(16)30399-7/fulltext
    """
    returns_df = get_portfolio_returns_df(mmc2_dir, dgp)
    num_time_windows = get_num_time_windows(len(returns_df))
    assert time_window_id < num_time_windows
    start_training_week = time_window_id * 12 # inclusive
    end_training_week = start_training_week + in_sample_time_window # not inclusive
    start_test_week = end_training_week # inclusive
    end_test_week = start_test_week + out_of_sample_time_window # not inclusive
    training_data = returns_df.iloc[start_training_week: end_training_week].values
    test_data = returns_df.iloc[start_test_week: end_test_week].values
    return training_data, test_data

def get_portfolio_returns_df(mmc2_dir: Path, dgp: str) -> pd.DataFrame:
    dgp_temp = dgp
    if dgp == "DowJones-crash":
        dgp_temp = "DowJones"
    return pd.read_excel(mmc2_dir / "Datasets" / dgp_temp / f"{dgp_temp}.xlsx", sheet_name="Assets_Returns", header=None)

def get_num_time_windows(
    num_weeks: int,
    in_sample_time_window: int = IN_SAMPLE_TIME_WINDOW,
    out_of_sample_time_window: int = OUT_OF_SAMPLE_TIME_WINDOW
) -> int:
    return math.floor((num_weeks - in_sample_time_window) / out_of_sample_time_window)

# ----------------------------------------------------------------------
# California Housing experiment utilities
# ----------------------------------------------------------------------

CALIFORNIA_HOUSING_D = 8
CALIFORNIA_HOUSING_SPLIT_FRACS = (0.40, 0.10, 0.00, 0.50)  # train, select, val, test

def _resolve_realworld_data_root(data_root: Optional[Path] = None) -> Path:
    """Resolve the root directory for locally stored real-world datasets.

    This returns the *data root* (not a dataset-specific subdirectory).
    """
    if data_root is not None:
        return Path(data_root).expanduser()

    env = os.environ.get("CALIFORNIA_HOUSING_DATASET_DIR")
    if env is not None and str(env).strip() != "":
        return Path(env).expanduser()

    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / "data"


def _resolve_california_housing_local_path(data_root: Optional[Path] = None) -> Path:
    """Resolve the on-disk path to the UCI `cal_housing.data` file.

    Accepted layouts (where `data_root` is the result of `_resolve_realworld_data_root`):
      - <data_root>/california_housing/CaliforniaHousing/cal_housing.data
      - <data_root>/CaliforniaHousing/cal_housing.data
      - <data_root>/cal_housing.data

    You can also pass `data_root` as a direct path to `cal_housing.data`.
    """
    root = _resolve_realworld_data_root(data_root)
    root = Path(root).expanduser()

    if root.is_file():
        return root

    candidates = [
        root / "california_housing" / "CaliforniaHousing" / "cal_housing.data",
        root / "CaliforniaHousing" / "cal_housing.data",
        root / "cal_housing.data",
    ]
    for p in candidates:
        if p.exists():
            return p

    tried = "\n".join(f"  - {p}" for p in candidates)
    raise RuntimeError(
        "Cannot load California Housing data: `cal_housing.data` not found. Tried:\n"
        f"{tried}\n"
    )

def california_housing_split_sizes(
    train_frac: float = CALIFORNIA_HOUSING_SPLIT_FRACS[0],
    select_frac: float = CALIFORNIA_HOUSING_SPLIT_FRACS[1],
    val_frac: float = CALIFORNIA_HOUSING_SPLIT_FRACS[2],
    test_frac: float = CALIFORNIA_HOUSING_SPLIT_FRACS[3],
    data_root: Optional[Path] = None,
):
    """Return deterministic split sizes for California Housing given split fractions.
    """

    if any(frac < 0.0 for frac in (train_frac, select_frac, val_frac, test_frac)):
        raise ValueError("Split fractions must be non-negative.")
    if train_frac + select_frac + val_frac + test_frac > 1.0 + 1e-12:
        raise ValueError("Split fractions must sum to <= 1.")

    local_path = _resolve_california_housing_local_path(data_root)

    X = None
    if local_path.exists():
        # Robust reader: accepts whitespace-separated or comma-separated
        df = pd.read_csv(
            local_path,
            header=None,
            sep=r"\s+|,",
            engine="python",
        )
        arr = df.to_numpy(dtype=float)

        # UCI California Housing: 8 features + 1 target = 9 columns
        if arr.ndim != 2 or arr.shape[1] < 9:
            raise ValueError(
                f"Unexpected cal_housing.data shape {arr.shape}; expected >= 9 columns (8 features + target). "
                f"Path: {local_path}"
            )

        X = arr[:, :8]
        # y is the last column in the standard UCI format
        y = arr[:, 8]
        n = int(X.shape[0])
    else:
        raise RuntimeError(f"Cannot determine dataset size; local California Housing data not found at {local_path}.")

    n_train = int(np.floor(train_frac * n))
    n_select = int(np.floor(select_frac * n))
    n_val = int(np.floor(val_frac * n))
    n_test = n - n_train - n_select - n_val  # remainder

    if min(n_train, n_select, n_val, n_test) < 0:
        raise ValueError(
            "Invalid split sizes; got "
            f"(train={n_train}, select={n_select}, val={n_val}, test={n_test})."
        )

    return {
        "n_total": n,
        "n_train": n_train,
        "n_select": n_select,
        "n_val": n_val,
        "n_test": n_test,
    }


def california_housing_splits(
    replication: int,
    train_frac: float = CALIFORNIA_HOUSING_SPLIT_FRACS[0],
    select_frac: float = CALIFORNIA_HOUSING_SPLIT_FRACS[1],
    val_frac: float = CALIFORNIA_HOUSING_SPLIT_FRACS[2],
    test_frac: float = CALIFORNIA_HOUSING_SPLIT_FRACS[3],
    standardise_y: bool = True,
    data_root: Optional[Path] = None,
):
    """
    Load and split California Housing into TRAIN/SELECT/VAL/TEST with deterministic RNG seed.

    * Standardise X using TRAIN mean/std (z-score).
    * y is left in its original units by default.
    """

    # Basic validation
    if any(frac < 0.0 for frac in (train_frac, select_frac, val_frac, test_frac)):
        raise ValueError("Split fractions must be non-negative.")
    if train_frac + select_frac + val_frac + test_frac > 1.0 + 1e-12:
        raise ValueError("Split fractions must sum to <= 1.")

    local_path = _resolve_california_housing_local_path(data_root)

    X = None
    if local_path.exists():
        # Robust reader: accepts whitespace-separated or comma-separated
        df = pd.read_csv(
            local_path,
            header=None,
            sep=r"\s+|,",
            engine="python",
        )
        arr = df.to_numpy(dtype=float)

        # UCI California Housing: 8 features + 1 target = 9 columns
        if arr.ndim != 2 or arr.shape[1] < 9:
            raise ValueError(
                f"Unexpected cal_housing.data shape {arr.shape}; expected >= 9 columns (8 features + target). "
                f"Path: {local_path}"
            )

        X = arr[:, :8]
        # y is the last column in the standard UCI format
        y = arr[:, 8]
    else:
        raise RuntimeError(f"Cannot load California Housing data; local file not found at {local_path}.")

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    if X.ndim != 2:
        raise ValueError(f"Expected X to be 2D, got shape {X.shape}")
    if X.shape[1] != CALIFORNIA_HOUSING_D:
        raise ValueError(
            f"Expected California Housing to have d={CALIFORNIA_HOUSING_D} features, got {X.shape[1]}."
        )
    n = int(X.shape[0])
    if y.shape[0] != n:
        raise ValueError("X and y have inconsistent lengths.")

    sizes = california_housing_split_sizes(train_frac, select_frac, val_frac, test_frac)
    n_train, n_select, n_val, n_test = (
        sizes["n_train"], sizes["n_select"], sizes["n_val"], sizes["n_test"]
    )

    rng = np.random.default_rng(seed=int(replication))
    perm = rng.permutation(n)

    train_idx = perm[:n_train]
    select_idx = perm[n_train : n_train + n_select]
    val_idx = perm[n_train + n_select : n_train + n_select + n_val]
    test_idx = perm[n_train + n_select + n_val : n_train + n_select + n_val + n_test]

    X_train_raw = X[train_idx]
    x_mean = X_train_raw.mean(axis=0)
    x_std = X_train_raw.std(axis=0, ddof=0)
    x_std = np.where(x_std < 1e-12, 1.0, x_std)

    #X_std = (X - x_mean) / x_std
    X_std = X # do not standardise X at this stage
    X_train = X_std[train_idx]
    X_select = X_std[select_idx]
    X_val = X_std[val_idx]
    X_test = X_std[test_idx]

    if standardise_y:
        y_train_raw = y[train_idx]
        y_mean = float(np.mean(y_train_raw))
        y_std = float(np.std(y_train_raw, ddof=0))
        y_std = float(y_std if y_std >= 1e-12 else 1.0)

        y_out = (y - y_mean) / y_std
        y_train = y_out[train_idx]
        y_select = y_out[select_idx]
        y_val = y_out[val_idx]
        y_test = y_out[test_idx]
    else:
        y_mean = None
        y_std = None
        y_train = y[train_idx]
        y_select = y[select_idx]
        y_val = y[val_idx]
        y_test = y[test_idx]

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_select": X_select,
        "y_select": y_select,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "indices": {
            "train": train_idx,
            "select": select_idx,
            "val": val_idx,
            "test": test_idx,
        },
    }


class _EmpiricalMarginal:
    """
    Simple monotone empirical CDF / inverse-CDF pair using piecewise-linear interpolation.

    * CDF uses unique(sorted(x)) mapped to "mid-ranks" u in (0,1).
    * PPF uses linear interpolation in u-space.
    """
    def __init__(self, x_1d: np.ndarray):
        x = np.asarray(x_1d, dtype=float).reshape(-1)
        if x.size < 2:
            raise ValueError("Need at least two samples to build an empirical marginal.")

        xs = np.sort(x)
        n = int(xs.size)
        u = (np.arange(1, n + 1) - 0.5) / n  # in (0,1)

        xs_u, idx = np.unique(xs, return_index=True)
        u_u = u[idx]

        self._xs = xs_u
        self._u = u_u
        self._u_min = float(u_u[0])
        self._u_max = float(u_u[-1])

    def cdf(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        return np.interp(x_arr, self._xs, self._u, left=self._u_min, right=self._u_max)

    def ppf(self, u: np.ndarray) -> np.ndarray:
        u_arr = np.asarray(u, dtype=float)
        u_clip = np.clip(u_arr, self._u_min, self._u_max)
        return np.interp(u_clip, self._u, self._xs)


class GaussianCopula:
    """
    Gaussian copula fitted from data via:
      X -> U (empirical CDF per coordinate) -> Z = Phi^{-1}(U), then estimate cov(Z).
    """
    def __init__(self, X_train: np.ndarray, jitter: float = 1e-6):
        from scipy.stats import norm

        X = np.asarray(X_train, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"GaussianCopula expects a 2D array, got shape {X.shape}")
        n, d = X.shape
        if n < 3:
            raise ValueError("GaussianCopula needs at least 3 samples for a stable covariance estimate.")
        self.d = int(d)

        self.marginals = [_EmpiricalMarginal(X[:, j]) for j in range(self.d)]

        U = np.column_stack([self.marginals[j].cdf(X[:, j]) for j in range(self.d)])
        eps = 1.0 / (n + 1.0)  # avoid 0/1 which would map to +/- inf
        U = np.clip(U, eps, 1.0 - eps)

        Z = norm.ppf(U)
        Z = Z - Z.mean(axis=0, keepdims=True)

        from sklearn.covariance import GraphicalLassoCV
        gl = GraphicalLassoCV(cv=3, assume_centered=True).fit(Z)
        Sigma = gl.covariance_
        Sigma = Sigma + float(jitter) * np.eye(self.d)

        self.Sigma = Sigma

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        from scipy.stats import norm

        n = int(n)
        if n <= 0:
            raise ValueError("n must be positive.")
        z = rng.multivariate_normal(mean=np.zeros(self.d), cov=self.Sigma, size=n)
        u = norm.cdf(z)
        X = np.column_stack([self.marginals[j].ppf(u[:, j]) for j in range(self.d)])
        return X


class CopulaRidgeCentre:
    """
    Centre P_c: X ~ GaussianCopula; y|X ~ N(ridge(X), sigma_y^2).
    """
    def __init__(self, copula: GaussianCopula, w: np.ndarray, b: float, sigma_y: float):
        self.copula = copula
        self.w = np.asarray(w, dtype=float).reshape(-1)
        self.b = float(b)
        self.sigma_y = float(sigma_y)

    def sample_x(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return self.copula.sample(int(n), rng)

    def sample_xi(self, n: int, rng: np.random.Generator) -> np.ndarray:
        X = self.sample_x(int(n), rng)
        mean = X @ self.w + self.b
        y = mean + rng.normal(loc=0.0, scale=self.sigma_y, size=int(n))
        return np.concatenate([X, y.reshape(-1, 1)], axis=1)


def fit_copula_ridge_centre(
    X_train: np.ndarray,
    y_train: np.ndarray,
    ridge_alpha: float = 0.005,
    copula_jitter: float = 1e-6,
):
    """Fit the Gaussian copula + ridge centre P_c on TRAIN."""
    from sklearn.linear_model import Ridge

    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float).reshape(-1)
    if X.ndim != 2:
        raise ValueError(f"Expected X_train 2D, got {X.shape}")
    if y.shape[0] != X.shape[0]:
        raise ValueError("X_train and y_train have inconsistent lengths.")

    copula = GaussianCopula(X, jitter=copula_jitter)

    ridge = Ridge(alpha=float(ridge_alpha), fit_intercept=True)
    ridge.fit(X, y)
    w = np.asarray(ridge.coef_, dtype=float).reshape(-1)
    b = float(ridge.intercept_)

    resid = y - (X @ w + b)
    sigma_y = float(np.sqrt(np.mean(resid * resid))) if resid.size else 1e-8
    sigma_y = float(max(sigma_y, 1e-8))

    return CopulaRidgeCentre(copula, w, b, sigma_y)


def mahalanobis_scores(
    xi: np.ndarray,
    mu: np.ndarray,
    sqrt_Sigma: np.ndarray,
) -> np.ndarray:
    """
    Mahalanobis (ellipsoidal) score:
        s(xi) = || Sigma^{-1/2} (xi - mu) ||_2

    Computed via a Cholesky solve with sqrt_Sigma = chol(Sigma) (lower triangular),
    so Sigma = sqrt_Sigma @ sqrt_Sigma.T and Sigma^{-1/2}(·) is implemented as solve(sqrt_Sigma, ·).
    """
    X = np.asarray(xi, dtype=float)
    mu = np.asarray(mu, dtype=float).reshape(-1)
    L = np.asarray(sqrt_Sigma, dtype=float)

    if X.ndim == 1:
        X = X[None, :]
    if X.ndim != 2:
        raise ValueError("xi must be a 1D or 2D array.")
    if X.shape[1] != mu.size:
        raise ValueError(f"xi dimension {X.shape[1]} does not match mu dimension {mu.size}.")
    if L.shape != (mu.size, mu.size):
        raise ValueError(f"sqrt_Sigma must be {(mu.size, mu.size)}, got {L.shape}.")

    D = (X - mu).T                    # (p, n)
    Z = np.linalg.solve(L, D)         # (p, n), where L Z = D
    return np.sqrt(np.sum(Z * Z, axis=0))


def calibrate_dkw_ellipsoid_bulk_set(
    xi_train: np.ndarray,
    xi_select: np.ndarray,
    gamma: float,
    delta: float,
    ridge: float = 1e-8,
    max_cholesky_tries: int = 10,
):
    """
    Calibrate DKW ellipsoidal (Mahalanobis) bulk set:

      * mu, Sigma from TRAIN (mean + covariance)
      * t_hat from SELECT scores via DKW thresholding on Mahalanobis scores.

    The bulk set is:
        Xi0 = { xi : ||Sigma^{-1/2}(xi - mu)||_2 <= t_hat }.

    Returns:
      - mu: (p,)
      - Sigma: (p,p) SPD (ridge-stabilised)
      - sqrt_Sigma: (p,p) lower-triangular Cholesky factor of Sigma
      - t_hat: scalar threshold (DKW)
    """
    from .lv_bulk_set import dkw_select_threshold

    xi_tr = np.asarray(xi_train, dtype=float)
    xi_sel = np.asarray(xi_select, dtype=float)

    if xi_tr.ndim != 2 or xi_sel.ndim != 2:
        raise ValueError("xi_train and xi_select must be 2D arrays.")
    if xi_tr.shape[1] != xi_sel.shape[1]:
        raise ValueError("xi_train and xi_select must have the same dimension.")
    if xi_tr.shape[0] < 2 or xi_sel.shape[0] < 2:
        raise ValueError("Need at least 2 points in each split for DKW calibration.")

    p = int(xi_tr.shape[1])
    mu = xi_tr.mean(axis=0)

    Xc = xi_tr - mu
    # ddof=0 analogue: divide by n
    Sigma_emp = (Xc.T @ Xc) / float(xi_tr.shape[0])
    Sigma_emp = 0.5 * (Sigma_emp + Sigma_emp.T)

    # Stabilise to SPD and compute Cholesky with increasing jitter if needed
    base = Sigma_emp
    jitter = float(ridge)
    Sigma = None
    sqrt_Sigma = None
    for _ in range(int(max_cholesky_tries)):
        try:
            Sigma_try = base + jitter * np.eye(p)
            sqrt_Sigma_try = np.linalg.cholesky(Sigma_try)
            Sigma = Sigma_try
            sqrt_Sigma = sqrt_Sigma_try
            break
        except np.linalg.LinAlgError:
            jitter *= 10.0

    if Sigma is None or sqrt_Sigma is None:
        raise np.linalg.LinAlgError(
            "Failed to compute Cholesky factor for ellipsoidal bulk covariance. "
            f"Tried ridge up to {jitter:g}."
        )

    audit_scores = mahalanobis_scores(xi_sel, mu, sqrt_Sigma)
    dkw_info = dkw_select_threshold(audit_scores, gamma=float(gamma), delta=float(delta))

    if (not dkw_info.get("exists", False)) or (not np.isfinite(dkw_info.get("t_hat", np.nan))):
        r = dkw_info["r"]
        warnings.warn(
                f"DKW certificate for gamma={gamma}, delta={delta} does not exist; "
                f"using t_hat=max(score) (coverage margin r={r}).",
                UserWarning,
            )
        t_hat = float(np.max(audit_scores)) if audit_scores.size else 0.0
    else:
        t_hat = float(dkw_info["t_hat"])

    return {
        "mu": mu,
        "Sigma": Sigma,
        "sqrt_Sigma": sqrt_Sigma,
        "t_hat": t_hat,
        "audit_scores": audit_scores,
        "dkw_info": dkw_info,
    }

def _dkw_select_threshold_or_max(audit_scores: np.ndarray, *, gamma: float, delta: float, warn_prefix: str):
    """
    Helper: run repo DKW selector on a 1D score array; fall back to max(score) if the certificate fails.

    Returns:
      - t_hat: float
      - dkw_info: dict (as returned by the repo's dkw_select_threshold)
    """
    from .lv_bulk_set import dkw_select_threshold

    scores = np.asarray(audit_scores, dtype=float).reshape(-1)
    dkw_info = dkw_select_threshold(scores, gamma=float(gamma), delta=float(delta))

    if (not dkw_info.get("exists", False)) or (not np.isfinite(dkw_info.get("t_hat", np.nan))):
        r = dkw_info.get("r", np.nan)
        warnings.warn(
            f"{warn_prefix} DKW certificate for gamma={gamma}, delta={delta} does not exist; "
            f"using t_hat=max(score) (coverage margin r={r}).",
            UserWarning,
        )
        t_hat = float(np.max(scores)) if scores.size else 0.0
    else:
        t_hat = float(dkw_info["t_hat"])

    if (not np.isfinite(t_hat)) or (t_hat < 0.0):
        warnings.warn(
            f"{warn_prefix} Selected t_hat is invalid (t_hat={t_hat}); using t_hat=max(score).",
            UserWarning,
        )
        t_hat = float(np.max(scores)) if scores.size else 0.0

    return t_hat, dkw_info


def _cholesky_spd_from_train_cov(
    X: np.ndarray,
    *,
    ridge: float = 1e-8,
    max_cholesky_tries: int = 10,
    warn_prefix: str = "",
):
    """
    Compute an SPD covariance estimate (ddof=0 analogue) and its lower-triangular Cholesky factor.

    Returns:
      - Sigma: (p,p) SPD
      - sqrt_Sigma: (p,p) lower-triangular, Sigma = sqrt_Sigma @ sqrt_Sigma.T
    """

    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2D.")

    n = int(X.shape[0])
    if n < 2:
        raise ValueError("Need at least 2 points to compute covariance.")

    mu = X.mean(axis=0)
    Xc = X - mu

    Sigma_emp = (Xc.T @ Xc) / float(n)
    Sigma_emp = 0.5 * (Sigma_emp + Sigma_emp.T)

    base = Sigma_emp
    jitter = float(ridge)
    Sigma = None
    sqrt_Sigma = None
    for _ in range(int(max_cholesky_tries)):
        try:
            Sigma_try = base + jitter * np.eye(int(base.shape[0]))
            sqrt_Sigma_try = np.linalg.cholesky(Sigma_try)
            Sigma = Sigma_try
            sqrt_Sigma = sqrt_Sigma_try
            break
        except np.linalg.LinAlgError:
            jitter *= 10.0

    if Sigma is None or sqrt_Sigma is None:
        raise np.linalg.LinAlgError(
            f"{warn_prefix} Failed to compute Cholesky factor for covariance. Tried ridge up to {jitter:g}."
        )

    if not np.all(np.isfinite(sqrt_Sigma)):
        warnings.warn(f"{warn_prefix} Non-finite entries in sqrt_Sigma.", UserWarning)

    return Sigma, sqrt_Sigma


def calibrate_dkw_ellipsoid_x_interval_y_bulk_set(
    xi_train: np.ndarray,
    xi_select: np.ndarray,
    gamma: float,
    delta: float,
    ridge: float = 1e-8,
    max_cholesky_tries: int = 10,
) -> dict:
    """
    Geometry 4: ellipsoid in x AND independent interval in y (intersection; two thresholds).

    Scores on SELECT:
      - s_x,i = ||L_x^{-1}(x_i - mu_x)||_2
      - s_y,i = |y_i - mu_y|

    DKW uses 1D selection per score with a union bound split (gamma/2 each; delta/2 each).

    Bulk set:
        Xi0 = { (x,y): s_x <= t_x  and  s_y <= r_y }.
    """

    xi_tr = np.asarray(xi_train, dtype=float)
    xi_sel = np.asarray(xi_select, dtype=float)

    if xi_tr.ndim != 2 or xi_sel.ndim != 2:
        raise ValueError("xi_train and xi_select must be 2D arrays.")
    if xi_tr.shape[1] != xi_sel.shape[1]:
        raise ValueError("xi_train and xi_select must have the same dimension.")
    if xi_tr.shape[0] < 2 or xi_sel.shape[0] < 2:
        raise ValueError("Need at least 2 points in each split for DKW calibration.")

    p = int(xi_tr.shape[1])
    d = int(p - 1)

    X_tr = xi_tr[:, :d]
    y_tr = xi_tr[:, d].reshape(-1)

    X_sel = xi_sel[:, :d]
    y_sel = xi_sel[:, d].reshape(-1)

    mu_x = X_tr.mean(axis=0)
    mu_y = float(np.mean(y_tr))

    _, sqrt_Sigma_x = _cholesky_spd_from_train_cov(
        X_tr,
        ridge=float(ridge),
        max_cholesky_tries=int(max_cholesky_tries),
        warn_prefix="[DKW ell_x_int_y]",
    )

    # Mahalanobis scores in x: ||L^{-1}(x - mu_x)||_2
    Xc_sel = X_sel - mu_x.reshape(1, -1)
    Z = np.linalg.solve(sqrt_Sigma_x, Xc_sel.T).T
    audit_scores_x = np.linalg.norm(Z, axis=1)

    # Absolute deviation scores in y
    audit_scores_y = np.abs(y_sel - float(mu_y))

    gamma_half = float(gamma) / 2.0
    delta_half = float(delta) / 2.0

    t_x, dkw_info_x = _dkw_select_threshold_or_max(
        audit_scores_x,
        gamma=gamma_half,
        delta=delta_half,
        warn_prefix="[DKW ell_x_int_y:x]",
    )
    r_y, dkw_info_y = _dkw_select_threshold_or_max(
        audit_scores_y,
        gamma=gamma_half,
        delta=delta_half,
        warn_prefix="[DKW ell_x_int_y:y]",
    )

    if (not np.isfinite(r_y)) or (r_y < 0.0):
        warnings.warn(f"[DKW ell_x_int_y] Invalid r_y={r_y}; clipping to nonnegative.", UserWarning)
        r_y = float(max(0.0, r_y))

    return {
        "mu_x": np.asarray(mu_x, dtype=float),
        "mu_y": float(mu_y),
        "sqrt_Sigma_x": np.asarray(sqrt_Sigma_x, dtype=float),
        "t_x": float(t_x),
        "r_y": float(r_y),
        "audit_scores_x": audit_scores_x,
        "audit_scores_y": audit_scores_y,
        "dkw_info_x": dkw_info_x,
        "dkw_info_y": dkw_info_y,
    }


def in_ellipsoid_bulk(
    xi: np.ndarray,
    mu: np.ndarray,
    sqrt_Sigma: np.ndarray,
    t_hat: float,
    atol: float = 1e-10,
) -> np.ndarray:
    """Boolean mask of points in Xi0(t_hat) for the ellipsoidal (Mahalanobis) bulk set."""
    return mahalanobis_scores(xi, mu, sqrt_Sigma) <= float(t_hat) + float(atol)


def rejection_sample_centre_in_ellipsoid_bulk(
    centre: CopulaRidgeCentre,
    mu: np.ndarray,
    sqrt_Sigma: np.ndarray,
    t_hat: float,
    n_accept: int,
    rng: np.random.Generator,
    max_draws_factor: int = 5000,
):
    """
    Rejection sample exactly n_accept points xi ~ P_c conditioned on xi in Xi0 (ellipsoid).

    Returns: (xi_acc, total_draws, accept_rate)
    """
    n_accept = int(n_accept)
    if n_accept <= 0:
        raise ValueError("n_accept must be positive.")

    accepted = []
    n_acc = 0
    total_draws = 0
    max_draws = int(max_draws_factor) * n_accept

    while n_acc < n_accept and total_draws < max_draws:
        remaining = n_accept - n_acc
        batch_size = max(2 * remaining, 256)
        xi_batch = centre.sample_xi(batch_size, rng)
        mask = in_ellipsoid_bulk(xi_batch, mu, sqrt_Sigma, t_hat)
        if np.any(mask):
            acc = xi_batch[mask]
            accepted.append(acc)
            n_acc += int(acc.shape[0])
        total_draws += int(batch_size)

    if not accepted:
        raise RuntimeError(
            "Rejection sampling failed: no accepted samples inside Xi0 (ellipsoid). "
            "Pc might put extremely small mass on the bulk set."
        )

    xi_acc = np.concatenate(accepted, axis=0)
    if xi_acc.shape[0] < n_accept:
        idx = rng.choice(xi_acc.shape[0], size=n_accept - xi_acc.shape[0], replace=True)
        xi_acc = np.vstack([xi_acc, xi_acc[idx]])
    accept_rate = float(xi_acc.shape[0]) / float(max(total_draws, 1))

    if total_draws > n_accept * 2:
        warnings.warn(
            "Rejection sampling took more than twice n_accept; "
            "Pc may put very small mass on the bulk set.",
            UserWarning,
        )
    xi_acc = xi_acc[:n_accept]
    return xi_acc, total_draws, accept_rate



def empirical_cvar(losses: np.ndarray, tail_mass: float) -> float:
    """
    Empirical CVaR of losses at level 1 - tail_mass (i.e., mean of worst tail_mass fraction).
    """
    L = np.asarray(losses, dtype=float).reshape(-1)
    n = int(L.size)
    if n == 0:
        return float("nan")

    tail_mass = float(tail_mass)
    if tail_mass <= 0.0:
        return float(np.max(L))

    if tail_mass >= 1.0:
        return float(np.mean(L))

    k = int(np.ceil(tail_mass * n))
    k = max(1, min(k, n))
    # Mean of largest k values:
    thresh_idx = n - k
    part = np.partition(L, thresh_idx)
    tail = part[thresh_idx:]
    return float(np.mean(tail))


def california_housing_split_geographic(
    *,
    axis: str,
    seed: int,
    test_side: str = "west",
    standardise_y: bool = False,
    gap_ratio: float = 0.0,
    data_root: Optional[Path] = None,
) -> dict:

    local_path = _resolve_california_housing_local_path(data_root)

    X = None
    if local_path.exists():
        # Robust reader: accepts whitespace-separated or comma-separated
        df = pd.read_csv(
            local_path,
            header=None,
            sep=r"\s+|,",
            engine="python",
        )
        arr = df.to_numpy(dtype=float)

        # UCI California Housing: 8 features + 1 target = 9 columns
        if arr.ndim != 2 or arr.shape[1] < 9:
            raise ValueError(
                f"Unexpected cal_housing.data shape {arr.shape}; expected >= 9 columns (8 features + target). "
                f"Path: {local_path}"
            )

        X = arr[:, :8]
        # y is the last column in the standard UCI format
        y = arr[:, 8]
    else:
        raise FileNotFoundError(f"California Housing data file not found at {local_path}.")

    X_all = np.asarray(X, dtype=float)
    y_all = np.asarray(y, dtype=float).reshape(-1)

    if X.ndim != 2:
        raise ValueError(f"Expected X to be 2D, got shape {X.shape}")
    if X.shape[1] != CALIFORNIA_HOUSING_D:
        raise ValueError(
            f"Expected California Housing to have d={CALIFORNIA_HOUSING_D} features, got {X.shape[1]}."
        )
    n = int(X.shape[0])
    if y.shape[0] != n:
        raise ValueError("X and y have inconsistent lengths.")

    sizes = california_housing_split_sizes(CALIFORNIA_HOUSING_SPLIT_FRACS[0], CALIFORNIA_HOUSING_SPLIT_FRACS[1], CALIFORNIA_HOUSING_SPLIT_FRACS[2], CALIFORNIA_HOUSING_SPLIT_FRACS[3])
    n_train, n_select, n_val, n_test = (
        sizes["n_train"], sizes["n_select"], sizes["n_val"], sizes["n_test"]
    )

    N = int(X_all.shape[0])

    if N != (n_train + n_select + n_val + n_test):
        raise ValueError(
            "Unexpected split sizes: full dataset size does not match sum of splits. "
            f"N={N}, sum={n_train + n_select + n_val + n_test}."
        )

    gap_ratio = float(gap_ratio)
    if gap_ratio < 0.0:
        raise ValueError(f"gap_ratio must be >= 0, got {gap_ratio}")
    if gap_ratio >= 1.0:
        raise ValueError(f"gap_ratio must be < 1, got {gap_ratio}")

    n_gap = int(np.floor(gap_ratio * N))
    if n_gap < 0:
        n_gap = 0
    if n_gap >= n_test:
        raise ValueError(
            f"gap_ratio too large: floor(gap_ratio*N)={n_gap} must be < n_test={n_test}."
        )
    n_test_eff = int(n_test - n_gap)
    if n_test_eff <= 0:
        raise ValueError(
            f"gap_ratio too large: effective test size n_test_eff={n_test_eff} must be positive."
        )

    # ---- choose which coordinate defines the geographic region
    # normalise axis_key / side_key (keep prints for compatibility)
    axis_key = axis
    side_key = test_side
    if isinstance(axis, tuple) and len(axis) == 1:
        axis_key = axis[0]
    if isinstance(test_side, tuple) and len(test_side) == 1:
        side_key = test_side[0]

    axis_key = str(axis_key).strip().lower()
    side_key = str(side_key).strip().lower()

    # [Longitude, Latitude] are the first two columns.
    lon_idx = 0
    lat_idx = 1

    gap_idx = np.asarray([], dtype=int)

    degrees_train = None
    try:
        degrees_train = float(axis_key)
        if not np.isfinite(degrees_train):
            degrees_train = None
    except Exception:
        degrees_train = None

    if degrees_train is None:
        # Backwards compatible mapping to a training direction
        if axis_key in {"lat", "latitude"}:
            # latitude increases south -> north
            if side_key in {"south", "s", "low", "min"}:
                # TEST south => TRAIN north
                degrees_train = 90.0
            elif side_key in {"north", "n", "high", "max"}:
                # TEST north => TRAIN south
                degrees_train = 270.0
            else:
                raise ValueError("For axis='latitude', test_side must be one of {'south','north'}.")
        elif axis_key in {"lon", "longitude", "long"}:
            # west/coastal corresponds to more negative longitude
            if side_key in {"west", "w", "coastal", "coast", "low", "min"}:
                # TEST west => TRAIN east
                degrees_train = 0.0
            elif side_key in {"east", "e", "inland", "high", "max"}:
                # TEST east => TRAIN west
                degrees_train = 180.0
            else:
                raise ValueError("For axis='longitude', test_side must be one of {'west','east'}.")
        else:
            raise ValueError(
                "axis must be one of {'latitude','longitude'} OR a numeric degrees value "
                "(passed via `axis`) to define the training direction."
            )

    degrees_train = float(degrees_train) % 360.0
    theta = float(np.deg2rad(degrees_train))

    # Coordinate system for the angle:
    #   x_east  := longitude   (larger => more east)
    #   y_north := latitude    (larger => more north)
    x_east = np.asarray(X_all[:, lon_idx], dtype=float)
    y_north = np.asarray(X_all[:, lat_idx], dtype=float)
    # Projection score; high score => more aligned with TRAIN direction
    score = float(np.cos(theta)) * x_east + float(np.sin(theta)) * y_north
    order = np.argsort(score)  # low score => opposite direction => TEST
    test_idx = order[:n_test_eff]
    if n_gap > 0:
        gap_idx = order[n_test_eff : n_test_eff + n_gap]

    test_idx = np.asarray(test_idx, dtype=int)
    gap_idx = np.asarray(gap_idx, dtype=int)

    if test_idx.size != n_test_eff:
        raise RuntimeError("Internal error: test_idx does not have size n_test_eff.")
    if n_gap > 0 and gap_idx.size != n_gap:
        raise RuntimeError("Internal error: gap_idx does not have size n_gap.")

    # ---- training-region pool is the complement (exclude TEST and the gap)
    mask = np.ones(N, dtype=bool)
    mask[test_idx] = False
    if gap_idx.size > 0:
        mask[gap_idx] = False
    train_pool_idx = np.where(mask)[0]

    if train_pool_idx.size != (n_train + n_select + n_val):
        raise RuntimeError(
            "Internal error: training pool size mismatch. "
            f"got {train_pool_idx.size}, expected {n_train + n_select + n_val}."
        )

    # ---- random split within training region into train/select/val
    rng = np.random.default_rng(seed=int(seed))
    perm = rng.permutation(train_pool_idx)

    train_idx = perm[:n_train]
    select_idx = perm[n_train : n_train + n_select]
    val_idx = perm[n_train + n_select : n_train + n_select + n_val]

    # ---- assemble the new splits
    X_train_new = X_all[train_idx]
    y_train_new = y_all[train_idx]

    X_select_new = X_all[select_idx]
    y_select_new = y_all[select_idx]

    X_val_new = X_all[val_idx]
    y_val_new = y_all[val_idx]

    X_test_new = X_all[test_idx]
    y_test_new = y_all[test_idx]


    # ---- standardise using TRAIN only (no leakage)
    mu = X_train_new.mean(axis=0)
    sigma = X_train_new.std(axis=0, ddof=0)
    sigma = np.where(sigma > 0.0, sigma, 1.0)

    X_train_new = (X_train_new - mu) / sigma
    X_select_new = (X_select_new - mu) / sigma
    X_val_new = (X_val_new - mu) / sigma
    X_test_new = (X_test_new - mu) / sigma

    if standardise_y:
        y_train_raw = y_train_new
        y_mean = float(np.mean(y_train_raw))
        y_std = float(np.std(y_train_raw, ddof=0))
        y_std = float(y_std if y_std >= 1e-12 else 1.0)

        y_out = (y_all - y_mean) / y_std
        y_train_new = y_out[train_idx]
        y_select_new = y_out[select_idx]
        y_val_new = y_out[val_idx]
        y_test_new = y_out[test_idx]

    return {
        "X_train": X_train_new,
        "y_train": y_train_new,
        "X_select": X_select_new,
        "y_select": y_select_new,
        "X_val": X_val_new,
        "y_val": y_val_new,
        "X_test": X_test_new,
        "y_test": y_test_new,
    }

def _california_housing_geo_block_folds(
    *,
    X: np.ndarray,
    n_folds: int,
    seed: int,
    n_lon_bins: int = 6,
    n_lat_bins: int = 6,
) -> dict:
    """
    Construct geographically-blocked CV folds from a single region (e.g. EAST).

    IMPORTANT:
      - Longitude/latitude are assumed to be the FIRST TWO columns of X.
      - We form blocks by quantile-binning lon/lat into a grid of
            n_lon_bins x n_lat_bins
        then randomly assign *blocks* (not points) to folds.

    Returns:
      dict with:
        - "folds": list of {"train_idx": np.ndarray, "val_idx": np.ndarray}
        - "n_folds": effective number of folds used
        - "n_blocks": number of non-empty blocks
        - "n_lon_bins", "n_lat_bins": bin counts used
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2D (n, d).")
    n, d = X.shape
    if d < 2:
        raise ValueError("X must have at least 2 columns (lon, lat).")

    n_folds = int(n_folds)
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2.")

    n_lon_bins = int(n_lon_bins)
    n_lat_bins = int(n_lat_bins)
    if n_lon_bins < 1 or n_lat_bins < 1:
        raise ValueError("n_lon_bins and n_lat_bins must be positive.")

    lon = X[:, 0].astype(float)
    lat = X[:, 1].astype(float)

    def _make_edges(v: np.ndarray, n_bins: int) -> np.ndarray:
        v = np.asarray(v, dtype=float).reshape(-1)
        qs = np.linspace(0.0, 1.0, int(n_bins) + 1)
        edges = np.quantile(v, qs)

        # If quantiles collapse (e.g., constant coordinate), fall back to linear edges.
        if np.unique(edges).size < 3:
            vmin = float(np.min(v))
            vmax = float(np.max(v))
            if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
                edges = np.linspace(vmin, vmax, int(n_bins) + 1)
            else:
                edges = np.linspace(-1.0, 1.0, int(n_bins) + 1)

        edges = edges.astype(float)
        edges[0] = -np.inf
        edges[-1] = np.inf
        return edges

    lon_edges = _make_edges(lon, n_lon_bins)
    lat_edges = _make_edges(lat, n_lat_bins)

    # Bin indices in {0,...,n_bins-1}
    lon_bin = np.searchsorted(lon_edges[1:-1], lon, side="right").astype(int)
    lat_bin = np.searchsorted(lat_edges[1:-1], lat, side="right").astype(int)

    # Unique block id per (lon_bin, lat_bin)
    block_id = lon_bin * n_lat_bins + lat_bin
    blocks = np.unique(block_id)
    n_blocks = int(blocks.size)
    if n_blocks < 2:
        raise ValueError("Too few non-empty geographic blocks to build folds (need >= 2).")

    # Don't create too many folds (also can't exceed number of blocks)
    n_folds_eff = int(min(n_folds, n_blocks))
    if n_folds_eff < 2:
        raise ValueError("Not enough blocks for at least 2 folds.")

    rng = np.random.default_rng(seed=int(seed))
    blocks_shuf = rng.permutation(blocks)
    block_groups = np.array_split(blocks_shuf, n_folds_eff)

    fold_map = {}
    for k, grp in enumerate(block_groups):
        for b in grp:
            fold_map[int(b)] = int(k)

    fold_id = np.array([fold_map[int(b)] for b in block_id], dtype=int)

    folds = []
    for k in range(n_folds_eff):
        val_idx = np.where(fold_id == k)[0]
        train_idx = np.where(fold_id != k)[0]
        if val_idx.size == 0 or train_idx.size == 0:
            # Guard: should not happen if blocks are non-empty.
            continue
        folds.append({"train_idx": train_idx, "val_idx": val_idx})

    if len(folds) < 2:
        raise RuntimeError("Failed to construct at least 2 non-empty folds.")

    return {
        "folds": folds,
        "n_folds": int(len(folds)),
        "n_blocks": int(n_blocks),
        "n_lon_bins": int(n_lon_bins),
        "n_lat_bins": int(n_lat_bins),
    }


def calibrate_dkw_box_xi_bulk_set(
    xi_train: np.ndarray,
    xi_select: np.ndarray,
    gamma: float,
    delta: float,
    scale_floor: float = 1e-8,
) -> dict:
    """
    Geometry 2: full axis-aligned box in xi=(x,y) via a scaled l_infty score.

      TRAIN stats: mu (mean) and q (per-coordinate scale from std with floor).
      Score on SELECT: s_i = ||diag(1/q)(xi_i - mu)||_inf.
      DKW selects t_hat; half-widths are r = t_hat * q.

    Bulk set:
        Xi0 = { xi : |xi_j - mu_j| <= r_j for all j }.
    """

    xi_tr = np.asarray(xi_train, dtype=float)
    xi_sel = np.asarray(xi_select, dtype=float)

    if xi_tr.ndim != 2 or xi_sel.ndim != 2:
        raise ValueError("xi_train and xi_select must be 2D arrays.")
    if xi_tr.shape[1] != xi_sel.shape[1]:
        raise ValueError("xi_train and xi_select must have the same dimension.")
    if xi_tr.shape[0] < 2 or xi_sel.shape[0] < 2:
        raise ValueError("Need at least 2 points in each split for DKW calibration.")

    p = int(xi_tr.shape[1])
    d = int(p - 1)

    mu = xi_tr.mean(axis=0)
    q = xi_tr.std(axis=0, ddof=0)
    q = np.asarray(q, dtype=float).reshape(-1)
    q = np.where(q >= float(scale_floor), q, float(scale_floor))

    scaled = np.abs(xi_sel - mu.reshape(1, -1)) / q.reshape(1, -1)
    audit_scores = np.max(scaled, axis=1)

    t_hat, dkw_info = _dkw_select_threshold_or_max(
        audit_scores,
        gamma=float(gamma),
        delta=float(delta),
        warn_prefix="[DKW box_xi]",
    )

    r = float(t_hat) * q
    if np.any(~np.isfinite(r)) or np.any(r < 0.0):
        warnings.warn("[DKW box_xi] Non-finite or negative half-widths detected; clipping.", UserWarning)
        r = np.clip(r, 0.0, np.inf)

    return {
        # Generic
        "mu": mu,
        "q": q,
        "t_hat": float(t_hat),
        "r": r,
        # Parameters for the LV-BAS CVXPY problem
        "mu_x": np.asarray(mu[:d], dtype=float),
        "mu_y": float(mu[d]),
        "r_x": np.asarray(r[:d], dtype=float),
        "r_y": float(r[d]),
        # Audit and certificate info
        "audit_scores": audit_scores,
        "dkw_info": dkw_info,
    }