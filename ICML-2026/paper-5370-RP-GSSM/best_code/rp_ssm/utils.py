from functools import wraps

import jax.numpy as np
from jax import vmap
from jax.scipy.linalg import solve_triangular

import numpy as onp

from PIL import Image

from sklearn.feature_selection import r_regression
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression


def inv_quad_form_symmetric(S, m):
    """Compute m.T * inv(S) * m and log(det(S))"""
    L = np.linalg.cholesky(S)
    x = solve_triangular(L, m, lower=True)
    logdet = 2 * np.sum(np.log(np.diag(L)))
    return x.T @ x, logdet


def inv_quad_form(S, m1, m2):
    """Compute m1.T * inv(S) * m2"""
    L = np.linalg.cholesky(S)
    x1 = solve_triangular(L, m1, lower=True)
    x2 = solve_triangular(L, m2, lower=True)
    return x1.T @ x2


def spd_inverse(S):
    """Invert a spd matrix via Cholesky"""
    L = np.linalg.cholesky(S)
    Linv = solve_triangular(L, np.eye(S.shape[-1]), lower=True)
    return Linv.T @ Linv


def logdet(S):
    L = np.linalg.cholesky(S)
    logdet = 2 * np.sum(np.log(np.diag(L)))
    return logdet


def scale_sv(A, eps):
    """
    Given a matrix A, scale its singular values to <=1-EPS.
    Only do so if there is a singular value already >1-EPS.

    Intuitively, this should work better than clipping
    if all singular values are relatively close to 1, as it
    preserves the relative sizes between singular values.
    If there is a single singular value that is very large,
    this will push the others to 0, so clipping may be
    more appropriate.
    """
    _, s, _ = np.linalg.svd(A)
    scale = np.maximum(np.max(s), 1 - eps) / (1 - eps)
    return A / scale


def clip_sv(A, eps):
    """
    Clip the SVs of a matrix to be less than 1-EPS.
    """
    u, s, vt = np.linalg.svd(A)
    return u @ np.diag(np.clip(s, 0.0, 1.0 - eps)) @ vt


def auto_vmap(param_name, expected_ndim):
    """
    Augmented function wrapper that acts as vmap, but automatically
    computes the number of vmaps that need to be applied, by
    computing the difference between the expected input dimension
    (`expected_ndim`) and the actual input dimension.

    The number of extra dimensions is taken from a designated param
    given by `param_name`. E.g. if the input is a Gaussian distparam,
    can use `param_name=mean` and `expected_ndim=1`, because the
    precision-weighted mean is a 1D array.
    """

    def decorator(f):
        @wraps(f)
        def wrapper(x):
            extra_dims = x.params[param_name].ndim - expected_ndim

            vmapped_f = f
            for _ in range(extra_dims):
                vmapped_f = vmap(vmapped_f)

            return vmapped_f(x)

        return wrapper

    return decorator


def linear_r2(x, y):
    """Compute linear regression R^2 between x, y"""
    x_reshaped = np.reshape(x, (-1, x.shape[-1]))
    y_reshaped = np.reshape(y, (-1, y.shape[-1]))
    linreg = LinearRegression()
    linreg.fit(x_reshaped, y_reshaped)
    return linreg.score(x_reshaped, y_reshaped), linreg


def krr_r2(x, y):
    """
    Compute kernel ridge regression score between x, y.
    Use RBF kernel with default hyperparameters.
    """
    x_reshaped = np.reshape(x, (-1, x.shape[-1]))
    y_reshaped = np.reshape(y, (-1, y.shape[-1]))
    krr = KernelRidge(kernel="rbf")
    krr.fit(x_reshaped, y_reshaped)
    return krr.score(x_reshaped, y_reshaped), krr


def linear_corr(x, y):
    """Compute linear correlation coefficient between x, y"""
    x_reshaped = np.reshape(x, (-1, x.shape[-1]))
    y_reshaped = np.reshape(y, (-1, y.shape[-1]))
    coef = r_regression(x_reshaped, y_reshaped.ravel())[0]
    return coef


def linear_predict(x, y, cov=None):
    """
    Apply a linear transformation to match x to y.
    Optionally pass a cov array of covariance
    matrices that will also get transformed.
    """
    x_reshaped = np.reshape(x, (-1, x.shape[-1]))
    y_reshaped = np.reshape(y, (-1, y.shape[-1]))
    linreg = LinearRegression()
    linreg.fit(x_reshaped, y_reshaped)
    transformed_means = linreg.predict(x_reshaped).reshape(
        x.shape[:-1] + (y.shape[-1],)
    )
    if cov is None:
        return transformed_means
    else:
        transformed_covs = linreg.coef_ @ cov @ linreg.coef_.T
        return transformed_means, transformed_covs


def krr_predict(x, y):
    """
    Match x to y via kernel ridge regression with default RBF kernel.
    """
    x_reshaped = np.reshape(x, (-1, x.shape[-1]))
    y_reshaped = np.reshape(y, (-1, y.shape[-1]))
    krr = KernelRidge(kernel="rbf")
    krr.fit(x_reshaped, y_reshaped)
    transformed_means = krr.predict(x_reshaped).reshape(x.shape[:-1] + (y.shape[-1],))
    return transformed_means


def fig2img(fig):
    """
    Transform matplotlib.pyplot.figure object
    into PIL.Image object suitable for wandb.
    """
    fig.canvas.draw()
    image_array = onp.frombuffer(fig.canvas.tostring_argb(), dtype=onp.uint8)
    width, height = fig.canvas.get_width_height()
    image_array = image_array.reshape((height, width, 4))
    image_array = image_array[:, :, [1, 2, 3]]
    img = Image.fromarray(image_array)
    return img