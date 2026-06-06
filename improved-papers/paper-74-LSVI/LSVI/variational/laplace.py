from typing import Callable

import jax
import jax.numpy as jnp
import scipy
from jax.typing import ArrayLike


def newton_descent(loss: Callable, init: ArrayLike):
    """
    Newton descent algorithm.
    """

    def update(x):
        grad = jax.grad(loss)(x)
        hess = jax.hessian(loss)(x)
        x = x - jax.scipy.linalg.solve(hess, grad)
        return x

    x = jax.lax.fori_loop(0, 100, lambda i, x: update(x), init)
    return x, jnp.linalg.pinv(jax.hessian(loss)(x))


def bfgs_wrapper(loss: Callable, init: ArrayLike):
    res = scipy.optimize.minimize(loss, init, method="BFGS")
    return res.x, res.hess_inv


def laplace_approximation(log_density: Callable, init: ArrayLike, optimization_method=bfgs_wrapper):
    """
    Compute the Laplace approximation of a density.
    """

    def loss(theta):
        return -log_density(theta)

    x, hess_inv = optimization_method(loss, init)
    return -log_density(x), x, hess_inv
