# TODO
from __future__ import annotations
from typing import Callable, Any

import pytest

from jax import numpy as jnp
import jax


def pytest_addoption(parser):
    parser.addoption(
        '--norm_tol', action='store', default=1e-3, type=float,
        help="Numerical tolerance for checking whether a "
             "discrete distribution normalizes. This is used as, "
             "`(sum(q) - 1.0) < norm_tol`."
    )
    parser.addoption(
        '--trust_region_tol', action='store', default=1e-3, type=float,
        help="Numerical tolerance for checking whether a "
             "distribution satisfies a trust-region constraint. "
             "This is used as, "
             "`f_divergence(q_star, prior) < trust_region_tol`."
    )
    parser.addoption(
        '--trust_region_strictness', action='store',
        choices=["interior", "vertex"], default='interior', type=str,
        help="Strictness of the test to see whether the solution sufficiently "
             "maximizes its objective within the Trust-Region constraint. "
             "If `interior` then we only check if the divergences of the "
             "are less than epsilon. If `vertex` then we "
             "assert whether the solution lies on the vertex of the feasible "
             "region, though this is often in practice too strict."
    )
    parser.addoption(
        '--multiplier_resolution', action='store', default=50, type=int,
        help="Number of grid-points to test proposal distribution solutions "
             "for. Choosing a higher value gives a stronger guarantee for "
             "stability during testing, but takes longer to run. "
    )


@pytest.fixture
def prior(request: Any) -> tuple[str, Callable[[jax.Array], jax.Array]]:
    if request.param == 'gaussian':
        return request.param, request.getfixturevalue('gaussian_prior')
    elif request.param == 'uniform':
        return request.param, request.getfixturevalue('uniform_prior')
    elif request.param == 'mixture':
        return request.param, request.getfixturevalue('gmm_prior')


@pytest.fixture
def function(request: Any) -> tuple[str, Callable[[jax.Array], jax.Array]]:
    if request.param == 'sine':
        return request.param, request.getfixturevalue('sine_function')
    elif request.param == 'uniform':
        return request.param, request.getfixturevalue('uniform_function')
    elif request.param == 'uniform_but_one':
        return (
            request.param,
            request.getfixturevalue('uniform_but_one_function')
        )
    elif request.param == 'step':
        return request.param, request.getfixturevalue('step_function')


@pytest.fixture
def gaussian_prior():
    # Gaussian + uniform mixture to ensure f(xs) > w/len(xs) \forall xs

    def prior(
            xs: jax.Array,
            mu: float = 0.0, scale: float = 0.5,
            w: float = 0.05
    ) -> jax.Array:
        v = jax.scipy.stats.norm.pdf(xs, loc=mu, scale=scale)
        unif = jnp.ones_like(v) / (xs.max() - xs.min())
        return v * (1 - w) + w * unif

    return prior


@pytest.fixture
def gmm_prior():
    # GMM + uniform mixture to ensure f(xs) > w/len(xs) \forall xs

    def prior(
            xs: jax.Array,
            mus: float = jnp.asarray([-1.0, 0.0, 1.0]),
            scales: float = jnp.asarray([0.3, 0.5, 0.2]),
            w: float = 0.05
    ) -> jax.Array:

        v = jax.vmap(
            jax.scipy.stats.norm.pdf,
            in_axes=(None, 0, 0)
        )(xs, mus, scales)

        v = v.mean(axis=0)
        unif = jnp.ones_like(v) / (xs.max() - xs.min())

        return v * (1 - w) + w * unif

    return prior


@pytest.fixture
def uniform_prior() -> Callable[[jax.Array], jax.Array]:

    def f(xs: jax.Array) -> jax.Array:
        return jnp.ones_like(xs) / (xs.max() - xs.min())

    return f


@pytest.fixture
def sine_function() -> Callable[[jax.Array], jax.Array]:
    # Manually tuned wave function

    def f(x: jax.Array) -> jax.Array:
        return 1e3 * jnp.square(
            .09 + jnp.cos(2*x) / (x**2 / 5 + 10)
        ) * jnp.sin(x)**2 - 2 + x

    return f


@pytest.fixture
def uniform_function() -> Callable[[jax.Array], jax.Array]:

    def f(x: jax.Array) -> jax.Array:
        return jnp.ones_like(x) / x.size

    return f


@pytest.fixture
def uniform_but_one_function() -> Callable[[jax.Array], jax.Array]:

    def f(x: jax.Array) -> jax.Array:
        z = jnp.ones_like(x)
        z = z.at[z.size // 2].set(z.size)
        return z / z.sum()

    return f


@pytest.fixture
def step_function() -> Callable[[jax.Array], jax.Array]:

    def f(x: jax.Array) -> jax.Array:
        return jnp.sign(jnp.sin(x - jnp.pi/2))

    return f
