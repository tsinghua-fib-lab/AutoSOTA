"""
von Mises distribution utilities

"""
__date__ = "January - February 2024"


from jax import jit
import jax.numpy as jnp
from jax.scipy.special import i0e, i1e



@jit
def vm_entropy(kappa):
    """Evaluate the vM entropy given κ."""
    return -kappa * i1e(kappa) / i0e(kappa) + jnp.log(2 * jnp.pi * i0e(kappa)) + kappa


@jit
def vm_log_pdf(a, b, x):
    """Evaluate the vM log pdf given κcos(μ), κsin(μ), and x."""
    kappa = jnp.sqrt(a**2 + b**2)
    return -jnp.log(2 * jnp.pi * i0e(kappa)) - kappa + a * jnp.cos(x) + b * jnp.sin(x)


@jit
def nu(kappa):
    """Calculate the absolute value of the mean of a vM distribution."""
    return i1e(kappa) / i0e(kappa)


@jit
def expanded_complex_nu(arr):
    assert arr.shape[-1] == 2
    res = complex_nu(arr[...,0] + 1j * arr[...,1])
    return jnp.stack([res.real, res.imag], -1)


@jit
def complex_nu(arr):
    return jnp.exp(1j * jnp.angle(arr)) * nu(jnp.abs(arr))


@jit
def estimate_kappa_fixed(means):
    """
    Approximate kappa from a vector of mean values.

    Use a fixed rational function approximation.
    """
    a1, a3, a5 = 2.171734789260238, -1.760730760853115, 0.5759380948510845
    x1, x2, x3, x5 = means, means**2, means**3, means**5
    return (a1 * x1 + a3 * x3 + a5 * x5) / (1.0 - x2)


def estimate_kappa_iterative(means, tol=1e-2, max_iter=20):
    """
    Approximate kappa from a vector of mean values.

    Use the exact iterative scheme.
    """
    kappa = means * (2 - means**2) / (1 - means**2)
    diff = jnp.inf
    itr = 0
    while diff > tol:
        nu_k = nu(kappa)
        corr = (nu_k - means) / (1.0 - nu_k**2 - nu_k / kappa)
        diff = jnp.max(jnp.abs(corr))
        kappa = kappa - corr
        itr += 1
        if itr == max_iter:
            break
    return kappa

