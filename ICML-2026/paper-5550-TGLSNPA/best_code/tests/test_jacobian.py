"""
Test the Jacobian term in the loss function.

"""

__date__ = "February 2023"

import sys
from os.path import abspath, join, dirname
sys.path.append(abspath(join(dirname(__file__), '..')))

import jax.numpy as jnp
from jax import vjp, jacfwd

from src.stats import flat_stats


if __name__ == "__main__":
    d = 10
    x = 0.5 * jnp.arange(d)
    phi = 0.5 * jnp.arange(2 * d**2)
    stats = flat_stats(x)

    # Option 1:
    jac = jacfwd(flat_stats)(x)  # [2d^2, d]
    res1 = jac.T @ phi
    print("res1", res1.shape)
    print(res1[:5])

    # Option 2:
    _, f_vjp = vjp(flat_stats, x)
    (jac_term,) = f_vjp(phi)
    print("jac_term", jac_term.shape)
    print(jac_term[:5])

    print("diff:", jnp.linalg.norm(res1 - jac_term))


###
