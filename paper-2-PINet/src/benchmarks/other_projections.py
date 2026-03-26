"""Projection layers using other approaches."""

from typing import Callable

import cvxpy as cp
import jax
import jax.numpy as jnp
import jaxopt
from cvxpylayers.jax import CvxpyLayer

jax.config.update("jax_enable_x64", True)


def get_jaxopt_projection(
    A: jnp.ndarray, C: jnp.ndarray, d: jnp.ndarray, dim: int, tol=1e-3
) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Compute a batched projection function for polyhedral constraints using JAXopt.

    This function creates a projection operator using the jaxopt.OSQP solver.
    The projection is formulated as the quadratic program:
    minimize   (1/2) * ||x - xx||^2
    subject to A x = b
               C x <= d,
    where the quadratic term is given by the identity matrix of size `dim`.

    The resulting function is JIT-compiled and vectorized.

    Args:
        A (jnp.ndarray): Coefficient matrix for equality constraints.
        C (jnp.ndarray): Coefficient matrix for inequality constraints.
        d (jnp.ndarray): Right-hand side vector for inequality constraints.
        dim (int): Dimension of the variable x.
        tol (float, optional): Tolerance for the solver. Defaults to 1e-3.

    Returns:
        Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
        A JIT-compiled and vectorized function
        that takes a batch of input vectors (shape: (batch_size, dim))
        and returns their corresponding projections.
    """
    qp = jaxopt.OSQP(tol=tol)
    Q = jnp.eye(dim)
    jaxopt_proj = jax.jit(
        jax.vmap(
            lambda xx, bb: qp.run(
                params_obj=(Q, -xx), params_eq=(A, bb[:, 0]), params_ineq=(C, d)
            ).params.primal,
            in_axes=[0, 0],
        )
    )

    return jaxopt_proj


def get_cvxpy_projection(
    A: jnp.ndarray,
    C: jnp.ndarray,
    d: jnp.ndarray,
    dim: int,
) -> Callable[[jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray]]:
    """Constructs and returns a CVXPY-based projection layer callable.

    The projection is formulated as a quadratic minimization problem that minimizes
    the squared distance between the projection variable and an input point xproj, subject
    to the constraints:
        A @ y = b   (equality constraints)
        C @ y <= d  (inequality constraints)

    Args:
        A (jnp.ndarray): Coefficient matrix for equality constraints.
        C (jnp.ndarray): Coefficient matrix for inequality constraints..
        d (jnp.ndarray): Right-hand side vector for inequality constraints.
        dim (int): Dimension of the variable x.

    Returns:
        Callable[[jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray]]:
        A callable CVXPY layer that takes two parameters:
        an input vector (xproj) to be projected and a corresponding vector b for
        the equality constraints.
        The callable returns the projected vector as a jnp.ndarray.
    """
    n_eq = A.shape[0]
    ycvxpy = cp.Variable(dim)
    xproj = cp.Parameter(dim)
    b = cp.Parameter(n_eq)
    constraints = [
        A @ ycvxpy == b,
        C @ ycvxpy <= d,
    ]
    objective = cp.Minimize(cp.sum_squares(ycvxpy - xproj))
    problem_cvxpy = cp.Problem(objective=objective, constraints=constraints)
    assert problem_cvxpy.is_dpp()

    cvxpylayer = CvxpyLayer(
        problem_cvxpy,
        parameters=[xproj, b],
        variables=[ycvxpy],
    )

    return cvxpylayer
