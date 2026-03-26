"""Module for the Alternating Direction Method of Multipliers (ADMM) solver."""

from typing import Callable

import jax.numpy as jnp

from pinet.constraints import (
    AffineInequalityConstraint,
    BoxConstraint,
    ConstraintParser,
    EqualityConstraint,
)
from pinet.dataclasses import ProjectionInstance


def initialize(
    yraw: jnp.ndarray,
    ineq_constraint: AffineInequalityConstraint,
    box_constraint: BoxConstraint,
    dim: int,
    dim_lifted: int,
    d_r: jnp.ndarray,
) -> ProjectionInstance:
    """Initialize the ADMM solver state.

    Args:
        yraw (jnp.ndarray): Point to be projected. Shape (batch_size, dimension, 1).
        ineq_constraint (AffineInequalityConstraint): Inequality constraint.
        box_constraint (BoxConstraint): Box constraint.
        dim (int): Dimension of the original problem.
        dim_lifted (int): Dimension of the lifted problem.
        d_r (jnp.ndarray): Scaling factor for the lifted dimension.

    Returns:
        ProjectionInstance: Initial state for the ADMM solver.
    """
    # Preprocess
    if yraw.eq is not None:
        if yraw.eq.A is not None:
            # Lift the equality constraint
            parser = ConstraintParser(
                eq_constraint=EqualityConstraint(yraw.eq.A, yraw.eq.b, method="pinv"),
                ineq_constraint=ineq_constraint,
                box_constraint=box_constraint,
            )
            lifted_eq_constraint, _, _ = parser.parse(method="pinv")
            yraw = yraw.update(
                eq=yraw.eq.update(
                    A=lifted_eq_constraint.A, Apinv=lifted_eq_constraint.Apinv
                )
            )

        if yraw.eq.b is not None:
            b_lifted = (
                jnp.concatenate(
                    [
                        yraw.eq.b,
                        jnp.zeros(shape=(yraw.eq.b.shape[0], dim_lifted - dim, 1)),
                    ],
                    axis=1,
                )
                * d_r
            )
            yraw = yraw.update(eq=yraw.eq.update(b=b_lifted))

    # Return updated value
    return yraw.update(x=jnp.zeros((yraw.x.shape[0], dim_lifted, 1)))


def build_iteration_step(
    eq_constraint: EqualityConstraint,
    box_constraint: BoxConstraint,
    dim: int,
    scale: jnp.ndarray = 1.0,
) -> tuple[
    Callable[[ProjectionInstance, jnp.ndarray, float, float], ProjectionInstance],
    Callable[[ProjectionInstance], jnp.ndarray],
]:
    """Build the iteration and result retrieval step for the ADMM solver.

    See https://web.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf for details.
    Args:
        eq_constraint (EqualityConstraint): (Lifted) Equality constraint.
        box_constraint (BoxConstraint): (Lifted) Box constraint.
        dim (int): Dimension of the original problem.
        scale (jnp.ndarray): Scaling of primal variables.

    Returns:
        tuple[
            Callable[[ProjectionInstance, jnp.ndarray, float, float], ProjectionInstance],
            Callable[[ProjectionInstance], ProjectionInstance]
        ]:
            The first element is the iteration step,
            the second element is the result retrieval step.
    """

    def iteration_step(
        sk: ProjectionInstance,
        yraw: ProjectionInstance,
        sigma: float = 1.0,
        omega: float = 1.7,
    ) -> ProjectionInstance:
        """One iteration of the ADMM solver.

        Args:
            sk (ProjectionInstance): State iterate for the ADMM solver.
                .x of Shape (batch_size, lifted_dimension, 1).
            yraw (ProjectionInstance):
                Point to be projected. .x of Shape (batch_size, dimension, 1).
            sigma (float, optional): ADMM parameter.
            omega (float, optional): ADMM parameter.

        Returns:
            jnp.ndarray: Next state iterate of the ADMM solver.
        """
        zk = eq_constraint.project(sk)
        # Reflection
        reflect = 2 * zk.x - sk.x
        tobox = jnp.concatenate(
            (
                (2 * sigma * scale * yraw.x + reflect[:, :dim, :])
                / (1 + 2 * sigma * scale**2),
                reflect[:, dim:, :],
            ),
            axis=1,
        )
        tk = box_constraint.project(sk.update(x=tobox))
        sk = sk.update(x=sk.x + omega * (tk.x - zk.x))
        return sk

    # The second element is used to extract the projection from the auxiliary
    return (iteration_step, lambda y: eq_constraint.project(y))
