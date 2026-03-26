"""Affine inequality constraint module."""

from jax import numpy as jnp

from pinet.dataclasses import ProjectionInstance

from .base import Constraint


class AffineInequalityConstraint(Constraint):
    """Affine inequality constraint set.

    The (affine) inequality constraint set is defined as:
    lb <= C @ x <= ub
    where the matrix C and the vectors lb and ub are the parameters.
    """

    def __init__(self, C: jnp.ndarray, lb: jnp.ndarray, ub: jnp.ndarray) -> None:
        """Initialize the affine inequality constraint.

        Args:
            C (jnp.ndarray): The matrix C in the inequality.
                Shape (batch_size, n_constraints, dimension).
            lb (jnp.ndarray): The lower bound in the inequality.
                Shape (batch_size, n_constraints, 1).
            ub (jnp.ndarray): The upper bound in the inequality.
                Shape (batch_size, n_constraints, 1).
        """
        self.C = C
        self.lb = lb
        self.ub = ub

        # Check if batch sizes for C and l are consistent.
        # They should either be the same, or one of them should be 1.
        assert (
            self.C.shape[0] == self.lb.shape[0]
            or self.C.shape[0] == 1
            or self.lb.shape[0] == 1
        ), f"Batch sizes are inconsistent: C{self.C.shape}, l{self.lb.shape}"

        # Check if batch sizes for C and u are consistent.
        # They should either be the same, or one of them should be 1.
        assert (
            self.C.shape[0] == self.ub.shape[0]
            or self.C.shape[0] == 1
            or self.ub.shape[0] == 1
        ), f"Batch sizes are inconsistent: C{self.C.shape}, ub{self.ub.shape}"

        assert (
            self.C.shape[1] == self.lb.shape[1]
        ), "Number of rows in C must equal size of l."
        assert (
            self.C.shape[1] == self.ub.shape[1]
        ), "Number of rows in C must equal size of u."

    def project(self, inp: ProjectionInstance) -> ProjectionInstance:
        """Project x onto the affine inequality constraint set.

        Args:
            inp (ProjectionInstance): ProjectionInstance to projection.
                The .x attribute is the point to project.

        Returns:
            ProjectionInstance: The projected point for each point in the batch.
                Shape (batch_size, dimension, 1).
        """
        raise NotImplementedError(
            "The 'project' method is not implemented and should not be called."
        )

    @property
    def dim(self) -> int:
        """Return the dimension of the constraint set."""
        return self.C.shape[-1]

    @property
    def n_constraints(self) -> int:
        """Return the number of constraints."""
        return self.C.shape[1]

    def cv(self, inp: ProjectionInstance) -> jnp.ndarray:
        """Compute the constraint violation.

        Args:
            inp (ProjectionInstance): ProjectionInstance to evaluate.

        Returns:
            jnp.ndarray: The constraint violation for each point in the batch.
                Shape (batch_size, 1, 1).
        """
        Cx = self.C @ inp.x
        cv_ub = jnp.maximum(Cx - self.ub, 0)
        cv_lb = jnp.maximum(self.lb - Cx, 0)
        return jnp.max(jnp.maximum(cv_ub, cv_lb), axis=1, keepdims=True)
