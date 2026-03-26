"""Equality constraint module."""

from typing import Optional

import jax.numpy as jnp

from pinet.dataclasses import ProjectionInstance

from .base import Constraint


class EqualityConstraint(Constraint):
    """Equality constraint set.

    The (affine) equality constraint set is defined as:
    A @ x == b
    where the matrix A and the vector b are the parameters.
    It might be worth to consider masking, so that the
    constraint acts only on a subset of dimensions.
    """

    def __init__(
        self,
        A: jnp.ndarray,
        b: jnp.ndarray,
        method: Optional[str] = "pinv",
        var_b: Optional[bool] = False,
        var_A: Optional[bool] = False,
    ) -> None:
        """Initialize the equality constraint.

        Args:
            A (jnp.ndarray): Left hand side matrix.
                Shape (batch_size, n_constraints, dimension).
            b (jnp.ndarray): Right hand side vector.
                Shape (batch_size, n_constraints, 1).
            method (str): A string that specifies the method used to solve
                linear systems. Valid methods are "pinv", and None.
            var_b (bool): Boolean that indicates whether the b vector
                changes or is constant.
            var_A (bool): Boolean that indicates whether the A matrix
                changes or is constant.
        """
        assert A is not None, "Matrix A must be provided."
        assert b is not None, "Vector b must be provided."

        self.A = A
        self.b = b
        self.method = method
        self.var_b = var_b
        self.var_A = var_A
        self.setup()

    def setup(self) -> None:
        """Sets up the equality constraint."""
        assert (
            self.A.ndim == 3
        ), "A is a matrix with shape (batch_size, n_constraints, dimension)."
        assert (
            self.b.ndim == 3
        ), "b is a matrix with shape (batch_size, n_constraints, 1)."
        assert self.b.shape[2] == 1, "b must have shape (batch_size, n_constraints, 1)."

        # Check if batch sizes are consistent.
        # They should either be the same, or one of them should be 1.
        assert (
            self.A.shape[0] == self.b.shape[0]
            or self.A.shape[0] == 1
            or self.b.shape[0] == 1
        ), f"Batch sizes are inconsistent: A{self.A.shape}, b{self.b.shape}"

        assert (
            self.A.shape[1] == self.b.shape[1]
        ), "Number of rows in A must equal size of b."

        # List of valid methods
        valid_methods = ["pinv", None]

        if self.method == "pinv":
            if not self.var_A:
                self.Apinv = self.Apinv = jnp.linalg.pinv(self.A)

            self.project = self.project_pinv
        elif self.method is None:

            def raise_not_implemented_error():
                raise NotImplementedError("No projection method set.")

            self.project = lambda *args: raise_not_implemented_error()
        else:
            raise Exception(
                f"Invalid method {self.method}. Valid methods are: {valid_methods}"
            )

    def get_params(
        self, inp: ProjectionInstance
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Get A, b, Apinv depending on varying constraints.

        Args:
            inp (ProjectionInstance): ProjectionInstance to get parameters from.

        Returns:
            tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]: Tuple containing
                b (jnp.ndarray): Right hand side vector.
                    Shape (batch_size, n_constraints, 1).
                A (jnp.ndarray): Left hand side matrix.
                    Shape (batch_size, n_constraints, dimension).
                Apinv (jnp.ndarray): Pseudo-inverse of A.
                    Shape (batch_size, n_constraints, dimension).
        """
        b = inp.eq.b if inp.eq and inp.eq.b is not None else self.b
        A = inp.eq.A if inp.eq and self.var_A else self.A
        Apinv = inp.eq.Apinv if inp.eq and self.var_A else self.Apinv

        return b, A, Apinv

    def project_pinv(self, yraw: ProjectionInstance) -> ProjectionInstance:
        """Project onto equality constraints using pseudo-inverse.

        Args:
            yraw (ProjectionInstance): ProjectionInstance to projection.
                The .x attribute is the point to project.

        Returns:
            ProjectionInstance: The projected point for each point in the batch.
        """
        b, A, Apinv = self.get_params(yraw)

        return yraw.update(x=yraw.x - Apinv @ (A @ yraw.x - b))

    @property
    def dim(self) -> int:
        """Return the dimension of the constraint set."""
        return self.A.shape[-1]

    @property
    def n_constraints(self) -> int:
        """Return the number of constraints."""
        return self.A.shape[1]

    def cv(self, inp: ProjectionInstance) -> jnp.ndarray:
        """Compute the constraint violation.

        Args:
            inp (ProjectionInstance): ProjectionInstance to evaluate.

        Returns:
            jnp.ndarray: The constraint violation for each point in the batch.
                Shape (batch_size, 1, 1).
        """
        b, A, _ = self.get_params(inp)

        return jnp.linalg.norm(A @ inp.x - b, ord=jnp.inf, axis=1, keepdims=True)
