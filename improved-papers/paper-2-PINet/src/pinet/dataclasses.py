"""This file contains dataclasses used to encapsulate inputs for the Pinet layer."""

from dataclasses import dataclass, replace
from typing import Optional

import jax
import jax.numpy as jnp


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class EqualityConstraintsSpecification:
    """Dataclass representing inputs used in forming equality constraints.

    Attributes:
        b (Optional[jnp.ndarray]): Vector representing the RHS of the equality constraint.
            Shape (batch_size, n_constraints, 1)
        A (Optional[jnp.ndarray]): Matrix representing the LHS of the equality constraint.
            Shape (batch_size, n_constraints, dimension).
        Apinv (Optional[jnp.ndarray]): The pseudoinverse of the matrix A.
            Shape (batch_size, dimension, n_constraints).
    """

    b: Optional[jnp.ndarray] = None
    A: Optional[jnp.ndarray] = None
    Apinv: Optional[jnp.ndarray] = None

    def validate(self) -> None:
        """Validate the equality constraints specification.

        NOTE: This checks cannot be done after tracing, but this function
        can be used to validate the inputs before tracing.
        """
        if self.A is not None and self.b is None:
            raise ValueError("If A is provided, b must also be provided.")

    def update(self, **kwargs) -> "EqualityConstraintsSpecification":
        """Update some attribute by keyword."""
        return replace(self, **kwargs)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class BoxConstraintSpecification:
    """Dataclass representing inputs used in forming box constraints.

    Attributes:
        lb (jnp.ndarray): Lower bound of the box. Shape (batch_size, n_constraints, 1).
        ub (jnp.ndarray): Upper bound of the box. Shape (batch_size, n_constraints, 1).
        maskidx (Optional[jnp.ndarray]):
            Mask to apply the constraint only to some dimensions.
    """

    lb: Optional[jnp.ndarray] = None
    ub: Optional[jnp.ndarray] = None
    mask: Optional[jnp.ndarray] = None

    def update(self, **kwargs) -> "BoxConstraintSpecification":
        """Update some attribute by keyword."""
        return replace(self, **kwargs)

    def validate(self) -> None:
        """Validate the box constraint specification.

        NOTE: This checks cannot be done after tracing, but this function
        can be used to validate the inputs before tracing.
        """
        if self.lb is None and self.ub is None:
            raise ValueError("At least one of lower or upper bounds must be provided.")

        if self.lb is not None and hasattr(self.lb, "ndim") and self.lb.ndim != 3:
            raise ValueError(
                "Lower bound must have shape (batch_size, n_constraints, 1). "
                f"Received shape: {getattr(self.lb, 'shape', None)}."
            )
        if self.ub is not None and hasattr(self.ub, "ndim") and self.ub.ndim != 3:
            raise ValueError(
                "Upper bound must have shape (batch_size, n_constraints, 1). "
                f"Received shape: {getattr(self.ub, 'shape', None)}."
            )

        if self.lb is not None and self.ub is not None:
            if hasattr(self.lb, "shape") and hasattr(self.ub, "shape"):
                if self.lb.shape[1:] != self.ub.shape[1:]:
                    raise ValueError(
                        "Lower and upper bounds must have the same shape. "
                        f"Received shapes: {self.lb.shape} and {self.ub.shape}."
                    )
                if (
                    self.lb.shape[0] != self.ub.shape[0]
                    and self.lb.shape[0] != 1
                    and self.ub.shape[0] != 1
                ):
                    raise ValueError(
                        "Batch size of lower and upper bounds must be the same "
                        "or one of them must be 1. "
                        f"Received shapes: {self.lb.shape} and {self.ub.shape}."
                    )

            if not jnp.all(self.lb <= self.ub):
                raise ValueError(
                    "Lower bound must be less than or equal to the upper bound."
                )

        if self.mask is not None:
            if getattr(self.mask, "dtype", None) != jnp.bool_:
                raise TypeError("Mask must be a boolean array.")
            if getattr(self.mask, "ndim", None) != 1:
                raise ValueError("Mask must be a 1D array.")

            dim = getattr(self.lb, "shape", None) or getattr(self.ub, "shape", None)
            if dim is not None:
                if dim[1] != int(jnp.sum(self.mask)):
                    raise ValueError(
                        "Number of active entries in the mask must match the bounds. "
                        f"Received mask shape: {getattr(self.mask, 'shape', None)}, "
                        f"bound shape: {dim}."
                    )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ProjectionInstance:
    """A dataclass for encapsulating model input parameters.

    Attributes:
        x (jnp.ndarray): The point to be projected.
            Shape (batch_size, dimension, 1)
        eq (Optional[EqualityConstraintsSpecification]):
            Specification of the equality constraints, if any.
        box (Optional[BoxConstraintSpecification]):
            Specification of the box constraints, if any.
    """

    x: jnp.ndarray
    eq: Optional[EqualityConstraintsSpecification] = None
    box: Optional[BoxConstraintSpecification] = None

    def validate(self) -> None:
        """Validate the projection instance.

        NOTE: This checks cannot be done after tracing, but this function
        can be used to validate the inputs before tracing.
        """
        if self.x.ndim != 3:
            raise ValueError(
                "x must have shape (batch_size, dimension, 1). "
                f"Received shape: {self.x.shape}."
            )

    def update(self, **kwargs) -> "ProjectionInstance":
        """Update some attribute by keyword."""
        return replace(self, **kwargs)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class EquilibrationParams:
    """A dataclass for encapsulating the equilibration parameters.

    Attributes:
        max_iter (int): Maximum number of iterations for the equilibration.
        tol (float): Tolerance for convergence of the equilibration.
        ord (float): Order of the norm used for convergence check.
        col_scaling (bool): Whether to apply column scaling.
        update_mode (str): Update mode for the equilibration.
            Available options are:
                - "Jacobi" means compute both row and column norms and update.
                - "Gauss" means compute row, update, compute column, update.
        safeguard (bool): Check if the condition number of A has decreased.
    """

    max_iter: int = 0
    tol: float = 1e-3
    ord: float = 2.0
    col_scaling: bool = False
    update_mode: str = "Gauss"
    safeguard: bool = False

    def validate(self) -> None:
        """Validate the equilibration parameters."""
        if self.max_iter < 0:
            raise ValueError("max_iter must be non-negative.")
        if self.tol <= 0:
            raise ValueError("tol must be positive.")
        if self.ord not in [1, 2, float("inf")]:
            raise ValueError("ord must be 1, 2, or infinity.")
        if self.update_mode not in ["Gauss", "Jacobi"]:
            raise ValueError('update_mode must be either "Gauss" or "Jacobi".')

    def update(self, **kwargs) -> "EquilibrationParams":
        """Update some attribute by keyword."""
        return replace(self, **kwargs)
