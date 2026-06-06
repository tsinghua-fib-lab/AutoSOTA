"""Box constraint module."""

import numpy as np
from jax import numpy as jnp

from pinet.dataclasses import BoxConstraintSpecification, ProjectionInstance

from .base import Constraint


class BoxConstraint(Constraint):
    """Box constraint set.

    The box constraint set is defined as the Cartesian product of intervals.
    The interval is defined by a lower and an upper bound.
    The constraint possibly acts only on a subset of the dimensions,
    defined by a mask.
    """

    def __init__(
        self,
        box_spec: BoxConstraintSpecification,
    ) -> None:
        """Initialize the box constraint.

        Args:
            box_spec (BoxConstraintSpecification): Specification of the box constraint.
                For variable bounds, provide an example of the bounds.
        """
        self.lb = box_spec.lb
        self.ub = box_spec.ub
        self.mask = box_spec.mask
        self._dim = self.lb.shape[1] if self.lb is not None else self.ub.shape[1]
        self.scale = jnp.ones((1, self._dim, 1))

        if self.mask is None:
            self.mask = np.ones(shape=(self.dim), dtype=jnp.bool_)

    def get_params(
        self, yraw: ProjectionInstance
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Get the parameters of the box constraint.

        Args:
            yraw (ProjectionInstance): ProjectionInstance to get the parameters from.

        Returns:
            tuple: A tuple containing the lower and upper bounds and the mask.
                Each of shape (batch_size, n_constraints, 1).
        """
        lb = (
            (yraw.box.lb * self.scale)
            if yraw.box and yraw.box.lb is not None
            else self.lb
        )
        ub = (
            (yraw.box.ub * self.scale)
            if yraw.box and yraw.box.ub is not None
            else self.ub
        )
        mask = yraw.box.mask if yraw.box and yraw.box.mask is not None else self.mask
        if lb is None:
            lb = -jnp.inf * jnp.ones_like(ub)
        if ub is None:
            ub = jnp.inf * jnp.ones_like(lb)
        # NOTE: Mask is never None

        return lb, ub, mask

    def project(self, yraw: ProjectionInstance) -> ProjectionInstance:
        """Project the input to the feasible region.

        Args:
            yraw (ProjectionInstance): ProjectionInstance to projection.
                The .x attribute is the point to project.

        Returns:
            ProjectionInstance: The projected point for each point in the batch.
                .x of shape (batch_size, dimension, 1).
        """
        lb, ub, mask = self.get_params(yraw)
        return yraw.update(
            x=yraw.x.at[:, mask, :].set(jnp.clip(yraw.x[:, mask, :], lb, ub))
        )

    def cv(self, y: ProjectionInstance) -> jnp.ndarray:
        """Compute the constraint violation.

        Args:
            y (ProjectionInstance): ProjectionInstance to evaluate.

        Returns:
            jnp.ndarray: The constraint violation for each point in the batch.
                Shape (batch_size, 1, 1).
        """
        lb, ub, mask = self.get_params(y)
        cvs = jnp.maximum(
            jnp.max(y.x[:, mask, :] - ub, axis=1, keepdims=True),
            jnp.max(lb - y.x[:, mask, :], axis=1, keepdims=True),
        )
        return jnp.maximum(cvs, 0)

    @property
    def dim(self) -> int:
        """Return the dimension of the constraint set.

        Returns:
            int: The dimension of the constraint set.
        """
        return self._dim

    @property
    def n_constraints(self) -> int:
        """Return the number of constraints.

        Returns:
            int: The number of constraints.
        """
        return self._dim
