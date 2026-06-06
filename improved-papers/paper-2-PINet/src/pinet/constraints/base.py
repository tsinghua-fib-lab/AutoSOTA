"""Abstract class for constraint sets."""

from abc import abstractmethod

import jax.numpy as jnp

from pinet.dataclasses import ProjectionInstance


class Constraint:
    """Abstract class for constraint sets."""

    @abstractmethod
    def project(self, yraw: ProjectionInstance) -> ProjectionInstance:
        """Project the input to the feasible region.

        Args:
            yraw (ProjectionInstance): ProjectionInstance to project.

        Returns:
            ProjectionInstance: The projected input.
        """

    @abstractmethod
    def cv(self, yraw: ProjectionInstance) -> jnp.ndarray:
        """Compute the constraint violation.

        Args:
            yraw (ProjectionInstance): ProjectionInstance to evaluate.

        Returns:
            jnp.ndarray: The constraint violation for each point in the batch.
                Shape (batch_size, 1, 1).
        """

    @property
    @abstractmethod
    def dim(self) -> int:
        """Return the dimension of the constraint set.

        Returns:
            int: The dimension of the constraint set.
        """

    @property
    @abstractmethod
    def n_constraints(self) -> int:
        """Return the number of constraints.

        Returns:
            int: The number of constraints.
        """
