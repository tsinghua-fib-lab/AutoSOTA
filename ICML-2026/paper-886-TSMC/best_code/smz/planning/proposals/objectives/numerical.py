from __future__ import annotations
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import typing as jxt

from optax.projections import projection_simplex

from smz.planning.proposals.mixins import SandwichTrustRegionMixin
from smz.planning.proposals.interface import PolicyObjective
from smz.planning.proposals.solver import bisection_method


@dataclass
class TotalVariationL2(
    SandwichTrustRegionMixin,
    PolicyObjective
):
    """Implements the Total Variation distance using a L2-simplex projection.
    """

    num_init: int = 10
    recursive_steps: int = 10
    bounds: tuple[float, float] = (-20.0, 10.0)

    _epsilon_rtol: float = 0.02

    def __post_init__(self):

        if self.solver is None:
            self.solver = bisection_method(
                lambda x, k: self._objective(x, **k),
                self.num_init, self.recursive_steps, step_size=0.0
            )

    def _objective(self, x, **kwargs) -> jax.Array:
        return jnp.asarray(self.lagrangian(x, **kwargs)[1])

    def lagrangian(
            self,
            log_beta: jax.Array,
            q: jax.Array,
            pi: jax.Array,
            *,
            epsilon: jax.Array
    ) -> tuple:
        # Based on: https://stanford.edu/~jduchi/projects/DuchiShSiCh08.pdf
        q_star = projection_simplex(q * jnp.exp(log_beta) + pi)
        return 1.0 - q_star.sum(), epsilon - self.divergence(q_star, pi)

    @staticmethod
    def divergence(q_star: jax.Array, pi: jax.Array) -> jxt.ArrayLike:
        return jnp.abs(q_star - pi).sum() / 2

    def _trust_region_interior(
            self,
            q: jax.Array,
            pi: jax.Array,
            *,
            epsilon: jax.typing.ArrayLike | None = None,
            inv_beta: jax.typing.ArrayLike | None = None
    ) -> jax.Array:
        q = q - q.max()

        if epsilon is not None:
            log_beta = self.solver(
                self.bounds, kwargs=dict(q=q, pi=pi, epsilon=epsilon)
            )
            beta = jnp.exp(log_beta)
        else:
            beta = 1.0 / inv_beta

        q_star = projection_simplex(q * jnp.clip(beta, 1e-32) + pi)

        if self.logits:
            return jnp.log(q_star)

        return q_star

    def trust_region_upperbound(
            self, q: jax.Array, pi: jax.Array
    ) -> jax.Array:
        return self.divergence(self.epsilon_greedy(q, 0.0), pi)
