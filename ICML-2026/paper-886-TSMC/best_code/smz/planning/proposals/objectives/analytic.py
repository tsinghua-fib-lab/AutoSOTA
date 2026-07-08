from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import typing as jxt

from smz.planning.proposals.interface import AnalyticObjective

from dataclasses import dataclass


@dataclass
class Greedy(AnalyticObjective):
    do_epsilon_greedy: bool = False

    @staticmethod
    def divergence(q_star: jax.Array, pi: jax.Array) -> jxt.ArrayLike:
        # Undefined
        return 0.0

    def trust_region_upperbound(
            self, q: jax.Array, pi: jax.Arra
    ) -> jax.Array:
        return jnp.zeros(())

    def __call__(
            self,
            q: jax.Array,
            pi: jax.Array,
            *,
            epsilon: jxt.ArrayLike | None = None,
            inv_beta: jxt.ArrayLike | None = None,
    ) -> jax.Array:
        self._validate_args(inv_beta, epsilon)

        if epsilon is None:
            epsilon = 0.0

        epsilon = jnp.clip(epsilon, 0.0, 1.0) * int(self.do_epsilon_greedy)
        qstar = self.epsilon_greedy(q, eps=epsilon)

        # Warning, will return -infs for epsilon = 0.0.
        return jnp.log(qstar) if self.logits else qstar


@dataclass
class Uniform(AnalyticObjective):

    @staticmethod
    def divergence(q_star: jax.Array, pi: jax.Array) -> jxt.ArrayLike:
        # Undefined
        return 0.0

    def trust_region_upperbound(
            self, q: jax.Array, pi: jax.Array
    ) -> jax.Array:
        return jnp.zeros(())

    def __call__(
            self,
            q: jax.Array,
            pi: jax.Array,
            *,
            epsilon: jxt.ArrayLike | None = None,
            inv_beta: jxt.ArrayLike | None = None,
    ) -> jax.Array:
        if self.logits:
            return jnp.full(pi.size, fill_value=-jnp.log(pi.size))
        return jnp.full(pi.size, fill_value=1.0 / pi.size)


@dataclass
class Prior(AnalyticObjective):

    @staticmethod
    def divergence(q_star: jax.Array, pi: jax.Array) -> jxt.ArrayLike:
        # Undefined
        return 0.0

    def trust_region_upperbound(
            self, q: jax.Array, pi: jax.Array
    ) -> jax.Array:
        return jnp.zeros(())

    def __call__(
            self,
            q: jax.Array,
            pi: jax.Array,
            *,
            epsilon: jxt.ArrayLike | None = None,
            inv_beta: jxt.ArrayLike | None = None,
    ) -> jax.Array:
        return jnp.log(pi) if self.logits else pi
