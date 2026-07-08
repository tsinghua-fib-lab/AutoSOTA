from __future__ import annotations
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import typing as jxt

from smz.planning.proposals.interface import NumericalTrustRegionObjective
from smz.planning.proposals.mixins import SandwichTrustRegionMixin


@dataclass
class VariationalKullbackLeibler(
    SandwichTrustRegionMixin, NumericalTrustRegionObjective
):
    """Implements the Kullback-Leibler divergence from model to prior.

    KL(model || prior) = int_X model(x) log(model(x) / prior(x)) dx

    This is the KL as derived uniquely through the evidence lower bound.
    It is also referred to, ambiguously, as the reverse-KL.

    For parametric model-fitting, this divergence induces mode-finding
    behaviour.
    """

    num_init: int = 16
    recursive_steps: int = 5

    bounds: tuple[float, float] = (-10.0, 5.0)

    _greedy_jitter = 0.00
    _epsilon_ltol = 0.0001

    def lagrangian(
        self,
        min_log_beta: jax.Array,
        q: jax.Array,
        log_pi: jax.Array,
        *,
        epsilon: jax.Array,
    ) -> tuple:
        """Computes the partial log-Lagrangian for inv_beta in log-space."""
        logits = log_pi + q * jnp.exp(-min_log_beta)
        log_z = jax.nn.logsumexp(logits)

        log_q_star = logits - log_z
        return jnp.log(epsilon) - jax.nn.logsumexp(
            log_q_star, b=log_q_star - log_pi
        ),

    def solve_trust_region(
        self, q: jax.Array, log_pi: jax.Array, epsilon: jxt.ArrayLike
    ) -> jax.Array:
        # = log(inv_beta)
        return self.solver(
            self.bounds, kwargs=dict(q=q, log_pi=log_pi, epsilon=epsilon)
        )

    @staticmethod
    def divergence(q_star: jax.Array, pi: jax.Array) -> jxt.ArrayLike:
        divergence = q_star * (
            jnp.clip(jnp.log(q_star), -1e3) - jnp.clip(jnp.log(pi), -1e3)
        )
        return divergence.sum()

    def _trust_region_interior(
        self,
        q: jax.Array,
        pi: jax.Array,
        *,
        epsilon: jxt.ArrayLike | None = None,
        inv_beta: jxt.ArrayLike | None = None,
    ) -> jax.Array:
        q = q - q.max()
        log_pi = jnp.log(jnp.clip(pi, 1e-16))

        if epsilon is not None:
            min_log_beta = self.solve_trust_region(q, log_pi, epsilon=epsilon)
            inv_beta = jnp.exp(min_log_beta)

        logits = log_pi + q / jnp.clip(inv_beta, 1e-16)

        if self.logits:
            return jax.nn.log_softmax(logits)

        return jax.nn.softmax(logits)

    def trust_region_upperbound(
            self, q: jax.Array, pi: jax.Array
    ) -> jax.Array:
        # reference = self(q, pi, inv_beta=jnp.exp(self.bounds[0]))
        # reference = jnp.exp(reference) if self.logits else reference
        #
        # max_supported = self.divergence(reference, pi)
        jittered = self.divergence(
            self.epsilon_greedy(q, self._greedy_jitter), pi
        )
        # return jnp.minimum(max_supported, jittered)
        return jittered
