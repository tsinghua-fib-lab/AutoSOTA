from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import typing as jxt

from smz.planning.proposals.interface import NumericalNormalizerObjective
from smz.planning.proposals.mixins import (
    SolveNormalizerMixin, SandwichTrustRegionMixin
)


@dataclass
class ExPropKullbackLeibler(
    SandwichTrustRegionMixin,
    SolveNormalizerMixin,
    NumericalNormalizerObjective
):
    """Implements the Kullback-Leibler divergence from prior to model.

    KL(prior || model) = int_X prior(x) log(prior(x) / model(x)) dx

    This KL is often obtained in empirical density fitting, as its
    minimization w.r.t. the model is equivalent to minimization of the
    cross-entropy loss. It is also obtained by swapping the model and
    prior arguments for the Variational KL. This is known in literature as
    Expectation-Propagation.

    This divergence is also referred to, ambiguously, as the forward-KL.

    For parametric model-fitting, this divergence induces moment-matching
    behaviour.
    """

    num_init: int = 16
    recursive_steps: int = 5
    bounds: tuple[float, float] = (-10, 5.0)

    _greedy_jitter: float = 1e-2
    _epsilon_ltol: float = 0.0001

    def lagrangian(
        self,
        log_eta: jax.Array,
        q: jax.Array,
        log_pi: jax.Array,
        *,
        inv_beta: jax.Array | None = None,
        epsilon: jax.Array | None = None,
    ) -> tuple:
        """Computes the partial log-Lagrangian for the normalizer eta."""
        self._validate_args(inv_beta, epsilon, True)

        log_z = jax.nn.logsumexp(
            jnp.asarray([
                jnp.broadcast_to(log_eta, q.shape),
                jnp.log(-q + 1e-32)
            ]), axis=0,
        )

        if epsilon is not None:
            log_inv_beta = jnp.sum(jnp.exp(log_pi) * log_z) - epsilon
        else:
            log_inv_beta = jnp.log(inv_beta)

        log_q_star = log_pi + log_inv_beta - log_z

        # 1 - exp(logits) = 0 --> logits = 0
        return (jax.nn.logsumexp(log_q_star),)

    def get_search_bounds(
        self,
        q: jax.Array,
        log_pi: jax.Array,
        *,
        inv_beta: jax.Array | None = None,
        epsilon: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        # Bound for eta depends on beta_inv. Given epsilon, the solution to
        # beta_inv depends on eta. So unless the constraint is soft, we
        # need to expand the search-domain for the hard-constraint program.
        if epsilon is not None:
            return (
                jnp.max(q + jnp.exp(self.bounds[0]) * jnp.exp(log_pi)),
                q.max() + jnp.exp(self.bounds[-1]),
            )

        return jnp.max(q + inv_beta * jnp.exp(log_pi)), q.max() + inv_beta

    @staticmethod
    def divergence(q_star: jax.Array, pi: jax.Array) -> jxt.ArrayLike:
        divergence = pi * (
            jnp.clip(jnp.log(pi), -1e3) - jnp.clip(jnp.log(q_star), -1e3)
        )
        return divergence.sum()

    @staticmethod
    def inv_beta(
        q: jax.Array,
        pi: jax.Array,
        eta: jax.Array,
        epsilon: jxt.ArrayLike | None = None,
    ) -> jax.Array:
        # Get analytical solution to the trust-region multiplier.
        log_z = jnp.log(eta - q)
        return jnp.exp(jnp.sum(pi * log_z) - epsilon)

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

        log_eta = self.solve_normalizer(
            q, log_pi, inv_beta=inv_beta, epsilon=epsilon
        )

        log_z = jax.nn.logsumexp(
            jnp.asarray([
                jnp.broadcast_to(log_eta, q.shape),
                jnp.log(-q + 1e-32)
            ]), axis=0,
        )

        if epsilon is not None:
            log_inv_beta = jnp.sum(jnp.exp(log_pi) * log_z) - epsilon
        else:
            log_inv_beta = jnp.log(inv_beta)

        logits = log_pi + log_inv_beta - log_z
        return logits if self.logits else jnp.exp(logits)

    def trust_region_upperbound(
            self, q: jax.Array, pi: jax.Array
    ) -> jax.Array:
        """Uses the KL-divergence from pi to an epsilon-greedy policy.

        We have to use an epsilon-greedy policy as a heuristic since the
        KL from pi to greedy is undefined. The greedy policy is the dirac
        measure on the maximum of q. The logarithm inside the KL grows to
        infinity due to this measure.
        """
        reference = self(q, pi, inv_beta=jnp.exp(self.bounds[0]))
        reference = jnp.exp(reference) if self.logits else reference

        max_supported = self.divergence(reference, pi)
        jittered = self.divergence(
            self.epsilon_greedy(q, self._greedy_jitter),
            pi
        )
        return jnp.minimum(max_supported, jittered)


@dataclass
class SquaredHellinger(
    SandwichTrustRegionMixin,
    SolveNormalizerMixin,
    NumericalNormalizerObjective
):
    """Implements the squared Hellinger distance from prior to model.

    H^2(prior, model) = 2 - 2int_X sqrt(prior(x) * model(x)) dx

    This objective is symmetric in its arguments and bounded between [0, 1].

    Note: as a result of the divergence being bounded, epsilon and inv_beta
          are also bounded. If epsilon exceeds tuned bounds, we opt for a
          uniform or greedy distribution to prevent numerical problems in
          the solver. For inv_beta we cannot check the bounds without knowing
          the normalizer, so we clip it to the epsilon-bound in the solver.
    """

    num_init: int = 16
    recursive_steps: int = 5
    bounds: tuple[float, float] = (-10, 5.0)

    _epsilon_rtol: float = 0.05
    _epsilon_ltol: float = 0.0001

    def lagrangian(
        self,
        log_eta: jax.Array,
        q: jax.Array,
        log_pi: jax.Array,
        *,
        inv_beta: jax.Array | None = None,
        epsilon: jax.Array | None = None,
    ) -> tuple:
        """Computes the partial log-Lagrangian for the normalizer eta."""
        self._validate_args(inv_beta, epsilon, True)
        eta = jnp.exp(log_eta)

        log_z = jnp.log((2 * jnp.abs(eta - q)) + 1e-32)
        norm = jax.nn.logsumexp(log_pi - log_z)

        if epsilon is not None:
            log_inv_beta = jnp.log(1 - epsilon) - norm
        else:
            log_inv_beta = jnp.log(jnp.clip(inv_beta, 1e-32))

        log_q_star = log_pi + 2 * (log_inv_beta - log_z)

        # 1 - exp(logits) = 0 --> logits = 0
        return (jax.nn.logsumexp(log_q_star),)

    def get_search_bounds(
        self,
        q: jax.Array,
        log_pi: jax.Array,
        *,
        inv_beta: jax.Array | None = None,
        epsilon: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        # Bound for eta depends on beta_inv. Given epsilon, the solution to
        # beta_inv depends on eta. So unless the constraint is soft, we
        # need to expand the search-domain for the hard-constraint program.
        if epsilon is not None:
            return (
                jnp.max(q + jnp.exp(self.bounds[0]) * jnp.exp(0.5 * log_pi)),
                q.max() + jnp.exp(self.bounds[-1]),
            )

        return (jnp.max(q + inv_beta * jnp.exp(0.5 * log_pi)),
                q.max() + inv_beta)

    @staticmethod
    def inv_beta(
        q: jax.Array,
        pi: jax.Array,
        eta: jax.Array,
        epsilon: jxt.ArrayLike | None = None,
    ) -> jax.Array:
        # Get analytical solution to the trust-region multiplier.
        z = (2 * jnp.abs(eta - q)) + 1e-8
        norm = jnp.sum(pi / z)

        return (1 - epsilon) / norm

    @staticmethod
    def divergence(q_star: jax.Array, pi: jax.Array) -> jxt.ArrayLike:
        return 1 - jnp.sqrt(q_star * pi).sum()

    def _trust_region_interior(
        self,
        q: jax.Array,
        pi: jax.Array,
        *,
        epsilon: jxt.ArrayLike | None = None,
        inv_beta: jxt.ArrayLike | None = None,
    ) -> jax.Array:
        q = q - q.max()
        log_pi = jnp.log(jnp.clip(pi, 1e-32))

        log_eta = self.solve_normalizer(
            q, log_pi, inv_beta=inv_beta, epsilon=epsilon
        )

        log_z = jnp.log((2 * jnp.abs(jnp.exp(log_eta) - q)) + 1e-32)
        norm = jax.nn.logsumexp(log_pi - log_z)

        if epsilon is not None:
            log_inv_beta = jnp.log(1 - epsilon) - norm
        else:
            log_inv_beta = jnp.log(jnp.clip(inv_beta, 1e-32))

        log_q_star = log_pi + 2 * (log_inv_beta - log_z)

        return log_q_star if self.logits else jnp.exp(log_q_star)

    def trust_region_upperbound(
            self, q: jax.Array, pi: jax.Array
    ) -> jax.Array:
        reference = self(q, pi, inv_beta=jnp.exp(self.bounds[0]))
        reference = jnp.exp(reference) if self.logits else reference

        max_supported = self.divergence(reference, pi)
        jittered = self.divergence(
            self.epsilon_greedy(q, self._greedy_jitter),
            pi
        )

        return jnp.minimum(max_supported, jittered)
