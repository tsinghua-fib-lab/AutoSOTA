from __future__ import annotations
from typing import Literal
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import typing as jxt

from smz.utils import lambertw, log_lambertw_exp
from smz.planning.proposals.interface import (
    NumericalNormalizerAndTrustRegionObjective
)
from smz.planning.proposals.mixins import (
    SolveNormalizerAndTrustRegionMixin,
    SandwichTrustRegionMixin,
)


@dataclass
class Jeffrey(
    SandwichTrustRegionMixin,
    SolveNormalizerAndTrustRegionMixin,
    NumericalNormalizerAndTrustRegionObjective,
):
    """ """

    num_init: int = 8  # Normalization
    num_init_tr: int = 16  # Trust-Region

    recursive_steps: int = 8  # Normalization
    recursive_steps_tr: int = 10  # Trust-Region

    bounds: tuple[float, float] = (-5.0, 10.0)
    bounds_slack: tuple[float, float] = (1.5, 0.0)

    method: Literal["exact", "softplus"] = "softplus"

    _epsilon_ltol: float = 0.01
    _epsilon_rtol: float = 0.1

    @staticmethod
    def transform(x):
        return x

    def _log_lambertw_exp(self, x: jax.Array) -> jax.Array:
        """Computes the composition of the lambert W_0 and the exponential.

        If specified (default), it computes an approximation based on the
        lambertW's asymptotic behaviour: W_0(x) ~ ln(x) - ln(ln(x)).
        """

        if self.method == "exact":
            # True value, but, numerically unstable for large `x`
            return jnp.log(lambertw(jnp.exp(x)))

        elif self.method == "softplus":
            # Softplus based approximation of ln( lambertW( exp(x) ) ).
            return log_lambertw_exp(x)

        else:
            raise NotImplementedError(f"Invalid lambertw_exp: {self.method}")

    def lagrangian(
            self,
            eta: jax.typing.ArrayLike,
            q: jax.Array,
            log_pi: jax.Array,
            *,
            log_inv_beta: jax.typing.ArrayLike | None = None,
            epsilon: jax.typing.ArrayLike | None = None,
    ) -> tuple:
        """Computes the partial Lagrangian for inv_beta in log-space.

        Normalizer = exp(-1 - beta * eta)
        eta = -inv_beta + inv_beta * sum(exp(log_pi + beta * q))
        """
        self._validate_args(log_inv_beta, epsilon, False)

        z = 1 + (eta - q) * jnp.exp(-log_inv_beta)

        log_w_exp_z = self._log_lambertw_exp(z)
        log_q_star = log_pi - log_w_exp_z

        # Partial derivative of the Lagrangian wrt log_eta.
        partial_log_eta = jax.nn.logsumexp(log_q_star)

        if epsilon is None:
            return partial_log_eta, 0.0

        # Partial derivative of the Lagrangian wrt log_inv_beta
        log_q_to_pi, sign_q = jax.nn.logsumexp(
            log_q_star, b=-log_w_exp_z, return_sign=True
        )
        log_pi_to_q, sign_pi = jax.nn.logsumexp(
            log_pi, b=log_w_exp_z, return_sign=True
        )

        log_divergence = jnp.logaddexp(log_q_to_pi, log_pi_to_q)
        return partial_log_eta, jnp.log(epsilon) - log_divergence

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

        low = inv_beta if epsilon is None else jnp.exp(self.bounds[1])
        return jnp.max(q + low * (jnp.expm1(log_pi) + log_pi)), q.max()

    def _trust_region_interior(
            self,
            q: jax.Array,
            pi: jax.Array,
            *,
            epsilon: jax.typing.ArrayLike | None = None,
            inv_beta: jax.typing.ArrayLike | None = None,
    ) -> jax.Array:
        q = q - q.max()
        log_pi = jnp.log(jnp.clip(pi, 1e-16))

        if inv_beta is not None:
            inv_beta = jnp.clip(inv_beta, jnp.exp(self.bounds[0]))

        eta, log_inv_beta = self.solve(
            q, log_pi, inv_beta=inv_beta, epsilon=epsilon
        )

        z = 1 + (eta - q) / (jnp.exp(log_inv_beta) + 1e-8)

        logits = log_pi - self._log_lambertw_exp(z)

        if self.logits:
            return logits

        return jnp.exp(logits)

    def divergence(self, q_star: jax.Array, pi: jax.Array) -> jxt.ArrayLike:
        pi_to_q = pi * (jnp.clip(jnp.log(pi), -1e3) -
                        jnp.clip(jnp.log(q_star), -1e3))
        q_to_pi = q_star * (
                jnp.clip(jnp.log(q_star), -1e3) - jnp.clip(jnp.log(pi), -1e3)
        )
        return pi_to_q.sum() + q_to_pi.sum()

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


@dataclass
class JensenShannon(
    SandwichTrustRegionMixin,
    SolveNormalizerAndTrustRegionMixin,
    NumericalNormalizerAndTrustRegionObjective,
):
    """ """

    num_init: int = 16  # Normalization
    num_init_tr: int = 16  # Trust-Region

    recursive_steps: int = 8  # Normalization
    recursive_steps_tr: int = 10  # Trust-Region

    bounds: tuple[float, float] = (-3.0, 10.0)
    bounds_slack: tuple[float, float] = (0.0, 0.0)

    _epsilon_ltol: float = 0.01
    _epsilon_rtol: float = 0.02

    @staticmethod
    def transform(x):
        return x

    def lagrangian(
            self,
            eta: jax.typing.ArrayLike,
            q: jax.Array,
            log_pi: jax.Array,
            *,
            log_inv_beta: jax.typing.ArrayLike | None = None,
            epsilon: jax.typing.ArrayLike | None = None,
    ) -> tuple:
        """Computes the partial Lagrangian for inv_beta in log-space.

        Normalizer = exp(-1 - beta * eta)
        eta = -inv_beta + inv_beta * sum(exp(log_pi + beta * q))
        """
        self._validate_args(log_inv_beta, epsilon, False)

        z = jnp.log(2) + 2 * jnp.exp(-log_inv_beta) * (eta - q)
        z = jnp.clip(z, 0.0)

        log_denom = jnp.clip(jnp.log(jnp.expm1(z)), -1e6)
        log_q_star = log_pi - log_denom

        # Partial derivative of the Lagrangian wrt log_eta.
        partial_log_eta = jax.nn.logsumexp(log_q_star)

        if epsilon is None:
            return partial_log_eta, 0.0

        # Partial derivative of the Lagrangian wrt log_inv_beta
        log_mixture = jax.nn.logsumexp(jnp.asarray([log_q_star, log_pi]),
                                       b=0.5, axis=0)

        q_to_m = jax.nn.logsumexp(log_q_star, b=log_q_star - log_mixture)
        pi_to_m = jax.nn.logsumexp(log_pi, b=log_pi - log_mixture)

        log_divergence = jnp.logaddexp(q_to_m, pi_to_m) - jnp.log(2)

        return partial_log_eta, jnp.log(epsilon) - log_divergence

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
        low = inv_beta if epsilon is None else jnp.exp(self.bounds[1])

        # Note: Using no prior term improves reliability of the bounds.
        # b = jnp.max(q + low * (jax.nn.softplus(log_pi) - jnp.log(2) / 2))
        b = jnp.max(q - low * jnp.log(2) / 2)

        return b, q.max()

    def _trust_region_interior(
            self,
            q: jax.Array,
            pi: jax.Array,
            *,
            epsilon: jax.typing.ArrayLike | None = None,
            inv_beta: jax.typing.ArrayLike | None = None,
    ) -> jax.Array:
        q = q - q.max()
        log_pi = jnp.log(jnp.clip(pi, 1e-16))

        if inv_beta is not None:
            inv_beta = jnp.clip(inv_beta, jnp.exp(self.bounds[0]))

        eta, log_inv_beta = self.solve(q, log_pi, inv_beta=inv_beta,
                                       epsilon=epsilon)

        z = jnp.log(2) + 2 * jnp.exp(-log_inv_beta) * (eta - q)
        z = jnp.clip(z, 0.0)

        log_denom = jnp.clip(jnp.log(jnp.expm1(z)), -1e6)
        logits = log_pi - log_denom

        if self.logits:
            return logits

        return jnp.exp(logits)

    def divergence(self, q_star: jax.Array, pi: jax.Array) -> jxt.ArrayLike:
        m = (q_star + pi) / 2.0

        pi_to_m = pi * (
                    jnp.clip(jnp.log(pi), -1e3) - jnp.clip(jnp.log(m), -1e3))
        q_to_m = q_star * (
                    jnp.clip(jnp.log(q_star), -1e3) - jnp.clip(jnp.log(m),
                                                               -1e3))

        return (pi_to_m.sum() + q_to_m.sum()) / 2.0

    def trust_region_upperbound(
            self, q: jax.Array, pi: jax.Array
    ) -> jax.Array:
        reference = self(q, pi, inv_beta=jnp.exp(self.bounds[0]))
        reference = jnp.exp(reference) if self.logits else reference

        max_supported = self.divergence(reference, pi)
        jittered = self.divergence(
            self.epsilon_greedy(q, self._greedy_jitter), pi
        )
        return jnp.minimum(max_supported, jittered)
