from __future__ import annotations
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import typing as jxt

from smz.planning.proposals.interface import PolicyObjective
from smz.planning.proposals.objectives.solve_norm import (
    ExPropKullbackLeibler
)


@dataclass
class Muesli(PolicyObjective):
    """

    References:
        Muesli: Combining Improvements in Policy Optimization. Hessel et al.,
        2022. Version 2. URL: https://arxiv.org/abs/2104.06159.
        Accessed 02-05-2024.
    """

    num_init: int = 50
    recursive_steps: int = 10

    bounds: tuple[float, float] = (-20, 10.0)

    _greedy_jitter: float = 1e-2

    clip: float = 1.0  # Default value in Muesli paper (Section 4.1)
    cmpo_temperature: float = 1.0  # Default value in Muesli paper

    _exkl: ExPropKullbackLeibler | None = None

    def __post_init__(self):
        self._exkl = ExPropKullbackLeibler(
            num_init=self.num_init,
            recursive_steps=self.recursive_steps,
            bounds=self.bounds,
            logits=self.logits,
            _epsilon_rtol=self._epsilon_rtol,
            _epsilon_ltol=self._epsilon_ltol,
            _greedy_jitter=self._greedy_jitter
        )

    @staticmethod
    def clipping_from_tvd(epsilon: jax.typing.ArrayLike) -> jax.Array:
        """Determine the optimal clipping threshold from a TVD-bound.

        Epsilon bounds the *worst* case total-variation from the prior.
        """
        return 2 * jnp.arctanh(epsilon)

    @staticmethod
    def max_tvd_from_clipping(clip: jax.typing.ArrayLike) -> jax.Array:
        """Determine the worst TVD from a clip-threshold. `clip` in [0, 1]. """
        return jnp.tanh(clip / 2)

    @staticmethod
    def divergence(
            q_star: jax.Array,
            pi: jax.Array
    ) -> jxt.ArrayLike:
        # Undefined. Muesli regularizer is not a divergence.
        return 0.0

    def lagrangian(
            self,
            log_eta: jax.Array, q: jax.Array, log_pi: jax.Array,
            *,
            inv_beta: jax.Array | None = None,
            epsilon: jax.Array | None = None
    ) -> tuple:
        # Unused
        log_pi_cmpo = self.get_cmpo_prior(q, jnp.exp(log_pi), log=True)
        return self._exkl.lagrangian(
            log_eta, q, log_pi_cmpo, inv_beta=inv_beta, epsilon=epsilon
        )

    def get_cmpo_prior(
            self, q: jax.Array, pi: jax.Array,
            *,
            log: bool = True, v: jax.Array | None = None
    ) -> jax.Array:
        if v is None:
            v = q.mean()  # Note: Clipping from q.mean() is most reliable.

        q_clipped = jnp.clip(q - v, -self.clip - 1e-8, self.clip + 1e-8)
        log_pi_cmpo = jnp.log(pi) + q_clipped / (1e-8 + self.cmpo_temperature)

        return log_pi_cmpo if log else jax.nn.softmax(log_pi_cmpo)

    def trust_region_upperbound(
            self, q: jax.Array, pi: jax.Array
    ) -> jax.Array:
        # Unused
        return self._exkl.trust_region_upperbound(q, pi)

    def __call__(
            self,
            q: jax.Array,
            pi: jax.Array,
            *,
            epsilon: jax.typing.ArrayLike | None = None,
            inv_beta: jax.typing.ArrayLike | None = None
    ) -> jax.Array:
        self._validate_args(inv_beta, epsilon, True)

        pi_cmpo = self.get_cmpo_prior(q, pi, log=False)
        return self._exkl(q=q, pi=pi_cmpo, epsilon=epsilon, inv_beta=inv_beta)
