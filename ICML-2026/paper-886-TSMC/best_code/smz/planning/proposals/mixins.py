"""Module for useful mixin classes to extend base class functionalities.

"""
from typing import Callable, Any
from functools import partial

import abc

import jax
import jax.numpy as jnp

from smz.planning.proposals.solver import bisection_method


class TypesMixin:
    num_init: int
    recursive_steps: int
    solver: Any
    bounds: tuple[float, float]

    bounds_slack: tuple[float, float] = (1.0, 0.1)

    _validate_args: Callable[[jax.Array, jax.Array, bool], None]
    _objective: Callable[..., jax.Array]


class SolveNormalizerMixin(TypesMixin):
    """Mixin class to implement a generic solver for the normalization value.
    """

    @staticmethod
    def transform(x):
        return jnp.log(x)

    def get_search_bounds(
            self,
            q: jax.Array,
            log_pi: jax.Array,
            *,
            inv_beta: jax.Array | None = None,
            epsilon: jax.Array | None = None
    ) -> tuple[jax.Array, jax.Array]:
        """Get informative bounds in the true search-space."""
        ...

    def solve_normalizer(
            self,
            q: jax.Array,
            log_pi: jax.Array,
            *,
            inv_beta: jax.Array | None = None,
            epsilon: jax.Array | None = None
    ) -> jax.Array:
        """Do a recursive grid-search + local hill-climber.

        Search is optionally performed (default = log) in a monotonically
        transformed search-space. This helps the stability of search,
        especially at the boundaries.

        This method returns the results in the *transformed* space.
        """
        self._validate_args(inv_beta, epsilon, True)

        # Compute bounds to speed up search.
        canonical_low, canonical_high = self.get_search_bounds(
            q, log_pi, inv_beta=inv_beta, epsilon=epsilon
        )

        results = self.solver(
            (self.transform(canonical_low) - self.bounds_slack[0],
             self.transform(canonical_high) + self.bounds_slack[1]),
            kwargs=dict(log_pi=log_pi, q=q, inv_beta=inv_beta, epsilon=epsilon)
        )

        return results


class SolveNormalizerAndTrustRegionMixin(TypesMixin):
    # Split up compute-budget for normalization and the trust-region.
    num_init_tr: int
    recursive_steps_tr: int

    def __post_init__(self):
        super().__post_init__()  # type: ignore

        self.inv_beta_solver = bisection_method(
            self._tr_objective,
            self.num_init_tr, self.recursive_steps_tr, step_size=0.1
        )

    def _tr_objective(self, x: jax.Array, kwargs: dict[str, Any]):
        unpack = {a: b for a, b in kwargs.items() if a != 'bounds'}

        # For a given log_inv_beta = 'x', normalize the solution
        f_eta = self.solver(
            kwargs['bounds'],
            kwargs=dict(log_inv_beta=x) | unpack
        )

        # Then return the lagrangian of the solution for 'x'
        return self._objective(f_eta, log_inv_beta=x, **unpack)[1]

    @staticmethod
    def transform(x):
        return jnp.log(x)

    def get_search_bounds(
            self,
            q: jax.Array,
            log_pi: jax.Array,
            *,
            inv_beta: jax.Array | None = None,
            epsilon: jax.Array | None = None
    ) -> tuple[jax.Array, jax.Array]:
        """Get informative bounds in the true search-space."""
        ...

    def solve(
            self,
            q: jax.Array,
            log_pi: jax.Array,
            *,
            inv_beta: jax.Array | None = None,
            epsilon: jax.Array | None = None
    ):
        """
        """
        self._validate_args(inv_beta, epsilon, True)

        if inv_beta is None:
            low, high = self.get_search_bounds(
                q, log_pi, epsilon=epsilon
            )
            search_eta_low = self.transform(low) - self.bounds_slack[0]
            search_eta_high = self.transform(high) + self.bounds_slack[1]

            log_inv_beta = self.inv_beta_solver(
                self.bounds,  # log_inv_beta bounds
                kwargs=dict(
                    bounds=(search_eta_low, search_eta_high),  # eta bounds
                    q=q, log_pi=log_pi, epsilon=epsilon
                )
            )
        else:
            log_inv_beta = jnp.log(inv_beta)

        # Find the normalizer using tight bounds.
        low, high = self.get_search_bounds(
            q, log_pi, inv_beta=jnp.exp(log_inv_beta)
        )
        search_eta_low = self.transform(low) - self.bounds_slack[0]
        search_eta_high = self.transform(high) + self.bounds_slack[1]

        # If inv_beta is given, we can reduce to 1D optimization
        search_eta_star = self.solver(
            (search_eta_low, search_eta_high),
            kwargs=dict(q=q, log_pi=log_pi, log_inv_beta=log_inv_beta)
        )
        return search_eta_star, log_inv_beta


class SandwichTrustRegionMixin(TypesMixin, abc.ABC):
    """

    """

    # Core Constants
    logits: bool
    _epsilon_ltol: float
    _epsilon_rtol: float

    # Core Methods
    epsilon_greedy: Callable[[jax.Array, jax.typing.ArrayLike], jax.Array]
    trust_region_upperbound: Callable[[jax.Array, jax.Array], jax.Array]

    # Mixin-functionality
    _greedy_jitter: float = 0.0

    def _trust_region_interior(
            self,
            q: jax.Array,
            pi: jax.Array,
            *,
            epsilon: jax.typing.ArrayLike | None = None,
            inv_beta: jax.typing.ArrayLike | None = None
    ) -> jax.Array:
        ...

    def _trust_region_boundary(
            self,
            q: jax.Array,
            pi: jax.Array,
            *,
            use_prior: bool
    ):
        """Get q-star when the trust-region constraint gives an extreme-case

        If epsilon is tiny, then return the prior. If epsilon exceeds the
        divergence between the greedy policy and the prior, return the greedy
        policy.
        """
        greedy = self.epsilon_greedy(q, self._greedy_jitter)
        out = jax.lax.select(use_prior, pi, greedy)

        # Warning will return -infs for epsilon \approx 1.0
        return jnp.log(out) if self.logits else out

    def __call__(
            self,
            q: jax.Array,
            pi: jax.Array,
            *,
            epsilon: jax.typing.ArrayLike | None = None,
            inv_beta: jax.typing.ArrayLike | None = None
    ) -> jax.Array:
        self._validate_args(inv_beta, epsilon, True)

        all_same = jnp.isclose(q, q[0]).all()

        if epsilon is not None:
            epsilon_ub = self.trust_region_upperbound(q, pi)

            use_prior = all_same | (epsilon < self._epsilon_ltol)
            use_greedy = epsilon > jnp.clip(epsilon_ub - self._epsilon_rtol, 0)

            return jax.lax.cond(
                use_prior | use_greedy,
                partial(self._trust_region_boundary, use_prior=use_prior),
                partial(self._trust_region_interior, epsilon=epsilon),
                q, pi
            )

        # We cannot check bounds for inv_beta without knowing the normalizer.
        inv_beta = jnp.clip(inv_beta, jnp.exp(self.bounds[0]))
        return jax.lax.cond(
            ~all_same,
            partial(self._trust_region_interior, inv_beta=inv_beta),
            lambda *_: pi,
            q, pi
        )
