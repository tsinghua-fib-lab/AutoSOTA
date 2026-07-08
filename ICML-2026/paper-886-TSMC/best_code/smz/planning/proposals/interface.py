from typing import Any

import abc

import jax
import jax.numpy as jnp
import jax.typing as jxt

from dataclasses import dataclass

from smz.planning.proposals.solver import bisection_method


@dataclass
class PolicyObjective(abc.ABC):
    """Base class for implementing proposal distributions for control.

    The proposal distribution solves a constrained program,
        max_q <Q, q>
        s.t.,
        D(q, pi) < epsilon
        int q(a) da = 1

    For the hard-constraint, one needs to specify epsilon to __call__.
    For the soft-constraint solution, one can also specify the Lagrange
    multiplier directly.

    Only discrete spaces are supported for a very good reason. For
    compatibility with continuous spaces one needs to sample from `pi`
    directly, and then passing a uniform distribution to __call__.
    """
    solver: Any | None = None
    logits: bool = False

    # Tolerance constants for validating solutions.
    _norm_tolerance: float = 0.001
    _epsilon_ltol: float = 0.001
    _epsilon_rtol: float = 0.01

    def set_solver(self, solver):
        self.solver = solver

    @staticmethod
    def _validate_args(
            inv_beta: jxt.ArrayLike | None,
            epsilon: jxt.ArrayLike | None,
            raise_ambiguity: bool = False
    ):
        if (epsilon is None) and (inv_beta is None):
            raise ValueError("Both `epsilon` and `inv_beta` cannot be None!")

        if raise_ambiguity:
            if (epsilon is not None) and (inv_beta is not None):
                raise ValueError(
                    "Ambiguity Error. Values given for both "
                    "`epsilon` and `inv_beta`! "
                )

    @staticmethod
    def epsilon_greedy(
            q: jax.Array,
            eps: jax.typing.ArrayLike = 0.0
    ) -> jax.Array:
        greedy = (q == q.max())
        return (greedy / greedy.sum()) * (1 - eps) + eps / q.size

    @abc.abstractmethod
    def trust_region_upperbound(
            self, q: jax.Array, pi: jax.Array
    ) -> jax.Array:
        pass

    @abc.abstractmethod
    def lagrangian(self, *args, **kwargs) -> tuple:
        pass

    @staticmethod
    @abc.abstractmethod
    def divergence(
            q_star: jax.Array,
            pi: jax.Array
    ) -> jxt.ArrayLike:
        pass

    @abc.abstractmethod
    def __call__(
            self,
            q: jax.Array,
            pi: jax.Array,
            *,
            epsilon: jxt.ArrayLike | None = None,
            inv_beta: jxt.ArrayLike | None = None
    ) -> jax.Array:
        # Compute Lagrangian solution w.r.t. q
        pass


@dataclass
class AnalyticObjective(PolicyObjective, abc.ABC):
    """Base class for objectives that are fully analytical.

    """

    def __init__(self, *, logits: bool = False):
        super().__init__(solver=object(), logits=logits)

    def lagrangian(self, *args, **kwargs) -> tuple:
        return 0,  # Solution is fully analytical


@dataclass
class NumericalNormalizerObjective(PolicyObjective, abc.ABC):
    """Base class for objectives without analytical trust-region constraint.

    Provides solver utility to estimate the optimal Lagrange multiplier.
    """
    num_init: int = 30
    recursive_steps: int = 10

    def __post_init__(self):
        if self.solver is None:
            self.solver = bisection_method(
                lambda x, k: self._objective(x, **k),
                self.num_init, self.recursive_steps
            )

    def _objective(self, x, **kwargs) -> jax.Array:
        return self.lagrangian(x, **kwargs)[0]

    @abc.abstractmethod
    def get_search_bounds(
            self,
            q: jax.Array,
            log_pi: jax.Array,
            *,
            inv_beta: jax.Array | None = None,
            epsilon: jax.Array | None = None
    ) -> tuple[jax.Array, jax.Array]:
        """Get informative bounds for the normalizer in the true search-space.
        """
        pass

    @abc.abstractmethod
    def solve_normalizer(self, *args, **kwargs) -> jxt.ArrayLike:
        pass


@dataclass
class NumericalTrustRegionObjective(PolicyObjective, abc.ABC):
    """Base class for objectives without analytical trust-region constraint.

    Provides solver utility to estimate the optimal Lagrange multiplier.
    """
    num_init: int = 30
    recursive_steps: int = 10

    def __post_init__(self):
        if self.solver is None:
            self.solver = bisection_method(
                lambda x, k: self._objective(x, **k),
                self.num_init, self.recursive_steps
            )

    def _objective(self, x, **kwargs) -> jax.Array:
        return self.lagrangian(x, **kwargs)[0]

    @abc.abstractmethod
    def solve_trust_region(self, *args, **kwargs) -> jxt.ArrayLike:
        pass


@dataclass
class NumericalNormalizerAndTrustRegionObjective(PolicyObjective, abc.ABC):
    """Base class for objectives without analytical trust-region constraint.

    Provides solver utility to estimate the optimal Lagrange multiplier.
    """
    num_init: int = 30
    recursive_steps: int = 10

    def __post_init__(self):

        if self.solver is None:
            self.solver = bisection_method(
                lambda x, k: self._objective(x, **k)[0],
                self.num_init, self.recursive_steps, step_size=0.0
            )

    def _objective(self, x, **kwargs) -> jax.Array:
        return jnp.asarray(self.lagrangian(x, **kwargs))

    @abc.abstractmethod
    def get_search_bounds(
            self,
            q: jax.Array,
            log_pi: jax.Array,
            *,
            inv_beta: jax.Array | None = None,
            epsilon: jax.Array | None = None
    ) -> tuple[jax.Array, jax.Array]:
        """Get informative bounds for the normalizer in the true search-space.
        """
        pass

    @abc.abstractmethod
    def solve(self, *args, **kwargs) -> jxt.ArrayLike:
        pass
