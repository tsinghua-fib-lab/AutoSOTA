"""
Baseline Algorithms for Alibaba Experiment

This module implements baseline algorithms for comparison with SP-UCB-OLP:

1. SPGreedyOLP: Greedy selection without exploration bonus (alpha=0)
2. OraclePolicy: Uses oracle mixture and price (upper bound)
3. RandomPolicy: Uniform random selection (lower bound)
"""

import numpy as np
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional, Any
from .base import BaseAlgorithm


class SPGreedyOLP(BaseAlgorithm):
    """
    Greedy Saddle-Point OLP without exploration.

    Same as SP-UCB-OLP but with alpha=0 (no confidence bonus).
    This is a pure exploitation baseline that doesn't explore.

    Expected behavior:
    - May get stuck on suboptimal config early
    - No regret guarantees
    """

    def __init__(
        self,
        K: int,
        d: int,
        T: int,
        B: np.ndarray,
        config: Dict[str, Any] = None
    ):
        config = config or {}
        config['alpha'] = 0.0  # No exploration
        super().__init__(K, d, T, B, config)

        self.n_restarts = self.config.get('n_restarts', 2)
        self.warm_start = self.config.get('warm_start', True)

        # Caching for efficiency
        self.last_solve_time = -1
        self.solve_frequency = max(1, int(np.sqrt(T)))
        self._last_optimal_p = None

    def _solve_greedy(self, t: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve greedy problem (no UCB bonus) with warm starting.

        min_p { <p, b_safe> + max_theta g_hat_theta(p) }
        """
        def objective(p):
            linear_term = np.dot(p, self.b_safe)
            surpluses = np.zeros(self.K)
            for theta in range(self.K):
                surpluses[theta] = self.compute_empirical_surplus(theta, p)
            envelope = np.max(surpluses)
            return linear_term + envelope

        best_val = np.inf
        best_p = np.zeros(self.d)

        # Use previous solution as warm start if available
        initial_points = []
        if self._last_optimal_p is not None:
            initial_points.append(self._last_optimal_p)
        initial_points.append(np.zeros(self.d))
        if self._last_optimal_p is None:
            for _ in range(self.n_restarts - 1):
                initial_points.append(np.random.uniform(0, self.P_max / 2, self.d))

        for p_init in initial_points:
            result = minimize(
                objective,
                p_init,
                method='L-BFGS-B',
                bounds=[(0, self.P_max)] * self.d,
                options={'maxiter': 100}
            )

            if result.fun < best_val:
                best_val = result.fun
                best_p = result.x.copy()

        # Store for warm start next time
        self._last_optimal_p = best_p.copy()

        # Compute mixture at optimal price
        surpluses = np.zeros(self.K)
        for theta in range(self.K):
            surpluses[theta] = self.compute_empirical_surplus(theta, best_p)

        max_surplus = np.max(surpluses)
        achieving_max = (surpluses >= max_surplus - 1e-8)
        w = achieving_max.astype(float)
        w /= w.sum()

        return w, best_p

    def select_config(self, t: int) -> Tuple[int, np.ndarray, np.ndarray]:
        """Select config greedily with caching."""
        self.t = t

        if self.warm_start and t < self.K:
            theta = t % self.K
            w = np.zeros(self.K)
            w[theta] = 1.0
            p = np.zeros(self.d)
            self.current_theta = theta
            self.current_w = w
            self.current_p = p
            return theta, w, p

        # Only recompute every solve_frequency steps
        if t - self.last_solve_time >= self.solve_frequency or self.current_w is None:
            w, p = self._solve_greedy(t)
            self.current_w = w
            self.current_p = p
            self.last_solve_time = t
        else:
            w = self.current_w
            p = self.current_p

        theta = np.random.choice(self.K, p=w)

        self.current_theta = theta
        return theta, w, p

    def decide_admission(
        self,
        t: int,
        theta: int,
        r: float,
        a: np.ndarray,
        p: np.ndarray
    ) -> bool:
        """Accept if surplus non-negative and budget allows."""
        if np.any(a > self.B_remaining + 1e-9):
            return False
        return r >= np.dot(p, a) - 1e-6

    def __repr__(self) -> str:
        return f"SPGreedyOLP(K={self.K}, d={self.d}, T={self.T})"


class OraclePolicy(BaseAlgorithm):
    """
    Oracle Policy with perfect information.

    Uses the true optimal mixture w* and price p* computed from
    the population distribution (not samples).

    This is an UPPER BOUND on achievable performance, not a
    practical algorithm.

    Parameters
    ----------
    w_star : np.ndarray
        Optimal mixture from oracle computation (K,)
    p_star : np.ndarray
        Optimal price from oracle computation (d,)
    """

    def __init__(
        self,
        K: int,
        d: int,
        T: int,
        B: np.ndarray,
        config: Dict[str, Any] = None
    ):
        super().__init__(K, d, T, B, config)

        # Oracle values must be provided in config
        self.w_star = self.config.get('w_star', np.ones(K) / K)
        self.p_star = self.config.get('p_star', np.zeros(d))

        if isinstance(self.w_star, list):
            self.w_star = np.array(self.w_star)
        if isinstance(self.p_star, list):
            self.p_star = np.array(self.p_star)

    def select_config(self, t: int) -> Tuple[int, np.ndarray, np.ndarray]:
        """Sample from oracle mixture."""
        self.t = t

        theta = np.random.choice(self.K, p=self.w_star)
        w = self.w_star.copy()
        p = self.p_star.copy()

        self.current_theta = theta
        self.current_w = w
        self.current_p = p
        return theta, w, p

    def decide_admission(
        self,
        t: int,
        theta: int,
        r: float,
        a: np.ndarray,
        p: np.ndarray
    ) -> bool:
        """Accept using oracle price."""
        if np.any(a > self.B_remaining + 1e-9):
            return False
        return r >= np.dot(p, a) - 1e-6

    def __repr__(self) -> str:
        return f"OraclePolicy(K={self.K}, d={self.d}, T={self.T})"


class RandomPolicy(BaseAlgorithm):
    """
    Uniform random policy.

    Selects configs uniformly at random and accepts all
    arrivals that fit in budget.

    This is a LOWER BOUND baseline with no learning.
    """

    def __init__(
        self,
        K: int,
        d: int,
        T: int,
        B: np.ndarray,
        config: Dict[str, Any] = None
    ):
        super().__init__(K, d, T, B, config)

    def select_config(self, t: int) -> Tuple[int, np.ndarray, np.ndarray]:
        """Select config uniformly at random."""
        self.t = t

        theta = np.random.randint(0, self.K)
        w = np.ones(self.K) / self.K
        p = np.zeros(self.d)  # Accept everything that fits

        self.current_theta = theta
        self.current_w = w
        self.current_p = p
        return theta, w, p

    def decide_admission(
        self,
        t: int,
        theta: int,
        r: float,
        a: np.ndarray,
        p: np.ndarray
    ) -> bool:
        """Accept if budget allows (greedy)."""
        return np.all(a <= self.B_remaining + 1e-9)

    def __repr__(self) -> str:
        return f"RandomPolicy(K={self.K}, d={self.d}, T={self.T})"
