"""
SP-UCB-OLP: Saddle-Point UCB for Online Linear Programming

This is the main algorithm from the paper. It combines:
- UCB-style exploration over configurations
- Saddle-point optimization for mixture weights and prices
- OLP-style admission control

Key features:
1. Warm start: Round-robin for first K timesteps to ensure samples from all configs
2. Saddle-point solve: min_p max_w {<p, b_safe> + sum_theta w_theta (g_hat_theta(p) + beta_theta)}
3. Mixture sampling: theta_t ~ w_t
4. Global price admission: accept if r >= <p_t, a>
"""

import numpy as np
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional, Any
from .base import BaseAlgorithm


class SPUCBOLP(BaseAlgorithm):
    """
    Saddle-Point UCB-OLP Algorithm.

    This algorithm solves the optimistic saddle-point problem:

        min_p  max_w  { <p, b_safe> + sum_theta w_theta * (g_hat_theta(p) + beta_theta(t)) }

    where:
    - g_hat_theta(p) = (1/N_theta) * sum (r - <p, a>)_+ is empirical surplus
    - beta_theta(t) = UCB confidence radius
    - b_safe = (1 - epsilon) * b is the slack-adjusted per-period budget

    The optimal mixture w* concentrates on configs achieving the max in the envelope.
    We sample theta_t ~ w_t and use the global price p_t for admission.

    Parameters
    ----------
    K : int
        Number of configurations
    d : int
        Number of resource dimensions
    T : int
        Time horizon
    B : np.ndarray
        Total budget vector (d,)
    config : Dict[str, Any]
        Configuration with keys:
        - 'alpha': UCB exploration parameter (default: 0.1, scaled for numerical stability)
        - 'epsilon': Slack parameter (default: sqrt(log(T)/T))
        - 'R_max': Upper bound on rewards (default: 10.0)
        - 'A_max': Upper bound on consumption (default: 2.0)
        - 'delta': Confidence parameter (default: 1/T^2)
        - 'beta_max': Maximum confidence bonus (default: 10.0)
        - 'n_price_grid': Grid size for price optimization (default: 50)
        - 'warm_start': Whether to do round-robin warm start (default: True)
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

        # Algorithm-specific parameters
        # alpha=0.1 is more stable for small T; increase for more exploration
        self.alpha = self.config.get('alpha', 0.1)
        self.delta = self.config.get('delta', 1.0 / (T ** 2))
        self.n_price_grid = self.config.get('n_price_grid', 50)
        self.warm_start = self.config.get('warm_start', True)
        self.warm_start_rounds = self.config.get('warm_start_rounds', 1)  # Samples per arm during warm-start

        # Cap on confidence bonus to prevent numerical issues
        # Should be O(R_max) scale for reasonable behavior
        self.beta_max = self.config.get('beta_max', 10.0 * self.R_max)

        # Adaptive alpha schedule (ALGO-04)
        self.use_adaptive_alpha = self.config.get('use_adaptive_alpha', False)
        if self.use_adaptive_alpha:
            self.alpha_0 = self.config.get('alpha_0', self.alpha * 5.0)  # Initial alpha
            self.alpha_min_ratio = self.config.get('alpha_min_ratio', 0.2)  # Min alpha as fraction of alpha_0
            self.alpha_tau = self.config.get('alpha_tau', T / 4.0)  # Decay timescale

        # Adaptive safe budget (CODE-02)
        # Ma et al. (2023): use d_t = B_remaining / (T - t) instead of fixed b_safe
        self.use_adaptive_budget = self.config.get('use_adaptive_budget', False)
        self.adaptive_budget_ema_alpha = self.config.get('adaptive_budget_ema_alpha', 0.1)
        self._B_remaining_ema = None  # EMA of B_remaining for stability

        # Optimization settings - reduced for speed
        self.n_restarts = self.config.get('n_restarts', 2)  # Reduced from 3

        # Internal state
        self.last_solve_time = -1
        # Recompute frequency (configurable)
        self.solve_frequency = self.config.get('solve_frequency', 10)

        # Warm start: store previous optimal price
        self._last_optimal_p = None

        # Gurobi is available but scipy with caching is faster for this problem
        # (LP grows with O(n_samples) variables, scipy L-BFGS-B is O(n) per eval)
        self.use_gurobi = self.config.get('use_gurobi', False)
        self._gurobi_available = None  # Lazy check

    def _check_gurobi(self) -> bool:
        """Check if Gurobi is available (cached)."""
        if self._gurobi_available is None:
            try:
                import gurobipy
                self._gurobi_available = True
            except ImportError:
                self._gurobi_available = False
        return self._gurobi_available

    def _get_alpha(self, t: int) -> float:
        """Get time-dependent alpha for adaptive exploration (ALGO-04)."""
        if not self.use_adaptive_alpha:
            return self.alpha
        # alpha_t = alpha_0 * max(alpha_min_ratio, 1.0 / sqrt(1 + t/tau))
        alpha_t = self.alpha_0 * max(self.alpha_min_ratio, 1.0 / np.sqrt(1.0 + max(t, 1) / self.alpha_tau))
        return alpha_t

    def _compute_confidence_radii(self, t: int) -> np.ndarray:
        """Compute confidence radii for all configs, capped at beta_max."""
        # Temporarily override alpha for adaptive schedule
        saved_alpha = self.alpha
        if self.use_adaptive_alpha:
            self.alpha = self._get_alpha(t)
        
        beta = np.zeros(self.K)
        for theta in range(self.K):
            raw_beta = self.compute_confidence_radius(theta, t)
            beta[theta] = min(raw_beta, self.beta_max)
        
        self.alpha = saved_alpha
        return beta

    def _solve_saddle_point(self, t: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the optimistic saddle-point problem.

        min_p { <p, b_safe> + max_theta (g_hat_theta(p) + beta_theta) }

        Uses Gurobi LP if available, falls back to scipy L-BFGS-B.

        Returns
        -------
        w : np.ndarray
            Optimal mixture (K,)
        p : np.ndarray
            Optimal price (d,)
        """
        beta = self._compute_confidence_radii(t)

        # Try Gurobi first (much faster for this LP structure)
        if self.use_gurobi and self._check_gurobi():
            try:
                w, p, _ = self.solve_saddle_point_gurobi(self.b_safe, beta, self.P_max)
                self._last_optimal_p = p.copy()
                return w, p
            except Exception:
                # Fall back to scipy on any Gurobi error
                pass

        # Fallback: scipy L-BFGS-B
        return self._solve_saddle_point_scipy(beta)

    def _get_adaptive_b_safe(self) -> np.ndarray:
        """Compute adaptive safe budget using remaining budget and time (CODE-02).
        
        d_t = B_remaining / max(T - t, 1) * (1 - epsilon)
        Uses EMA of B_remaining for stability.
        """
        T_remaining = max(self.T - self.t, 1)
        # Use EMA of B_remaining to reduce oscillation
        if self._B_remaining_ema is None:
            self._B_remaining_ema = self.B_remaining.copy()
        else:
            self._B_remaining_ema = (self.adaptive_budget_ema_alpha * self.B_remaining + 
                                     (1 - self.adaptive_budget_ema_alpha) * self._B_remaining_ema)
        b_safe_t = self._B_remaining_ema / T_remaining * (1 - self.epsilon)
        # Ensure b_safe_t is at least a small fraction of original b_safe for numerical stability
        b_safe_t = np.maximum(b_safe_t, self.b_safe * 0.01)
        return b_safe_t

    def _solve_saddle_point_scipy(self, beta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Solve saddle-point using scipy (fallback)."""
        # Use adaptive b_safe if enabled
        if self.use_adaptive_budget:
            b_safe_current = self._get_adaptive_b_safe()
        else:
            b_safe_current = self.b_safe

        def objective(p):
            """Objective: <p, b_safe> + max_theta (g_hat_theta(p) + beta_theta)"""
            linear_term = np.dot(p, b_safe_current)

            # Compute g_hat_theta(p) + beta_theta for each config
            ucb_values = np.zeros(self.K)
            for theta in range(self.K):
                g_hat = self.compute_empirical_surplus(theta, p)
                ucb_values[theta] = g_hat + beta[theta]

            envelope = np.max(ucb_values)
            return linear_term + envelope

        # Optimize price via scipy minimize with warm starting
        best_val = np.inf
        best_p = np.zeros(self.d)

        # Use previous solution as warm start if available
        initial_points = []
        if self._last_optimal_p is not None:
            initial_points.append(self._last_optimal_p)
        initial_points.append(np.zeros(self.d))
        # Add random restarts only if we don't have a warm start
        if self._last_optimal_p is None:
            for _ in range(self.n_restarts - 1):
                initial_points.append(np.random.uniform(0, self.P_max / 2, self.d))

        for p_init in initial_points:
            result = minimize(
                objective,
                p_init,
                method='L-BFGS-B',
                bounds=[(0, self.P_max)] * self.d,
                options={'maxiter': 100, 'ftol': 1e-6}
            )

            if result.fun < best_val:
                best_val = result.fun
                best_p = result.x.copy()

        # Store for warm start next time
        self._last_optimal_p = best_p.copy()

        # Compute optimal mixture at the optimal price
        # w* puts weight on configs achieving the envelope
        ucb_values = np.zeros(self.K)
        for theta in range(self.K):
            g_hat = self.compute_empirical_surplus(theta, best_p)
            ucb_values[theta] = g_hat + beta[theta]

        max_ucb = np.max(ucb_values)
        eps_tie = 1e-8
        achieving_max = (ucb_values >= max_ucb - eps_tie)

        # Uniform over configs achieving the max
        w = achieving_max.astype(float)
        w /= w.sum()

        return w, best_p

    def select_config(self, t: int) -> Tuple[int, np.ndarray, np.ndarray]:
        """
        Select configuration for timestep t.

        During warm-start (t < K), use round-robin.
        After warm-start, solve saddle-point and sample from mixture.

        Returns
        -------
        theta : int
            Selected configuration
        w : np.ndarray
            Mixture weights (K,)
        p : np.ndarray
            Price vector (d,)
        """
        self.t = t

        # Warm-start: round-robin to ensure samples from each config
        # Paper: x_t = 0 during warm-start (observe only, don't consume budget)
        warm_start_length = self.K * self.warm_start_rounds
        if self.warm_start and t < warm_start_length:
            theta = t % self.K
            w = np.zeros(self.K)
            w[theta] = 1.0
            p = np.full(self.d, self.P_max)  # High price = reject all (observe only)

            self.current_theta = theta
            self.current_w = w
            self.current_p = p
            return theta, w, p

        # Solve saddle-point (with caching for efficiency)
        if t - self.last_solve_time >= self.solve_frequency or self.current_w is None:
            w, p = self._solve_saddle_point(t)
            self.current_w = w
            self.current_p = p
            self.last_solve_time = t
        else:
            w = self.current_w
            p = self.current_p

        # Sample theta from mixture w
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
        """
        Decide whether to accept arrival using global price.

        Accept if:
        1. Surplus is non-negative: r >= <p, a> - eps (with numerical tolerance)
        2. Budget constraint is satisfied: a <= B_remaining

        Returns
        -------
        accept : bool
            True to accept, False to reject
        """
        # Check budget feasibility
        if np.any(a > self.B_remaining + 1e-9):
            return False

        # Check surplus condition with small tolerance for numerical stability
        # This handles edge cases where optimizer finds prices slightly above breakeven
        surplus = r - np.dot(p, a)
        return surplus >= -1e-6

    def get_statistics(self) -> Dict[str, Any]:
        """Get algorithm statistics including saddle-point info."""
        stats = super().get_statistics()

        # Add algorithm-specific info
        stats['algorithm'] = 'SP-UCB-OLP'
        stats['alpha'] = self.alpha
        stats['epsilon'] = self.epsilon
        stats['solve_frequency'] = self.solve_frequency

        if self.current_w is not None:
            stats['current_w'] = self.current_w.tolist()
        if self.current_p is not None:
            stats['current_p'] = self.current_p.tolist()

        # Compute current confidence radii
        if self.t > 0:
            beta = self._compute_confidence_radii(self.t)
            stats['confidence_radii'] = beta.tolist()

        return stats

    def __repr__(self) -> str:
        return (
            f"SPUCBOLP(K={self.K}, d={self.d}, T={self.T}, "
            f"alpha={self.alpha}, epsilon={self.epsilon:.4f})"
        )
