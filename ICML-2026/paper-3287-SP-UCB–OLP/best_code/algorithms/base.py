"""
Base Algorithm Class

Abstract base class for all algorithms in the SP-UCB-OLP framework.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any


class BaseAlgorithm(ABC):
    """
    Abstract base class for online learning algorithms.

    All algorithms must implement:
    - select_config(): Choose configuration theta_t
    - decide_admission(): Make accept/reject decision
    - update(): Update internal state after observing outcome

    The algorithm maintains:
    - Sample storage S_theta for each configuration
    - Running budget B_remaining
    - Statistics for analysis
    """

    def __init__(
        self,
        K: int,
        d: int,
        T: int,
        B: np.ndarray,
        config: Dict[str, Any] = None
    ):
        """
        Initialize base algorithm.

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
            Algorithm-specific configuration
        """
        self.K = K
        self.d = d
        self.T = T
        self.B = B.copy()
        self.b = B / T  # Per-period budget
        self.config = config or {}

        # Budget tracking
        self.B_remaining = B.copy()

        # Sample storage: list of (r, a) tuples for each config
        self.samples: Dict[int, List[Tuple[float, np.ndarray]]] = {
            theta: [] for theta in range(K)
        }
        self.N_theta = np.zeros(K, dtype=np.int32)  # Sample counts

        # Cached sample arrays for efficiency (avoid rebuilding each call)
        self._cached_rewards: Dict[int, np.ndarray] = {}
        self._cached_consumptions: Dict[int, np.ndarray] = {}
        self._cache_valid: Dict[int, bool] = {theta: False for theta in range(K)}

        # Current time step
        self.t = 0

        # Algorithm parameters (use self.config which is always a dict)
        self.R_max = self.config.get('R_max', 10.0)
        self.A_max = self.config.get('A_max', 2.0)
        self.P_max = self.config.get('P_max', None)
        if self.P_max is None:
            b_min = np.min(self.b[self.b > 0]) if np.any(self.b > 0) else 0.1
            self.P_max = self.R_max / b_min + 1.0

        # Slack parameter
        self.epsilon = self.config.get('epsilon', None)
        if self.epsilon is None:
            self.epsilon = np.sqrt(np.log(T) / T) if T > 1 else 0.1
        self.b_safe = (1 - self.epsilon) * self.b

        # Current state (to be set by select_config)
        self.current_w: Optional[np.ndarray] = None  # Mixture
        self.current_p: Optional[np.ndarray] = None  # Price
        self.current_theta: Optional[int] = None     # Selected config

        # Statistics
        self.total_reward = 0.0
        self.total_accepts = 0

    def reset(self):
        """Reset algorithm state for new run."""
        self.B_remaining = self.B.copy()
        self.samples = {theta: [] for theta in range(self.K)}
        self.N_theta = np.zeros(self.K, dtype=np.int32)
        self.t = 0
        self.current_w = None
        self.current_p = None
        self.current_theta = None
        self.total_reward = 0.0
        self.total_accepts = 0
        # Reset cache
        self._cached_rewards = {}
        self._cached_consumptions = {}
        self._cache_valid = {theta: False for theta in range(self.K)}

    @abstractmethod
    def select_config(self, t: int) -> Tuple[int, np.ndarray, np.ndarray]:
        """
        Select configuration for timestep t.

        Parameters
        ----------
        t : int
            Current timestep

        Returns
        -------
        theta : int
            Selected configuration
        w : np.ndarray
            Mixture weights (K,)
        p : np.ndarray
            Price vector (d,)
        """
        pass

    @abstractmethod
    def decide_admission(
        self,
        t: int,
        theta: int,
        r: float,
        a: np.ndarray,
        p: np.ndarray
    ) -> bool:
        """
        Decide whether to accept or reject arrival.

        Parameters
        ----------
        t : int
            Current timestep
        theta : int
            Current configuration
        r : float
            Observed reward
        a : np.ndarray
            Observed consumption (d,)
        p : np.ndarray
            Current price vector (d,)

        Returns
        -------
        accept : bool
            True to accept, False to reject
        """
        pass

    def update(
        self,
        t: int,
        theta: int,
        r: float,
        a: np.ndarray,
        accepted: bool
    ):
        """
        Update algorithm state after observing outcome.

        Parameters
        ----------
        t : int
            Current timestep
        theta : int
            Configuration used
        r : float
            Observed reward
        a : np.ndarray
            Observed consumption (d,)
        accepted : bool
            Whether arrival was accepted
        """
        # Store sample (always, regardless of acceptance)
        self.samples[theta].append((r, a.copy()))
        self.N_theta[theta] += 1
        # Invalidate cache for this config
        self._cache_valid[theta] = False

        # Update budget and statistics if accepted
        if accepted:
            self.B_remaining -= a
            self.total_reward += r
            self.total_accepts += 1

        self.t = t + 1

    def get_samples_arrays(self, theta: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get samples for config theta as arrays (with caching).

        Returns
        -------
        rewards : np.ndarray
            Rewards array (n,)
        consumptions : np.ndarray
            Consumptions array (n, d)
        """
        # Check cache first
        if self._cache_valid.get(theta, False):
            return self._cached_rewards[theta], self._cached_consumptions[theta]

        samples = self.samples[theta]
        if len(samples) == 0:
            return np.array([]), np.zeros((0, self.d))

        rewards = np.array([s[0] for s in samples])
        consumptions = np.array([s[1] for s in samples])

        # Update cache
        self._cached_rewards[theta] = rewards
        self._cached_consumptions[theta] = consumptions
        self._cache_valid[theta] = True

        return rewards, consumptions

    def get_all_samples_dict(self) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """Get all samples as dictionary of arrays."""
        return {
            theta: self.get_samples_arrays(theta)
            for theta in range(self.K)
        }

    def compute_empirical_surplus(self, theta: int, p: np.ndarray) -> float:
        """Compute empirical surplus g_hat_theta(p)."""
        rewards, consumptions = self.get_samples_arrays(theta)
        if len(rewards) == 0:
            return 0.0
        # Safeguard: clip prices to valid range and handle numerical issues
        p_safe = np.clip(np.nan_to_num(p, nan=0.0, posinf=self.P_max, neginf=0.0), 0, self.P_max)
        margins = rewards - consumptions @ p_safe
        return float(np.mean(np.maximum(margins, 0)))

    def compute_empirical_consumption(self, theta: int, p: np.ndarray) -> np.ndarray:
        """Compute empirical threshold consumption h_hat_theta(p)."""
        rewards, consumptions = self.get_samples_arrays(theta)
        if len(rewards) == 0:
            return np.zeros(self.d)
        # Safeguard: clip prices to valid range and handle numerical issues
        p_safe = np.clip(np.nan_to_num(p, nan=0.0, posinf=self.P_max, neginf=0.0), 0, self.P_max)
        margins = rewards - consumptions @ p_safe
        accept_mask = (margins > 0).astype(float)
        return np.mean(consumptions * accept_mask[:, np.newaxis], axis=0)

    def compute_confidence_radius(self, theta: int, t: int) -> float:
        """
        Compute UCB confidence radius for config theta.

        beta_theta(t) = alpha * c_g * R_max * sqrt((d log t + log(KT/delta)) / N_theta)

        The scale factor c_g * R_max comes from the bounded range of (r - <p,a>)_+ in [0, R_max].
        This matches the paper's concentration lemma (Lemma conc-g in Appendix).
        """
        n = max(1, self.N_theta[theta])
        alpha = self.config.get('alpha', 1.0)
        delta = self.config.get('delta', 1.0 / (self.T ** 2))
        c_g = self.config.get('c_g', 1.0)  # Constant from concentration lemma

        scale = c_g * self.R_max  # Correct scale - matches paper theory
        # Paper formula: d * log(c_0 * d * P_max * A_max * T / R_max) + log(2KT/delta)
        # Key: log term is fixed (uses T), not growing (uses t)
        c_0 = self.config.get('c_0', 1.0)  # Constant from covering argument
        log_term = self.d * np.log(max(c_0 * self.d * self.P_max * self.A_max * self.T / self.R_max, 2.0)) + np.log(2 * self.K * self.T / delta)

        return alpha * scale * np.sqrt(log_term / n)

    def solve_saddle_point_gurobi(
        self,
        b: np.ndarray,
        beta: np.ndarray,
        P_max: float = None
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Solve saddle-point problem using Gurobi LP.

        min_p { <p, b> + max_θ (ĝ_θ(p) + β_θ) }

        Reformulated as LP:
            min   <p, b> + z
            s.t.  z ≥ (1/N_θ) Σ u_θi + β_θ   ∀θ
                  u_θi ≥ r_θi - <p, a_θi>    ∀θ, i
                  u_θi ≥ 0                    ∀θ, i
                  0 ≤ p ≤ P_max

        Parameters
        ----------
        b : np.ndarray
            Per-period budget vector (d,)
        beta : np.ndarray
            Confidence radii for each config (K,)
        P_max : float
            Upper bound on prices (default: self.P_max)

        Returns
        -------
        w : np.ndarray
            Optimal mixture (K,) - uniform over configs achieving envelope
        p : np.ndarray
            Optimal price vector (d,)
        obj_val : float
            Optimal objective value
        """
        try:
            import gurobipy as gp
            from gurobipy import GRB
        except ImportError:
            raise ImportError("Gurobi not installed. Install with: pip install gurobipy")

        if P_max is None:
            P_max = self.P_max

        # Collect samples for all configs
        samples_data = []
        total_vars = 0
        for theta in range(self.K):
            rewards, consumptions = self.get_samples_arrays(theta)
            n_samples = len(rewards)
            samples_data.append({
                'n': n_samples,
                'rewards': rewards,
                'consumptions': consumptions,
                'start_idx': total_vars
            })
            total_vars += n_samples

        # Create model with minimal output
        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            env.start()

            with gp.Model(env=env) as model:
                # Variables
                p = model.addVars(self.d, lb=0, ub=P_max, name="p")
                z = model.addVar(lb=-GRB.INFINITY, name="z")

                # u variables for each config's samples
                u = {}
                for theta in range(self.K):
                    n = samples_data[theta]['n']
                    if n > 0:
                        u[theta] = model.addVars(n, lb=0, name=f"u_{theta}")

                # Objective: min <p, b> + z
                obj = gp.quicksum(b[j] * p[j] for j in range(self.d)) + z
                model.setObjective(obj, GRB.MINIMIZE)

                # Constraints: z >= (1/N_θ) Σ u_θi + β_θ for all θ
                for theta in range(self.K):
                    n = samples_data[theta]['n']
                    if n > 0:
                        model.addConstr(
                            z >= (1.0 / n) * gp.quicksum(u[theta][i] for i in range(n)) + beta[theta],
                            name=f"envelope_{theta}"
                        )
                    else:
                        # No samples: surplus is 0, constraint is z >= beta_theta
                        model.addConstr(z >= beta[theta], name=f"envelope_{theta}")

                # Constraints: u_θi >= r_θi - <p, a_θi> for all θ, i
                for theta in range(self.K):
                    n = samples_data[theta]['n']
                    if n > 0:
                        rewards = samples_data[theta]['rewards']
                        consumptions = samples_data[theta]['consumptions']
                        for i in range(n):
                            model.addConstr(
                                u[theta][i] >= rewards[i] - gp.quicksum(
                                    consumptions[i, j] * p[j] for j in range(self.d)
                                ),
                                name=f"surplus_{theta}_{i}"
                            )

                # Optimize
                model.optimize()

                if model.status != GRB.OPTIMAL:
                    # Fallback to zero price
                    p_opt = np.zeros(self.d)
                    w_opt = np.ones(self.K) / self.K
                    return w_opt, p_opt, 0.0

                # Extract optimal price
                p_opt = np.array([p[j].X for j in range(self.d)])
                obj_val = model.ObjVal

        # Compute mixture: uniform over configs achieving the envelope
        ucb_values = np.zeros(self.K)
        for theta in range(self.K):
            g_hat = self.compute_empirical_surplus(theta, p_opt)
            ucb_values[theta] = g_hat + beta[theta]

        max_ucb = np.max(ucb_values)
        eps_tie = 1e-8
        achieving_max = (ucb_values >= max_ucb - eps_tie)
        w_opt = achieving_max.astype(float)
        w_opt /= w_opt.sum()

        return w_opt, p_opt, obj_val

    def get_statistics(self) -> Dict[str, Any]:
        """Get algorithm statistics."""
        return {
            'total_reward': self.total_reward,
            'total_accepts': self.total_accepts,
            'acceptance_rate': self.total_accepts / max(1, self.t),
            'N_theta': self.N_theta.tolist(),
            'B_remaining': self.B_remaining.tolist(),
            'budget_utilization': 1 - self.B_remaining / self.B,
        }
