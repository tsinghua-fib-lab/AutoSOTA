# This file is licensed under the MIT License.
# See the LICENSE file in the project root for full license information.
#
# VarDE sampling rule + Q-learning update (tabular)

from __future__ import annotations
import numpy as np
from typing import NamedTuple
from .agent import Agent, Experience, AgentParameters  # :contentReference[oaicite:0]{index=0}


class VarDEParameters(NamedTuple):
    tau: float = 0.1              # softmax temperature for influence weights w_tau(s,a)
    var_floor: float = 0.1       # lower bound on variance estimate (stability)
    init_variance: float = 1.0    # variance used when N(s,a) < 2


class VarDE(Agent):
    """
    Q-learning agent where action selection follows a VarDE-style sampling rule:

        a_t ∈ argmax_a  w_tau(s,a)^2 * sigma_hat^2(s,a) / ( N(s,a)*(N(s,a)+1) )

    with w_tau(s,·) = softmax(Q(s,·)/tau), and sigma_hat^2 estimated online
    from TD-target samples y = r + gamma * max_a' Q(s',a').
    """

    def __init__(self, parameters: VarDEParameters, agent_parameters: AgentParameters):
        super().__init__(agent_parameters)
        self.parameters = parameters

        # Same optimistic init as your QLearning implementation
        self.Q = np.ones((self.ns, self.na), dtype=np.float64) / (1.0 - self.discount_factor)

        # Online variance tracking for TD-target samples y(s,a)
        self._mean_y = np.zeros((self.ns, self.na), dtype=np.float64)
        self._M2_y = np.zeros((self.ns, self.na), dtype=np.float64)

    @staticmethod
    def suggested_exploration_parameter(dim_state: int, dim_action: int) -> float:
        # Not used directly (VarDE rule drives exploration), but keep the interface consistent.
        return 1.0

    # ---------- helpers ----------
    def _softmax(self, x: np.ndarray, tau: float) -> np.ndarray:
        # numerically-stable softmax
        z = x / max(tau, 1e-12)
        z = z - np.max(z)
        e = np.exp(z)
        s = e.sum()
        if s <= 0 or not np.isfinite(s):
            return np.ones_like(x) / x.size
        return e / s

    def _var_hat(self, s: int, a: int) -> float:
        n = int(self.state_action_visits[s, a])
        if n < 2:
            return float(self.parameters.init_variance)
        return float(self._M2_y[s, a] / (n - 1))

    # ---------- VarDE forward ----------
    def forward(self, state: int, step: int) -> int:
        # Influence weights w_tau(s,·) from current Q
        w = self._softmax(self.Q[state], self.parameters.tau)

        scores = np.empty(self.na, dtype=np.float64)

        # VarDE score for each action
        for a in range(self.na):
            n = int(self.state_action_visits[state, a])

            # Force at least one sample per action in each state
            if n == 0:
                scores[a] = np.inf
                continue

            sigma2 = max(self._var_hat(state, a), self.parameters.var_floor)
            scores[a] = (w[a] ** 2) * sigma2 / (n * (n + 1))

        # Random tie-break among maxima
        max_score = np.max(scores)
        best = np.flatnonzero(np.isclose(scores, max_score))
        return int(np.random.choice(best))

    # ---------- Q-learning update + variance tracking ----------
    def process_experience(self, experience: Experience, step: int) -> None:
        s, a, r, sp = experience.s_t, experience.a_t, experience.r_t, experience.s_tp1

        # TD target sample (used BOTH for Q update and variance tracking)
        y = r + self.discount_factor * self.Q[sp].max()

        # Online variance update (Welford) on y for (s,a)
        n = int(self.state_action_visits[s, a])  # AFTER backward(), this already includes the current sample
        # In this codebase, backward() calls process_experience() after incrementing state_action_visits.
        if n == 1:
            self._mean_y[s, a] = y
            self._M2_y[s, a] = 0.0
        else:
            delta = y - self._mean_y[s, a]
            self._mean_y[s, a] += delta / n
            delta2 = y - self._mean_y[s, a]
            self._M2_y[s, a] += delta * delta2

        # Same learning-rate schedule as your QLearning
        k = self.exp_visits[s, a].sum()
        H = 1.0 / (1.0 - self.discount_factor)
        alpha_t = (H + 1.0) / (H + k)

        # Q-learning update
        self.Q[s, a] = (1.0 - alpha_t) * self.Q[s, a] + alpha_t * y

        # Update greedy_policy with random tie-breaking (same style as QLearning)
        self.greedy_policy = (np.random.random(self.Q.shape) * (self.Q == self.Q.max(1, keepdims=True))).argmax(1)