# This file is licensed under the MIT License.
# See the LICENSE file in the project root for full license information.
#

from __future__ import annotations

import numpy as np
from typing import NamedTuple

from .agent import AgentParameters  # for typing
from .mfbpi import MFBPI, MFBPIParameters

golden_ratio = (1 + np.sqrt(5)) / 2
golden_ratio_sq = golden_ratio ** 2


class VarDEMFBPIParameters(NamedTuple):
    # MF-BPI core params
    kbar: int
    ensemble_size: int

    # VarDE sampling params
    varde_tau: float = 1.0           # softmax temperature for w_tau(s,a)
    varde_var_floor: float = 1e-8    # floor for variance proxy
    force_min_exploration: float = 1e-3  # keep the same safety exploration


class VarDEMFBPI(MFBPI):
    """
    MF-BPI updates + VarDE sampling rule in forward().

    VarDE sampling score (per current state s):
        score(a) = w_tau(s,a)^2 * sigma2_hat(s,a) / (N(s,a) (N(s,a)+1))

    Mapping in this implementation:
      - N(s,a) := exp_visits[s,a,:].sum()
      - sigma2_hat(s,a) := m_values[s,a] (quantile head used by MF-BPI), floored
      - w_tau(s,·) := softmax(q_values[s,·] / varde_tau)
    """

    def __init__(self, parameters: VarDEMFBPIParameters, agent_parameters: AgentParameters):
        # Build a base MFBPIParameters and pass to parent
        super().__init__(MFBPIParameters(kbar=parameters.kbar, ensemble_size=parameters.ensemble_size), agent_parameters)
        self.varde_params = parameters

        # storage for the last (q_values, m_values) used when compute_omega() ran
        self._q_values_for_sampling = None
        self._m_values_for_sampling = None

    # --------- small helpers ---------
    @staticmethod
    def _softmax(x: np.ndarray, tau: float) -> np.ndarray:
        tau = max(float(tau), 1e-12)
        z = x / tau
        z = z - np.max(z)
        e = np.exp(z)
        s = e.sum()
        if (not np.isfinite(s)) or s <= 0:
            return np.ones_like(x) / x.size
        return e / s

    # --------- override MF-BPI omega computation to ALSO cache q,m used ---------
    def compute_omega(self):
        # Same logic as MF-BPI, but cache (q_values, m_values) for forward()

        if self.ensemble_size == 1:
            q_values = self.Q[0]
            m_values = self.M[0]
        else:
            x = np.random.uniform()
            q_values = np.quantile(self.Q, x, axis=0)
            m_values = np.quantile(self.M, x, axis=0)

        # Cache for VarDE forward
        self._q_values_for_sampling = q_values
        self._m_values_for_sampling = m_values

        # --- original MF-BPI compute_omega below (unchanged) ---
        greedy_policy = q_values.argmax(1)

        idxs_subopt_actions = np.array(
            [[False if greedy_policy[s] == a else True for a in range(self.na)] for s in range(self.ns)],
            dtype=np.bool_,
        )

        delta = np.clip((q_values.max(-1, keepdims=True) - q_values), a_min=1e-8, a_max=None)
        delta_subopt = delta[idxs_subopt_actions]
        delta_min = delta_subopt.min()

        delta[~idxs_subopt_actions] = delta_min * (1 - self.discount_factor) / (1 + self.discount_factor)

        Hsa = (2 + 8 * golden_ratio_sq * m_values) / (delta ** 2)

        C = np.max(np.maximum(4, 16 * (self.discount_factor ** 2) * golden_ratio_sq * m_values[~idxs_subopt_actions]))
        Hopt = C / (delta[~idxs_subopt_actions] ** 2)

        Hsa[~idxs_subopt_actions] = np.sqrt(Hopt * Hsa[idxs_subopt_actions].sum() / self.ns)

        self.omega = Hsa / Hsa.sum()
        self.policy = self.omega / self.omega.sum(-1, keepdims=True)

    # --------- VarDE sampling rule (replaces MF-BPI forward sampling from policy) ---------
    def forward(self, state: int, step: int) -> int:
        # Keep MF-BPI safety exploration
        eps = self.forced_exploration_callable(state, step, minimum_exploration=self.varde_params.force_min_exploration)
        if np.random.uniform() < eps:
            return int(np.random.choice(self.na))

        # Ensure q/m cache exists (if forward called before any update)
        if self._q_values_for_sampling is None or self._m_values_for_sampling is None:
            # deterministic fallback: use median quantile
            if self.ensemble_size == 1:
                self._q_values_for_sampling = self.Q[0]
                self._m_values_for_sampling = self.M[0]
            else:
                self._q_values_for_sampling = np.quantile(self.Q, 0.5, axis=0)
                self._m_values_for_sampling = np.quantile(self.M, 0.5, axis=0)

        q_values = self._q_values_for_sampling
        m_values = self._m_values_for_sampling

        # VarDE influence weights w_tau(s,·)
        w = self._softmax(q_values[state], self.varde_params.varde_tau)

        scores = np.empty(self.na, dtype=np.float64)
        for a in range(self.na):
            N_sa = float(self.exp_visits[state, a].sum())  # total transitions observed from (s,a)

            # Force at least one pull for each action in this state
            if N_sa <= 0:
                scores[a] = np.inf
                continue

            sigma2 = max(float(m_values[state, a]), float(self.varde_params.varde_var_floor))
            scores[a] = (w[a] ** 2) * sigma2 / (N_sa * (N_sa + 1.0))

        # random tie-break among maxima
        mx = np.max(scores)
        best = np.flatnonzero(np.isclose(scores, mx))
        return int(np.random.choice(best))
