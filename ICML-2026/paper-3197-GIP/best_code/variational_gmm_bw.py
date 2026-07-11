# variational_gmm_bw.py — Explicit Euler discretization of the N-particle BW flow
from __future__ import annotations
import numpy as np
from typing import Callable, Tuple

from variational_gp import VGPConfig


class VariationalGMMBW:
    """
    Explicit Euler scheme for the interacting BW flow with fixed-weight Gaussian mixture

        q_i = N(m_i, S_i),     P_mu = (1/N) sum_i q_i

        m_i' = -c * E_{q_i}[ ∇ log(P_mu / pi)(X) ]
        S_i' = -c * ( C_i + C_i^T )

    where
        C_i = E_{q_i}[ ∇ log(P_mu / pi)(X) (X - m_i)^T ].

    Equivalently, in Hessian form,
        S_i' = -c * ( A_i S_i + S_i A_i ),
        A_i  = E_{q_i}[ ∇² log(P_mu / pi)(X) ].

    You must provide two callables:

      - expect_grad(M, S)  -> G   with shape (N, d)
            G[i] = E_{q_i}[ ∇ log(P_mu / pi)(X) ]

      - expect_cross(M, S) -> C   with shape (N, d, d)
            C[i] = E_{q_i}[ ∇ log(P_mu / pi)(X) (X - m_i)^T ]

    Notes
    -----
    * M has shape (N, d)
    * S has shape (N, d, d)
    * flow_scale = 1.0 corresponds to the averaged product metric convention
    * flow_scale = 1.0 / N corresponds to the unnormalized product metric convention
    """

    def __init__(
        self,
        expect_grad: Callable[[np.ndarray, np.ndarray], np.ndarray],
        expect_cross: Callable[[np.ndarray, np.ndarray], np.ndarray],
        cfg: VGPConfig,
        flow_scale: float = 1.0,
    ):
        self.expect_grad = expect_grad
        self.expect_cross = expect_cross
        self.cfg = cfg
        self.flow_scale = float(flow_scale)
        self.rng = np.random.default_rng(cfg.seed)

    @staticmethod
    def _project_spd(S: np.ndarray, eps: float) -> np.ndarray:
        S = 0.5 * (S + S.T)
        w, U = np.linalg.eigh(S)
        w = np.clip(w, eps, None)
        return U @ np.diag(w) @ U.T

    @staticmethod
    def _sym_batch(S: np.ndarray) -> np.ndarray:
        return 0.5 * (S + np.swapaxes(S, -1, -2))

    @staticmethod
    def _check_shapes(M: np.ndarray, S: np.ndarray) -> Tuple[int, int]:
        if M.ndim != 2:
            raise ValueError(f"M must have shape (N, d), got {M.shape}")
        if S.ndim != 3:
            raise ValueError(f"S must have shape (N, d, d), got {S.shape}")

        N, d = M.shape
        if S.shape != (N, d, d):
            raise ValueError(f"S must have shape {(N, d, d)}, got {S.shape}")
        return N, d

    def step(self, M: np.ndarray, S: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        M = np.asarray(M, dtype=float)
        S = np.asarray(S, dtype=float)
        N, d = self._check_shapes(M, S)

        # Symmetrize each covariance
        S = self._sym_batch(S)

        # 1) Expectations under q_i = N(m_i, S_i), coupled through P_mu
        G = np.asarray(self.expect_grad(M, S), dtype=float)    # shape (N, d)
        C = np.asarray(self.expect_cross(M, S), dtype=float)   # shape (N, d, d)

        if G.shape != (N, d):
            raise ValueError(f"expect_grad must return shape {(N, d)}, got {G.shape}")
        if C.shape != (N, d, d):
            raise ValueError(f"expect_cross must return shape {(N, d, d)}, got {C.shape}")

        # harmless symmetrization, same style as your single-particle code
        C = self._sym_batch(C)

        # 2) Explicit Euler updates
        M_next = M - self.cfg.dt * self.flow_scale * G

        S_next = np.empty_like(S)
        for i in range(N):
            Sdot_i = -self.flow_scale * (C[i] + C[i].T)
            S_i_next = S[i] + self.cfg.dt * Sdot_i
            S_next[i] = self._project_spd(S_i_next, self.cfg.eig_floor)

        return M_next, S_next

    def run(self, M0: np.ndarray, S0: np.ndarray):
        M0 = np.asarray(M0, dtype=float)
        S0 = np.asarray(S0, dtype=float)
        self._check_shapes(M0, S0)

        traj = [(M0.copy(), S0.copy())]
        M, S = M0.copy(), S0.copy()

        for _ in range(self.cfg.steps):
            M, S = self.step(M, S)
            traj.append((M.copy(), S.copy()))

        return traj