# variational_gp.py — Explicit Euler discretization of your new covariance ODE (no Hessian)
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Callable, Tuple

@dataclass
class VGPConfig:
    dt: float = 0.05
    steps: int = 200
    seed: int = 0
    eig_floor: float = 1e-10  # SPD safety after Euler step

class VariationalGP:
    """
    Explicit Euler scheme for:
        m'      = - E_q[ ∇U(X) ]
        Σ'      =  2 I - ( C + C^T ),  with  C = E_q[ ∇U(X) (X-m)^T ].

    You must provide two callables:
      - expect_grad(m, S) -> E_q[∇U(X)]      (shape d,)
      - expect_cross(m, S) -> E_q[∇U(X)(X-m)^T] (shape d×d)
    """

    def __init__(
        self,
        expect_grad: Callable[[np.ndarray, np.ndarray], np.ndarray],
        expect_cross: Callable[[np.ndarray, np.ndarray], np.ndarray],
        cfg: VGPConfig,
    ):
        self.expect_grad = expect_grad
        self.expect_cross = expect_cross
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

    @staticmethod
    def _project_spd(S: np.ndarray, eps: float) -> np.ndarray:
        S = 0.5 * (S + S.T)
        w, U = np.linalg.eigh(S)
        w = np.clip(w, eps, None)
        return (U @ np.diag(w) @ U.T)

    def step(self, m: np.ndarray, S: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        S = 0.5 * (S + S.T)

        # 1) Expectations under q = N(m, S)
        g = self.expect_grad(m, S)      # E[∇U(X)]          (d,)
        C = self.expect_cross(m, S)     # E[∇U(X)(X-m)^T]   (d,d)
        C = 0.5 * (C + C.T)             # harmless symmetrization 

        # 2) Explicit Euler updates
        m_next = m - self.cfg.dt * g

        Sdot   = 2.0 * np.eye(S.shape[0]) - (C + C.T)  # = 2I - 2*sym(C)
        S_next = S + self.cfg.dt * Sdot
        S_next = self._project_spd(S_next, self.cfg.eig_floor)  # keep SPD

        return m_next, S_next

    def run(self, m0: np.ndarray, S0: np.ndarray):
        traj = [(m0.copy(), S0.copy())]
        m, S = m0.copy(), S0.copy()
        for _ in range(self.cfg.steps):
            m, S = self.step(m, S)
            traj.append((m.copy(), S.copy()))
        return traj
