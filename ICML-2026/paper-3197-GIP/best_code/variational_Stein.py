# variational_Stein.py — Gaussian-SVGD update with kernel K1(x, y) = x^T y + 1
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Callable, Tuple

@dataclass
class VGPConfig:
    dt: float = 0.05
    steps: int = 200
    seed: int = 0
    eig_floor: float = 1e-5  # SPD safety after update

class VariationalStein:
    """
    Time stepping scheme for Gaussian-SVGD with kernel K1(x, y) = x^T y + 1:

        m' = (I - Γ Σ) m - (1 + m^T m) g
        Σ' = G Σ + Σ G^T,   with   G = I - Γ Σ - g m^T

    where
        g = E_q[∇U(X)]
        Γ = E_q[∇²U(X)]

    We discretize as
        m_{n+1} = m_n + dt * F_n
        M_{n+1} = I + dt * G_n
        Σ_{n+1} = M_{n+1} Σ_n M_{n+1}^T

    You must provide two callables:
      - expect_grad(m, S) -> E_q[∇U(X)]   (shape d,)
      - expect_hess(m, S) -> E_q[∇²U(X)]  (shape d×d)
    """

    def __init__(
        self,
        expect_grad: Callable[[np.ndarray, np.ndarray], np.ndarray],
        expect_hess: Callable[[np.ndarray, np.ndarray], np.ndarray],
        cfg: VGPConfig,
    ):
        self.expect_grad = expect_grad
        self.expect_hess = expect_hess
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

    @staticmethod
    def _project_spd(S: np.ndarray, eps: float) -> np.ndarray:
        S = 0.5 * (S + S.T)
        w, U = np.linalg.eigh(S)
        w = np.clip(w, eps, None)
        return U @ np.diag(w) @ U.T

    def step(self, m: np.ndarray, S: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        S = 0.5 * (S + S.T)
        d = S.shape[0]
        I = np.eye(d)

        # 1) Expectations under q = N(m, S)
        g = self.expect_grad(m, S)      # E[∇U(X)]     (d,)
        H = self.expect_hess(m, S)      # E[∇²U(X)]    (d,d)
        H = 0.5 * (H + H.T)             # numerical symmetry

        # 2) Gaussian-SVGD drift
        F = (I - H @ S) @ m - (1.0 + m @ m) * g
        G = I - H @ S - np.outer(g, m)

        # 3) Time stepping
        m_next = m + self.cfg.dt * F

        M_next = I + self.cfg.dt * G
        S_next = M_next @ S @ M_next.T
        S_next = self._project_spd(S_next, self.cfg.eig_floor)

        return m_next, S_next

    def run(self, m0: np.ndarray, S0: np.ndarray):
        traj = [(m0.copy(), S0.copy())]
        m, S = m0.copy(), S0.copy()
        for _ in range(self.cfg.steps):
            m, S = self.step(m, S)
            traj.append((m.copy(), S.copy()))
        return traj