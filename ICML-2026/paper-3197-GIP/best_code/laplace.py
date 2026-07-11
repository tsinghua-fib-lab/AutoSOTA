
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

@dataclass
class LaplaceConfig:
    maxiter: int = 200
    tol: float = 1e-8
    step: float = 1.0

def laplace_approx(x0: np.ndarray, V, gradV, hessV, cfg: LaplaceConfig=LaplaceConfig()):
    x = x0.copy()
    for _ in range(cfg.maxiter):
        g = gradV(x)
        H = hessV(x)
        H = 0.5*(H+H.T)
        w, U = np.linalg.eigh(H)
        w = np.clip(w, 1e-6, None)
        Hpd = (U * w) @ U.T
        step_dir = np.linalg.solve(Hpd, g)
        x_next = x - cfg.step * step_dir
        if np.linalg.norm(x_next - x) < cfg.tol * (1 + np.linalg.norm(x)):
            x = x_next
            break
        x = x_next
    H = hessV(x)
    H = 0.5*(H+H.T)
    w, U = np.linalg.eigh(H)
    w = np.clip(w, 1e-6, None)
    S = U @ (np.diag(1.0/w)) @ U.T
    return x, S
