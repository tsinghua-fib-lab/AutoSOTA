# cbo.py — LBW-only CBO with annealing schedules and diagonal covariance option
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, List, Tuple, Optional

from geometry import (bw_exp, bw_log,
    lbw_weighted_average,
    _sym, _clip_eigs
)  # BW log/exp-backed


@dataclass
class GaussianParticle:
    """
    A particle represented by:
      - m : Euclidean mean (reference mean is fixed to 0, so no mean encode/decode)
      - V : BW tangent matrix at the *current* reference covariance S_ref
    """
    m: np.ndarray          # shape (d,)
    V: np.ndarray          # shape (d,d), symmetric tangent at S_ref


@dataclass
class CBOConfig:
    alpha: float = 25.0     # weight sharpness in softmin
    sigma: float = 0.2      # noise scale (for m and V updates)
    lmbda: float = 1.0      # pull toward LBW barycenter (for both m, V)
    dt: float = 0.05
    seed: int = 0
    n_update_base: int = 20  # how often to reset S_ref to current LBW barycenter

    # Annealing schedules (disabled by default; set sigma_decay/alpha_growth to enable)
    sigma_decay: float = 0.0     # sigma(t) = sigma_0 / (1 + sigma_decay * t)
    alpha_growth: float = 0.0    # alpha(t) = alpha_0 * (1 + alpha_growth * t)
    alpha_max: float = 1e5       # cap for alpha growth
    sigma_min: float = 0.01      # floor for sigma decay

    # Diagonal covariance restriction for high-dimensional problems
    diagonal: bool = False       # If True, only diagonal elements of V are updated

    def get_sigma(self, t: int) -> float:
        """Compute sigma at time step t (0-indexed)."""
        if self.sigma_decay <= 0:
            return self.sigma
        return max(self.sigma / (1.0 + self.sigma_decay * t), self.sigma_min)

    def get_alpha(self, t: int) -> float:
        """Compute alpha at time step t (0-indexed)."""
        if self.alpha_growth <= 0:
            return self.alpha
        return min(self.alpha * (1.0 + self.alpha_growth * t), self.alpha_max)


class LBWCBO:
    """
    Zeroth-order CBO in LBW coordinates with:
      - Reference mean fixed to 0 at all times (no mean encode/decode).
      - Particles are (m, V) where V lies in the BW tangent at current S_ref.
      - Only decode when we need covariances (for computing weights or re-encoding).
      - Every `n_update_base` steps, set S_ref to current LBW barycenter covariance and
        re-encode particle tangents V at the new S_ref.
      - We track only the LBW barycenter (m_bar, S_bar) at each step.
      - Supports sigma/alpha annealing and diagonal covariance restriction.

    NOTE (base-update invariance):
      When changing base from S_ref(old) -> S_ref(new) = S_bar, covariances of particles
      must remain unchanged. To ensure this, *decode at the old base*, re-encode at the
      new base, and validate by *decoding at the new base* when checking drift.
    """
    def __init__(self, energy: Callable[[np.ndarray, np.ndarray], float], cfg: CBOConfig):
        self.energy = energy
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        # Reference objects: mean fixed to 0, covariance set lazily.
        self._m_ref: np.ndarray | None = None  # zero vector once dimension is known
        self._S_ref: np.ndarray | None = None  # SPD reference covariance for BW log/exp

        self._t = 0  # iteration counter

    # -------------------------
    # Helpers
    # -------------------------
    def _current_sigma(self) -> float:
        return self.cfg.get_sigma(self._t)

    def _current_alpha(self) -> float:
        return self.cfg.get_alpha(self._t)

    def _ensure_refs(self, particles: List[GaussianParticle]):
        """Initialize reference mean (zeros) and a reasonable reference covariance."""
        if self._m_ref is None:
            d = particles[0].m.shape[0]
            self._m_ref = np.zeros(d)
        if self._S_ref is None:
            d = particles[0].m.shape[0]
            self._S_ref = np.eye(d)  # simple, safe default

    def _weights_and_barycenter(
        self, particles: List[GaussianParticle]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute softmin weights and LBW barycenter (m_bar, V_bar) using
        lbw_weighted_average((vm, V), w). With m_ref = 0, vm == m.
        Returns: w, m_bar, V_bar
        """
        alpha_t = self._current_alpha()

        # 1) Weights via energy(m_i, S_i) with S_i decoded from V_i at current S_ref
        vals = []
        for p in particles:
            S_i = bw_exp(self._S_ref, p.V)
            vals.append(self.energy(p.m, S_i))
        vals = np.asarray(vals)
        vals -= vals.min()
        w = np.exp(-alpha_t * vals)
        s = w.sum()
        w = w / s if s > 0 else np.ones_like(w) / len(w)

        # 2) LBW weighted average (vm = m since m_ref = 0)
        lbw_states = [(p.m, p.V) for p in particles]
        m_bar, V_bar = lbw_weighted_average(lbw_states, w)

        return w, m_bar, V_bar

    def _maybe_update_base(self, particles: List[GaussianParticle], V_bar: np.ndarray):
        """Every n_update_base steps, set S_ref := S_bar and re-encode particles."""
        if self.cfg.n_update_base <= 0:
            return particles
        if self._t > 0 and (self._t % self.cfg.n_update_base == 0):
            S_bar = bw_exp(self._S_ref, V_bar)

            new_particles: list[GaussianParticle] = []
            for p in particles:
                S = bw_exp(self._S_ref, p.V)
                V_new = bw_log(S_bar, S)
                S_newbase = bw_exp(S_bar, V_new)
                if not np.allclose(S, S_newbase, rtol=1e-8, atol=1e-10):
                    return  # numerical issue, skip update
                new_particles.append(GaussianParticle(p.m, V_new))

            self._S_ref = S_bar
            particles[:] = new_particles
            return particles
        else:
            return particles

    # -------------------------
    # Core CBO step
    # -------------------------
    def step(
        self, particles: List[GaussianParticle]
    ) -> Tuple[List[GaussianParticle], Tuple[np.ndarray, np.ndarray]]:
        """
        One CBO step. Returns (updated_particles, (m_bar, S_bar)).
        """
        self._ensure_refs(particles)

        sigma_t = self._current_sigma()
        diagonal = self.cfg.diagonal

        # Weights + barycenter (tangent)
        _, m_bar, V_bar = self._weights_and_barycenter(particles)

        # OU-type dynamics (Euclidean in m, linear in LBW tangent for V)
        out: list[GaussianParticle] = []

        # Matrix for noise preconditioning
        eig, Q = np.linalg.eigh(self._S_ref)
        D = (eig[:, None] + eig[None, :]) / 2

        for p in particles:
            # Mean update
            noise_m = sigma_t * self.rng.normal(size=p.m.shape)
            m_diff = m_bar - p.m
            m_next = p.m + self.cfg.dt * self.cfg.lmbda * m_diff \
                     + np.sqrt(self.cfg.dt) * (m_diff * noise_m)
            m_next = np.clip(m_next, -100, 100)

            # Tangent (V) update
            if diagonal:
                # Diagonal-only: restrict to diagonal elements
                V_diff = np.diag(np.diag(V_bar - p.V))
                noise_diag = sigma_t * self.rng.normal(size=p.V.shape[0])
                V_next = p.V + self.cfg.dt * self.cfg.lmbda * V_diff \
                         + np.sqrt(self.cfg.dt) * np.diag(np.diag(V_diff) * noise_diag)
                # Keep only diagonal
                V_next = np.diag(np.diag(V_next))
            else:
                # Full covariance update
                noise_V = _sym(self.rng.normal(size=p.V.shape))
                noise_V = sigma_t * (Q @ (noise_V / D) @ Q.T)
                V_diff = V_bar - p.V
                V_next = p.V + self.cfg.dt * self.cfg.lmbda * V_diff \
                         + np.sqrt(self.cfg.dt) * _sym(V_diff * noise_V)
                V_next = _sym(V_next)

            V_next = np.clip(V_next, -100, 100)
            out.append(GaussianParticle(m_next, V_next))

        particles[:] = out

        # Decode S_bar for trajectory (at current base)
        S_bar = bw_exp(self._S_ref, V_bar)

        # Maybe update base and re-encode tangents
        particles = self._maybe_update_base(particles, V_bar)

        self._t += 1
        return particles, (m_bar, S_bar)

    # -------------------------
    # Driver
    # -------------------------
    def run(self, particles: List[GaussianParticle], steps: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Run for `steps` iterations. Returns the trajectory of the LBW barycenter:
          [(m_bar_0, S_bar_0), (m_bar_1, S_bar_1), ...]
        """
        self._ensure_refs(particles)

        # Initial barycenter at t=0
        _, m_bar0, V_bar0 = self._weights_and_barycenter(particles)
        S_bar0 = bw_exp(self._S_ref, V_bar0)
        traj_bary: list[Tuple[np.ndarray, np.ndarray]] = [(m_bar0, S_bar0)]

        for _ in range(steps):
            particles, (m_bar, S_bar) = self.step(particles)
            traj_bary.append((m_bar, S_bar))

        return (traj_bary, particles)
