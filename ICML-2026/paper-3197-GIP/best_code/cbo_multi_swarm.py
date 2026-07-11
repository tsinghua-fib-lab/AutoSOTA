# cbo_multi_swarm.py

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Callable, List, Tuple

from cbo import LBWCBO, CBOConfig, GaussianParticle
from geometry import bw_exp, lbw_weighted_average, _sym, _clip_eigs


@dataclass
class MultiSwarmCBOConfig(CBOConfig):
    """
    Extra options for the multi-swarm variants.
    """
    min_mixture: float = 1e-300
    eig_floor: float = 1e-12
    clip_value: float = 100.0


class MultiSwarmLBWCBO(LBWCBO):
    """
    Base multi-swarm Gaussian CBO class.

    Shared mechanics:
      - no base update
      - other-swarm information uses previous-iteration swarm barycenters
      - anisotropic noise kept as in the current LBWCBO implementation

    Subclasses can override `_particle_swarm_score(...)` to change the
    objective used to compute within-swarm weights.
    """

    def __init__(
        self,
        potential: Callable[[np.ndarray], float],
        quad_rule: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]],
        cfg: MultiSwarmCBOConfig,
    ):
        # energy is not used directly here; weights are swarm-coupled
        super().__init__(energy=lambda m, S: 0.0, cfg=cfg)
        self.potential = potential
        self.quad_rule = quad_rule
        self.cfg: MultiSwarmCBOConfig = cfg

        # cached previous-iteration swarm barycenters in LBW coordinates:
        # [(mbar_k, Vbar_k), ...]
        self._prev_swarm_barycenters: list[tuple[np.ndarray, np.ndarray]] | None = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _softmin_weights(self, vals: np.ndarray) -> np.ndarray:
        vals = np.asarray(vals, dtype=float)
        vals = vals - np.min(vals)
        w = np.exp(-self.cfg.alpha * vals)
        s = np.sum(w)
        if s <= 0 or not np.isfinite(s):
            return np.ones_like(w) / len(w)
        return w / s

    def _gaussian_logpdf_batch(
        self,
        xs: np.ndarray,   # (nq, d)
        m: np.ndarray,    # (d,)
        S: np.ndarray     # (d, d)
    ) -> np.ndarray:
        """
        Vectorized log N(xs | m, S).
        """
        S = _clip_eigs(_sym(S), self.cfg.eig_floor, None)
        L = np.linalg.cholesky(S)
        Y = np.linalg.solve(L, (xs - m).T)   # (d, nq)
        quad = np.sum(Y * Y, axis=0)
        logdet = 2.0 * np.sum(np.log(np.diag(L)))
        d = m.size
        return -0.5 * (d * np.log(2.0 * np.pi) + logdet + quad)

    def _uniform_swarm_barycenters(
        self,
        swarms: List[List[GaussianParticle]]
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Initial fallback barycenters: uniform average inside each swarm.
        """
        barys = []
        for swarm in swarms:
            n = len(swarm)
            w = np.ones(n, dtype=float) / n
            states = [(p.m, p.V) for p in swarm]
            barys.append(lbw_weighted_average(states, w))
        return barys

    def _particle_swarm_score(
        self,
        p: GaussianParticle,
        swarm_index: int,
        prev_barys_cov: list[tuple[np.ndarray, np.ndarray]],
    ) -> float:
        """
        Hook for subclasses.

        Must return the scalar score used in the within-swarm softmin.
        """
        raise NotImplementedError

    def _compute_swarm_consensuses(
        self,
        swarms: List[List[GaussianParticle]],
        prev_barys: list[tuple[np.ndarray, np.ndarray]],
    ) -> tuple[
        list[np.ndarray],                        # weights per swarm
        list[np.ndarray],                        # scores per swarm
        list[tuple[np.ndarray, np.ndarray]],     # current barycenters in LBW coords
        list[tuple[np.ndarray, np.ndarray]],     # current barycenters decoded (mbar, Sbar)
    ]:
        """
        Compute current within-swarm weights using previous-iteration consensuses
        from the other swarms.
        """
        prev_barys_cov = [(mbar, bw_exp(self._S_ref, Vbar)) for (mbar, Vbar) in prev_barys]

        weights_all = []
        scores_all = []
        barys_all = []

        for k, swarm in enumerate(swarms):
            scores_k = np.array(
                [self._particle_swarm_score(p, k, prev_barys_cov) for p in swarm],
                dtype=float
            )
            w_k = self._softmin_weights(scores_k)

            states = [(p.m, p.V) for p in swarm]
            mbar_k, Vbar_k = lbw_weighted_average(states, w_k)

            scores_all.append(scores_k)
            weights_all.append(w_k)
            barys_all.append((mbar_k, Vbar_k))

        barys_all_cov = [(mbar, bw_exp(self._S_ref, Vbar)) for (mbar, Vbar) in barys_all]
        return weights_all, scores_all, barys_all, barys_all_cov

    # ------------------------------------------------------------------
    # Core step
    # ------------------------------------------------------------------
    def step(
        self,
        swarms: List[List[GaussianParticle]]
    ) -> tuple[
        List[List[GaussianParticle]],
        list[tuple[np.ndarray, np.ndarray]],
    ]:
        """
        One step of the simplified multi-swarm dynamics.
        """
        flat = [p for swarm in swarms for p in swarm]
        self._ensure_refs(flat)

        K = len(swarms)
        if K <= 1:
            raise ValueError("Expected at least two swarms.")

        if self._prev_swarm_barycenters is None:
            prev_barys = self._uniform_swarm_barycenters(swarms)
        else:
            prev_barys = self._prev_swarm_barycenters

        _, _, swarm_barys, swarm_barys_cov = self._compute_swarm_consensuses(swarms, prev_barys)

        eig, Q = np.linalg.eigh(self._S_ref)
        eig = np.clip(eig, self.cfg.eig_floor, None)
        D = (eig[:, None] + eig[None, :]) / 2.0

        out_swarms: list[list[GaussianParticle]] = []

        for k, swarm in enumerate(swarms):
            mbar_k, Vbar_k = swarm_barys[k]
            out_swarm: list[GaussianParticle] = []

            for p in swarm:
                noise_m = self.cfg.sigma * self.rng.normal(size=p.m.shape)
                m_diff = mbar_k - p.m

                noise_V = _sym(self.rng.normal(size=p.V.shape))
                noise_V = self.cfg.sigma * (Q @ (noise_V / D) @ Q.T)
                V_diff = Vbar_k - p.V

                d_i = np.sqrt(np.linalg.norm(V_diff, 'fro')**2 + np.linalg.norm(m_diff)**2)

                m_next = (
                    p.m
                    + self.cfg.dt * self.cfg.lmbda * m_diff
                    + np.sqrt(self.cfg.dt) * (m_diff * noise_m)
                )
                m_next = np.clip(m_next, -self.cfg.clip_value, self.cfg.clip_value)


                V_next = (
                    p.V
                    + self.cfg.dt * self.cfg.lmbda * V_diff
                    + np.sqrt(self.cfg.dt) * _sym(V_diff * noise_V)
                )
                V_next = np.clip(V_next, -self.cfg.clip_value, self.cfg.clip_value)
                V_next = _sym(V_next)

                out_swarm.append(GaussianParticle(m_next, V_next))

            out_swarms.append(out_swarm)

        self._prev_swarm_barycenters = swarm_barys
        self._t += 1
        return out_swarms, swarm_barys_cov

    # ------------------------------------------------------------------
    # Driver
    # ------------------------------------------------------------------
    def run(
        self,
        swarms: List[List[GaussianParticle]],
        steps: int
    ) -> tuple[
        list[list[tuple[np.ndarray, np.ndarray]]],
        List[List[GaussianParticle]]
    ]:
        """
        Run for `steps` iterations.

        Returns
        -------
        traj : list of length steps+1
            traj[t][k] = (mbar_k(t), Sbar_k(t))

        swarms : final particles
        """
        flat = [p for swarm in swarms for p in swarm]
        self._ensure_refs(flat)

        K = len(swarms)
        if K <= 1:
            raise ValueError("Expected at least two swarms.")

        barys0 = self._uniform_swarm_barycenters(swarms)
        self._prev_swarm_barycenters = barys0
        barys0_cov = [(mbar, bw_exp(self._S_ref, Vbar)) for (mbar, Vbar) in barys0]

        traj: list[list[tuple[np.ndarray, np.ndarray]]] = [barys0_cov]

        for _ in range(steps):
            swarms, swarm_barys_cov = self.step(swarms)
            traj.append(swarm_barys_cov)

        return traj, swarms


class MultiSwarmKLLBWCBO(MultiSwarmLBWCBO):
    r"""
    Multi-swarm Gaussian CBO where particle scores are based on the KL of the
    surrogate mixture

        P_{k,j} = (1/K) [ q_{k,j} + sum_{ell != k} \bar q_ell^{prev} ].

    The score used in the within-swarm softmin is

        J_{k,j} = E_{P_{k,j}}[ log P_{k,j}(X) + V(X) ],

    which equals KL(P_{k,j} || pi) up to the additive constant log Z_pi,
    irrelevant for the softmin weights.

    Notes
    -----
    - The other-swarm Gaussians are the previous-iteration barycenters.
    - The drift/noise dynamics are inherited unchanged from MultiSwarmLBWCBO.
    - Each component in the surrogate mixture has weight 1/K.
    """

    def _surrogate_components(
        self,
        p: GaussianParticle,
        swarm_index: int,
        prev_barys_cov: list[tuple[np.ndarray, np.ndarray]],
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Return the K Gaussian components of the surrogate mixture:
            [candidate particle] + [other swarm barycenters].
        """
        S_p = bw_exp(self._S_ref, p.V)
        comps = [(p.m, S_p)]

        for ell, (mbar, Sbar) in enumerate(prev_barys_cov):
            if ell != swarm_index:
                comps.append((mbar, Sbar))

        return comps

    def _log_component_mixture_batch(
        self,
        xs: np.ndarray,
        components: list[tuple[np.ndarray, np.ndarray]],
    ) -> np.ndarray:
        """
        Compute log of the equal-weight Gaussian mixture determined by `components`.
        """
        L = np.stack(
            [self._gaussian_logpdf_batch(xs, m, S) for (m, S) in components],
            axis=1
        )  # shape (nq, K)

        a = np.max(L, axis=1, keepdims=True)
        out = a[:, 0] + np.log(np.sum(np.exp(L - a), axis=1)) - np.log(len(components))

        return np.maximum(out, np.log(self.cfg.min_mixture))

    def _component_free_energy(
        self,
        m: np.ndarray,
        S: np.ndarray,
        components: list[tuple[np.ndarray, np.ndarray]],
    ) -> float:
        """
        Approximate E_{N(m,S)}[ V(X) + log P(X) ] where P is the surrogate mixture.
        """
        xs, ws = self.quad_rule(m, S)
        ws = np.asarray(ws, dtype=float)
        ws = ws / np.sum(ws)

        pot = np.array([self.potential(x) for x in xs], dtype=float)
        log_mix = self._log_component_mixture_batch(xs, components)

        return float(np.dot(ws, pot + log_mix))

    def _particle_swarm_score(
        self,
        p: GaussianParticle,
        swarm_index: int,
        prev_barys_cov: list[tuple[np.ndarray, np.ndarray]],
    ) -> float:
        """
        Score particle p by the surrogate-mixture KL:
            J_{k,j} = E_{P_{k,j}}[ log P_{k,j}(X) + V(X) ].

        Since P_{k,j} is an equal-weight mixture of K Gaussians, we compute this as
        the arithmetic mean of the component expectations.
        """
        components = self._surrogate_components(p, swarm_index, prev_barys_cov)

        vals = [
            self._component_free_energy(m, S, components)
            for (m, S) in components
        ]

        return float(np.mean(vals))