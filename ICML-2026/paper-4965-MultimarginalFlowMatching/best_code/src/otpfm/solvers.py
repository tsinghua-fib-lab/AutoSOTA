"""
Fixed-point solvers for the self-consistent X_tk equations in OTP-FM.

Solvers:

- :class:`DirectSolver`: Direct solution for W2-based potentials.
- :class:`PicardSolver`: Simple Picard iteration.
- :class:`AndersonSolver`: Improves upon the simple Picard solver using Anderson acceleration.
- :class:`AndersonSafe`: Safeguarded Anderson acceleration with adaptive damping (default recommendation for non-W2 potentials).
- :class:`AndersonHomotopy`: Homotopy continuation with Anderson acceleration (can be helpful for high-strength potentials).
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import Tensor

from .potentials import Potential

logger = logging.getLogger(__name__)


@dataclass
class FixedPointSolver(ABC):
    """
    Base class for fixed-point solvers.

    Solves the coupled system for K potentials (Eq. 18 of :cite:t:`kansal2026otpfm`):
        X(t_i) = X_base(t_i) + Σ_j x_time_dep_j(t_i) * dV_j(xm_j, X(t_j))
    """

    picard_steps: int

    def compute_dV_tks(
        self,
        X_tks: Tensor,  # (K, bs, d)
        potentials: list[Potential],
        xms: Tensor,  # (K, bs, d)
    ) -> Tensor:
        """Compute dV for all potentials in parallel. Returns (K, bs, d)."""
        K = X_tks.shape[0]
        # TODO: could potentially batch this if potentials support it
        dV_tks = torch.stack([potentials[k].dV_tk(xms[k], X_tks[k]) for k in range(K)])
        return dV_tks

    def G(
        self,
        X_tks: Tensor,  # (K, bs, d)
        X_bases: Tensor,  # (K, bs, d)
        x_time_deps: Tensor,  # (K, K, bs, 1) - x_time_deps[i, j] for potential j at time t_i
        potentials: list[Potential],
        xms: Tensor,  # (K, bs, d)
        dV_tks: Tensor | None = None,  # If provided, skip recomputing potential gradients
        alpha: float = 1.0,  # Scaling factor for corrections (used in homotopy)
    ) -> tuple[Tensor, Tensor]:
        """
        Fixed-point map G(X) for K potentials (Eq. 21 of :cite:t:`kansal2026otpfm`).

        G_i(X) = X_base_i + alpha * Σ_j x_time_dep[i,j] * dV_j

        Args:
            dV_tks: If provided, use this instead of recomputing. Must correspond to X_tks.
            alpha: Scaling factor for corrections (1.0 for normal, <1.0 for homotopy).

        Returns:
            G_z: (K, bs, d) updated positions
            dV_tks: (K, bs, d) potential gradients
        """
        if dV_tks is None:
            dV_tks = self.compute_dV_tks(X_tks, potentials, xms)

        # corrections[i] = Σ_j x_time_deps[i, j] * dV_tks[j]
        corrections = torch.einsum("ijb...,jb...->ib...", x_time_deps, dV_tks)
        G_z = X_bases + alpha * corrections

        return G_z, dV_tks

    def compute_residuals(
        self,
        X_tks: Tensor,  # (K, bs, d)
        G_z: Tensor,  # (K, bs, d)
    ) -> tuple[Tensor, Tensor]:
        """
        Compute residuals r_i = G_i(X) - X_i.

        Returns:
            residuals: (K, bs, d)
            residual_norms: (K, bs) per-sample norms for each potential
        """
        residuals = G_z - X_tks
        # Per-potential, per-sample norms
        residual_norms = residuals.reshape(residuals.shape[0], residuals.shape[1], -1).norm(dim=-1)
        return residuals, residual_norms

    def _flatten(self, X_tks: Tensor) -> Tensor:
        """Flatten (K, bs, d) to (bs, K*d) for Anderson."""
        K, bs, d = X_tks.shape
        return X_tks.permute(1, 0, 2).reshape(bs, -1)

    def _unflatten(self, X_flat: Tensor, K: int, d: int) -> Tensor:
        """Unflatten (bs, K*d) back to (K, bs, d)."""
        bs = X_flat.shape[0]
        return X_flat.reshape(bs, K, d).permute(1, 0, 2)

    @abstractmethod
    def __call__(
        self,
        X_inits: Tensor,  # (K, bs, d)
        X_bases: Tensor,  # (K, bs, d)
        x_time_deps: Tensor,  # (K, K, bs, 1)
        potentials: list[Potential],
        xms: Tensor,  # (K, bs, d)
        debug: bool = False,
        **kwargs,  # For solver-specific precomputed values
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Solve for X_tks.

        Args:
            X_inits: (K, bs, d) initial guess for positions (e.g. from the consistency model)
            X_bases: (K, bs, d) base positions
            x_time_deps: (K, K, bs, 1) X time-dependent factors evaluated at for each X_tk at each time t_k (K x K)
            potentials: list[Potential] list of potentials
            xms: (K, bs, d) intermediate marginal samples
            debug: whether to print debug information
            **kwargs: additional keyword arguments for the solver

        Returns:
            X_tks: (K, bs, d) final positions
            dV_tks: (K, bs, d) potential gradients
            residual_norms: (K, bs) per-sample residual norms
        """
        pass


@dataclass
class DirectSolver(FixedPointSolver):
    """
    Direct solution for the linear gradients of W2 potentials (Eq. 20 of :cite:t:`kansal2026otpfm`).

    The fixed-point equation:
        z_i = z_base_i + Σ_j x_time_dep[i,j] * w_j * (z_j - xm_j)

    Rearranges to a linear system:
        (I - A) * X = X_base - A * xm

    where A[i,j] = x_time_dep[i,j] * w_j

    This solver computes the exact solution in one step, optionally using precomputed matrices for efficiency.
    """

    def __call__(
        self,
        X_inits: Tensor,  # Not used by direct solver
        X_bases: Tensor,  # (K, bs, d)
        x_time_deps: Tensor,  # (K, K, 1, 1) or (K, K, bs, 1)
        potentials: list[Potential],
        xms: Tensor,  # (K, bs, d)
        debug: bool = False,
        A_matrix: Tensor | None = None,  # Precomputed A matrix (K, K)
        I_minus_A_inv: Tensor | None = None,  # Precomputed (I - A)^{-1} (K, K)
        **kwargs,
    ) -> tuple[Tensor, Tensor, Tensor]:
        K, bs, d = X_bases.shape
        device = X_bases.device
        dtype = X_bases.dtype

        # Use precomputed matrices if available, otherwise compute
        if A_matrix is not None and I_minus_A_inv is not None:
            A = A_matrix.to(dtype=dtype)
            I_minus_A_inv_local = I_minus_A_inv.to(dtype=dtype)
        else:
            # Fallback: compute A and inverse on the fly
            strengths = torch.tensor([p.strength for p in potentials], device=device, dtype=dtype)
            A = x_time_deps[:, :, 0, 0] * strengths.unsqueeze(0)  # (K, K)
            I_minus_A = torch.eye(K, device=device, dtype=dtype) - A
            I_minus_A_inv_local = torch.linalg.inv(I_minus_A)

        # rhs = z_base - A * xm, computed via einsum
        # A: (K, K), xms: (K, bs, d) -> A_xm: (K, bs, d)
        A_xm = torch.einsum("ij,jbd->ibd", A, xms)
        rhs = X_bases - A_xm  # (K, bs, d)

        # Solve for X: X = (I-A)^{-1} * rhs
        X_tks = torch.einsum("ij,jbd->ibd", I_minus_A_inv_local, rhs)

        dV_tks = self.compute_dV_tks(X_tks, potentials, xms)

        # Verify solution (residual should be ~0 at machine precision)
        G_z, _ = self.G(X_tks, X_bases, x_time_deps, potentials, xms, dV_tks)
        _, residual_norms = self.compute_residuals(X_tks, G_z)

        if debug:
            logger.debug(
                f"DirectSolver: residual={residual_norms.mean():.2e}, max={residual_norms.max():.2e}"
            )

        return X_tks, dV_tks, residual_norms


@dataclass
class PicardSolver(FixedPointSolver):
    """Simple Picard iterations."""

    def __call__(
        self,
        X_inits: Tensor,  # (K, bs, d)
        X_bases: Tensor,  # (K, bs, d)
        x_time_deps: Tensor,  # (K, K, bs, 1)
        potentials: list[Potential],
        xms: Tensor,  # (K, bs, d)
        debug: bool = False,
        **kwargs,
    ) -> tuple[Tensor, Tensor, Tensor]:
        X_tks = X_inits

        for k in range(self.picard_steps):
            G_z, dV_tks = self.G(X_tks, X_bases, x_time_deps, potentials, xms)
            _, residual_norms = self.compute_residuals(X_tks, G_z)

            if debug:
                logger.debug(f"Picard step {k}: r={residual_norms[:, 0]}")

            X_tks = G_z

        # Final evaluation
        G_z, dV_tks = self.G(X_tks, X_bases, x_time_deps, potentials, xms)
        _, residual_norms = self.compute_residuals(X_tks, G_z)

        return X_tks, dV_tks, residual_norms


@dataclass
class AndersonSolver(FixedPointSolver):
    """
    Improves upon the simple Picard solver using Anderson acceleration;
    i.e., using the history of previous iterations to choose a more optimal step.

    Rank-1 acceleration (exact formula):
        r_k = G(z_k) - z_k                          (residual)
        Δz_k = z_k - z_{k-1}
        Δr_k = r_k - r_{k-1}
        γ_k = (r_k · Δr_k) / ||Δr_k||²              (mixing coefficient)
        z_{k+1} = z_k + β*r_k - β*γ_k*Δz_k          (Anderson update)

    Rank-m acceleration (requires solving a least-squares problem):
        ΔZ = [Δz_{k-m+1}, ..., Δz_k]                (history of m steps)
        ΔR = [Δr_{k-m+1}, ..., Δr_k]
        γ = argmin ||r_k - ΔR @ γ||²                (least-squares)
        z_{k+1} = z_k + β*r_k - β*(ΔZ @ γ)

    Falls back to Picard (z_{k+1} = z_k + β * r_k) when history is insufficient
    or the least-squares problem is ill-conditioned.
    """

    beta: float = 1.0
    eps: float = 1e-8
    rank: int = 1

    def _anderson_step_rank1(
        self,
        X_flat: Tensor,
        r: Tensor,
        X_prev: Tensor,
        r_prev: Tensor,
        beta: float,
        clamp_gamma: bool = False,
    ) -> Tensor:
        """
        Rank-1 Anderson acceleration using a simple scalar mixing coefficient.

        Computes:
            γ = (r · Δr) / ||Δr||²
            z_{k+1} = z_k + β*r - β*γ*Δz

        Falls back to Picard if Δr is too small.
        """
        dz = X_flat - X_prev
        dr = r - r_prev

        num = (r * dr).sum(dim=-1)
        den = (dr * dr).sum(dim=-1)

        use_picard = den < self.eps
        den = den.clamp_min(self.eps)
        gamma = (num / den).unsqueeze(-1)

        if clamp_gamma:
            gamma = torch.clamp(gamma, -2.0, 2.0)

        X_anderson = X_flat + beta * r - beta * gamma * dz
        X_picard = X_flat + beta * r

        return torch.where(use_picard.unsqueeze(-1), X_picard, X_anderson)

    def _anderson_step_rankm(
        self,
        X_flat: Tensor,
        r: Tensor,
        dz_list: list[Tensor],
        dr_list: list[Tensor],
        beta: float,
        clamp_gamma: bool = False,
        debug: bool = False,
    ) -> Tensor | None:
        """
        Rank-m Anderson acceleration using least-squares to find mixing coefficients.

        Solves:
            γ = argmin ||r - ΔR @ γ||²   via normal equations (ΔR^T ΔR) γ = ΔR^T r
            z_{k+1} = z_k + β*r - β*(ΔZ @ γ)

        Returns None if the solve fails (caller should fall back to Picard).
        """
        m = len(dz_list)
        DZ = torch.stack(dz_list, dim=-1)  # (bs, D, m)
        DR = torch.stack(dr_list, dim=-1)  # (bs, D, m)

        # Compute ΔR^T ΔR: (bs, m, m)
        RTR = torch.bmm(DR.transpose(1, 2), DR)

        # Compute ΔR^T r: (bs, m)
        RTr = torch.bmm(DR.transpose(1, 2), r.unsqueeze(-1)).squeeze(-1)

        # Regularize for numerical stability
        RTR = RTR + self.eps * torch.eye(m, device=RTR.device, dtype=RTR.dtype)

        # Solve (ΔR^T ΔR) γ = ΔR^T r
        try:
            if debug:
                logger.debug(f"Solving least-squares problem: RTR={RTR.shape}, RTr={RTr.shape}")
            gamma = torch.linalg.solve(RTR, RTr)  # (bs, m)
            if debug:
                logger.debug(f"Solved least-squares problem: gamma={gamma[0]}, ({gamma.shape})")
        except RuntimeError:
            return None  # Signal to fall back to Picard

        if clamp_gamma:
            gamma = torch.clamp(gamma, -2.0, 2.0)

        # z_{k+1} = z_k + β*r - β*(ΔZ @ γ)
        DZ_gamma = torch.bmm(DZ, gamma.unsqueeze(-1)).squeeze(-1)  # (bs, D)
        return X_flat + beta * r - beta * DZ_gamma

    def _anderson_step(
        self,
        X_flat: Tensor,
        G_flat: Tensor,
        X_hist: list[Tensor],
        r_hist: list[Tensor],
        beta: float | None = None,
        clamp_gamma: bool = False,
        debug: bool = False,
    ) -> tuple[Tensor, Tensor, list[Tensor], list[Tensor]]:
        """
        Single Anderson acceleration step - dispatches to rank-1 or rank-m.

        Args:
            X_flat: Current iterate (bs, D) where D = K*d
            G_flat: Fixed-point map output (bs, D)
            X_hist: List of up to m previous iterates
            r_hist: List of up to m previous residuals
            beta: Step size (defaults to self.beta)
            clamp_gamma: Whether to clamp gamma coefficients

        Returns:
            X_next: Next iterate (bs, D)
            r: Current residual (bs, D)
            X_hist_new: Updated history for next iteration
            r_hist_new: Updated history for next iteration
        """
        if beta is None:
            beta = self.beta

        r = G_flat - X_flat

        # Fall back to Picard if no history
        if len(X_hist) == 0:
            X_next = X_flat + beta * r
        else:
            # Build difference vectors from history
            dz_list = []
            dr_list = []

            # Consecutive differences in history
            for i in range(1, len(X_hist)):
                dz_list.append(X_hist[i] - X_hist[i - 1])
                dr_list.append(r_hist[i] - r_hist[i - 1])

            # Most recent difference: X_flat - X_hist[-1] and r - r_hist[-1]
            dz_list.append(X_flat - X_hist[-1])
            dr_list.append(r - r_hist[-1])

            m_actual = len(dz_list)

            if m_actual == 1:
                # Rank-1: simple scalar formula
                X_next = self._anderson_step_rank1(
                    X_flat, r, X_hist[-1], r_hist[-1], beta, clamp_gamma
                )
            else:
                # Rank-m: least-squares
                X_next = self._anderson_step_rankm(
                    X_flat, r, dz_list, dr_list, beta, clamp_gamma, debug
                )
                if X_next is None:
                    # Fall back to Picard if solve failed
                    X_next = X_flat + beta * r

        # Update history (keep last `rank` entries)
        X_hist_new = (X_hist + [X_flat])[-self.rank :]
        r_hist_new = (r_hist + [r])[-self.rank :]

        return X_next, r, X_hist_new, r_hist_new

    def __call__(
        self,
        X_inits: Tensor,  # (K, bs, d)
        X_bases: Tensor,  # (K, bs, d)
        x_time_deps: Tensor,  # (K, K, bs, 1)
        potentials: list[Potential],
        xms: Tensor,  # (K, bs, d)
        debug: bool = False,
        **kwargs,
    ) -> tuple[Tensor, Tensor, Tensor]:
        X_tks = X_inits
        K, bs, d = X_tks.shape

        X_hist: list[Tensor] = []
        r_hist: list[Tensor] = []

        for k in range(self.picard_steps):
            G_z, dV_tks = self.G(X_tks, X_bases, x_time_deps, potentials, xms)

            X_flat = self._flatten(X_tks)
            G_flat = self._flatten(G_z)

            X_next, _, X_hist, r_hist = self._anderson_step(
                X_flat, G_flat, X_hist, r_hist, debug=debug
            )

            X_tks = self._unflatten(X_next, K, d)

            if debug:
                r_norm = self._unflatten(G_flat - X_flat, K, d).norm(dim=-1)
                logger.debug(
                    f"Anderson(m={self.rank}) step {k}: prev r={r_norm[:, 0]}, prev dV_tk={dV_tks[:, 0].squeeze()}, new X_tk={X_tks[:, 0].squeeze()}"
                )

        # Final evaluation
        G_z, dV_tks = self.G(X_tks, X_bases, x_time_deps, potentials, xms)
        _, residual_norms = self.compute_residuals(X_tks, G_z)

        if debug:
            logger.debug("--- Final ---")
            logger.debug(
                f"X_tk={X_tks[:, 0].squeeze()}, dV_tk={dV_tks[:, 0].squeeze()}, r={residual_norms[:, 0]}"
            )

        return X_tks, dV_tks, residual_norms


@dataclass
class AndersonSafe(AndersonSolver):
    """
    Safeguarded Anderson acceleration with adaptive damping.

    Key improvements over basic Anderson:
    1. Safeguarded updates: Only accept step if it reduces total residual
    2. Adaptive damping: Reduce step size when residuals increase
    3. Early stopping: Stop when residual is small enough
    """

    beta: float = 0.5
    beta_min: float = 0.01
    rtol: float = 1e-3
    atol: float = 1e-3

    def _safeguarded_step(
        self,
        X_tks: Tensor,  # (K, bs, d)
        X_bases: Tensor,  # (K, bs, d)
        x_time_deps: Tensor,  # (K, K, bs, 1)
        potentials: list[Potential],
        xms: Tensor,  # (K, bs, d)
        X_hist: list[Tensor],  # list of (K, bs, d)
        r_hist: list[Tensor],  # list of (K, bs, d)
        debug: bool = False,
        dV_tks_prev: (
            Tensor | None
        ) = None,  # Precomputed dV_tks from previous iteration, if available
        alpha: float = 1.0,  # Scaling factor for corrections (used in homotopy)
    ) -> tuple[Tensor, Tensor, list[Tensor], list[Tensor], bool]:
        """
        Single safeguarded Anderson step with backtracking.

        Args:
            dV_tks_prev: If provided, use this instead of recomputing potential gradients.
                         Must correspond to the current X_tks.
            alpha: Scaling factor for corrections (1.0 for normal, <1.0 for homotopy).

        Returns:
            X_tks_new: Updated X_tks (K, bs, d)
            dV_tks: Potential gradients (K, bs, d)
            X_hist_new: Updated history for next iteration
            r_hist_new: Updated history for next iteration
            converged: Whether convergence criterion was met
        """
        K, bs, d = X_tks.shape

        G_z, dV_tks = self.G(X_tks, X_bases, x_time_deps, potentials, xms, dV_tks_prev, alpha)

        if debug:
            logger.debug(f"dV_tk={dV_tks[:, 0].squeeze()}")

        X_flat = self._flatten(X_tks)
        G_flat = self._flatten(G_z)
        r = G_flat - X_flat
        r_norm = r.norm(dim=-1).mean().item()

        # Early stopping check
        X_norm = X_flat.norm(dim=-1).mean().item()
        if r_norm < self.atol + self.rtol * X_norm:
            return X_tks, dV_tks, X_hist, r_hist, True

        # Compute Anderson candidate
        X_candidate, _, X_hist_new, r_hist_new = self._anderson_step(
            X_flat, G_flat, X_hist, r_hist, self.beta, clamp_gamma=True, debug=debug
        )

        # Safeguard: check if candidate improves
        X_cand_tks = self._unflatten(X_candidate, K, d)
        G_cand, dV_cand = self.G(X_cand_tks, X_bases, x_time_deps, potentials, xms, alpha=alpha)
        G_cand_flat = self._flatten(G_cand)
        r_cand_norm = (G_cand_flat - X_candidate).norm(dim=-1).mean().item()

        if r_cand_norm <= r_norm:
            if debug:
                logger.debug(f"  Accepted: r_cand_norm={r_cand_norm:.6e}")
            return X_cand_tks, dV_cand, X_hist_new, r_hist_new, False

        # Backtrack with smaller step (use simple Picard step, update history)
        beta = self.beta
        for bt in range(5):
            beta = beta * 0.5
            if beta < self.beta_min:
                break
            X_bt = X_flat + beta * r
            X_bt_tks = self._unflatten(X_bt, K, d)
            G_bt, dV_bt = self.G(X_bt_tks, X_bases, x_time_deps, potentials, xms, alpha=alpha)
            G_bt_flat = self._flatten(G_bt)
            r_bt_norm = (G_bt_flat - X_bt).norm(dim=-1).mean().item()

            if r_bt_norm <= r_norm:
                if debug:
                    logger.debug(f"  Backtrack {bt}: beta={beta:.4f}, r_bt_norm={r_bt_norm:.6e}")
                # Update history with the backtracked step
                X_hist_bt = (X_hist + [X_flat])[-self.rank :]
                r_hist_bt = (r_hist + [r])[-self.rank :]
                return X_bt_tks, dV_bt, X_hist_bt, r_hist_bt, False

        if debug:
            logger.debug("  No improvement, keeping current")
        # Still update history even if no improvement
        X_hist_new = (X_hist + [X_flat])[-self.rank :]
        r_hist_new = (r_hist + [r])[-self.rank :]
        return X_tks, dV_tks, X_hist_new, r_hist_new, False

    def __call__(
        self,
        X_inits: Tensor,
        X_bases: Tensor,
        x_time_deps: Tensor,
        potentials: list[Potential],
        xms: Tensor,
        debug: bool = False,
        **kwargs,
    ) -> tuple[Tensor, Tensor, Tensor]:
        X_tks = X_inits
        K, bs, d = X_tks.shape

        X_hist: list[Tensor] = []
        r_hist: list[Tensor] = []
        dV_tks = None

        if debug:
            logger.debug(f"SafeAnderson(m={self.rank}) starting with X_tk={X_tks[:, 0].squeeze()}")

        for k in range(self.picard_steps):
            X_tks, dV_tks, X_hist, r_hist, converged = self._safeguarded_step(
                X_tks,
                X_bases,
                x_time_deps,
                potentials,
                xms,
                X_hist,
                r_hist,
                debug,
                dV_tks_prev=dV_tks,
            )

            if debug:
                r_norm = self._unflatten(r_hist[-1], K, d).norm(dim=-1)
                logger.debug(
                    f"SafeAnderson(m={self.rank}) step {k}: prev r={r_norm[:, 0]}, new X_tk={X_tks[:, 0].squeeze()}"
                )
                # print("printed")

            if converged:
                if debug:
                    logger.debug(f"  Converged at step {k}")
                break

        # Final evaluation
        G_z, dV_tks = self.G(X_tks, X_bases, x_time_deps, potentials, xms)
        _, residual_norms = self.compute_residuals(X_tks, G_z)

        if debug:
            logger.debug("--- Final ---")
            logger.debug(
                f"X_tk={X_tks[:, 0].squeeze()}, dV_tk={dV_tks[:, 0].squeeze()}, r={residual_norms[:, 0]}"
            )

        return X_tks, dV_tks, residual_norms


@dataclass
class AndersonHomotopy(AndersonSafe):
    """
    Homotopy continuation with Anderson acceleration for high-strength potentials.

    Gradually increases the effective strength of all potentials:
        X(t_i) = X_base(t_i) + alpha * Σ_j x_time_dep_j(t_i) * dV_j

    Starting with alpha << 1 and increasing to alpha = 1.
    """

    n_stages: int = 5
    inner_steps: int = 3
    alpha_start: float = 0.1
    alpha_schedule: str = "adaptive"  # "linear" or "adaptive"

    def _inner_solve(
        self,
        X_tks: Tensor,
        X_bases: Tensor,
        x_time_deps: Tensor,
        potentials: list[Potential],
        xms: Tensor,
        alpha: float,
        max_steps: int,
        debug: bool = False,
    ) -> tuple[Tensor, Tensor, float, bool]:
        """Run inner safeguarded Anderson iterations at fixed alpha."""
        K, bs, d = X_tks.shape

        X_hist: list[Tensor] = []
        r_hist: list[Tensor] = []
        dV_tks = None

        for k in range(max_steps):
            X_tks, dV_tks, X_hist, r_hist, converged = self._safeguarded_step(
                X_tks,
                X_bases,
                x_time_deps,
                potentials,
                xms,
                X_hist,
                r_hist,
                debug=debug,
                dV_tks_prev=dV_tks,
                alpha=alpha,
            )

            if debug:
                logger.debug(
                    f"Inner step {k}: r_norm={r_hist[-1][:, 0]}, dV_tk={dV_tks[:, 0].squeeze()}, new X_tk={X_tks[:, 0].squeeze()}, converged={converged}"
                )

            if converged:
                G_z, _ = self.G(X_tks, X_bases, x_time_deps, potentials, xms, alpha=alpha)
                r_norm = (G_z - X_tks).norm(dim=-1)
                return X_tks, dV_tks, r_norm, True

        # Final residual
        G_z, dV_tks = self.G(X_tks, X_bases, x_time_deps, potentials, xms, alpha=alpha)
        r_norm = (G_z - X_tks).norm(dim=-1)

        return X_tks, dV_tks, r_norm, False

    def __call__(
        self,
        X_inits: Tensor,
        X_bases: Tensor,
        x_time_deps: Tensor,
        potentials: list[Potential],
        xms: Tensor,
        debug: bool = False,
        **kwargs,
    ) -> tuple[Tensor, Tensor, Tensor]:
        X_tks = X_inits
        K, bs, d = X_tks.shape

        if debug:
            logger.debug(f"Homotopy starting with X_tk={X_tks[:, 0].squeeze()}")

        if self.alpha_schedule == "adaptive":
            alpha = self.alpha_start
            total_steps = 0
            max_total_steps = self.picard_steps

            while alpha < 1.0 and total_steps < max_total_steps:
                steps_this_stage = min(self.inner_steps, max_total_steps - total_steps)
                X_tks, dV_tks, r_norm, converged = self._inner_solve(
                    X_tks,
                    X_bases,
                    x_time_deps,
                    potentials,
                    xms,
                    alpha,
                    steps_this_stage,
                    debug=debug,
                )
                total_steps += steps_this_stage

                if debug:
                    logger.debug(
                        f"Homotopy alpha={alpha:.4f}: r_norm={r_norm[:, 0]}, dV_tk={dV_tks[:, 0].squeeze()}, new X_tk={X_tks[:, 0].squeeze()}, converged={converged}"
                    )

                # norm along K potentials then average over batch
                r_norm_k = r_norm.norm(dim=0).mean().item()
                # Adaptive alpha increase
                if converged or r_norm_k < 0.1:
                    alpha = min(1.0, alpha + (1.0 - alpha) * 0.5)
                elif r_norm_k < 1.0:
                    alpha = min(1.0, alpha + (1.0 - alpha) * 0.3)
                else:
                    alpha = min(1.0, alpha + (1.0 - alpha) * 0.15)

            # Final solve at alpha=1
            if alpha >= 1.0 - 1e-6:
                remaining_steps = max(1, max_total_steps - total_steps)
                X_tks, dV_tks, r_norm, _ = self._inner_solve(
                    X_tks, X_bases, x_time_deps, potentials, xms, 1.0, remaining_steps, debug
                )
                if debug:
                    logger.debug(
                        f"Homotopy alpha=1.0 (final): r_norm={r_norm[:, 0]}, dV_tk={dV_tks[:, 0].squeeze()}, new X_tk={X_tks[:, 0].squeeze()}"
                    )
        else:
            # Fixed linear schedule
            alphas = [
                self.alpha_start + (1.0 - self.alpha_start) * i / (self.n_stages - 1)
                for i in range(self.n_stages)
            ]
            steps_per_stage = max(1, self.picard_steps // self.n_stages)

            for stage, alpha in enumerate(alphas):
                X_tks, dV_tks, r_norm, converged = self._inner_solve(
                    X_tks, X_bases, x_time_deps, potentials, xms, alpha, steps_per_stage, debug
                )
                if debug:
                    logger.debug(f"Stage {stage} (alpha={alpha:.4f}): r_norm={r_norm[:, 0]}")

        # Final evaluation at full strength
        G_z, dV_tks = self.G(X_tks, X_bases, x_time_deps, potentials, xms)
        _, residual_norms = self.compute_residuals(X_tks, G_z)

        if debug:
            logger.debug("--- Final (alpha=1.0) ---")
            logger.debug(
                f"X_tk={X_tks[:, 0].squeeze()}, dV_tk={dV_tks[:, 0].squeeze()}, r={residual_norms[:, 0]}"
            )

        return X_tks, dV_tks, residual_norms


def get_solver(solver_type: str, picard_steps: int, **kwargs) -> FixedPointSolver:
    """Factory function to create a solver from string type."""
    match solver_type:
        case "direct":
            return DirectSolver(picard_steps=1)  # picard_steps not used
        case "picard":
            return PicardSolver(picard_steps=picard_steps)
        case "anderson":
            return AndersonSolver(picard_steps=picard_steps, **kwargs)
        case "anderson_safe":
            return AndersonSafe(picard_steps=picard_steps, **kwargs)
        case "anderson_homotopy":
            return AndersonHomotopy(picard_steps=picard_steps, **kwargs)
        case _:
            raise ValueError(
                f"Unknown solver: {solver_type}. Use 'direct', 'picard', 'anderson', 'anderson_safe', or 'anderson_homotopy'."
            )


__all__ = [
    "FixedPointSolver",
    "DirectSolver",
    "PicardSolver",
    "AndersonSolver",
    "AndersonSafe",
    "AndersonHomotopy",
    "get_solver",
]
