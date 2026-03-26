# OLLA_NIPS/src/samplers/chmc.py

import math
import torch
import numpy as np
from typing import Callable, Dict, List, Optional
from torch.func import vmap, grad
from .base_sampler import BaseSampler
from tqdm.auto import trange

class CHMC(BaseSampler):
    """
    Constrained Hamiltonian Monte Carlo sampler using a RATTLE integrator.
    Handles inequality constraints via a slack-squared variable transformation.
    """
    def __init__(
        self,
        constraint_funcs: Dict[str, List[Callable]],
        step_size: float,
        num_steps: int,
        gamma: float = 1.0,
        proj_iters: int = 10,
        proj_tol: float = 1e-4,
        proj_damp: float = 1e-6,
        dtype: torch.dtype = torch.float64,
        device: Optional[torch.device] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(step_size=step_size, num_steps=num_steps, seed=seed)
        self.h_fns = constraint_funcs.get('h', [])
        self.g_fns = constraint_funcs.get('g', [])
        self.gamma = gamma
        self.proj_iters = proj_iters
        self.proj_tol = proj_tol
        self.proj_damp = proj_damp
        self.dtype = dtype
        self.device = device or torch.device('cpu')

        self._h_vmapped = [vmap(fn) for fn in self.h_fns]
        self._g_vmapped = [vmap(fn) for fn in self.g_fns]

    def _solve_multipliers(self, G_mat: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        """Solves the linear system (G @ G.T) * lambda = rhs to find multipliers."""
        A = torch.einsum('bmd,bnd->bmn', G_mat, G_mat)
        A += self.proj_damp * torch.eye(G_mat.shape[1], device=G_mat.device, dtype=G_mat.dtype)
        try:
            return torch.linalg.solve(A, rhs)
        except RuntimeError:
            return torch.linalg.lstsq(A, rhs).solution

    def sample(
        self,
        x0: np.ndarray,
        potential_fn: Callable[[torch.Tensor], torch.Tensor],
        grad_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> Dict[str, np.ndarray]:
        # --- 1. Initial Setup ---
        q_arr = np.atleast_2d(x0.astype(float))
        P, orig_d = q_arr.shape
        K, orig_m, orig_l = self.num_steps, len(self.h_fns), len(self.g_fns)

        if self.seed is not None:
            torch.manual_seed(self.seed)
        q_init = torch.tensor(q_arr, dtype=self.dtype, device=self.device, requires_grad=True)
        p_init = torch.randn((P, orig_d), dtype=self.dtype, device=self.device)

        # --- 2. Setup Slack Variables ---
        q, p = q_init, p_init
        if orig_l > 0:
            g0 = torch.stack([fn(q_init) for fn in self._g_vmapped], dim=1)
            s0 = torch.sqrt(torch.clamp(-g0, min=0.0))
            q = torch.cat([q_init, s0], dim=1)
            p = torch.cat([p_init, torch.zeros_like(s0)], dim=1)

        # --- 3. History Tracking & Constraint Setup ---
        traj_q = torch.zeros((K + 1, P, orig_d), dtype=self.dtype, device=self.device)
        traj_q[0] = q[:, :orig_d]
        h_hist = torch.zeros((K, P, orig_m), dtype=self.dtype, device=self.device)
        g_hist = torch.zeros((K, P, orig_l), dtype=self.dtype, device=self.device)

        eff_h_fns = [lambda x_ext, fn=fn: fn(x_ext[:orig_d]) for fn in self.h_fns]
        eff_h_fns.extend([
            lambda x_ext, g_fn=g_fn, j=j: g_fn(x_ext[:orig_d]) + x_ext[orig_d + j]**2
            for j, g_fn in enumerate(self.g_fns)
        ])
        d_ext, m_eff = q.shape[1], len(eff_h_fns)
        eff_h_vmapped = [vmap(fn) for fn in eff_h_fns]

        # --- 4. Extended Potential and Gradient Setup ---
        potential_fn_ext = lambda x_s: potential_fn(x_s[:orig_d])
        grad_potential_vmap = vmap(grad(potential_fn_ext))
        n_proj_fails = 0

        alpha = 0.25 * self.step_size * self.gamma

        # --- Time-Stepping Loop ---
        with torch.no_grad():
            with trange(K, desc="CHMC Sampling", leave=False, dynamic_ncols=True) as pbar:
                for k in pbar:
                    h_vals = torch.stack([fn(q[:,:orig_d]) for fn in self._h_vmapped],dim=1) if orig_m > 0 else torch.zeros((P,0), device=q.device)
                    g_vals = torch.stack([fn(q[:,:orig_d]) for fn in self._g_vmapped],dim=1) if orig_l > 0 else torch.zeros((P,0), device=q.device)
                    h_hist[k], g_hist[k] = h_vals, g_vals

                    h_viol = h_vals.abs().mean().item() if orig_m > 0 else 0.0
                    g_viol = torch.relu(g_vals).max().item() if orig_l > 0 else 0.0

                    # progress bar (include cumulative projection failures)
                    pbar.set_postfix({
                        'h_viol (mean)': f'{h_viol:.4f}',
                        'g_viol (max)': f'{g_viol:.4f}',
                        'n_proj_fails': n_proj_fails
                    })
                    q_prev, p_prev = q, p
                    # --- Step 1: Update p -> p_{n+1/4} (OU) ---
                    noise_1 = torch.randn_like(p)
                    p_1_4 = (1 - alpha) * p + math.sqrt(2 * alpha) * noise_1

                    if m_eff > 0:
                        Gq_n = torch.stack([vmap(grad(fn))(q) for fn in eff_h_fns], dim=1)  # (P,m_eff,d_ext)
                        res_1 = torch.einsum('pmd,pd->pm', Gq_n, p_1_4)
                        lam_1 = self._solve_multipliers(Gq_n, res_1)
                        p_1_4 = p_1_4 - torch.einsum('pmd,pm->pd', Gq_n, lam_1)

                    p_1_4 = p_1_4 / (1 + alpha)

                    # --- Step 2: p_{n+1/4} -> p_{n+1/2} (potential kick) ---
                    grad_f_1 = grad_potential_vmap(q)
                    p_1_2 = p_1_4 - 0.5 * self.step_size * grad_f_1

                    # --- Step 3: q -> q_{n+1} (position drift + projection) ---
                    q_n_1 = q + self.step_size * p_1_2
                    max_h = 0.0
                    if m_eff > 0:
                        # iterative SHAKE-like projection
                        for _ in range(self.proj_iters):
                            h_vals_prop = torch.stack([h(q_n_1) for h in eff_h_vmapped], dim=1)
                            if h_vals_prop.abs().max() < self.proj_tol:
                                break
                            Gq_n_1 = torch.stack([vmap(grad(fn))(q_n_1) for fn in eff_h_fns], dim=1)
                            # note: using Gq_n from current q as "frozen" columns
                            J_gram = self.step_size * (torch.einsum('pij,pkj->pik', Gq_n_1, Gq_n) + self.proj_damp * torch.eye(m_eff, device=q.device, dtype=q.dtype))
                            try:
                                delta_lam = -torch.linalg.solve(J_gram, h_vals_prop)
                            except RuntimeError:
                                delta_lam = -torch.linalg.lstsq(J_gram, h_vals_prop).solution
                            q_n_1 = q_n_1 + self.step_size * torch.einsum('pmd,pm->pd', Gq_n, delta_lam)

                        # final violation after projection
                        h_final = torch.stack([h(q_n_1) for h in eff_h_vmapped], dim=1)
                        max_h = h_final.abs().max().item()

                    # --- check projection quality; continue if bad ---
                    if max_h > self.proj_tol:
                        # reject: stay at previous state & count failure
                        n_proj_fails += 1
                        q, p = q_prev.detach().requires_grad_(), -p_prev.detach().requires_grad_()
                        traj_q[k+1] = q[:, :orig_d].detach()
                        continue

                    # --- Step 4: p_{n+1/2} -> p_{n+3/4} (potential kick at q_{n+1}) ---
                    grad_f_2 = grad_potential_vmap(q_n_1)
                    p_3_4 = p_1_2 - 0.5 * self.step_size * grad_f_2

                    # --- Step 5: p_{n+3/4} -> p_{n+1} (OU) ---
                    noise_2 = torch.randn_like(p)
                    p_n_1 = (1 - alpha) * p_3_4 + math.sqrt(2 * alpha) * noise_2

                    if m_eff > 0:
                        Gq_n_1_final = torch.stack([vmap(grad(fn))(q_n_1) for fn in eff_h_fns], dim=1)
                        res_2 = torch.einsum('pmd,pd->pm', Gq_n_1_final, p_n_1)
                        lam_2 = self._solve_multipliers(Gq_n_1_final, res_2)
                        p_n_1 = p_n_1 - torch.einsum('pmd,pm->pd', Gq_n_1_final, lam_2)

                    p_n_1 = p_n_1 / (1 + alpha)

                    # accept this proposal
                    q, p = q_n_1.detach().requires_grad_(), p_n_1

                    # record (whether accepted or rejected)
                    traj_q[k+1] = q[:, :orig_d].detach()

        return {
            'trajectory': traj_q.detach().cpu().numpy(),
            'h_vals': h_hist.detach().cpu().numpy(),
            'g_vals': g_hist.detach().cpu().numpy(),
        }

