# OLLA_NIPS/src/samplers/cghmc.py

import math
import torch
import numpy as np
from typing import Callable, Dict, List, Optional
from torch.func import vmap, grad
from .base_sampler import BaseSampler
from tqdm.auto import trange

class CGHMC(BaseSampler):
    """
    Constrained generalized Hamiltonian Monte Carlo (CGHMC) sampler.

    This sampler uses a RATTLE integrator for Hamiltonian dynamics on a
    manifold defined by equality constraints. It includes Ornstein-Uhlenbeck
    steps for thermalization and a Metropolis-Hastings acceptance step.
    Inequality constraints are handled via rejection in the MH filter.
    """
    def __init__(
        self,
        constraint_funcs: Dict[str, List[Callable]],
        step_size: float,
        num_steps: int,
        gamma: float = 1.0,
        proj_iters: int = 10,
        proj_tol: float = 1e-6,
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
        A = torch.einsum('pmd,pnd->pmn', G_mat, G_mat)
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
        P, d = q_arr.shape
        K, m, l = self.num_steps, len(self.h_fns), len(self.g_fns)

        if self.seed is not None:
            torch.manual_seed(self.seed)
        q = torch.tensor(q_arr, dtype=self.dtype, device=self.device, requires_grad=True)
        p = torch.randn((P, d), dtype=self.dtype, device=self.device)

        # --- 2. History Tracking ---
        traj_q = torch.zeros((K + 1, P, d), dtype=self.dtype, device=self.device)
        traj_q[0] = q
        h_hist = torch.zeros((K, P, m), dtype=self.dtype, device=self.device)
        g_hist = torch.zeros((K, P, l), dtype=self.dtype, device=self.device)
        
        # Buffer to store acceptance history for the last 10 steps.
        recent_accepts = torch.zeros((20, P), dtype=self.dtype, device=self.device)

        # --- 3. Setup Functions ---
        grad_potential_vmap = vmap(grad_fn if grad_fn is not None else grad(potential_fn))
        n_proj_fails = 0

        # --- Time-Stepping Loop ---
        with torch.no_grad():
            with trange(K, desc="CGHMC Sampling", leave=False, dynamic_ncols=True) as pbar:
                for k in pbar:
                    h_vals = torch.stack([fn(q) for fn in self._h_vmapped],dim=1) if m > 0 else torch.zeros((P,0), device=q.device)
                    g_vals = torch.stack([fn(q) for fn in self._g_vmapped],dim=1) if l > 0 else torch.zeros((P,0), device=q.device)
                    h_hist[k], g_hist[k] = h_vals, g_vals
                    q_prev, p_prev = q, p
                    # --- Step A: First OU half-step for momentum ---
                    alpha = 0.25 * self.step_size * self.gamma
                    noise_1 = torch.randn_like(p)
                    p_half_ou = (1 - alpha) * p + math.sqrt(2 * alpha) * noise_1
                    p_1_4 = p_half_ou
                    Gq_n = None
                    if m > 0:
                        Gq_n = torch.stack([vmap(grad(fn))(q) for fn in self.h_fns], dim=1)
                        res_1 = torch.einsum('pmd,pd->pm', Gq_n, p_half_ou)
                        lam_1 = self._solve_multipliers(Gq_n, res_1)
                        p_1_4 = p_half_ou - torch.einsum('pmd,pm->pd', Gq_n, lam_1)
                    
                    p_1_4 = p_1_4 / (1 + alpha)
                    
                    # --- Store Old Hamiltonian for MH Step ---
                    U_old = vmap(potential_fn)(q)
                    K_old = 0.5 * (p_1_4 ** 2).sum(dim=1)
                    H_old = U_old + K_old

                    # --- Step B: RATTLE Proposal ---
                    grad_f_1 = grad_potential_vmap(q)
                    p_1_2 = p_1_4 - 0.5 * self.step_size * grad_f_1
                    q_prop = q + self.step_size * p_1_2

                    max_h = 0.0
                    if m > 0:
                        for _ in range(self.proj_iters):
                            h_vals_prop = torch.stack([h(q_prop) for h in self._h_vmapped], dim=1)
                            if h_vals_prop.abs().max() < self.proj_tol: break
                            Gq_prop = torch.stack([vmap(grad(fn))(q_prop) for fn in self.h_fns], dim=1)
                            J_gram = self.step_size * (torch.einsum('pij,pkj->pik', Gq_prop, Gq_n) + self.proj_damp * torch.eye(m, device=q.device, dtype=q.dtype))
                            try: delta_lam = -torch.linalg.solve(J_gram, h_vals_prop)
                            except RuntimeError: delta_lam = -torch.linalg.lstsq(J_gram, h_vals_prop).solution
                            q_prop += self.step_size * torch.einsum('pmd,pm->pd', Gq_n, delta_lam)
                        p_1_2 = (q_prop - q) / self.step_size
                        h_final = torch.stack([h(q_prop) for h in self._h_vmapped], dim=1)
                        max_h = h_final.abs().max().item()
                    
                    if max_h > self.proj_tol:
                        n_proj_fails += 1
                        q, p = q_prev.detach().requires_grad_(), -p_prev.detach().requires_grad_()
                        traj_q[k+1] = q[:, :].detach()
                        continue

                    grad_f_2 = grad_potential_vmap(q_prop)
                    p_3_4 = p_1_2 - 0.5 * self.step_size * grad_f_2
                    p_prop = p_3_4
                    if m > 0:
                        Gq_prop_final = torch.stack([vmap(grad(fn))(q_prop) for fn in self.h_fns], dim=1)
                        res_2 = torch.einsum('pmd,pd->pm', Gq_prop_final, p_3_4)
                        lam_2 = self._solve_multipliers(Gq_prop_final, res_2)
                        p_prop = p_3_4 - torch.einsum('pmd,pm->pd', Gq_prop_final, lam_2)

                    # --- Step C: Metropolis-Hastings Acceptance ---
                    U_new = vmap(potential_fn)(q_prop)
                    K_new = 0.5 * (p_prop ** 2).sum(dim=1)
                    H_new = U_new + K_new
                    
                    alpha_mh = torch.exp(H_old - H_new).clamp(max=1.0)
                    u = torch.rand_like(alpha_mh)
                    accept = u < alpha_mh
                    if l > 0:
                        g_vals_prop = torch.stack([fn(q_prop) for fn in self._g_vmapped], dim=1)
                        accept &= (g_vals_prop <= 0).all(dim=1)
                    
                    # Update acceptance history buffer.
                    recent_accepts[k % 20] = accept.to(self.dtype)
                    
                    # --- Update Progress Bar ---
                    # Calculate the acceptance rate over the last 10 steps.
                    recent_rate = recent_accepts.mean().item()
                    h_viol = h_vals.abs().mean().item() if m > 0 else 0.0
                    g_viol = torch.relu(g_vals).max().item() if l > 0 else 0.0
                    pbar.set_postfix({'h_viol (mean)': f'{h_viol:.4f}', 'g_viol (max)': f'{g_viol:.4f}', 
                                     'n_proj_fails': n_proj_fails, 'acc_rate': f'{recent_rate:.2f}'})

                    q = torch.where(accept.unsqueeze(1), q_prop, q)
                    p = torch.where(accept.unsqueeze(1), p_prop, -p_1_4) # Reflect momentum on rejection

                    # --- Step D: Final OU Half-Step ---
                    noise_2 = torch.randn_like(p)
                    p_final_ou = (1 - alpha) * p + math.sqrt(2 * alpha) * noise_2
                    p = p_final_ou
                    if m > 0:
                        Gq_final = torch.stack([vmap(grad(fn))(q) for fn in self.h_fns], dim=1)
                        res_final = torch.einsum('pmd,pd->pm', Gq_final, p_final_ou)
                        lam_final = self._solve_multipliers(Gq_final, res_final)
                        p = p_final_ou - torch.einsum('pmd,pm->pd', Gq_final, lam_final)
                        
                    p = p / (1 + alpha)
                    q = q.detach().requires_grad_()
                    traj_q[k+1] = q

        return {
            'trajectory': traj_q.detach().cpu().numpy(),
            'h_vals': h_hist.detach().cpu().numpy(),
            'g_vals': g_hist.detach().cpu().numpy(),
        }
