# OLLA_NIPS/src/samplers/clangevin.py

import math
import torch
import numpy as np
from typing import Callable, Dict, List, Optional
from torch.func import vmap, grad
from .base_sampler import BaseSampler
from tqdm.auto import trange



class CLangevin(BaseSampler):
    """
    Constrained Langevin sampler using a slack-squared variable transformation
    for inequality constraints, combined with an iterative SHAKE-style projection.
    """
    def __init__(
        self,
        constraint_funcs: Dict[str, List[Callable]],
        step_size: float,
        num_steps: int,
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
        self.proj_iters = proj_iters
        self.proj_tol = proj_tol
        self.damp = proj_damp
        self.dtype = dtype
        self.device = device or torch.device('cpu')

        self.MAX_RETRIES = 0                  
        self.REJECT_H_THRESH = max(self.proj_tol, 1e-8)  

        self._h_vmapped = [vmap(fn) for fn in self.h_fns]
        self._g_vmapped = [vmap(fn) for fn in self.g_fns]

    def sample(
        self,
        x0: np.ndarray,
        potential_fn: Callable[[torch.Tensor], torch.Tensor],
        grad_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> Dict[str, np.ndarray]:
        # --- 1. Initial Setup ---
        x0_arr = np.atleast_2d(x0.astype(np.float64))
        P, orig_d = x0_arr.shape
        K, orig_m, orig_l = self.num_steps, len(self.h_fns), len(self.g_fns)

        if self.seed is not None:
            torch.manual_seed(self.seed)
        x_init = torch.tensor(x0_arr, dtype=self.dtype, device=self.device, requires_grad=True)

        # --- 2. Setup Slack Variables ---
        x = x_init
        if orig_l > 0:
            g0 = torch.stack([fn(x_init) for fn in self._g_vmapped], dim=1)
            s0 = torch.sqrt(torch.clamp(-g0, min=0.0))
            x = torch.cat([x_init, s0], dim=1)

        # --- 3. History Tracking ---
        traj = torch.zeros((K + 1, P, orig_d), dtype=self.dtype, device=self.device)
        traj[0] = x[:, :orig_d].detach()
        h_hist = torch.zeros((K, P, orig_m), dtype=self.dtype, device=self.device)
        g_hist = torch.zeros((K, P, orig_l), dtype=self.dtype, device=self.device)

        # --- 4. Effective Constraints (post-slack) ---
        eff_h_fns = [lambda x_ext, fn=fn: fn(x_ext[:orig_d]) for fn in self.h_fns]
        eff_h_fns.extend([
            lambda x_ext, g_fn=g_fn, j=j: g_fn(x_ext[:orig_d]) + x_ext[orig_d + j]**2
            for j, g_fn in enumerate(self.g_fns)
        ])
        d_ext, m_eff = x.shape[1], len(eff_h_fns)
        eff_h_vmapped = [vmap(fn) for fn in eff_h_fns]
        eff_h_grad_vmapped = [vmap(grad(fn)) for fn in eff_h_fns]   
        
        # --- 5. Extended Potential and Gradient ---
        potential_fn_ext = lambda x_s: potential_fn(x_s[:orig_d])
        grad_potential_vmap = vmap(grad(potential_fn_ext))

        # --- 6. Time-Stepping Loop ---
        noise_scale = math.sqrt(2.0 * self.step_size)
        n_proj_fails = 0
        with torch.no_grad():
            with trange(K, desc="CLangevin Sampling", leave=False, dynamic_ncols=True) as pbar:
                for k in pbar:
                    # Record histories and update progress bar
                    h_vals = torch.stack([fn(x[:, :orig_d]) for fn in self._h_vmapped], dim=1) if orig_m > 0 else torch.zeros((P,0), device=x.device)
                    g_vals = torch.stack([fn(x[:, :orig_d]) for fn in self._g_vmapped], dim=1) if orig_l > 0 else torch.zeros((P,0), device=x.device)
                    h_hist[k], g_hist[k] = h_vals, g_vals

                    # --- Update Progress Bar with Constraint Violations ---
                    h_viol = h_vals.abs().mean().item() if orig_m > 0 else 0.0
                    g_viol = torch.relu(g_vals).max().item() if orig_l > 0 else 0.0
                    pbar.set_postfix({'h_viol (mean)': f'{h_viol:.4f}', 'g_viol (max)': f'{g_viol:.4f}', 'n_proj_fails': n_proj_fails})

                    grad_f_ext = grad_potential_vmap(x)
                    # ---------- proposal + projection with rejection / resampling ----------
                    x_prev = x 
                    # propose
                    noise = torch.randn((P, d_ext), dtype=self.dtype, device=self.device)
                    y_prop = x - self.step_size * grad_f_ext + noise_scale * noise

                    # project via iterative SHAKE
                    y_proj = y_prop
                    if m_eff > 0:
                        G_proj_init = torch.stack([fn(y_proj) for fn in eff_h_grad_vmapped], dim=1)
                        for _ in range(self.proj_iters):
                            h_vals_eff = torch.stack([h(y_proj) for h in eff_h_vmapped], dim=1)
                            if h_vals_eff.abs().max() < self.proj_tol:
                                break
                            G_proj = torch.stack([fn(y_proj) for fn in eff_h_grad_vmapped], dim=1)
                            J_gram = torch.einsum('pmd,pnd->pmn', G_proj, G_proj_init)
                            J_gram += self.damp * torch.eye(m_eff, device=x.device, dtype=x.dtype)
                            try:
                                lam = -torch.linalg.solve(J_gram, h_vals_eff)
                            except RuntimeError:
                                lam = -torch.linalg.lstsq(J_gram, h_vals_eff).solution
                            y_proj = y_proj + torch.einsum('pmd,pm->pd', G_proj_init, lam)

                        # final constraint violation check on equality manifold
                        h_final = torch.stack([h(y_proj) for h in eff_h_vmapped], dim=1)
                        max_h = h_final.abs().max().item()
                    else:
                        # no equality constraints to project
                        h_final = torch.zeros((P,0), device=x.device, dtype=self.dtype)
                        max_h = 0.0

                    if max_h <= self.proj_tol:
                        # accept this projected proposal
                        x = y_proj.detach().requires_grad_()
                    else:
                        n_proj_fails += 1
                        x = x_prev.detach().requires_grad_()
                    traj[k+1] = x[:, :orig_d].detach()

        return {
            'trajectory': traj.detach().cpu().numpy(),
            'h_vals': h_hist.detach().cpu().numpy(),
            'g_vals': g_hist.detach().cpu().numpy(),
        }

