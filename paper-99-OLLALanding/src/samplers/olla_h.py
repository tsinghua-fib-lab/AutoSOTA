# OLLA_NIPS/src/samplers/olla_h.py

import math
import torch
import numpy as np
from typing import Callable, Dict, List, Optional
from torch.func import vmap, grad, jvp
from .base_sampler import BaseSampler
from ..utils import project
from tqdm.auto import trange

def _hutchinson_trace_per_constraint(
    grad_fn: Callable,
    x: torch.Tensor,
    J: torch.Tensor,
    G_inv: torch.Tensor,
    num_samples: int
) -> torch.Tensor:
    """
    Estimates trace(P @ Hess h_i) where P = I - J G^{-1} J^T
    using Rademacher probes.
    """
    d = x.shape[0]
    # Use Rademacher probes { -1, 1 } for lower variance
    u = (torch.randint(0, 2, (num_samples, d), device=x.device) * 2 - 1).to(x.dtype)
    hvp = vmap(lambda ui: jvp(grad_fn, (x,), (ui,))[1])(u)  # (S, d)
    
    # Project all probes in one shot
    P_u = project(u, J.unsqueeze(0), G_inv.unsqueeze(0))
    
    return (hvp * P_u).sum(-1).mean()

class OLLA_H(BaseSampler):
    """
    OLLA sampler using the Hutchinson trace estimator. This version is highly
    optimized for performance by:
    1. Avoiding explicit formation of the projection matrix.
    2. Only computing the trace estimate for active inequality constraints.
    """
    def __init__(
        self,
        constraint_funcs: Dict[str, List[Callable]],
        alpha: float,
        step_size: float,
        num_steps: int,
        num_hutchinson_samples: int = 5,
        epsilon: float = 1e-4,
        proj_damp: float = 1e-6,
        dtype: torch.dtype = torch.float64,
        device: Optional[torch.device] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(
            step_size=step_size,
            num_steps=num_steps,
            seed=seed,
            alpha=alpha,
            epsilon=epsilon,
        )
        self.h_fns = constraint_funcs.get('h', [])
        self.g_fns = constraint_funcs.get('g', [])
        self.num_hutchinson_samples = num_hutchinson_samples
        self.alpha   = alpha
        self.epsilon = epsilon
        self.proj_damp = proj_damp
        self.dtype   = dtype
        self.device  = device or torch.device('cpu')

        self._h_vmapped = [vmap(fn) for fn in self.h_fns]
        self._g_vmapped = [vmap(fn) for fn in self.g_fns]
        self._h_grad_vmapped = [vmap(grad(fn)) for fn in self.h_fns]
        self._g_grad_vmapped = [vmap(grad(fn)) for fn in self.g_fns]

    def sample(
        self,
        x0: np.ndarray,
        potential_fn: Callable[[torch.Tensor], torch.Tensor],
        grad_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> Dict[str, np.ndarray]:
        
        x0_arr = np.atleast_2d(x0.astype(float))
        P, d   = x0_arr.shape
        K      = self.num_steps

        x = torch.tensor(x0_arr, dtype=self.dtype, device=self.device, requires_grad=True)
        noise_scale = math.sqrt(2.0 * self.step_size)

        traj = torch.zeros((K + 1, P, d), dtype=self.dtype, device=self.device)
        traj[0] = x
        m, l = len(self.h_fns), len(self.g_fns)
        h_hist = torch.zeros((K, P, m), dtype=self.dtype, device=self.device)
        g_hist = torch.zeros((K, P, l), dtype=self.dtype, device=self.device)

        grad_potential_vmap = vmap(grad_fn if grad_fn is not None else grad(potential_fn))
        with torch.no_grad():
            with trange(K, desc="OLLA-H Sampling", leave=False, dynamic_ncols=True) as pbar:
                for k in pbar:
                    h_vals = torch.stack([h_fn(x) for h_fn in self._h_vmapped], dim=1) if m > 0 else torch.zeros((P,0), device=self.device)
                    g_vals = torch.stack([g_fn(x) for g_fn in self._g_vmapped], dim=1) if l > 0 else torch.zeros((P,0), device=self.device)
                    h_hist[k], g_hist[k] = h_vals, g_vals
                    
                    # --- Update Progress Bar with Constraint Violations ---
                    h_viol = h_vals.abs().mean().item() if m > 0 else 0.0
                    g_viol = torch.relu(g_vals).max().item() if l > 0 else 0.0
                    pbar.set_postfix({'h_viol (mean)': f'{h_viol:.4f}', 'g_viol (max)': f'{g_viol:.4f}'})

                    grad_f = grad_potential_vmap(x)
                    
                    grad_h = torch.stack([fn(x) for fn in self._h_grad_vmapped], dim=1) if m > 0 else torch.zeros((P,0,d), device=self.device)

                    mask = (g_vals >= 0)
                    grad_g = torch.zeros((P,l,d), device=self.device, dtype=self.dtype)
                    if l > 0:
                        for j, fn in enumerate(self._g_grad_vmapped):
                            idx = mask[:,j].nonzero(as_tuple=False).squeeze(1)
                            if idx.numel(): grad_g[idx, j] = fn(x[idx])
                    
                    cols = [grad_h[:,i].unsqueeze(2) for i in range(m)]
                    cols += [(grad_g[:,j] * mask[:,j:j+1]).unsqueeze(2) for j in range(l)]
                    
                    J = torch.cat(cols, dim=2) if cols else torch.zeros((P,d,0), device=self.device)
                    p = J.shape[2]
                    
                    G_inv = None
                    if p > 0:
                        G = torch.einsum('bdi,bdj->bij', J, J) + self.proj_damp * torch.eye(p, device=J.device, dtype=J.dtype).unsqueeze(0)
                        try: G_inv = torch.linalg.inv(G)
                        except RuntimeError: G_inv = torch.linalg.pinv(G)

                    H_corr = torch.zeros((P,d), device=self.device, dtype=self.dtype)
                    if p > 0 and self.num_hutchinson_samples > 0:
                        trace_terms = []
                        
                        # Equalities (always active)
                        for fn in self.h_fns:
                            ti_estimate = vmap(_hutchinson_trace_per_constraint, in_dims=(None, 0, 0, 0, None), randomness='different')(grad(fn), x, J, G_inv, self.num_hutchinson_samples)
                            trace_terms.append(ti_estimate)

                        # Inequalities (computed only for active particles)
                        for j, fn in enumerate(self.g_fns):
                            idx = mask[:, j].nonzero(as_tuple=False).squeeze(1)
                            full_trace = torch.zeros(P, dtype=self.dtype, device=self.device)
                            if idx.numel() > 0:
                                ti_estimate = vmap(_hutchinson_trace_per_constraint, in_dims=(None, 0, 0, 0, None), randomness='different')(grad(fn), x[idx], J[idx], G_inv[idx], self.num_hutchinson_samples)
                                full_trace[idx] = ti_estimate
                            trace_terms.append(full_trace)

                        trace_vec = torch.stack(trace_terms, dim=1) if trace_terms else torch.zeros((P,0), device=self.device)
                        
                        tmp = torch.einsum('bij,bj->bi', G_inv, trace_vec)
                        H_corr = -torch.einsum('bdi,bi->bd', J, tmp)
                    else:
                        H_corr = torch.zeros((P,d), device=self.device, dtype=self.dtype)

                    drift = -grad_f
                    if p > 0 and G_inv is not None:
                        drift = project(-grad_f, J, G_inv)
                        parts = [h_vals] if m > 0 else []
                        if l > 0: parts.append((g_vals + self.epsilon) * mask.to(self.dtype))
                        
                        J_vals = torch.cat(parts, dim=1)
                        tmp2   = torch.einsum('bij,bj->bi', G_inv, J_vals)
                        drift  = drift - self.alpha * torch.einsum('bdi,bi->bd', J, tmp2) + H_corr

                    xi = torch.randn_like(x)
                    projected_noise = project(xi, J, G_inv) if p > 0 and G_inv is not None else xi

                    x  = x + self.step_size * drift + noise_scale * projected_noise
                    x = x.detach().requires_grad_()
                    traj[k+1] = x

        return {
            'trajectory': traj.detach().cpu().numpy(),
            'h_vals':     h_hist.detach().cpu().numpy(),
            'g_vals':     g_hist.detach().cpu().numpy()
        }

