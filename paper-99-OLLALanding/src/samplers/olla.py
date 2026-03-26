# OLLA_NIPS/src/samplers/olla.py

import math
import torch
import numpy as np
from typing import Callable, Dict, List, Optional
from torch.func import vmap, grad, hessian
from .base_sampler import BaseSampler
from ..utils import project
from tqdm.auto import trange

class OLLA(BaseSampler):
    """
    A vectorized OLLA sampler with an analytical trace correction.
    NOTE: The analytical trace calculation can be slow in high dimensions 
    (Also, this code uses explicit projection matrix (dxd) which is not scalable as d increases).
    For better performance in various cases, OLLA_H is recommended. 
    """
    def __init__(
        self,
        constraint_funcs: Dict[str, List[Callable]],
        alpha: float,
        step_size: float,
        num_steps: int,
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

        traj   = torch.zeros((K+1, P, d), dtype=self.dtype, device=self.device)
        traj[0] = x
        m, l   = len(self.h_fns), len(self.g_fns)
        h_hist = torch.zeros((K, P, m), dtype=self.dtype, device=self.device)
        g_hist = torch.zeros((K, P, l), dtype=self.dtype, device=self.device)
        
        grad_potential_vmap = vmap(grad_fn if grad_fn is not None else grad(potential_fn))
        eye_d = torch.eye(d, dtype=self.dtype, device=self.device).unsqueeze(0)
        with torch.no_grad():
            with trange(K, desc="OLLA Sampling", leave=False, dynamic_ncols=True) as pbar:
                for k in pbar:
                    h_vals = torch.stack([h_fn(x) for h_fn in self._h_vmapped], dim=1) if m > 0 else torch.zeros((P,0), device=self.device)
                    g_vals = torch.stack([g_fn(x) for g_fn in self._g_vmapped], dim=1) if l > 0 else torch.zeros((P,0), device=self.device)
                    h_hist[k], g_hist[k] = h_vals, g_vals

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

                    P_mat, G_inv = eye_d.expand(P, -1, -1), None
                    if p > 0:
                        G = torch.einsum('bdi,bdj->bij', J, J) + self.proj_damp * torch.eye(p, device=J.device, dtype=J.dtype).unsqueeze(0)
                        try: G_inv = torch.linalg.inv(G)
                        except RuntimeError: G_inv = torch.linalg.pinv(G)
                        J_G_inv = torch.einsum('bdi,bij->bdj', J, G_inv)
                        P_mat = eye_d - torch.einsum('bid,bdj->bij', J_G_inv, J.transpose(1,2))

                    H_corr = torch.zeros((P,d), device=self.device, dtype=self.dtype)
                    if p > 0 and G_inv is not None:
                        H_h = torch.stack([vmap(hessian(fn))(x) for fn in self.h_fns], dim=1) if m > 0 else torch.zeros((P,0,d,d), device=self.device)
                        H_g = torch.zeros((P,l,d,d), device=self.device, dtype=self.dtype)
                        if l > 0:
                            for j, fn in enumerate(self.g_fns):
                                idx = mask[:,j].nonzero(as_tuple=False).squeeze(1)
                                if idx.numel(): H_g[idx, j] = vmap(hessian(fn))(x[idx])
                        
                        trace_terms = []
                        for i in range(m):
                            ti = (P_mat * H_h[:, i]).sum(dim=(-1, -2))
                            trace_terms.append(ti)
                        for j in range(l):
                            tj = (P_mat * H_g[:, j]).sum(dim=(-1, -2))
                            trace_terms.append(tj * mask[:,j].to(self.dtype))
                        
                        trace_vec = torch.stack(trace_terms, dim=1)
                        tmp = torch.einsum('bij,bj->bi', G_inv, trace_vec)
                        H_corr = -torch.einsum('bdi,bi->bd', J, tmp)

                    drift = -torch.einsum('bij,bj->bi', P_mat, grad_f)
                    if p > 0 and G_inv is not None:
                        parts = [h_vals] if m > 0 else []
                        if l > 0: parts.append((g_vals + self.epsilon) * mask.to(self.dtype))
                        
                        J_vals = torch.cat(parts, dim=1)
                        tmp2   = torch.einsum('bij,bj->bi', G_inv, J_vals)
                        drift  = drift - self.alpha * torch.einsum('bdi,bi->bd', J, tmp2) + H_corr

                    xi = torch.randn_like(x)
                    projected_noise = torch.einsum('bij,bj->bi', P_mat, xi)

                    x  = x + self.step_size * drift + noise_scale * projected_noise
                    x = x.detach().requires_grad_()
                    traj[k+1] = x

        return {
            'trajectory': traj.detach().cpu().numpy(),
            'h_vals':     h_hist.detach().cpu().numpy(),
            'g_vals':     g_hist.detach().cpu().numpy()
        }

