from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from cycloreps.translator.translator import MultiSpaceBase

def _l2norm_rows(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(dim=1, keepdim=True) + eps)


@torch.no_grad()
def _build_consensus(U: Dict[str, torch.Tensor], eps: float = 1e-8) -> torch.Tensor:
    """Mean of per-view unit directions, re-normalized."""
    names = list(U.keys())
    acc = 0.0
    for n in names:
        acc = acc + _l2norm_rows(U[n], eps=eps)
    return _l2norm_rows(acc / float(len(names)), eps=eps)

def _sample_batch(
    U: Dict[str, torch.Tensor],
    C: torch.Tensor,
    names: list[str],
    batch_size: int,
    idxs: torch.Tensor,
    *,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    m = int(torch.randint(0, len(names), (1,), device=device).item())
    name = names[m]
    jj = torch.randint(0, idxs.numel(), (batch_size,), device=device)
    ii = idxs[jj]
    return U[name][ii], C[ii]


# Geometry Correction
class _GCCorrector(nn.Module):
    def __init__(
        self,
        d: int,
        *,
        hidden_mult: float = 2.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        h = max(8, int(round(hidden_mult * d)))

        self.ln = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, h)
        self.fc2 = nn.Linear(h, d)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.gate = nn.Linear(d, d)

        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, u: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
        u_hat = _l2norm_rows(u, eps=eps)
        x = self.ln(u_hat)

        h = F.gelu(self.fc1(x))
        h = self.drop(h)
        delta = self.fc2(h)

        g = torch.sigmoid(self.gate(x))
        u_corr = u_hat + (g * delta)
        return _l2norm_rows(u_corr, eps=eps)


class GeneralizedProcrustesTranslator(MultiSpaceBase):
    """Generalized Procrustes with optional post-hoc geometry correction."""

    def __init__(
        self,
        *,
        max_iter: int = 50,
        tol: float = 1e-6,
        alignment_kind: str = "cycle-consistent",
        device: str = "cpu",
        gc_enabled: bool = False,
        gc_epochs: int = 50,
        gc_steps_per_epoch: int = 200,
        gc_batch_size: int = 4096,
        gc_lr: float = 2e-3,
        gc_weight_decay: float = 1e-4,
        gc_hidden_mult: float = 2.0,
        gc_dropout: float = 0.0,
        gc_val_fraction: float = 0.1,
        gc_tau: float = 0.08,
        gc_lam: float = 0.7,
        gc_num_restarts: int = 1,
        gc_val_batches: int = 50,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(
            name="gpa_multi_space_translator",
            max_iter=max_iter,
            tol=tol,
            device=device,
        )
        self._alignment_kind = alignment_kind
        self._uses_padding = True

        self.gc_enabled = bool(gc_enabled)
        self.gc_epochs = int(gc_epochs)
        self.gc_steps_per_epoch = int(gc_steps_per_epoch)
        self.gc_train_batch_size = int(gc_batch_size)
        self.gc_val_batch_size = int(gc_batch_size)
        self.gc_lr = float(gc_lr)
        self.gc_weight_decay = float(gc_weight_decay)
        self.gc_hidden_mult = float(gc_hidden_mult)
        self.gc_dropout = float(gc_dropout)
        self.gc_val_fraction = float(gc_val_fraction)
        self.gc_tau = float(gc_tau)
        self.gc_lam = float(gc_lam)
        self.gc_num_restarts = int(gc_num_restarts)

        self.gc_val_batches = int(gc_val_batches)

        self.eps = float(eps)

        self.R_out: Dict[str, torch.Tensor] = {}
        self.T_out: Dict[str, torch.Tensor] = {}
        self._corrector: Optional[_GCCorrector] = None
        self._use_gc_override: Optional[bool] = None
        self._gc_rescale_override: Optional[bool] = None

    def _fit_impl(self, spaces_std: Dict[str, torch.Tensor], dims: Dict[str, int]) -> None:
        self.R_out = self.align(
            spaces=spaces_std,
            dims=dims,
            max_iter=self.max_iter,
            tol=self.tol,
            device=self.device,
        )
        self.T_out.clear()
        self._corrector = None

        if not self.gc_enabled:
            return

        names = list(spaces_std.keys())
        X = {n: spaces_std[n].to(self.device, torch.float32) for n in names}
        N, d = X[names[0]].shape
        if any(t.shape != (N, d) for t in X.values()):
            raise ValueError("All spaces must share shape (N, d) — pad first if needed.")

        U = {n: X[n] @ self.R_out[n].to(self.device, torch.float32) for n in names}
        # per view unit directions after GPA is fit
        C = _build_consensus(U, eps=self.eps)

        # MLP init for the post hoc refinement
        gc_device = torch.device("cuda") if torch.cuda.is_available() else self.device
        corrector = _GCCorrector(
            d,
            hidden_mult=self.gc_hidden_mult,
            dropout=self.gc_dropout,
        ).to(gc_device)

        U_gc = {n: U[n].to(gc_device) for n in names}
        C_gc = C.to(gc_device)
        with torch.enable_grad():
            self._fit_gc(corrector, U=U_gc, C=C_gc)
        if gc_device != self.device:
            corrector.to(self.device)
        self._corrector = corrector

    # override parent class to provide option for using refinement or not
    def to_universe(
        self,
        x: torch.Tensor,
        *,
        src: str,
        use_gc: Optional[bool] = None,
        gc_rescale: Optional[bool] = None,
    ) -> torch.Tensor:
        """Convert to universe coordinates."""
        self._use_gc_override = use_gc
        self._gc_rescale_override = gc_rescale
        try:
            return super().to_universe(x, src=src)
        finally:
            self._use_gc_override = None
            self._gc_rescale_override = None

    def _to_universe_impl(
        self,
        z: torch.Tensor,
        *,
        src: str,
        use_gc: Optional[bool] = None,
        gc_rescale: Optional[bool] = None,
    ) -> torch.Tensor:
        U = z @ self.R_out[src].to(z.device, z.dtype)

        if use_gc is None:
            use_gc = self._use_gc_override
        if gc_rescale is None:
            gc_rescale = self._gc_rescale_override
        if gc_rescale is None:
            gc_rescale = False
        want_gc = self.gc_enabled if use_gc is None else bool(use_gc)
        # corrected should have been run
        if want_gc and self._corrector is not None:
            U_corr_dir = self._corrector(U, eps=self.eps)  # unit directions
            if bool(gc_rescale):
                r = U.norm(dim=1, keepdim=True).clamp_min(self.eps)  # (N,1)
                return U_corr_dir * r
            return U_corr_dir
        return U

    def _from_universe_impl(self, u: torch.Tensor, *, tgt: str) -> torch.Tensor:
        return u @ self.R_out[tgt].to(u.device, u.dtype).T

    def _pairwise_map_impl(self, src: str, tgt: str) -> torch.Tensor:
        return self.R_out[src] @ self.R_out[tgt].T

    # GPA alignment
    @torch.no_grad()
    def align(
        self,
        spaces: Dict[str, torch.Tensor],
        dims,
        *,
        max_iter: int = 50,
        tol: float = 1e-6,
        device: str = "cpu",
    ) -> Dict[str, torch.Tensor]:
        names = list(spaces)
        N, d = spaces[names[0]].shape
        if any(t.shape != (N, d) for t in spaces.values()):
            raise ValueError("All spaces must share shape (N, d) — pad first if needed.")

        X = {n: t.to(device, torch.float32) for n, t in spaces.items()}

        # SVD-based initialization: use optimal pairwise Procrustes rotation
        # from each view to the first (reference) view
        ref_name = names[0]
        R = {}
        for n in names:
            if n == ref_name:
                R[n] = torch.eye(d, device=device)
            else:
                d_n = int(dims[n])
                d_ref = int(dims[ref_name])
                G = X[n][:, :d_n].T @ X[ref_name][:, :d_ref]
                U_, _, Vt = torch.linalg.svd(G, full_matrices=False)
                R_new = torch.eye(d, device=device)
                md = min(d_n, d_ref)
                R_new[:md, :md] = (U_ @ Vt)[:md, :md]
                R[n] = R_new
        M = len(names)

        consensus = sum(X[m] @ R[m] for m in spaces) / M

        for _ in tqdm(range(max_iter), desc="GPA", leave=False):
            for m in spaces:
                d_m = int(dims[m])
                G_block = X[m][:, :d_m].T @ consensus[:, :d_m]  # (d_m, d_m)
                U_, _, Vt = torch.linalg.svd(G_block, full_matrices=False)
                R_block = U_ @ Vt

                R_new = torch.eye(d, device=device)
                R_new[:d_m, :d_m] = R_block
                R[m] = R_new

            new_consensus = sum(X[m] @ R[m] for m in spaces) / M
            rel = torch.norm(new_consensus - consensus) / (torch.norm(consensus) + 1e-12)
            consensus = new_consensus
            if rel < tol:
                break

        return {n: R[n] for n in names}

    # Geometry correction refinement
    def _fit_gc(self, corrector: _GCCorrector, *, U: Dict[str, torch.Tensor], C: torch.Tensor) -> None:
        """
        Train corrector to predict consensus direction:
            minimize mean_i,m (1 - cos(corrector(u_m,i), C_i))
        """
        names = list(U.keys())
        M = len(names)
        N = C.shape[0]
        device = C.device

        assert self.gc_val_fraction > 0.0, "validation fraction should be greater than 0"

        # take a subset for validation
        n_val = max(1, int(round(self.gc_val_fraction * N)))
        perm = torch.randperm(N, device=device)
        val_idx = perm[:n_val]
        tr_idx = perm[n_val:]
        train_bs = min(len(tr_idx),  self.gc_train_batch_size) # use large batch size for large datasets
        steps_per_epoch = self.gc_steps_per_epoch
        if (steps_per_epoch * train_bs) < len(tr_idx): # while we are sampling with replacement, still ensuring more steps to match
            steps_per_epoch = int(math.ceil(len(tr_idx) / max(1, train_bs)))
        
        val_bs = min(len(val_idx), self.gc_val_batch_size)

        val_batches = self.gc_val_batches
        if (val_batches * val_bs) < len(val_idx):
            val_batches = int(math.ceil(len(val_idx) / max(1, val_bs)))

        val_check_every = 20 if val_bs < self.gc_val_batch_size else 40 # avoid checking frequently for larger datasets
        bad_steps_setting = 50 if val_bs < self.gc_val_batch_size else 100 # more patience for larger datasets

        opt = torch.optim.AdamW(
            corrector.parameters(),
            lr=self.gc_lr,
            weight_decay=self.gc_weight_decay,
        )

        best_val = 10000
        best_state = None
        bad_steps = 0

        total_steps = self.gc_epochs * steps_per_epoch
        corrector.train()
        pbar = tqdm(range(total_steps), desc="GC", leave=False)
        for step in pbar:
            u_b, t_b = _sample_batch(
                U,
                C,
                names,
                train_bs,
                tr_idx,
                device=device,
            )
            u_hat = _l2norm_rows(u_b, eps=self.eps)
            y_b   = corrector(u_b, eps=self.eps)
            t_b   = _l2norm_rows(t_b, eps=self.eps)

            loss_align = (1.0 - (y_b * t_b).sum(dim=1))
            drift      = (1.0 - (y_b * u_hat).sum(dim=1))

            tau = self.gc_tau
            lam = self.gc_lam

            loss_trust = torch.relu(drift - tau)
            loss = loss_align.mean() + lam * loss_trust.mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(corrector.parameters(), 1.0)
            opt.step()
            pbar.set_postfix(loss=float(loss.item()))

            if (step + 1) % val_check_every != 0: # check validation every few steps
                continue

            corrector.eval()
            with torch.no_grad():
                val = 0.0
                n_batches = 0
                for _ in range(val_batches):
                    ii = val_idx[
                        torch.randint(0, val_idx.numel(), (val_bs,), device=device)
                    ]
                    m = int(torch.randint(0, len(names), (1,), device=device).item())
                    name = names[m]
                    u_b = U[name][ii]
                    t_b = C[ii]

                    u_hat = _l2norm_rows(u_b, eps=self.eps)
                    y_b   = corrector(u_b, eps=self.eps)
                    t_b   = _l2norm_rows(t_b, eps=self.eps)

                    loss_align = (1.0 - (y_b * t_b).sum(dim=1))
                    drift      = (1.0 - (y_b * u_hat).sum(dim=1))
                    loss_trust = torch.relu(drift - tau)

                    val += (loss_align.mean() + lam * loss_trust.mean()).item()
                    n_batches += 1

                val /= max(1, n_batches)

            if val < best_val:
                best_val = val
                bad_steps = 0
                best_state = {k: v.detach().cpu().clone() for k, v in corrector.state_dict().items()}
            else:
                bad_steps += 1

            if bad_steps >= bad_steps_setting:
                break

            corrector.train()

        if best_state is not None:
            corrector.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})
        return best_val
