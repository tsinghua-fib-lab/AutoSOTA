# core/ekfac_blocks.py

import torch
import torch.distributed as dist
from typing import List, Dict, Optional


class EKFAC_QK_Head:
    """
    EK-FAC for QK-only (single block):
      W_QK: d_in=d_model, d_out=2*d_head
    A = E[X^T X], S = E[G_qk^; dK]
    Lambda is fitted by full-pass ge^2 averaging in the A/S eigenbasis.
    """
    def __init__(self, d_model: int, d_head: int, damping: float = 1e-5, damping_alpha: float = 0.1):
        self.d_model = d_model
        self.d_head = d_head
        self.block = {'name': 'W_QK', 'd_in': d_model, 'd_out': 2 * d_head}
        self.damping = float(damping)
        self.damping_alpha = float(damping_alpha)

        self.A_accum: Optional[torch.Tensor] = None
        self.S_accum: Optional[torch.Tensor] = None
        self.token_count: int = 0

        self.Q_A: Optional[torch.Tensor] = None
        self.Q_S: Optional[torch.Tensor] = None
        self.Lambda: Optional[torch.Tensor] = None

    def accumulate_A_S(self, X_flat_f32, dQ_f32, dK_f32):
        BStok = X_flat_f32.shape[0]
        if BStok == 0:
            return

        X = X_flat_f32.detach()
        dQ = dQ_f32.detach()
        dK = dK_f32.detach()

        G_qk = torch.cat([dQ, dK], dim=-1)  # [B*S, 2*d_head]
        A = X.t() @ X
        S = G_qk.t() @ G_qk

        if self.A_accum is None:
            self.A_accum = A
            self.S_accum = S
        else:
            self.A_accum.add_(A)
            self.S_accum.add_(S)

        self.token_count += int(BStok)

    def finalize_eigendecomposition(self, device=None):
        if self.token_count == 0:
            raise RuntimeError("No tokens accumulated for EK-FAC A/S.")

        if dist.is_initialized():
            dist.all_reduce(self.A_accum, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.S_accum, op=dist.ReduceOp.SUM)
            tok_t = torch.tensor([self.token_count], device=device, dtype=torch.long) if device is not None else torch.tensor([self.token_count], dtype=torch.long)
            dist.all_reduce(tok_t, op=dist.ReduceOp.SUM)
            global_tokens = tok_t.item()
        else:
            global_tokens = self.token_count

        A = self.A_accum / float(global_tokens)
        S = self.S_accum / float(global_tokens)
        A = 0.5 * (A + A.t())
        S = 0.5 * (S + S.t())

        eps_A = (1e-6 * torch.trace(A).abs() / A.shape[0]).to(A.dtype)
        eps_S = (1e-6 * torch.trace(S).abs() / S.shape[0]).to(S.dtype)
        A = A + eps_A * torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
        S = S + eps_S * torch.eye(S.shape[0], device=S.device, dtype=S.dtype)

        _, self.Q_A = torch.linalg.eigh(A.float())
        _, self.Q_S = torch.linalg.eigh(S.float())

    def inverse_hvp(self, grad_matrix: torch.Tensor) -> torch.Tensor:
        if self.Q_A is None or self.Q_S is None or self.Lambda is None:
            raise RuntimeError("EK-FAC not ready for inverse_hvp")
        G = grad_matrix.float()
        ge = self.Q_A.t() @ G @ self.Q_S
        lam = self.Lambda
        denom = lam + self.damping_alpha * lam.mean()
        denom = torch.clamp(denom, min=self.damping)
        ihvp_eig_flat = ge.flatten() / denom
        d_in = self.block['d_in']
        d_out = self.block['d_out']
        ihvp = self.Q_A @ ihvp_eig_flat.reshape(d_in, d_out) @ self.Q_S.t()
        return ihvp

    @property
    def block_dims(self) -> List[Dict[str, int]]:
        return [{'d_in': self.block['d_in'], 'd_out': self.block['d_out']}]


class EKFAC_QKVO_Head:
    """
    EK-FAC for Q):
      0: W_QK -> d_in=d_model, d_out=2*d_head
      1: W_V  -> d_in=d_model, d_out=d_head
      2: W_O  -> d_in=d_head,  d_out=d_model
    A/S are token expectations; Lambda is sequence-level ge^2 averaged.
    """
    def __init__(self, d_model: int, d_head: int, damping: float = 1e-5, damping_alpha: float = 0.1):
        self.d_model = d_model
        self.d_head = d_head
        self.damping = float(damping)
        self.damping_alpha = float(damping_alpha)
        self.blocks = [
            {'name': 'W_QK', 'd_in': d_model, 'd_out': 2 * d_head},
            {'name': 'W_V',  'd_in': d_model, 'd_out': d_head},
            {'name':'W_O',  'd_in': d_head,  'd_out': d_model},
        ]
        self.A_accum = [None, None, None]
        self.S_accum = [None, None, None]
        self.token_count = 0

        self.Q_A = [None, None, None]
        self.Q_S = [None, None, None]
        self.Lambda = [None, None, None]

    def accumulate_A_S(self, X_flat_f32, dQ_f32, dK_f32, dV_f32, Z_flat_f32, dR_f32):
        BStok = X_flat_f32.shape[0] if X_flat_f32 is not None else (Z_flat_f32.shape[0] if Z_flat_f32 is not None else 0)
        if BStok == 0:
            return

        X = X_flat_f32.detach()
        Z = Z_flat_f32.detach()
        dQ = dQ_f32.detach()
        dK = dK_f32.detach()
        dV = dV_f32.detach()
        dR = dR_f32.detach()

        # Block 0: QK
        G_qk = torch.cat([dQ, dK], dim=-1)
        A0 = X.t() @ X
        S0 = G_qk.t() @ G_qk
        if self.A_accum[0] is None:
            self.A_accum[0] = A0
            self.S_accum[0] = S0
        else:
            self.A_accum[0].add_(A0)
            self.S_accum[0].add_(S0)

        # Block 1: V
        A1 = X.t() @ X
        S1 = dV.t() @ dV
        if self.A_accum[1] is None:
            self.A_accum[1] = A1
            self.S_accum[1] = S1
        else:
            self.A_accum[1].add_(A1)
            self.S_accum[1].add_(S1)

        # Block 2: O
        A2 = Z.t() @ Z
        S2 = dR.t() @ dR
        if self.A_accum[2] is None:
            self.A_accum[2] = A2
            self.S_accum[2] = S2
        else:
            self.A_accum[2].add_(A2)
            self.S_accum[2].add_(S2)

        self.token_count += int(BStok)

    def finalize_eigendecomposition(self, device=None):
        if self.token_count == 0:
            raise RuntimeError("No tokens accumulated for EK-FAC A/S.")

        if dist.is_initialized():
            for i in range(3):
                dist.all_reduce(self.A_accum[i], op=dist.ReduceOp.SUM)
                dist.all_reduce(self.S_accum[i], op=dist.ReduceOp.SUM)
            tok_t = torch.tensor([self.token_count], device=device, dtype=torch.long) if device is not None else torch.tensor([self.token_count], dtype=torch.long)
            dist.all_reduce(tok_t, op=dist.ReduceOp.SUM)
            global_tokens = tok_t.item()
        else:
            global_tokens = self.token_count

        for i in range(3):
            A = self.A_accum[i] / float(global_tokens)
            S = self.S_accum[i] / float(global_tokens)
            A = 0.5 * (A + A.t())
            S = 0.5 * (S + S.t())
            eps_A = (1e-6 * torch.trace(A).abs() / A.shape[0]).to(A.dtype)
            eps_S = (1e-6 * torch.trace(S).abs() / S.shape[0]).to(S.dtype)
            A = A + eps_A * torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
            S = S + eps_S * torch.eye(S.shape[0], device=S.device, dtype=S.dtype)

            _, self.Q_A[i] = torch.linalg.eigh(A.float())
            _, self.Q_S[i] = torch.linalg.eigh(S.float())

    def inverse_hvp(self, block_idx: int, grad_matrix: torch.Tensor) -> torch.Tensor:
        assert 0 <= block_idx < 3
        if self.Q_A[block_idx] is None or self.Q_S[block_idx] is None or self.Lambda[block_idx] is None:
            raise RuntimeError("EK-FAC not ready for block %d" % block_idx)
        G = grad_matrix.float()
        QA, QS = self.Q_A[block_idx], self.Q_S[block_idx]
        ge = QA.t() @ G @ QS
        lam = self.Lambda[block_idx]
        denom = lam + self.damping_alpha * lam.mean()
        denom = torch.clamp(denom, min=self.damping)
        ihvp_eig_flat = ge.flatten() / denom
        d_in = self.blocks[block_idx]['d_in']
        d_out = self.blocks[block_idx]['d_out']
        ihvp = QA @ ihvp_eig_flat.reshape(d_in, d_out) @ QS.t()
        return ihvp

    @property
    def block_dims(self) -> List[Dict[str, int]]:
        return [{'d_in': b['d_in'], 'd_out': b['d_out']} for b in self.blocks]
