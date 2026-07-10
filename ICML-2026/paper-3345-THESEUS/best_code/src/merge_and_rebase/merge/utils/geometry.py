from __future__ import annotations

import torch


def orthonormal_basis(x: torch.Tensor) -> torch.Tensor:
    return torch.linalg.qr(x.to(dtype=torch.float64), mode="reduced")[0].contiguous()


def normalized_subspace_similarity(basis_i: torch.Tensor, basis_j: torch.Tensor) -> float:
    if int(basis_i.shape[1]) != int(basis_j.shape[1]):
        raise ValueError("Subspace similarity requires matching subspace rank.")
    rank = int(basis_i.shape[1])
    overlap = basis_i.to(dtype=torch.float64).T @ basis_j.to(dtype=torch.float64)
    score = torch.sum(overlap * overlap) / max(1, rank)
    return float(score.item())


def pairwise_subspace_similarity(basis_i: torch.Tensor, basis_j: torch.Tensor) -> float:
    rank_i = int(basis_i.shape[1])
    rank_j = int(basis_j.shape[1])
    if rank_i == 0 or rank_j == 0:
        return 0.0
    overlap = basis_i.to(dtype=torch.float64).T @ basis_j.to(dtype=torch.float64)
    denom = float(min(rank_i, rank_j))
    score = torch.sum(overlap * overlap) / max(1.0, denom)
    return float(score.item())
