from __future__ import annotations

import torch


def sort_by_fitness(xp: torch.Tensor, minimize: bool = True) -> torch.Tensor:
    """Sort population tensor by fitness.

    Args:
        xp: Tensor of shape (B, N, D) where fitness is stored in xp[..., 0].
        minimize: If True, lower fitness is better. If False, higher is better.

    Returns:
        A tensor with the same shape as xp, sorted along the population dimension.
    """
    if xp.ndim != 3:
        raise ValueError(f"Expected xp with shape (B, N, D), got {tuple(xp.shape)}")
    fitness = xp[..., 0]
    idx = torch.argsort(fitness, dim=1, descending=not bool(minimize))
    return torch.gather(xp, dim=1, index=idx.unsqueeze(-1).expand_as(xp))


def sortIndivBND(batchPop: torch.Tensor, minimize: bool = True) -> torch.Tensor:
    """Backward-compatible alias of sort_by_fitness()."""
    return sort_by_fitness(batchPop, minimize=minimize)


def one2one_selection(x: torch.Tensor, u: torch.Tensor, minimize: bool = True) -> torch.Tensor:
    """One-to-one selection between parent x and candidate u.

    For each individual i, choose the better one according to fitness (index 0).

    Args:
        x: Parent population, shape (B, N, D).
        u: Candidate population, shape (B, N, D).
        minimize: If True, lower fitness is better. If False, higher is better.

    Returns:
        Next population, shape (B, N, D).
    """
    if x.shape != u.shape:
        raise ValueError(f"Shape mismatch: x={tuple(x.shape)} u={tuple(u.shape)}")
    if x.ndim != 3:
        raise ValueError(f"Expected x/u with shape (B, N, D), got {tuple(x.shape)}")

    if minimize:
        keep_parent = x[..., 0] < u[..., 0]
    else:
        keep_parent = x[..., 0] > u[..., 0]

    return torch.where(keep_parent.unsqueeze(-1), x, u)
