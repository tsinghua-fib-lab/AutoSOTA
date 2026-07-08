from __future__ import annotations
from typing import List, Tuple, Union
import numpy as np
import torch

TensorLike = Union[np.ndarray, torch.Tensor]

# ---------------------- Candidate generation & screening (GPU) -----------------

@torch.no_grad()
def screen_pairs_uncertain(marginals: TensorLike, top_k: int) -> List[Tuple[int, int]]:
    """Vectorized uncertainty screening.

    Matches generation.screen_pairs_uncertain:
      - considers unordered pairs (i<j)
      - score = max(p_ij*(1-p_ij), p_ji*(1-p_ji))
      - returns top_k pairs sorted by descending score

    Accepts numpy array or torch tensor. Computation uses torch.
    """
    if isinstance(marginals, np.ndarray):
        M = torch.from_numpy(marginals)
    else:
        M = marginals
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"marginals must be (D,D), got {tuple(M.shape)}")

    D = M.shape[0]
    device = M.device
    M = M.to(dtype=torch.float64)

    # Upper-triangular indices (unordered pairs)
    iu, ju = torch.triu_indices(D, D, offset=1, device=device)
    p_ij = M[iu, ju]
    p_ji = M[ju, iu]

    s_ij = p_ij * (1.0 - p_ij)
    s_ji = p_ji * (1.0 - p_ji)
    score = torch.maximum(s_ij, s_ji)

    k = min(int(top_k), score.numel())
    if k <= 0:
        return []

    topv, topidx = torch.topk(score, k=k, largest=True, sorted=True)
    ii = iu[topidx].tolist()
    jj = ju[topidx].tolist()
    return list(zip(ii, jj))


@torch.no_grad()
def screen_pairs_uncertain_ordered(marginals: TensorLike, top_k: int) -> List[Tuple[int, int]]:
    """Vectorized ordered uncertainty screening.

    Matches generation.screen_pairs_uncertain_ordered:
      - considers ordered pairs i!=j
      - score = p*(1-p)
      - returns top_k pairs sorted by descending score
    """
    if isinstance(marginals, np.ndarray):
        M = torch.from_numpy(marginals)
    else:
        M = marginals
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"marginals must be (D,D), got {tuple(M.shape)}")

    D = M.shape[0]
    device = M.device
    M = M.to(dtype=torch.float64)

    # all ordered pairs excluding diagonal
    ii, jj = torch.meshgrid(torch.arange(D, device=device), torch.arange(D, device=device), indexing='ij')
    mask = ii != jj
    ii = ii[mask]
    jj = jj[mask]
    p = M[ii, jj]
    score = p * (1.0 - p)

    k = min(int(top_k), score.numel())
    if k <= 0:
        return []

    _, topidx = torch.topk(score, k=k, largest=True, sorted=True)
    return list(zip(ii[topidx].tolist(), jj[topidx].tolist()))
