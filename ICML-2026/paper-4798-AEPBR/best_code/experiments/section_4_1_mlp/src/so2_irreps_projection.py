# so2_equivariant_net.py
from typing import Callable, Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------- radial basis helper ---------------------------
def _gaussian_radial_basis(
    r: torch.Tensor, C: int, r_max: float, sigma_scale: float = 0.5
) -> torch.Tensor:
    centres = torch.linspace(0.0, r_max, C, device=r.device, dtype=r.dtype)  # (C,)
    sigma = sigma_scale * (r_max / C)
    return torch.exp(-((r[..., None] - centres[None, :]) ** 2) / (2.0 * sigma**2))


# ---------------------------------------------------------------------------


def project_to_irreps_radial(
    x: torch.Tensor, M: int, C: int = 4, r_max: float = 4.0, eps: float = 1e-6
) -> torch.Tensor:
    r"""
    Circular-harmonic ⇄ radial embedding.

    Returns
    -------
    feat : (B, 2*M+1, C) complex tensor
           slice [:, i, :] transforms with weight m = i−M.
    """
    assert x.dim() == 2 and x.size(1) == 2, "input must be (B,2)"
    B = x.size(0)

    # polar coordinates ------------------------------------------------------
    r = torch.linalg.norm(x, dim=-1).clamp_min(eps)  # (B,)
    z = (x[:, 0] / r) + 1j * (x[:, 1] / r)  # e^{iθ} (B,) complex

    # radial channels --------------------------------------------------------
    R = _gaussian_radial_basis(r, C=C, r_max=r_max)  # (B,C) real

    # angular × radial -------------------------------------------------------
    blocks = []
    for m in range(-M, M + 1):
        ang = (z**m) if m >= 0 else torch.conj(z ** (-m))  # (B,) complex
        # (B,1,1) × (B,1,C) → (B,1,C)  then keep extra dim 1 for cat
        block = (ang[:, None, None] * R[:, None, :]).to(torch.cfloat)  # (B,1,C)
        blocks.append(block)

    feat = torch.cat(blocks, dim=1)  # (B, 2M+1, C)
    return feat
