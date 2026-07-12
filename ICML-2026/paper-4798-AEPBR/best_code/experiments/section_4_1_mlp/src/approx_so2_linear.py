# relax_so2_linear.py
import math

import torch
import torch.nn as nn
from typing import Dict, List, Tuple


class RelaxSO2Linear(nn.Module):
    r"""
    A linear map that can **mix different SO(2) irreps**, but stores
    a *mask* so we can
      • extract the equivariant (block-diagonal) projection **W_eq**,
      • measure the non-equivariant part **W_non = W − W_eq**.

    Parameters
    ----------
    in_irreps  : dict {m : Cin(m)}
    out_irreps : dict {m : Cout(m)}
    bias       : bool  (optional)
    """

    def __init__(
        self, in_irreps: Dict[int, int], out_irreps: Dict[int, int], bias: bool = False
    ):
        super().__init__()

        self.m_vals: List[int] = sorted(in_irreps)
        assert self.m_vals == sorted(out_irreps)

        # ---- flatten channel layout ---------------------------------------
        self.in_slices, self.C_in_tot = self._build_slices(in_irreps)
        self.out_slices, self.C_out_tot = self._build_slices(out_irreps)

        # ---- trainable full weight ----------------------------------------
        w = torch.randn(self.C_out_tot, self.C_in_tot, dtype=torch.cfloat) / math.sqrt(
            self.C_in_tot
        )
        self.weight = nn.Parameter(w)

        if bias:
            self.bias = nn.Parameter(torch.zeros(self.C_out_tot, dtype=torch.cfloat))
        else:
            self.bias = None

        # ---- binary mask for equivariant projection -----------------------
        mask = torch.zeros_like(self.weight)
        for m in self.m_vals:
            r = self.out_slices[m]
            c = self.in_slices[m]
            mask[r, c] = 1.0
        self.register_buffer("mask_eq", mask, persistent=False)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_slices(ir: Dict[int, int]) -> Tuple[Dict[int, slice], int]:
        slices, start = {}, 0
        for m in sorted(ir):
            dim = ir[m]
            slices[m] = slice(start, start + dim)
            start += dim
        return slices, start

    # ------------------------------------------------------------------ #
    # forward pass: flatten -> matmul -> split
    # ------------------------------------------------------------------ #
    def forward(self, x: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        # concatenate channel axis
        x_cat = torch.cat([x[m] for m in self.m_vals], dim=-1)  # (..., Cin)
        W = self.weight.to(x_cat.dtype)  # <── cast
        y_cat = torch.einsum("oi,...i->...o", W, x_cat)
        if self.bias is not None:
            y_cat = y_cat + self.bias

        # split back into blocks
        out: Dict[int, torch.Tensor] = {}
        for m in self.m_vals:
            sl = self.out_slices[m]
            out[m] = y_cat[..., sl]
        return out

    # ------------------------------------------------------------------ #
    #   PENALTY helpers
    # ------------------------------------------------------------------ #
    def equivariant_projection(self) -> torch.Tensor:
        """Block-diagonal part W_eq"""
        return self.weight * self.mask_eq

    def penalty_terms(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns  (‖W_eq‖_F , ‖W − W_eq‖_F)
        You can weigh them differently in the loss.
        """
        W_eq = self.equivariant_projection()
        W_non = self.weight - W_eq
        return W_eq.norm(), W_non.norm()
