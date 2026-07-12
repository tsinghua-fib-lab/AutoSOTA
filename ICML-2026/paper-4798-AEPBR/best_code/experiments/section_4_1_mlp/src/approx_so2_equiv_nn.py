# approx_harmonic_invariant.py
import math
from typing import Dict, List

import torch
import torch.nn as nn

from so2_irreps_projection import project_to_irreps_radial
from approx_so2_linear import RelaxSO2Linear
from so2_tensor_product import so2_tensor_product

class ApproxHarmonicInvariantMLP(nn.Module):
    """
    Relaxed SO(2)-equivariant network

        project → RelaxLinear → RelaxLinear → RelaxTensorProduct → invariant head
    """

    def __init__(self, M: int = 4, C: int = 4, hidden_c: int = 8):
        super().__init__()
        self.M, self.C = M, C
        self.m_vals: List[int] = list(range(-M, M + 1))

        # channel dictionaries per irrep
        in_ch = {m: C for m in self.m_vals}
        hid_ch = {m: hidden_c for m in self.m_vals}

        # layers
        self.lin1 = RelaxSO2Linear(in_ch, hid_ch, bias=False)
        self.lin2 = RelaxSO2Linear(hid_ch, hid_ch, bias=False)

        inv_dim = len(self.m_vals) * hidden_c * hidden_c
        self.fc = nn.Linear(inv_dim, 1)

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # harmonic+radial embed : (B, 2M+1, C) → dict {m:(B,C)}
        feat = project_to_irreps_radial(x, self.M, self.C)
        feat = {m: feat[:, i] for i, m in enumerate(self.m_vals)}

        # Linear → Linear
        feat = self.lin1(feat)
        feat = self.lin2(feat)

        # exact invariant contraction
        conj = {-m: torch.conj(t) for m, t in feat.items()}
        inv = so2_tensor_product(feat, conj)[0].real  # (B, h*h*(2M+1))

        return self.fc(inv.reshape(inv.size(0), -1))  # (B,1)

    # ------------------------------------------------------------------ #
    def compute_non_equivariance_penalty(self) -> Dict[str, torch.Tensor]:
        """
        Returns Frobenius-norm sums over *all* relaxed layers:
          {'equivariant_part': Σ‖W_eq‖ , 'nonequiv_part': Σ‖W_non‖}
        """
        peq, pne = 0.0, 0.0
        for mod in self.modules():
            if isinstance(mod, (RelaxSO2Linear)):
                eq, ne = mod.penalty_terms()
                peq = peq + eq
                pne = pne + ne
        return {"equivariant_part": peq, "nonequiv_part": pne}
