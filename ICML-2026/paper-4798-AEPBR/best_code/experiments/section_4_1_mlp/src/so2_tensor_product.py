import torch
import torch.nn as nn
from typing import Dict, Any, Tuple


def so2_tensor_product(
    a: Dict[int, torch.Tensor], b: Dict[int, torch.Tensor]
) -> Dict[int, torch.Tensor]:
    """
    Computes the outer product of two SO(2)-equivariant feature dictionaries.

    For every pair (m1, m2) it produces channels transforming with weight
        m_out = m1 + m2
    and concatenates all contributions that land in the same m_out.

    Parameters
    ----------
    a, b : dict  {m : (..., C_m)}   complex Tensors

    Returns
    -------
    out : dict {m_out : (..., C_out(m_out))}
           where  C_out(m_out) = Σ_{m1+m2=m_out} C_a(m1)·C_b(m2)
    """
    out: Dict[int, torch.Tensor] = {}

    for m1, A in a.items():
        for m2, B in b.items():
            m_out = m1 + m2

            # Outer product on the last (channel) axis; flatten it back
            # (..., Ca, Cb) -> (..., Ca*Cb)
            prod = torch.einsum("...i,...j->...ij", A, B).reshape(*A.shape[:-1], -1)

            if m_out in out:
                out[m_out] = torch.cat([out[m_out], prod], dim=-1)
            else:
                out[m_out] = prod

    return out
