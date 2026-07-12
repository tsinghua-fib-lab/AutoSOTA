# test_relax_so2_tensor.py
import math

import torch
import pytest

from approx_so2_tensor_product import RelaxSO2TensorProduct


# -------------------------------------------------------------------- #
def rand_dict(batch, irreps):
    """helper: {m: (B, C_m) complex}"""
    return {m: torch.randn(batch, c, dtype=torch.cfloat) for m, c in irreps.items()}


def _rotate(feat, alpha):
    return {
        m: v
        * (torch.cos(torch.tensor(alpha * m)) + 1j * torch.sin(torch.tensor(alpha * m)))
        for m, v in feat.items()
    }


# -------------------------------------------------------------------- #
@pytest.mark.parametrize("batch", [2, 5])
def test_shapes_forward(batch):
    in_irA = {-2: 3, -1: 2, 0: 1, 1: 2, 2: 3}
    in_irB = {m: c + 1 for m, c in in_irA.items()}
    out_ir = {m: c + 2 for m, c in in_irA.items()}

    tp = RelaxSO2TensorProduct(in_irA, in_irB, out_ir)

    a = rand_dict(batch, in_irA)
    b = rand_dict(batch, in_irB)
    z = tp(a, b)

    for m in out_ir:
        assert z[m].shape == (batch, out_ir[m])
        assert z[m].dtype == torch.cfloat


# -------------------------------------------------------------------- #
def test_mask_and_penalty():
    in_ir = {-1: 2, 0: 2, 1: 2}
    tp = RelaxSO2TensorProduct(in_ir, in_ir, in_ir)

    W = tp.weight.detach()
    mask = tp.mask_eq.detach()

    # --- 1. mask has ones exactly on blocks with m_out = m1+m2 ----------
    for mo in tp.mO:
        o_sl = tp.slice_o[mo]
        for m1 in tp.mA:
            a_sl = tp.slice_a[m1]
            for m2 in tp.mB:
                b_sl = tp.slice_b[m2]
                block = mask[o_sl, a_sl, b_sl]
                should_keep = mo == m1 + m2
                if should_keep:
                    assert block.all(), "mask zeroed a required block"
                else:
                    assert not block.any(), "mask kept a forbidden block"

    # --- 2. penalty_terms equals manual computation ---------------------
    p_eq, p_ne = tp.penalty_terms()
    W_eq = W * mask
    W_non = W - W_eq
    assert torch.allclose(p_eq, W_eq.norm())
    assert torch.allclose(p_ne, W_non.norm())


# -------------------------------------------------------------------- #
def test_grad_through_penalty():
    in_ir = {0: 3, 1: 3}
    tp = RelaxSO2TensorProduct(in_ir, in_ir, in_ir)

    a = rand_dict(4, in_ir)
    out = tp(a, a)
    loss = sum(v.real.mean() for v in out.values())

    peq, pne = tp.penalty_terms()
    total = loss + 0.05 * (peq + pne)
    total.backward()

    # every parameter in weight tensor gets gradient
    assert tp.weight.grad is not None and tp.weight.grad.abs().sum() > 0


# -------------------------------------------------------------------- #
@pytest.mark.parametrize("batch", [3])
def test_projected_tensor_equivariant(batch):
    irreps = {-1: 2, 0: 2, 1: 2}
    tp = RelaxSO2TensorProduct(irreps, irreps, irreps)

    with torch.no_grad():  # keep only legal blocks
        tp.weight.mul_(tp.mask_eq)

    a = rand_dict(batch, irreps)
    b = rand_dict(batch, irreps)
    alpha = torch.rand(1).item() * 2 * math.pi
    a_rot = _rotate(a, alpha)
    b_rot = _rotate(b, alpha)

    z = tp(a, b)
    z_rot = tp(a_rot, b_rot)

    for m in irreps:
        phase = torch.exp(1j * torch.tensor(m * alpha))
        assert torch.allclose(z_rot[m], z[m] * phase, atol=1e-6, rtol=1e-6)


# -------------------------------------------------------------------- #
@pytest.mark.parametrize("batch", [3])
def test_projected_tensor_equivariant_not(batch):
    irreps = {-1: 2, 0: 2, 1: 2}
    tp = RelaxSO2TensorProduct(irreps, irreps, irreps)

    a = rand_dict(batch, irreps)
    b = rand_dict(batch, irreps)
    alpha = torch.rand(1).item() * 2 * math.pi
    a_rot = _rotate(a, alpha)
    b_rot = _rotate(b, alpha)

    z = tp(a, b)
    z_rot = tp(a_rot, b_rot)

    for m in irreps:
        phase = torch.exp(1j * torch.tensor(m * alpha))
        assert not torch.allclose(z_rot[m], z[m] * phase, atol=1e-6, rtol=1e-6)
