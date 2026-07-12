# test_relax_so2_linear.py
import torch
import pytest
import math

from approx_so2_linear import RelaxSO2Linear


# -------------------------------------------------------------
def _random_feature_dict(batch, irreps, dtype=torch.cfloat):
    """helper to build {m : (B, C_m)} complex tensors"""
    return {m: torch.randn(batch, c, dtype=dtype) for m, c in irreps.items()}


def _rand_dict(batch, irreps):
    return {m: torch.randn(batch, c, dtype=torch.cfloat) for m, c in irreps.items()}


def _rotate(feat, alpha):
    return {
        m: v
        * (torch.cos(torch.tensor(alpha * m)) + 1j * torch.sin(torch.tensor(alpha * m)))
        for m, v in feat.items()
    }


# -------------------------------------------------------------
@pytest.mark.parametrize("batch", [2, 5])
def test_shapes_forward(batch):
    in_ir = {-2: 3, -1: 2, 0: 4, 1: 2, 2: 3}
    out_ir = {m: c + 1 for m, c in in_ir.items()}  # arbitrary

    layer = RelaxSO2Linear(in_ir, out_ir, bias=True)

    x = _random_feature_dict(batch, in_ir)
    y = layer(x)

    for m in out_ir:
        assert y[m].shape == (batch, out_ir[m])


# -------------------------------------------------------------
def test_projection_correctness():
    in_ir = {-1: 2, 0: 3, 1: 2}
    out_ir = in_ir
    layer = RelaxSO2Linear(in_ir, out_ir, bias=False)

    W = layer.weight.detach()
    W_eq = layer.equivariant_projection().detach()

    outside_mask = torch.ones_like(W_eq, dtype=torch.bool)
    for m in in_ir:
        r, c = layer.out_slices[m], layer.in_slices[m]
        assert torch.allclose(W_eq[r, c], W[r, c])  # inside slice check
        outside_mask[r, c] = False  # mark as inside

    assert torch.allclose(W_eq[outside_mask], torch.zeros_like(W_eq[outside_mask]))


# -------------------------------------------------------------
def test_grad_through_penalty():
    in_ir = {0: 2, 1: 2}
    layer = RelaxSO2Linear(in_ir, in_ir, bias=True)

    # dummy loss = sum of both penalties
    pen_eq, pen_ne = layer.penalty_terms()
    loss = pen_eq + pen_ne
    loss.backward()

    assert layer.weight.grad is not None, "no gradient through penalty"


# -------------------------------------------------------------
@pytest.mark.parametrize("batch", [4])
def test_projected_linear_equivariant(batch):
    in_ir = {-2: 2, -1: 2, 0: 2, 1: 2, 2: 2}
    out_ir = {m: c for m, c in in_ir.items()}
    lin = RelaxSO2Linear(in_ir, out_ir, bias=False)

    # keep only equivariant blocks
    with torch.no_grad():
        lin.weight.mul_(lin.mask_eq)

    x = _rand_dict(batch, in_ir)
    alpha = torch.rand(1).item() * 2 * math.pi
    x_rot = _rotate(x, alpha)

    y = lin(x)
    y_rot = lin(x_rot)

    for m in out_ir:
        assert torch.allclose(y_rot[m], _rotate(y, alpha)[m], atol=1e-6, rtol=1e-6)


# -------------------------------------------------------------
@pytest.mark.parametrize("batch", [4])
def test_projected_linear_equivariant_not(batch):
    in_ir = {-2: 2, -1: 2, 0: 2, 1: 2, 2: 2}
    out_ir = {m: c for m, c in in_ir.items()}
    lin = RelaxSO2Linear(in_ir, out_ir, bias=False)

    x = _rand_dict(batch, in_ir)
    alpha = torch.rand(1).item() * 2 * math.pi
    x_rot = _rotate(x, alpha)

    y = lin(x)
    y_rot = lin(x_rot)

    for m in out_ir:
        assert not torch.allclose(y_rot[m], _rotate(y, alpha)[m], atol=1e-6, rtol=1e-6)
