# test_harmonic_invariant.py
import math
import torch
import pytest

from approx_so2_equiv_nn import ApproxHarmonicInvariantMLP
from approx_so2_linear import RelaxSO2Linear
from approx_so2_tensor_product import RelaxSO2TensorProduct


# --------------------------------------------------------------------------- #
# 1. forward pass: shape and dtype
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("batch,M,C,hid", [(7, 2, 3, 4), (3, 4, 2, 5)])
def test_forward_shapes(batch, M, C, hid):
    model = ApproxHarmonicInvariantMLP(M=M, C=C, hidden_c=hid)

    x = torch.randn(batch, 2)
    out = model(x)  # (B,1)

    assert out.shape == (batch, 1)
    assert out.dtype == torch.cfloat or out.dtype == torch.float32


# --------------------------------------------------------------------------- #
# 3. gradients flow through penalty + data loss
# --------------------------------------------------------------------------- #
def test_gradients_through_penalty():
    model = ApproxHarmonicInvariantMLP(M=1, C=2, hidden_c=3)

    x = torch.randn(4, 2, requires_grad=False)
    logits = model(x).real  # use real part for BCE

    target = torch.randint(0, 2, (4, 1)).float()
    data_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)

    pen = model.compute_non_equivariance_penalty()
    loss = data_loss + pen["nonequiv_part"] * 0.1 + pen["equivariant_part"] * 0.01
    loss.backward()

    for name, p in model.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
