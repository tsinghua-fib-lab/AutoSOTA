# test_harmonic_invariant.py
import math
import torch
import pytest

from so2_equiv_nn import HarmonicInvariantMLP


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def rotate(points: torch.Tensor, alpha: float) -> torch.Tensor:
    """Rotate a batch of 2-D points counter-clockwise by α (radians)."""
    R = torch.tensor(
        [[math.cos(alpha), -math.sin(alpha)], [math.sin(alpha), math.cos(alpha)]],
        dtype=points.dtype,
        device=points.device,
    )
    return points @ R.T


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("batch", [4, 16])
def test_so2_invariance(batch, tol=1e-5, device="cpu"):
    """f(R·x) == f(x)  for a random rotation R ∈ SO(2)."""
    net = HarmonicInvariantMLP().to(device).eval()

    # random input batch
    x = torch.randn(batch, 2, device=device)
    out = net(x)  # (B,1)

    # random rotation
    alpha = torch.rand(1).item() * 2 * math.pi
    x_rot = rotate(x, alpha)
    out_rot = net(x_rot)

    assert torch.allclose(out, out_rot, atol=tol, rtol=tol)


def test_backward_pass(device="cpu"):
    """Gradients flow through the whole model."""
    net = HarmonicInvariantMLP().to(device).train()

    x = torch.randn(8, 2, device=device, requires_grad=True)
    out = net(x)  # (8,1)
    loss = out.mean()
    loss.backward()

    # every parameter must receive a gradient
    for name, p in net.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
