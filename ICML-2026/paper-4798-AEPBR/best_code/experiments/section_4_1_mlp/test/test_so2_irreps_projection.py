# test_project_to_irreps_simple.py
import math
import torch
import pytest

from so2_irreps_projection import project_to_irreps_simple, project_to_irreps_radial

# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch,M", [(10, 4), (3, 1)])
def test_so2_equivariance_simple(batch, M, tol=1e-5, device="cpu"):
    """
    Check that for every weight m = -M…M
        y(R·x) = e^{i m α} · y(x)
    where R is a random in-plane rotation by α.
    """
    # random input points
    x = torch.randn(batch, 2, device=device)

    # random rotation angle α  and its matrix  R
    alpha = torch.rand(1).item() * 2 * math.pi
    R = torch.tensor(
        [[math.cos(alpha), -math.sin(alpha)], [math.sin(alpha), math.cos(alpha)]],
        device=device,
    )
    x_rot = x @ R.T

    # project to irreps
    y = project_to_irreps_simple(x, M)  # (B, 2M+1) complex
    y_rot = project_to_irreps_simple(x_rot, M)

    # phase factors e^{i m α}  for m = -M … M
    m_vals = torch.arange(-M, M + 1, device=device, dtype=torch.float32)
    phase = torch.exp(1j * m_vals * alpha)[None, :]  # (1, 2M+1)

    # equivariance assertion
    assert torch.allclose(y_rot, y * phase, atol=tol, rtol=tol)


def test_zero_vector_no_nan_simple():
    """A zero input vector should not create NaNs (eps guard works)."""
    x = torch.zeros(1, 2)
    y = project_to_irreps_simple(x, M=2)
    assert not torch.isnan(y).any()


# --------------------------------------------------------------------------- #
# 1.  Equivariance  y(R·x) = phase(m) · y(x)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("batch,M,C", [(12, 4, 5), (3, 2, 3)])
def test_so2_equivariance(batch, M, C, tol=1e-5, device="cpu"):
    x = torch.randn(batch, 2, device=device)

    # random rotation α
    alpha = torch.rand(1).item() * 2 * math.pi
    R = torch.tensor(
        [[math.cos(alpha), -math.sin(alpha)], [math.sin(alpha), math.cos(alpha)]],
        device=device,
    )
    x_rot = x @ R.T

    y = project_to_irreps_radial(x, M, C)  # (B, 2M+1, C)
    y_rot = project_to_irreps_radial(x_rot, M, C)

    m_vals = torch.arange(-M, M + 1, device=device)  # (2M+1,)
    phase = torch.exp(1j * m_vals * alpha)  # (2M+1,)
    phase = phase[None, :, None]  # (1,2M+1,1) broadcast

    assert torch.allclose(y_rot, y * phase, atol=tol, rtol=tol)


# --------------------------------------------------------------------------- #
# 2.  Radius is preserved (same angle, different radii → different features)
# --------------------------------------------------------------------------- #
def test_radius_sensitivity(device="cpu"):
    theta = 0.876
    v1 = torch.tensor([[math.cos(theta), math.sin(theta)]], device=device) * 1.0
    v2 = torch.tensor([[math.cos(theta), math.sin(theta)]], device=device) * 2.5

    y1 = project_to_irreps_radial(v1, M=2, C=4)  # (1, 5, 4)
    y2 = project_to_irreps_radial(v2, M=2, C=4)

    m0_index = 2  # since m=-2…2, m=0 lives at index 2
    assert not torch.allclose(y1[:, m0_index], y2[:, m0_index]), "radius info lost"


# --------------------------------------------------------------------------- #
# 3.  Zero vector produces no NaNs
# --------------------------------------------------------------------------- #
def test_zero_vector_no_nan():
    z = torch.zeros(1, 2)
    yz = project_to_irreps_radial(z, M=3, C=4)
    assert not torch.isnan(yz).any()
