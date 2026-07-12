# test_so2_linear.py
import math, torch, pytest
from so2_linear import SO2Linear  # adjust if your file/module is named differently


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def rotation_phase_tensor(alpha, *, device="cpu"):
    """Returns a 0-D complex tensor e^{i α}"""
    return torch.cos(torch.tensor(alpha, device=device)) + 1j * torch.sin(
        torch.tensor(alpha, device=device)
    )


def make_phases(m_vals, alpha, *, device="cpu"):
    eia = rotation_phase_tensor(alpha, device=device)
    return {m: eia**m for m in m_vals}


def apply_phases(x_dict, phases):
    return {m: phases[m] * t for m, t in x_dict.items()}


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "batch,Cin,Cout,m_vals", [(8, 3, 5, (-2, 0, 3)), (4, 2, 2, (-1, 1))]
)
@pytest.mark.parametrize("use_bias", [False, True])
def test_equivariance(batch, Cin, Cout, m_vals, use_bias, tol=1e-5, device="cpu"):
    in_irreps = {m: Cin for m in m_vals}
    out_irreps = {m: Cout for m in m_vals}

    layer = SO2Linear(in_irreps, out_irreps, bias=use_bias).to(device)

    # random COMPLEX input (dtype = cfloat)
    x = {m: torch.randn(batch, Cin, dtype=torch.cfloat, device=device) for m in m_vals}

    # random rotation
    alpha = torch.rand(1).item() * 2 * math.pi
    phases = make_phases(m_vals, alpha, device=device)
    x_rot = apply_phases(x, phases)  # ρ(α)·x

    # forward
    y = layer(x)
    y_rot = layer(x_rot)
    y_ref = apply_phases(y, phases)  # ρ(α)·T(x)

    for m in m_vals:
        assert torch.allclose(y_rot[m], y_ref[m], atol=tol, rtol=tol)


def test_shape_and_bias():
    layer = SO2Linear({0: 2, 1: 3}, {0: 4, 1: 1}, bias=True)
    x = {
        0: torch.zeros(1, 2, dtype=torch.cfloat),
        1: torch.zeros(1, 3, dtype=torch.cfloat),
    }
    y = layer(x)
    assert y[0].shape == (1, 4)
    assert y[1].shape == (1, 1)

    # zero weights → output should equal bias
    for p in layer.weight.values():
        p.data.zero_()
    y_bias = layer(x)
    for m in y_bias:
        expected = layer.bias[str(m)][None, :].to(y_bias[m].dtype)
        assert torch.allclose(y_bias[m], expected)


def test_mismatch_keys_raises():
    with pytest.raises(ValueError):
        SO2Linear({0: 2, 1: 1}, {0: 2})  # m=1 missing on output
