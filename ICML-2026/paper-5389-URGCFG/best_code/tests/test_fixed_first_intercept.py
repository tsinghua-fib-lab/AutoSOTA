import torch
from models.helpers import FixedFirstIntercept


def test_fixed_first_intercept_shapes_and_gauge():
    ny, dim = 5, 3
    ffi = FixedFirstIntercept(ny, dim, init_std=0.0)
    b = ffi.value
    assert b.shape == (ny, dim)
    # First row is always zero
    assert torch.allclose(b[0], torch.zeros(dim, dtype=b.dtype, device=b.device))


def test_fixed_first_intercept_project_from_b_and_value():
    ny, dim = 4, 2
    ffi = FixedFirstIntercept(ny, dim)
    b_raw = torch.randn(ny, dim)

    ffi.project_from_b(b_raw)
    b = ffi.value

    # Gauge: first row is zero, remaining rows match raw[1:]
    assert torch.allclose(b[0], torch.zeros(dim, dtype=b.dtype, device=b.device))
    assert torch.allclose(b[1:], b_raw[1:])


def test_fixed_first_intercept_set_column_from_raw():
    ny, dim = 6, 3
    ffi = FixedFirstIntercept(ny, dim, init_std=0.0)

    raw_col = torch.linspace(-1.0, 1.0, ny)
    ffi.set_column_from_raw_(1, raw_col)
    b = ffi.value

    # Column 1 should have gauge-fixed first entry 0 and the rest from raw_col[1:]
    assert b[0, 1].item() == 0.0
    assert torch.allclose(b[1:, 1], raw_col[1:])


def test_fixed_first_intercept_gradients_flow():
    ny, dim = 5, 2
    ffi = FixedFirstIntercept(ny, dim)
    b = ffi.value
    loss = (b ** 2).sum()
    loss.backward()
    # Gradients should propagate to theta
    assert ffi.theta.grad is not None
    assert torch.isfinite(ffi.theta.grad).all()

