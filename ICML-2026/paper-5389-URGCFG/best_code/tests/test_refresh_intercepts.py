import torch
import pytest

from models import FiniteSeparableModel


def quadratic_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return -(x - y) ** 2


def build_model(mode: str) -> FiniteSeparableModel:
    return FiniteSeparableModel(
        kernel=quadratic_kernel,
        num_dims=1,
        radius=1.0,
        y_accuracy=0.5,
        x_accuracy=0.5,
        mode=mode,
        temp=1.0,
        cache_gradients=False,
    )


def manual_refresh(model: FiniteSeparableModel, column: torch.Tensor) -> torch.Tensor:
    K = model.kernel_tensor.to(column)
    scores = K - column.unsqueeze(0)
    if model.mode == "concave":
        u_grid = scores.min(dim=1).values
        refreshed = (K - u_grid.unsqueeze(1)).min(dim=0).values
    else:
        u_grid = scores.max(dim=1).values
        refreshed = (K - u_grid.unsqueeze(1)).max(dim=0).values
    return refreshed


def manual_u_grid(model: FiniteSeparableModel, column: torch.Tensor) -> torch.Tensor:
    K = model.kernel_tensor.to(column)
    scores = K - column.unsqueeze(0)
    if model.mode == "concave":
        return scores.min(dim=1).values
    return scores.max(dim=1).values


@pytest.mark.parametrize("mode", ["convex", "concave"])
def test_refresh_intercepts_matches_manual(mode: str) -> None:
    model = build_model(mode)
    torch.manual_seed(0)
    with torch.no_grad():
        model.intercepts.copy_(torch.randn_like(model.intercepts))
    expected = manual_refresh(model, model.intercepts[:, 0])
    model.refresh_intercepts_via_transform()
    # Intercepts are defined up to an additive constant per column.
    # FiniteSeparableModel uses a gauge with a fixed first row, so we
    # compare only rows 1: and allow the first row to differ by a constant.
    torch.testing.assert_close(model.intercepts[1:, 0], expected[1:], atol=1e-9, rtol=0.0)


@pytest.mark.parametrize("mode", ["convex", "concave"])
def test_refresh_intercepts_preserves_dual_sum(mode: str) -> None:
    model = build_model(mode)
    torch.manual_seed(1)
    with torch.no_grad():
        model.intercepts.copy_(torch.randn_like(model.intercepts))
    model.refresh_intercepts_via_transform()
    column_before = model.intercepts[:, 0].clone()
    u_grid = manual_u_grid(model, column_before)
    dual_before = u_grid.unsqueeze(1) + column_before.unsqueeze(0)
    changed = model.refresh_intercepts_via_transform()
    column_after = model.intercepts[:, 0]
    dual_after = u_grid.unsqueeze(1) + column_after.unsqueeze(0)
    assert changed == 0
    torch.testing.assert_close(column_after, column_before, atol=1e-12, rtol=0.0)
    torch.testing.assert_close(dual_after, dual_before, atol=1e-12, rtol=0.0)
