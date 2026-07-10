from __future__ import annotations

import pytest
import torch

from merge_and_rebase.merge.methods.functional import merge_functional
from merge_and_rebase.merge.methods.wudi_merge import WUDIMerge
from merge_and_rebase.merge.registry import get_method
from merge_and_rebase.merge.task_vectors import TaskVector


def _toy_checkpoints() -> tuple[dict[str, torch.Tensor], list[dict[str, torch.Tensor]]]:
    base = {
        "w": torch.zeros((2, 2), dtype=torch.float32),
        "b": torch.zeros((2,), dtype=torch.float32),
    }
    tuned = [
        {
            "w": torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
            "b": torch.tensor([1.0, -1.0], dtype=torch.float32),
        },
        {
            "w": torch.tensor([[0.0, 2.0], [2.0, 0.0]], dtype=torch.float32),
            "b": torch.tensor([3.0, 1.0], dtype=torch.float32),
        },
    ]
    return base, tuned


def test_wudi_registered() -> None:
    assert isinstance(get_method("wudi"), WUDIMerge)
    assert isinstance(get_method("wudi_merge"), WUDIMerge)


def test_wudi_alpha_zero_is_base() -> None:
    base, tuned = _toy_checkpoints()
    method = WUDIMerge()

    merged = method.merge(base=base, tuned=tuned, alpha=0.0, strict=True)

    assert torch.allclose(merged["w"], base["w"])
    assert torch.allclose(merged["b"], base["b"])


def test_wudi_single_task_recovers_task_vector() -> None:
    base = {"w": torch.zeros((2, 2), dtype=torch.float32)}
    tuned = [{"w": torch.tensor([[1.5, -0.5], [0.25, 2.0]], dtype=torch.float32)}]
    method = WUDIMerge()

    prepared_base, direction = method.prepare(base=base, tuned=tuned, strict=True)
    delta = TaskVector.from_checkpoints(base, tuned[0], strict=True).delta

    assert prepared_base is base
    assert torch.allclose(direction["w"], delta["w"], atol=1e-5, rtol=1e-5)


def test_wudi_single_task_gd_recovers_task_vector() -> None:
    base = {"w": torch.zeros((2, 2), dtype=torch.float32)}
    tuned = [{"w": torch.tensor([[1.5, -0.5], [0.25, 2.0]], dtype=torch.float32)}]
    method = WUDIMerge()

    _, direction = method.prepare(
        base=base,
        tuned=tuned,
        strict=True,
        method_params={"solver": "gd", "steps": 5, "lr": 1e-4},
    )
    delta = TaskVector.from_checkpoints(base, tuned[0], strict=True).delta

    assert torch.allclose(direction["w"], delta["w"], atol=1e-5, rtol=1e-5)


def test_wudi_functional_gd_matches_closed_form_on_toy_matrix() -> None:
    matrices = [
        torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
        torch.tensor([[0.0, 2.0], [2.0, 0.0]], dtype=torch.float32),
    ]

    closed = merge_functional("wudi", matrices=matrices, method_params={"solver": "closed_form"})
    gd = merge_functional("wudi", matrices=matrices, method_params={"solver": "gd", "steps": 400, "lr": 0.05})

    assert torch.allclose(gd, closed, atol=5e-3, rtol=5e-3)


def test_wudi_1d_average_fallback_matches_mean_delta() -> None:
    base, tuned = _toy_checkpoints()
    method = WUDIMerge()

    _, direction = method.prepare(base=base, tuned=tuned, strict=True, method_params={"vector_1d_merge": "average"})

    expected = torch.stack([t["b"] - base["b"] for t in tuned], dim=0).mean(dim=0)
    assert torch.allclose(direction["b"], expected)


def test_wudi_1d_zero_fallback_skips_bias_delta() -> None:
    base, tuned = _toy_checkpoints()
    method = WUDIMerge()

    _, direction = method.prepare(base=base, tuned=tuned, strict=True, method_params={"vector_1d_merge": "zero"})

    assert torch.equal(direction["b"], torch.zeros_like(base["b"]))


def test_wudi_invalid_solver_raises() -> None:
    matrices = [torch.eye(2), torch.ones((2, 2))]

    try:
        merge_functional("wudi", matrices=matrices, method_params={"solver": "nope"})
    except ValueError as exc:
        assert "solver" in str(exc)
    else:
        raise AssertionError("Expected invalid wudi solver to raise ValueError")


@pytest.mark.parametrize("solver", ["closed_form", "gd"])
def test_wudi_functional_preserves_input_device(solver: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matrices = [
        torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32, device=device),
        torch.tensor([[0.0, 2.0], [2.0, 0.0]], dtype=torch.float32, device=device),
    ]

    params = {"solver": solver}
    if solver == "gd":
        params.update({"steps": 10, "lr": 1e-5})

    merged = merge_functional("wudi", matrices=matrices, method_params=params)

    assert merged.device == matrices[0].device
