from __future__ import annotations

import pytest
import torch

from merge_and_rebase.merge.methods.dc_merge import DCMerge
from merge_and_rebase.merge.methods.functional import merge_functional
from merge_and_rebase.merge.registry import get_method
from merge_and_rebase.merge.task_vectors import TaskVector


def _toy_checkpoints() -> tuple[dict[str, torch.Tensor], list[dict[str, torch.Tensor]]]:
    base = {
        "w": torch.zeros((3, 3), dtype=torch.float32),
        "b": torch.zeros((3,), dtype=torch.float32),
    }
    tuned = [
        {
            "w": torch.tensor(
                [
                    [1.0, 0.2, -0.1],
                    [0.0, 0.7, 0.3],
                    [0.4, -0.2, 0.5],
                ],
                dtype=torch.float32,
            ),
            "b": torch.tensor([0.5, -0.25, 0.75], dtype=torch.float32),
        },
        {
            "w": torch.tensor(
                [
                    [0.6, -0.4, 0.1],
                    [0.5, 0.1, -0.3],
                    [-0.2, 0.3, 0.8],
                ],
                dtype=torch.float32,
            ),
            "b": torch.tensor([-0.5, 0.75, 0.25], dtype=torch.float32),
        },
    ]
    return base, tuned


def test_dc_merge_registered() -> None:
    assert isinstance(get_method("dc_merge"), DCMerge)


def test_dc_merge_alpha_zero_is_base() -> None:
    base, tuned = _toy_checkpoints()

    merged = DCMerge().merge(base=base, tuned=tuned, alpha=0.0, strict=True)

    assert torch.allclose(merged["w"], base["w"])
    assert torch.allclose(merged["b"], base["b"])


@pytest.mark.parametrize("mode", ["none", "average", "linear"])
def test_dc_merge_functional_supports_energy_smoothing_modes(mode: str) -> None:
    matrices = [checkpoint["w"] for checkpoint in _toy_checkpoints()[1]]

    merged = merge_functional(
        "dc_merge",
        matrices=matrices,
        method_params={
            "energy_smoothing": mode,
            "sv_reduction": 1.0,
            "cover_merge_method": "task_arithmetic",
            "mask_mode": "block",
            "svd_dtype": "fp32",
        },
    )

    assert merged.shape == matrices[0].shape
    assert torch.isfinite(merged).all()


def test_dc_merge_prepare_uses_mean_for_1d_average_fallback() -> None:
    base, tuned = _toy_checkpoints()

    _, direction = DCMerge().prepare(
        base=base,
        tuned=tuned,
        strict=True,
        method_params={"vector_1d_merge": "average"},
    )

    expected = torch.stack([checkpoint["b"] - base["b"] for checkpoint in tuned], dim=0).mean(dim=0)
    assert torch.allclose(direction["b"], expected)


def test_dc_merge_single_task_recovers_task_vector_without_smoothing() -> None:
    base = {"w": torch.zeros((2, 2), dtype=torch.float32)}
    tuned = [{"w": torch.tensor([[1.25, -0.5], [0.25, 2.0]], dtype=torch.float32)}]

    prepared_base, direction = DCMerge().prepare(
        base=base,
        tuned=tuned,
        strict=True,
        method_params={
            "energy_smoothing": "none",
            "sv_reduction": 1.0,
            "mask_mode": "none",
            "cover_merge_method": "task_arithmetic",
        },
    )

    delta = TaskVector.from_checkpoints(base, tuned[0], strict=True).delta
    assert prepared_base is base
    assert torch.allclose(direction["w"], delta["w"], atol=1e-5, rtol=1e-5)


def test_dc_merge_rejects_invalid_energy_smoothing_mode() -> None:
    matrices = [checkpoint["w"] for checkpoint in _toy_checkpoints()[1]]

    with pytest.raises(ValueError, match="energy_smoothing"):
        merge_functional(
            "dc_merge",
            matrices=matrices,
            method_params={"energy_smoothing": "mystery"},
        )
