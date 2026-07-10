from __future__ import annotations

import pytest
import torch

from merge_and_rebase.merge.methods.actmerge import ActMerge
from merge_and_rebase.merge.methods.functional import merge_functional
from merge_and_rebase.merge.registry import get_method


def test_actmerge_registered_with_alias() -> None:
    assert isinstance(get_method("actmerge"), ActMerge)
    assert isinstance(get_method("actmat"), ActMerge)


def test_actmerge_alpha_zero_is_base() -> None:
    base = {
        "linear.weight": torch.eye(2, dtype=torch.float32),
        "linear.bias": torch.tensor([0.25, -0.5], dtype=torch.float32),
    }
    tuned = [
        {
            "linear.weight": base["linear.weight"] + torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float32),
            "linear.bias": base["linear.bias"] + torch.tensor([1.0, -1.0], dtype=torch.float32),
        },
        {
            "linear.weight": base["linear.weight"] + torch.tensor([[0.0, 0.0], [0.0, 2.0]], dtype=torch.float32),
            "linear.bias": base["linear.bias"] + torch.tensor([-1.0, 2.0], dtype=torch.float32),
        },
    ]

    merged = ActMerge().merge(base=base, tuned=tuned, alpha=0.0, strict=True)

    assert torch.allclose(merged["linear.weight"], base["linear.weight"])
    assert torch.allclose(merged["linear.bias"], base["linear.bias"])


def test_actmerge_prepare_matches_closed_form_for_linear_weights() -> None:
    base = {
        "linear.weight": torch.zeros((2, 2), dtype=torch.float32),
        "linear.bias": torch.zeros((2,), dtype=torch.float32),
    }
    tuned = [
        {
            "linear.weight": torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float32),
            "linear.bias": torch.tensor([1.0, -1.0], dtype=torch.float32),
        },
        {
            "linear.weight": torch.tensor([[0.0, 0.0], [0.0, 2.0]], dtype=torch.float32),
            "linear.bias": torch.tensor([3.0, 1.0], dtype=torch.float32),
        },
    ]

    _, direction = ActMerge().prepare(base=base, tuned=tuned, strict=True, weights=[2.0, 1.0])

    expected_weight = torch.tensor([[1.0, 0.0], [0.0, 2.0]], dtype=torch.float32)
    expected_bias = torch.tensor([5.0 / 3.0, -1.0 / 3.0], dtype=torch.float32)

    assert torch.allclose(direction["linear.weight"], expected_weight)
    assert torch.allclose(direction["linear.bias"], expected_bias)


def test_actmerge_averages_non_linear_2d_parameters_by_default() -> None:
    base = {
        "linear.weight": torch.zeros((2, 2), dtype=torch.float32),
        "positional_embedding": torch.zeros((2, 2), dtype=torch.float32),
    }
    tuned = [
        {
            "linear.weight": torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float32),
            "positional_embedding": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        },
        {
            "linear.weight": torch.tensor([[0.0, 0.0], [0.0, 2.0]], dtype=torch.float32),
            "positional_embedding": torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32),
        },
    ]

    _, direction = ActMerge().prepare(base=base, tuned=tuned, strict=True)

    expected = torch.tensor([[3.0, 4.0], [5.0, 6.0]], dtype=torch.float32)
    assert torch.allclose(direction["positional_embedding"], expected)


def test_actmerge_functional_accepts_absolute_form() -> None:
    matrices = [
        torch.tensor([[2.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
        torch.tensor([[1.0, 0.0], [0.0, 3.0]], dtype=torch.float32),
    ]

    merged = merge_functional("actmerge", matrices=matrices, method_params={"form": "absolute"})

    assert merged.shape == matrices[0].shape
    assert torch.isfinite(merged).all()


def test_actmerge_rejects_negative_weights() -> None:
    matrices = [
        torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float32),
        torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
    ]

    with pytest.raises(ValueError, match="non-negative weights"):
        merge_functional("actmerge", matrices=matrices, weights=[1.0, -1.0])
