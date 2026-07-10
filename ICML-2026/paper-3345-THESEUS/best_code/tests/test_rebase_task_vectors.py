from __future__ import annotations

import torch

from merge_and_rebase.rebase.registry import get_method, list_methods
from merge_and_rebase.rebase.task_vectors import (
    merge_task_vectors,
    rebase_merged_task_vectors,
    transport_task_vector,
)


def test_rebase_methods_registered() -> None:
    methods = list_methods()
    assert "identity" in methods
    assert "orthogonal_shift" in methods
    assert get_method("identity").name == "identity"


def test_identity_transport_keeps_delta() -> None:
    source_base = {"w": torch.tensor([0.0, 0.0])}
    target_base = {"w": torch.tensor([10.0, -3.0])}
    tuned = [{"w": torch.tensor([1.0, 2.0])}]

    merged = merge_task_vectors(base=source_base, tuned=tuned, strict=True)
    transported = transport_task_vector(
        source_base=source_base,
        target_base=target_base,
        task_vector=merged,
        method="identity",
        strict=True,
    )

    assert torch.allclose(transported.delta["w"], torch.tensor([1.0, 2.0]))


def test_orthogonal_shift_removes_shift_component() -> None:
    source_base = {"w": torch.tensor([0.0, 0.0])}
    target_base = {"w": torch.tensor([2.0, 0.0])}  # shift along x
    tuned = [{"w": torch.tensor([3.0, 4.0])}]  # delta [3,4]

    merged = merge_task_vectors(base=source_base, tuned=tuned, strict=True)
    transported = transport_task_vector(
        source_base=source_base,
        target_base=target_base,
        task_vector=merged,
        method="orthogonal_shift",
        strict=True,
    )

    # remove x component -> [0, 4]
    assert torch.allclose(transported.delta["w"], torch.tensor([0.0, 4.0]), atol=1e-6)


def test_rebase_merged_task_vectors_end_to_end() -> None:
    source_base = {"w": torch.tensor([0.0, 0.0])}
    target_base = {"w": torch.tensor([10.0, 10.0])}
    tuned = [
        {"w": torch.tensor([1.0, 2.0])},
        {"w": torch.tensor([3.0, 6.0])},
    ]

    # merged delta = 0.5*[1,2] + 0.25*[3,6] = [1.25, 2.5]
    rebased = rebase_merged_task_vectors(
        source_base=source_base,
        target_base=target_base,
        tuned=tuned,
        weights=[0.5, 0.25],
        alpha=2.0,
        transport_method="identity",
        strict=True,
    )

    # target + 2 * merged_delta = [12.5, 15.0]
    assert torch.allclose(rebased["w"], torch.tensor([12.5, 15.0]))
