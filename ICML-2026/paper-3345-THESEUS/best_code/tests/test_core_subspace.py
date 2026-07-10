from __future__ import annotations

import pytest
import torch

from merge_and_rebase.merge.subspaces.core_space import CoreSpace


def _peft_state_for_layer(prefix: str, a: torch.Tensor, b: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        f"{prefix}.lora_A.weight": a,
        f"{prefix}.lora_B.weight": b,
    }


@pytest.mark.parametrize(
    ("method_params", "expected_method"),
    [
        ({}, "qr"),
        ({"basis_method": "qr"}, "qr"),
        ({"core_basis_method": "qr"}, "qr"),
    ],
)
def test_core_prepare_selects_basis_method(
    method_params: dict[str, str],
    expected_method: str,
) -> None:
    layer = "visual.transformer.resblocks.0.attn.q_proj"
    lora_by_task = {
        "t1": _peft_state_for_layer(layer, torch.randn(2, 4), torch.randn(6, 2)),
        "t2": _peft_state_for_layer(layer, torch.randn(2, 4), torch.randn(6, 2)),
    }

    prepared = CoreSpace().prepare(
        lora_by_task=lora_by_task,
        peft_cfg={"lora_alpha": 4},
        method_params=method_params,
    )

    assert prepared.basis_method == expected_method


@pytest.mark.parametrize("basis_method", ["svd", "qr"])
def test_core_project_and_lift_reconstruct_full_update(basis_method: str) -> None:
    torch.manual_seed(0)
    layer = "visual.transformer.resblocks.0.attn.v_proj"
    a1 = torch.randn(2, 5)
    b1 = torch.randn(7, 2)
    a2 = torch.randn(2, 5)
    b2 = torch.randn(7, 2)

    lora_by_task = {
        "t1": _peft_state_for_layer(layer, a1, b1),
        "t2": _peft_state_for_layer(layer, a2, b2),
    }

    core = CoreSpace()
    prepared = core.prepare(
        lora_by_task=lora_by_task,
        peft_cfg={"lora_alpha": 8},
        method_params={"basis_method": basis_method},
    )
    projected = core.project(prepared, lora_by_task=lora_by_task, peft_cfg={"lora_alpha": 8})
    lifted = core.lift(
        prepared,
        merged_core={layer: projected["t1"][layer]},
        lora_template=lora_by_task["t1"],
        peft_cfg={"lora_alpha": 8},
    )

    assert torch.allclose(lifted[f"{layer}.weight"], b1 @ a1, atol=1e-5, rtol=1e-5)


def test_core_prepare_rejects_unknown_basis_method() -> None:
    layer = "visual.transformer.resblocks.0.attn.k_proj"
    lora_by_task = {
        "t1": _peft_state_for_layer(layer, torch.randn(2, 3), torch.randn(4, 2)),
    }

    with pytest.raises(ValueError, match="basis_method"):
        CoreSpace().prepare(
            lora_by_task=lora_by_task,
            peft_cfg={"lora_alpha": 4},
            method_params={"basis_method": "fastest"},
        )
