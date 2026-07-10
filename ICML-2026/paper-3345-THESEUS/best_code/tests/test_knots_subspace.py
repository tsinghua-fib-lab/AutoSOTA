from __future__ import annotations

import pytest
import torch

from merge_and_rebase.merge.methods.task_arithmetic import TaskArithmeticMerge
from merge_and_rebase.merge.subspaces.knots_space import KnotsSpace


def _peft_state_for_layer(prefix: str, a: torch.Tensor, b: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        f"{prefix}.lora_A.weight": a,
        f"{prefix}.lora_B.weight": b,
    }


def test_knots_project_and_lift_reconstructs_full_update() -> None:
    torch.manual_seed(0)
    layer = "visual.transformer.resblocks.0.attn.q_proj"
    peft_cfg = {"lora_alpha": 4}

    a1 = torch.randn(2, 4)
    b1 = torch.randn(6, 2)
    a2 = torch.randn(2, 4)
    b2 = torch.randn(6, 2)

    lora_by_task = {
        "t1": _peft_state_for_layer(layer, a1, b1),
        "t2": _peft_state_for_layer(layer, a2, b2),
    }

    knots = KnotsSpace()
    prepared = knots.prepare(lora_by_task=lora_by_task, peft_cfg=peft_cfg)
    projected = knots.project(prepared, lora_by_task=lora_by_task, peft_cfg=peft_cfg)

    lifted = knots.lift(
        prepared,
        merged_core={layer: projected["t1"][layer]},
        lora_template=lora_by_task["t1"],
        peft_cfg=peft_cfg,
    )
    scale = float(peft_cfg["lora_alpha"]) / float(a1.shape[0])
    expected = scale * (b1 @ a1)
    assert torch.allclose(lifted[f"{layer}.weight"], expected, atol=1e-5, rtol=1e-5)


def test_knots_merges_on_v_and_reconstructs_with_u_and_s() -> None:
    torch.manual_seed(1)
    layer = "visual.transformer.resblocks.0.attn.v_proj"
    peft_cfg = {"lora_alpha": 8}

    a1 = torch.randn(2, 5)
    b1 = torch.randn(7, 2)
    a2 = torch.randn(2, 5)
    b2 = torch.randn(7, 2)

    lora_by_task = {
        "t1": _peft_state_for_layer(layer, a1, b1),
        "t2": _peft_state_for_layer(layer, a2, b2),
    }

    knots = KnotsSpace()
    prepared = knots.prepare(lora_by_task=lora_by_task, peft_cfg=peft_cfg)
    projected = knots.project(prepared, lora_by_task=lora_by_task, peft_cfg=peft_cfg)

    method = TaskArithmeticMerge()
    base = {layer: torch.zeros_like(projected["t1"][layer])}
    merged_v = method.merge(
        base=base,
        tuned=[projected["t1"], projected["t2"]],
        weights=[0.25, 0.75],
        alpha=1.0,
        strict=True,
    )

    expected_v = 0.25 * projected["t1"][layer] + 0.75 * projected["t2"][layer]
    assert torch.allclose(merged_v[layer], expected_v, atol=1e-6, rtol=1e-6)

    lifted = knots.lift(
        prepared,
        merged_core=merged_v,
        lora_template=lora_by_task["t1"],
        peft_cfg=peft_cfg,
    )
    basis = prepared.layers[layer]
    expected_delta = basis.U @ (basis.S[:, None] * expected_v)
    assert torch.allclose(lifted[f"{layer}.weight"], expected_delta, atol=1e-5, rtol=1e-5)


def test_knots_project_rejects_unknown_task() -> None:
    layer = "visual.transformer.resblocks.0.attn.k_proj"
    peft_cfg = {"lora_alpha": 4}
    a = torch.randn(2, 3)
    b = torch.randn(4, 2)

    lora_by_task = {"t1": _peft_state_for_layer(layer, a, b)}
    knots = KnotsSpace()
    prepared = knots.prepare(lora_by_task=lora_by_task, peft_cfg=peft_cfg)

    with pytest.raises(ValueError, match="not available in prepared KnOTS basis"):
        knots.project(
            prepared,
            lora_by_task={"t2": _peft_state_for_layer(layer, a, b)},
            peft_cfg=peft_cfg,
        )


def test_knots_svd_eps_parameter_controls_rank_truncation() -> None:
    torch.manual_seed(2)
    layer = "visual.transformer.resblocks.0.attn.q_proj"

    a1 = torch.randn(2, 4)
    b1 = torch.randn(6, 2)
    a2 = torch.randn(2, 4)
    b2 = torch.randn(6, 2)

    lora_by_task = {
        "t1": _peft_state_for_layer(layer, a1, b1),
        "t2": _peft_state_for_layer(layer, a2, b2),
    }

    knots = KnotsSpace()
    prepared = knots.prepare(
        lora_by_task=lora_by_task,
        peft_cfg={"lora_alpha": 4, "knots_svd_eps": 1e9},
    )
    assert prepared.layers[layer].S.numel() == 0
