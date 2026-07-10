from __future__ import annotations

import torch

from merge_and_rebase.merge.methods.task_arithmetic import TaskArithmeticMerge
from merge_and_rebase.merge.runtime import build_dense_delta_branch, build_merged_state_for_alpha
from merge_and_rebase.merge.subspaces.core_space import CoreSpace


def _peft_state_for_layer(prefix: str, a: torch.Tensor, b: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        f"{prefix}.lora_A.weight": a,
        f"{prefix}.lora_B.weight": b,
    }


def test_build_merged_state_for_alpha_preserves_dense_delta_branch_in_peft_subspace() -> None:
    torch.manual_seed(0)
    layer = "visual.transformer.resblocks.0.attn.q_proj"
    peft_cfg = {"lora_alpha": 4}
    tasks = ["t1", "t2"]

    a1 = torch.randn(2, 4)
    b1 = torch.randn(6, 2)
    a2 = torch.randn(2, 4)
    b2 = torch.randn(6, 2)
    lora_by_task = {
        "t1": _peft_state_for_layer(layer, a1, b1),
        "t2": _peft_state_for_layer(layer, a2, b2),
    }

    subspace = CoreSpace()
    prepared = subspace.prepare(lora_by_task=lora_by_task, peft_cfg=peft_cfg)
    projected = subspace.project(prepared, lora_by_task=lora_by_task, peft_cfg=peft_cfg)

    lora_only_tuned_by_task = {
        "t1": {f"{layer}.weight": b1 @ a1},
        "t2": {f"{layer}.weight": b2 @ a2},
    }
    full_tuned_by_task = {
        "t1": {f"{layer}.weight": b1 @ a1, "visual.ln_post.weight": torch.tensor([0.2, -0.1])},
        "t2": {f"{layer}.weight": b2 @ a2, "visual.ln_post.weight": torch.tensor([-0.4, 0.3])},
    }
    base_sd = {
        f"{layer}.weight": torch.zeros_like(b1 @ a1),
        "visual.ln_post.weight": torch.zeros(2),
    }
    dense_base_sd, dense_tuned_sds = build_dense_delta_branch(
        tasks=tasks,
        full_tuned_by_task=full_tuned_by_task,
        lora_only_tuned_by_task=lora_only_tuned_by_task,
        base_sd=base_sd,
    )

    merged = build_merged_state_for_alpha(
        method=TaskArithmeticMerge(),
        prepared=None,
        base_sd_for_merge={key: torch.zeros_like(value) for key, value in projected["t1"].items()},
        tuned_sds_list=[projected[t] for t in tasks],
        weights=[0.25, 0.75],
        method_params={},
        alpha=1.2,
        peft_subspace="core",
        subspace=subspace,
        subspace_prepared=prepared,
        peft_cfg=peft_cfg,
        peft_state_by_task=lora_by_task,
        tasks=tasks,
        merge_base_sd=base_sd,
        dense_base_sd_for_merge=dense_base_sd,
        dense_tuned_sds_list=dense_tuned_sds,
    )

    expected_lora = 1.2 * (0.25 * (b1 @ a1) + 0.75 * (b2 @ a2))
    expected_dense = 1.2 * (0.25 * torch.tensor([0.2, -0.1]) + 0.75 * torch.tensor([-0.4, 0.3]))
    assert torch.allclose(merged[f"{layer}.weight"], expected_lora, atol=1e-5, rtol=1e-5)
    assert torch.allclose(merged["visual.ln_post.weight"], expected_dense, atol=1e-6, rtol=1e-6)
