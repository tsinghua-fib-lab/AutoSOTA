from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.func import functional_call

from merge_and_rebase.postmerge import PostMergeContext, get_postmerge_method
from merge_and_rebase.postmerge.methods.adamerging import (
    clamped_alpha_from_raw,
    init_clamped_raw_alpha,
    prediction_entropy,
)
from merge_and_rebase.postmerge.task_delta_bank import TaskDeltaBank, layer_group_for_key


def test_task_delta_bank_materializes_task_alpha() -> None:
    base = {"w": torch.tensor([1.0, 2.0])}
    tuned = [
        {"w": torch.tensor([2.0, 4.0])},
        {"w": torch.tensor([0.0, 1.0])},
    ]
    bank = TaskDeltaBank.build(base=base, tuned=tuned, tasks=["a", "b"], weights=[2.0, 0.5], kind="vision")

    out = bank.materialize(torch.tensor([0.25, 0.5]), mode="task")

    expected = torch.tensor([1.0, 2.0]) + 2.0 * 0.25 * torch.tensor([1.0, 2.0]) + 0.5 * 0.5 * torch.tensor([-1.0, -1.0])
    assert torch.allclose(out["w"], expected)


def test_task_delta_bank_materializes_layer_alpha() -> None:
    base = {
        "visual.transformer.resblocks.0.attn.weight": torch.tensor([1.0]),
        "visual.transformer.resblocks.1.attn.weight": torch.tensor([1.0]),
    }
    tuned = [
        {
            "visual.transformer.resblocks.0.attn.weight": torch.tensor([2.0]),
            "visual.transformer.resblocks.1.attn.weight": torch.tensor([3.0]),
        }
    ]
    bank = TaskDeltaBank.build(base=base, tuned=tuned, tasks=["a"], weights=None, kind="vision")
    alphas = torch.zeros((len(bank.tensor_keys), 1))
    alphas[bank.tensor_index_by_key["visual.transformer.resblocks.0.attn.weight"], 0] = 0.5
    alphas[bank.tensor_index_by_key["visual.transformer.resblocks.1.attn.weight"], 0] = 0.25

    out = bank.materialize(alphas, mode="layer")

    assert torch.allclose(out["visual.transformer.resblocks.0.attn.weight"], torch.tensor([1.5]))
    assert torch.allclose(out["visual.transformer.resblocks.1.attn.weight"], torch.tensor([1.5]))


def test_task_delta_bank_layer_alpha_is_per_tensor_and_task() -> None:
    base = {"a": torch.tensor([1.0]), "b": torch.tensor([10.0])}
    tuned = [
        {"a": torch.tensor([2.0]), "b": torch.tensor([12.0])},
        {"a": torch.tensor([4.0]), "b": torch.tensor([16.0])},
    ]
    bank = TaskDeltaBank.build(base=base, tuned=tuned, tasks=["t1", "t2"], weights=None, kind="vision")
    alphas = torch.zeros(bank.alpha_shape("layer"))
    alphas[bank.tensor_index_by_key["a"], 0] = 0.5
    alphas[bank.tensor_index_by_key["a"], 1] = 0.25
    alphas[bank.tensor_index_by_key["b"], 0] = 0.0
    alphas[bank.tensor_index_by_key["b"], 1] = 1.0

    out = bank.materialize(alphas, mode="layer")

    assert torch.allclose(out["a"], torch.tensor([2.25]))
    assert torch.allclose(out["b"], torch.tensor([16.0]))


def test_layer_group_for_key_defaults_for_vision_and_llm() -> None:
    assert (
        layer_group_for_key("visual.transformer.resblocks.11.mlp.c_fc.weight", kind="vision")
        == "visual.transformer.resblocks.11"
    )
    assert layer_group_for_key("model.layers.3.self_attn.q_proj.weight", kind="llm") == "model.layers.3"
    assert layer_group_for_key("encoder.block.2.layer.0.weight", kind="llm") == "encoder.block.2"
    assert layer_group_for_key("logit_scale", kind="vision") == "__other__"


def test_official_clamped_alpha_initialization_matches_prior() -> None:
    raw = init_clamped_raw_alpha((2,), init_alpha=0.3, alpha_min=0.0, alpha_max=1.0, device="cpu")
    alpha = clamped_alpha_from_raw(raw, alpha_min=0.0, alpha_max=1.0)
    assert torch.allclose(alpha, torch.full((2,), 0.3))

    raw = init_clamped_raw_alpha((1,), init_alpha=2.0, alpha_min=0.0, alpha_max=1.0, device="cpu")
    assert torch.allclose(clamped_alpha_from_raw(raw, alpha_min=0.0, alpha_max=1.0), torch.ones(1))


def test_adamerging_toy_module_updates_alpha() -> None:
    model = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    base = {k: v.detach().clone() for k, v in model.state_dict().items()}
    tuned = [{"weight": torch.tensor([[1.0, 0.0], [-1.0, 0.0]])}]
    x = torch.tensor([[1.0, 0.0], [1.0, 0.0]])

    def entropy_loss(bank: TaskDeltaBank, alpha_values: torch.Tensor, alpha_mode: str) -> torch.Tensor:
        params = bank.merged_parameter_dict(model, alpha_values, mode=alpha_mode, device="cpu")
        logits = functional_call(model, params, (x,))
        return prediction_entropy(logits)

    result = get_postmerge_method("adamerging").run(
        PostMergeContext(
            kind="llm",
            model=model,
            base=base,
            tuned=tuned,
            tasks=["toy"],
            config={
                "alpha_mode": "task",
                "steps": 20,
                "lr": 0.2,
                "alpha_min": 0.0,
                "alpha_max": 1.0,
                "init_alpha": 0.1,
                "log_every": 0,
                "device": "cpu",
            },
            entropy_loss_fn=entropy_loss,
        )
    )

    final_alpha = float(result.metadata["alpha_values"][0])
    assert final_alpha > 0.1
    assert torch.allclose(result.merged_state["weight"], tuned[0]["weight"] * final_alpha)


def test_adamerging_rejects_peft_subspace() -> None:
    model = nn.Linear(1, 1, bias=False)
    base = {k: v.detach().clone() for k, v in model.state_dict().items()}
    tuned = [{k: v.detach().clone() for k, v in model.state_dict().items()}]

    with pytest.raises(ValueError, match="peft_subspace='full'"):
        get_postmerge_method("adamerging").run(
            PostMergeContext(
                kind="llm",
                model=model,
                base=base,
                tuned=tuned,
                tasks=["toy"],
                peft_subspace="core",
                entropy_loss_fn=lambda _bank, _alpha, _mode: torch.tensor(0.0),
            )
        )


def test_adamerging_layer_mode_uses_tensor_by_task_alpha_shape() -> None:
    model = nn.Linear(2, 2, bias=False)
    base = {k: v.detach().clone() for k, v in model.state_dict().items()}
    tuned = [{"weight": torch.tensor([[1.0, 0.0], [-1.0, 0.0]])}]

    result = get_postmerge_method("adamerging").run(
        PostMergeContext(
            kind="llm",
            model=model,
            base=base,
            tuned=tuned,
            tasks=["toy"],
            config={"alpha_mode": "layer", "steps": 0, "init_alpha": 0.3, "device": "cpu"},
            entropy_loss_fn=lambda _bank, _alpha, _mode: torch.tensor(0.0),
        )
    )

    assert result.metadata["alpha_shape"] == [1, 1]
    assert torch.allclose(result.merged_state["weight"], base["weight"] + 0.3 * (tuned[0]["weight"] - base["weight"]))
