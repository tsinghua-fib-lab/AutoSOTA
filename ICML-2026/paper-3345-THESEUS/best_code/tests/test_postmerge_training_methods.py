from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from merge_and_rebase.postmerge import PostMergeContext, get_postmerge_method
from merge_and_rebase.postmerge.methods._vision_training import VisionPostmergeTrainer


class _ToyVisual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Parameter(torch.zeros(2, 2))
        self.ln_post = nn.LayerNorm(2)
        self.other = nn.Parameter(torch.ones(2, 2))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.ln_post(images) @ self.proj + 0.0 * self.other.sum()


class _ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.visual = _ToyVisual()


class _ToyClassifier:
    def __init__(self, model: _ToyModel) -> None:
        self.model = model
        self.normalize = False
        self.logit_scale = 1.0
        self._zs_text_features = torch.empty(0)

    def build_zeroshot_text_features(self, *_args, **_kwargs) -> None:
        self._zs_text_features = torch.eye(2)


def _toy_context(
    *,
    model: _ToyModel | None = None,
    tuned_proj: torch.Tensor | None = None,
    tuned_other: torch.Tensor | None = None,
    peft_subspace: str = "full",
    config: dict | None = None,
) -> PostMergeContext:
    model = model or _ToyModel()
    base = {k: v.detach().clone() for k, v in model.state_dict().items()}
    tuned = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if tuned_proj is not None:
        tuned["visual.proj"] = tuned_proj.detach().clone()
    if tuned_other is not None:
        tuned["visual.other"] = tuned_other.detach().clone()

    images = torch.eye(2).repeat(4, 1)
    labels = torch.tensor([0, 1] * 4)
    loader = DataLoader(TensorDataset(images, labels), batch_size=2, shuffle=False)
    clf = _ToyClassifier(model)
    per_task = [
        {
            "task": "toy",
            "loaders": SimpleNamespace(train=loader),
            "classnames": ["zero", "one"],
            "build_cfg_task": object(),
            "text_features": torch.eye(2),
        }
    ]
    return PostMergeContext(
        kind="vision",
        model=model,
        base=base,
        tuned=[tuned],
        tasks=["toy"],
        peft_subspace=peft_subspace,
        config=config or {},
        resources={"classifier": clf, "per_task": per_task, "device": "cpu"},
    )


def test_vision_postmerge_trainer_ce_uses_labels_and_entropy_ignores_them() -> None:
    ce_trainer = VisionPostmergeTrainer(_toy_context(config={"loss": "ce"}), {"loss": "ce", "device": "cpu"})
    entropy_trainer = VisionPostmergeTrainer(
        _toy_context(config={"loss": "entropy"}), {"loss": "entropy", "device": "cpu"}
    )
    logits = torch.tensor([[3.0, -3.0]])

    ce_correct = ce_trainer.loss_from_logits(logits, torch.tensor([0]))
    ce_wrong = ce_trainer.loss_from_logits(logits, torch.tensor([1]))
    assert ce_correct < ce_wrong

    entropy_a = entropy_trainer.loss_from_logits(logits, torch.tensor([0]))
    entropy_b = entropy_trainer.loss_from_logits(logits, torch.tensor([1]))
    assert torch.allclose(entropy_a, entropy_b)


def test_task_vector_finetune_updates_delta_tensors_with_ce() -> None:
    context = _toy_context(config={"loss": "ce", "steps": 5, "lr": 0.5, "log_every": 0, "device": "cpu"})

    result = get_postmerge_method("task_vector_finetune").run(context)

    assert result.metadata["loss"] == "ce"
    assert result.metadata["num_trainable_tensors"] == 4
    assert not torch.allclose(result.merged_state["visual.proj"], context.base["visual.proj"])


def test_task_vector_finetune_rejects_peft_subspace() -> None:
    context = _toy_context(peft_subspace="core")

    with pytest.raises(ValueError, match="peft_subspace='full'"):
        get_postmerge_method("task_vector_finetune").run(context)


def test_merged_delta_finetune_updates_single_composed_delta_with_ce() -> None:
    context = _toy_context(config={"loss": "ce", "steps": 5, "lr": 0.5, "log_every": 0, "device": "cpu"})

    result = get_postmerge_method("merged_delta_finetune").run(context)

    assert result.metadata["loss"] == "ce"
    assert result.metadata["num_trainable_tensors"] == 4
    assert not torch.allclose(result.merged_state["visual.proj"], context.base["visual.proj"])


def test_merged_delta_finetune_initializes_from_weighted_task_vector_sum() -> None:
    model = _ToyModel()
    tuned_proj = torch.eye(2)
    tuned_other = model.visual.other.detach() + 4.0
    context = _toy_context(
        model=model,
        tuned_proj=tuned_proj,
        tuned_other=tuned_other,
        config={"loss": "ce", "steps": 0, "log_every": 0, "device": "cpu"},
    )

    result = get_postmerge_method("merged_delta_finetune").run(context)

    assert torch.allclose(result.merged_state["visual.proj"], tuned_proj)
    assert torch.allclose(result.merged_state["visual.other"], tuned_other)


def test_merged_delta_finetune_rejects_peft_subspace() -> None:
    context = _toy_context(peft_subspace="core")

    with pytest.raises(ValueError, match="peft_subspace='full'"):
        get_postmerge_method("merged_delta_finetune").run(context)


def test_vision_head_probe_trains_only_final_vision_head_task_vector_entries() -> None:
    model = _ToyModel()
    tuned_proj = torch.eye(2)
    tuned_ln_weight = model.visual.ln_post.weight.detach() + 0.5
    tuned_other = model.visual.other.detach() + 10.0
    context = _toy_context(
        model=model,
        tuned_proj=tuned_proj,
        tuned_other=tuned_other,
        config={"loss": "ce", "steps": 5, "lr": 0.5, "init_alpha": 0.3, "log_every": 0, "device": "cpu"},
    )
    context.tuned[0]["visual.ln_post.weight"] = tuned_ln_weight
    expected_fixed_other = context.base["visual.other"] + 0.3 * (tuned_other - context.base["visual.other"])

    result = get_postmerge_method("vision_head_probe").run(context)

    assert result.metadata["trainable_tensor_keys"] == [
        "visual.ln_post.bias",
        "visual.ln_post.weight",
        "visual.proj",
    ]
    assert not torch.allclose(result.merged_state["visual.proj"], tuned_proj)
    assert not torch.allclose(result.merged_state["visual.ln_post.weight"], tuned_ln_weight)
    assert torch.allclose(result.merged_state["visual.other"], expected_fixed_other)
    assert result.metadata["init_alpha"] == 0.3
