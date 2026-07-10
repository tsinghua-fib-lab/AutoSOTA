from __future__ import annotations

import torch
import torch.nn as nn

from merge_and_rebase.models.parameter_subsets import resolve_parameter_subset, resolve_visual_parameter_plan


class _ToyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = nn.Module()
        self.attn.q_proj = nn.Linear(4, 4)
        self.attn.k_proj = nn.Linear(4, 4)
        self.attn.v_proj = nn.Linear(4, 4)
        self.attn.out_proj = nn.Linear(4, 4)
        self.mlp = nn.Module()
        self.mlp.c_fc = nn.Linear(4, 8)
        self.mlp.c_proj = nn.Linear(8, 4)
        self.ln_pre = nn.LayerNorm(4)


class _ToyVisual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.resblocks = nn.ModuleList([_ToyBlock()])
        self.class_embedding = nn.Parameter(torch.zeros(4))
        self.proj = nn.Parameter(torch.zeros(4, 4))


class _ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.visual = _ToyVisual()


def test_regularized_only_includes_visual_proj() -> None:
    model = _ToyModel()

    resolution = resolve_parameter_subset(model, "regularized_only")

    assert resolution.supported is True
    assert "visual.proj" in resolution.parameter_names
    assert "visual.class_embedding" in resolution.parameter_names
    assert "visual.transformer.resblocks.0.attn.q_proj.weight" in resolution.parameter_names
    assert "visual.transformer.resblocks.0.mlp.c_fc.weight" in resolution.parameter_names
    assert "visual.transformer.resblocks.0.ln_pre.weight" in resolution.parameter_names


def test_regularized_only_plan_splits_lora_hosts_and_dense_params() -> None:
    model = _ToyModel()

    plan = resolve_visual_parameter_plan(model, "regularized_only")

    assert plan.supported is True
    assert "visual.proj" in plan.lora_parameter_names
    assert "visual.transformer.resblocks.0.attn.q_proj.weight" in plan.lora_parameter_names
    assert "visual.transformer.resblocks.0.mlp.c_proj.weight" in plan.lora_parameter_names
    assert "visual.transformer.resblocks.0.attn.q_proj.bias" in plan.dense_parameter_names
    assert "visual.transformer.resblocks.0.ln_pre.weight" in plan.dense_parameter_names
    assert "visual.class_embedding" in plan.dense_parameter_names
    assert set(plan.lora_target_modules) == {"c_fc", "c_proj", "k_proj", "lin_proj", "out_proj", "q_proj", "v_proj"}
