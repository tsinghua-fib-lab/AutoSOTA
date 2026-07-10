from __future__ import annotations

import torch
import torch.nn as nn

from merge_and_rebase.finetune.train_vision import _save_peft_visual_adapter
from merge_and_rebase.finetune._vision_runtime import ImageEncoder
from merge_and_rebase.finetune.regularizers.kfac_ggn import ensure_openclip_kfac_surface
from merge_and_rebase.finetune.strategies.peft_lora import PeftLoraVision
from merge_and_rebase.models.patch_openclip_attention import LoRAableMHA, split_openclip_vit_attn


class _TinyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(8)
        self.attn = nn.MultiheadAttention(embed_dim=8, num_heads=2, batch_first=True)
        self.ln_2 = nn.LayerNorm(8)
        self.mlp = nn.Module()
        self.mlp.c_fc = nn.Linear(8, 16)
        self.mlp.c_proj = nn.Linear(16, 8)


class _TinyTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.resblocks = nn.ModuleList([_TinyBlock(), _TinyBlock()])


class _TinyVisual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Identity()
        self.class_embedding = nn.Parameter(torch.zeros(8))
        self.positional_embedding = nn.Parameter(torch.zeros(2, 8))
        self.proj = nn.Parameter(torch.zeros(8, 4))
        self.ln_pre = nn.LayerNorm(8)
        self.ln_post = nn.LayerNorm(8)
        self.transformer = _TinyTransformer()


class _TinyOpenClipModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = nn.Linear(1, 1)
        self.visual = _TinyVisual()


class _TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _TinyOpenClipModel()
        self.logit_scale = nn.Parameter(torch.ones(()))
        self.preprocess = object()
        self.train_preprocess = object()
        self.normalize = True
        self.register_buffer("_zs_text_features", torch.empty(0), persistent=False)


def _build_model() -> ImageEncoder:
    return ImageEncoder(_TinyClassifier())


def _peft_cfg() -> dict[str, object]:
    return {
        "target_modules": "auto",
        "r": 2,
        "lora_alpha": 4,
        "lora_dropout": 0.0,
        "bias": "none",
    }


def test_peft_lora_reuses_existing_softmax_attention_surface() -> None:
    model = _build_model()
    visual = model.clip_model.model.visual

    first_patch_count = split_openclip_vit_attn(visual, proj_dropout=0.0, attn_impl="softmax")
    second_patch_count = split_openclip_vit_attn(visual, proj_dropout=0.0, attn_impl="softmax")

    assert first_patch_count == 2
    assert second_patch_count == 0
    assert isinstance(visual.transformer.resblocks[0].attn, LoRAableMHA)

    _opt, _sched, info = PeftLoraVision().configure(
        model=model,
        lr=1e-4,
        weight_decay=0.0,
        warmup_length=1,
        steps=4,
        device=torch.device("cpu"),
        peft_cfg=_peft_cfg(),
        strategy_cfg={"params": {"trainable_params": "regularized_only"}},
    )

    assert getattr(model, "peft_patched_attn", False) is True
    assert getattr(model, "peft_patched_proj", False) is True
    assert info["lora_params"] > 0
    assert info["dense_trainable_params"] > 0
    assert any("lora_" in name and param.requires_grad for name, param in model.named_parameters())
    assert any(name.endswith("base_model.model.class_embedding") and param.requires_grad for name, param in model.clip_model.model.visual.named_parameters())
    assert any(name.endswith("base_model.model.transformer.resblocks.0.attn.q_proj.base_layer.bias") and param.requires_grad for name, param in model.clip_model.model.visual.named_parameters())
    assert any(name.endswith("base_model.model.ln_pre.weight") and param.requires_grad for name, param in model.clip_model.model.visual.named_parameters())
    assert any(name.endswith("base_model.model.lin_proj.lora_A.default.weight") for name, _ in model.clip_model.model.visual.named_parameters())


def test_peft_lora_configures_after_ekfac_surface_finalization() -> None:
    model = _build_model()

    surface = ensure_openclip_kfac_surface(model)
    assert surface["patched_blocks"] == 2
    assert isinstance(model.clip_model.model.visual.transformer.resblocks[0].attn, LoRAableMHA)

    _opt, _sched, info = PeftLoraVision().configure(
        model=model,
        lr=1e-4,
        weight_decay=0.0,
        warmup_length=1,
        steps=4,
        device=torch.device("cpu"),
        peft_cfg=_peft_cfg(),
        strategy_cfg={"params": {"trainable_params": "regularized_only"}},
    )

    assert getattr(model, "peft_patched_attn", False) is True
    assert getattr(model, "peft_patched_proj", False) is True
    assert info["lora_params"] > 0
    assert info["trainable_params"] >= info["lora_params"]


def test_peft_lora_delta_parameterizes_dense_visual_params() -> None:
    model = _build_model()

    _opt, _sched, info = PeftLoraVision().configure(
        model=model,
        lr=1e-4,
        weight_decay=0.0,
        warmup_length=1,
        steps=4,
        device=torch.device("cpu"),
        peft_cfg=_peft_cfg(),
        strategy_cfg={"params": {"parameterization": "delta", "trainable_params": "regularized_only"}},
    )

    assert info["parameterization"] == "delta"
    assert info["lora_params"] > 0
    assert info["dense_trainable_params"] > 0
    assert info["dense_delta_params"] > 0
    assert callable(getattr(model, "_current_param_map", None))
    assert callable(getattr(model, "_materialized_state_dict", None))
    assert any("lora_" in name and param.requires_grad for name, param in model.named_parameters())

    delta_module = getattr(model, "_delta_module", None)
    assert delta_module is not None
    assert "clip_model.model.visual.base_model.model.class_embedding" in delta_module.names
    assert any(name.endswith("base_model.model.class_embedding") and not param.requires_grad for name, param in model.clip_model.model.visual.named_parameters())


def test_peft_lora_regularized_only_supports_split_lrs_for_lora_and_dense() -> None:
    model = _build_model()

    opt, _sched, info = PeftLoraVision().configure(
        model=model,
        lr=1e-4,
        dense_lr=3e-5,
        weight_decay=0.0,
        warmup_length=1,
        steps=4,
        device=torch.device("cpu"),
        peft_cfg=_peft_cfg(),
        strategy_cfg={"params": {"parameterization": "delta", "trainable_params": "regularized_only"}},
    )

    assert [group.get("name") for group in opt.param_groups] == ["lora", "dense"]
    assert opt.param_groups[0]["lr"] == 1e-4
    assert opt.param_groups[1]["lr"] == 3e-5
    assert info["lr_lora"] == 1e-4
    assert info["lr_dense"] == 3e-5
    assert info["lora_group_params"] > 0
    assert info["dense_group_params"] > 0
    assert info["other_group_params"] == 0


def test_peft_lora_regularized_only_defaults_dense_lr_to_lora_lr() -> None:
    model = _build_model()

    opt, _sched, info = PeftLoraVision().configure(
        model=model,
        lr=1e-4,
        weight_decay=0.0,
        warmup_length=1,
        steps=4,
        device=torch.device("cpu"),
        peft_cfg=_peft_cfg(),
        strategy_cfg={"params": {"parameterization": "delta", "trainable_params": "regularized_only"}},
    )

    assert [group.get("name") for group in opt.param_groups] == ["lora", "dense"]
    assert opt.param_groups[0]["lr"] == 1e-4
    assert opt.param_groups[1]["lr"] == 1e-4
    assert info["lr_dense"] == 1e-4


def test_peft_lora_save_materializes_dense_delta_state(tmp_path) -> None:
    model = _build_model()

    _opt, _sched, _info = PeftLoraVision().configure(
        model=model,
        lr=1e-4,
        weight_decay=0.0,
        warmup_length=1,
        steps=4,
        device=torch.device("cpu"),
        peft_cfg=_peft_cfg(),
        strategy_cfg={"params": {"parameterization": "delta", "trainable_params": "regularized_only"}},
    )

    delta_module = getattr(model, "_delta_module", None)
    assert delta_module is not None
    class_embedding_index = delta_module.names.index("clip_model.model.visual.base_model.model.class_embedding")
    with torch.no_grad():
        delta_module.params[class_embedding_index].fill_(0.25)

    payload = _save_peft_visual_adapter(
        model=model,
        task_dir=tmp_path,
        strategy="peft_lora",
        suffix=None,
        peft_cfg=_peft_cfg(),
        patched_attn=bool(getattr(model, "peft_patched_attn", False)),
        attn_patch_cfg=getattr(model, "peft_attn_patch_cfg", None),
    )

    assert "base_model.model.class_embedding" in payload["peft_dense_state"]
    assert torch.allclose(
        payload["peft_dense_state"]["base_model.model.class_embedding"],
        torch.full((8,), 0.25),
    )
