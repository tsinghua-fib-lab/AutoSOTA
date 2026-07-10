from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import torch
import torch.nn as nn
import torch.optim as optim

from merge_and_rebase.finetune import train_vision
from merge_and_rebase.finetune.reference_tasks import build_reference_task_resolution_context
from merge_and_rebase.finetune.regularizers.kfac_ggn import (
    KfacGgnConfig,
    KfacGgnRegularizer,
    TaskCurvatureStats,
    _cls_only_from_sequence,
    _flatten_sequence,
    _matrix_gram_from_rows,
    _sum_over_sequence_axis,
    _base_snapshot,
    _delta_params,
    _visual_param_map,
    collect_curvature,
    ensure_openclip_kfac_surface,
    load_task_curvature,
    metadata_compatible,
    save_task_curvature,
    select_tracked_parameters,
    task_cache_path,
)
from merge_and_rebase.finetune.regularizers.registry import list_regularizers
from merge_and_rebase.finetune.strategies.peft_lora import PeftLoraVision
from merge_and_rebase.finetune.train_vision import ImageEncoder
from merge_and_rebase.models.openclip_classifier import OpenClipBuildConfig, OpenClipClassifier, zero_shot_logits_from_features


def _dummy_transform(x):
    return x


class _ToyMLP(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.c_fc = nn.Linear(width, 2 * width)
        self.c_proj = nn.Linear(2 * width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(torch.relu(self.c_fc(x)))


class _ToyOpenClipBlock(nn.Module):
    def __init__(self, width: int, *, batch_first: bool = False) -> None:
        super().__init__()
        self.batch_first = batch_first
        self.ln_1 = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, num_heads=1, batch_first=batch_first)
        self.ln_2 = nn.LayerNorm(width)
        self.mlp = _ToyMLP(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_in = self.ln_1(x)
        attn_out = self.attn(attn_in, attn_in, attn_in, need_weights=False)[0]
        x = x + attn_out
        return x + self.mlp(self.ln_2(x))


class _ToyOpenClipVisual(nn.Module):
    def __init__(self, width: int = 4, out_dim: int = 3) -> None:
        super().__init__()
        self.class_embedding = nn.Parameter(torch.randn(width) * 0.01)
        self.positional_embedding = nn.Parameter(torch.randn(3, width) * 0.01)
        self.conv1 = nn.Linear(width, width, bias=False)
        self.ln_pre = nn.LayerNorm(width)
        self.transformer = nn.Module()
        self.transformer.resblocks = nn.ModuleList([_ToyOpenClipBlock(width, batch_first=False)])
        self.ln_post = nn.LayerNorm(width)
        self.proj = nn.Parameter(torch.randn(width, out_dim) * 0.02)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        patch = self.conv1(images)
        patches = torch.stack([patch, 0.5 * patch + 0.25], dim=1)
        cls = self.class_embedding.to(dtype=patches.dtype).view(1, 1, -1).expand(patches.shape[0], 1, -1)
        x = torch.cat([cls, patches], dim=1)
        x = x + self.positional_embedding.to(dtype=x.dtype)
        x = self.ln_pre(x)
        x = x.transpose(0, 1)
        for block in self.transformer.resblocks:
            x = block(x)
        x = x.transpose(0, 1)
        x = self.ln_post(x[:, 0, :])
        return x @ self.proj


class _ToyClsTokenLayer(nn.Module):
    def forward(self, x: torch.Tensor, class_embedding: torch.Tensor) -> torch.Tensor:
        cls = class_embedding.to(dtype=x.dtype).view(1, 1, -1).expand(x.shape[0], 1, -1)
        return torch.cat([cls, x], dim=1)


class _ToyBatchFirstVisual(nn.Module):
    def __init__(self, width: int = 4, out_dim: int = 3) -> None:
        super().__init__()
        self.class_embedding = nn.Parameter(torch.randn(width) * 0.01)
        self.positional_embedding = nn.Parameter(torch.randn(3, width) * 0.01)
        self.conv1 = nn.Linear(width, width, bias=False)
        self.ln_pre = nn.LayerNorm(width)
        self.transformer = nn.Module()
        self.transformer.resblocks = nn.ModuleList([_ToyOpenClipBlock(width, batch_first=True)])
        self.ln_post = nn.LayerNorm(width)
        self.proj = nn.Parameter(torch.randn(width, out_dim) * 0.02)
        self.pool_type = "tok"

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        patch = self.conv1(images)
        patches = torch.stack([patch, 0.5 * patch + 0.25], dim=1)
        cls = self.class_embedding.to(dtype=patches.dtype).view(1, 1, -1).expand(patches.shape[0], 1, -1)
        x = torch.cat([cls, patches], dim=1)
        x = x + self.positional_embedding.to(dtype=x.dtype)
        x = self.ln_pre(x)
        for block in self.transformer.resblocks:
            x = block(x)
        x = self.ln_post(x)
        return x[:, 0, :] @ self.proj


class _ToyMammothStyleVisual(nn.Module):
    def __init__(self, width: int = 4, out_dim: int = 3) -> None:
        super().__init__()
        self.class_embedding = nn.Parameter(torch.randn(width) * 0.01)
        self.positional_embedding = nn.Parameter(torch.randn(3, width) * 0.01)
        self.conv1 = nn.Linear(width, width, bias=False)
        self.cls_token_layer = _ToyClsTokenLayer()
        self.ln_pre = nn.LayerNorm(width)
        self.transformer = nn.Module()
        self.transformer.resblocks = nn.ModuleList([_ToyOpenClipBlock(width, batch_first=False)])
        self.ln_post = nn.LayerNorm(width)
        self.lin_proj = nn.Linear(width, out_dim, bias=False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        patch = self.conv1(images)
        patches = torch.stack([patch, 0.5 * patch + 0.25], dim=1)
        x = self.cls_token_layer(patches, self.class_embedding)
        x = x + self.positional_embedding.to(dtype=x.dtype)
        x = self.ln_pre(x)
        x = x.transpose(0, 1)
        for block in self.transformer.resblocks:
            x = block(x)
        x = x.transpose(0, 1)
        x = self.ln_post(x[:, 0, :])
        return self.lin_proj(x)


def _copy_batch_first_to_mammoth(source: _ToyBatchFirstVisual, target: _ToyMammothStyleVisual) -> None:
    with torch.no_grad():
        target.class_embedding.copy_(source.class_embedding)
        target.positional_embedding.copy_(source.positional_embedding)
        target.conv1.weight.copy_(source.conv1.weight)
        target.ln_pre.weight.copy_(source.ln_pre.weight)
        target.ln_pre.bias.copy_(source.ln_pre.bias)
        target.ln_post.weight.copy_(source.ln_post.weight)
        target.ln_post.bias.copy_(source.ln_post.bias)
        target.lin_proj.weight.copy_(source.proj.T)
        for src_block, tgt_block in zip(source.transformer.resblocks, target.transformer.resblocks):
            tgt_block.ln_1.load_state_dict(src_block.ln_1.state_dict())
            tgt_block.ln_2.load_state_dict(src_block.ln_2.state_dict())
            tgt_block.mlp.load_state_dict(src_block.mlp.state_dict())
            tgt_block.attn.load_state_dict(src_block.attn.state_dict())


class _ToyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.visual = _ToyOpenClipVisual()
        self.transformer = nn.Identity()


def _build_classifier(num_classes: int = 3) -> OpenClipClassifier:
    classifier = OpenClipClassifier(
        model=_ToyBackbone(),
        tokenizer=lambda texts: torch.zeros(len(texts), 1, dtype=torch.long),
        preprocess=_dummy_transform,
        normalize=True,
        logit_scale=3.5,
    )
    classifier._zs_text_features = torch.eye(num_classes, dtype=torch.float32)
    return classifier


class _TrainDummyClipModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = nn.Linear(1, 1)
        self.visual = nn.Sequential(nn.Flatten(), nn.Linear(12, 2))


class _TrainDummyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _TrainDummyClipModel()
        self.preprocess = object()
        self.normalize = True
        self.logit_scale = 1.0
        self.register_buffer("_zs_text_features", torch.empty(0), persistent=False)

    def build_zeroshot_text_features(self, classnames, build_cfg):
        del build_cfg
        self._zs_text_features = torch.eye(len(classnames), dtype=torch.float32)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        image_features = self.model.visual(images)
        return zero_shot_logits_from_features(self, image_features, normalize_image_features=self.normalize)


class _DummyStrategy:
    def configure(self, *, model, lr, weight_decay, warmup_length, optimizer="adamw", steps=1, device, **kwargs):
        del weight_decay, warmup_length, steps, device, kwargs
        params = [p for p in model.parameters() if p.requires_grad]
        opt = optim.Adam(params, lr=lr)
        scheduler = lambda step: None
        return opt, scheduler, {"trainable_params": sum(p.numel() for p in params)}


def _build_identity_curvature(plan) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    aaT = {}
    ggT = {}
    ffT = {}
    for key, block in plan.matrix_blocks.items():
        shape = plan.param_shapes[key]
        if block.is_projection and key == "visual.proj":
            rows = shape[1]
            cols = shape[0]
        else:
            rows = shape[0]
            cols = shape[1]
        cols = cols + (1 if block.bias_key is not None else 0)
        aaT[key] = torch.eye(cols)
        ggT[key] = torch.eye(rows)
    for key in plan.full_blocks:
        rows = plan.param_shapes[key][0]
        ffT[key] = torch.eye(rows)
    return aaT, ggT, ffT


def test_regularizer_registry_exposes_kfac_ggn() -> None:
    assert "kfac_ggn" in list_regularizers()


def test_select_tracked_parameters_and_delta_params_support_peft_lora_weight_space() -> None:
    classifier = _build_classifier()
    model = ImageEncoder(classifier)
    PeftLoraVision().configure(
        model=model,
        lr=1e-4,
        weight_decay=0.0,
        warmup_length=1,
        steps=4,
        device=torch.device("cpu"),
        peft_cfg={
            "target_modules": "auto",
            "r": 2,
            "lora_alpha": 4,
            "lora_dropout": 0.0,
            "bias": "none",
        },
        strategy_cfg={"params": {"trainable_params": "regularized_only"}},
    )

    plan = select_tracked_parameters(model)
    assert plan.ignored_trainable == []
    assert "visual.proj" in plan.matrix_blocks
    assert "visual.class_embedding" in plan.full_blocks

    base = _base_snapshot(model, plan)
    named_params = dict(model.clip_model.model.visual.named_parameters())
    named_params["base_model.model.transformer.resblocks.0.attn.q_proj.lora_B.default.weight"].data.add_(0.25)
    named_params["base_model.model.class_embedding"].data.add_(0.1)
    named_params["base_model.model.lin_proj.lora_B.default.weight"].data.add_(0.15)

    deltas = _delta_params(model, base)
    assert torch.count_nonzero(deltas["visual.transformer.resblocks.0.attn.q_proj.weight"]).item() > 0
    assert torch.count_nonzero(deltas["visual.class_embedding"]).item() > 0
    assert torch.count_nonzero(deltas["visual.proj"]).item() > 0


def test_delta_params_support_peft_lora_delta_regularized_only_weight_space() -> None:
    classifier = _build_classifier()
    model = ImageEncoder(classifier)
    PeftLoraVision().configure(
        model=model,
        lr=1e-4,
        weight_decay=0.0,
        warmup_length=1,
        steps=4,
        device=torch.device("cpu"),
        peft_cfg={
            "target_modules": "auto",
            "r": 2,
            "lora_alpha": 4,
            "lora_dropout": 0.0,
            "bias": "none",
        },
        strategy_cfg={"params": {"parameterization": "delta", "trainable_params": "regularized_only"}},
    )

    plan = select_tracked_parameters(model)
    base = _base_snapshot(model, plan)
    current = _visual_param_map(model)
    assert "visual.transformer.resblocks.0.attn.q_proj.weight" in current
    assert "visual.class_embedding" in current
    assert "visual.ln_pre.weight" in current
    assert "visual.proj" in current

    named_params = dict(model.clip_model.model.visual.named_parameters())
    named_params["base_model.model.transformer.resblocks.0.attn.q_proj.lora_B.default.weight"].data.add_(0.25)
    named_params["base_model.model.lin_proj.lora_B.default.weight"].data.add_(0.15)

    delta_module = getattr(model, "_delta_module", None)
    assert delta_module is not None
    class_embedding_idx = delta_module.names.index("clip_model.model.visual.base_model.model.class_embedding")
    ln_pre_weight_idx = delta_module.names.index("clip_model.model.visual.base_model.model.ln_pre.weight")
    with torch.no_grad():
        delta_module.params[class_embedding_idx].add_(0.1)
        delta_module.params[ln_pre_weight_idx].add_(0.05)

    deltas = _delta_params(model, base)
    assert torch.count_nonzero(deltas["visual.transformer.resblocks.0.attn.q_proj.weight"]).item() > 0
    assert torch.count_nonzero(deltas["visual.class_embedding"]).item() > 0
    assert torch.count_nonzero(deltas["visual.ln_pre.weight"]).item() > 0
    assert torch.count_nonzero(deltas["visual.proj"]).item() > 0


def test_select_tracked_parameters_reports_supported_blocks_and_ignored_trainables() -> None:
    visual = _ToyOpenClipVisual()
    ensure_openclip_kfac_surface(visual)
    for param in visual.parameters():
        param.requires_grad_(True)
    plan = select_tracked_parameters(visual)
    assert "visual.transformer.resblocks.0.attn.q_proj.weight" in plan.matrix_blocks
    assert "visual.transformer.resblocks.0.attn.k_proj.weight" in plan.matrix_blocks
    assert "visual.transformer.resblocks.0.attn.v_proj.weight" in plan.matrix_blocks
    assert "visual.transformer.resblocks.0.attn.out_proj.weight" in plan.matrix_blocks
    assert "visual.transformer.resblocks.0.mlp.c_fc.weight" in plan.matrix_blocks
    assert "visual.proj" in plan.matrix_blocks
    assert "visual.class_embedding" in plan.full_blocks
    assert "visual.conv1.weight" in plan.ignored_trainable
    assert "visual.positional_embedding" in plan.ignored_trainable


def test_collect_curvature_respects_train_percent_and_seed() -> None:
    torch.manual_seed(0)
    visual = _ToyOpenClipVisual()
    ensure_openclip_kfac_surface(visual)
    data = [(torch.randn(4), torch.tensor(0)) for _ in range(12)]
    loader = torch.utils.data.DataLoader(data, batch_size=2, shuffle=False)
    cfg = KfacGgnConfig(train_percent=0.5, fisher_seed=17, fisher_num_samples_expectation=1)
    stats_a = collect_curvature(visual, loader, tracked_params=select_tracked_parameters(visual), config=cfg)
    stats_b = collect_curvature(visual, loader, tracked_params=select_tracked_parameters(visual), config=cfg)
    assert stats_a.num_examples_aaT == 6
    assert stats_a.num_examples_ggT == 6
    assert set(stats_a.aaT) == set(stats_b.aaT)
    for key in stats_a.aaT:
        assert torch.allclose(stats_a.aaT[key], stats_b.aaT[key])
        assert torch.allclose(stats_a.ggT[key], stats_b.ggT[key])


def test_flatten_sequence_matches_between_batch_first_and_sequence_first_layouts() -> None:
    x_batch_first = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    x_sequence_first = x_batch_first.permute(1, 0, 2)

    rows_bf, denom_bf = _flatten_sequence(
        x_batch_first,
        "runtime_batch_size_inferred",
        current_batch_size=2,
        target="bf",
    )
    rows_sf, denom_sf = _flatten_sequence(
        x_sequence_first,
        "runtime_batch_size_inferred",
        current_batch_size=2,
        target="sf",
    )

    assert denom_bf == denom_sf == 3
    assert torch.allclose(
        _matrix_gram_from_rows(rows_bf, normalize_by=denom_bf),
        _matrix_gram_from_rows(rows_sf, normalize_by=denom_sf),
    )


def test_layer_norm_sequence_reduction_matches_between_layouts_for_cls_only_gradients() -> None:
    normalized_batch_first = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) / 10.0
    grad_batch_first = torch.zeros_like(normalized_batch_first)
    grad_batch_first[:, 0, :] = torch.tensor([[1.0, -2.0, 3.0, -4.0], [0.5, 1.5, -0.5, 2.0]])

    grad_weight_bf = _sum_over_sequence_axis(
        grad_batch_first * normalized_batch_first,
        layout="runtime_batch_size_inferred",
        current_batch_size=2,
        target="ln_post_bf",
    )
    grad_weight_sf = _sum_over_sequence_axis(
        grad_batch_first.permute(1, 0, 2) * normalized_batch_first.permute(1, 0, 2),
        layout="runtime_batch_size_inferred",
        current_batch_size=2,
        target="ln_post_sf",
    )

    assert torch.allclose(grad_weight_bf, grad_weight_sf)
    assert torch.allclose(grad_weight_bf.T @ grad_weight_bf, grad_weight_sf.T @ grad_weight_sf)


def test_cls_only_reduction_matches_between_layouts_and_rejects_ambiguous_batch_size() -> None:
    batch_first = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) / 10.0
    sequence_first = batch_first.permute(1, 0, 2)

    pooled_bf = _cls_only_from_sequence(
        batch_first,
        layout="runtime_batch_size_inferred",
        current_batch_size=2,
        target="ln_post_bf",
    )
    pooled_sf = _cls_only_from_sequence(
        sequence_first,
        layout="runtime_batch_size_inferred",
        current_batch_size=2,
        target="ln_post_sf",
    )

    assert torch.allclose(pooled_bf, pooled_sf)
    assert torch.allclose(_cls_only_from_sequence(pooled_bf, layout="non_sequence", current_batch_size=2, target="ln_post_2d"), pooled_bf)

    ambiguous = torch.randn(2, 2, 4)
    try:
        _cls_only_from_sequence(
            ambiguous,
            layout="runtime_batch_size_inferred",
            current_batch_size=2,
            target="ambiguous_ln_post",
        )
    except ValueError as exc:
        assert "Ambiguous KFAC layout" in str(exc)
    else:
        raise AssertionError("Expected ambiguous batch/sequence layout to raise ValueError")


def test_ln_post_cls_only_full_block_ignores_patch_tokens_even_with_nonzero_patch_gradients() -> None:
    inputs_a = torch.tensor(
        [
            [[1.0, 2.0, -1.0, 0.5], [8.0, 9.0, 10.0, 11.0], [12.0, 13.0, 14.0, 15.0]],
            [[-1.0, 0.0, 1.0, 2.0], [3.0, 4.0, 5.0, 6.0], [7.0, 8.0, 9.0, 10.0]],
        ],
        dtype=torch.float32,
    )
    grad_a = torch.tensor(
        [
            [[0.5, -1.0, 1.5, -2.0], [4.0, 5.0, 6.0, 7.0], [8.0, 9.0, 10.0, 11.0]],
            [[-0.25, 0.75, -1.25, 1.5], [2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0]],
        ],
        dtype=torch.float32,
    )
    inputs_b = inputs_a.clone()
    grad_b = grad_a.clone()
    inputs_b[:, 1:, :] = inputs_b[:, 1:, :] * -3.0
    grad_b[:, 1:, :] = grad_b[:, 1:, :] * 2.5

    pooled_inputs_a = _cls_only_from_sequence(
        inputs_a,
        layout="runtime_batch_size_inferred",
        current_batch_size=2,
        target="ln_post_inputs_a",
    )
    pooled_inputs_b = _cls_only_from_sequence(
        inputs_b,
        layout="runtime_batch_size_inferred",
        current_batch_size=2,
        target="ln_post_inputs_b",
    )
    pooled_grad_a = _cls_only_from_sequence(
        grad_a,
        layout="runtime_batch_size_inferred",
        current_batch_size=2,
        target="ln_post_grad_a",
    )
    pooled_grad_b = _cls_only_from_sequence(
        grad_b,
        layout="runtime_batch_size_inferred",
        current_batch_size=2,
        target="ln_post_grad_b",
    )

    norm_a = torch.nn.functional.layer_norm(pooled_inputs_a, (pooled_inputs_a.shape[-1],))
    norm_b = torch.nn.functional.layer_norm(pooled_inputs_b, (pooled_inputs_b.shape[-1],))
    fft_weight_a = (pooled_grad_a * norm_a).T @ (pooled_grad_a * norm_a)
    fft_weight_b = (pooled_grad_b * norm_b).T @ (pooled_grad_b * norm_b)
    fft_bias_a = pooled_grad_a.T @ pooled_grad_a
    fft_bias_b = pooled_grad_b.T @ pooled_grad_b

    assert torch.allclose(pooled_inputs_a, pooled_inputs_b)
    assert torch.allclose(pooled_grad_a, pooled_grad_b)
    assert torch.allclose(fft_weight_a, fft_weight_b)
    assert torch.allclose(fft_bias_a, fft_bias_b)


def test_collect_curvature_matches_between_ported_batch_first_and_mammoth_style_visuals() -> None:
    torch.manual_seed(3)
    batch_first_visual = _ToyBatchFirstVisual()
    mammoth_visual = _ToyMammothStyleVisual()
    _copy_batch_first_to_mammoth(batch_first_visual, mammoth_visual)

    ensure_openclip_kfac_surface(batch_first_visual)
    ensure_openclip_kfac_surface(mammoth_visual)

    data = [(torch.randn(4), torch.tensor(0)) for _ in range(8)]
    loader = torch.utils.data.DataLoader(data, batch_size=2, shuffle=False)
    cfg = KfacGgnConfig(train_percent=1.0, fisher_seed=19, fisher_num_samples_expectation=1)

    stats_batch_first = collect_curvature(
        batch_first_visual,
        loader,
        tracked_params=select_tracked_parameters(batch_first_visual),
        config=cfg,
    )
    stats_mammoth = collect_curvature(
        mammoth_visual,
        loader,
        tracked_params=select_tracked_parameters(mammoth_visual),
        config=cfg,
    )

    assert set(stats_batch_first.aaT) == set(stats_mammoth.aaT)
    assert set(stats_batch_first.ggT) == set(stats_mammoth.ggT)
    assert set(stats_batch_first.ffT) == set(stats_mammoth.ffT)
    for key in stats_batch_first.aaT:
        assert torch.allclose(stats_batch_first.aaT[key], stats_mammoth.aaT[key], atol=1e-5, rtol=1e-4), key
        assert torch.allclose(stats_batch_first.ggT[key], stats_mammoth.ggT[key], atol=1e-5, rtol=1e-4), key
    for key in stats_batch_first.ffT:
        assert torch.allclose(stats_batch_first.ffT[key], stats_mammoth.ffT[key], atol=1e-5, rtol=1e-4), key
    assert torch.allclose(
        stats_batch_first.ffT["visual.ln_post.weight"],
        stats_mammoth.ffT["visual.ln_post.weight"],
        atol=1e-5,
        rtol=1e-4,
    )
    assert torch.allclose(
        stats_batch_first.ffT["visual.ln_post.bias"],
        stats_mammoth.ffT["visual.ln_post.bias"],
        atol=1e-5,
        rtol=1e-4,
    )


def test_cache_roundtrip_and_metadata_compatibility() -> None:
    classifier = _build_classifier()
    model = ImageEncoder(classifier)
    ensure_openclip_kfac_surface(model)
    config = KfacGgnConfig()
    build_cfg = OpenClipBuildConfig(model_name="ViT-B-32", pretrained="openai", device="cpu", dtype="fp32")
    regularizer = KfacGgnRegularizer()
    plan, expected_meta = regularizer._expected_cache_metadata(  # type: ignore[attr-defined]
        model=model,
        task="Cars",
        build_cfg=build_cfg,
        config=config,
        attn_patch_cfg=getattr(model, "peft_attn_patch_cfg", None),
    )
    aaT, ggT, ffT = _build_identity_curvature(plan)
    with TemporaryDirectory() as tmpdir:
        path = task_cache_path(cache_dir=tmpdir, build_cfg=build_cfg, task="Cars")
        save_task_curvature(
            path,
            TaskCurvatureStats(
                aaT=aaT,
                ggT=ggT,
                ffT=ffT,
                num_examples_aaT=8,
                num_examples_ggT=8,
                metadata=expected_meta,
            ),
        )
        loaded = load_task_curvature(path, device="cpu", precision="fp32")
        assert metadata_compatible(loaded.metadata, expected_meta)
        assert loaded.metadata is not None
        assert "visual.conv1.weight" in loaded.metadata["ignored_trainable"]
        assert "visual.positional_embedding" in loaded.metadata["ignored_trainable"]
        assert loaded.metadata["layout_policy"] == "batch_size_inferred_v1"
        assert loaded.metadata["ln_post_full_block_policy"] == "cls_only_v1"


def test_kfac_cache_recomputes_when_metadata_uses_old_layout_policy() -> None:
    classifier = _build_classifier()
    model = ImageEncoder(classifier)
    regularizer = KfacGgnRegularizer()
    build_cfg = OpenClipBuildConfig(model_name="ViT-B-32", pretrained="openai", device="cpu", dtype="fp32")
    surface = ensure_openclip_kfac_surface(model)
    loader = torch.utils.data.DataLoader([(torch.randn(4), torch.tensor(0)) for _ in range(4)], batch_size=2)
    plan, expected_meta = regularizer._expected_cache_metadata(  # type: ignore[attr-defined]
        model=model,
        task="Cars",
        build_cfg=build_cfg,
        config=KfacGgnConfig(),
        attn_patch_cfg=surface["attn_patch_cfg"],
    )
    aaT, ggT, ffT = _build_identity_curvature(plan)
    stale_meta = dict(expected_meta)
    stale_meta["layout_policy"] = "static_layout_v0"
    with TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        path = task_cache_path(cache_dir=cache_dir, build_cfg=build_cfg, task="Cars")
        save_task_curvature(
            path,
            TaskCurvatureStats(
                aaT=aaT,
                ggT=ggT,
                ffT=ffT,
                num_examples_aaT=8,
                num_examples_ggT=8,
                metadata=stale_meta,
            ),
        )
        with patch(
            "merge_and_rebase.finetune.regularizers.kfac_ggn.collect_curvature",
            return_value=TaskCurvatureStats(
                aaT=aaT,
                ggT=ggT,
                ffT=ffT,
                num_examples_aaT=2,
                num_examples_ggT=2,
                metadata=expected_meta,
            ),
        ) as collect_mock:
            _, recomputed = regularizer._ensure_cache(  # type: ignore[attr-defined]
                model=model,
                loader=loader,
                task="Cars",
                build_cfg=build_cfg,
                config=KfacGgnConfig(cache_dir=cache_dir),
                attn_patch_cfg=surface["attn_patch_cfg"],
                device=torch.device("cpu"),
            )
        assert recomputed is True
        assert collect_mock.call_count == 1


def test_kfac_cache_recomputes_when_metadata_uses_old_ln_post_policy() -> None:
    classifier = _build_classifier()
    model = ImageEncoder(classifier)
    regularizer = KfacGgnRegularizer()
    build_cfg = OpenClipBuildConfig(model_name="ViT-B-32", pretrained="openai", device="cpu", dtype="fp32")
    surface = ensure_openclip_kfac_surface(model)
    loader = torch.utils.data.DataLoader([(torch.randn(4), torch.tensor(0)) for _ in range(4)], batch_size=2)
    plan, expected_meta = regularizer._expected_cache_metadata(  # type: ignore[attr-defined]
        model=model,
        task="Cars",
        build_cfg=build_cfg,
        config=KfacGgnConfig(),
        attn_patch_cfg=surface["attn_patch_cfg"],
    )
    aaT, ggT, ffT = _build_identity_curvature(plan)
    stale_meta = dict(expected_meta)
    stale_meta.pop("ln_post_full_block_policy")
    with TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        path = task_cache_path(cache_dir=cache_dir, build_cfg=build_cfg, task="Cars")
        save_task_curvature(
            path,
            TaskCurvatureStats(
                aaT=aaT,
                ggT=ggT,
                ffT=ffT,
                num_examples_aaT=8,
                num_examples_ggT=8,
                metadata=stale_meta,
            ),
        )
        with patch(
            "merge_and_rebase.finetune.regularizers.kfac_ggn.collect_curvature",
            return_value=TaskCurvatureStats(
                aaT=aaT,
                ggT=ggT,
                ffT=ffT,
                num_examples_aaT=2,
                num_examples_ggT=2,
                metadata=expected_meta,
            ),
        ) as collect_mock:
            _, recomputed = regularizer._ensure_cache(  # type: ignore[attr-defined]
                model=model,
                loader=loader,
                task="Cars",
                build_cfg=build_cfg,
                config=KfacGgnConfig(cache_dir=cache_dir),
                attn_patch_cfg=surface["attn_patch_cfg"],
                device=torch.device("cpu"),
            )
        assert recomputed is True
        assert collect_mock.call_count == 1


def test_kfac_cache_reuses_when_sampling_and_precision_metadata_differs() -> None:
    classifier = _build_classifier()
    model = ImageEncoder(classifier)
    regularizer = KfacGgnRegularizer()
    build_cfg = OpenClipBuildConfig(model_name="ViT-B-32", pretrained="openai", device="cpu", dtype="fp32")
    surface = ensure_openclip_kfac_surface(model)
    loader = torch.utils.data.DataLoader([(torch.randn(4), torch.tensor(0)) for _ in range(4)], batch_size=2)
    plan, cached_meta = regularizer._expected_cache_metadata(  # type: ignore[attr-defined]
        model=model,
        task="Cars",
        build_cfg=build_cfg,
        config=KfacGgnConfig(),
        attn_patch_cfg=surface["attn_patch_cfg"],
    )
    aaT, ggT, ffT = _build_identity_curvature(plan)
    with TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        path = task_cache_path(cache_dir=cache_dir, build_cfg=build_cfg, task="Cars")
        save_task_curvature(
            path,
            TaskCurvatureStats(
                aaT=aaT,
                ggT=ggT,
                ffT=ffT,
                num_examples_aaT=8,
                num_examples_ggT=8,
                metadata=cached_meta,
            ),
        )
        with patch("merge_and_rebase.finetune.regularizers.kfac_ggn.collect_curvature") as collect_mock:
            _, recomputed = regularizer._ensure_cache(  # type: ignore[attr-defined]
                model=model,
                loader=loader,
                task="Cars",
                build_cfg=build_cfg,
                config=KfacGgnConfig(
                    cache_dir=cache_dir,
                    precision="fp64",
                    train_percent=0.5,
                    fisher_seed=7,
                    fisher_num_samples_expectation=3,
                ),
                attn_patch_cfg=surface["attn_patch_cfg"],
                device=torch.device("cpu"),
            )
        assert recomputed is False
        assert collect_mock.call_count == 0


def test_task_cache_path_separates_openclip_and_openai_clip_loaders() -> None:
    openclip_cfg = OpenClipBuildConfig(
        loader="openclip",
        model_name="ViT-B-32",
        pretrained="openai",
        device="cpu",
        dtype="fp32",
    )
    openai_cfg = OpenClipBuildConfig(
        loader="openai_clip",
        model_name="ViT-B-32",
        pretrained="openai",
        device="cpu",
        dtype="fp32",
    )

    openclip_path = task_cache_path(cache_dir="tmp/kfac", build_cfg=openclip_cfg, task="Cars")
    openai_path = task_cache_path(cache_dir="tmp/kfac", build_cfg=openai_cfg, task="Cars")

    assert openclip_path != openai_path
    assert openclip_path == Path("tmp/kfac/openclip/ViT-B-32/openai/Cars/curvature.pt")
    assert openai_path == Path("tmp/kfac/openai_clip/ViT-B-32/openai/Cars/curvature.pt")


def test_kfac_regularizer_zero_at_base_and_positive_after_supported_update() -> None:
    torch.manual_seed(0)
    classifier = _build_classifier()
    model = ImageEncoder(classifier)
    device = torch.device("cpu")
    build_cfg = OpenClipBuildConfig(model_name="ViT-B-32", pretrained="openai", device="cpu", dtype="fp32")
    regularizer = KfacGgnRegularizer()
    train_loader = torch.utils.data.DataLoader([(torch.randn(4), torch.tensor(0)) for _ in range(4)], batch_size=2)
    loaders = SimpleNamespace(train=train_loader)
    surface = ensure_openclip_kfac_surface(model)
    plan = select_tracked_parameters(model)
    aaT, ggT, ffT = _build_identity_curvature(plan)
    with TemporaryDirectory() as tmpdir:
        cache_path = task_cache_path(cache_dir=tmpdir, build_cfg=build_cfg, task="task_b")
        save_task_curvature(
            cache_path,
            TaskCurvatureStats(
                aaT=aaT,
                ggT=ggT,
                ffT=ffT,
                num_examples_aaT=8,
                num_examples_ggT=8,
                metadata={"source": "test"},
            ),
        )
        with patch(
            "merge_and_rebase.finetune.regularizers.kfac_ggn.build_vision_regularizer_task_context",
            return_value=SimpleNamespace(task="task_b", build_cfg=build_cfg, loader=train_loader, model=model),
        ):
            def _fake_ensure_cache(self, *, task, **kwargs):
                return task_cache_path(cache_dir=tmpdir, build_cfg=build_cfg, task=str(task)), False

            with patch.object(KfacGgnRegularizer, "_ensure_cache", new=_fake_ensure_cache):
                prepared, info = regularizer.prepare(
                    model=model,
                    device=device,
                    regularization_cfg={"cache_dir": tmpdir, "reg_lambda": 1.0, "cadence": 1},
                    task="task_a",
                    build_cfg=build_cfg,
                    loaders=loaders,
                    strategy_cfg={},
                    reference_tasks=["task_b"],
                    batch_size=2,
                    num_workers=0,
                    val_fraction=0.1,
                    seed=42,
                )
        zero_loss = regularizer.apply(prepared, model=model, step=0, batch_index=0)
        assert float(zero_loss.detach()) == 0.0
        first_key = next(iter(plan.matrix_blocks))
        local_name = first_key[len("visual.") :]
        dict(model.clip_model.model.visual.named_parameters())[local_name].data.add_(0.25)
        moved_loss = regularizer.apply(prepared, model=model, step=0, batch_index=0)
        assert float(moved_loss.detach()) > 0.0
        assert info["kfac_reference_tasks"] == 1
        assert info["kfac_ignored_trainable"] == len(plan.ignored_trainable)


def test_kfac_regularizer_auto_excludes_current_task_from_reference_tasks() -> None:
    classifier = _build_classifier()
    model = ImageEncoder(classifier)
    device = torch.device("cpu")
    build_cfg = OpenClipBuildConfig(model_name="ViT-B-32", pretrained="openai", device="cpu", dtype="fp32")
    regularizer = KfacGgnRegularizer()
    train_loader = torch.utils.data.DataLoader([(torch.randn(4), torch.tensor(0)) for _ in range(4)], batch_size=2)
    loaders = SimpleNamespace(train=train_loader)
    with TemporaryDirectory() as tmpdir:
        def _fake_ensure_cache(self, *, task, **kwargs):
            return task_cache_path(cache_dir=tmpdir, build_cfg=build_cfg, task=str(task)), False

        with patch.object(KfacGgnRegularizer, "_ensure_cache", new=_fake_ensure_cache):
            prepared, info = regularizer.prepare(
                model=model,
                device=device,
                regularization_cfg={"cache_dir": tmpdir, "reg_lambda": 1.0, "cadence": 1},
                task="task_a",
                build_cfg=build_cfg,
                loaders=loaders,
                strategy_cfg={},
                reference_tasks=["task_a"],
                batch_size=2,
                num_workers=0,
                val_fraction=0.1,
                seed=42,
            )
    loss = regularizer.apply(prepared, model=model, step=0, batch_index=0)
    assert float(loss.detach()) == 0.0
    assert info["kfac_reference_tasks"] == 0


def test_kfac_regularizer_prefers_its_own_reference_config_over_inherited_reference_tasks() -> None:
    classifier = _build_classifier()
    model = ImageEncoder(classifier)
    device = torch.device("cpu")
    build_cfg = OpenClipBuildConfig(model_name="ViT-B-32", pretrained="openai", device="cpu", dtype="fp32")
    regularizer = KfacGgnRegularizer()
    train_loader = torch.utils.data.DataLoader([(torch.randn(4), torch.tensor(0)) for _ in range(4)], batch_size=2)
    loaders = SimpleNamespace(train=train_loader)
    plan = select_tracked_parameters(model)
    aaT, ggT, ffT = _build_identity_curvature(plan)
    inherited_reference_tasks = list(train_vision.SUITES["vision8"].tasks)
    context = build_reference_task_resolution_context(training_tasks=["Cars", "DTD"], suite="vision8")

    with TemporaryDirectory() as tmpdir:
        cache_path = task_cache_path(cache_dir=tmpdir, build_cfg=build_cfg, task="ImageNet1K")
        save_task_curvature(
            cache_path,
            TaskCurvatureStats(
                aaT=aaT,
                ggT=ggT,
                ffT=ffT,
                num_examples_aaT=8,
                num_examples_ggT=8,
                metadata={"source": "test"},
            ),
        )

        seen_tasks: list[str] = []

        def _fake_ensure_cache(self, *, task, **kwargs):
            seen_tasks.append(str(task))
            return task_cache_path(cache_dir=tmpdir, build_cfg=build_cfg, task=str(task)), False

        with patch(
            "merge_and_rebase.finetune.regularizers.kfac_ggn.build_vision_regularizer_task_context",
            return_value=SimpleNamespace(task="ImageNet1K", build_cfg=build_cfg, loader=train_loader, model=model),
        ):
            with patch.object(KfacGgnRegularizer, "_ensure_cache", new=_fake_ensure_cache):
                prepared, info = regularizer.prepare(
                    model=model,
                    device=device,
                    regularization_cfg={
                        "cache_dir": tmpdir,
                        "reg_lambda": 1.0,
                        "cadence": 1,
                        "reference_datasets": ["ImageNet1K"],
                    },
                    task="Cars",
                    build_cfg=build_cfg,
                    loaders=loaders,
                    strategy_cfg={},
                    reference_tasks=inherited_reference_tasks,
                    reference_resolution_context=context,
                    batch_size=2,
                    num_workers=0,
                    val_fraction=0.1,
                    seed=42,
                )

        assert seen_tasks == ["ImageNet1K"]
        assert prepared.aggregated.reference_tasks == ["ImageNet1K"]
        assert info["kfac_reference_tasks"] == 1


def test_train_task_passes_regularizer_context(monkeypatch, tmp_path: Path) -> None:
    class _ContextRegularizer:
        def __init__(self) -> None:
            self.prepare_kwargs = None
            self.apply_calls: list[dict[str, object]] = []

        def prepare(self, **kwargs):
            self.prepare_kwargs = dict(kwargs)
            return {"prepared": True}, {"kfac_reference_tasks": 0}

        def apply(self, prepared, **kwargs):
            self.apply_calls.append({"prepared": prepared, **kwargs})
            model = kwargs["model"]
            return next(model.parameters()).sum() * 0.0

    regularizer = _ContextRegularizer()
    loaders = SimpleNamespace(
        train=[(torch.randn(2, 3, 2, 2), torch.tensor([0, 1]))],
        val=[(torch.randn(2, 3, 2, 2), torch.tensor([0, 1]))],
        test=[(torch.randn(2, 3, 2, 2), torch.tensor([0, 1]))],
        classnames=["a", "b"],
    )

    monkeypatch.setattr(train_vision, "load_hf_splits", lambda *args, **kwargs: {})
    monkeypatch.setattr(train_vision.OpenClipClassifier, "build", staticmethod(lambda cfg: _TrainDummyClassifier()))
    monkeypatch.setattr(train_vision, "build_vision_loaders", lambda **kwargs: loaders)
    monkeypatch.setattr(train_vision, "get_strategy", lambda name: _DummyStrategy())
    monkeypatch.setattr(train_vision, "get_regularizer", lambda name: regularizer)

    build_cfg = OpenClipBuildConfig(model_name="ViT-B-32", pretrained="openai", device="cpu", dtype="fp32")
    reference_resolution_context = build_reference_task_resolution_context(
        training_tasks=["Cars", "DTD"],
        suite="vision8",
    )
    train_vision.train_task(
        task="Cars",
        hf_path="dummy",
        hf_config=None,
        split_map={"train": "train", "test": "test"},
        build_cfg=build_cfg,
        strategy="full",
        epochs=1,
        lr=1e-3,
        weight_decay=0.0,
        warmup_length=1,
        clip_grad_norm=-1.0,
        accumulate_grad_batches=1,
        batch_size=2,
        num_workers=0,
        val_fraction=0.1,
        early_stopping=False,
        early_stopping_patience=1,
        text_only=False,
        seed=42,
        device="cpu",
        out_dir=tmp_path,
        save_format="full",
        save_checkpoints=False,
        save_last_epoch=False,
        strategy_cfg={},
        regularization_cfg={"name": "kfac_ggn"},
        all_tasks=["Cars", "DTD"],
        reference_tasks=["DTD", "SVHN"],
        reference_resolution_context=reference_resolution_context,
    )

    assert regularizer.prepare_kwargs is not None
    assert regularizer.prepare_kwargs["build_cfg"] == build_cfg
    assert regularizer.prepare_kwargs["loaders"] is loaders
    assert regularizer.prepare_kwargs["all_tasks"] == ["Cars", "DTD"]
    assert regularizer.prepare_kwargs["reference_tasks"] == ["DTD", "SVHN"]
    assert regularizer.prepare_kwargs["reference_resolution_context"] == reference_resolution_context
    assert regularizer.prepare_kwargs["batch_size"] == 2
    assert len(regularizer.apply_calls) >= 1
    assert regularizer.apply_calls[0]["prepared"] == {"prepared": True}


def test_resolve_reference_tasks_defaults_to_suite_when_available() -> None:
    args = SimpleNamespace(
        datasets=None,
        suite="vision8",
        reference_suite=None,
        reference_datasets=None,
    )
    refs, explicit = train_vision.resolve_reference_tasks(
        args,
        training_tasks=["Cars", "DTD"],
        regularization_cfg={"name": "kfac_ggn"},
        require_reference=True,
    )
    assert refs == list(train_vision.SUITES["vision8"].tasks)
    assert explicit is False


def test_resolve_reference_tasks_defaults_to_training_list_without_suite() -> None:
    args = SimpleNamespace(
        datasets="Cars,DTD",
        suite=None,
        reference_suite=None,
        reference_datasets=None,
    )
    refs, explicit = train_vision.resolve_reference_tasks(
        args,
        training_tasks=["Cars", "DTD"],
        regularization_cfg={"name": "kfac_ggn"},
        require_reference=True,
    )
    assert refs == ["Cars", "DTD"]
    assert explicit is False


def test_resolve_reference_tasks_prefers_explicit_overrides_and_requires_single_task_reference() -> None:
    args = SimpleNamespace(
        datasets="Cars",
        suite=None,
        reference_suite=None,
        reference_datasets="SVHN,DTD",
    )
    refs, explicit = train_vision.resolve_reference_tasks(
        args,
        training_tasks=["Cars"],
        regularization_cfg={"name": "kfac_ggn", "reference_suite": "vision8"},
        require_reference=True,
    )
    assert refs == ["SVHN", "DTD"]
    assert explicit is True

    missing_args = SimpleNamespace(
        datasets="Cars",
        suite=None,
        reference_suite=None,
        reference_datasets=None,
    )
    try:
        train_vision.resolve_reference_tasks(
            missing_args,
            training_tasks=["Cars"],
            regularization_cfg={"name": "kfac_ggn"},
            require_reference=True,
        )
    except ValueError as exc:
        assert "explicit regularization dataset" in str(exc)
    else:
        raise AssertionError("Expected single-task regularization resolution to fail without explicit reference datasets.")


def test_resolve_reference_tasks_accepts_imagenet1k_as_reference_dataset() -> None:
    args = SimpleNamespace(
        datasets="Cars,DTD",
        suite="vision8",
        reference_suite=None,
        reference_datasets="ImageNet1K",
    )
    refs, explicit = train_vision.resolve_reference_tasks(
        args,
        training_tasks=["Cars", "DTD"],
        regularization_cfg={"name": "kfac_ggn"},
        require_reference=True,
    )
    assert refs == ["ImageNet1K"]
    assert explicit is True
