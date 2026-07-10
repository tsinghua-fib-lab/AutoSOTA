from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

from merge_and_rebase.finetune.regularizers.ekfac_ggn import (
    EkfacGgnConfig,
    EkfacGgnRegularizer,
    EkfacTaskCurvatureStats,
    _per_example_grad_weight,
    _related_kfac_cache_dir,
    collect_ekfac_curvature,
    compute_ekfac_penalty,
    load_task_ekfac,
    save_task_ekfac,
)
from merge_and_rebase.finetune.regularizers.kfac_ggn import (
    KfacGgnConfig,
    TaskCurvatureStats,
    _metadata as _kfac_metadata,
    collect_curvature,
    ensure_openclip_kfac_surface,
    load_task_curvature,
    metadata_compatible,
    save_task_curvature,
    select_tracked_parameters,
    task_cache_path,
    task_cache_path as kfac_task_cache_path,
)
from merge_and_rebase.finetune.regularizers.registry import list_regularizers
from merge_and_rebase.finetune.train_vision import ImageEncoder
from merge_and_rebase.models.openclip_classifier import OpenClipBuildConfig



def _dummy_transform(x):
    return x


class _ToyMLP(torch.nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.c_fc = torch.nn.Linear(width, 2 * width)
        self.c_proj = torch.nn.Linear(2 * width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(torch.relu(self.c_fc(x)))


class _ToyOpenClipBlock(torch.nn.Module):
    def __init__(self, width: int, *, batch_first: bool = False) -> None:
        super().__init__()
        self.batch_first = batch_first
        self.ln_1 = torch.nn.LayerNorm(width)
        self.attn = torch.nn.MultiheadAttention(width, num_heads=1, batch_first=batch_first)
        self.ln_2 = torch.nn.LayerNorm(width)
        self.mlp = _ToyMLP(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_in = self.ln_1(x)
        attn_out = self.attn(attn_in, attn_in, attn_in, need_weights=False)[0]
        x = x + attn_out
        return x + self.mlp(self.ln_2(x))


class _ToyOpenClipVisual(torch.nn.Module):
    def __init__(self, width: int = 4, out_dim: int = 3) -> None:
        super().__init__()
        self.class_embedding = torch.nn.Parameter(torch.randn(width) * 0.01)
        self.positional_embedding = torch.nn.Parameter(torch.randn(3, width) * 0.01)
        self.conv1 = torch.nn.Linear(width, width, bias=False)
        self.ln_pre = torch.nn.LayerNorm(width)
        self.transformer = torch.nn.Module()
        self.transformer.resblocks = torch.nn.ModuleList([_ToyOpenClipBlock(width, batch_first=False)])
        self.ln_post = torch.nn.LayerNorm(width)
        self.proj = torch.nn.Parameter(torch.randn(width, out_dim) * 0.02)

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


class _ToyBatchFirstVisual(torch.nn.Module):
    def __init__(self, width: int = 4, out_dim: int = 3) -> None:
        super().__init__()
        self.class_embedding = torch.nn.Parameter(torch.randn(width) * 0.01)
        self.positional_embedding = torch.nn.Parameter(torch.randn(3, width) * 0.01)
        self.conv1 = torch.nn.Linear(width, width, bias=False)
        self.ln_pre = torch.nn.LayerNorm(width)
        self.transformer = torch.nn.Module()
        self.transformer.resblocks = torch.nn.ModuleList([_ToyOpenClipBlock(width, batch_first=True)])
        self.ln_post = torch.nn.LayerNorm(width)
        self.proj = torch.nn.Parameter(torch.randn(width, out_dim) * 0.02)
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


class _ToyMammothStyleVisual(torch.nn.Module):
    def __init__(self, width: int = 4, out_dim: int = 3) -> None:
        super().__init__()
        self.class_embedding = torch.nn.Parameter(torch.randn(width) * 0.01)
        self.positional_embedding = torch.nn.Parameter(torch.randn(3, width) * 0.01)
        self.conv1 = torch.nn.Linear(width, width, bias=False)
        self.cls_token_layer = _ToyClsTokenLayer()
        self.ln_pre = torch.nn.LayerNorm(width)
        self.transformer = torch.nn.Module()
        self.transformer.resblocks = torch.nn.ModuleList([_ToyOpenClipBlock(width, batch_first=False)])
        self.ln_post = torch.nn.LayerNorm(width)
        self.lin_proj = torch.nn.Linear(width, out_dim, bias=False)

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


class _ToyBackbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.visual = _ToyOpenClipVisual()
        self.transformer = torch.nn.Identity()


class _TrainDummyClipModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = torch.nn.Linear(1, 1)
        self.visual = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(12, 2))


class _TrainDummyClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _TrainDummyClipModel()
        self.preprocess = object()
        self.normalize = True
        self.logit_scale = 1.0
        self.register_buffer('_zs_text_features', torch.empty(0), persistent=False)

    def build_zeroshot_text_features(self, classnames, build_cfg):
        del build_cfg
        self._zs_text_features = torch.eye(len(classnames), dtype=torch.float32)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        image_features = self.model.visual(images)
        image_features = torch.nn.functional.normalize(image_features, dim=-1)
        return self.logit_scale * (image_features @ self._zs_text_features.T)



def _build_classifier(num_classes: int = 3):
    from merge_and_rebase.models.openclip_classifier import OpenClipClassifier
    classifier = OpenClipClassifier(
        model=_ToyBackbone(),
        tokenizer=lambda texts: torch.zeros(len(texts), 1, dtype=torch.long),
        preprocess=_dummy_transform,
        normalize=True,
        logit_scale=3.5,
    )
    classifier._zs_text_features = torch.eye(num_classes, dtype=torch.float32)
    return classifier


def _build_identity_ekfac(plan):
    UA = {}
    UG = {}
    D = {}
    ffT = {}
    for key, block in plan.matrix_blocks.items():
        shape = plan.param_shapes[key]
        if block.is_projection and key == 'visual.proj':
            rows = shape[1]
            cols = shape[0]
        else:
            rows = shape[0]
            cols = shape[1]
        cols = cols + (1 if block.bias_key is not None else 0)
        UA[key] = torch.eye(cols)
        UG[key] = torch.eye(rows)
        D[key] = torch.ones(rows, cols)
    for key in plan.full_blocks:
        rows = plan.param_shapes[key][0]
        ffT[key] = torch.eye(rows)
    return UA, UG, D, ffT


def test_regularizer_registry_exposes_ekfac_ggn() -> None:
    assert 'ekfac_ggn' in list_regularizers()


def test_collect_ekfac_curvature_respects_train_percent_and_seed() -> None:
    torch.manual_seed(0)
    visual = _ToyOpenClipVisual()
    ensure_openclip_kfac_surface(visual)
    data = [(torch.randn(4), torch.tensor(0)) for _ in range(12)]
    loader = torch.utils.data.DataLoader(data, batch_size=2, shuffle=False)
    cfg = EkfacGgnConfig(train_percent=0.5, fisher_seed=17, fisher_num_samples_expectation=1)
    stats_a = collect_ekfac_curvature(visual, loader, tracked_params=select_tracked_parameters(visual), config=cfg)
    stats_b = collect_ekfac_curvature(visual, loader, tracked_params=select_tracked_parameters(visual), config=cfg)
    assert stats_a.num_examples == 6
    assert set(stats_a.UA) == set(stats_b.UA)
    assert set(stats_a.D) == set(stats_b.D)
    for key in stats_a.UA:
        assert torch.allclose(stats_a.UA[key], stats_b.UA[key])
        assert torch.allclose(stats_a.UG[key], stats_b.UG[key])
        assert torch.allclose(stats_a.D[key], stats_b.D[key])


def test_per_example_grad_weight_matches_between_batch_first_and_sequence_first_layouts() -> None:
    inputs_batch_first = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) / 10.0
    grad_batch_first = torch.arange(18, dtype=torch.float32).reshape(2, 3, 3) / 7.0

    grad_weight_bf = _per_example_grad_weight(
        inputs=inputs_batch_first,
        grad_output=grad_batch_first,
        layout="runtime_batch_size_inferred",
        current_batch_size=2,
        target="bf",
        include_bias=True,
    )
    grad_weight_sf = _per_example_grad_weight(
        inputs=inputs_batch_first.permute(1, 0, 2),
        grad_output=grad_batch_first.permute(1, 0, 2),
        layout="runtime_batch_size_inferred",
        current_batch_size=2,
        target="sf",
        include_bias=True,
    )

    assert torch.allclose(grad_weight_bf, grad_weight_sf)


def test_collect_ekfac_curvature_matches_between_ported_batch_first_and_mammoth_style_visuals() -> None:
    torch.manual_seed(5)
    batch_first_visual = _ToyBatchFirstVisual()
    mammoth_visual = _ToyMammothStyleVisual()
    _copy_batch_first_to_mammoth(batch_first_visual, mammoth_visual)

    ensure_openclip_kfac_surface(batch_first_visual)
    ensure_openclip_kfac_surface(mammoth_visual)

    data = [(torch.randn(4), torch.tensor(0)) for _ in range(8)]
    loader = torch.utils.data.DataLoader(data, batch_size=2, shuffle=False)
    cfg = EkfacGgnConfig(train_percent=1.0, fisher_seed=29, fisher_num_samples_expectation=1)

    stats_batch_first = collect_ekfac_curvature(
        batch_first_visual,
        loader,
        tracked_params=select_tracked_parameters(batch_first_visual),
        config=cfg,
    )
    stats_mammoth = collect_ekfac_curvature(
        mammoth_visual,
        loader,
        tracked_params=select_tracked_parameters(mammoth_visual),
        config=cfg,
    )

    assert set(stats_batch_first.D) == set(stats_mammoth.D)
    assert set(stats_batch_first.ffT) == set(stats_mammoth.ffT)
    for key in stats_batch_first.D:
        assert torch.allclose(stats_batch_first.D[key], stats_mammoth.D[key], atol=1e-5, rtol=1e-4), key
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


def test_collect_ekfac_curvature_reuses_kfac_ln_post_ffT() -> None:
    torch.manual_seed(11)
    visual = _ToyBatchFirstVisual()
    ensure_openclip_kfac_surface(visual)
    data = [(torch.randn(4), torch.tensor(0)) for _ in range(8)]
    loader = torch.utils.data.DataLoader(data, batch_size=2, shuffle=False)
    kfac_cfg = KfacGgnConfig(train_percent=1.0, fisher_seed=31, fisher_num_samples_expectation=1)
    base_stats = collect_curvature(
        visual,
        loader,
        tracked_params=select_tracked_parameters(visual),
        config=kfac_cfg,
    )
    ekfac_stats = collect_ekfac_curvature(
        visual,
        loader,
        tracked_params=select_tracked_parameters(visual),
        config=EkfacGgnConfig(train_percent=1.0, fisher_seed=31, fisher_num_samples_expectation=1),
        base_stats=base_stats,
    )
    assert torch.allclose(ekfac_stats.ffT["visual.ln_post.weight"], base_stats.ffT["visual.ln_post.weight"])
    assert torch.allclose(ekfac_stats.ffT["visual.ln_post.bias"], base_stats.ffT["visual.ln_post.bias"])


def test_collect_ekfac_curvature_averages_D_over_examples() -> None:
    torch.manual_seed(1)
    plan_model = _ToyOpenClipVisual()
    ensure_openclip_kfac_surface(plan_model)
    plan = select_tracked_parameters(plan_model)
    examples = [(torch.randn(4), torch.tensor(0)) for _ in range(8)]
    loader_once = torch.utils.data.DataLoader(examples, batch_size=2, shuffle=False)
    loader_twice = torch.utils.data.DataLoader(examples + examples, batch_size=2, shuffle=False)
    cfg = EkfacGgnConfig(train_percent=1.0, fisher_seed=23, fisher_num_samples_expectation=0)

    torch.manual_seed(2)
    visual_once = _ToyOpenClipVisual()
    ensure_openclip_kfac_surface(visual_once)
    stats_once = collect_ekfac_curvature(visual_once, loader_once, tracked_params=plan, config=cfg)

    torch.manual_seed(2)
    visual_twice = _ToyOpenClipVisual()
    ensure_openclip_kfac_surface(visual_twice)
    stats_twice = collect_ekfac_curvature(visual_twice, loader_twice, tracked_params=plan, config=cfg)

    assert stats_once.num_examples * 2 == stats_twice.num_examples
    for key in stats_once.D:
        assert torch.allclose(stats_once.D[key], stats_twice.D[key], atol=1e-5, rtol=1e-4)


def test_related_kfac_cache_dir_swaps_regularizer_segment() -> None:
    assert _related_kfac_cache_dir(Path('src/checkpoints/ekfac_ggn')) == Path('src/checkpoints/kfac_ggn')
    assert _related_kfac_cache_dir(Path('src/checkpoints/foo/ekfac_ggn')) == Path('src/checkpoints/foo/kfac_ggn')


def test_ekfac_cache_path_separates_openclip_and_openai_clip_loaders() -> None:
    openclip_cfg = OpenClipBuildConfig(
        loader='openclip',
        model_name='ViT-B-32',
        pretrained='openai',
        device='cpu',
        dtype='fp32',
    )
    openai_cfg = OpenClipBuildConfig(
        loader='openai_clip',
        model_name='ViT-B-32',
        pretrained='openai',
        device='cpu',
        dtype='fp32',
    )

    openclip_path = task_cache_path(cache_dir='tmp/ekfac', build_cfg=openclip_cfg, task='Cars')
    openai_path = task_cache_path(cache_dir='tmp/ekfac', build_cfg=openai_cfg, task='Cars')

    assert openclip_path != openai_path
    assert openclip_path == Path('tmp/ekfac/openclip/ViT-B-32/openai/Cars/curvature.pt')
    assert openai_path == Path('tmp/ekfac/openai_clip/ViT-B-32/openai/Cars/curvature.pt')


def test_ekfac_uses_cached_kfac_when_available() -> None:
    torch.manual_seed(0)
    classifier = _build_classifier()
    model = ImageEncoder(classifier)
    device = torch.device('cpu')
    build_cfg = OpenClipBuildConfig(model_name='ViT-B-32', pretrained='openai', device='cpu', dtype='fp32')
    regularizer = EkfacGgnRegularizer()
    config = EkfacGgnConfig(cache_dir=Path('src/checkpoints/ekfac_ggn'))
    surface = ensure_openclip_kfac_surface(model)
    plan, meta = regularizer._expected_cache_metadata(  # type: ignore[attr-defined]
        model=model,
        task='Cars',
        build_cfg=build_cfg,
        config=config,
        attn_patch_cfg=surface['attn_patch_cfg'],
    )
    with TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        ekfac_cache_dir = tmp_path / 'ekfac_ggn'
        kfac_cache_dir = tmp_path / 'kfac_ggn'
        kfac_cfg = {
            'cache_dir': str(ekfac_cache_dir),
            'precision': 'fp32',
            'reg_lambda': 0.0,
            'full_block_scaler': 10.0,
            'projection_scaler': 0.01,
            'cadence': 1,
            'force_recompute': False,
            'train_percent': 1.0,
            'fisher_seed': None,
            'fisher_num_samples_expectation': 1,
        }
        expected_kfac_meta = _kfac_metadata(
            task='Cars',
            build_cfg=build_cfg,
            config=KfacGgnConfig(**kfac_cfg),
            plan=plan,
            attn_patch_cfg=surface['attn_patch_cfg'],
        )
        aaT = {}
        ggT = {}
        for key, block in plan.matrix_blocks.items():
            shape = plan.param_shapes[key]
            if block.is_projection and key == 'visual.proj':
                rows = shape[1]
                cols = shape[0]
            else:
                rows = shape[0]
                cols = shape[1] + (1 if block.bias_key is not None else 0)
            aaT[key] = torch.eye(cols)
            ggT[key] = torch.eye(rows)
        ffT = {key: torch.eye(plan.param_shapes[key][0]) for key in plan.full_blocks}
        save_task_curvature(
            kfac_task_cache_path(cache_dir=kfac_cache_dir, build_cfg=build_cfg, task='Cars'),
            TaskCurvatureStats(aaT=aaT, ggT=ggT, ffT=ffT, num_examples_aaT=8, num_examples_ggT=8, metadata=expected_kfac_meta),
        )
        loader = torch.utils.data.DataLoader([(torch.randn(4), torch.tensor(0)) for _ in range(4)], batch_size=2, shuffle=False)
        with patch('merge_and_rebase.finetune.regularizers.ekfac_ggn.collect_curvature', side_effect=AssertionError('should reuse cached kfac')):
            path, _ = regularizer._collect_and_store(  # type: ignore[attr-defined]
                model=model,
                loader=loader,
                task='Cars',
                build_cfg=build_cfg,
                config=EkfacGgnConfig(cache_dir=ekfac_cache_dir),
                attn_patch_cfg=surface['attn_patch_cfg'],
                device=device,
            )
        assert path.exists()
        loaded = load_task_ekfac(path, device='cpu', precision='fp32')
        assert metadata_compatible(loaded.metadata, meta)



def test_ekfac_populates_kfac_cache_when_missing() -> None:
    torch.manual_seed(0)
    classifier = _build_classifier()
    model = ImageEncoder(classifier)
    device = torch.device('cpu')
    build_cfg = OpenClipBuildConfig(model_name='ViT-B-32', pretrained='openai', device='cpu', dtype='fp32')
    regularizer = EkfacGgnRegularizer()
    surface = ensure_openclip_kfac_surface(model)
    with TemporaryDirectory() as tmpdir:
        ekfac_cache_dir = Path(tmpdir) / 'ekfac_ggn'
        ekfac_cfg = EkfacGgnConfig(cache_dir=ekfac_cache_dir, train_percent=0.5, fisher_seed=7, fisher_num_samples_expectation=0)
        loader = torch.utils.data.DataLoader([(torch.randn(4), torch.tensor(0)) for _ in range(6)], batch_size=2, shuffle=False)
        ekfac_path, _ = regularizer._collect_and_store(  # type: ignore[attr-defined]
            model=model,
            loader=loader,
            task='Cars',
            build_cfg=build_cfg,
            config=ekfac_cfg,
            attn_patch_cfg=surface['attn_patch_cfg'],
            device=device,
        )
        assert ekfac_path.exists()
        kfac_path = kfac_task_cache_path(cache_dir=_related_kfac_cache_dir(ekfac_cache_dir), build_cfg=build_cfg, task='Cars')
        assert kfac_path.exists()
        kfac_stats = load_task_curvature(kfac_path, device='cpu', precision='fp32')
        assert kfac_stats.num_examples_aaT == kfac_stats.num_examples_ggT
        assert kfac_stats.metadata is not None
        expected_kfac_meta = _kfac_metadata(
            task='Cars',
            build_cfg=build_cfg,
            config=KfacGgnConfig(
                cache_dir=_related_kfac_cache_dir(ekfac_cache_dir),
                precision=ekfac_cfg.precision,
                reg_lambda=ekfac_cfg.reg_lambda,
                full_block_scaler=ekfac_cfg.full_block_scaler,
                projection_scaler=ekfac_cfg.projection_scaler,
                cadence=ekfac_cfg.cadence,
                force_recompute=ekfac_cfg.force_recompute,
                train_percent=ekfac_cfg.train_percent,
                fisher_seed=ekfac_cfg.fisher_seed,
                fisher_num_samples_expectation=ekfac_cfg.fisher_num_samples_expectation,
            ),
            plan=select_tracked_parameters(model),
            attn_patch_cfg=surface['attn_patch_cfg'],
        )
        assert metadata_compatible(kfac_stats.metadata, expected_kfac_meta)


def test_ekfac_cache_roundtrip_and_metadata_compatibility() -> None:
    classifier = _build_classifier()
    model = ImageEncoder(classifier)
    ensure_openclip_kfac_surface(model)
    config = EkfacGgnConfig()
    build_cfg = OpenClipBuildConfig(model_name='ViT-B-32', pretrained='openai', device='cpu', dtype='fp32')
    regularizer = EkfacGgnRegularizer()
    plan, expected_meta = regularizer._expected_cache_metadata(  # type: ignore[attr-defined]
        model=model,
        task='Cars',
        build_cfg=build_cfg,
        config=config,
        attn_patch_cfg=getattr(model, 'peft_attn_patch_cfg', None),
    )
    UA, UG, D, ffT = _build_identity_ekfac(plan)
    with TemporaryDirectory() as tmpdir:
        path = task_cache_path(cache_dir=tmpdir, build_cfg=build_cfg, task='Cars')
        save_task_ekfac(
            path,
            EkfacTaskCurvatureStats(
                UA=UA,
                UG=UG,
                D=D,
                ffT=ffT,
                num_examples=8,
                metadata=expected_meta,
            ),
        )
        loaded = load_task_ekfac(path, device='cpu', precision='fp32')
        assert metadata_compatible(loaded.metadata, expected_meta)
        assert loaded.metadata is not None
        assert loaded.metadata['regularizer'] == 'ekfac_ggn'
        assert 'visual.conv1.weight' in loaded.metadata['ignored_trainable']
        assert loaded.metadata['layout_policy'] == 'batch_size_inferred_v1'
        assert loaded.metadata['ln_post_full_block_policy'] == 'cls_only_v1'


def test_ekfac_cache_recomputes_when_metadata_uses_old_layout_policy() -> None:
    classifier = _build_classifier()
    model = ImageEncoder(classifier)
    device = torch.device('cpu')
    build_cfg = OpenClipBuildConfig(model_name='ViT-B-32', pretrained='openai', device='cpu', dtype='fp32')
    regularizer = EkfacGgnRegularizer()
    surface = ensure_openclip_kfac_surface(model)
    plan, expected_meta = regularizer._expected_cache_metadata(  # type: ignore[attr-defined]
        model=model,
        task='Cars',
        build_cfg=build_cfg,
        config=EkfacGgnConfig(),
        attn_patch_cfg=surface['attn_patch_cfg'],
    )
    UA, UG, D, ffT = _build_identity_ekfac(plan)
    stale_meta = dict(expected_meta)
    stale_meta['layout_policy'] = 'static_layout_v0'
    with TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        path = task_cache_path(cache_dir=cache_dir, build_cfg=build_cfg, task='Cars')
        save_task_ekfac(
            path,
            EkfacTaskCurvatureStats(
                UA=UA,
                UG=UG,
                D=D,
                ffT=ffT,
                num_examples=8,
                metadata=stale_meta,
            ),
        )
        loader = torch.utils.data.DataLoader([(torch.randn(4), torch.tensor(0)) for _ in range(4)], batch_size=2)
        with patch(
            'merge_and_rebase.finetune.regularizers.ekfac_ggn.collect_ekfac_curvature',
            return_value=EkfacTaskCurvatureStats(
                UA=UA,
                UG=UG,
                D=D,
                ffT=ffT,
                num_examples=2,
                metadata=expected_meta,
            ),
        ) as collect_mock:
            _, recomputed = regularizer._ensure_cache(  # type: ignore[attr-defined]
                model=model,
                loader=loader,
                task='Cars',
                build_cfg=build_cfg,
                config=EkfacGgnConfig(cache_dir=cache_dir),
                attn_patch_cfg=surface['attn_patch_cfg'],
                device=device,
            )
        assert recomputed is True
        assert collect_mock.call_count == 1


def test_ekfac_cache_recomputes_when_metadata_uses_old_ln_post_policy() -> None:
    classifier = _build_classifier()
    model = ImageEncoder(classifier)
    device = torch.device('cpu')
    build_cfg = OpenClipBuildConfig(model_name='ViT-B-32', pretrained='openai', device='cpu', dtype='fp32')
    regularizer = EkfacGgnRegularizer()
    surface = ensure_openclip_kfac_surface(model)
    plan, expected_meta = regularizer._expected_cache_metadata(  # type: ignore[attr-defined]
        model=model,
        task='Cars',
        build_cfg=build_cfg,
        config=EkfacGgnConfig(),
        attn_patch_cfg=surface['attn_patch_cfg'],
    )
    UA, UG, D, ffT = _build_identity_ekfac(plan)
    stale_meta = dict(expected_meta)
    stale_meta.pop('ln_post_full_block_policy')
    with TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        path = task_cache_path(cache_dir=cache_dir, build_cfg=build_cfg, task='Cars')
        save_task_ekfac(
            path,
            EkfacTaskCurvatureStats(
                UA=UA,
                UG=UG,
                D=D,
                ffT=ffT,
                num_examples=8,
                metadata=stale_meta,
            ),
        )
        loader = torch.utils.data.DataLoader([(torch.randn(4), torch.tensor(0)) for _ in range(4)], batch_size=2)
        with patch(
            'merge_and_rebase.finetune.regularizers.ekfac_ggn.collect_ekfac_curvature',
            return_value=EkfacTaskCurvatureStats(
                UA=UA,
                UG=UG,
                D=D,
                ffT=ffT,
                num_examples=2,
                metadata=expected_meta,
            ),
        ) as collect_mock:
            _, recomputed = regularizer._ensure_cache(  # type: ignore[attr-defined]
                model=model,
                loader=loader,
                task='Cars',
                build_cfg=build_cfg,
                config=EkfacGgnConfig(cache_dir=cache_dir),
                attn_patch_cfg=surface['attn_patch_cfg'],
                device=device,
            )
        assert recomputed is True
        assert collect_mock.call_count == 1


def test_ekfac_cache_reuses_when_sampling_and_precision_metadata_differs() -> None:
    classifier = _build_classifier()
    model = ImageEncoder(classifier)
    device = torch.device('cpu')
    build_cfg = OpenClipBuildConfig(model_name='ViT-B-32', pretrained='openai', device='cpu', dtype='fp32')
    regularizer = EkfacGgnRegularizer()
    surface = ensure_openclip_kfac_surface(model)
    plan, cached_meta = regularizer._expected_cache_metadata(  # type: ignore[attr-defined]
        model=model,
        task='Cars',
        build_cfg=build_cfg,
        config=EkfacGgnConfig(),
        attn_patch_cfg=surface['attn_patch_cfg'],
    )
    UA, UG, D, ffT = _build_identity_ekfac(plan)
    with TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        path = task_cache_path(cache_dir=cache_dir, build_cfg=build_cfg, task='Cars')
        save_task_ekfac(
            path,
            EkfacTaskCurvatureStats(
                UA=UA,
                UG=UG,
                D=D,
                ffT=ffT,
                num_examples=8,
                metadata=cached_meta,
            ),
        )
        loader = torch.utils.data.DataLoader([(torch.randn(4), torch.tensor(0)) for _ in range(4)], batch_size=2)
        with patch('merge_and_rebase.finetune.regularizers.ekfac_ggn.collect_ekfac_curvature') as collect_mock:
            _, recomputed = regularizer._ensure_cache(  # type: ignore[attr-defined]
                model=model,
                loader=loader,
                task='Cars',
                build_cfg=build_cfg,
                config=EkfacGgnConfig(
                    cache_dir=cache_dir,
                    precision='fp64',
                    train_percent=0.5,
                    fisher_seed=7,
                    fisher_num_samples_expectation=3,
                ),
                attn_patch_cfg=surface['attn_patch_cfg'],
                device=device,
            )
        assert recomputed is False
        assert collect_mock.call_count == 0


def test_ekfac_regularizer_zero_at_base_and_positive_after_supported_update() -> None:
    torch.manual_seed(0)
    classifier = _build_classifier()
    model = ImageEncoder(classifier)
    device = torch.device('cpu')
    build_cfg = OpenClipBuildConfig(model_name='ViT-B-32', pretrained='openai', device='cpu', dtype='fp32')
    regularizer = EkfacGgnRegularizer()
    train_loader = torch.utils.data.DataLoader([(torch.randn(4), torch.tensor(0)) for _ in range(4)], batch_size=2)
    loaders = SimpleNamespace(train=train_loader)
    plan = select_tracked_parameters(model)
    UA, UG, D, ffT = _build_identity_ekfac(plan)
    with TemporaryDirectory() as tmpdir:
        cache_path = task_cache_path(cache_dir=tmpdir, build_cfg=build_cfg, task='task_b')
        save_task_ekfac(
            cache_path,
            EkfacTaskCurvatureStats(
                UA=UA,
                UG=UG,
                D=D,
                ffT=ffT,
                num_examples=8,
                metadata={'source': 'test'},
            ),
        )
        with patch(
            'merge_and_rebase.finetune.regularizers.ekfac_ggn.build_vision_regularizer_task_context',
            return_value=SimpleNamespace(task='task_b', build_cfg=build_cfg, loader=train_loader, model=model),
        ):
            prepared, info = regularizer.prepare(
                model=model,
                device=device,
                regularization_cfg={'cache_dir': tmpdir, 'reg_lambda': 1.0, 'cadence': 1},
                task='task_a',
                build_cfg=build_cfg,
                loaders=loaders,
                strategy_cfg={},
                reference_tasks=['task_a', 'task_b'],
                batch_size=2,
                num_workers=0,
                val_fraction=0.1,
                seed=42,
            )
        zero_loss = regularizer.apply(prepared, model=model, step=0, batch_index=0)
        assert float(zero_loss.detach()) == 0.0
        first_key = next(iter(plan.matrix_blocks))
        local_name = first_key[len('visual.') :]
        dict(model.clip_model.model.visual.named_parameters())[local_name].data.add_(0.25)
        moved_loss = regularizer.apply(prepared, model=model, step=0, batch_index=0)
        assert float(moved_loss.detach()) > 0.0
        assert info['ekfac_reference_tasks'] == 1
        assert info['ekfac_ignored_trainable'] == len(prepared.plan.ignored_trainable)


def test_ekfac_regularizer_auto_excludes_current_task_from_reference_tasks() -> None:
    classifier = _build_classifier()
    model = ImageEncoder(classifier)
    device = torch.device('cpu')
    build_cfg = OpenClipBuildConfig(model_name='ViT-B-32', pretrained='openai', device='cpu', dtype='fp32')
    regularizer = EkfacGgnRegularizer()
    train_loader = torch.utils.data.DataLoader([(torch.randn(4), torch.tensor(0)) for _ in range(4)], batch_size=2)
    loaders = SimpleNamespace(train=train_loader)
    prepared, info = regularizer.prepare(
        model=model,
        device=device,
        regularization_cfg={'reg_lambda': 1.0, 'cadence': 1},
        task='task_a',
        build_cfg=build_cfg,
        loaders=loaders,
        strategy_cfg={},
        reference_tasks=['task_a'],
        batch_size=2,
        num_workers=0,
        val_fraction=0.1,
        seed=42,
    )
    loss = regularizer.apply(prepared, model=model, step=0, batch_index=0)
    assert float(loss.detach()) == 0.0
    assert info['ekfac_reference_tasks'] == 0


def test_compute_ekfac_penalty_matches_manual_projection_formula() -> None:
    delta = torch.tensor([[2.0, -1.0], [0.5, 3.0]])
    refs = [
        (
            'task_b',
            1.0,
            EkfacTaskCurvatureStats(
                UA={'visual.block.weight': torch.eye(2)},
                UG={'visual.block.weight': torch.eye(2)},
                D={'visual.block.weight': torch.tensor([[1.0, 2.0], [3.0, 4.0]])},
                ffT={},
                num_examples=5,
                metadata=None,
            ),
        )
    ]
    breakdown = compute_ekfac_penalty({'visual.block.weight': delta}, refs)
    manual = (refs[0][2].D['visual.block.weight'] * delta.pow(2)).sum()
    assert torch.allclose(breakdown.loss_reg_matrix, manual)
    assert float(breakdown.loss_reg_ffT) == 0.0


def test_ekfac_multi_task_penalty_sums_weighted_references() -> None:
    delta = torch.tensor([[1.0, 2.0]])
    refs = [
        (
            'task_b',
            0.25,
            EkfacTaskCurvatureStats(
                UA={'visual.block.weight': torch.eye(2)},
                UG={'visual.block.weight': torch.eye(1)},
                D={'visual.block.weight': torch.tensor([[2.0, 1.0]])},
                ffT={},
                num_examples=1,
                metadata=None,
            ),
        ),
        (
            'task_c',
            0.75,
            EkfacTaskCurvatureStats(
                UA={'visual.block.weight': torch.eye(2)},
                UG={'visual.block.weight': torch.eye(1)},
                D={'visual.block.weight': torch.tensor([[4.0, 3.0]])},
                ffT={},
                num_examples=3,
                metadata=None,
            ),
        ),
    ]
    breakdown = compute_ekfac_penalty({'visual.block.weight': delta}, refs)
    manual = 0.25 * (refs[0][2].D['visual.block.weight'] * delta.pow(2)).sum() + 0.75 * (
        refs[1][2].D['visual.block.weight'] * delta.pow(2)
    ).sum()
    assert torch.allclose(breakdown.loss_reg_matrix, manual)


def test_ekfac_ffT_penalty_uses_reference_weights_without_extra_example_division() -> None:
    delta = torch.tensor([2.0, -1.0])
    refs = [
        (
            'task_b',
            0.25,
            EkfacTaskCurvatureStats(
                UA={},
                UG={},
                D={},
                ffT={'visual.block.bias': torch.tensor([[2.0, 0.0], [0.0, 4.0]])},
                num_examples=1,
                metadata=None,
            ),
        ),
        (
            'task_c',
            0.75,
            EkfacTaskCurvatureStats(
                UA={},
                UG={},
                D={},
                ffT={'visual.block.bias': torch.tensor([[6.0, 0.0], [0.0, 12.0]])},
                num_examples=3,
                metadata=None,
            ),
        ),
    ]
    breakdown = compute_ekfac_penalty({'visual.block.bias': delta}, refs)
    manual = sum(
        float(coeff) * torch.trace(delta.reshape(1, -1) @ stats.ffT['visual.block.bias'] @ delta.reshape(1, -1).T)
        for _, coeff, stats in refs
    )
    assert torch.allclose(breakdown.loss_reg_ffT, manual)
