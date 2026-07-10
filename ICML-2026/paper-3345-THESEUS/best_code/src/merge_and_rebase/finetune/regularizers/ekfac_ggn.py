from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from merge_and_rebase.finetune.reference_tasks import resolve_reference_tasks_from_kwargs
from merge_and_rebase.models.openclip_classifier import OpenClipBuildConfig

from ._vision_collection import build_vision_regularizer_task_context
from .kfac_ggn import (
    FullBlock,
    KfacGgnConfig,
    MatrixBlock,
    TaskCurvatureStats,
    TrackedCurvaturePlan,
    _acc_values,
    _as_config,
    _base_snapshot,
    _delta_params,
    _dtype_from_precision,
    _flatten_sequence,
    _format_cache_completed,
    _format_cache_status,
    _infer_sequence_layout,
    _images_from_batch,
    _kahan_add,
    _load_cache_metadata,
    _module_by_name,
    _progress_total,
    _projection_rows_from_ln_post,
    _resolve_num_batches,
    _run_visual,
    _tensor_from_output,
    _visual_module,
    collect_curvature,
    ensure_openclip_kfac_surface,
    load_task_curvature,
    metadata_compatible,
    normalize_attn_patch_cfg,
    save_task_curvature,
    select_tracked_parameters,
    task_cache_path,
)
from .kfac_ggn import (
    _metadata as _kfac_metadata,
)
from .registry import register

_VERSION = 3


@dataclass
class EkfacTaskCurvatureStats:
    UA: dict[str, torch.Tensor]
    UG: dict[str, torch.Tensor]
    D: dict[str, torch.Tensor]
    ffT: dict[str, torch.Tensor]
    num_examples: int
    metadata: dict[str, Any] | None = None


@dataclass
class EkfacPenaltyBreakdown:
    loss_reg_matrix: torch.Tensor
    loss_reg_ffT: torch.Tensor
    loss_reg_proj: torch.Tensor
    loss_reg_cls: torch.Tensor

    @property
    def total_unscaled(self) -> torch.Tensor:
        return self.loss_reg_matrix + self.loss_reg_ffT + self.loss_reg_proj + self.loss_reg_cls


@dataclass(frozen=True)
class PreparedEkfacGgn:
    config: EkfacGgnConfig
    plan: TrackedCurvaturePlan
    base: dict[str, torch.Tensor]
    references: list[tuple[str, float, EkfacTaskCurvatureStats]]
    ignored_trainable: int


@dataclass(frozen=True)
class EkfacGgnConfig:
    cache_dir: Path = Path('src/checkpoints/ekfac_ggn')
    precision: str = 'fp32'
    reg_lambda: float = 0.0
    full_block_scaler: float = 1.0e4
    projection_scaler: float = 1.0e-3
    cadence: int = 1
    force_recompute: bool = False
    train_percent: float | int = 1.0
    fisher_seed: int | None = None
    fisher_num_samples_expectation: int = 1


def _as_ekfac_config(raw: Mapping[str, Any] | None) -> EkfacGgnConfig:
    cfg = dict(raw or {})
    cfg.pop('name', None)
    cfg.pop('reference_suite', None)
    cfg.pop('reference_datasets', None)
    allowed = set(EkfacGgnConfig.__dataclass_fields__)
    unknown = sorted(k for k in cfg if k not in allowed)
    if unknown:
        raise ValueError(f'Unknown ekfac_ggn config keys: {unknown}')
    if 'cache_dir' in cfg:
        cfg['cache_dir'] = Path(str(cfg['cache_dir']))
    if 'train_percent' in cfg and isinstance(cfg['train_percent'], str):
        raw_value = cfg['train_percent'].strip()
        cfg['train_percent'] = float(raw_value) if any(c in raw_value for c in '.eE') else int(raw_value)
    out = EkfacGgnConfig(**cfg)
    if out.precision not in {'fp32', 'fp64'}:
        raise ValueError('ekfac_ggn precision must be one of: fp32, fp64')
    if out.cadence < 1:
        raise ValueError('ekfac_ggn cadence must be >= 1')
    if isinstance(out.train_percent, float) and not (0 < out.train_percent <= 1.0):
        raise ValueError('ekfac_ggn train_percent float must be in (0, 1].')
    if isinstance(out.train_percent, int) and out.train_percent < 1:
        raise ValueError('ekfac_ggn train_percent int must be >= 1.')
    if out.fisher_num_samples_expectation < 0:
        raise ValueError('ekfac_ggn fisher_num_samples_expectation must be >= 0.')
    return out


def _metadata(
    *,
    task: str,
    build_cfg: OpenClipBuildConfig,
    config: EkfacGgnConfig,
    plan: TrackedCurvaturePlan,
    attn_patch_cfg: Mapping[str, Any] | None,
) -> dict[str, Any]:
    kfac_cfg = KfacGgnConfig(
        cache_dir=config.cache_dir,
        precision=config.precision,
        reg_lambda=config.reg_lambda,
        full_block_scaler=config.full_block_scaler,
        projection_scaler=config.projection_scaler,
        cadence=config.cadence,
        force_recompute=config.force_recompute,
        train_percent=config.train_percent,
        fisher_seed=config.fisher_seed,
        fisher_num_samples_expectation=config.fisher_num_samples_expectation,
    )
    meta = _kfac_metadata(
        task=task,
        build_cfg=build_cfg,
        config=kfac_cfg,
        plan=plan,
        attn_patch_cfg=attn_patch_cfg,
    )
    meta['version'] = _VERSION
    meta['regularizer'] = 'ekfac_ggn'
    return meta


def save_task_ekfac(path: str | Path, stats: EkfacTaskCurvatureStats) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'UA': {k: v.detach().cpu() for k, v in stats.UA.items()},
        'UG': {k: v.detach().cpu() for k, v in stats.UG.items()},
        'D': {k: v.detach().cpu() for k, v in stats.D.items()},
        'ffT': {k: v.detach().cpu() for k, v in stats.ffT.items()},
        'num_examples': int(stats.num_examples),
        'metadata': dict(stats.metadata or {}),
    }
    torch.save(payload, p)


def load_task_ekfac(
    path: str | Path,
    *,
    device: torch.device | str = 'cpu',
    precision: str = 'fp32',
) -> EkfacTaskCurvatureStats:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    dtype = _dtype_from_precision(precision)
    metadata = dict(payload.get('metadata', {}))

    def _load_dict(name: str) -> dict[str, torch.Tensor]:
        raw = payload.get(name, {})
        if not isinstance(raw, dict):
            raise ValueError(f"Invalid EKFAC cache: '{name}' is not a dict")
        return {str(k): v.to(dtype=dtype) for k, v in raw.items()}

    return EkfacTaskCurvatureStats(
        UA=_load_dict('UA'),
        UG=_load_dict('UG'),
        D=_load_dict('D'),
        ffT=_load_dict('ffT'),
        num_examples=int(payload['num_examples']),
        metadata=metadata,
    )


def _set_fisher_seed(seed: int | None) -> None:
    if seed is None:
        return
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _per_example_grad_weight(
    *,
    inputs: torch.Tensor,
    grad_output: torch.Tensor,
    layout: str,
    current_batch_size: int | None,
    target: str,
    include_bias: bool,
) -> torch.Tensor:
    x = inputs
    grad = grad_output
    if x.ndim == 3:
        inferred_layout = _infer_sequence_layout(
            x,
            layout=layout,
            current_batch_size=current_batch_size,
            target=target,
        )
        if inferred_layout == 'sequence_first':
            x = x.permute(1, 0, 2)
            grad = grad.permute(1, 0, 2)
        grad_weight = torch.einsum('bso,bsi->boi', grad, x)
        if include_bias:
            grad_bias = grad.sum(dim=1)
            grad_weight = torch.cat([grad_weight, grad_bias.unsqueeze(2)], dim=2)
        return grad_weight
    if x.ndim == 2:
        grad_weight = torch.einsum('bo,bi->boi', grad, x)
        if include_bias:
            grad_weight = torch.cat([grad_weight, grad.unsqueeze(2)], dim=2)
        return grad_weight
    raise ValueError(f'EKFAC hooks expect 2D or 3D tensors, got inputs={tuple(x.shape)}')


def _project_batch_grad(
    grad_weight: torch.Tensor,
    *,
    UA: torch.Tensor,
    UG: torch.Tensor,
) -> torch.Tensor:
    projected = torch.einsum('oi,bij->boj', UG.T.to(device=grad_weight.device, dtype=grad_weight.dtype), grad_weight)
    projected = torch.einsum('boj,jk->bok', projected, UA.to(device=grad_weight.device, dtype=grad_weight.dtype))
    return projected


def collect_ekfac_curvature(
    model: nn.Module,
    data_loader: Iterable[Any],
    tracked_params: TrackedCurvaturePlan | None = None,
    config: EkfacGgnConfig | Mapping[str, Any] | None = None,
    *,
    device: torch.device | str | None = None,
    base_stats: TaskCurvatureStats | None = None,
    progress_label: str | None = None,
) -> EkfacTaskCurvatureStats:
    cfg = config if isinstance(config, EkfacGgnConfig) else _as_ekfac_config(config)
    visual = _visual_module(model)
    plan = tracked_params or select_tracked_parameters(visual)
    dtype = _dtype_from_precision(cfg.precision)
    dev = torch.device(device) if device is not None else next(visual.parameters()).device

    kfac_cfg = KfacGgnConfig(
        cache_dir=cfg.cache_dir,
        precision=cfg.precision,
        reg_lambda=cfg.reg_lambda,
        full_block_scaler=cfg.full_block_scaler,
        projection_scaler=cfg.projection_scaler,
        cadence=cfg.cadence,
        force_recompute=cfg.force_recompute,
        train_percent=cfg.train_percent,
        fisher_seed=cfg.fisher_seed,
        fisher_num_samples_expectation=cfg.fisher_num_samples_expectation,
    )
    if base_stats is None:
        base_stats = collect_curvature(
            model,
            data_loader,
            tracked_params=plan,
            config=kfac_cfg,
            device=dev,
        )
    num_examples = max(1, int(base_stats.num_examples_ggT))
    UA = {
        key: torch.linalg.svd((aaT / float(num_examples)).to(dtype=dtype))[0].to(dtype=dtype)
        for key, aaT in base_stats.aaT.items()
    }
    UG = {
        key: torch.linalg.svd((base_stats.ggT[key] / float(num_examples)).to(dtype=dtype))[0].to(dtype=dtype)
        for key in base_stats.ggT
    }

    orig_training = visual.training
    orig_requires_grad = {name: param.requires_grad for name, param in visual.named_parameters()}
    handles: list[Any] = []
    d_accs: dict[str, Any] = {}
    modules = dict(visual.named_modules())
    matrix_inputs: dict[str, torch.Tensor] = {}
    projection_rows: dict[str, torch.Tensor] = {}
    max_batches = _resolve_num_batches(data_loader, kfac_cfg)
    proj_block = next((b for b in plan.matrix_blocks.values() if b.is_projection and b.module_name is None), None)
    current_batch_size: int | None = None

    def _register_matrix_hooks() -> None:
        for block in plan.matrix_blocks.values():
            if block.is_projection and block.module_name is None:
                ln_post = modules.get('ln_post', None)
                if ln_post is None:
                    raise RuntimeError('Cannot collect visual.proj EKFAC block: visual.ln_post was not found.')

                def proj_hook(_module, _inputs, output, *, key=block.key):
                    hook_input = _tensor_from_output(output).detach().to(dtype=dtype)
                    rows = _projection_rows_from_ln_post(hook_input, visual)
                    projection_rows[key] = rows

                handles.append(ln_post.register_forward_hook(proj_hook))
                continue

            module = _module_by_name(visual, block.module_name)
            if module is None:
                raise RuntimeError(f'Tracked EKFAC module not found: {block.module_name}')

            def fwd_hook(_module, inputs, _output, *, key=block.key):
                matrix_inputs[key] = inputs[0].detach().to(dtype=dtype)

            def bwd_hook(_module, _grad_input, grad_output, *, b=block):
                with torch.no_grad():
                    x = matrix_inputs.get(b.key, None)
                    if x is None:
                        return
                    grad = grad_output[0].detach().to(dtype=dtype)
                    grad_weight = _per_example_grad_weight(
                        inputs=x,
                        grad_output=grad,
                        layout=b.layout,
                        current_batch_size=current_batch_size,
                        target=b.key,
                        include_bias=b.bias_key is not None,
                    )
                    projected = _project_batch_grad(grad_weight, UA=UA[b.key], UG=UG[b.key])
                    _kahan_add(d_accs, b.key, projected.pow(2).sum(dim=0))

            handles.append(module.register_forward_hook(fwd_hook))
            handles.append(module.register_full_backward_hook(bwd_hook))

    try:
        _set_fisher_seed(cfg.fisher_seed)
        visual.eval()
        for param in visual.parameters():
            param.requires_grad_(False)
        named_params = dict(visual.named_parameters())
        for key in plan.param_shapes:
            local_name = key[len('visual.') :] if key.startswith('visual.') else key
            param = named_params.get(local_name, None)
            if param is not None:
                param.requires_grad_(True)

        _register_matrix_hooks()
        batch_iter = data_loader if max_batches is None else islice(data_loader, max_batches)
        ekfac_desc = f"[{progress_label}] EKFAC curvature" if progress_label else "EKFAC curvature"
        with tqdm(
            batch_iter,
            total=_progress_total(data_loader, max_batches),
            desc=ekfac_desc,
            unit='batch',
        ) as pbar:
            for batch in pbar:
                visual.zero_grad(set_to_none=True)
                matrix_inputs.clear()
                projection_rows.clear()
                images = _images_from_batch(batch).to(dev)
                current_batch_size = int(images.shape[0])
                fake_param = torch.tensor([1.0], device=dev, requires_grad=True)
                raw_features = _run_visual(visual, images * fake_param)
                if raw_features.ndim != 2:
                    raise RuntimeError(
                        'EKFAC curvature expects visual(image) to produce pooled features with shape (batch, channels); '
                        f'got shape={tuple(raw_features.shape)}'
                    )
                if proj_block is not None:
                    def proj_grad_hook(grad, *, key=proj_block.key):
                        with torch.no_grad():
                            hook_input = grad.detach().to(dtype=dtype)
                            rows = projection_rows.get(key, None)
                            if rows is None:
                                return
                            grad_weight = torch.einsum('bo,bi->boi', hook_input, rows)
                            projected = _project_batch_grad(grad_weight, UA=UA[key], UG=UG[key])
                            _kahan_add(d_accs, key, projected.pow(2).sum(dim=0))

                    raw_features.register_hook(proj_grad_hook)
                features = F.normalize(raw_features, dim=-1)
                if cfg.fisher_num_samples_expectation > 0:
                    for sample_idx in range(int(cfg.fisher_num_samples_expectation)):
                        probe = torch.randn_like(features)
                        backward_source = features * probe
                        backward_target = backward_source.sum()
                        backward_target.backward(retain_graph=sample_idx < int(cfg.fisher_num_samples_expectation) - 1)
                else:
                    summed = features.sum(0)
                    for feat_idx, feat in enumerate(summed):
                        visual.zero_grad(set_to_none=True)
                        feat.backward(retain_graph=feat_idx < summed.shape[0] - 1)

        visual.zero_grad(set_to_none=True)
        D = {key: value / float(num_examples) for key, value in _acc_values(d_accs).items()}
        missing = sorted(set(plan.matrix_blocks) - set(D))
        if missing:
            raise RuntimeError(f'EKFAC/GGN collection missed tracked blocks: D={missing}')
        return EkfacTaskCurvatureStats(
            UA=UA,
            UG=UG,
            D=D,
            ffT=base_stats.ffT,
            num_examples=num_examples,
        )
    finally:
        for handle in handles:
            handle.remove()
        for name, param in visual.named_parameters():
            if name in orig_requires_grad:
                param.requires_grad_(orig_requires_grad[name])
        visual.train(orig_training)



def _related_kfac_cache_dir(cache_dir: Path) -> Path:
    parts = list(cache_dir.parts)
    for idx in range(len(parts) - 1, -1, -1):
        part = parts[idx]
        if part.startswith('ekfac_ggn'):
            parts[idx] = 'kfac_ggn' + part[len('ekfac_ggn') :]
            return Path(*parts)
    return cache_dir.parent / ('kfac_ggn' + cache_dir.name[len('ekfac_ggn') :]) if cache_dir.name.startswith('ekfac_ggn') else cache_dir.parent / 'kfac_ggn'


def _load_cached_kfac_stats(
    *,
    task: str,
    build_cfg: OpenClipBuildConfig,
    config: EkfacGgnConfig,
    plan: TrackedCurvaturePlan,
    attn_patch_cfg: Mapping[str, Any] | None,
    device: torch.device,
) -> TaskCurvatureStats | None:
    kfac_cache_dir = _related_kfac_cache_dir(config.cache_dir)
    kfac_path = task_cache_path(cache_dir=kfac_cache_dir, build_cfg=build_cfg, task=task)
    if not kfac_path.exists():
        return None
    kfac_cfg = KfacGgnConfig(
        cache_dir=kfac_cache_dir,
        precision=config.precision,
        reg_lambda=config.reg_lambda,
        full_block_scaler=config.full_block_scaler,
        projection_scaler=config.projection_scaler,
        cadence=config.cadence,
        force_recompute=config.force_recompute,
        train_percent=config.train_percent,
        fisher_seed=config.fisher_seed,
        fisher_num_samples_expectation=config.fisher_num_samples_expectation,
    )
    expected = _kfac_metadata(
        task=task,
        build_cfg=build_cfg,
        config=kfac_cfg,
        plan=plan,
        attn_patch_cfg=attn_patch_cfg,
    )
    existing = _load_cache_metadata(kfac_path)
    if not metadata_compatible(existing, expected):
        return None
    return load_task_curvature(kfac_path, device=device, precision=config.precision)



def _store_related_kfac_cache(
    *,
    task: str,
    build_cfg: OpenClipBuildConfig,
    config: EkfacGgnConfig,
    plan: TrackedCurvaturePlan,
    attn_patch_cfg: Mapping[str, Any] | None,
    stats: TaskCurvatureStats,
) -> Path:
    kfac_cache_dir = _related_kfac_cache_dir(config.cache_dir)
    kfac_cfg = KfacGgnConfig(
        cache_dir=kfac_cache_dir,
        precision=config.precision,
        reg_lambda=config.reg_lambda,
        full_block_scaler=config.full_block_scaler,
        projection_scaler=config.projection_scaler,
        cadence=config.cadence,
        force_recompute=config.force_recompute,
        train_percent=config.train_percent,
        fisher_seed=config.fisher_seed,
        fisher_num_samples_expectation=config.fisher_num_samples_expectation,
    )
    meta = _kfac_metadata(
        task=task,
        build_cfg=build_cfg,
        config=kfac_cfg,
        plan=plan,
        attn_patch_cfg=attn_patch_cfg,
    )
    path = task_cache_path(cache_dir=kfac_cache_dir, build_cfg=build_cfg, task=task)
    save_task_curvature(
        path,
        TaskCurvatureStats(
            aaT=stats.aaT,
            ggT=stats.ggT,
            ffT=stats.ffT,
            num_examples_aaT=int(stats.num_examples_aaT),
            num_examples_ggT=int(stats.num_examples_ggT),
            metadata=meta,
        ),
    )
    return path


def _zero_like_context(delta_params: Mapping[str, torch.Tensor], references: list[tuple[str, float, EkfacTaskCurvatureStats]]) -> torch.Tensor:
    for tensor in delta_params.values():
        return tensor.sum() * 0.0
    for _task, _coeff, stats in references:
        for groups in (stats.UA, stats.ffT):
            for tensor in groups.values():
                return tensor.sum() * 0.0
    return torch.tensor(0.0)


def compute_ekfac_penalty(
    delta_params: Mapping[str, torch.Tensor],
    references: list[tuple[str, float, EkfacTaskCurvatureStats]],
) -> EkfacPenaltyBreakdown:
    zero = _zero_like_context(delta_params, references)
    loss_reg_matrix = zero
    loss_reg_ffT = zero
    loss_reg_proj = zero
    loss_reg_cls = zero

    for task, coeff, stats in references:
        del task
        for key, UA in stats.UA.items():
            if key not in delta_params:
                continue
            delta = delta_params[key]
            is_projection = key == 'visual.proj'
            if is_projection:
                delta_w = delta.T
            else:
                delta_w = delta
                if key.endswith('.weight'):
                    bias_key = key[: -len('.weight')] + '.bias'
                    if bias_key in delta_params and UA.shape[0] == delta_w.shape[1] + 1:
                        delta_w = torch.cat([delta_w, delta_params[bias_key].unsqueeze(1)], dim=1)
            UA_d = UA.to(device=delta_w.device, dtype=delta_w.dtype)
            UG_d = stats.UG[key].to(device=delta_w.device, dtype=delta_w.dtype)
            D_d = stats.D[key].to(device=delta_w.device, dtype=delta_w.dtype)
            projected = UG_d.T @ delta_w @ UA_d
            value = float(coeff) * (D_d * projected.pow(2)).sum()
            if is_projection:
                loss_reg_proj = loss_reg_proj + value
            else:
                loss_reg_matrix = loss_reg_matrix + value

        for key, ffT in stats.ffT.items():
            if key not in delta_params:
                continue
            delta = delta_params[key].reshape(1, -1)
            ffT_d = ffT.to(device=delta.device, dtype=delta.dtype)
            value = float(coeff) * torch.trace(delta @ ffT_d @ delta.T)
            if key == 'visual.class_embedding':
                loss_reg_cls = loss_reg_cls + value
            else:
                loss_reg_ffT = loss_reg_ffT + value

    return EkfacPenaltyBreakdown(
        loss_reg_matrix=loss_reg_matrix,
        loss_reg_ffT=loss_reg_ffT,
        loss_reg_proj=loss_reg_proj,
        loss_reg_cls=loss_reg_cls,
    )


class EkfacGgnRegularizer:
    name = 'ekfac_ggn'

    def finalize_model(
        self,
        *,
        model: nn.Module,
        device: torch.device,
        regularization_cfg: dict | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        del regularization_cfg
        build_cfg = kwargs.get('build_cfg', None)
        if not isinstance(build_cfg, OpenClipBuildConfig):
            return {}
        strategy_cfg = kwargs.get('strategy_cfg', None)
        attn_patch_cfg = self._resolve_attn_patch_cfg(model=model, strategy_cfg=strategy_cfg)
        surface = ensure_openclip_kfac_surface(model, attn_patch_cfg=attn_patch_cfg)
        model.to(device)
        return {
            'patched_blocks': int(surface['patched_blocks']),
            'patched_attn_impl': str(surface['attn_patch_cfg']['attn_impl']),
        }

    def _resolve_attn_patch_cfg(
        self,
        *,
        model: nn.Module,
        strategy_cfg: Mapping[str, Any] | None,
    ) -> dict[str, Any] | None:
        existing_attn_cfg = getattr(model, 'peft_attn_patch_cfg', None)
        attn_patch_cfg = existing_attn_cfg if isinstance(existing_attn_cfg, dict) else None
        if isinstance(strategy_cfg, Mapping) and isinstance(strategy_cfg.get('attention', None), dict):
            attn_patch_cfg = attn_patch_cfg or dict(strategy_cfg['attention'])
        return attn_patch_cfg

    def _expected_cache_metadata(
        self,
        *,
        model: nn.Module,
        task: str,
        build_cfg: OpenClipBuildConfig,
        config: EkfacGgnConfig,
        attn_patch_cfg: Mapping[str, Any] | None,
    ) -> tuple[TrackedCurvaturePlan, dict[str, Any]]:
        visual = _visual_module(model)
        plan = select_tracked_parameters(visual)
        meta = _metadata(
            task=task,
            build_cfg=build_cfg,
            config=config,
            plan=plan,
            attn_patch_cfg=attn_patch_cfg,
        )
        return plan, meta

    def _collect_and_store(
        self,
        *,
        model: nn.Module,
        loader: Iterable[Any],
        task: str,
        build_cfg: OpenClipBuildConfig,
        config: EkfacGgnConfig,
        attn_patch_cfg: Mapping[str, Any] | None,
        device: torch.device,
    ) -> tuple[Path, EkfacTaskCurvatureStats]:
        surface = ensure_openclip_kfac_surface(model, attn_patch_cfg=attn_patch_cfg)
        plan, meta = self._expected_cache_metadata(
            model=model,
            task=task,
            build_cfg=build_cfg,
            config=config,
            attn_patch_cfg=surface['attn_patch_cfg'],
        )
        kfac_stats = _load_cached_kfac_stats(
            task=task,
            build_cfg=build_cfg,
            config=config,
            plan=plan,
            attn_patch_cfg=surface['attn_patch_cfg'],
            device=device,
        )
        kfac_path = task_cache_path(cache_dir=_related_kfac_cache_dir(config.cache_dir), build_cfg=build_cfg, task=task)
        if kfac_stats is None:
            print(
                _format_cache_status(
                    regularizer='ekfac_ggn',
                    task=task,
                    stage='related KFAC curvature',
                    path=kfac_path,
                    cached=False,
                )
            )
            kfac_cfg = KfacGgnConfig(
                cache_dir=_related_kfac_cache_dir(config.cache_dir),
                precision=config.precision,
                reg_lambda=config.reg_lambda,
                full_block_scaler=config.full_block_scaler,
                projection_scaler=config.projection_scaler,
                cadence=config.cadence,
                force_recompute=config.force_recompute,
                train_percent=config.train_percent,
                fisher_seed=config.fisher_seed,
                fisher_num_samples_expectation=config.fisher_num_samples_expectation,
            )
            kfac_stats = collect_curvature(
                model,
                loader,
                tracked_params=plan,
                config=kfac_cfg,
                device=device,
                progress_label=f'{task} related',
            )
            _store_related_kfac_cache(
                task=task,
                build_cfg=build_cfg,
                config=config,
                plan=plan,
                attn_patch_cfg=surface['attn_patch_cfg'],
                stats=kfac_stats,
            )
            print(
                _format_cache_completed(
                    regularizer='ekfac_ggn',
                    task=task,
                    stage='related KFAC curvature',
                    path=kfac_path,
                )
            )
        else:
            print(
                _format_cache_status(
                    regularizer='ekfac_ggn',
                    task=task,
                    stage='related KFAC curvature',
                    path=kfac_path,
                    cached=True,
                )
            )
        stats = collect_ekfac_curvature(
            model,
            loader,
            tracked_params=plan,
            config=config,
            device=device,
            base_stats=kfac_stats,
            progress_label=task,
        )
        stats.metadata = meta
        path = task_cache_path(cache_dir=config.cache_dir, build_cfg=build_cfg, task=task)
        save_task_ekfac(path, stats)
        print(
            _format_cache_completed(
                regularizer='ekfac_ggn',
                task=task,
                stage='EKFAC curvature',
                path=path,
            )
        )
        return path, stats

    def _ensure_cache(
        self,
        *,
        model: nn.Module,
        loader: Iterable[Any],
        task: str,
        build_cfg: OpenClipBuildConfig,
        config: EkfacGgnConfig,
        attn_patch_cfg: Mapping[str, Any] | None,
        device: torch.device,
    ) -> tuple[Path, bool]:
        surface = ensure_openclip_kfac_surface(model, attn_patch_cfg=attn_patch_cfg)
        _, expected = self._expected_cache_metadata(
            model=model,
            task=task,
            build_cfg=build_cfg,
            config=config,
            attn_patch_cfg=surface['attn_patch_cfg'],
        )
        path = task_cache_path(cache_dir=config.cache_dir, build_cfg=build_cfg, task=task)
        if path.exists() and not config.force_recompute:
            existing = _load_cache_metadata(path)
            if metadata_compatible(existing, expected):
                print(
                    _format_cache_status(
                        regularizer='ekfac_ggn',
                        task=task,
                        stage='EKFAC curvature',
                        path=path,
                        cached=True,
                    )
                )
                return path, False
        print(
            _format_cache_status(
                regularizer='ekfac_ggn',
                task=task,
                stage='EKFAC curvature',
                path=path,
                cached=False,
            )
        )
        self._collect_and_store(
            model=model,
            loader=loader,
            task=task,
            build_cfg=build_cfg,
            config=config,
            attn_patch_cfg=surface['attn_patch_cfg'],
            device=device,
        )
        return path, True

    def prepare(
        self,
        *,
        model: nn.Module,
        device: torch.device,
        regularization_cfg: dict | None = None,
        **kwargs,
    ) -> tuple[PreparedEkfacGgn, dict[str, int]]:
        config = _as_ekfac_config(regularization_cfg)
        task = str(kwargs.get('task', '')).strip()
        build_cfg = kwargs.get('build_cfg', None)
        if not isinstance(build_cfg, OpenClipBuildConfig):
            raise ValueError('ekfac_ggn.prepare requires build_cfg from train_vision.')
        loaders = kwargs.get('loaders', None)
        train_loader = getattr(loaders, 'train', None)
        if train_loader is None:
            raise ValueError('ekfac_ggn.prepare requires loaders.train from train_vision.')
        run_logger = kwargs.get('run_logger', None)
        strategy_cfg = kwargs.get('strategy_cfg', None)
        attn_patch_cfg = self._resolve_attn_patch_cfg(model=model, strategy_cfg=strategy_cfg)
        surface = ensure_openclip_kfac_surface(model, attn_patch_cfg=attn_patch_cfg)
        model.to(device)
        plan = select_tracked_parameters(model)
        base = _base_snapshot(model, plan)

        selected_tasks = list(
            resolve_reference_tasks_from_kwargs(
                regularization_cfg=regularization_cfg,
                kwargs=kwargs,
                task=task,
                require_reference=True,
            )
        )
        required_tasks = list(dict.fromkeys(selected_tasks))
        batch_size = int(kwargs.get('batch_size', getattr(train_loader, 'batch_size', 128) or 128))
        num_workers = int(kwargs.get('num_workers', getattr(train_loader, 'num_workers', 0)))
        val_fraction = float(kwargs.get('val_fraction', 0.1))
        seed = int(kwargs.get('seed', 42))

        loaded_refs: list[tuple[str, EkfacTaskCurvatureStats]] = []
        for cache_task in required_tasks:
            if cache_task == task:
                cache_model = model
                cache_loader = train_loader
            else:
                ctx = build_vision_regularizer_task_context(
                    task=cache_task,
                    build_cfg=build_cfg,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    val_fraction=val_fraction,
                    seed=seed,
                )
                cache_model = ctx.model
                cache_loader = ctx.loader
            path, recomputed = self._ensure_cache(
                model=cache_model,
                loader=cache_loader,
                task=cache_task,
                build_cfg=build_cfg,
                config=config,
                attn_patch_cfg=surface['attn_patch_cfg'],
                device=device,
            )
            if run_logger is not None:
                run_logger.log_event(
                    'ekfac_ggn_cache',
                    metrics={},
                    context={'task': cache_task, 'path': str(path), 'recomputed': bool(recomputed)},
                )

        for ref_task in selected_tasks:
            if any(task_name == ref_task for task_name, _ in loaded_refs):
                continue
            if ref_task == task:
                continue
            path = task_cache_path(cache_dir=config.cache_dir, build_cfg=build_cfg, task=ref_task)
            if not path.exists():
                continue
            loaded_refs.append((
                ref_task,
                load_task_ekfac(path, device=device, precision=config.precision),
            ))
        total_examples = max(1, sum(max(1, int(stats.num_examples)) for _, stats in loaded_refs))
        references = [
            (ref_task, float(max(1, int(stats.num_examples))) / float(total_examples), stats)
            for ref_task, stats in loaded_refs
        ]
        ignored_trainable = len(plan.ignored_trainable)
        prepared = PreparedEkfacGgn(
            config=config,
            plan=plan,
            base=base,
            references=references,
            ignored_trainable=int(ignored_trainable),
        )
        info = {
            'ekfac_reference_tasks': len(references),
            'ekfac_matrix_blocks': len(plan.matrix_blocks),
            'ekfac_full_blocks': len(plan.full_blocks),
            'ekfac_ignored_trainable': int(ignored_trainable),
        }
        return prepared, info

    def apply(
        self,
        prepared: PreparedEkfacGgn,
        *,
        model: nn.Module,
        step: int,
        batch_index: int,
        **kwargs,
    ) -> torch.Tensor:
        del batch_index, kwargs
        if not prepared.references or prepared.config.reg_lambda == 0.0 or (int(step) % prepared.config.cadence) != 0:
            return next(model.parameters()).sum() * 0.0
        deltas = _delta_params(model, prepared.base)
        breakdown = compute_ekfac_penalty(deltas, prepared.references)
        loss = (
            prepared.config.reg_lambda * breakdown.loss_reg_matrix
            + prepared.config.reg_lambda * prepared.config.full_block_scaler * breakdown.loss_reg_ffT
            + prepared.config.reg_lambda * prepared.config.projection_scaler * breakdown.loss_reg_proj
            + prepared.config.reg_lambda * prepared.config.full_block_scaler * breakdown.loss_reg_cls
        )
        model._ekfac_ggn_last_breakdown = {  # type: ignore[attr-defined]
            'matrix': float(breakdown.loss_reg_matrix.detach().cpu()),
            'ffT': float(breakdown.loss_reg_ffT.detach().cpu()),
            'projection': float(breakdown.loss_reg_proj.detach().cpu()),
            'class_embedding': float(breakdown.loss_reg_cls.detach().cpu()),
        }
        return loss


register(EkfacGgnRegularizer())
