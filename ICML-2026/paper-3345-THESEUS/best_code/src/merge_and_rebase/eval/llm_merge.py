from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.func import functional_call

from merge_and_rebase.hyperparam_search import (
    SearchEvaluation,
    build_search_planner,
    describe_candidate,
    summarize_search_results,
)
from merge_and_rebase.utils.helpers import load_json, parse_csv

from ..cli_args import (
    add_alpha_args,
    add_config_arg,
    add_device_dtype_args,
    add_logging_args,
    add_merge_io_args,
    add_postmerge_args,
    add_suite_arg,
    add_tasks_arg,
    build_common_eval_overrides,
    build_common_merge_overrides,
    build_logging_overrides,
    build_postmerge_overrides,
    merge_non_none,
    parse_json_object_arg,
)
from ..data.text_loaders import (
    NLI_TASKS,
    NLITaskData,
    NLITokenizedData,
    build_nli_task_data,
    build_nli_tokenized_loader,
    default_head_class_ids_for_task,
)
from ..io.ckpt import align_to_base_keys, load_ckpt, load_into_model
from ..io.peft_helpers import (
    is_peft_adapter_dir_ckpt,
    load_peft_adapter_dir_components,
    normalize_peft_adapter_dir_checkpoint,
)
from ..merge import runtime as _merge_utils
from ..merge import subspaces as _subspaces  # noqa: F401
from ..merge.base import PreparedMergeMethod
from ..merge.methods._common import get_method_params, resolve_merge_weights
from ..merge.registry import get_method, list_methods
from ..merge.subspaces.registry import get_subspace, list_subspaces
from ..merge.task_vectors import default_key_filter
from ..models.text_lm import TextBuildConfig, TextLM
from ..postmerge import PostMergeContext, get_postmerge_method
from ..postmerge.methods.adamerging import prediction_entropy
from ..run_logging import default_summary_path, merge_logging_config, start_run
from .print_utils import pretty_print_task_accuracies
from .utils import stable_method_params_cache_key

_build_merged_state_for_alpha = _merge_utils.build_merged_state_for_alpha
_ensure_peft_cfg_map = _merge_utils.ensure_peft_cfg_map
_extract_peft_components = _merge_utils.extract_peft_components
_get_peft_cfg = _merge_utils.get_peft_cfg
_is_peft_checkpoint = _merge_utils.is_peft_checkpoint
_load_prepared_direction_into_model = _merge_utils.load_prepared_direction_into_model
_prepared_base_direction = _merge_utils.prepared_base_direction
_to_cpu_fp32 = _merge_utils.to_cpu_fp32

NLI_SUITES: dict[str, tuple[str, ...]] = {
    "nli6": tuple(NLI_TASKS),
}


def _resolve_tuned_ckpts(tuned_cfg: Any, *, tasks: list[str] | None = None) -> list[str]:
    if isinstance(tuned_cfg, dict):
        keyed = {str(k).strip().lower(): str(v) for k, v in tuned_cfg.items()}
        if tasks:
            missing = [t for t in tasks if t not in keyed]
            if missing:
                raise ValueError(f"tuned_ckpts is missing task keys: {missing}. Provided keys: {sorted(keyed)}")
            return [keyed[t] for t in tasks]
        return [keyed[k] for k in sorted(keyed)]
    if isinstance(tuned_cfg, (list, tuple)):
        out = [str(x) for x in tuned_cfg]
        if tasks is not None and len(out) != len(tasks):
            raise ValueError(f"tuned_ckpts list length ({len(out)}) must match number of tasks ({len(tasks)}): {tasks}")
        return out
    raise ValueError("tuned_ckpts must be a list/tuple of checkpoint paths (or a task->path dict).")


@dataclass(frozen=True)
class _MergeRuntimeOptions:
    use_stream_prepare: bool
    use_inplace_apply: bool
    use_low_memory_prepare: bool


def _resolve_merge_runtime_options(
    *,
    cfg: dict[str, Any],
    method: Any,
    peft_subspace: str,
    method_params: dict[str, Any],
) -> tuple[_MergeRuntimeOptions, dict[str, Any]]:
    stream_requested = bool(cfg.get("stream_task_arithmetic", True))
    inplace_requested = bool(cfg.get("inplace_task_arithmetic", True))
    low_memory_requested = bool(cfg.get("low_memory_all_methods", True))
    method_name = str(getattr(method, "name", "")).strip()

    out_method_params = dict(method_params)
    if method_name == "tsv_merge" and low_memory_requested and out_method_params.get("low_memory", None) is None:
        out_method_params["low_memory"] = True
        print("Auto-enabled tsv_merge memory-bounded mode (set method_params.low_memory=false to disable).")

    if peft_subspace != "full":
        if cfg.get("stream_task_arithmetic", None) is True:
            print("[warn] stream_task_arithmetic is ignored when peft_subspace != 'full'.")
        if cfg.get("inplace_task_arithmetic", None) is True:
            print("[warn] inplace_task_arithmetic is ignored when peft_subspace != 'full'.")
        if cfg.get("low_memory_all_methods", None) is True:
            print("[warn] low_memory_all_methods is ignored when peft_subspace != 'full'.")
        return _MergeRuntimeOptions(False, False, False), out_method_params

    supports_stream_prepare = method_name == "task_arithmetic"
    if stream_requested and (not supports_stream_prepare) and cfg.get("stream_task_arithmetic", None) is True:
        print("[warn] stream_task_arithmetic is ignored for non-task_arithmetic methods.")

    return (
        _MergeRuntimeOptions(
            use_stream_prepare=stream_requested and supports_stream_prepare,
            use_inplace_apply=inplace_requested,
            use_low_memory_prepare=low_memory_requested,
        ),
        out_method_params,
    )


def _tensor_dict_stats(sd: dict[str, torch.Tensor]) -> tuple[float, float]:
    sq = 0.0
    max_abs = 0.0
    for v in sd.values():
        if not isinstance(v, torch.Tensor):
            continue
        vf = v.detach().float()
        sq += float((vf * vf).sum().item())
        if vf.numel() > 0:
            max_abs = max(max_abs, float(vf.abs().max().item()))
    return math.sqrt(max(0.0, sq)), max_abs


def _load_peft_components_from_adapter_ref(adapter_ref: str) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    try:
        from peft import PeftConfig
        from peft.utils import load_peft_weights
    except Exception as e:
        raise ImportError("Loading PEFT adapters requires `peft`.") from e

    peft_cfg_obj = PeftConfig.from_pretrained(adapter_ref)
    cfg_dict = peft_cfg_obj.to_dict() if hasattr(peft_cfg_obj, "to_dict") else dict(peft_cfg_obj.__dict__)
    peft_state = load_peft_weights(adapter_ref, device="cpu")
    if not isinstance(peft_state, dict):
        raise ValueError(f"Invalid PEFT adapter '{adapter_ref}': adapter weights are not a dict.")
    state = {str(k): v.detach().cpu() for k, v in peft_state.items() if torch.is_tensor(v)}
    if not state:
        raise ValueError(f"Invalid PEFT adapter '{adapter_ref}': adapter state has no tensors.")
    return state, {"default": cfg_dict}


def _load_peft_components_for_subspace(
    *,
    ckpt_ref: str,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], str]:
    resolved_ref = _resolve_checkpoint_reference(str(ckpt_ref))
    p = Path(resolved_ref)
    if p.exists() and p.is_file():
        obj = torch.load(str(p), map_location="cpu", weights_only=False)
        obj = normalize_peft_adapter_dir_checkpoint(obj, checkpoint_path=str(p))
        if is_peft_adapter_dir_ckpt(obj):
            adapter_dir = str(obj["peft_adapter_dir"])
            state, cfg_map = load_peft_adapter_dir_components(adapter_dir, checkpoint_path=str(p))
            return state, cfg_map, adapter_dir
        if isinstance(obj, dict) and isinstance(obj.get("peft_adapter_dir"), str):
            adapter_dir = str(obj["peft_adapter_dir"])
            state, cfg_map = load_peft_adapter_dir_components(adapter_dir, checkpoint_path=str(p))
            return state, cfg_map, adapter_dir
        if _is_peft_checkpoint(obj):
            state, cfg_map = _extract_peft_components(obj)
            return state, cfg_map, resolved_ref
        raise ValueError(f"peft_subspace requires PEFT checkpoints. Got non-PEFT checkpoint payload: {resolved_ref}")

    if _is_adapter_reference(resolved_ref):
        state, cfg_map = _load_peft_components_from_adapter_ref(resolved_ref)
        return state, cfg_map, resolved_ref

    raise ValueError(f"peft_subspace requires PEFT adapter references or PEFT checkpoints. Got: {ckpt_ref}")


@dataclass(frozen=True)
class _AlphaMergeContext:
    method: Any
    prepared: Any
    base_sd_for_merge: dict[str, torch.Tensor]
    tuned_sds_list: Sequence[Mapping[str, torch.Tensor]]
    weights: Any
    method_params: dict[str, Any]
    peft_subspace: str
    subspace: Any
    subspace_prepared: Any
    peft_cfg: dict[str, Any] | None
    peft_state_by_task: dict[str, dict[str, torch.Tensor]] | None
    tasks: list[str] | None
    merge_base_sd: dict[str, torch.Tensor] | None


def _build_merged_state_from_context(ctx: _AlphaMergeContext, *, alpha: float) -> dict[str, torch.Tensor]:
    return _build_merged_state_for_alpha(
        method=ctx.method,
        prepared=ctx.prepared,
        base_sd_for_merge=ctx.base_sd_for_merge,
        tuned_sds_list=ctx.tuned_sds_list,
        weights=ctx.weights,
        method_params=ctx.method_params,
        alpha=float(alpha),
        peft_subspace=ctx.peft_subspace,
        subspace=ctx.subspace,
        subspace_prepared=ctx.subspace_prepared,
        peft_cfg=ctx.peft_cfg,
        peft_state_by_task=ctx.peft_state_by_task,
        tasks=ctx.tasks,
        merge_base_sd=ctx.merge_base_sd,
    )


@dataclass(frozen=True)
class _LoRAFactors:
    a: torch.Tensor
    b: torch.Tensor
    scale: float


def _lookup_layer_pattern(
    pattern: dict[str, Any] | None,
    *,
    layer_key: str,
    default: Any,
) -> Any:
    if not pattern:
        return default
    # Common key variants across PEFT save formats.
    candidates = [layer_key]
    if layer_key.startswith("base_model.model."):
        tail = layer_key[len("base_model.model.") :]
        candidates.append(tail)
        candidates.append(f"model.{tail}")
    elif layer_key.startswith("model."):
        tail = layer_key[len("model.") :]
        candidates.append(tail)
        candidates.append(f"base_model.model.{tail}")
    else:
        candidates.append(f"base_model.model.{layer_key}")
        candidates.append(f"model.{layer_key}")
    for k in candidates:
        if k in pattern:
            return pattern[k]
    return default


def _lora_scaling_for_layer(
    *,
    layer_key: str,
    a: torch.Tensor,
    peft_cfg: dict[str, Any],
) -> float:
    rank_pattern = peft_cfg.get("rank_pattern", {}) if isinstance(peft_cfg.get("rank_pattern", {}), dict) else {}
    alpha_pattern = peft_cfg.get("alpha_pattern", {}) if isinstance(peft_cfg.get("alpha_pattern", {}), dict) else {}
    default_alpha = float(peft_cfg.get("lora_alpha", max(1, int(a.shape[0]))))
    use_rslora = bool(peft_cfg.get("use_rslora", False))

    r_eff = int(a.shape[0])
    r_cfg = int(_lookup_layer_pattern(rank_pattern, layer_key=layer_key, default=r_eff))
    if r_cfg <= 0:
        r_cfg = r_eff
    alpha = float(_lookup_layer_pattern(alpha_pattern, layer_key=layer_key, default=default_alpha))
    denom = (r_cfg**0.5) if use_rslora else float(r_cfg)
    return float(alpha / max(1e-12, denom))


def _strip_known_key_prefixes(key: str) -> list[str]:
    out: list[str] = [key]
    queue: list[str] = [key]
    seen: set[str] = set()
    prefixes = ("base_model.model.", "model.", "module.", "clip_model.model.", "clip_model.")
    while queue:
        cur = queue.pop(0)
        if cur in seen:
            continue
        seen.add(cur)
        for p in prefixes:
            if cur.startswith(p):
                nxt = cur[len(p) :]
                if nxt and nxt not in seen:
                    out.append(nxt)
                    queue.append(nxt)
    uniq: list[str] = []
    seen2: set[str] = set()
    for k in out:
        if k not in seen2:
            uniq.append(k)
            seen2.add(k)
    return uniq


def _aligned_key_from_candidates(
    *,
    candidates: list[str],
    shape: tuple[int, ...],
    base_shapes: dict[str, tuple[int, ...]],
) -> str | None:
    queue = list(candidates)
    seen: set[str] = set()
    while queue:
        k = queue.pop(0)
        if k in seen:
            continue
        seen.add(k)
        if base_shapes.get(k, None) == shape:
            return k

        for p in ("model.", "module.", "clip_model.model.", "clip_model."):
            if k.startswith(p):
                queue.append(k[len(p) :])
        if k.startswith("visual.transformer."):
            queue.append("transformer." + k[len("visual.transformer.") :])
        if k.startswith("transformer."):
            queue.append("visual.transformer." + k[len("transformer.") :])
    return None


def _base_key_candidates_from_lora_prefix(prefix: str) -> list[str]:
    cands: list[str] = []
    for p in _strip_known_key_prefixes(prefix):
        if p.endswith(".base_layer"):
            cands.append(p[: -len(".base_layer")] + ".weight")
        cands.append(f"{p}.weight")
    uniq: list[str] = []
    seen: set[str] = set()
    for c in cands:
        if c not in seen:
            uniq.append(c)
            seen.add(c)
    return uniq


def _base_key_candidates_from_modules_to_save_key(key: str) -> list[str]:
    marker = ".modules_to_save."
    if marker not in key:
        return _strip_known_key_prefixes(key)
    head, rest = key.split(marker, 1)
    parts = rest.split(".")
    if len(parts) >= 2:
        tail = ".".join(parts[1:])
        canonical = f"{head}.{tail}"
    else:
        canonical = head
    return _strip_known_key_prefixes(canonical)


class _LoRAAlignedAdapterView(Mapping[str, torch.Tensor]):
    """
    Aligned tuned-checkpoint view backed by LoRA factors.
    Computes tuned tensors on-demand per key as: base + scale * (B @ A).
    """

    def __init__(
        self,
        *,
        adapter_ref: str,
        base_sd: Mapping[str, torch.Tensor],
        lora_by_key: dict[str, _LoRAFactors],
        direct_overrides: dict[str, torch.Tensor],
    ) -> None:
        self._adapter_ref = str(adapter_ref)
        self._base_sd = base_sd
        self._lora_by_key = dict(lora_by_key)
        self._direct = {k: v.detach().cpu() for k, v in direct_overrides.items()}
        keys = set(self._direct.keys()).union(self._lora_by_key.keys())
        self._keys = tuple(sorted(keys))

    @property
    def adapter_ref(self) -> str:
        return self._adapter_ref

    def __iter__(self) -> Iterator[str]:
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)

    def __getitem__(self, key: str) -> torch.Tensor:
        k = str(key)
        direct = self._direct.get(k, None)
        if direct is not None:
            base = self._base_sd.get(k, None)
            if isinstance(base, torch.Tensor):
                return direct.to(dtype=base.dtype, device="cpu")
            return direct

        factors = self._lora_by_key.get(k, None)
        if factors is None:
            raise KeyError(k)
        base = self._base_sd.get(k, None)
        if not isinstance(base, torch.Tensor):
            raise KeyError(k)

        base_cpu = base.detach().to(device="cpu")
        work_dtype = (
            torch.float32 if base_cpu.dtype in {torch.float16, torch.bfloat16, torch.float32} else torch.float64
        )
        a = factors.a.detach().to(device="cpu", dtype=work_dtype)
        b = factors.b.detach().to(device="cpu", dtype=work_dtype)
        delta = torch.matmul(b, a).mul_(float(factors.scale))
        tuned = base_cpu.to(dtype=work_dtype).add_(delta)
        return tuned.to(dtype=base_cpu.dtype)

    def __repr__(self):
        return f"_LoRAAlignedAdapterView(adapter_ref={self._adapter_ref})"

    def items(self) -> Iterator[tuple[str, torch.Tensor]]:
        for k in self._keys:
            yield k, self[k]


def _resolve_checkpoint_reference(ckpt_ref: str) -> str:
    resolved_ref = str(ckpt_ref)
    p = Path(resolved_ref)
    if p.exists() and p.is_file():
        try:
            obj = torch.load(str(p), map_location="cpu", weights_only=False)
            obj = normalize_peft_adapter_dir_checkpoint(obj, checkpoint_path=str(p))
            if is_peft_adapter_dir_ckpt(obj):
                adapter_dir = str(obj["peft_adapter_dir"])
                print(f"Resolved PEFT adapter checkpoint metadata {resolved_ref} -> {adapter_dir}")
                return adapter_dir
        except Exception:
            return resolved_ref
    return resolved_ref


def _build_lora_aligned_adapter_view(
    *,
    adapter_ref: str,
    base_sd: Mapping[str, torch.Tensor],
) -> _LoRAAlignedAdapterView | None:
    try:
        from peft import PeftConfig
        from peft.utils import load_peft_weights
    except Exception as e:
        raise ImportError("LoRA adapter loading requires `peft`.") from e

    peft_cfg_obj = PeftConfig.from_pretrained(adapter_ref)
    peft_cfg = peft_cfg_obj.to_dict() if hasattr(peft_cfg_obj, "to_dict") else dict(peft_cfg_obj.__dict__)
    peft_type_raw = peft_cfg.get("peft_type", "")
    if hasattr(peft_type_raw, "value"):
        peft_type = str(peft_type_raw.value)
    else:
        peft_type = str(peft_type_raw)
    peft_type = peft_type.split(".")[-1].strip().upper()
    if peft_type != "LORA":
        return None
    if bool(peft_cfg.get("use_dora", False)):
        print(f"[warn] Adapter {adapter_ref} uses DoRA; falling back to full materialization.")
        return None

    peft_state = load_peft_weights(adapter_ref, device="cpu")
    if not isinstance(peft_state, dict):
        return None
    state = {str(k): v.detach().cpu() for k, v in peft_state.items() if isinstance(v, torch.Tensor)}
    if any("lora_magnitude_vector" in k for k in state):
        print(f"[warn] Adapter {adapter_ref} has lora_magnitude_vector; falling back to full materialization.")
        return None

    base_shapes = {k: tuple(v.shape) for k, v in base_sd.items() if isinstance(v, torch.Tensor)}
    a_by_prefix: dict[str, torch.Tensor] = {}
    b_by_prefix: dict[str, torch.Tensor] = {}
    direct_overrides: dict[str, torch.Tensor] = {}

    for k, v in state.items():
        prefix: str | None = None
        if ".lora_A." in k and k.endswith(".weight"):
            prefix = k.split(".lora_A.", 1)[0]
            a_by_prefix[prefix] = v
            continue
        if k.endswith(".lora_A.weight"):
            prefix = k[: -len(".lora_A.weight")]
            a_by_prefix[prefix] = v
            continue
        if ".lora_B." in k and k.endswith(".weight"):
            prefix = k.split(".lora_B.", 1)[0]
            b_by_prefix[prefix] = v
            continue
        if k.endswith(".lora_B.weight"):
            prefix = k[: -len(".lora_B.weight")]
            b_by_prefix[prefix] = v
            continue
        if ".modules_to_save." in k:
            candidates = _base_key_candidates_from_modules_to_save_key(k)
            resolved = _aligned_key_from_candidates(
                candidates=candidates, shape=tuple(v.shape), base_shapes=base_shapes
            )
            if resolved is not None:
                direct_overrides[resolved] = v

    lora_by_key: dict[str, _LoRAFactors] = {}
    for prefix in sorted(set(a_by_prefix.keys()).intersection(b_by_prefix.keys())):
        a = a_by_prefix[prefix]
        b = b_by_prefix[prefix]
        if a.ndim != 2 or b.ndim != 2:
            # Non-linear LoRA layers (e.g., conv/embedding) are not handled by this fast path.
            continue
        shape = (int(b.shape[0]), int(a.shape[1]))
        base_candidates = _base_key_candidates_from_lora_prefix(prefix)
        base_key = _aligned_key_from_candidates(candidates=base_candidates, shape=shape, base_shapes=base_shapes)
        if base_key is None:
            continue
        scale = _lora_scaling_for_layer(layer_key=prefix, a=a, peft_cfg=peft_cfg)
        lora_by_key[base_key] = _LoRAFactors(a=a, b=b, scale=scale)

    if not lora_by_key and not direct_overrides:
        return None
    return _LoRAAlignedAdapterView(
        adapter_ref=adapter_ref,
        base_sd=base_sd,
        lora_by_key=lora_by_key,
        direct_overrides=direct_overrides,
    )


class _LazyAlignedTunedSequence(Sequence[Mapping[str, torch.Tensor]]):
    """
    Loads and aligns each tuned checkpoint on-demand.
    This avoids storing all tuned checkpoints simultaneously in memory.
    """

    def __init__(
        self,
        *,
        tuned_refs: list[str],
        base_sd: dict[str, torch.Tensor],
        build_cfg: TextBuildConfig,
        model: Any,
        force_fp32: bool = True,
        prefer_lora_view: bool = True,
    ) -> None:
        self._refs = [str(x) for x in tuned_refs]
        self._base_sd = base_sd
        self._build_cfg = build_cfg
        self._model = model
        self._force_fp32 = bool(force_fp32)
        self._prefer_lora_view = bool(prefer_lora_view)
        self._index_cache: dict[int, Mapping[str, torch.Tensor]] = {}
        self._adapter_view_cache: dict[str, _LoRAAlignedAdapterView] = {}

    def __len__(self) -> int:
        return len(self._refs)

    def __getitem__(self, idx: int) -> Mapping[str, torch.Tensor]:
        i = int(idx)
        if i < 0:
            i += len(self._refs)
        if i < 0 or i >= len(self._refs):
            raise IndexError(idx)

        cached = self._index_cache.get(i, None)
        if cached is not None:
            return cached

        aligned = _load_aligned_tuned_from_ref(
            ckpt_ref=self._refs[i],
            base_sd=self._base_sd,
            build_cfg=self._build_cfg,
            model=self._model,
            prefer_lora_view=self._prefer_lora_view,
            adapter_view_cache=self._adapter_view_cache,
        )
        if self._force_fp32:
            out = _to_cpu_fp32(aligned)
            del aligned
            return out
        if isinstance(aligned, _LoRAAlignedAdapterView):
            self._index_cache[i] = aligned
        return aligned

    def __repr__(self):
        return f"_LazyAlignedTunedSequence(refs={self._refs}, prefer_lora_view={self._prefer_lora_view})"


def _load_tuned_sequence_for_preparation(
    *,
    tuned_refs: list[str],
    base_sd: dict[str, torch.Tensor],
    build_cfg: TextBuildConfig,
    model: Any,
    strict_load: bool,
    use_low_memory_prepare: bool,
) -> tuple[Sequence[Mapping[str, torch.Tensor]], dict[str, torch.Tensor]]:
    if use_low_memory_prepare:
        print("Using low-memory lazy checkpoint loading for method preparation.")
        tuned_sds_list: Sequence[Mapping[str, torch.Tensor]] = _LazyAlignedTunedSequence(
            tuned_refs=tuned_refs,
            base_sd=base_sd,
            build_cfg=build_cfg,
            model=model,
            force_fp32=False,
            prefer_lora_view=(not strict_load),
        )
        return tuned_sds_list, {k: v.detach().cpu() for k, v in base_sd.items()}

    eager_list: list[dict[str, torch.Tensor]] = []
    for ckpt_ref in tuned_refs:
        aligned = _load_aligned_tuned_from_ref(
            ckpt_ref=ckpt_ref,
            base_sd=base_sd,
            build_cfg=build_cfg,
            model=model,
            prefer_lora_view=(not strict_load),
        )
        eager_list.append(_to_cpu_fp32(aligned))
        del aligned
    return eager_list, _to_cpu_fp32(base_sd)


def _resolve_tasks(tasks_raw: Any, *, suite_name: str | None = None) -> list[str]:
    allowed = list(NLI_TASKS) if suite_name is None else list(NLI_SUITES[suite_name])
    if tasks_raw is None:
        return allowed
    if isinstance(tasks_raw, str):
        if tasks_raw.strip().lower() == "all":
            return allowed
        tasks = [t.strip().lower() for t in parse_csv(tasks_raw)]
    elif isinstance(tasks_raw, (list, tuple)):
        tasks = [str(t).strip().lower() for t in tasks_raw]
    else:
        raise ValueError("tasks must be 'all', a CSV string, or a list.")

    bad = [t for t in tasks if t not in allowed]
    if bad:
        if suite_name is None:
            raise ValueError(f"Unknown tasks: {bad}. Supported: {list(NLI_TASKS)}")
        raise ValueError(f"Unknown tasks for suite '{suite_name}': {bad}. Allowed: {allowed}")
    return tasks


def _resolve_suite_name(raw: Any) -> str | None:
    if raw is None:
        return None
    name = str(raw).strip().lower()
    if not name:
        return None
    if name not in NLI_SUITES:
        raise ValueError(f"Unknown suite '{name}'. Available: {sorted(NLI_SUITES)}")
    return name


def _resolve_task_mask_class(raw: Any) -> dict[str, int | None]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("task_mask_class must be a dict task->masked_class (or null).")
    out: dict[str, int | None] = {}
    for k, v in raw.items():
        key = str(k).strip().lower()
        if not key:
            continue
        out[key] = None if v is None else int(v)
    return out


def _head_class_ids_for_task(
    *,
    task: str,
    task_num_labels: int,
    head_num_labels: int,
    masked_class: int | None,
) -> list[int]:
    if int(task_num_labels) <= 0:
        raise ValueError(f"Invalid task_num_labels for '{task}': {task_num_labels}")
    if int(head_num_labels) <= 0:
        raise ValueError(f"Invalid head_num_labels for '{task}': {head_num_labels}")

    if masked_class is None:
        if int(head_num_labels) == int(task_num_labels):
            return list(range(int(task_num_labels)))

        # Match finetune/train_text.py default mapping for shared 3-way NLI heads.
        t = str(task).strip().lower()
        if int(task_num_labels) == 2 and int(head_num_labels) >= 3:
            if t in {"qnli", "rte"}:
                return [0, 2]
            if t == "scitail":
                return [0, 1]

        out = default_head_class_ids_for_task(task, num_labels=int(head_num_labels))
        if len(out) == int(task_num_labels):
            return out
        raise ValueError(
            f"Could not infer head_class_ids for task '{task}': "
            f"task_num_labels={task_num_labels}, head_num_labels={head_num_labels}. "
            "Set config['task_mask_class'] explicitly."
        )

    masked = int(masked_class)
    if masked < 0 or masked >= int(head_num_labels):
        raise ValueError(
            f"Invalid masked class for task '{task}': {masked}. Allowed range is [0, {int(head_num_labels) - 1}]"
        )
    keep = [c for c in range(int(head_num_labels)) if c != masked]
    if len(keep) != int(task_num_labels):
        raise ValueError(
            f"Mask-derived class ids for task '{task}' are incompatible: keep={keep}, "
            f"task_num_labels={task_num_labels}, head_num_labels={head_num_labels}."
        )
    return keep


def _adapt_legacy_knots_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """
    Accept KnOTS-like nested config and fill this script's flat keys.
    Existing flat keys take precedence.
    """
    out = dict(cfg)

    model_cfg = out.get("model", None)
    if isinstance(model_cfg, dict):
        ptm = str(model_cfg.get("ptm_path", "") or "").strip()
        model_name = str(model_cfg.get("name", "") or "").strip()
        if out.get("model_name_or_path", None) is None:
            out["model_name_or_path"] = ptm if ptm else model_name
        if out.get("tuned_ckpts", None) is None and isinstance(model_cfg.get("bases", None), list):
            out["tuned_ckpts"] = [str(x) for x in model_cfg["bases"]]
        if out.get("peft_config", None) is None and isinstance(model_cfg.get("peft_config", None), dict):
            out["peft_config"] = dict(model_cfg["peft_config"])

    merge_cfg = out.get("task_merge_config", None)
    if isinstance(merge_cfg, dict):
        if out.get("method", None) is None and merge_cfg.get("merge_method", None) is not None:
            out["method"] = str(merge_cfg["merge_method"])
        if out.get("alpha", None) is None and merge_cfg.get("scaling_coeffs", None) is not None:
            sc = merge_cfg["scaling_coeffs"]
            if isinstance(sc, (int, float)):
                out["alpha"] = float(sc)

    if out.get("eval_mode", None) is None:
        eval_type = str(out.get("eval_type", "")).strip().lower()
        if eval_type == "logits":
            out["eval_mode"] = "head_logits"

    if out.get("tasks", None) is None and isinstance(out.get("dataset", None), list):
        ds_names = []
        mask_map: dict[str, int | None] = {}
        for row in out["dataset"]:
            if not isinstance(row, dict):
                continue
            n = str(row.get("name", "")).strip().lower()
            if not n:
                continue
            ds_names.append(n)
            mc = row.get("mask_class", None)
            mask_map[n] = None if mc is None else int(mc)
        if ds_names:
            out["tasks"] = ",".join(ds_names)
        if mask_map and out.get("task_mask_class", None) is None:
            out["task_mask_class"] = mask_map

    return out


def _default_prompt_for_task(task_data: NLITaskData) -> str:
    label_space = ", ".join(task_data.label_texts)
    return (
        "You are an NLI classifier.\n"
        f"Given a premise and a hypothesis, predict one label from: {label_space}.\n"
        "Premise: {premise}\n"
        "Hypothesis: {hypothesis}\n"
        "Label:"
    )


def _resolve_fine_tuned_acc(
    *,
    cfg: dict[str, Any],
    tasks: list[str],
) -> dict[str, float] | None:
    raw = cfg.get("fine_tuned_acc", None)
    if isinstance(raw, dict):
        out: dict[str, float] = {}
        for k, v in raw.items():
            try:
                out[str(k).strip().lower()] = float(v)
            except Exception:
                continue
        missing = [t for t in tasks if t not in out]
        if missing:
            print(
                f"[warn] fine_tuned_acc missing tasks {missing}. Normalized accuracy will be skipped for missing tasks."
            )
        return out
    if raw is None:
        return None
    raise ValueError("fine_tuned_acc must be a dict when provided.")


def _to_unit_acc(ref_acc: float) -> float:
    v = float(ref_acc)
    # Backward compatibility: accept percentage inputs (e.g., 46.7) and unit inputs (e.g., 0.467).
    if v > 1.0:
        v = v / 100.0
    return v


def _normalized_acc(acc: float, fine_tuned_acc_ref: float) -> float:
    denom = _to_unit_acc(fine_tuned_acc_ref)
    if denom <= 0:
        return 0.0
    return float(acc) / denom


def _resolve_eval_mode(eval_mode: str, task_heads_path: str | None) -> str:
    mode = str(eval_mode).strip().lower()
    if mode == "auto":
        return "head_logits" if task_heads_path else "prompt"
    if mode not in {"prompt", "head_logits"}:
        raise ValueError("eval_mode must be one of: auto, prompt, head_logits")
    return mode


def _load_task_heads(path: str) -> dict[str, Any]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        raise ValueError(f"task_heads file must contain a dict. Got: {type(obj)}")
    out: dict[str, Any] = {}
    for k, v in obj.items():
        out[str(k).strip().lower()] = v
    return out


def _task_head_tensor_for_param(
    *,
    task_key: str,
    name: str,
    param: torch.Tensor,
    value: torch.Tensor,
    head_class_ids: list[int] | None = None,
) -> torch.Tensor:
    tgt = param.detach().clone()
    src = value.to(device=tgt.device, dtype=tgt.dtype)
    if tuple(src.shape) == tuple(tgt.shape):
        return src

    mapped_class_ids: list[int] | None = None
    if head_class_ids is not None:
        mapped_class_ids = [int(x) for x in head_class_ids]
        if len(set(mapped_class_ids)) != len(mapped_class_ids):
            raise ValueError(f"head_class_ids for task '{task_key}' must be unique. Got: {mapped_class_ids}")

    if mapped_class_ids is not None:
        if (
            name.endswith("classification_head.out_proj.weight")
            and src.ndim == 2
            and tgt.ndim == 2
            and src.shape[0] == len(mapped_class_ids)
            and src.shape[1] == tgt.shape[1]
        ):
            if min(mapped_class_ids) < 0 or max(mapped_class_ids) >= tgt.shape[0]:
                raise ValueError(
                    f"head_class_ids out of range for task '{task_key}', param '{name}': "
                    f"ids={mapped_class_ids}, target_rows={tgt.shape[0]}"
                )
            for i, cls_id in enumerate(mapped_class_ids):
                tgt[int(cls_id)].copy_(src[i])
            return tgt
        if (
            name.endswith("classification_head.out_proj.bias")
            and src.ndim == 1
            and tgt.ndim == 1
            and src.shape[0] == len(mapped_class_ids)
        ):
            if min(mapped_class_ids) < 0 or max(mapped_class_ids) >= tgt.shape[0]:
                raise ValueError(
                    f"head_class_ids out of range for task '{task_key}', param '{name}': "
                    f"ids={mapped_class_ids}, target_rows={tgt.shape[0]}"
                )
            for i, cls_id in enumerate(mapped_class_ids):
                tgt[int(cls_id)].copy_(src[i])
            return tgt

    if name.endswith("classification_head.out_proj.weight"):
        if src.ndim == 2 and tgt.ndim == 2 and src.shape[1] == tgt.shape[1] and src.shape[0] < tgt.shape[0]:
            tgt[: src.shape[0]].copy_(src)
            return tgt
    if name.endswith("classification_head.out_proj.bias"):
        if src.ndim == 1 and tgt.ndim == 1 and src.shape[0] < tgt.shape[0]:
            tgt[: src.shape[0]].copy_(src)
            return tgt

    raise ValueError(
        f"Head shape mismatch for task '{task_key}', param '{name}': "
        f"model={tuple(tgt.shape)} payload={tuple(src.shape)}"
    )


def _task_head_param_overrides(
    *,
    model: Any,
    task: str,
    task_heads: dict[str, Any],
    head_key_pattern: str,
    head_class_ids: list[int] | None = None,
) -> dict[str, torch.Tensor]:
    task_key = str(task).strip().lower()
    if task_key not in task_heads:
        raise KeyError(f"Task '{task_key}' not found in task_heads.")
    payload = task_heads[task_key]
    named_params = {n: p for n, p in model.named_parameters()}
    pattern = str(head_key_pattern)
    out: dict[str, torch.Tensor] = {}

    def _add_override(name: str, param: torch.Tensor, value: torch.Tensor) -> None:
        out[name] = _task_head_tensor_for_param(
            task_key=task_key,
            name=name,
            param=param,
            value=value,
            head_class_ids=head_class_ids,
        )

    if isinstance(payload, torch.Tensor):
        cands = [(n, p) for n, p in named_params.items() if pattern in n and tuple(p.shape) == tuple(payload.shape)]
        if not cands:
            by_shape = [(n, p) for n, p in named_params.items() if tuple(p.shape) == tuple(payload.shape)]
            preferred = [
                (n, p)
                for n, p in by_shape
                if n.endswith("score.weight")
                or n.endswith("classifier.weight")
                or n.endswith("classification_head.weight")
            ]
            if len(preferred) == 1:
                cands = preferred
            elif len(by_shape) == 1:
                cands = by_shape
        if len(cands) != 1:
            names = [n for n, _ in cands]
            shape_only = [n for n, p in named_params.items() if tuple(p.shape) == tuple(payload.shape)]
            raise ValueError(
                f"Could not uniquely match tensor head for task '{task_key}'. "
                f"pattern='{pattern}', shape={tuple(payload.shape)}, candidates={names}, "
                f"shape_only_matches={shape_only[:8]}"
            )
        name, param = cands[0]
        _add_override(name, param, payload)
        return out

    if not isinstance(payload, dict):
        raise ValueError(f"task_heads['{task_key}'] must be a Tensor or dict. Got: {type(payload)}")

    for hk, hv in payload.items():
        if not isinstance(hv, torch.Tensor):
            continue
        key = str(hk)
        if key in named_params:
            _add_override(key, named_params[key], hv)
            continue

        suffix_matches = [(n, p) for n, p in named_params.items() if pattern in n and n.endswith(key)]
        if len(suffix_matches) == 1:
            n, p = suffix_matches[0]
            _add_override(n, p, hv)
            continue
        if len(suffix_matches) == 0:
            any_suffix_matches = [(n, p) for n, p in named_params.items() if n.endswith(key)]
            if len(any_suffix_matches) == 1:
                n, p = any_suffix_matches[0]
                _add_override(n, p, hv)
                continue
        if len(suffix_matches) > 1:
            raise ValueError(
                f"Ambiguous suffix match for task '{task_key}', key='{key}', "
                f"matches={[n for n, _ in suffix_matches]}"
            )
        raise KeyError(f"No parameter match for task '{task_key}' head key '{key}'.")
    return out


def _inject_task_head(
    *,
    model: Any,
    task: str,
    task_heads: dict[str, Any],
    head_key_pattern: str,
    head_class_ids: list[int] | None = None,
) -> None:
    task_key = str(task).strip().lower()
    if task_key not in task_heads:
        raise KeyError(f"Task '{task_key}' not found in task_heads.")
    payload = task_heads[task_key]

    named_params = {n: p for n, p in model.named_parameters()}
    pattern = str(head_key_pattern)

    def _copy_param(name: str, param: torch.Tensor, value: torch.Tensor) -> None:
        param.copy_(
            _task_head_tensor_for_param(
                task_key=task_key,
                name=name,
                param=param,
                value=value,
                head_class_ids=head_class_ids,
            )
        )

    with torch.no_grad():
        if isinstance(payload, torch.Tensor):
            cands = [(n, p) for n, p in named_params.items() if pattern in n and tuple(p.shape) == tuple(payload.shape)]
            if not cands:
                # Fallback for plain sequence-classification models (no modules_to_save wrapper).
                by_shape = [(n, p) for n, p in named_params.items() if tuple(p.shape) == tuple(payload.shape)]
                preferred = [
                    (n, p)
                    for n, p in by_shape
                    if n.endswith("score.weight")
                    or n.endswith("classifier.weight")
                    or n.endswith("classification_head.weight")
                ]
                if len(preferred) == 1:
                    cands = preferred
                elif len(by_shape) == 1:
                    cands = by_shape
            if len(cands) != 1:
                names = [n for n, _ in cands]
                shape_only = [n for n, p in named_params.items() if tuple(p.shape) == tuple(payload.shape)]
                raise ValueError(
                    f"Could not uniquely match tensor head for task '{task_key}'. "
                    f"pattern='{pattern}', shape={tuple(payload.shape)}, candidates={names}, "
                    f"shape_only_matches={shape_only[:8]}"
                )
            name, param = cands[0]
            _copy_param(name, param, payload)
            return

        if not isinstance(payload, dict):
            raise ValueError(f"task_heads['{task_key}'] must be a Tensor or dict. Got: {type(payload)}")

        for hk, hv in payload.items():
            if not isinstance(hv, torch.Tensor):
                continue
            key = str(hk)
            if key in named_params:
                p = named_params[key]
                _copy_param(key, p, hv)
                continue

            # Fallback: suffix match when payload keys are local/submodule keys.
            suffix_matches = [(n, p) for n, p in named_params.items() if pattern in n and n.endswith(key)]
            if len(suffix_matches) == 1:
                n, p = suffix_matches[0]
                _copy_param(n, p, hv)
                continue
            if len(suffix_matches) == 0:
                # Fallback when model does not use pattern wrappers.
                any_suffix_matches = [(n, p) for n, p in named_params.items() if n.endswith(key)]
                if len(any_suffix_matches) == 1:
                    n, p = any_suffix_matches[0]
                    _copy_param(n, p, hv)
                    continue
            if len(suffix_matches) > 1:
                raise ValueError(
                    f"Ambiguous suffix match for task '{task_key}', key='{key}', "
                    f"matches={[n for n, _ in suffix_matches]}"
                )
            raise KeyError(f"No parameter match for task '{task_key}' head key '{key}'.")


def _is_adapter_reference(ref: str) -> bool:
    p = Path(ref)
    if p.exists():
        if p.is_file():
            return False
        if p.is_dir():
            return (p / "adapter_config.json").exists()
    if "/" in ref and not ref.endswith((".pt", ".bin", ".safetensors", ".ckpt", ".pth")):
        return True
    return False


def _load_aligned_tuned_from_ref(
    *,
    ckpt_ref: str,
    base_sd: dict[str, torch.Tensor],
    build_cfg: TextBuildConfig,
    model: Any,
    prefer_lora_view: bool = False,
    adapter_view_cache: dict[str, _LoRAAlignedAdapterView] | None = None,
) -> Mapping[str, torch.Tensor]:
    resolved_ref = _resolve_checkpoint_reference(str(ckpt_ref))
    used_adapter = False

    if _is_adapter_reference(resolved_ref):
        if prefer_lora_view:
            if adapter_view_cache is not None and resolved_ref in adapter_view_cache:
                view = adapter_view_cache[resolved_ref]
                print(f"Reusing cached LoRA adapter view: {resolved_ref} ({len(view)} aligned tensors)")
                return view
            try:
                view = _build_lora_aligned_adapter_view(adapter_ref=resolved_ref, base_sd=base_sd)
            except Exception as exc:
                print(
                    f"[warn] LoRA adapter fast-path failed for {resolved_ref}: {exc}. Falling back to full materialization."
                )
                view = None
            if view is not None:
                if adapter_view_cache is not None:
                    adapter_view_cache[resolved_ref] = view
                print(f"Loaded LoRA adapter view {resolved_ref}: {len(view)} aligned tensors")
                return view

        print(f"Materializing HF/PEFT adapter into full checkpoint: {resolved_ref}")
        sd = _materialize_adapter_state_dict(
            adapter_ref=resolved_ref,
            build_cfg=build_cfg,
            model=model,
        )
        used_adapter = True
    else:
        sd = load_ckpt(resolved_ref)

    try:
        aligned: Mapping[str, torch.Tensor] = align_to_base_keys(sd, base_sd)
        if not aligned:
            raise ValueError(
                f"No tensors from tuned checkpoint aligned to base keys: {resolved_ref}. "
                "Check checkpoint key prefixes and model compatibility."
            )
        print(f"Aligned tuned checkpoint {resolved_ref}: {len(aligned)} tensors")
        return aligned
    finally:
        del sd
        if used_adapter:
            miss, unexp = load_into_model(model, base_sd, strict=False)
            print(f"Restored base model after adapter materialization. missing={miss}, unexpected={unexp}")


def _prepare_task_arithmetic_streaming(
    *,
    base_sd: dict[str, torch.Tensor],
    tuned_refs: list[str],
    weights: Any,
    strict: bool,
    build_cfg: TextBuildConfig,
    model: Any,
    prefer_lora_view: bool = False,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    if not tuned_refs:
        raise ValueError("No tuned checkpoints provided for streaming task_arithmetic.")

    w = resolve_merge_weights(len(tuned_refs), weights)
    active_keys: set[str] | None = None
    direction: dict[str, torch.Tensor] = {}
    expected_base_keys = {k for k, v in base_sd.items() if default_key_filter(k, v)}

    for i, ckpt_ref in enumerate(tuned_refs):
        aligned = _load_aligned_tuned_from_ref(
            ckpt_ref=ckpt_ref,
            base_sd=base_sd,
            build_cfg=build_cfg,
            model=model,
            prefer_lora_view=prefer_lora_view,
        )
        current_keys: set[str] = set()
        wi = float(w[i])

        for k, t in aligned.items():
            b = base_sd.get(k, None)
            if b is None:
                continue
            if not default_key_filter(k, b):
                continue
            if not isinstance(t, torch.Tensor):
                continue
            if t.shape != b.shape:
                continue
            current_keys.add(k)

        if active_keys is None:
            active_keys = set(current_keys)
            for k in active_keys:
                b = base_sd[k]
                t = aligned[k].to(dtype=b.dtype, device="cpu")
                direction[k] = wi * (t - b)
        else:
            shared = active_keys.intersection(current_keys)
            dropped = active_keys - shared
            for k in dropped:
                direction.pop(k, None)
            for k in shared:
                b = base_sd[k]
                d = direction[k]
                t = aligned[k].to(dtype=d.dtype, device="cpu")
                d.add_(wi * (t - b.to(dtype=d.dtype, device="cpu")))
            active_keys = shared

        print(
            f"[stream] processed {i + 1}/{len(tuned_refs)} tuned checkpoints; "
            f"active merged keys={0 if active_keys is None else len(active_keys)}"
        )
        del aligned

    if not active_keys:
        raise RuntimeError("Streaming task_arithmetic found no common mergeable keys across checkpoints.")
    if strict and active_keys != expected_base_keys:
        missing = sorted(expected_base_keys - active_keys)
        raise ValueError(
            "Strict mode: tuned checkpoints do not match base floating-point keyspace.\n"
            f"Missing keys (sample): {missing[:10]}"
        )
    return base_sd, direction


def _build_hf_model_for_materialization(
    *,
    build_cfg: TextBuildConfig,
):
    try:
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoModelForSeq2SeqLM,
            AutoModelForSequenceClassification,
        )
    except Exception as e:
        raise ImportError("Hugging Face materialization requires transformers.") from e

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    torch_dtype = dtype_map.get(build_cfg.dtype, None)
    arch = str(build_cfg.model_arch).strip().lower()
    if arch not in {"llama", "t5", "auto"}:
        raise ValueError("model_arch must be one of: llama, t5, auto")
    kind = str(build_cfg.model_kind).strip().lower()
    common = {
        "pretrained_model_name_or_path": build_cfg.model_name_or_path,
        "trust_remote_code": bool(build_cfg.trust_remote_code),
        "torch_dtype": torch_dtype,
    }
    if kind == "sequence_classification":
        model = AutoModelForSequenceClassification.from_pretrained(
            **common,
            num_labels=int(build_cfg.num_labels),
        )
    elif kind == "causal_lm":
        hf_cfg = AutoConfig.from_pretrained(
            build_cfg.model_name_or_path,
            trust_remote_code=bool(build_cfg.trust_remote_code),
        )
        is_encoder_decoder = bool(getattr(hf_cfg, "is_encoder_decoder", False))
        use_seq2seq = (arch == "t5") or (arch == "auto" and is_encoder_decoder)
        if use_seq2seq:
            model = AutoModelForSeq2SeqLM.from_pretrained(**common)
        else:
            model = AutoModelForCausalLM.from_pretrained(**common)
    else:
        raise ValueError("model_kind must be one of: causal_lm, sequence_classification")
    return model.to(build_cfg.device)


def _materialize_adapter_state_dict(
    *,
    adapter_ref: str,
    build_cfg: TextBuildConfig,
    model: Any | None = None,
) -> dict[str, torch.Tensor]:
    try:
        from peft import PeftModel
    except Exception as e:
        raise ImportError("Adapter materialization from HF requires `peft`.") from e

    owns_model = model is None
    if model is None:
        model = _build_hf_model_for_materialization(build_cfg=build_cfg)
    peft_model = PeftModel.from_pretrained(
        model,
        adapter_ref,
        is_trainable=False,
    )
    if not hasattr(peft_model, "merge_and_unload"):
        raise RuntimeError(f"PEFT model from '{adapter_ref}' does not support merge_and_unload().")
    merged = peft_model.merge_and_unload()
    sd = {k: v.detach().cpu() for k, v in merged.state_dict().items() if torch.is_tensor(v)}

    if owns_model:
        del merged
        del peft_model
        del model
    if torch.cuda.is_available() and str(build_cfg.device).lower() != "cpu":
        torch.cuda.empty_cache()
    return sd


def main() -> None:
    run_logger = None
    p = argparse.ArgumentParser("Merge text-model checkpoints and evaluate on NLI benchmarks.")
    add_config_arg(p)
    add_suite_arg(p, choices=sorted(NLI_SUITES.keys()), default=None)
    p.add_argument(
        "--stream-task-arithmetic",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Stream tuned checkpoints one-by-one for task_arithmetic to reduce RAM usage.",
    )
    p.add_argument(
        "--inplace-task-arithmetic",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Apply task_arithmetic prepared merge directly into model params per alpha to avoid merged state_dict allocation.",
    )
    p.add_argument(
        "--low-memory-all-methods",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use lazy on-demand tuned checkpoint loading for non-task_arithmetic methods.",
    )
    p.add_argument(
        "--eval-single-task-tuned",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Evaluate each tuned checkpoint on its matching task for sanity-checking.",
    )
    p.add_argument(
        "--zero-shot-only",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Skip all merge/tuned-checkpoint logic and run only base-model zero-shot evaluation.",
    )
    p.add_argument(
        "--allow-prompt-eval",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable prompt-mode evaluation explicitly. Disabled by default.",
    )

    # LLM build
    p.add_argument("--model-name-or-path", type=str, default=None)
    p.add_argument(
        "--model-arch",
        type=str,
        default=None,
        choices=["llama", "t5", "auto"],
        help="Text model architecture family for loading/materialization.",
    )
    add_device_dtype_args(p, device_default=None, dtype_default=None)
    p.add_argument(
        "--model-kind",
        type=str,
        default=None,
        choices=["causal_lm", "sequence_classification"],
        help="Model wrapper kind. If omitted, auto-switches to sequence_classification in head_logits mode.",
    )
    p.add_argument("--num-labels", type=int, default=None, help="Num labels for sequence_classification model kind.")
    p.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--use-fast-tokenizer", action=argparse.BooleanOptionalAction, default=None)

    # NLI eval
    add_tasks_arg(p, default=None, help_text=f"CSV task list or 'all'. Supported: {', '.join(NLI_TASKS)}")
    p.add_argument("--split", type=str, default=None, choices=["train", "validation", "test"])
    p.add_argument("--max-samples-per-task", type=int, default=None)
    p.add_argument(
        "--prompt-template", type=str, default=None, help="Optional prompt template with {premise}/{hypothesis}."
    )
    p.add_argument("--max-prompt-tokens", type=int, default=None)
    p.add_argument("--print-every", type=int, default=None, help="Progress print frequency during task evaluation.")
    p.add_argument(
        "--single-acc-zero-shot",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Also compute pre-merge base-model accuracy per task (reported only; not used for normalization).",
    )
    p.add_argument(
        "--fine-tuned-acc-json",
        type=str,
        default=None,
        help="Optional JSON dict task->reference accuracy for normalization (accepts 0-1 or percent values).",
    )
    p.add_argument(
        "--eval-mode",
        type=str,
        default=None,
        choices=["auto", "prompt", "head_logits"],
        help="Evaluation mode. auto => head_logits when --task-heads is provided else prompt.",
    )
    p.add_argument("--task-heads", type=str, default=None, help="Path to heads.pt mapping task -> head payload.")
    p.add_argument(
        "--head-key-pattern",
        type=str,
        default=None,
        help="Substring used to find head params (default: modules_to_save).",
    )
    p.add_argument("--batch-size", type=int, default=None, help="Batch size for head_logits mode.")
    p.add_argument("--num-workers", type=int, default=None, help="Num workers for head_logits mode.")
    p.add_argument("--max-length", type=int, default=None, help="Tokenization max length for head_logits mode.")

    add_merge_io_args(
        p,
        method_choices=list_methods(),
        subspace_choices=list_subspaces(),
        tuned_help="Paths to tuned checkpoints.",
        weights_help="Optional merge weights.",
        strict_mode="bool_optional",
    )
    # Alpha
    add_alpha_args(
        p,
        alpha_default=None,
        alpha_min_default=None,
        alpha_max_default=None,
        alpha_step_default=None,
        alpha_search_default=None,
    )
    add_postmerge_args(p)
    add_logging_args(p)

    args = p.parse_args()

    method_params_cli = parse_json_object_arg(args.method_params, arg_name="--method-params")
    postmerge_cli = build_postmerge_overrides(args).get("postmerge", {})

    cfg: dict[str, Any] = {}
    if args.config is not None:
        cfg = load_json(args.config)
        if not isinstance(cfg, dict):
            raise ValueError("--config must contain a JSON object.")
        cfg = _adapt_legacy_knots_config(cfg)

    cli_overrides = {
        "model_name_or_path": args.model_name_or_path,
        "stream_task_arithmetic": args.stream_task_arithmetic,
        "inplace_task_arithmetic": args.inplace_task_arithmetic,
        "low_memory_all_methods": args.low_memory_all_methods,
        "eval_single_task_tuned": args.eval_single_task_tuned,
        "zero_shot_only": args.zero_shot_only,
        "allow_prompt_eval": args.allow_prompt_eval,
        "model_arch": args.model_arch,
        "model_kind": args.model_kind,
        "num_labels": args.num_labels,
        "trust_remote_code": args.trust_remote_code,
        "use_fast_tokenizer": args.use_fast_tokenizer,
        "split": args.split,
        "max_samples_per_task": args.max_samples_per_task,
        "prompt_template": args.prompt_template,
        "max_prompt_tokens": args.max_prompt_tokens,
        "print_every": args.print_every,
        "single_acc_zero_shot": args.single_acc_zero_shot,
        "fine_tuned_acc": (json.loads(args.fine_tuned_acc_json) if args.fine_tuned_acc_json else None),
        "eval_mode": args.eval_mode,
        "task_heads": args.task_heads,
        "head_key_pattern": args.head_key_pattern,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "max_length": args.max_length,
    }
    cli_overrides = merge_non_none(cli_overrides, build_common_eval_overrides(args))
    cli_overrides = merge_non_none(
        cli_overrides,
        build_common_merge_overrides(args=args, method_params=method_params_cli, strict_as_bool=False),
    )
    cfg = merge_non_none(cfg, cli_overrides)
    if postmerge_cli:
        existing_postmerge = cfg.get("postmerge", {})
        if existing_postmerge is None:
            existing_postmerge = {}
        if not isinstance(existing_postmerge, dict):
            raise ValueError("config['postmerge'] must be a dict when provided.")
        cfg["postmerge"] = merge_non_none(dict(existing_postmerge), dict(postmerge_cli))
    logging_cfg = merge_logging_config(cfg.get("logging", {}), build_logging_overrides(args))
    cfg["logging"] = logging_cfg

    model_name_or_path = cfg.get("model_name_or_path", None)
    if not isinstance(model_name_or_path, str) or not model_name_or_path.strip():
        raise ValueError("You must provide --model-name-or-path (or config['model_name_or_path']).")

    method_params = dict(get_method_params({"method_params": cfg.get("method_params", {})}))

    strict_load = bool(cfg.get("strict_load", False))
    merge_weights = cfg.get("weights", None)
    merge_weights_raw = merge_weights
    peft_subspace = str(cfg.get("peft_subspace", "full"))
    task_heads_path = cfg.get("task_heads", None)
    if task_heads_path is not None:
        task_heads_path = str(task_heads_path)
    eval_mode = _resolve_eval_mode(str(cfg.get("eval_mode", "auto")), task_heads_path)
    head_key_pattern = str(cfg.get("head_key_pattern", "modules_to_save"))

    search_planner = build_search_planner(cfg=cfg, base_method_params=method_params)
    run_summary_path = default_summary_path(
        entrypoint="eval.llm_merge",
        logging_cfg=logging_cfg,
        default_parent=(Path(str(cfg["save_merged"])).parent if cfg.get("save_merged") else None),
    )

    cfg_model_kind = cfg.get("model_kind", None)
    if cfg_model_kind is None:
        model_kind = "sequence_classification" if eval_mode == "head_logits" else "causal_lm"
    else:
        model_kind = str(cfg_model_kind)
    model_arch = str(cfg.get("model_arch", "auto"))
    num_labels = int(cfg.get("num_labels", 3))
    build_cfg = TextBuildConfig(
        model_name_or_path=str(model_name_or_path),
        model_arch=model_arch,
        device=str(cfg.get("device", "cuda")),
        dtype=cfg.get("dtype", None),
        model_kind=model_kind,
        num_labels=num_labels,
        trust_remote_code=bool(cfg.get("trust_remote_code", False)),
        use_fast_tokenizer=bool(cfg.get("use_fast_tokenizer", True)),
    )
    run_logger = start_run(
        entrypoint="eval.llm_merge",
        logging_cfg=logging_cfg,
        summary_path=run_summary_path,
        metadata={
            "config_path": args.config,
            "resolved_config": cfg,
            "summary_path": str(run_summary_path),
        },
    )
    subspace_artifact_dir = run_summary_path.with_name(f"{run_summary_path.stem}.artifacts")
    llm = TextLM.build(build_cfg)
    print(f"Using eval mode: {eval_mode}")
    print(f"Using model arch: {model_arch}")
    print(f"Using model kind: {model_kind}")
    allow_prompt_eval = bool(cfg.get("allow_prompt_eval", False))
    if eval_mode == "prompt" and not allow_prompt_eval:
        raise ValueError(
            "Prompt evaluation is disabled for llm_merge unless explicitly enabled. "
            "Set --allow-prompt-eval (or config['allow_prompt_eval']=true)."
        )
    if eval_mode == "prompt" and model_kind != "causal_lm":
        raise ValueError(
            "prompt mode requires model_kind='causal_lm'. "
            f"Got model_kind='{model_kind}'. "
            "Set --model-kind causal_lm (or config['model_kind']='causal_lm')."
        )
    if eval_mode == "head_logits" and model_kind != "sequence_classification":
        raise ValueError("head_logits mode requires model_kind='sequence_classification'.")

    base_ckpt = cfg.get("base_ckpt", None)
    if base_ckpt is None:
        print(f"Using pretrained base weights from {build_cfg.model_name_or_path}")
    else:
        print(f"Loading base checkpoint from {base_ckpt}")
        sd0 = load_ckpt(str(base_ckpt))
        miss, unexp = load_into_model(llm.model, sd0, strict=strict_load)
        print(f"Loaded base checkpoint into model. missing={miss}, unexpected={unexp}")
    base_sd = {k: v.detach().cpu() for k, v in llm.model.state_dict().items()}

    suite_name = _resolve_suite_name(cfg.get("suite", None))
    if suite_name is not None:
        print(f"Using suite: {suite_name}")
    tasks = _resolve_tasks(cfg.get("tasks", None), suite_name=suite_name)

    tuned_cfg = cfg.get("tuned_ckpts", None)
    zero_shot_only = bool(cfg.get("zero_shot_only", False)) or (tuned_cfg is None)
    tuned_ckpts: list[str] = []
    if zero_shot_only:
        if tuned_cfg is not None:
            print("zero_shot_only=True: ignoring tuned_ckpts and merge method.")
        print("No tuned_ckpts provided; running zero-shot-only evaluation.")
    else:
        tuned_ckpts = _resolve_tuned_ckpts(tuned_cfg, tasks=tasks)
        if not tuned_ckpts:
            raise ValueError("No tuned checkpoints provided.")

    method = get_method(str(cfg.get("method", "task_arithmetic"))) if not zero_shot_only else None
    prepared = None
    runtime_options = _MergeRuntimeOptions(False, False, False)
    if method is not None:
        runtime_options, method_params = _resolve_merge_runtime_options(
            cfg=cfg,
            method=method,
            peft_subspace=peft_subspace,
            method_params=method_params,
        )
    use_stream_task_arithmetic = runtime_options.use_stream_prepare
    use_inplace_task_arithmetic = runtime_options.use_inplace_apply
    use_low_memory_all_methods = runtime_options.use_low_memory_prepare

    tuned_sds_list: Sequence[Mapping[str, torch.Tensor]] = []
    peft_state_by_task: dict[str, dict[str, torch.Tensor]] = {}
    peft_cfg_map: dict[str, Any] | None = None
    peft_cfg: dict[str, Any] | None = None
    subspace = None
    subspace_prepared = None
    merge_base_sd = _to_cpu_fp32(base_sd)
    if not zero_shot_only and peft_subspace != "full":
        if method is None:
            raise RuntimeError("Internal error: merge method was not initialized.")

        for i, task in enumerate(tasks):
            state, cfg_map, src_ref = _load_peft_components_for_subspace(ckpt_ref=tuned_ckpts[i])
            peft_state_by_task[task] = state
            peft_cfg_map = _ensure_peft_cfg_map(peft_cfg_map, cfg_map)
            print(f"Loaded PEFT checkpoint for task '{task}' from {src_ref}: tensors={len(state)}")

        if peft_cfg_map is None:
            raise ValueError(f"peft_subspace='{peft_subspace}' requires peft_config in checkpoints.")
        peft_cfg = _get_peft_cfg(peft_cfg_map)
        subspace = get_subspace(peft_subspace)
        subspace_prepared = subspace.prepare(
            lora_by_task=peft_state_by_task,
            peft_cfg=peft_cfg,
            method_params=method_params,
            weights=resolve_merge_weights(len(tasks), merge_weights),
            artifact_dir=subspace_artifact_dir,
        )
        if getattr(subspace_prepared, "merge_weight_override", None) is not None:
            merge_weights = list(subspace_prepared.merge_weight_override)
        projected_by_task = subspace.project(
            subspace_prepared,
            lora_by_task=peft_state_by_task,
            peft_cfg=peft_cfg,
        )
        missing_projected = [t for t in tasks if t not in projected_by_task]
        if missing_projected:
            raise ValueError(f"Subspace projection missing task outputs: {missing_projected}")
        tuned_sds_list = [projected_by_task[t] for t in tasks]
        if not tuned_sds_list or not tuned_sds_list[0]:
            raise ValueError("Subspace projection returned no mergeable tensors.")
        base_sd_for_merge = {k: torch.zeros_like(v) for k, v in tuned_sds_list[0].items()}
        base_sd_for_merge = _to_cpu_fp32(base_sd_for_merge)
        use_stream_task_arithmetic = False
        use_inplace_task_arithmetic = False
        print(f"Using PEFT subspace '{peft_subspace}' with {len(base_sd_for_merge)} projected tensors.")
    elif not zero_shot_only and use_stream_task_arithmetic:
        print("Using low-memory task_arithmetic streaming mode.")
        base_sd_for_merge, direction = _prepare_task_arithmetic_streaming(
            base_sd=base_sd,
            tuned_refs=tuned_ckpts,
            weights=merge_weights,
            strict=strict_load,
            build_cfg=build_cfg,
            model=llm.model,
            prefer_lora_view=(not strict_load),
        )
        prepared = (base_sd_for_merge, direction)
    elif not zero_shot_only and method is not None:
        tuned_sds_list, base_sd_for_merge = _load_tuned_sequence_for_preparation(
            tuned_refs=tuned_ckpts,
            base_sd=base_sd,
            build_cfg=build_cfg,
            model=llm.model,
            strict_load=strict_load,
            use_low_memory_prepare=use_low_memory_all_methods,
        )

    split = str(cfg.get("split", "validation"))
    max_samples_per_task = cfg.get("max_samples_per_task", None)
    if max_samples_per_task is not None:
        max_samples_per_task = int(max_samples_per_task)
    max_prompt_tokens = cfg.get("max_prompt_tokens", None)
    if max_prompt_tokens is not None:
        max_prompt_tokens = int(max_prompt_tokens)
    print_every = cfg.get("print_every", None)
    if print_every is not None:
        print_every = int(print_every)

    task_data: list[NLITaskData] = []
    for t in tasks:
        td = build_nli_task_data(task=t, split=split, max_samples=max_samples_per_task)
        task_data.append(td)
        print(f"Loaded task {t}: {td.meta}")

    tokenized_task_data: list[NLITokenizedData] = []
    task_heads: dict[str, Any] | None = None
    task_mask_class = _resolve_task_mask_class(cfg.get("task_mask_class", None))
    if eval_mode == "head_logits":
        if task_heads_path is None:
            raise ValueError("head_logits mode requires --task-heads (or config['task_heads']).")
        task_heads = _load_task_heads(task_heads_path)
        batch_size = int(cfg.get("batch_size", 8))
        num_workers = int(cfg.get("num_workers", 0))
        max_length = int(cfg.get("max_length", 512))
        head_num_labels = int(getattr(llm.model.config, "num_labels", num_labels))
        for td in task_data:
            masked_class = task_mask_class.get(td.task, None)
            class_ids = _head_class_ids_for_task(
                task=td.task,
                task_num_labels=len(td.labels),
                head_num_labels=head_num_labels,
                masked_class=masked_class,
            )
            tk = build_nli_tokenized_loader(
                task_data=td,
                tokenizer=llm.tokenizer,
                batch_size=batch_size,
                num_workers=num_workers,
                max_length=max_length,
                head_class_ids=class_ids,
            )
            tokenized_task_data.append(tk)
            print(f"Tokenized task {td.task}: {tk.meta}")
    external_ref_acc = _resolve_fine_tuned_acc(cfg=cfg, tasks=tasks)
    if external_ref_acc is not None:
        print(f"External reference accs: {external_ref_acc}")

    merge_context = {
        "kind": "llm",
        "cfg": cfg,
        "model": llm.model,
        "llm": llm,
        "build_cfg": build_cfg,
        "base_sd": base_sd,
        "tuned_ckpts": tuned_ckpts,
        "task_data": task_data,
        "eval_mode": eval_mode,
        "task_heads": task_heads,
        "head_key_pattern": head_key_pattern,
        "task_mask_class": task_mask_class,
        "num_labels": num_labels,
        "strict_load": strict_load,
        "peft_subspace": peft_subspace,
        "subspace_prepared": subspace_prepared,
        "peft_state_by_task": peft_state_by_task,
        "suite_name": suite_name,
        "batch_size": int(cfg.get("batch_size", 8)),
        "num_workers": int(cfg.get("num_workers", 0)),
        "max_length": int(cfg.get("max_length", 512)),
        "load_aligned_tuned": _load_aligned_tuned_from_ref,
        "inject_task_head": _inject_task_head,
        "head_class_ids_for_task": _head_class_ids_for_task,
    }

    enable_global_prepare = (not search_planner.is_multi_param()) and (cfg.get("hyperparam_search") is None)
    if (not zero_shot_only) and prepared is None and isinstance(method, PreparedMergeMethod) and enable_global_prepare:
        print(f"\nPreparing merge directions with method: {method.name}")
        prepared = method.prepare(
            base=base_sd_for_merge,
            tuned=tuned_sds_list,
            weights=merge_weights,
            strict=strict_load,
            merge_context=merge_context,
            method_params=method_params,
        )
    prepared_base_direction = _prepared_base_direction(prepared)
    if prepared_base_direction is not None:
        direction = prepared_base_direction[1]
        l2, max_abs = _tensor_dict_stats(direction)
        print(f"Prepared merge direction stats: keys={len(direction)} l2={l2:.4e} max_abs={max_abs:.4e}")

    user_prompt_template = cfg.get("prompt_template", None)
    if user_prompt_template is not None and not isinstance(user_prompt_template, str):
        raise ValueError("prompt_template must be a string when provided.")

    if zero_shot_only:
        print("Single-accuracy baseline mode: zero-shot only (no tuned checkpoints)")
    else:
        print("Single-accuracy baseline mode: single-task tuned (used for normalization)")
    compute_zero_shot_acc = True if zero_shot_only else bool(cfg.get("single_acc_zero_shot", False))
    if compute_zero_shot_acc:
        print("Zero-shot base-model accuracies will also be computed (not used for normalization).")
    base_accs: list[float] = []
    base_acc_by_task: dict[str, float] = {}
    if compute_zero_shot_acc:
        for i, td in enumerate(task_data):
            if eval_mode == "head_logits":
                if task_heads is None:
                    raise RuntimeError("head_logits mode requires loaded task_heads.")
                tk = tokenized_task_data[i]
                _inject_task_head(
                    model=llm.model,
                    task=td.task,
                    task_heads=task_heads,
                    head_key_pattern=head_key_pattern,
                    head_class_ids=list(tk.meta.get("head_class_ids", [])),
                )
                acc = llm.sequence_classification_accuracy(
                    tk.loader,
                    device=build_cfg.device,
                    mask_class=tk.mask_class,
                    print_every=print_every,
                )
            else:
                tpl = user_prompt_template if isinstance(user_prompt_template, str) else _default_prompt_for_task(td)
                acc = llm.nli_accuracy(
                    examples=td.examples,
                    label_texts=td.label_texts,
                    prompt_template=tpl,
                    device=build_cfg.device,
                    max_prompt_tokens=max_prompt_tokens,
                    print_every=print_every,
                )
            base_accs.append(acc)
            base_acc_by_task[td.task] = float(acc)
            print(f"{td.task}: zero_shot_acc={acc:.6f}")
        print(f"Average zero-shot acc across {len(base_accs)} tasks: {sum(base_accs) / len(base_accs):.6f}")

    eval_single_task_tuned = False if zero_shot_only else bool(cfg.get("eval_single_task_tuned", True))
    single_task_ref_acc: dict[str, float] = {}
    if eval_single_task_tuned:
        print("\nEvaluating single-task tuned checkpoints (sanity check)")
        tuned_task_accs: list[float] = []
        for i, td in enumerate(task_data):
            # Ensure each single-task evaluation starts from the same base weights.
            load_into_model(llm.model, base_sd, strict=False)
            aligned = _load_aligned_tuned_from_ref(
                ckpt_ref=tuned_ckpts[i],
                base_sd=base_sd,
                build_cfg=build_cfg,
                model=llm.model,
                prefer_lora_view=(not strict_load),
            )
            miss, unexp = load_into_model(llm.model, _to_cpu_fp32(aligned), strict=False)
            del aligned

            if eval_mode == "head_logits":
                if task_heads is None:
                    raise RuntimeError("head_logits mode requires loaded task_heads.")
                tk = tokenized_task_data[i]
                _inject_task_head(
                    model=llm.model,
                    task=td.task,
                    task_heads=task_heads,
                    head_key_pattern=head_key_pattern,
                    head_class_ids=list(tk.meta.get("head_class_ids", [])),
                )
                acc = llm.sequence_classification_accuracy(
                    tk.loader,
                    device=build_cfg.device,
                    mask_class=tk.mask_class,
                    print_every=print_every,
                )
            else:
                tpl = user_prompt_template if isinstance(user_prompt_template, str) else _default_prompt_for_task(td)
                acc = llm.nli_accuracy(
                    examples=td.examples,
                    label_texts=td.label_texts,
                    prompt_template=tpl,
                    device=build_cfg.device,
                    max_prompt_tokens=max_prompt_tokens,
                    print_every=print_every,
                )
            tuned_task_accs.append(acc)
            single_task_ref_acc[td.task] = float(acc)

            print(f"{td.task}: single_tuned_acc={acc:.6f} (load missing={miss} unexpected={unexp})")

        print(
            f"Average single-task tuned accuracy across {len(tuned_task_accs)} tasks: "
            f"{sum(tuned_task_accs) / len(tuned_task_accs):.6f}"
        )
        # Restore base model state before merged alpha sweep.
        load_into_model(llm.model, base_sd, strict=False)

    norm_reference_acc = single_task_ref_acc if single_task_ref_acc else external_ref_acc
    if norm_reference_acc is not None and compute_zero_shot_acc:
        baseline_name = "single-task tuned" if single_task_ref_acc else "external reference"
        print(f"Normalized accuracy baseline: {baseline_name}")
        base_norm_accs: list[float] = []
        for td in task_data:
            if td.task in norm_reference_acc:
                n = _normalized_acc(base_acc_by_task[td.task], norm_reference_acc[td.task])
                base_norm_accs.append(n)
                print(f"{td.task}: zero_shot_norm_acc={n:.3f}")
        if base_norm_accs:
            print(f"Average zero-shot normalized accuracy: {sum(base_norm_accs) / len(base_norm_accs):.3f}")

    if zero_shot_only:
        if not base_accs:
            raise RuntimeError("Zero-shot-only mode produced no accuracy values.")
        print("\n=== Zero-shot summary ===")
        for i, td in enumerate(task_data):
            print(f"{td.task}: acc={base_accs[i]:.6f}")
        print(f"avg_acc={sum(base_accs) / len(base_accs):.6f}")
        if norm_reference_acc is not None:
            per_task_rows = [{"task": td.task} for td in task_data]
            single_accs = [
                _to_unit_acc(norm_reference_acc[td.task]) if td.task in norm_reference_acc else 0.0 for td in task_data
            ]
            norm_ratio = [(base_accs[i] / single_accs[i]) if single_accs[i] > 0 else 0.0 for i in range(len(base_accs))]
            pretty_print_task_accuracies(
                suite_name or "nli6",
                "zero_shot",
                peft_subspace,
                per_task_rows,
                base_accs,
                norm_ratio,
                single_accs=single_accs,
            )
        if run_logger is not None:
            run_logger.log_summary(
                {
                    "mode": "zero_shot_only",
                    "tasks": [td.task for td in task_data],
                    "per_task_acc": {td.task: float(base_accs[i]) for i, td in enumerate(task_data)},
                    "avg_acc": float(sum(base_accs) / len(base_accs)),
                }
            )
            run_logger.finish("success")
        return

    alpha_to_task_accs: dict[float, list[float]] = {}
    alpha_to_task_norm_accs: dict[float, list[float]] = {}
    search_results: list[SearchEvaluation] = []
    best_result: SearchEvaluation | None = None
    subspace_state_cache: dict[str, dict[str, Any]] = {}
    prepared_cache: dict[str, Any] = {}

    if method is None:
        raise RuntimeError("Internal error: merge method was not initialized.")
    merge_context = _AlphaMergeContext(
        method=method,
        prepared=prepared,
        base_sd_for_merge=base_sd_for_merge,
        tuned_sds_list=tuned_sds_list,
        weights=merge_weights,
        method_params=method_params,
        peft_subspace=peft_subspace,
        subspace=subspace,
        subspace_prepared=subspace_prepared,
        peft_cfg=peft_cfg,
        peft_state_by_task=peft_state_by_task,
        tasks=tasks,
        merge_base_sd=merge_base_sd,
    )

    def _subspace_state_for(candidate_method_params: dict[str, Any]) -> dict[str, Any]:
        if peft_subspace == "full" or subspace is None or peft_cfg is None:
            return {
                "subspace_prepared": subspace_prepared,
                "tuned_sds_list": tuned_sds_list,
                "base_sd_for_merge": base_sd_for_merge,
                "weights": merge_weights,
            }

        cache_key = stable_method_params_cache_key(candidate_method_params)
        if cache_key in subspace_state_cache:
            return subspace_state_cache[cache_key]

        print(f"\nPreparing PEFT subspace: {peft_subspace} ({candidate_method_params})")
        candidate_subspace_prepared = subspace.prepare(
            lora_by_task=peft_state_by_task,
            peft_cfg=peft_cfg,
            method_params=candidate_method_params,
            weights=resolve_merge_weights(len(tasks), merge_weights),
            artifact_dir=subspace_artifact_dir,
        )
        candidate_weights = (
            list(candidate_subspace_prepared.merge_weight_override)
            if getattr(candidate_subspace_prepared, "merge_weight_override", None) is not None
            else merge_weights_raw
        )
        projected_by_task = subspace.project(
            candidate_subspace_prepared,
            lora_by_task=peft_state_by_task,
            peft_cfg=peft_cfg,
        )
        missing_projected = [t for t in tasks if t not in projected_by_task]
        if missing_projected:
            raise ValueError(f"Subspace projection missing task outputs: {missing_projected}")
        candidate_tuned_sds_list = [projected_by_task[t] for t in tasks]
        if not candidate_tuned_sds_list or not candidate_tuned_sds_list[0]:
            raise ValueError("Subspace projection returned no mergeable tensors.")
        candidate_base_sd_for_merge = _to_cpu_fp32({k: torch.zeros_like(v) for k, v in candidate_tuned_sds_list[0].items()})
        state = {
            "subspace_prepared": candidate_subspace_prepared,
            "tuned_sds_list": candidate_tuned_sds_list,
            "base_sd_for_merge": candidate_base_sd_for_merge,
            "weights": candidate_weights,
        }
        subspace_state_cache[cache_key] = state
        return state

    def _prepared_for(candidate_method_params: dict[str, Any]) -> Any:
        if not isinstance(method, PreparedMergeMethod):
            return None
        candidate_subspace_state = _subspace_state_for(candidate_method_params)
        cache_key = stable_method_params_cache_key(candidate_method_params)
        if cache_key in prepared_cache:
            return prepared_cache[cache_key]
        print(f"\nPreparing merge directions with method: {method.name} ({candidate_method_params})")
        candidate_merge_context = _AlphaMergeContext(
            method=method,
            prepared=prepared,
            base_sd_for_merge=candidate_subspace_state["base_sd_for_merge"],
            tuned_sds_list=candidate_subspace_state["tuned_sds_list"],
            weights=candidate_subspace_state["weights"],
            method_params=candidate_method_params,
            peft_subspace=peft_subspace,
            subspace=subspace,
            subspace_prepared=candidate_subspace_state["subspace_prepared"],
            peft_cfg=peft_cfg,
            peft_state_by_task=peft_state_by_task,
            tasks=tasks,
            merge_base_sd=merge_base_sd,
        )
        prepared_value = method.prepare(
            base=candidate_subspace_state["base_sd_for_merge"],
            tuned=candidate_subspace_state["tuned_sds_list"],
            weights=candidate_subspace_state["weights"],
            strict=strict_load,
            merge_context=candidate_merge_context,
            method_params=candidate_method_params,
        )
        prepared_cache[cache_key] = prepared_value
        return prepared_value

    postmerge_cfg_raw = cfg.get("postmerge", None)
    if postmerge_cfg_raw is not None and not isinstance(postmerge_cfg_raw, dict):
        raise ValueError("config['postmerge'] must be a dict when provided.")
    postmerge_cfg = dict(postmerge_cfg_raw) if isinstance(postmerge_cfg_raw, dict) else {}
    postmerge_name = postmerge_cfg.get("method", None)
    if postmerge_name is not None:
        if eval_mode != "head_logits":
            raise ValueError("AdaMerging v1 for llm_merge supports eval_mode='head_logits' only.")
        if task_heads is None:
            raise RuntimeError("AdaMerging head_logits mode requires loaded task_heads.")
        if not tuned_sds_list:
            tuned_sds_list, base_sd_for_merge = _load_tuned_sequence_for_preparation(
                tuned_refs=tuned_ckpts,
                base_sd=base_sd,
                build_cfg=build_cfg,
                model=llm.model,
                strict_load=strict_load,
                use_low_memory_prepare=True,
            )
        postmerge_cfg.setdefault("device", build_cfg.device)
        postmerge_cfg.setdefault("init_alpha", float(cfg.get("alpha", 1.0)))
        postmerge_method = get_postmerge_method(str(postmerge_name))
        max_batches_per_task = postmerge_cfg.get("max_batches_per_task", None)
        max_batches_per_task = None if max_batches_per_task is None else int(max_batches_per_task)
        entropy_temperature = float(postmerge_cfg.get("entropy_temperature", 1.0))

        def _llm_entropy_loss(bank, alpha_values: torch.Tensor, alpha_mode: str) -> torch.Tensor:
            base_params = bank.merged_parameter_dict(
                llm.model,
                alpha_values,
                mode=alpha_mode,
                device=build_cfg.device,
            )
            losses: list[torch.Tensor] = []
            for i, td in enumerate(task_data):
                tk = tokenized_task_data[i]
                head_overrides = _task_head_param_overrides(
                    model=llm.model,
                    task=td.task,
                    task_heads=task_heads,
                    head_key_pattern=head_key_pattern,
                    head_class_ids=list(tk.meta.get("head_class_ids", [])),
                )
                task_params = dict(base_params)
                task_params.update(head_overrides)
                for batch_idx, batch in enumerate(tk.loader):
                    if max_batches_per_task is not None and batch_idx >= max_batches_per_task:
                        break
                    model_kwargs = {
                        k: v.to(build_cfg.device, non_blocking=True)
                        for k, v in batch.items()
                        if k != "labels" and torch.is_tensor(v)
                    }
                    out = functional_call(llm.model, task_params, (), kwargs=model_kwargs)
                    logits = out.logits
                    if logits.ndim != 2:
                        raise ValueError(
                            f"AdaMerging head_logits mode expects [B, C] logits, got shape {tuple(logits.shape)}."
                        )
                    if tk.mask_class is not None:
                        idx = torch.tensor(tk.mask_class, device=logits.device, dtype=torch.long)
                        logits = logits.index_select(dim=1, index=idx)
                    losses.append(prediction_entropy(logits, temperature=entropy_temperature))
            if not losses:
                raise RuntimeError("AdaMerging LLM loss did not receive any validation batches.")
            return torch.stack(losses).mean()

        print(f"\n=== Postmerge method = {postmerge_method.name} ===")
        postmerge_result = postmerge_method.run(
            PostMergeContext(
                kind="llm",
                model=llm.model,
                base=base_sd_for_merge,
                tuned=tuned_sds_list,
                tasks=tasks,
                weights=merge_weights,
                peft_subspace=peft_subspace,
                config=postmerge_cfg,
                entropy_loss_fn=_llm_entropy_loss,
            )
        )
        miss, unexp = load_into_model(llm.model, postmerge_result.merged_state, strict=strict_load)
        print(f"Loaded postmerged weights. missing={miss}, unexpected={unexp}")

        postmerge_accs: list[float] = []
        postmerge_norm_accs: list[float] = []
        for i, td in enumerate(task_data):
            tk = tokenized_task_data[i]
            _inject_task_head(
                model=llm.model,
                task=td.task,
                task_heads=task_heads,
                head_key_pattern=head_key_pattern,
                head_class_ids=list(tk.meta.get("head_class_ids", [])),
            )
            acc = llm.sequence_classification_accuracy(
                tk.loader,
                device=build_cfg.device,
                mask_class=tk.mask_class,
                print_every=print_every,
            )
            postmerge_accs.append(acc)
            if norm_reference_acc is not None and td.task in norm_reference_acc:
                norm = _normalized_acc(acc, norm_reference_acc[td.task])
                postmerge_norm_accs.append(norm)
                print(f"{td.task}: postmerge_acc={acc:.6f}  norm_acc={norm:.3f}")
            else:
                print(f"{td.task}: postmerge_acc={acc:.6f}")

        avg_acc = sum(postmerge_accs) / max(1, len(postmerge_accs))
        avg_norm_acc = (
            sum(postmerge_norm_accs) / max(1, len(postmerge_norm_accs)) if postmerge_norm_accs else 0.0
        )
        print(f"\nPostmerge avg_acc={avg_acc:.6f}")
        if postmerge_norm_accs:
            print(f"Postmerge avg_norm_acc={avg_norm_acc:.3f}")
        if norm_reference_acc is not None:
            per_task_rows = [{"task": td.task} for td in task_data]
            single_accs = [
                _to_unit_acc(norm_reference_acc[td.task]) if td.task in norm_reference_acc else 0.0
                for td in task_data
            ]
            norm_ratio = [
                (postmerge_accs[i] / single_accs[i]) if single_accs[i] > 0 else 0.0
                for i in range(len(postmerge_accs))
            ]
            pretty_print_task_accuracies(
                suite_name or "nli6",
                f"{method.name}+{postmerge_method.name}",
                peft_subspace,
                per_task_rows,
                postmerge_accs,
                norm_ratio,
                single_accs=single_accs,
            )
        if cfg.get("save_merged", None) is not None:
            outp = Path(str(cfg["save_merged"]))
            outp.parent.mkdir(parents=True, exist_ok=True)
            torch.save(postmerge_result.merged_state, str(outp))
            print(f"Saved postmerged state_dict to {outp}")
        if run_logger is not None:
            run_logger.log_summary(
                {
                    "tasks": [td.task for td in task_data],
                    "method": method.name,
                    "postmerge": postmerge_result.metadata,
                    "peft_subspace": peft_subspace,
                    "test_results": {
                        "per_task_acc": {td.task: float(postmerge_accs[i]) for i, td in enumerate(task_data)},
                        "per_task_norm_acc": {
                            td.task: float(postmerge_norm_accs[i])
                            for i, td in enumerate(task_data[: len(postmerge_norm_accs)])
                        },
                        "avg_acc": float(avg_acc),
                        "avg_norm_acc": float(avg_norm_acc),
                    },
                    "saved_merged_path": cfg.get("save_merged"),
                }
            )
            run_logger.finish("success")
        return

    while True:
        batch = search_planner.next_batch()
        if batch is None:
            break
        batch_results: list[SearchEvaluation] = []
        prev_avg_norm_acc: float | None = None
        for candidate in batch:
            t0 = time.time()
            candidate_subspace_state = _subspace_state_for(candidate.method_params)
            candidate_prepared = prepared if prepared is not None else _prepared_for(candidate.method_params)
            candidate_prepared_base_direction = _prepared_base_direction(candidate_prepared)
            candidate_context = _AlphaMergeContext(
                method=method,
                prepared=candidate_prepared,
                base_sd_for_merge=candidate_subspace_state["base_sd_for_merge"],
                tuned_sds_list=candidate_subspace_state["tuned_sds_list"],
                weights=candidate_subspace_state["weights"],
                method_params=candidate.method_params,
                peft_subspace=peft_subspace,
                subspace=subspace,
                subspace_prepared=candidate_subspace_state["subspace_prepared"],
                peft_cfg=peft_cfg,
                peft_state_by_task=peft_state_by_task,
                tasks=tasks,
                merge_base_sd=merge_base_sd,
            )
            merged_sd = None
            can_use_inplace = (
                peft_subspace == "full"
                and use_inplace_task_arithmetic
                and candidate_prepared_base_direction is not None
            )
            if can_use_inplace:
                miss, unexp = _load_prepared_direction_into_model(
                    model=llm.model,
                    base=candidate_prepared_base_direction[0],
                    direction=candidate_prepared_base_direction[1],
                    alpha=float(candidate.alpha),
                    strict=strict_load,
                )
            else:
                merged_sd = _build_merged_state_from_context(candidate_context, alpha=float(candidate.alpha))
                miss, unexp = load_into_model(llm.model, merged_sd, strict=strict_load)

            accs: list[float] = []
            norm_accs: list[float] = []
            print(f"\n{describe_candidate(candidate)}  missing={miss}  unexpected={unexp}")
            for i, td in enumerate(task_data):
                if eval_mode == "head_logits":
                    if task_heads is None:
                        raise RuntimeError("head_logits mode requires loaded task_heads.")
                    tk = tokenized_task_data[i]
                    _inject_task_head(
                        model=llm.model,
                        task=td.task,
                        task_heads=task_heads,
                        head_key_pattern=head_key_pattern,
                        head_class_ids=list(tk.meta.get("head_class_ids", [])),
                    )
                    acc = llm.sequence_classification_accuracy(
                        tk.loader,
                        device=build_cfg.device,
                        mask_class=tk.mask_class,
                        print_every=print_every,
                    )
                else:
                    tpl = user_prompt_template if isinstance(user_prompt_template, str) else _default_prompt_for_task(td)
                    acc = llm.nli_accuracy(
                        examples=td.examples,
                        label_texts=td.label_texts,
                        prompt_template=tpl,
                        device=build_cfg.device,
                        max_prompt_tokens=max_prompt_tokens,
                        print_every=print_every,
                    )
                accs.append(acc)
                if norm_reference_acc is not None and td.task in norm_reference_acc:
                    norm = _normalized_acc(acc, norm_reference_acc[td.task])
                    norm_accs.append(norm)
                    print(f"{td.task}: acc={acc:.6f}  norm_acc={norm:.3f}")
                else:
                    print(f"{td.task}: acc={acc:.6f}")

            avg_acc = sum(accs) / max(1, len(accs))
            avg_norm_acc = sum(norm_accs) / max(1, len(norm_accs)) if norm_accs else 0.0
            score = avg_norm_acc if norm_accs else avg_acc
            result = SearchEvaluation(
                candidate=candidate,
                score=float(score),
                avg_acc=float(avg_acc),
                avg_norm_acc=float(avg_norm_acc),
                per_task_acc=[float(v) for v in accs],
                per_task_norm_acc=[float(v) for v in norm_accs],
            )
            batch_results.append(result)
            search_results.append(result)
            if not search_planner.is_multi_param():
                alpha_to_task_accs[float(candidate.alpha)] = [float(v) for v in accs]
                alpha_to_task_norm_accs[float(candidate.alpha)] = [float(v) for v in norm_accs]
            if run_logger is not None:
                run_logger.log_event(
                    "alpha_eval_end",
                    metrics={
                        "alpha/value": float(candidate.alpha),
                        "alpha/avg_acc": float(avg_acc),
                        "alpha/avg_norm_acc": float(avg_norm_acc),
                    },
                    context={
                        "search_stage": int(candidate.stage),
                        "method_params": candidate.method_params,
                        "search_values": candidate.values,
                        "per_task_acc": {td.task: float(accs[i]) for i, td in enumerate(task_data)},
                        "per_task_norm_acc": {
                            td.task: float(norm_accs[i]) for i, td in enumerate(task_data[: len(norm_accs)])
                        },
                    },
                )
            if norm_accs:
                print(
                    f"{describe_candidate(candidate)}  avg_acc={avg_acc:.6f}  avg_norm_acc={avg_norm_acc:.3f}  "
                    f"seconds={time.time() - t0:.2f}"
                )
            else:
                print(f"{describe_candidate(candidate)}  avg_acc={avg_acc:.6f}  seconds={time.time() - t0:.2f}")

            if best_result is None or result.score > best_result.score:
                best_result = result

            if norm_accs:
                if prev_avg_norm_acc is not None and avg_norm_acc < prev_avg_norm_acc:
                    print(
                        f"avg_norm_acc dropped ({avg_norm_acc:.3f} < {prev_avg_norm_acc:.3f}); stopping this alpha sweep early."
                    )
                    if merged_sd is not None:
                        del merged_sd
                    if torch.cuda.is_available() and build_cfg.device != "cpu":
                        torch.cuda.empty_cache()
                    break
                prev_avg_norm_acc = float(avg_norm_acc)

            if merged_sd is not None:
                del merged_sd
            if torch.cuda.is_available() and build_cfg.device != "cpu":
                torch.cuda.empty_cache()

        search_planner.observe(batch_results)

    if best_result is None:
        raise RuntimeError("Alpha sweep produced no results.")

    print("\n=== Alpha search summary (higher is better) ===")
    for result in search_results:
        if result.per_task_norm_acc:
            print(
                f"{describe_candidate(result.candidate)}  avg_acc={result.avg_acc:.6f}  "
                f"avg_norm_acc={result.avg_norm_acc:.3f}"
            )
        else:
            print(f"{describe_candidate(result.candidate)}  avg_acc={result.avg_acc:.6f}")
    best_alpha = float(best_result.candidate.alpha)
    best_method_params = dict(best_result.candidate.method_params)
    best_acc = float(best_result.avg_acc)
    best_norm_vals = list(best_result.per_task_norm_acc)
    if best_norm_vals:
        best_norm = float(best_result.avg_norm_acc)
        print(
            f"\nBest setting: {describe_candidate(best_result.candidate)} -> avg_acc={best_acc:.6f}  "
            f"avg_norm_acc={best_norm:.3f}"
        )
    else:
        print(f"\nBest setting: {describe_candidate(best_result.candidate)} -> avg_acc={best_acc:.6f}")

    print("\nPer-task accuracy at best alpha:")
    best_vals = list(best_result.per_task_acc)
    best_norm_vals = list(best_result.per_task_norm_acc)
    for i, td in enumerate(task_data):
        if i < len(best_norm_vals):
            print(f"{td.task}: acc={best_vals[i]:.6f}  norm_acc={best_norm_vals[i]:.3f}")
        else:
            print(f"{td.task}: acc={best_vals[i]:.6f}")
    if best_norm_vals:
        normalized_results = (
            " & ".join([f"{v:.2f}" for v in best_norm_vals])
            + f" & {sum(best_norm_vals) / len(best_norm_vals):.2f} \\\\"
        )
        print(f"Normalized Test results: {normalized_results}")
    if norm_reference_acc is not None:
        per_task_rows = [{"task": td.task} for td in task_data]
        single_accs = [
            _to_unit_acc(norm_reference_acc[td.task]) if td.task in norm_reference_acc else 0.0 for td in task_data
        ]
        norm_ratio = [(best_vals[i] / single_accs[i]) if single_accs[i] > 0 else 0.0 for i in range(len(best_vals))]
        pretty_print_task_accuracies(
            suite_name or "nli6",
            method.name,
            peft_subspace,
            per_task_rows,
            best_vals,
            norm_ratio,
            single_accs=single_accs,
        )

    if peft_subspace != "full":
        subspace_prepared = _subspace_state_for(best_method_params)["subspace_prepared"]

    if cfg.get("save_merged", None) is not None:
        best_subspace_state = _subspace_state_for(best_method_params)
        best_context = _AlphaMergeContext(
            method=method,
            prepared=(prepared if prepared is not None else _prepared_for(best_method_params)),
            base_sd_for_merge=best_subspace_state["base_sd_for_merge"],
            tuned_sds_list=best_subspace_state["tuned_sds_list"],
            weights=best_subspace_state["weights"],
            method_params=best_method_params,
            peft_subspace=peft_subspace,
            subspace=subspace,
            subspace_prepared=best_subspace_state["subspace_prepared"],
            peft_cfg=peft_cfg,
            peft_state_by_task=peft_state_by_task,
            tasks=tasks,
            merge_base_sd=merge_base_sd,
        )
        merged_best_sd = _build_merged_state_from_context(best_context, alpha=float(best_alpha))
        outp = Path(str(cfg["save_merged"]))
        outp.parent.mkdir(parents=True, exist_ok=True)
        torch.save(merged_best_sd, str(outp))
        print(f"Saved best-alpha merged state_dict to {outp}")
    if run_logger is not None:
        run_logger.log_summary(
            {
                "tasks": [td.task for td in task_data],
                "method": method.name,
                "peft_subspace": peft_subspace,
                "best_alpha": float(best_alpha),
                "best_method_params": best_method_params,
                "search_strategy": search_planner.search_summary(),
                "search_results": summarize_search_results(search_results),
                "alpha_to_task_accs": {str(k): [float(v) for v in vals] for k, vals in alpha_to_task_accs.items()},
                "alpha_to_task_norm_accs": {
                    str(k): [float(v) for v in vals] for k, vals in alpha_to_task_norm_accs.items()
                },
                "best_per_task_acc": {td.task: float(best_vals[i]) for i, td in enumerate(task_data)},
                "best_per_task_norm_acc": {
                    td.task: float(best_norm_vals[i]) for i, td in enumerate(task_data[: len(best_norm_vals)])
                },
                "saved_merged_path": cfg.get("save_merged"),
                "subspace_artifacts": (
                    {"similarity_artifact_path": getattr(subspace_prepared, "similarity_artifact_path", None)}
                    if subspace_prepared is not None
                    else {}
                ),
            }
        )
        run_logger.finish("success")


if __name__ == "__main__":
    main()
