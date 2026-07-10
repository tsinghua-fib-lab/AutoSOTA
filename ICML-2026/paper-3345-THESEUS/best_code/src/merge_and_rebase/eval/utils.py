import inspect
import json
import math
import random
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency fallback
    tqdm = None

try:
    from peft import LoraConfig, TaskType, get_peft_model, set_peft_model_state_dict
except ImportError:
    LoraConfig = None
    TaskType = None
    get_peft_model = None
    set_peft_model_state_dict = None

from merge_and_rebase.io.ckpt import load_ckpt, load_into_model, resolve_ckpt_path
from merge_and_rebase.io.peft_helpers import (
    get_attn_patch_cfg,
    get_patched_attn_flag,
    is_peft_adapter_dir_ckpt,
    is_peft_adapter_reference,
    normalize_attn_patch_cfg,
    normalize_peft_visual_state_dict_keys,
    resolve_peft_adapter_dir,
    state_dict_looks_patched_attn,
)
from merge_and_rebase.models.openclip_classifier import OpenClipBuildConfig, OpenClipClassifier
from merge_and_rebase.models.patch_openclip_projection import patch_openclip_visual_proj, restore_openclip_proj_keyspace

from ..merge import runtime as _merge_utils

is_peft_checkpoint = _merge_utils.is_peft_checkpoint
extract_peft_components = _merge_utils.extract_peft_components
get_peft_cfg = _merge_utils.get_peft_cfg
apply_delta = _merge_utils.apply_delta
to_cpu_fp32 = _merge_utils.to_cpu_fp32
ensure_peft_cfg_map = _merge_utils.ensure_peft_cfg_map
build_merged_state_for_alpha = _merge_utils.build_merged_state_for_alpha
build_dense_delta_branch = _merge_utils.build_dense_delta_branch
compose_weighted_deltas = _merge_utils.compose_weighted_deltas


def _require_peft() -> None:
    if LoraConfig is None or TaskType is None or get_peft_model is None or set_peft_model_state_dict is None:
        raise ImportError("PEFT-dependent evaluation paths require `peft` with its runtime dependencies installed.")


def stable_method_params_cache_key(value: Any) -> str:
    def _json_safe(x: Any) -> Any:
        if x is None or isinstance(x, (bool, int, float, str)):
            return x
        if isinstance(x, Mapping):
            return {str(k): _json_safe(v) for k, v in x.items()}
        if isinstance(x, (list, tuple, set)):
            return [_json_safe(v) for v in x]
        return repr(x)

    return json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"))


def build_cpu_cfg(cfg: OpenClipBuildConfig) -> OpenClipBuildConfig:
    return OpenClipBuildConfig(
        loader=cfg.loader,
        model_name=cfg.model_name,
        pretrained=cfg.pretrained,
        device="cpu",
        dtype="fp32",
        normalize=cfg.normalize,
        logit_scale=cfg.logit_scale,
        prompt_template=cfg.prompt_template,
        prompt_templates=cfg.prompt_templates,
    )


def build_lora_config(cfg_dict: dict[str, Any]):
    _require_peft()

    cfg = dict(cfg_dict)
    task_type = cfg.get("task_type", None)
    if isinstance(task_type, str):
        try:
            cfg["task_type"] = TaskType[task_type]
        except KeyError as exc:
            raise ValueError(f"Unknown peft TaskType '{task_type}'.") from exc

    # Filter to LoraConfig signature to avoid unexpected fields.
    sig = inspect.signature(LoraConfig.__init__)
    allowed = set(sig.parameters)
    allowed.discard("self")
    cfg = {k: v for k, v in cfg.items() if k in allowed}
    return LoraConfig(**cfg)


@dataclass(frozen=True)
class TaskAttentionMeta:
    patched_attn: bool = False
    attn_patch_cfg: dict[str, Any] | None = None

    @property
    def linearized_attn(self) -> bool:
        if not self.patched_attn or self.attn_patch_cfg is None:
            return False
        return str(self.attn_patch_cfg.get("attn_impl", "softmax")) == "linear"


def humanize(s: str) -> str:
    s = s.replace("_", " ").replace("-", " ")
    return " ".join(s.split())


class FirstNBatches:
    """Wrapper that yields at most *n* batches from a dataloader."""

    def __init__(self, loader: DataLoader, n: int):
        self.loader = loader
        self.n = int(n)

    def __iter__(self):
        import itertools

        return itertools.islice(iter(self.loader), self.n)

    def __len__(self):
        try:
            return min(self.n, len(self.loader))
        except TypeError:
            return self.n


def balanced_sample_indices(
    dataset: Any,
    imgs_per_class: int,
    seed: int = 42,
) -> list[int]:
    """Sample *imgs_per_class* indices per class (random, balanced)."""
    class_indices: dict[int, list[int]] = defaultdict(list)
    if hasattr(dataset, "split") and hasattr(dataset.split, "__getitem__"):
        labels = dataset.split[dataset.label_key]
        for idx, label in enumerate(labels):
            class_indices[int(label)].append(idx)
    else:
        for idx in range(len(dataset)):
            _, label = dataset[idx]
            class_indices[int(label)].append(idx)

    rng = random.Random(seed)
    flat: list[int] = []
    for cls in sorted(class_indices):
        idxs = class_indices[cls]
        if len(idxs) <= imgs_per_class:
            flat.extend(idxs)
        else:
            flat.extend(rng.sample(idxs, imgs_per_class))
    return flat


def extract_dataset_labels(dataset: Any) -> list[int]:
    if isinstance(dataset, Subset):
        parent_labels = extract_dataset_labels(dataset.dataset)
        return [int(parent_labels[i]) for i in dataset.indices]

    if hasattr(dataset, "labels"):
        labels = dataset.labels
        return [int(y) for y in labels]

    if hasattr(dataset, "targets"):
        labels = dataset.targets
        return [int(y) for y in labels]

    if hasattr(dataset, "split") and hasattr(dataset.split, "__getitem__") and hasattr(dataset, "label_key"):
        labels = dataset.split[dataset.label_key]
        return [int(y) for y in labels]

    out: list[int] = []
    for idx in range(len(dataset)):
        item = dataset[idx]
        if isinstance(item, dict) and "labels" in item:
            out.append(int(item["labels"]))
            continue
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            out.append(int(item[1]))
            continue
        raise ValueError(
            "Unable to infer dataset labels for Fisher sampling. "
            "Expected .labels/.targets, split[label_key], dict['labels'], or tuple(item, label)."
        )
    return out


def load_vision_checkpoint_reference(
    *,
    ckpt_ref: str,
) -> tuple[str, Any]:
    resolved_ref = resolve_ckpt_path(str(ckpt_ref))
    if is_peft_adapter_reference(resolved_ref):
        adapter_dir = resolve_peft_adapter_dir(resolved_ref)
        return str(ckpt_ref), {"format": "peft", "peft_adapter_dir": str(Path(adapter_dir))}
    return resolved_ref, torch.load(resolved_ref, map_location="cpu", weights_only=False)


def proportional_sample_indices(
    dataset: Any,
    sample_size: int,
    *,
    seed: int = 42,
) -> list[int]:
    labels = extract_dataset_labels(dataset)
    total = len(labels)
    if sample_size <= 0:
        return []
    if sample_size >= total:
        return list(range(total))

    class_indices: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        class_indices[int(label)].append(idx)

    counts: dict[int, int] = {}
    remainders: list[tuple[float, int]] = []
    assigned = 0
    for cls, idxs in class_indices.items():
        raw = (float(len(idxs)) * float(sample_size)) / float(total)
        take = min(len(idxs), int(math.floor(raw)))
        counts[cls] = take
        assigned += take
        remainders.append((raw - float(take), cls))

    remaining = int(sample_size) - int(assigned)
    remainders.sort(key=lambda item: (-item[0], item[1]))
    while remaining > 0:
        progressed = False
        for _remainder, cls in remainders:
            capacity = len(class_indices[cls]) - counts[cls]
            if capacity <= 0:
                continue
            counts[cls] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            break

    rng = random.Random(seed)
    selected: list[int] = []
    for cls in sorted(class_indices):
        idxs = class_indices[cls]
        take = counts.get(cls, 0)
        if take <= 0:
            continue
        if take >= len(idxs):
            selected.extend(idxs)
        else:
            selected.extend(rng.sample(idxs, take))
    return selected


def _build_loader_like(
    loader: DataLoader,
    dataset: Any,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=bool(getattr(loader, "pin_memory", True)),
        drop_last=bool(getattr(loader, "drop_last", False)),
        collate_fn=getattr(loader, "collate_fn", None),
        persistent_workers=(bool(getattr(loader, "persistent_workers", False)) and int(num_workers) > 0),
    )


def build_grad_dataloader(
    train_loader: DataLoader,
    train_dataset: Any,
    *,
    grad_batch_size: int | None = None,
    grad_imgs_per_class: int | None = None,
    grad_data_percentage: float | None = None,
    grad_sampling_strategy: str = "random",
    grad_num_batches: int | None = None,
    num_workers: int = 6,
    seed: int = 42,
) -> DataLoader | FirstNBatches:
    """Build a smaller dataloader for gradient-sign computation when requested."""
    if grad_imgs_per_class is not None and grad_data_percentage is not None:
        raise ValueError("grad_imgs_per_class and grad_data_percentage are mutually exclusive; set only one of them.")
    if grad_data_percentage is not None:
        grad_data_percentage = float(grad_data_percentage)
        if grad_data_percentage <= 0.0 or grad_data_percentage > 100.0:
            raise ValueError("grad_data_percentage must be in the range (0, 100].")
    sampling_strategy = str(grad_sampling_strategy).strip().lower()
    if sampling_strategy not in {"random", "stratified"}:
        raise ValueError("grad_sampling_strategy must be one of: random, stratified")

    batch_size = grad_batch_size or train_loader.batch_size
    dataset_for_loader: Any = train_loader.dataset
    use_shuffle = True

    if grad_imgs_per_class is not None:
        indices = balanced_sample_indices(train_dataset, grad_imgs_per_class, seed=seed)
        dataset_for_loader = Subset(train_dataset, indices)
    elif grad_data_percentage is not None and grad_data_percentage < 100.0:
        total = len(train_dataset)
        sample_size = max(1, int(math.ceil((grad_data_percentage / 100.0) * float(total))))
        if sampling_strategy == "stratified":
            indices = proportional_sample_indices(train_dataset, sample_size, seed=seed)
        else:
            rng = random.Random(seed)
            indices = rng.sample(range(total), sample_size)
        dataset_for_loader = Subset(train_dataset, indices)

    if dataset_for_loader is train_loader.dataset and (
        grad_batch_size is None or grad_batch_size == train_loader.batch_size
    ):
        loader = train_loader
    else:
        loader = _build_loader_like(
            train_loader,
            dataset_for_loader,
            batch_size=int(batch_size),
            shuffle=use_shuffle,
            num_workers=num_workers,
        )

    if grad_num_batches is not None:
        return FirstNBatches(loader, grad_num_batches)

    return loader


def acc_cache_key(
    clip_model: str,
    clip_pretrained: str,
    dataset: str,
    chk_path: str,
    baseline_mode: str,
    forward_mode: str,
    forward_mode_params: Mapping[str, Any] | None,
    classnames_mode: str,
    text_features_mode: str = "zero_shot",
) -> str:
    # Include baseline mode/checkpoint/forward/classname mode to avoid stale cache collisions.
    forward_mode_params_key = json.dumps(dict(forward_mode_params or {}), sort_keys=True, separators=(",", ":"))
    return (
        f"{clip_model}::{clip_pretrained}::{dataset}::{baseline_mode}::"
        f"{chk_path}::{forward_mode}::{forward_mode_params_key}::{classnames_mode}::{text_features_mode}"
    )


def patch_base_for_attn(
    *,
    clf: OpenClipClassifier,
    base_ckpt: str | None,
    strict_load: bool,
    attn_patch_cfg: dict[str, Any] | None = None,
) -> dict[str, torch.Tensor]:
    from merge_and_rebase.models.patch_openclip_attention import split_openclip_vit_attn

    patch_cfg = dict(attn_patch_cfg or {})
    n = split_openclip_vit_attn(
        clf.model.visual,
        proj_dropout=0.0,
        attn_impl=str(patch_cfg.get("attn_impl", "softmax")),
        kernel=str(patch_cfg.get("kernel", "elu_plus_one")),
        eps=float(patch_cfg.get("eps", 1e-6)),
        linear_rule=str(patch_cfg.get("linear_rule", "kernel")),
        delta_eta=float(patch_cfg.get("delta_eta", 1.0)),
        delta_exclude_cls_from_store=bool(patch_cfg.get("delta_exclude_cls_from_store", True)),
        delta_cls_only_readout=bool(patch_cfg.get("delta_cls_only_readout", False)),
        delta_learn_w0=bool(patch_cfg.get("delta_learn_w0", False)),
        delta_w0_rank=int(patch_cfg.get("delta_w0_rank", 0)),
    )
    if n == 0:
        raise RuntimeError("patched_attn=True but patch_openclip_vit_attn patched 0 blocks.")
    print(f"Patched {n} attention blocks in base model.")

    if base_ckpt is None:
        return {k: v.detach().cpu() for k, v in clf.model.state_dict().items()}

    sd0 = load_ckpt(str(base_ckpt))
    load_into_model(clf.model, sd0, strict=strict_load)
    return {k: v.detach().cpu() for k, v in clf.model.state_dict().items()}


def extract_checkpoint_attn_patch_info(
    *,
    obj: Any,
    ckpt_path: str,
) -> TaskAttentionMeta:
    if not isinstance(obj, dict):
        return TaskAttentionMeta()

    sd_obj = obj.get("state_dict", None)
    sd_looks_patched = isinstance(sd_obj, dict) and state_dict_looks_patched_attn(sd_obj)
    cfg_obj = obj.get("attn_patch_cfg", None)
    cfg_norm = normalize_attn_patch_cfg(cfg_obj) if isinstance(cfg_obj, dict) else None

    if is_peft_adapter_dir_ckpt(obj) or is_peft_checkpoint(obj):
        patched_attn = get_patched_attn_flag(obj)
        if not patched_attn:
            return TaskAttentionMeta()
        return TaskAttentionMeta(patched_attn=True, attn_patch_cfg=normalize_attn_patch_cfg(get_attn_patch_cfg(obj)))

    if "patched_attn" in obj:
        patched_attn = bool(obj["patched_attn"])
        if not patched_attn:
            if sd_looks_patched:
                if cfg_norm is not None:
                    return TaskAttentionMeta(patched_attn=True, attn_patch_cfg=cfg_norm)
                raise ValueError(
                    f"Checkpoint '{ckpt_path}' has patched_attn=False but state_dict uses q_proj/k_proj/v_proj keys. "
                    "Add attn_patch_cfg metadata so vision merge can patch qkv before linearization."
                )
            return TaskAttentionMeta()
        if cfg_norm is None:
            raise ValueError(
                f"Checkpoint '{ckpt_path}' has patched_attn=True but no attn_patch_cfg. "
                "Cannot verify whether attention was linearized ('attn_impl=linear') for a reproducible vision merge."
            )
        return TaskAttentionMeta(patched_attn=True, attn_patch_cfg=cfg_norm)

    if sd_looks_patched:
        if cfg_norm is not None:
            return TaskAttentionMeta(patched_attn=True, attn_patch_cfg=cfg_norm)
        raise ValueError(
            f"Checkpoint '{ckpt_path}' appears to use patched attention (q_proj/k_proj/v_proj) but lacks "
            "patched_attn/attn_patch_cfg metadata. Cannot verify linearized-attention consistency in vision merge."
        )
    return TaskAttentionMeta()


def assert_qkv_patched_before_linearizing(
    *,
    needs_linear_attention: bool,
    base_patched_for_attn: bool,
    model_state_dict: Mapping[str, torch.Tensor],
) -> None:
    if not needs_linear_attention:
        return
    if not base_patched_for_attn:
        raise RuntimeError(
            "Linear attention requested by checkpoints, but base attention was not patched first. "
            "Ensure q/k/v patching runs before linearized attention evaluation."
        )

    keys = tuple(str(k) for k in model_state_dict.keys())
    has_qkv = any(".attn.q_proj." in k or ".attn.k_proj." in k or ".attn.v_proj." in k for k in keys)
    if not has_qkv:
        raise RuntimeError(
            "Linear attention requested, but model keyspace is not fully q/k/v patched "
            "(expected q_proj/k_proj/v_proj and no in_proj_* keys)."
        )


def maybe_patch_base_for_task_attn(
    *,
    task_meta: TaskAttentionMeta,
    base_patched_for_attn: bool,
    clf: OpenClipClassifier,
    base_ckpt: str | None,
    strict_load: bool,
    base_sd: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], bool]:
    if base_patched_for_attn or not task_meta.patched_attn:
        return base_sd, base_patched_for_attn
    patched_sd = patch_base_for_attn(
        clf=clf,
        base_ckpt=base_ckpt,
        strict_load=strict_load,
        attn_patch_cfg=task_meta.attn_patch_cfg,
    )
    return patched_sd, True


def eval_task_top1(
    *,
    clf: OpenClipClassifier,
    loaders: Any,
    classnames: list[str],
    build_cfg_task: OpenClipBuildConfig,
    device: str,
    split: str,
    text_features: torch.Tensor | None = None,
) -> float:
    if split == "val":
        eval_loader = loaders.val
    elif split == "test":
        eval_loader = loaders.test
    else:
        raise ValueError(f"Unknown split '{split}'. Expected one of: val, test.")
    if text_features is not None:
        return float(
            clf.top1_with_text_features(
                eval_loader,
                device=device,
                text_features=text_features,
                expected_num_classes=len(classnames),
            )
        )
    clf.build_zeroshot_text_features(classnames, build_cfg_task, cache_dir="src/.cache/zs_cache", force_rebuild=False)
    return float(clf.top1(eval_loader, device=device))


def eval_norm_accs_for_split(
    *,
    clf: OpenClipClassifier,
    per_task: list[dict[str, Any]],
    device: str,
    split: str,
    print_per_task: bool,
    result_label: str = "merged",
    baseline_label: str = "single",
) -> tuple[list[float], list[float]]:
    merged_accs: list[float] = []
    norm_accs: list[float] = []
    task_width = max((len(str(item["task"])) for item in per_task), default=0)
    pbar = tqdm(per_task, desc=f"Evaluating {split} accuracies", unit="task", disable=not split == "test")

    for item in pbar:
        task = str(item["task"])
        acc = eval_task_top1(
            device=device,
            clf=clf,
            loaders=item["loaders"],
            classnames=list(item["classnames"]),
            build_cfg_task=item["build_cfg_task"],
            split=split,
            text_features=item.get("text_features", None),
        )
        single_acc = float(item["single_acc"])
        norm = (acc / single_acc) if single_acc > 0 else 0.0
        merged_accs.append(acc)
        norm_accs.append(norm)
        if print_per_task:
            print(f"{task:<{task_width}}  {baseline_label}={single_acc:.6f}  {result_label}={acc:.6f}  norm={norm:.6f}")
    return merged_accs, norm_accs


def materialize_peft_sd_from_adapter(
    *,
    peft_state: dict[str, torch.Tensor],
    base_sd: dict[str, torch.Tensor],
    build_cfg: OpenClipBuildConfig,
    peft_cfg: dict[str, Any],
    peft_dense_state: dict[str, torch.Tensor] | None = None,
    strict_load: bool,
    patched_attn: bool,
    attn_patch_cfg: dict[str, Any] | None = None,
) -> dict[str, torch.Tensor]:
    _require_peft()

    """
    Rebuild a full OpenCLIP state_dict from:
      - base_sd: base (pretrained or base checkpoint) state dict in the *correct keyspace*
      - peft_state: adapter-only state dict (LoRA weights) as saved by PEFT
      - peft_cfg_map: adapter config map (like {"default": {...}})

    Returns:
      full_sd: dict[str, Tensor] suitable for load_into_model(clf.model, full_sd)
    """
    # 1) Build base model
    # Use a deterministic fp32 CPU build for adapter materialization so
    # full-space and core-space paths operate in the same numeric regime.
    clf = OpenClipClassifier.build(build_cpu_cfg(build_cfg))
    model = clf.model

    # 2) Patch attention if needed (must happen before loading base_sd)
    if patched_attn:
        from merge_and_rebase.models.patch_openclip_attention import split_openclip_vit_attn

        patch_cfg = dict(attn_patch_cfg or {})
        n = split_openclip_vit_attn(
            model.visual,
            proj_dropout=0.0,
            attn_impl=str(patch_cfg.get("attn_impl", "softmax")),
            kernel=str(patch_cfg.get("kernel", "elu_plus_one")),
            eps=float(patch_cfg.get("eps", 1e-6)),
            linear_rule=str(patch_cfg.get("linear_rule", "kernel")),
            delta_eta=float(patch_cfg.get("delta_eta", 1.0)),
            delta_exclude_cls_from_store=bool(patch_cfg.get("delta_exclude_cls_from_store", True)),
            delta_cls_only_readout=bool(patch_cfg.get("delta_cls_only_readout", False)),
            delta_learn_w0=bool(patch_cfg.get("delta_learn_w0", False)),
            delta_w0_rank=int(patch_cfg.get("delta_w0_rank", 0)),
        )
        if n == 0:
            raise RuntimeError("patched_attn=True but patch_openclip_vit_attn patched 0 blocks.")

    # 3) Load base weights into the (possibly patched) base model
    # base_sd must match this model's keys (patched keyspace if patched_attn=True)
    load_into_model(model, base_sd, strict=strict_load)

    target_modules = peft_cfg.get("target_modules", None)
    if target_modules is None:
        raise ValueError("peft_cfg_map must include target_modules to reconstruct the adapter.")
    if not isinstance(target_modules, list):
        raise ValueError("peft_cfg.target_modules must be a list when reconstructing PEFT adapters.")
    if "lin_proj" in target_modules:
        patched_proj = patch_openclip_visual_proj(model.visual)
        if patched_proj == 0 and not hasattr(model.visual, "lin_proj"):
            raise RuntimeError("target_modules requested 'lin_proj' but the visual projection surface was not patched.")

    # 5) Wrap ONLY visual with PEFT
    peft_visual = get_peft_model(model.visual, build_lora_config(peft_cfg))
    model.visual = peft_visual

    # 6) Load adapter weights
    # Ensure tensors are on the same device/dtype as the PEFT module expects
    dev = next(model.parameters()).device
    peft_state = normalize_peft_visual_state_dict_keys(peft_state)
    peft_state = {k: v.to(device=dev) for k, v in peft_state.items()}

    # Adapter-only state dict load. Missing base-model keys are expected here.
    res = set_peft_model_state_dict(model.visual, peft_state, adapter_name="default")

    # Some PEFT versions return None; others return _IncompatibleKeys / tuple.
    if res is not None:
        missing = getattr(res, "missing_keys", None)
        unexpected = getattr(res, "unexpected_keys", None)
        if missing is None or unexpected is None:
            missing, unexpected = res
        # Missing keys are expected when loading adapter-only weights into the wrapped module.
        if unexpected:
            msg = f"PEFT load: unexpected={len(unexpected)}, loaded {len(peft_state)} adapter params."
            if strict_load:
                raise RuntimeError(msg + f"\nunexpected[:20]={unexpected[:20]}")
            print("[warn]", msg)

    if peft_dense_state:
        dense_state = {k: v.to(device=dev) for k, v in peft_dense_state.items()}
        model.visual.load_state_dict(dense_state, strict=False)

    # Important: unwrap PEFT visual so state_dict keys match OpenCLIP base keyspace.
    if hasattr(model.visual, "merge_and_unload"):
        model.visual = model.visual.merge_and_unload()

    # 7) Export full weights on CPU
    full_sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    full_sd = restore_openclip_proj_keyspace(full_sd)
    return full_sd
