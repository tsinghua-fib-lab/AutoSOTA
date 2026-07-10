from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from merge_and_rebase.io.peft_helpers import normalize_peft_visual_state_dict_keys


def to_cpu_fp32(sd: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().to(dtype=torch.float32) for k, v in sd.items()}


def is_peft_checkpoint(obj: Any) -> bool:
    return isinstance(obj, dict) and (
        "peft_state_dict" in obj or obj.get("format") == "peft" or isinstance(obj.get("peft_adapter_dir"), str)
    )


def extract_peft_components(ckpt_obj: dict[str, Any]) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    peft_state = ckpt_obj.get("peft_state_dict", None)
    peft_cfg_map = ckpt_obj.get("peft_config", None)
    if not isinstance(peft_state, dict) or not peft_state:
        raise ValueError("Invalid PEFT checkpoint: missing 'peft_state_dict'.")
    if not isinstance(peft_cfg_map, dict) or not peft_cfg_map:
        raise ValueError("Invalid PEFT checkpoint: missing 'peft_config'.")
    state = {str(k): v.detach().cpu() for k, v in peft_state.items() if torch.is_tensor(v)}
    state = normalize_peft_visual_state_dict_keys(state)
    if not state:
        raise ValueError("Invalid PEFT checkpoint: 'peft_state_dict' contains no tensors.")
    return state, peft_cfg_map


def ensure_peft_cfg_map(
    peft_cfg_map: dict[str, Any] | None,
    cfg_map: dict[str, Any],
) -> dict[str, Any]:
    if peft_cfg_map is None:
        if "target_modules" in cfg_map.get("default", {}):
            cfg_map["default"]["target_modules"] = sorted(cfg_map["default"]["target_modules"])
        return cfg_map

    if "target_modules" in peft_cfg_map.get("default", {}):
        peft_cfg_map["default"]["target_modules"] = sorted(peft_cfg_map["default"]["target_modules"])
    if "target_modules" in cfg_map.get("default", {}):
        cfg_map["default"]["target_modules"] = sorted(cfg_map["default"]["target_modules"])
    if cfg_map != peft_cfg_map:
        raise ValueError("PEFT config mismatch across tuned checkpoints.")
    return peft_cfg_map


def get_peft_cfg(peft_cfg_map: dict[str, Any]) -> dict[str, Any]:
    cfg_name, cfg_dict = next(iter(peft_cfg_map.items()))
    if not isinstance(cfg_dict, dict):
        raise ValueError(f"Invalid PEFT config for adapter '{cfg_name}'.")
    return cfg_dict


def apply_delta(
    base: dict[str, torch.Tensor],
    delta: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    out = dict(base)
    for k, v in delta.items():
        if k in out:
            out[k] = out[k] + v.to(dtype=out[k].dtype, device=out[k].device)
        else:
            out[k] = v
    return out


def compose_weighted_deltas(
    deltas: Sequence[Mapping[str, torch.Tensor]],
    weights: Sequence[float],
) -> dict[str, torch.Tensor]:
    if not deltas:
        return {}
    if len(weights) != len(deltas):
        raise ValueError("weights length must match deltas length.")

    shared_keys = set(deltas[0].keys())
    for delta in deltas[1:]:
        shared_keys &= set(delta.keys())

    out: dict[str, torch.Tensor] = {}
    for key in sorted(shared_keys):
        acc = torch.zeros_like(deltas[0][key])
        for weight, delta in zip(weights, deltas, strict=True):
            acc = acc + float(weight) * delta[key].to(dtype=acc.dtype, device=acc.device)
        out[key] = acc
    return out


def build_dense_delta_branch(
    *,
    tasks: Sequence[str],
    full_tuned_by_task: Mapping[str, Mapping[str, torch.Tensor]],
    lora_only_tuned_by_task: Mapping[str, Mapping[str, torch.Tensor]],
    base_sd: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], list[dict[str, torch.Tensor]]]:
    dense_keys: set[str] = set()
    dense_by_task: dict[str, dict[str, torch.Tensor]] = {}

    for task in tasks:
        full_sd = full_tuned_by_task[task]
        lora_sd = lora_only_tuned_by_task[task]
        task_dense: dict[str, torch.Tensor] = {}
        candidate_keys = set(full_sd.keys()) | set(lora_sd.keys())
        for key in candidate_keys:
            if key not in base_sd:
                continue
            base_value = base_sd[key]
            full_value = full_sd.get(key, base_value)
            lora_value = lora_sd.get(key, base_value)
            if torch.equal(full_value, lora_value):
                continue
            task_dense[key] = full_value.to(dtype=base_value.dtype) - lora_value.to(dtype=base_value.dtype)
        dense_by_task[task] = task_dense
        dense_keys.update(task_dense.keys())

    if not dense_keys:
        return {}, []

    ordered_keys = sorted(dense_keys)
    dense_base = {key: torch.zeros_like(base_sd[key]) for key in ordered_keys}
    dense_tuned: list[dict[str, torch.Tensor]] = []
    for task in tasks:
        task_dense = dense_by_task[task]
        dense_tuned.append(
            {
                key: task_dense.get(key, torch.zeros_like(base_sd[key]))
                for key in ordered_keys
            }
        )
    return dense_base, dense_tuned


def build_merged_state_for_alpha(
    *,
    method: Any,
    prepared: Any,
    base_sd_for_merge: dict[str, torch.Tensor],
    tuned_sds_list: Sequence[Mapping[str, torch.Tensor]],
    weights: Any,
    method_params: dict[str, Any],
    alpha: float,
    peft_subspace: str = "full",
    subspace: Any = None,
    subspace_prepared: Any = None,
    peft_cfg: dict[str, Any] | None = None,
    peft_state_by_task: dict[str, dict[str, torch.Tensor]] | None = None,
    tasks: list[str] | None = None,
    merge_base_sd: dict[str, torch.Tensor] | None = None,
    dense_prepared: Any = None,
    dense_base_sd_for_merge: dict[str, torch.Tensor] | None = None,
    dense_tuned_sds_list: Sequence[Mapping[str, torch.Tensor]] | None = None,
) -> dict[str, torch.Tensor]:
    merge_alpha = 1.0 if peft_subspace != "full" else float(alpha)
    if prepared is not None:
        merged_sd = method.apply(prepared=prepared, alpha=merge_alpha)
    else:
        merged_sd = method.merge(
            base=base_sd_for_merge,
            tuned=tuned_sds_list,
            weights=weights,
            alpha=merge_alpha,
            method_params=method_params,
        )

    if peft_subspace == "full":
        return merged_sd

    if subspace is None or subspace_prepared is None or peft_cfg is None or merge_base_sd is None:
        raise RuntimeError(f"Subspace '{peft_subspace}' was not prepared.")
    if not tasks:
        raise RuntimeError("Subspace lifting requires non-empty task ordering.")
    if peft_state_by_task is None or tasks[0] not in peft_state_by_task:
        raise RuntimeError("Subspace lifting requires PEFT state templates by task.")

    refine_merged_core = getattr(subspace, "refine_merged_core", None)
    if callable(refine_merged_core):
        merged_sd = refine_merged_core(
            subspace_prepared,
            merged_core=merged_sd,
            tuned_cores=tuned_sds_list,
            weights=weights,
            method_params=method_params,
            tasks=tasks,
            peft_cfg=peft_cfg,
        )

    merged_delta = subspace.lift(
        subspace_prepared,
        merged_core=merged_sd,
        lora_template=peft_state_by_task[tasks[0]],
        peft_cfg=peft_cfg,
    )
    if dense_tuned_sds_list and dense_base_sd_for_merge:
        if dense_prepared is not None:
            merged_dense_delta = method.apply(prepared=dense_prepared, alpha=1.0)
        else:
            merged_dense_delta = method.merge(
                base=dense_base_sd_for_merge,
                tuned=dense_tuned_sds_list,
                weights=weights,
                alpha=1.0,
                method_params=method_params,
            )
        for key, value in merged_dense_delta.items():
            if key in merged_delta:
                merged_delta[key] = merged_delta[key] + value.to(dtype=merged_delta[key].dtype, device=merged_delta[key].device)
            else:
                merged_delta[key] = value
    if float(alpha) != 1.0:
        merged_delta = {k: v * float(alpha) for k, v in merged_delta.items()}
    return apply_delta(merge_base_sd, merged_delta)


def prepared_base_direction(prepared: Any) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]] | None:
    if not isinstance(prepared, tuple) or len(prepared) != 2:
        return None
    base, direction = prepared
    if not isinstance(base, dict) or not isinstance(direction, dict):
        return None
    return base, direction


def load_prepared_direction_into_model(
    *,
    model: Any,
    base: dict[str, torch.Tensor],
    direction: dict[str, torch.Tensor],
    alpha: float,
    strict: bool,
) -> tuple[int, int]:
    model_sd = model.state_dict()
    unexpected = len(set(base.keys()) - set(model_sd.keys()))
    missing = 0
    a = float(alpha)
    with torch.no_grad():
        for k, tgt in model_sd.items():
            src = base.get(k, None)
            if src is None:
                missing += 1
                continue
            src_cpu = src
            if src_cpu.device.type != "cpu":
                src_cpu = src_cpu.detach().cpu()
            if src_cpu.dtype != tgt.dtype:
                src_cpu = src_cpu.to(dtype=tgt.dtype)

            # Copy base weights first, then add alpha-scaled direction in-place.
            tgt.copy_(src_cpu.to(device=tgt.device))
            if k in direction:
                delta_cpu = direction[k]
                if delta_cpu.device.type != "cpu":
                    delta_cpu = delta_cpu.detach().cpu()
                if delta_cpu.dtype != tgt.dtype:
                    delta_cpu = delta_cpu.to(dtype=tgt.dtype)
                tgt.add_(delta_cpu.to(device=tgt.device), alpha=a)
    if strict and (missing > 0 or unexpected > 0):
        raise RuntimeError(
            f"Strict load failed.\nmissing({missing}) keys in base dict\nunexpected({unexpected}) keys in base dict"
        )
    return missing, unexpected
