from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from .registry import register


@dataclass(frozen=True)
class LoraLayer:
    a_key: str
    b_key: str
    a: torch.Tensor
    b: torch.Tensor


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
    if layer_key.startswith("visual."):
        tail = layer_key[len("visual.") :]
        candidates.append(tail)
        candidates.append(f"base_model.model.{tail}")
    candidates.append(f"base_model.model.{layer_key}")
    for k in candidates:
        if k in pattern:
            return pattern[k]
    return default


def _lora_scaling_for_layer(layer_key: str, layer: LoraLayer, peft_cfg: dict[str, Any]) -> float:
    # Match PEFT's per-layer scaling behavior.
    rank_pattern = peft_cfg.get("rank_pattern", {}) if isinstance(peft_cfg.get("rank_pattern", {}), dict) else {}
    alpha_pattern = peft_cfg.get("alpha_pattern", {}) if isinstance(peft_cfg.get("alpha_pattern", {}), dict) else {}
    default_alpha = float(peft_cfg.get("lora_alpha", 1.0))
    use_rslora = bool(peft_cfg.get("use_rslora", False))

    r_eff = int(layer.a.shape[0])
    r_cfg = int(_lookup_layer_pattern(rank_pattern, layer_key=layer_key, default=r_eff))
    if r_cfg <= 0:
        r_cfg = r_eff
    alpha = float(_lookup_layer_pattern(alpha_pattern, layer_key=layer_key, default=default_alpha))
    denom = (r_cfg**0.5) if use_rslora else float(r_cfg)
    return float(alpha / max(1e-12, denom))


def _normalize_prefix(prefix: str) -> str:
    # PEFT often prefixes with base_model.model.* even when wrapping a submodule.
    for p in ("base_model.model.", "model."):
        if prefix.startswith(p):
            prefix = prefix[len(p) :]
            break
    if prefix.startswith("encoder.layers."):
        prefix = "visual.transformer.resblocks." + prefix[len("encoder.layers.") :]
    prefix = prefix.replace(".self_attn.", ".attn.")
    # If adapter was applied to visual, PEFT keys may start at transformer.*
    if prefix.startswith("transformer."):
        prefix = "visual." + prefix
    return prefix


def build_lora_groups(peft_state: dict[str, torch.Tensor]) -> dict[str, LoraLayer]:
    """
    Group LoRA tensors by logical layer key.
    Expects keys like: <prefix>.lora_A.weight, <prefix>.lora_B.weight
    """
    groups: dict[str, dict[str, Any]] = {}
    for k, v in peft_state.items():
        if ".lora_A" in k:
            prefix = _normalize_prefix(k.split(".lora_A")[0])
            groups.setdefault(prefix, {})["a_key"] = k
            groups[prefix]["a"] = v
        elif ".lora_B" in k:
            prefix = _normalize_prefix(k.split(".lora_B")[0])
            groups.setdefault(prefix, {})["b_key"] = k
            groups[prefix]["b"] = v

    out: dict[str, LoraLayer] = {}
    for prefix, payload in groups.items():
        if "a" in payload and "b" in payload:
            out[prefix] = LoraLayer(
                a_key=payload["a_key"],
                b_key=payload["b_key"],
                a=payload["a"],
                b=payload["b"],
            )
    return out


@dataclass(frozen=True)
class CorePrepared:
    """
    Stores shared bases per layer.
    """

    basis_method: str
    bases: dict[str, dict[str, torch.Tensor]]  # layer_key -> {"U": U, "V": V}


@dataclass(frozen=True)
class CoreSpace:
    name: str = "core"

    @staticmethod
    def _resolve_basis_method(method_params: dict[str, Any] | None) -> str:
        params = method_params or {}
        basis_method = str(params.get("core_basis_method", params.get("basis_method", "qr"))).strip().lower()
        if basis_method not in {"svd", "qr"}:
            raise ValueError(f"core subspace basis_method must be one of: 'svd', 'qr' (got {basis_method!r}).")
        return basis_method

    @staticmethod
    def _build_basis(
        *,
        a_stack: torch.Tensor,
        b_stack: torch.Tensor,
        basis_method: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if basis_method == "svd":
            _u_a, _s_a, v_a_h = torch.linalg.svd(a_stack.to(torch.float64), full_matrices=False)
            u_b, _s_b, _v_b_h = torch.linalg.svd(b_stack.to(torch.float64), full_matrices=False)
            return u_b, v_a_h
        if basis_method == "qr":
            u_b = torch.linalg.qr(b_stack.double(), mode="reduced")[0]
            v_a_h = torch.linalg.qr(a_stack.T.double(), mode="reduced")[0].T
            return u_b, v_a_h
        raise AssertionError(f"Unhandled core basis method: {basis_method}")

    def prepare(
        self,
        *,
        lora_by_task: dict[str, dict[str, torch.Tensor]],
        peft_cfg: dict[str, Any],
        method_params: dict[str, Any] | None = None,
        weights: Sequence[float] | None = None,
        artifact_dir: str | Path | None = None,
    ) -> CorePrepared:
        if not lora_by_task:
            raise ValueError("lora_by_task is empty.")

        _ = weights
        _ = artifact_dir
        basis_method = self._resolve_basis_method(method_params)
        tasks = list(lora_by_task.keys())
        layer_groups = {t: build_lora_groups(lora_by_task[t]) for t in tasks}
        if not layer_groups[tasks[0]]:
            raise ValueError("No LoRA layers found in peft_state_dict.")

        bases: dict[str, dict[str, torch.Tensor]] = {}
        ref_layers = layer_groups[tasks[0]]

        for layer_key, _ in tqdm(ref_layers.items(), desc="Preparing Core basis", unit="layer"):
            a_list = []
            b_list = []
            for t in tasks:
                layer = layer_groups[t].get(layer_key, None)
                if layer is None:
                    raise ValueError(f"Missing LoRA layer '{layer_key}' for task '{t}'.")
                a_list.append(layer.a.to(dtype=torch.float32))
                b_list.append(layer.b.to(dtype=torch.float32))

            a_stack = torch.cat(a_list, dim=0)  # (N*r, in)
            b_stack = torch.cat(b_list, dim=1)  # (out, N*r)

            u_b, v_a_h = self._build_basis(
                a_stack=a_stack,
                b_stack=b_stack,
                basis_method=basis_method,
            )

            bases[layer_key] = {
                "U": u_b.to(dtype=torch.float32).contiguous(),
                "V": v_a_h.to(dtype=torch.float32).contiguous(),
            }

        return CorePrepared(basis_method=basis_method, bases=bases)

    def project(
        self,
        prepared: CorePrepared,
        *,
        lora_by_task: dict[str, dict[str, torch.Tensor]],
        peft_cfg: dict[str, Any],
    ) -> dict[str, dict[str, torch.Tensor]]:
        core_by_task: dict[str, dict[str, torch.Tensor]] = {}
        for task, peft_state in lora_by_task.items():
            layers = build_lora_groups(peft_state)
            out_layers: dict[str, torch.Tensor] = {}
            for layer_key, layer in layers.items():
                if layer_key not in prepared.bases:
                    continue
                U = prepared.bases[layer_key]["U"]
                V = prepared.bases[layer_key]["V"]
                b = layer.b.to(dtype=U.dtype, device=U.device)
                a = layer.a.to(dtype=U.dtype, device=U.device)
                # scale = _lora_scaling_for_layer(layer_key, layer, peft_cfg)
                core = (U.T @ b) @ (a @ V.T)
                out_layers[layer_key] = core
            core_by_task[task] = out_layers
        return core_by_task

    def lift(
        self,
        prepared: CorePrepared,
        *,
        merged_core: dict[str, torch.Tensor],
        lora_template: dict[str, torch.Tensor],
        peft_cfg: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        template_layers = build_lora_groups(lora_template)
        out: dict[str, torch.Tensor] = {}

        for layer_key, tpl in template_layers.items():
            if layer_key not in prepared.bases:
                continue
            core = merged_core.get(layer_key, None)
            if core is None:
                delta = torch.zeros_like(tpl.b @ tpl.a)
            else:
                U = prepared.bases[layer_key]["U"]
                V = prepared.bases[layer_key]["V"]
                c = core.to(dtype=U.dtype, device=U.device)
                delta = (U @ c @ V).to(dtype=torch.float32)

            # Map LoRA layer prefix to base weight key
            base_key = f"{layer_key}.weight"
            out[base_key] = delta.to(dtype=tpl.b.dtype)

        return out


register(CoreSpace())
