from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from .core_space import _lora_scaling_for_layer, build_lora_groups
from .registry import register


@dataclass(frozen=True)
class KnotsLayerBasis:
    U: torch.Tensor
    S: torch.Tensor
    V_by_task: dict[str, torch.Tensor]


@dataclass(frozen=True)
class KnotsPrepared:
    tasks: tuple[str, ...]
    layers: dict[str, KnotsLayerBasis]


@dataclass(frozen=True)
class KnotsSpace:
    name: str = "knots"

    def prepare(
        self,
        *,
        lora_by_task: dict[str, dict[str, torch.Tensor]],
        peft_cfg: dict[str, Any],
        method_params: dict[str, Any] | None = None,
        weights: Sequence[float] | None = None,
        artifact_dir: str | Path | None = None,
    ) -> KnotsPrepared:
        if not lora_by_task:
            raise ValueError("lora_by_task is empty.")

        _ = method_params
        _ = weights
        _ = artifact_dir
        svd_eps = float(peft_cfg.get("knots_svd_eps", 1e-5))

        tasks = tuple(lora_by_task.keys())
        layer_groups = {t: build_lora_groups(lora_by_task[t]) for t in tasks}
        if not layer_groups[tasks[0]]:
            raise ValueError("No LoRA layers found in peft_state_dict.")

        layers: dict[str, KnotsLayerBasis] = {}
        for layer_key in tqdm(layer_groups[tasks[0]], desc="Preparing KnOTS basis", unit="layer"):
            full_updates: list[torch.Tensor] = []
            in_dim: int | None = None
            for task in tasks:
                layer = layer_groups[task].get(layer_key, None)
                if layer is None:
                    raise ValueError(f"Missing LoRA layer '{layer_key}' for task '{task}'.")
                scale = _lora_scaling_for_layer(layer_key, layer, peft_cfg)
                delta = scale * (layer.b.to(dtype=torch.float32) @ layer.a.to(dtype=torch.float32))
                full_updates.append(delta)
                if in_dim is None:
                    in_dim = int(delta.shape[1])

            if in_dim is None:
                continue

            concat_update = torch.cat(full_updates, dim=1).to(dtype=torch.float64)
            u, s, vh = torch.linalg.svd(concat_update, full_matrices=False)

            mask = s > svd_eps
            u = u[:, mask].to(dtype=torch.float32)
            vh = vh[mask].to(dtype=torch.float32)
            s = s[mask].to(dtype=torch.float32)
            s[s <= svd_eps] = 0.0

            v_by_task: dict[str, torch.Tensor] = {}
            for i, task in enumerate(tasks):
                col_start = i * in_dim
                col_end = (i + 1) * in_dim
                v_by_task[task] = vh[:, col_start:col_end].contiguous()

            layers[layer_key] = KnotsLayerBasis(
                U=u.contiguous(),
                S=s.contiguous(),
                V_by_task=v_by_task,
            )

        return KnotsPrepared(tasks=tasks, layers=layers)

    def project(
        self,
        prepared: KnotsPrepared,
        *,
        lora_by_task: dict[str, dict[str, torch.Tensor]],
        peft_cfg: dict[str, Any],
    ) -> dict[str, dict[str, torch.Tensor]]:
        projected_by_task: dict[str, dict[str, torch.Tensor]] = {}
        prepared_tasks = set(prepared.tasks)

        for task in lora_by_task:
            if task not in prepared_tasks:
                raise ValueError(f"Task '{task}' not available in prepared KnOTS basis: {sorted(prepared_tasks)}")

            out_layers: dict[str, torch.Tensor] = {}
            for layer_key, layer_basis in prepared.layers.items():
                v_task = layer_basis.V_by_task.get(task, None)
                if v_task is not None:
                    out_layers[layer_key] = v_task
            projected_by_task[task] = out_layers

        return projected_by_task

    def lift(
        self,
        prepared: KnotsPrepared,
        *,
        merged_core: dict[str, torch.Tensor],
        lora_template: dict[str, torch.Tensor],
        peft_cfg: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        template_layers = build_lora_groups(lora_template)
        out: dict[str, torch.Tensor] = {}

        for layer_key, tpl in template_layers.items():
            basis = prepared.layers.get(layer_key, None)
            if basis is None:
                continue

            merged_v = merged_core.get(layer_key, None)
            if merged_v is None:
                delta = torch.zeros_like(tpl.b @ tpl.a)
            else:
                v = merged_v.to(dtype=basis.U.dtype, device=basis.U.device)
                delta = (basis.U @ (basis.S[:, None] * v)).to(dtype=torch.float32)

            base_key = f"{layer_key}.weight"
            out[base_key] = delta.to(dtype=tpl.b.dtype)

        return out


register(KnotsSpace())
