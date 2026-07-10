from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .registry import register


@dataclass(frozen=True)
class IdentitySpace:
    """
    Identity subspace: merges in full weight space.
    """

    name: str = "full"

    def prepare(
        self,
        *,
        lora_by_task: dict[str, dict[str, torch.Tensor]],
        peft_cfg: dict[str, Any],
        method_params: dict[str, Any] | None = None,
        weights: Sequence[float] | None = None,
        artifact_dir: str | Path | None = None,
    ) -> Any:
        _ = lora_by_task
        _ = peft_cfg
        _ = method_params
        _ = weights
        _ = artifact_dir
        return None

    def project(
        self,
        prepared: Any,
        *,
        lora_by_task: dict[str, dict[str, torch.Tensor]],
        peft_cfg: dict[str, Any],
    ) -> dict[str, dict[str, torch.Tensor]]:
        raise RuntimeError("Identity subspace does not project LoRA adapters.")

    def lift(
        self,
        prepared: Any,
        *,
        merged_core: dict[str, torch.Tensor],
        lora_template: dict[str, torch.Tensor],
        peft_cfg: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        raise RuntimeError("Identity subspace does not lift LoRA adapters.")


register(IdentitySpace())
