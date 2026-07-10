from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import torch


@runtime_checkable
class Subspace(Protocol):
    name: str

    def prepare(
        self,
        *,
        lora_by_task: dict[str, dict[str, torch.Tensor]],
        peft_cfg: dict[str, Any],
        method_params: dict[str, Any] | None = None,
        weights: Sequence[float] | None = None,
        artifact_dir: str | Path | None = None,
    ) -> Any:
        """
        Build any shared bases / metadata needed for projection.
        """

    def project(
        self,
        prepared: Any,
        *,
        lora_by_task: dict[str, dict[str, torch.Tensor]],
        peft_cfg: dict[str, Any],
    ) -> dict[str, dict[str, torch.Tensor]]:
        """
        Project LoRA adapters into subspace.
        Returns: task -> {layer_key -> core_tensor}
        """

    def lift(
        self,
        prepared: Any,
        *,
        merged_core: dict[str, torch.Tensor],
        lora_template: dict[str, torch.Tensor],
        peft_cfg: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        """
        Lift merged core representation back to full-space deltas keyed
        by the base model parameter names.
        """
