"""Shared model-level utilities.

Helpers for resolving HuggingFace model paths and collecting
non-persistent buffer names from model modules.
"""

from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download
from torch import nn


def resolve_model_path(model_id: str, **config_kwargs: Any) -> Path:
    """Resolve a HuggingFace Hub ID or local path to a local directory.

    Args:
        model_id: HuggingFace model ID or local path.
        **config_kwargs: Forwarded to snapshot_download
            (e.g. ``trust_remote_code``).
    """
    path = Path(model_id)
    if path.is_dir():
        return path
    return Path(
        snapshot_download(
            model_id,
            allow_patterns=["*.safetensors", "*.json"],
            **config_kwargs,
        )
    )


def non_persistent_buffer_names(model: nn.Module) -> set[str]:
    """Collect fully-qualified names of non-persistent buffers.

    Non-persistent buffers (e.g., rotary inv_freq) are recomputed from
    config and should not be serialized in checkpoints.
    """
    result: set[str] = set()
    for module_name, module in model.named_modules():
        for buf_name in module._non_persistent_buffers_set:
            full = f"{module_name}.{buf_name}" if module_name else buf_name
            result.add(full)
    return result
