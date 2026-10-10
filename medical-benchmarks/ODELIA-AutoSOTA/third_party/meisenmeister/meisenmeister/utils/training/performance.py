from __future__ import annotations

from contextlib import nullcontext

import torch


def configure_training_performance(device: torch.device) -> None:
    if device.type == "cuda" and torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
    if device.type == "cuda" and torch.backends.cuda.is_built():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")


def resolve_compile_enabled(
    device: torch.device,
    *,
    enabled: bool = True,
) -> tuple[bool, str]:
    if not enabled:
        return False, "disabled by caller"
    if not hasattr(torch, "compile"):
        return False, "torch.compile unavailable in this PyTorch build"
    if device.type != "cuda":
        return False, f"unsupported device type '{device.type}'"
    return True, "enabled for CUDA"


def maybe_compile_model(
    model: torch.nn.Module,
    *,
    device: torch.device,
    enabled: bool = True,
) -> tuple[torch.nn.Module, bool, str]:
    compile_enabled, message = resolve_compile_enabled(device, enabled=enabled)
    if not compile_enabled:
        return model, False, message

    try:
        compiled_model = torch.compile(model)
    except Exception as error:
        return model, False, f"compile failed, falling back to eager mode: {error}"
    return compiled_model, True, message


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "_orig_mod", model)


def is_amp_enabled(device: torch.device) -> bool:
    return device.type == "cuda"


def resolve_amp_dtype(device: torch.device) -> torch.dtype | None:
    if not is_amp_enabled(device):
        return None
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def create_grad_scaler(device: torch.device, amp_dtype: torch.dtype | None):
    if not is_amp_enabled(device) or amp_dtype != torch.float16:
        return None
    return torch.amp.GradScaler("cuda")


def autocast_context(
    device: torch.device,
    amp_dtype: torch.dtype | None,
):
    if not is_amp_enabled(device):
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=True)
