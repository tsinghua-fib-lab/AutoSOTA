import torch
from transformers import set_seed


def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def set_launch_seed(seed: int | None) -> None:
    if seed is not None:
        set_seed(seed)


def freeze_model_parameters(model: torch.nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def resolve_torch_dtype(dtype_str: str) -> torch.dtype:
    s = (dtype_str or "").lower()
    if s in {"auto"}:
        return "auto"
    if s in {"float32", "fp32"}:
        return torch.float32
    if s in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if s in {"float16", "fp16"}:
        return torch.float16
    # Fallback to float32 for unknown values
    return torch.float32
