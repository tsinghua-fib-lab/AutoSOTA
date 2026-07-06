from gitbud.gitbud import inject_repo_into_sys_path
inject_repo_into_sys_path()

import torch

from kernels.so_loader import load_extension


_EXTENSION_NAME = "flashsketch_ext"
_EXTENSION = None


def _get_extension():
    """Return the loaded extension, raising if missing."""
    global _EXTENSION
    if _EXTENSION is None:
        _EXTENSION = load_extension(_EXTENSION_NAME)
    return _EXTENSION


def flashsketch_cuda_forward(
    *,
    x: torch.Tensor,
    k_total: int,
    block_rows: int,
    kappa: int,
    s: int,
    scale: float,
    seed: int,
    skip_zeros: bool,
) -> torch.Tensor:
    """Apply the FlashSketch kernel with static parameters."""
    if not x.is_cuda:
        raise ValueError("flashsketch_cuda_forward requires CUDA tensors")
    if x.dtype != torch.float32:
        raise ValueError("flashsketch_cuda_forward only supports float32")

    ext = _get_extension()
    return ext.forward(
        x.contiguous(),
        int(k_total),
        int(block_rows),
        int(kappa),
        int(s),
        float(scale),
        int(seed),
        bool(skip_zeros),
    )
