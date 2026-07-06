from gitbud.gitbud import inject_repo_into_sys_path
inject_repo_into_sys_path()

import torch

from kernels.so_loader import load_extension


_EXTENSION_NAME = "flashblockrow_ext"
_EXTENSION = None


def _get_extension():
    """Return the loaded extension, raising if missing."""
    global _EXTENSION
    if _EXTENSION is None:
        _EXTENSION = load_extension(_EXTENSION_NAME)
    return _EXTENSION


def flashblockrow_cuda_forward(
    *,
    x: torch.Tensor,
    k_total: int,
    block_size: int,
    kappa: int,
    s: int,
    tc: int,
    alpha: float,
    seed: int,
) -> torch.Tensor:
    """Apply the FlashBlockRow CUDA kernel."""
    if not x.is_cuda:
        raise ValueError("flashblockrow_cuda_forward requires CUDA tensors")
    if x.dtype != torch.float32:
        raise ValueError("flashblockrow_cuda_forward only supports float32")

    ext = _get_extension()
    return ext.forward(
        x.contiguous(),
        int(k_total),
        int(block_size),
        int(kappa),
        int(s),
        int(tc),
        float(alpha),
        int(seed),
    )
