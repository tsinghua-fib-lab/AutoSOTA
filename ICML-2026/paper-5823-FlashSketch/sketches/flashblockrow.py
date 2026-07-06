from gitbud.gitbud import inject_repo_into_sys_path
inject_repo_into_sys_path()

from dataclasses import dataclass

import globals as g
from kernels.flashblockrow.flashblockrow import flashblockrow_cuda_apply


_STATIC_BLOCK_SIZE = 128
_STATIC_KAPPA = 2
_STATIC_S = 2
_STATIC_TC = 32


@dataclass(frozen=True)
class FlashBlockRowConfig:
    """Config for the FlashBlockRow CUDA kernel."""

    method: str = g.METHOD_FLASH_BLOCK_ROW
    k: int = 512
    seed: int = 0
    dtype: str = g.DTYPE_FP32
    block_size: int = _STATIC_BLOCK_SIZE
    kappa: int = _STATIC_KAPPA
    s: int = _STATIC_S
    tc: int = _STATIC_TC
    return_contiguous: bool = True


def sketch(A, cfg):
    """Apply the FlashBlockRow CUDA kernel."""
    return flashblockrow_cuda_apply(A, cfg)
