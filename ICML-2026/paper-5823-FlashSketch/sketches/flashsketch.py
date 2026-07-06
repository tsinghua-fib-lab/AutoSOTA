from gitbud.gitbud import inject_repo_into_sys_path
inject_repo_into_sys_path()

from dataclasses import dataclass

import globals as g
from kernels.flashsketch.flashsketch import flashsketch_cuda_apply


@dataclass(frozen=True)
class FlashSketchConfig:
    """Config for the FlashSketch SJLT CUDA kernel."""

    method: str = g.METHOD_FLASH_SKETCH
    k: int = 512
    kappa: int = 2
    s: int = 2
    seed: int = 0
    dtype: str = g.DTYPE_FP32
    block_rows: int = 0
    skip_zeros: bool = True


def sketch(A, cfg):
    """Apply the FlashSketch kernel."""
    return flashsketch_cuda_apply(A, cfg)
