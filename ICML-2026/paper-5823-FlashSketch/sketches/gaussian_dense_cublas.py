from gitbud.gitbud import inject_repo_into_sys_path
inject_repo_into_sys_path()

from dataclasses import dataclass
import math

import torch

import globals as g
from torch_utils import resolve_dtype


@dataclass(frozen=True)
class GaussianDenseCublasConfig:
    """Config for dense Gaussian sketch using GEMM."""

    method: str = g.METHOD_GAUSSIAN_DENSE_CUBLAS
    k: int = 512
    seed: int = 0
    dtype: str = g.DTYPE_FP32
    scale: float = 0.0


def sketch(A, cfg):
    """Apply a dense Gaussian sketch using torch GEMM (cuBLAS backend)."""
    if A.ndim != 2:
        raise ValueError("A must be 2D with shape (d, n)")

    dtype = resolve_dtype(cfg.dtype)
    device = A.device
    if dtype != torch.float32:
        raise ValueError("Only fp32 is supported for gaussian_dense_cublas.")
    if A.dtype != torch.float32:
        raise ValueError("Input A must be fp32 for gaussian_dense_cublas.")
    A_work = A
    if not A_work.is_contiguous():
        A_work = A_work.contiguous()

    d = A_work.shape[0]
    generator = torch.Generator(device=device)
    generator.manual_seed(int(cfg.seed))
    scale = cfg.scale if cfg.scale > 0 else 1.0 / math.sqrt(cfg.k)
    S = torch.randn((cfg.k, d), device=device, dtype=dtype, generator=generator) * scale
    SA = S @ A_work
    return SA
