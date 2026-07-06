from gitbud.gitbud import inject_repo_into_sys_path
inject_repo_into_sys_path()

from dataclasses import dataclass

import torch

import globals as g
from kernels.grass_sjlt.grass_sjlt import sjlt_projection_cuda
from torch_utils import resolve_dtype


@dataclass(frozen=True)
class SjltGrassKernelConfig:
    """Config for the GraSS SJLT CUDA kernel baseline."""

    method: str = g.METHOD_SJLT_GRASS_KERNEL
    k: int = 512
    s: int = 4
    seed: int = 0
    dtype: str = g.DTYPE_FP32
    threads: int = 1024
    fixed_blocks: int = 84
    expects_transposed: bool = True


def sketch(A, cfg):
    """Apply the GraSS SJLT kernel baseline."""
    if A.ndim != 2:
        raise ValueError("A must be 2D")
    if not A.is_cuda:
        raise ValueError("grass_sjlt_kernel requires CUDA tensors")

    dtype = resolve_dtype(cfg.dtype)
    if dtype != torch.float32:
        raise ValueError("grass_sjlt_kernel only supports fp32")
    if A.dtype != torch.float32:
        raise ValueError("Input A must be fp32 for grass_sjlt_kernel")

    if cfg.k <= 0:
        raise ValueError("k must be positive")
    if cfg.s <= 0:
        raise ValueError("s must be positive")
    if cfg.k < cfg.s:
        raise ValueError("k must be >= s")

    device = A.device
    batch_size, original_dim = A.shape
    generator = torch.Generator(device=device)
    generator.manual_seed(int(cfg.seed))

    rand_indices = torch.randint(
        0,
        cfg.k,
        (original_dim, cfg.s),
        generator=generator,
        device=device,
        dtype=torch.int64,
    )
    rand_signs = torch.randint(
        0,
        2,
        (original_dim, cfg.s),
        generator=generator,
        device=device,
        dtype=torch.int8,
    )
    rand_signs = rand_signs * 2 - 1

    output = sjlt_projection_cuda(
        A,
        rand_indices,
        rand_signs,
        cfg.k,
        cfg.s,
        cfg.threads,
        cfg.fixed_blocks,
    )[0]
    return output
