from gitbud.gitbud import inject_repo_into_sys_path
inject_repo_into_sys_path()

import math

import torch

from kernels.flashblockrow.flashblockrow_ext import flashblockrow_cuda_forward
from torch_utils import resolve_dtype


_STATIC_BLOCK_SIZE = 128
_STATIC_TC = 32
_SUPPORTED_KAPPA_S = {(1, 1), (2, 1), (4, 1), (1, 2), (2, 2), (1, 4)}


def _ceil_div(x, y):
    """Return ceil(x / y) for positive integers."""
    return (x + y - 1) // y


def _next_pow2(x):
    """Return the next power of two >= x for positive integers."""
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def _pad_rows(A, target_rows):
    """Pad A along rows with zeros up to target_rows."""
    if A.shape[0] == target_rows:
        return A
    if target_rows < A.shape[0]:
        raise ValueError("target_rows must be >= current rows")
    padded = torch.zeros((target_rows, A.shape[1]), device=A.device, dtype=A.dtype)
    padded[: A.shape[0], :] = A
    return padded


def flashblockrow_cuda_apply(A, cfg):
    """Apply the FlashBlockRow CUDA kernel."""
    if A.ndim != 2:
        raise ValueError("A must be 2D with shape (d, n)")
    if not A.is_cuda:
        raise ValueError("flashblockrow_cuda requires CUDA tensors")

    dtype = resolve_dtype(cfg.dtype)
    if dtype != torch.float32:
        raise ValueError("flashblockrow_cuda only supports fp32")
    if A.dtype != torch.float32:
        raise ValueError("Input A must be fp32 for flashblockrow_cuda")

    if cfg.block_size <= 0:
        raise ValueError("block_size must be positive")
    if cfg.block_size > 1024:
        raise ValueError("block_size must be <= 1024")
    if cfg.kappa <= 0:
        raise ValueError("kappa must be positive")
    if cfg.s <= 0:
        raise ValueError("s must be positive")
    if cfg.tc <= 0:
        raise ValueError("tc must be positive")
    if cfg.tc > 32:
        raise ValueError("tc must be <= 32")
    if cfg.k <= 0:
        raise ValueError("k must be positive")

    d, n = A.shape
    block_size = int(cfg.block_size)
    kappa = int(cfg.kappa)
    s = int(cfg.s)
    tc = int(cfg.tc)
    if block_size != _STATIC_BLOCK_SIZE or tc != _STATIC_TC or (kappa, s) not in _SUPPORTED_KAPPA_S:
        supported = ", ".join(f"(kappa={k}, s={s})" for k, s in sorted(_SUPPORTED_KAPPA_S))
        raise ValueError(
            "flashblockrow_cuda is static; supported config is "
            f"block_size={_STATIC_BLOCK_SIZE}, tc={_STATIC_TC}, {supported}"
        )

    d_blocks = _next_pow2(max(_ceil_div(d, block_size), kappa))
    d_pad = d_blocks * block_size
    k_pad = _ceil_div(cfg.k, block_size) * block_size

    A_pad = _pad_rows(A, d_pad)
    if not A_pad.is_contiguous():
        A_pad = A_pad.contiguous()

    seed = int(cfg.seed)
    alpha = math.sqrt(float(d) / (float(cfg.k) * float(kappa) * float(s)))

    y = flashblockrow_cuda_forward(
        x=A_pad,
        k_total=k_pad,
        block_size=block_size,
        kappa=kappa,
        s=s,
        tc=tc,
        alpha=alpha,
        seed=seed,
    )

    SA = y[: cfg.k, :]
    if cfg.return_contiguous and not SA.is_contiguous():
        SA = SA.contiguous()
    return SA
