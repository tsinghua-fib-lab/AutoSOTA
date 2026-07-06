from gitbud.gitbud import inject_repo_into_sys_path
inject_repo_into_sys_path()

import math

import torch

from kernels.flashsketch.flashsketch_ext import flashsketch_cuda_forward
from torch_utils import resolve_dtype


_ALLOWED_KAPPA = (1, 2, 4)
_ALLOWED_S = (1, 2, 4, 8)
_STATIC_TN = 32
_STATIC_TK = 128


def _ceil_div(x, y):
    """Return ceil(x / y) for positive integers."""
    return (x + y - 1) // y


def _pad_rows(A, target_rows):
    """Pad A along rows with zeros up to target_rows."""
    if A.shape[0] == target_rows:
        return A
    if target_rows < A.shape[0]:
        raise ValueError("target_rows must be >= current rows")
    padded = torch.zeros((target_rows, A.shape[1]), device=A.device, dtype=A.dtype)
    padded[: A.shape[0], :] = A
    return padded


def _select_block_rows(cfg, k):
    """Return the block-row size for the sketch."""
    if getattr(cfg, "block_rows", 0) > 0:
        return int(cfg.block_rows)
    if k >= 2048:
        return 64
    return 128 if k >= 128 else 64



def flashsketch_cuda_apply(A, cfg):
    """Apply the FlashSketch CUDA kernel."""
    if A.ndim != 2:
        raise ValueError("A must be 2D with shape (d, n)")
    if not A.is_cuda:
        raise ValueError("flashsketch_cuda requires CUDA tensors")

    dtype = resolve_dtype(cfg.dtype)
    if dtype != torch.float32:
        raise ValueError("flashsketch_cuda only supports fp32")
    if A.dtype != torch.float32:
        raise ValueError("Input A must be fp32 for flashsketch_cuda")

    if cfg.k <= 0:
        raise ValueError("k must be positive")
    if cfg.k < 64:
        raise ValueError("flashsketch_cuda requires k >= 64")
    if cfg.kappa <= 0:
        raise ValueError("kappa must be positive")
    if cfg.s <= 0:
        raise ValueError("s must be positive")

    if cfg.kappa not in _ALLOWED_KAPPA or cfg.s not in _ALLOWED_S:
        raise ValueError(
            "flashsketch_cuda is static; supported config is "
            f"kappa={_ALLOWED_KAPPA}, s={_ALLOWED_S}"
        )

    d, n = A.shape
    block_rows = _select_block_rows(cfg, cfg.k)
    if block_rows not in (64, 128):
        raise ValueError("block_rows must be 64 or 128")

    k_pad = _ceil_div(cfg.k, block_rows) * block_rows
    m_blocks = k_pad // block_rows

    d_pad = _ceil_div(d, m_blocks) * m_blocks
    A_pad = _pad_rows(A, d_pad)
    if not A_pad.is_contiguous():
        A_pad = A_pad.contiguous()

    scale = 1.0 / math.sqrt(float(cfg.kappa) * float(cfg.s))

    y = flashsketch_cuda_forward(
        x=A_pad,
        k_total=k_pad,
        block_rows=block_rows,
        kappa=cfg.kappa,
        s=cfg.s,
        scale=scale,
        seed=cfg.seed,
        skip_zeros=bool(getattr(cfg, "skip_zeros", False)),
    )

    SA = y[: cfg.k, :]
    return SA
