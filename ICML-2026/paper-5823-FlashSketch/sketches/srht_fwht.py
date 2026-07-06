from gitbud.gitbud import inject_repo_into_sys_path
inject_repo_into_sys_path()

from dataclasses import dataclass
import math

import torch

import globals as g
from torch_utils import resolve_dtype


@dataclass(frozen=True)
class SrhtFwhtConfig:
    """Config for an SRHT-style FWHT sketch."""

    method: str = g.METHOD_SRHT_FWHT
    k: int = 512
    seed: int = 0
    dtype: str = g.DTYPE_FP32
    backend: str = g.FWHT_BACKEND_FAST


_MAX_FAST_DIM = 32768


def _next_pow2(x):
    """Return the smallest power of two >= x."""
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


def _fwht_torch(x):
    """Apply an FWHT using torch ops along the last dimension."""
    n = x.shape[-1]
    if n & (n - 1) != 0:
        raise ValueError("FWHT requires power-of-two dimension")

    orig_shape = x.shape
    y = x.reshape(-1, n)
    h = 1
    while h < n:
        y = y.view(-1, n // (2 * h), 2 * h)
        a = y[..., :h]
        b = y[..., h:]
        y = torch.cat((a + b, a - b), dim=-1)
        h *= 2
    return y.view(orig_shape)


def _fwht_fast(x):
    """Apply an FWHT using the fast_hadamard_transform kernel with recursion."""
    n = x.shape[-1]
    if n & (n - 1) != 0:
        raise ValueError("FWHT requires power-of-two dimension")

    if n <= _MAX_FAST_DIM:
        from fast_hadamard_transform import hadamard_transform

        return hadamard_transform(x, scale=1.0)

    orig_shape = x.shape
    y = x.reshape(-1, n)
    half = n // 2
    a = y[..., :half]
    b = y[..., half:]
    y1 = _fwht_fast(a + b)
    y2 = _fwht_fast(a - b)
    y = torch.cat((y1, y2), dim=-1)
    return y.view(orig_shape)


def sketch(A, cfg):
    """Apply an SRHT sketch using the fast Hadamard transform."""
    if A.ndim != 2:
        raise ValueError("A must be 2D with shape (d, n)")
    if not A.is_cuda:
        raise ValueError("srht_fwht requires CUDA tensors")

    dtype = resolve_dtype(cfg.dtype)
    if dtype != torch.float32:
        raise ValueError("srht_fwht only supports fp32")
    if A.dtype != torch.float32:
        raise ValueError("Input A must be fp32 for srht_fwht")

    if cfg.k <= 0:
        raise ValueError("k must be positive")

    generator = torch.Generator(device=A.device)
    generator.manual_seed(int(cfg.seed))

    d, n = A.shape
    d_pad = _next_pow2(max(d, cfg.k))
    A_pad = _pad_rows(A, d_pad)
    if not A_pad.is_contiguous():
        A_pad = A_pad.contiguous()

    signs = torch.randint(
        0,
        2,
        (d_pad,),
        device=A.device,
        dtype=torch.int8,
        generator=generator,
    )
    signs = signs.to(A.dtype).mul_(2).add_(-1)
    A_signed = A_pad * signs[:, None]

    X = A_signed.transpose(0, 1).contiguous()
    scale = 1.0 / math.sqrt(float(cfg.k))
    if cfg.backend == g.FWHT_BACKEND_FAST:
        X = _fwht_fast(X) * scale
    elif cfg.backend == g.FWHT_BACKEND_TORCH:
        X = _fwht_torch(X) * scale
    else:
        raise ValueError(f"Unknown FWHT backend: {cfg.backend}")
    X = X.transpose(0, 1)

    perm = torch.randperm(d_pad, device=A.device, generator=generator)
    idx = perm[: cfg.k]
    return X.index_select(0, idx)
