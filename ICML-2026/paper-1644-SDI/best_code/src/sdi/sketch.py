"""
Sketching primitives for SDI/TracIn.

We use:
- CountSketch for vector-like gradients (biases, norm weights, etc.)
- TensorSketch for matrix gradients which factorize as outer products
  (Linear weights, Embedding weights).

Implementation notes:
- All sketch outputs are returned as **float32** for numerical stability and
  to ensure FFT support even when the model uses bf16/fp16.
- Hash/sign tensors are stored on the same device as the parameter they sketch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class CountSketchSpec:
    """CountSketch spec for a length-d vector into dimension m."""

    h: torch.Tensor  # (d,) int64, values in [0, m)
    s: torch.Tensor  # (d,) int8/int64, values in {-1, +1}
    m: int


@dataclass(frozen=True)
class TensorSketchSpec:
    """TensorSketch spec for a (d_out, d_in) matrix into dimension m."""

    h_out: torch.Tensor  # (d_out,) int64 in [0, m)
    s_out: torch.Tensor  # (d_out,) in {-1, +1}
    h_in: torch.Tensor  # (d_in,) int64 in [0, m)
    s_in: torch.Tensor  # (d_in,) in {-1, +1}
    m: int


def _rand_signs(
    *, size: int, device: torch.device, generator: Optional[torch.Generator]
) -> torch.Tensor:
    # int8 would be fine, but scatter_add expects same dtype as values; we cast later anyway.
    s = torch.randint(
        0, 2, (size,), device=device, generator=generator, dtype=torch.int64
    )
    return s * 2 - 1


def make_count_sketch(
    *,
    d: int,
    m: int,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> CountSketchSpec:
    if d <= 0:
        raise ValueError(f"CountSketch requires d>0, got d={d}")
    if m <= 0:
        raise ValueError(f"CountSketch requires m>0, got m={m}")
    # Pairwise-independent CW-trick hash: h(i) = (a*i + b) mod P mod m
    # where P = 2^31 - 1 (Mersenne prime). This is sufficient for CountSketch
    # and empirically yields tighter variance constants than fully independent hash.
    P = 2147483647  # 2^31 - 1
    a = torch.randint(1, P, (1,), device=device, generator=generator, dtype=torch.int64)
    b = torch.randint(0, P, (1,), device=device, generator=generator, dtype=torch.int64)
    indices = torch.arange(d, device=device, dtype=torch.int64)
    h = ((a * indices + b) % P) % m
    s = _rand_signs(size=d, device=device, generator=generator)
    return CountSketchSpec(h=h, s=s, m=int(m))


def make_tensor_sketch(
    *,
    d_out: int,
    d_in: int,
    m: int,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> TensorSketchSpec:
    if d_out <= 0 or d_in <= 0:
        raise ValueError(f"TensorSketch requires d_out,d_in>0, got {(d_out, d_in)}")
    if m <= 0:
        raise ValueError(f"TensorSketch requires m>0, got m={m}")
    # Pairwise-independent CW-trick hash with independent (a,b) per dimension.
    P = 2147483647  # 2^31 - 1
    # h_out
    a_out = torch.randint(1, P, (1,), device=device, generator=generator, dtype=torch.int64)
    b_out = torch.randint(0, P, (1,), device=device, generator=generator, dtype=torch.int64)
    idx_out = torch.arange(d_out, device=device, dtype=torch.int64)
    h_out = ((a_out * idx_out + b_out) % P) % m
    s_out = _rand_signs(size=d_out, device=device, generator=generator)
    # h_in (independent a,b)
    a_in = torch.randint(1, P, (1,), device=device, generator=generator, dtype=torch.int64)
    b_in = torch.randint(0, P, (1,), device=device, generator=generator, dtype=torch.int64)
    idx_in = torch.arange(d_in, device=device, dtype=torch.int64)
    h_in = ((a_in * idx_in + b_in) % P) % m
    s_in = _rand_signs(size=d_in, device=device, generator=generator)
    return TensorSketchSpec(h_out=h_out, s_out=s_out, h_in=h_in, s_in=s_in, m=int(m))


def _flatten_tokens(x: torch.Tensor) -> torch.Tensor:
    """
    Convert x shaped (B, ..., d) into (B, M, d) by flattening all non-batch, non-last dims.
    """
    if x.dim() < 2:
        raise ValueError(
            f"Expected tensor with dim>=2 (B,...), got shape={tuple(x.shape)}"
        )
    if x.dim() == 2:
        return x.unsqueeze(1)
    B = x.shape[0]
    d = x.shape[-1]
    return x.reshape(B, -1, d)


def count_sketch_tokens(
    x_bmd: torch.Tensor,
    *,
    spec: CountSketchSpec,
) -> torch.Tensor:
    """
    CountSketch a token/broadcasted tensor x of shape (B, M, d) into (B, M, m).
    """
    if x_bmd.dim() != 3:
        raise ValueError(f"Expected x shape (B,M,d), got {tuple(x_bmd.shape)}")
    B, M, d = x_bmd.shape
    if spec.h.numel() != d:
        raise ValueError(
            f"Hash length mismatch: h has {spec.h.numel()}, expected d={d}"
        )
    # Compute values and indices for scatter_add.
    x = x_bmd.to(dtype=torch.float32)
    val = x * spec.s.view(1, 1, d).to(dtype=torch.float32)
    idx = spec.h.view(1, 1, d).expand(B, M, d)
    out = torch.zeros(B, M, spec.m, device=x.device, dtype=torch.float32)
    out.scatter_add_(2, idx, val)
    return out


def count_sketch_vector(
    x_bd: torch.Tensor,
    *,
    spec: CountSketchSpec,
) -> torch.Tensor:
    """
    CountSketch a batch of vectors x of shape (B, d) into (B, m).
    """
    if x_bd.dim() != 2:
        raise ValueError(f"Expected x shape (B,d), got {tuple(x_bd.shape)}")
    B, d = x_bd.shape
    if spec.h.numel() != d:
        raise ValueError(
            f"Hash length mismatch: h has {spec.h.numel()}, expected d={d}"
        )
    x = x_bd.to(dtype=torch.float32)
    val = x * spec.s.view(1, d).to(dtype=torch.float32)
    idx = spec.h.view(1, d).expand(B, d)
    out = torch.zeros(B, spec.m, device=x.device, dtype=torch.float32)
    out.scatter_add_(1, idx, val)
    return out


def tensor_sketch_linear_weight(
    *,
    activations: torch.Tensor,
    backprops: torch.Tensor,
    spec: TensorSketchSpec,
) -> torch.Tensor:
    """
    TensorSketch for a Linear weight gradient, using outer-product factorization.

    activations: (..., d_in)
    backprops:   (..., d_out)

    Returns: (B, m)
    """
    a = _flatten_tokens(activations)
    bp = _flatten_tokens(backprops)
    if a.shape[0] != bp.shape[0] or a.shape[1] != bp.shape[1]:
        raise ValueError(
            f"Activation/backprop token shape mismatch: a={tuple(a.shape)} bp={tuple(bp.shape)}"
        )
    B, M, d_in = a.shape
    _, _, d_out = bp.shape
    if spec.h_in.numel() != d_in or spec.h_out.numel() != d_out:
        raise ValueError("TensorSketchSpec shapes do not match activations/backprops")

    # CountSketch along feature dims -> (B, M, m)
    s_bp = count_sketch_tokens(
        bp, spec=CountSketchSpec(h=spec.h_out, s=spec.s_out, m=spec.m)
    )
    s_a = count_sketch_tokens(
        a, spec=CountSketchSpec(h=spec.h_in, s=spec.s_in, m=spec.m)
    )

    # FFT-based convolution, then sum over tokens (M)
    f1 = torch.fft.rfft(s_bp, n=spec.m, dim=-1)
    f2 = torch.fft.rfft(s_a, n=spec.m, dim=-1)
    f_prod_sum = (f1 * f2).sum(dim=1)  # (B, F)
    sketch = torch.fft.irfft(f_prod_sum, n=spec.m, dim=-1)  # (B, m)
    return sketch


def tensor_sketch_embedding_weight(
    *,
    token_ids: torch.Tensor,
    backprops: torch.Tensor,
    spec: TensorSketchSpec,
) -> torch.Tensor:
    """
    TensorSketch for an Embedding weight gradient.

    token_ids: (B, ...) integer ids
    backprops: (B, ..., d_emb)

    Treats the embedding gradient as a sum over tokens of one_hot(token) ⊗ backprop.

    Returns: (B, m)
    """
    if token_ids.dim() < 2:
        raise ValueError(
            f"Expected token_ids with dim>=2 (B,...), got {tuple(token_ids.shape)}"
        )
    # Flatten tokens to (B, M) and backprops to (B, M, d_emb)
    B = token_ids.shape[0]
    ids = token_ids.reshape(B, -1)
    bp = _flatten_tokens(backprops)
    if bp.shape[0] != B or bp.shape[1] != ids.shape[1]:
        raise ValueError(
            f"token_ids/backprops mismatch: ids={tuple(ids.shape)} bp={tuple(bp.shape)}"
        )
    _, M = ids.shape
    d_emb = bp.shape[-1]
    if spec.h_in.numel() != d_emb:
        raise ValueError("Expected spec.h_in to match embedding_dim")
    # spec.h_out is indexed by token id; its length should be vocab size.

    # Build sketched one-hot counts for token ids: (B, M, m)
    if ids.dtype not in (
        torch.int64,
        torch.int32,
        torch.int16,
        torch.int8,
        torch.uint8,
    ):
        ids = ids.to(torch.int64)
    h_ids = spec.h_out[ids]  # (B, M)
    s_ids = spec.s_out[ids]  # (B, M)
    s1_act = torch.zeros(B, M, spec.m, device=bp.device, dtype=torch.float32)
    s1_act.scatter_add_(
        2,
        h_ids.unsqueeze(2),
        s_ids.unsqueeze(2).to(dtype=torch.float32),
    )

    # CountSketch backprops over embedding dim -> (B, M, m)
    s2_bp = count_sketch_tokens(
        bp, spec=CountSketchSpec(h=spec.h_in, s=spec.s_in, m=spec.m)
    )

    f1 = torch.fft.rfft(s1_act, n=spec.m, dim=-1)
    f2 = torch.fft.rfft(s2_bp, n=spec.m, dim=-1)
    f_prod_sum = (f1 * f2).sum(dim=1)
    sketch = torch.fft.irfft(f_prod_sum, n=spec.m, dim=-1)
    return sketch


def flatten_param_grad_sample(gs: torch.Tensor) -> torch.Tensor:
    """
    Flatten a per-sample gradient tensor to (B, D).
    """
    if gs.dim() < 2:
        raise ValueError(
            f"Expected grad_sample with dim>=2 (B,...), got {tuple(gs.shape)}"
        )
    return gs.flatten(1)


def count_sketch_from_grad_sample(
    *,
    gs: torch.Tensor,
    spec: CountSketchSpec,
) -> torch.Tensor:
    """CountSketch a per-sample gradient tensor gs to (B, m)."""
    return count_sketch_vector(flatten_param_grad_sample(gs), spec=spec)
