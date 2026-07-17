from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

from quant_layerwise.methods.zsic import _col_allreduce, _compute_entropy_synced


@dataclass(frozen=True)
class GPTQConfig:
    """GPTQ quantization configuration.

    target_rate: Target bits for quantization.
        maxq = int(2 ** (target_rate+1) - 1)
        - target_rate=4 -> maxq=31
        - target_rate=3 -> maxq=15
        - target_rate=2 -> maxq=7

    maxq: If specified, overrides the maxq computed from target_rate.
        target_rate is still used for initialization/logging.
    """

    target_rate: float = 4.0
    groupsize: int = -1
    blocksize: int = 128
    percdamp: float = 0.1  # GPTQ defaults to 0.1
    actorder: bool = False
    quiet: bool = False
    maxq: Optional[int] = None  # If set, overrides computed maxq

    def get_maxq(self) -> int:
        """Compute maxq from target_rate: maxq = int(2^(target_rate+1) - 1)"""
        if self.maxq is not None:
            return self.maxq
        return int(2 ** (self.target_rate + 1) - 1)


def _gptq_quantize(x: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, maxq: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize tensor x and return both quantized values and integer codes.

    Args:
        x: Input tensor to quantize
        scale: Per-channel scale factors
        zero: Per-channel zero points
        maxq: Maximum quantization level

    Returns:
        (quantized, q): Dequantized values and integer quantization codes
    """
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    quantized = scale * (q - zero)
    return quantized, q.int()


@torch.no_grad()
def compress_gptq(W: torch.Tensor, H: torch.Tensor, target_rate: float = 4.0,
                  groupsize: int = -1, blocksize: int = 128, percdamp: float = 0.1,  # GPTQ defaults to 0.1
                  actorder: bool = False, quiet: bool = False,
                  maxq_override: Optional[int] = None,
                  global_nrows: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """
    GPTQ quantization with cleaned-up implementation.

    This is a simplified version of GPTQ with:
    - sym=True (symmetric quantization)
    - mse=False (no MSE optimization for scale/zero)
    - perchannel=True (per-row quantization parameters)

    The core quantization loop is per-row (no cross-row dependencies), so it
    needs no multi-GPU sync.  Only entropy, loss, and rate overhead use
    global_nrows for correct normalization across ranks.

    Args:
        W: Weight matrix (a, n) - rows are output channels
        H: Hessian matrix (n, n)
        target_rate: Used to set maxq = int(2**target_rate - 1), i.e. number of bits
        groupsize: Column grouping for quantization parameters (-1 = no grouping)
        blocksize: Block size for processing columns (default 128)
        percdamp: Damping factor as percentage of mean diagonal (default 0.0)
        actorder: Whether to reorder columns by activation magnitude (default False)
        quiet: Suppress print output
        maxq_override: If specified, use this maxq value instead of computing from target_rate
        global_nrows: For multi-GPU ColumnParallel: total rows across all ranks

    Returns:
        (final_loss, final_rate, What, frame): Loss, rate, reconstructed weights, and locals dict
    """
    a, n = W.shape
    device = W.device
    dtype = W.dtype
    sync = global_nrows is not None
    a_eff = global_nrows if sync else a

    # maxq from target_rate (treat target_rate as bits), or use override
    if maxq_override is not None:
        maxq = maxq_override
    else:
        maxq = int(2 ** (target_rate + 1) - 1)

    if maxq % 2 == 0:
        maxq = maxq + 1 # maxq should be odd, so the symmetry would be preserved

    if not quiet:
        if maxq_override is not None:
            print(f"GPTQ: maxq={maxq} (override), target_rate={target_rate} bits (ignored)")
        else:
            print(f"GPTQ: target_rate={target_rate} bits -> maxq={maxq}")

    # Work in float64 for numerical stability
    W_orig = W
    H_orig = H
    W = W.clone()
    H = H.clone()
    W = W.double()
    H = H.double()

    # Handle dead columns (zero diagonal in H)
    dead = torch.diag(H) == 0

    H[dead, dead] = 1
    W[:, dead] = 0

    # Activation ordering (optional)
    perm = None
    invperm = None
    if actorder:
        perm = torch.argsort(torch.diag(H), descending=True)
        W = W[:, perm]
        H = H[perm][:, perm]
        invperm = torch.argsort(perm)

    # Add damping to Hessian diagonal
    damp = percdamp * torch.mean(torch.diag(H))
    diag_idx = torch.arange(n, device=device)

    H_damped = H.clone()
    H_damped[diag_idx, diag_idx] += damp

    # Compute inverse Hessian via Cholesky
    H_chol = torch.linalg.cholesky(H_damped)
    Hinv = torch.cholesky_inverse(H_chol)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)
    Hinv_diag = Hinv.diag().clone()  # Pre-extract diagonal to avoid GPU sync in loop

    # Initialize output tensors
    Q = torch.zeros_like(W)          # Quantized weights (dequantized values)
    Qint = torch.zeros_like(W, dtype=torch.int32)  # Integer quantization codes
    Losses = torch.zeros_like(W)     # Per-element loss

    # Storage for scale/zero per group
    scales_list = []
    zeros_list = []

    # Process columns in blocks
    for i1 in range(0, n, blocksize):
        i2 = min(i1 + blocksize, n)
        count = i2 - i1

        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Qint1 = torch.zeros_like(W1, dtype=torch.int32)
        Err1 = torch.zeros_like(W1)
        Losses1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        Hinv1_diag = Hinv_diag[i1:i2]  # Pre-extracted diagonal slice
        Hinv1_rows = [Hinv1[i, i:].clone() for i in range(count)]  # Pre-extract rows

        # Current scale/zero (will be updated per group)
        scale = None
        zero = None

        for i in range(count):
            col_idx = i1 + i
            w = W1[:, i]
            d = Hinv1_diag[i]  # Use pre-extracted diagonal

            # Update quantization parameters for new group
            if groupsize != -1:
                if col_idx % groupsize == 0:
                    # Find params for this group
                    group_end = min(col_idx + groupsize, n)
                    W_group = W[:, col_idx:group_end]

                    # Per-channel symmetric quantization (perchannel=True, sym=True)
                    # For each row, find max abs value
                    xmax = W_group.abs().max(dim=1)[0]
                    xmax = torch.clamp(xmax, min=1e-8)  # Avoid division by zero

                    # Symmetric: scale = 2 * xmax / maxq, zero = (maxq + 1) / 2
                    scale = (2 * xmax / maxq).unsqueeze(1)
                    zero = torch.full((a, 1), (maxq + 1) / 2, device=device, dtype=W.dtype)

                    scales_list.append(scale.squeeze(1))
                    zeros_list.append(zero.squeeze(1))
            else:
                # No grouping: compute scale/zero once at start
                if scale is None:
                    xmax = W.abs().max(dim=1)[0]
                    xmax = torch.clamp(xmax, min=1e-8)
                    scale = (2 * xmax / maxq).unsqueeze(1)
                    zero = torch.full((a, 1), (maxq + 1) / 2, device=device, dtype=W.dtype)
                    scales_list.append(scale.squeeze(1))
                    zeros_list.append(zero.squeeze(1))

            # Quantize column
            q_val, q_int = _gptq_quantize(w.unsqueeze(1), scale, zero, maxq)
            q_val = q_val.flatten()
            q_int = q_int.flatten()

            Q1[:, i] = q_val
            Qint1[:, i] = q_int
            Losses1[:, i] = (w - q_val) ** 2 / d ** 2

            # Error feedback
            err1 = (w - q_val) / d
            W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1_rows[i].unsqueeze(0))  # Use pre-extracted row
            Err1[:, i] = err1

        Q[:, i1:i2] = Q1
        Qint[:, i1:i2] = Qint1
        Losses[:, i1:i2] = Losses1 / 2

        # Propagate error to remaining columns
        W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

    # Undo activation ordering
    if actorder:
        Q = Q[:, invperm]
        Qint = Qint[:, invperm]

    What = Q.to(dtype)

    # GPTQ internal loss estimate (sum over all local elements, synced for multi-GPU)
    gptq_loss_local = torch.sum(Losses)
    if sync:
        _col_allreduce(gptq_loss_local.unsqueeze(0))
        gptq_loss_local = gptq_loss_local.squeeze(0)
    gptq_loss_estimate = gptq_loss_local.item()
    if not quiet:
        print(f"GPTQ internal loss estimate: {gptq_loss_estimate / (n * a_eff):.6g}")

    # Compute entropy-based rate (synced across ranks for multi-GPU)
    if sync:
        entropy = _compute_entropy_synced(Qint, a_eff * n)
    else:
        zsic_elts, zsic_counts = torch.unique(Qint.flatten(), return_counts=True)
        probs = zsic_counts.float() / Qint.numel()
        entropy = -torch.sum(probs * torch.log2(probs)).item()

    # Add overhead for scales (16 bits per group per row)
    if groupsize != -1:
        num_groups = (n + groupsize - 1) // groupsize
        scale_overhead = 16 * num_groups / n  # bits per entry for scales
    else:
        scale_overhead = 16 / n  # One scale per row

    final_rate = entropy + scale_overhead
    if not quiet:
        print(f"Qint: min={Qint.min()}, max={Qint.max()}, mean={Qint.float().mean():.3f}")
        print(f"Entropy: {entropy:.4f} bits, scale overhead: {scale_overhead:.4f} bits")
        print(f"Final rate: {final_rate:.4f} bits/entry")

    # Compute actual loss (trace is a partial sum over local rows, synced for multi-GPU)
    tr_loss = torch.trace((What - W_orig) @ H_orig @ (What - W_orig).T)
    tr_null = torch.trace(W_orig @ H_orig @ W_orig.T)
    if sync:
        tr_both = torch.stack([tr_loss, tr_null])
        _col_allreduce(tr_both)
        tr_loss, tr_null = tr_both[0], tr_both[1]
    final_loss = tr_loss / (n * a_eff)
    mse_null = tr_null / (n * a_eff)
    rel_mse = final_loss / mse_null

    if not quiet:
        print(f'Actual MSE loss: {final_loss:.6g}, relative MSE: {rel_mse:.6g}')

    # Build frame with useful info
    frame = {
        'Qint': Qint,
        'scales': torch.stack(scales_list) if scales_list else None,
        'zeros': torch.stack(zeros_list) if zeros_list else None,
        'maxq': maxq,
        'groupsize': groupsize,
        'blocksize': blocksize,
        'actorder': actorder,
        'entropy': float(entropy),
        'final_loss': float(final_loss.item()) if hasattr(final_loss, 'item') else float(final_loss),
        'relative_mse': float(rel_mse.item()) if hasattr(rel_mse, 'item') else float(rel_mse),
        'gptq_loss_estimate': gptq_loss_estimate,
    }

    return final_loss, final_rate, What, frame


# =============================================================================
# Wrapper for pipeline integration
# =============================================================================

@torch.no_grad()
def compress_gptq_wrapper(
    W: torch.Tensor,
    H: torch.Tensor,
    *,
    cfg: GPTQConfig,
    global_nrows: Optional[int] = None,
) -> Tuple[torch.Tensor, float, float, Dict[str, object]]:
    """
    GPTQ quantization - wrapper for pipeline.

    Args:
        W: Weight matrix (a, n) - rows are output channels
        H: Hessian matrix (n, n)
        cfg: GPTQConfig with quantization parameters
        global_nrows: For multi-GPU ColumnParallel: total rows across all ranks

    Returns:
        (What, loss, rate, frame): Reconstructed weights, loss, rate, and metadata dict
    """
    final_loss, final_rate, What, frame = compress_gptq(
        W, H,
        target_rate=cfg.target_rate,
        groupsize=cfg.groupsize,
        blocksize=cfg.blocksize,
        percdamp=cfg.percdamp,
        actorder=cfg.actorder,
        quiet=cfg.quiet,
        maxq_override=cfg.maxq,
        global_nrows=global_nrows,
    )

    # Convert frame for pipeline compatibility
    _rate = float(final_rate.item()) if hasattr(final_rate, 'item') else float(final_rate)
    _loss = float(final_loss.item()) if hasattr(final_loss, 'item') else float(final_loss)
    frame['rate_overhead'] = _rate - float(frame.get('entropy', 0))
    frame['loss'] = frame.get('final_loss')
    frame['percdamp'] = cfg.percdamp

    return What, _loss, _rate, frame


@torch.no_grad()
def dequantize_gptq(
    Qint: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    *,
    groupsize: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Dequantize a saved GPTQ artifact back to a float weight matrix.

    Args:
        Qint: (a,n) integer tensor (uint8/int32)
        scales: (num_groups, a)
        zeros: (num_groups, a)
        groupsize: -1 or group size along columns
        dtype: output dtype

    Returns:
        W_hat: (a,n)
    """
    a, n = Qint.shape

    q = Qint.to(torch.float32)
    scales_f = scales.to(torch.float32)
    zeros_f = zeros.to(torch.float32)

    if groupsize == -1:
        scale = scales_f[0].unsqueeze(1)
        zero = zeros_f[0].unsqueeze(1)
        out = scale * (q - zero)
        return out.to(dtype)

    num_groups = (n + groupsize - 1) // groupsize
    if scales_f.shape[0] != num_groups:
        raise ValueError(
            f"scales has {scales_f.shape[0]} groups but expected {num_groups} for n={n}, groupsize={groupsize}"
        )

    out = torch.empty((a, n), device=Qint.device, dtype=torch.float32)
    for g in range(num_groups):
        c0 = g * groupsize
        c1 = min(c0 + groupsize, n)
        scale = scales_f[g].unsqueeze(1)
        zero = zeros_f[g].unsqueeze(1)
        out[:, c0:c1] = scale * (q[:, c0:c1] - zero)
    return out.to(dtype)
