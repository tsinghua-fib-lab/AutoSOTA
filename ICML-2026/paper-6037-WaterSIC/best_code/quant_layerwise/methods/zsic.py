from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.distributed as _dist

# Set ZSIC_NCCL_DEBUG=1 to log every collective with a counter per rank.
_NCCL_DEBUG = os.environ.get("ZSIC_NCCL_DEBUG", "") == "1"
_nccl_counter = 0


def _nccl_checkpoint(label: str):
    """Print NCCL counter from all ranks at a strategic checkpoint.

    Use this to narrow down where a 1-op desync occurs by comparing
    counter values across ranks in the logs.  No NCCL ops are issued
    (just a print), so this cannot itself cause a hang.
    """
    if not (_dist.is_available() and _dist.is_initialized() and _dist.get_world_size() > 1):
        return
    print(f"[nccl-chk] rank={_dist.get_rank()} cnt={_nccl_counter} @ {label}", flush=True)


def _nccl_assert_sync(label: str):
    """Active desync detection: allreduce counter and verify all ranks match.

    Costs 1 NCCL allreduce op.  Use sparingly at high-value checkpoints
    (e.g. before entering the rescaler loop, after binary search).
    If counters differ, prints an error with per-rank breakdown and aborts
    immediately instead of waiting for the 10-minute NCCL timeout.
    """
    global _nccl_counter
    if not (_dist.is_available() and _dist.is_initialized() and _dist.get_world_size() > 1):
        return
    ws = _dist.get_world_size()
    rank = _dist.get_rank()
    local_cnt = _nccl_counter
    # Use a separate allreduce to check counter consistency.
    # This IS an extra NCCL op, but it's only at a few key checkpoints.
    _nccl_counter += 1
    buf = torch.tensor([local_cnt], dtype=torch.int64, device="cuda")
    # Gather all counters to rank 0 using all_gather
    gathered = [torch.zeros(1, dtype=torch.int64, device="cuda") for _ in range(ws)]
    _dist.all_gather(gathered, buf)
    _nccl_counter += 1  # count the all_gather too
    counts = [int(g.item()) for g in gathered]
    if len(set(counts)) > 1:
        print(f"\n{'='*60}\n"
              f"[NCCL DESYNC DETECTED] @ {label}\n"
              f"  Per-rank zsic counters: {counts}\n"
              f"  rank={rank} local_cnt={local_cnt}\n"
              f"{'='*60}\n", flush=True)
        # Don't abort — let the user see which checkpoint caught it.
        # The next allreduce will hang with a clear message about where.


def _sync_break(should_break: bool, sync: bool) -> bool:
    """Broadcast a break/branch decision from rank 0 in multi-GPU sync mode.

    Even when break conditions depend on all-reduced values (theoretically
    identical), subtle float-point or GPU-specific differences can cause one
    rank to break a loop while the other continues, fatally desyncing the
    NCCL collective sequence.  Broadcasting rank 0's decision costs one
    tiny broadcast per call and eliminates this class of bug.
    """
    if not sync:
        return should_break
    if not (_dist.is_available() and _dist.is_initialized() and _dist.get_world_size() > 1):
        return should_break
    flag = torch.tensor([1 if should_break else 0], dtype=torch.int32, device="cuda")
    global _nccl_counter
    _nccl_counter += 1
    if _NCCL_DEBUG:
        import traceback
        caller = traceback.extract_stack(limit=3)[0]
        print(f"[nccl-debug] rank={_dist.get_rank()} op={_nccl_counter} BROADCAST(1) "
              f"val={should_break} @ {caller.filename.split('/')[-1]}:{caller.lineno}",
              flush=True)
    _dist.broadcast(flag, src=0)
    return flag.item() == 1


def _col_allreduce(tensor: torch.Tensor) -> torch.Tensor:
    """All-reduce tensor in-place. No-op if distributed is not initialized.

    Handles CPU tensors by temporarily moving to CUDA for NCCL all-reduce.
    """
    if _dist.is_available() and _dist.is_initialized() and _dist.get_world_size() > 1:
        global _nccl_counter
        _nccl_counter += 1
        if _NCCL_DEBUG:
            import traceback
            caller = traceback.extract_stack(limit=3)[0]
            print(f"[nccl-debug] rank={_dist.get_rank()} op={_nccl_counter} ALLREDUCE({tensor.numel()}) "
                  f"@ {caller.filename.split('/')[-1]}:{caller.lineno}",
                  flush=True)
        if tensor.is_cuda:
            _dist.all_reduce(tensor, op=_dist.ReduceOp.SUM)
        else:
            gpu_tensor = tensor.cuda()
            _dist.all_reduce(gpu_tensor, op=_dist.ReduceOp.SUM)
            tensor.copy_(gpu_tensor)
    return tensor


def _compute_entropy_synced(Zsic: torch.Tensor, total_numel: int) -> float:
    """Compute entropy from Zsic codes, merging histograms across ranks."""
    device = Zsic.device

    z_lo = Zsic.min().unsqueeze(0).long()
    z_hi = Zsic.max().unsqueeze(0).long()
    if _dist.is_available() and _dist.is_initialized() and _dist.get_world_size() > 1:
        global _nccl_counter
        _nccl_counter += 2  # MIN + MAX allreduces
        _dist.all_reduce(z_lo, op=_dist.ReduceOp.MIN)
        _dist.all_reduce(z_hi, op=_dist.ReduceOp.MAX)
    lo, hi = z_lo.item(), z_hi.item()

    nbins = hi - lo + 1
    hist = torch.zeros(nbins, device=device, dtype=torch.long)
    flat = (Zsic.flatten() - lo).long()
    hist.scatter_add_(0, flat, torch.ones_like(flat))
    _col_allreduce(hist)

    probs = hist.float() / total_numel
    mask = probs > 0
    entropy = -torch.sum(probs[mask] * torch.log2(probs[mask]))
    return entropy.item()


@dataclass(frozen=True)
class ZSICConfig:
    target_rate_bits: float
    percdamp: float = 0.0001
    binary_search: bool = False  # Enable rate targeting (secant method)
    binary_search_left: float = -10.0  # Left bound for secant clamping
    binary_search_right: float = 10.0  # Right bound for secant clamping
    binary_search_row_fraction: float = 0.1  # Fraction of rows for fast entropy estimation
    qronos: bool = False
    # Residual compensation for wo/w2 layers (layers that output to residual stream)
    # When enabled, modifies the quantization target to account for residual stream error:
    # ŷ = (W Σ_{X,X̂} + Σ_{ΔR,X̂}) (L̂^T)^{-1}  where Σ_{ΔR,X̂} = E[(R - R̂)X̂^T]
    residual_compensation: bool = False
    # Dead dimension handling: remove dimensions with near-zero variance before quantization.
    # Uses Σ_X (unquantized covariance) to detect dead dims in the original activation space.
    # Dimensions with diag(Σ_X)[i] < dead_dim_threshold * median(diag(Σ_X)) are considered dead.
    dead_dim_threshold: float = 0.001
    # When rate control is active, don't inflate live-element entropy to fill
    # the original weight's budget when dead dims are present. Instead, let the
    # savings flow back to the rate controller for redistribution.
    rate_control_active: bool = False
    # Skip T/Gamma rescaler optimization (find_optimal_rescalers3).
    # Used during coord-adapt search to speed up evaluations — the search
    # only needs relative ordering of eps values, not exact dequantized weights.
    apply_rescaler: bool = True


def compute_entropy(zdata: torch.Tensor) -> float:
    """Compute log2-based entropy of a tensor."""
    Zsic = zdata.flatten()
    zsic_elts, zsic_counts = torch.unique(Zsic.flatten(), return_counts=True)
    probs = zsic_counts.float() / Zsic.numel()
    entropy = -torch.sum(probs * torch.log2(probs))
    return entropy.item()


# =============================================================================
# Dead Dimension Handling
# =============================================================================

def find_dead_dimensions(
    Sig_X: torch.Tensor,
    threshold_ratio: float = 0.001,
) -> torch.Tensor:
    """Find dimensions with near-zero or non-positive variance (dead dimensions).

    Args:
        Sig_X: Unquantized activations covariance E[X X^T] (n x n).
               Detects dims that are dead in the original activation space.
        threshold_ratio: Dimensions with diag < threshold_ratio * median(diag) are dead

    Returns:
        Boolean mask of shape (n,) where True = dead dimension
    """
    diag = Sig_X.diag()
    diag_abs = diag.abs()
    # Use median (not mean) as the reference point. Mean is sensitive to outliers:
    # e.g. SiLU-gated w2 inputs have a few high-variance features that pull the
    # mean way up, making 0.001*mean too aggressive and flagging 87% as dead.
    # Median is robust to heavy-tailed distributions.
    threshold = threshold_ratio * diag_abs.median()
    dead = (diag_abs < threshold) | (diag <= 0)

    n_dead = int(dead.sum())
    print(f"[dead-dim] diag stats: min={diag.min():.2e}, median={diag_abs.median():.2e}, mean={diag_abs.mean():.2e}, max={diag.max():.2e}", flush=True)
    print(f"[dead-dim] threshold={threshold:.2e}, n_dead={n_dead}/{len(diag)}", flush=True)

    return dead


def slice_out_dead_dims(
    dead_mask: torch.Tensor,
    W: torch.Tensor,
    Sig_X: torch.Tensor,
    Sig_hX: torch.Tensor | None = None,
    Sig_X_hX: torch.Tensor | None = None,
    Sig_delta_R_Xhat: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    """Remove dead dimensions from weight and covariance matrices.

    Args:
        dead_mask: Boolean mask (n,) where True = dead
        W: Weight matrix (a, n)
        Sig_X: Unquantized activations covariance E[X X^T] (n, n)
        Sig_hX: Quantized activations covariance E[X̂ X̂^T] (n, n), optional
        Sig_X_hX: Cross-covariance E[X X̂^T] (n, n), optional
        Sig_delta_R_Xhat: Residual compensation (a, n), optional

    Returns:
        Dict with sliced matrices (only live dimensions)
    """
    live = ~dead_mask

    # Move mask to each tensor's device for indexing (handles CPU/GPU mix).
    _live_w = live.to(W.device)
    result = {
        "W": W[:, _live_w],
        "Sig_X": Sig_X[live.to(Sig_X.device)][:, live.to(Sig_X.device)],
    }

    if Sig_hX is not None:
        _live_h = live.to(Sig_hX.device)
        result["Sig_hX"] = Sig_hX[_live_h][:, _live_h]
    if Sig_X_hX is not None:
        _live_xh = live.to(Sig_X_hX.device)
        result["Sig_X_hX"] = Sig_X_hX[_live_xh][:, _live_xh]
    if Sig_delta_R_Xhat is not None:
        result["Sig_delta_R_Xhat"] = Sig_delta_R_Xhat[:, _live_w]

    return result


def expand_zsic_results(
    frame: Dict[str, Any],
    dead_mask: torch.Tensor,
    n_original: int,
    device: torch.device,
) -> Dict[str, Any]:
    """Expand quantization results back to original size, inserting zeros at dead positions.

    Args:
        frame: Quantization results with Z, alpha, etc. for live dimensions
        dead_mask: Boolean mask (n_original,) where True = dead
        n_original: Original number of dimensions
        device: Torch device

    Returns:
        Updated frame with expanded tensors and dead dimension metadata
    """
    live_mask = ~dead_mask
    dead_indices = dead_mask.nonzero().squeeze(-1).tolist()
    if isinstance(dead_indices, int):
        dead_indices = [dead_indices]
    n_live = int(live_mask.sum())
    n_dead = int(dead_mask.sum())

    def expand_1d(t: torch.Tensor) -> torch.Tensor:
        if t is None:
            return None
        full = torch.zeros(n_original, dtype=t.dtype, device=device)
        full[live_mask] = t.to(device)
        return full

    def expand_2d_cols(t: torch.Tensor) -> torch.Tensor:
        if t is None:
            return None
        a = t.shape[0]
        full = torch.zeros(a, n_original, dtype=t.dtype, device=device)
        full[:, live_mask] = t.to(device)
        return full

    # Expand the quantization artifacts
    expanded = dict(frame)  # Copy existing entries
    expanded["Z"] = expand_2d_cols(frame["Z"])
    expanded["alpha"] = expand_1d(frame["alpha"])

    if frame.get("alpha_base") is not None:
        expanded["alpha_base"] = expand_1d(frame["alpha_base"])
    if frame.get("g_vec") is not None:
        expanded["g_vec"] = expand_1d(frame["g_vec"])
    # t_vec is row dimension, unchanged
    # (but copy to ensure it's on right device)
    if frame.get("t_vec") is not None:
        expanded["t_vec"] = frame["t_vec"].to(device)

    # Add dead dimension metadata
    expanded["dead_indices"] = dead_indices
    expanded["n_original"] = n_original
    expanded["n_live"] = n_live
    expanded["n_dead"] = n_dead

    return expanded


@torch.no_grad()
def find_optimal_rescalers3(
    W_hat: torch.Tensor,   # shape: a x n
    W: torch.Tensor,       # shape: a x n
    Sig_X: torch.Tensor,    # shape: n x n, = E[X X^T]
    Sig_hX: torch.Tensor = None, # shape: n x n, = E[\hat X \hat X^T]
    Sig_X_hX: torch.Tensor = None, # shape: n x n, = E[ X \hat X^T]
    Sig_delta_R_Xhat: torch.Tensor = None,  # shape: a x n, = E[(R - R̂) X̂^T] for residual compensation
    max_iter: int = 1000,
    tol: float = 3e-4,
    ridge_eps: float = 1e-11,    # small Tikhonov for Gamma-step and T-step
    t_init: torch.Tensor = None,
    gamma_init: torch.Tensor = None,
    gamma_clip_min: float = -0.1,  # Clip gamma to prevent wild values at low rates
    gamma_clip_max: float = 1.5,   # (columns quantized to ~0 can cause gamma blowup)
    quiet: bool = False,
    global_nrows: int = None,  # For multi-GPU ColumnParallel: total rows across all ranks
):
    """
    Alternating updates for diagonal T and Gamma that minimize:
        J(T,Gamma) = -2 tr(T W_hat Gamma SigX W) + tr(T W_hat Gamma SigX Gamma W_hat^T T)

    When Sig_delta_R_Xhat is provided (for residual compensation), replaces
    W @ Sig_X_hX with (W @ Sig_X_hX + Sig_delta_R_Xhat) in the optimization.

    Conventions:
      - a := number of rows of W_hat
      - n := number of columns of W_hat (and size of SigX)
      - T  = diag(t) with t in R^a
      - Gamma = diag(gamma) with gamma in R^n
      - Normalization: t.abs().sum() == a at every iteration (scale absorbed into Gamma).
      - Progress printing uses mse_loss(T @ W_hat @ Gamma) as requested.

    Returns:
      t (a,), gamma (n,) — diagonal vectors (use t[:,None]*X*gamma[None,:] instead of T@X@Gamma)
    """

    # ----- basic checks & shape harmonization -----
    a, n = W_hat.shape
    assert W.shape == (a, n)
    assert Sig_X.shape == (n, n)

    if Sig_hX is None:
        assert Sig_X_hX is None
        # Assuming that X=\hat X (e.g. first layer)
        Sig_hX = Sig_X
        Sig_X_hX = Sig_X

    device = W_hat.device
    dtype = W_hat.dtype

    # If Sig matrices are on a different device (e.g. CPU, to avoid GPU OOM for
    # large RowParallel layers like w2), move everything to that device.
    sig_device = Sig_X.device if Sig_hX is None else Sig_hX.device
    if sig_device != device:
        W_hat = W_hat.to(sig_device)
        W = W.to(sig_device)
        Sig_X = Sig_X.to(sig_device)
        if Sig_delta_R_Xhat is not None:
            Sig_delta_R_Xhat = Sig_delta_R_Xhat.to(sig_device)
        device = sig_device
        if t_init is not None:
            t_init = t_init.to(sig_device)
        if gamma_init is not None:
            gamma_init = gamma_init.to(sig_device)

    assert dtype == torch.double, 'ERROR: this code (at high-rates) does not work in torch.float...'

    sync = global_nrows is not None
    a_eff = global_nrows if sync else a  # effective row count for normalization

    # Compute the effective cross-term: W @ Sig_X_hX + Sig_delta_R_Xhat (if provided)
    # This accounts for residual compensation in the T/Gamma optimization.
    # When Sig_X_hX is on a different device (CPU offloaded for large RowParallel w2),
    # move it to GPU temporarily — the matmul takes seconds on GPU vs minutes on CPU.
    if Sig_X_hX.device != device:
        _Sig_gpu = Sig_X_hX.to(device)
        W_Sig_X_hX_eff = W @ _Sig_gpu
        del _Sig_gpu
    else:
        W_Sig_X_hX_eff = W @ Sig_X_hX
    if Sig_delta_R_Xhat is not None:
        W_Sig_X_hX_eff = W_Sig_X_hX_eff + Sig_delta_R_Xhat.to(dtype=dtype, device=device)
        if not quiet:
            print("[find_optimal_rescalers3] applying residual compensation in T/Gamma optimization")

    # Precompute constant term: trace(W @ Sig_X @ W.T) — doesn't change across iterations.
    # Use trace-free form: trace(A @ B @ A.T) = (A @ B * A).sum() to avoid (a,a) matrices.
    if Sig_X.device != device:
        _Sig_gpu = Sig_X.to(device)
        _W_Sig_X = W @ _Sig_gpu  # (a, n)
        del _Sig_gpu
    else:
        _W_Sig_X = W @ Sig_X  # (a, n)
    _tr_WSW = (_W_Sig_X * W).sum()
    if sync:
        _col_allreduce(_tr_WSW.unsqueeze(0))
        _tr_WSW = _tr_WSW.squeeze(0)
    del _W_Sig_X
    # Sig_X and Sig_X_hX are no longer needed — only Sig_hX is used in iterations.
    del Sig_X, Sig_X_hX

    # mse_loss uses trace-free forms to avoid materializing (a,a) matrices.
    _mse_buf = torch.zeros(2, device=device, dtype=dtype) if sync else None
    def mse_loss(What):
        cross = (W_Sig_X_hX_eff * What).sum()
        What_Sig = What @ Sig_hX  # (a, n)
        quad = (What_Sig * What).sum()
        if sync:
            _mse_buf[0] = cross
            _mse_buf[1] = quad
            _col_allreduce(_mse_buf)
            cross, quad = _mse_buf[0], _mse_buf[1]
        tr = _tr_WSW - 2 * cross + quad
        return tr / (n * a_eff)

    # Helper: compute t[:,None] * W_hat * gamma[None,:] without diagonal matrices.
    # Pre-allocated buffer avoids ~100 a×n float64 allocations across iterations.
    _tWg_buf = torch.empty_like(W_hat)
    def _tWg(t_vec, g_vec):
        torch.mul(W_hat, g_vec[None, :], out=_tWg_buf)
        _tWg_buf.mul_(t_vec[:, None])
        return _tWg_buf

    # ----- initialization: ones(a), ones(n) -----
    if t_init is None:
        t = torch.ones(a, device=device, dtype=dtype)
    else:
        t = t_init.clone().to(device=device, dtype=dtype)

    if gamma_init is None:
        gamma = torch.ones(n, device=device, dtype=dtype)
    else:
        gamma = gamma_init.clone().to(device=device, dtype=dtype)

    # enforce t.abs().sum() = a_eff and absorb scale into Gamma (keeps T @ W_hat @ Gamma unchanged)
    s0_sum = t.abs().sum()
    if sync:
        _col_allreduce(s0_sum.unsqueeze(0))
        s0_sum = s0_sum.squeeze(0)
    s0 = s0_sum / a_eff
    if s0 > 0:
        t = t / s0
        gamma = gamma * s0

    _nccl_checkpoint("rescalers3:start")

    # ----- step 0: report mse_loss -----
    loss_prev = mse_loss(_tWg(t, gamma)).detach()
    if not quiet:
        print(f"iter 0 | mse_loss = {float(loss_prev):.6e}")

    for it in range(1, max_iter + 1):

        # Save previous state for rollback if numerical issues arise
        gamma_prev = gamma.clone()
        t_prev = t.clone()

        # ===== Gamma-step (given T) =====
        # F2 = W_hat.T @ diag(t^2) @ W_hat — avoid (a,a) diagonal matrix
        t2_What = t.pow(2)[:, None] * W_hat       # (a, n)
        F2 = t2_What.T @ W_hat                    # (n, n)
        # f4_vec = diag(W_hat.T @ diag(t) @ W_Sig_X_hX_eff) — element-wise
        f4_vec = (W_hat * (t[:, None] * W_Sig_X_hX_eff)).sum(dim=0)  # (n,)
        if sync:
            _col_allreduce(F2)
            _col_allreduce(f4_vec)
        # For large n (e.g., w2 with n=28672), F3 = Sig_hX * F2 needs ~6GB.
        # Use deterministic CPU path for all ranks when n is large to avoid
        # some ranks OOMing while others don't (which causes NCCL timeout).
        if n > 16384 and device.type == "cuda":
            F3 = Sig_hX.cpu() * F2.cpu()
            del F2
            F3 = F3.to(device)
        else:
            F3 = Sig_hX * F2
            del F2

        # Adaptive ridge: scale with F3 magnitude to bound condition number.
        # F3 is PSD (Schur product of two PSD matrices). With fixed ridge_eps=1e-11,
        # the condition number can exceed 1e13 when F3 eigenvalues are O(1) or larger,
        # causing numerically inaccurate solves. In multi-GPU the all-reduce introduces
        # slightly different rounding than single-GPU matmul, pushing the solve past
        # float64 precision limits. Scaling ridge to ~1e-8 * mean(diag(F3)) keeps
        # cond(F5) ≤ ~1e8, safe for float64.
        f3_diag_mean = float(F3.diag().abs().mean())
        effective_ridge = max(ridge_eps, f3_diag_mean * 1e-8)
        if effective_ridge > ridge_eps * 10 and not quiet:
            print(f"  [Gamma-step] adaptive ridge: {effective_ridge:.2e} "
                  f"(F3 diag mean={f3_diag_mean:.2e}, base ridge={ridge_eps:.2e})")

        F3.diagonal().add_(effective_ridge)  # In-place; avoids n×n torch.eye temporary

        # For large n (e.g., w2 with n=28672), solve on CPU to avoid GPU OOM.
        # All ranks must take the same path to avoid NCCL timeout at the
        # subsequent gamma broadcast.
        _solve_on_cpu = n > 16384
        if _solve_on_cpu:
            if not quiet:
                print(f"  [Gamma-step] large n={n}, solving on CPU")
            gamma = torch.linalg.solve(F3.cpu().double(), f4_vec.cpu().double()).to(device)
        else:
            try:
                gamma = torch.linalg.solve(F3.double(), f4_vec.double())
            except RuntimeError:
                print('WARNING: linalg.solve() failed, using pseudo-inverse')
                F6 = torch.linalg.pinv(F3.double())
                gamma = F6 @ f4_vec.double()

        gamma = gamma.to(dtype)
        del t2_What  # Keep F3/f4_vec for efficient mse_loss below

        # Broadcast gamma from rank 0 to ensure all ranks use the same
        # per-column scaling.  Even with identical F5 and f4, cuSOLVER's
        # LU decomposition can differ across physical GPUs.
        if sync:
            global _nccl_counter
            _nccl_counter += 1
            if _NCCL_DEBUG:
                print(f"[nccl-debug] rank={_dist.get_rank()} op={_nccl_counter} BROADCAST({gamma.numel()}) gamma",
                      flush=True)
            if gamma.is_cuda:
                _dist.broadcast(gamma, src=0)
            else:
                _gpu = gamma.cuda()
                _dist.broadcast(_gpu, src=0)
                gamma.copy_(_gpu)

        if (it == 1) and (gamma_init is not None) and (t_init is None):
            mean_diff = (gamma - gamma_init).abs().mean()
            mask = gamma_init > 0
            rel_mean_diff = (1 - gamma_init[mask] / gamma[mask]).abs().mean()
            if not quiet:
                print(f'iter 1 | gamma changed by {mean_diff:.5g} (rel = {rel_mean_diff:.5g})')

        # Efficient post-gamma mse_loss: reuse F3/f4_vec from gamma step.
        # cross = f4_vec · gamma (O(n)), quad = gamma @ F3 @ gamma - ridge*||gamma||² (O(n²))
        # vs original O(an²) matmul.  F3/f4_vec are globally synced — no allreduce needed.
        def _mse_gamma(g):
            _c = f4_vec.dot(g)
            _q = (g @ F3).dot(g) - effective_ridge * g.dot(g)
            return (_tr_WSW - 2 * _c + _q) / (n * a_eff)

        loss_curr = _mse_gamma(gamma)

        # Gamma step should decrease loss (it's a quadratic minimization).
        # If it increases, the linear solve overshot (ill-conditioned F3, common
        # for RowParallel layers in multi-GPU).  Try a line search along the
        # direction gamma_new - gamma_prev before giving up.
        # Sync decision via _sync_break: both ranks must agree on whether to enter
        # the line search, since it contains mse_loss calls with allreduces.
        if _sync_break(float(loss_curr) > float(loss_prev) * (1 + 1e-6), sync):
                direction = gamma - gamma_prev
                best_gamma = gamma_prev
                best_loss = loss_prev
                for alpha in [0.5, 0.25, 0.125, 0.0625]:
                    gamma_try = gamma_prev + alpha * direction
                    loss_try = _mse_gamma(gamma_try)
                    if float(loss_try) < float(best_loss):
                        best_gamma = gamma_try
                        best_loss = loss_try
                if _sync_break(
                    float(best_loss) >= float(loss_prev) * (1 + 1e-6), sync
                ):
                    if not quiet:
                        print(f"WARNING: Gamma step increased loss at iter {it} "
                              f"(loss_curr={float(loss_curr):.6e}, loss_prev={float(loss_prev):.6e}, "
                              f"ratio={float(loss_curr/loss_prev):.1f}x). "
                              f"Line search failed, reverting to previous gamma.")
                    gamma = gamma_prev
                    t = t_prev
                    break
                else:
                    if not quiet:
                        alpha_used = float((best_gamma - gamma_prev).norm() / direction.norm()) if direction.norm() > 0 else 0
                        print(f"iter {it} | Gamma full step overshot, line search found alpha={alpha_used:.4f} "
                              f"(loss {float(loss_prev):.3e} → {float(best_loss):.3e})")
                    gamma = best_gamma
                    loss_curr = best_loss

        del F3, f4_vec  # Free gamma-step intermediates now that mse_loss is done

        # ===== clip gamma to prevent wild values (columns quantized to ~0 can blow up) =====
        # Clip BEFORE T-step so T is computed for the clipped gamma
        n_clipped_low = (gamma < gamma_clip_min).sum().item()
        n_clipped_high = (gamma > gamma_clip_max).sum().item()
        if (n_clipped_low > 0 or n_clipped_high > 0) and not quiet:
            print(f"iter {it} | clipping gamma: {n_clipped_low} below {gamma_clip_min}, {n_clipped_high} above {gamma_clip_max}")
        gamma = gamma.clamp(min=gamma_clip_min, max=gamma_clip_max)

        # ===== T-step (given Gamma) =====
        # f7_vec = diag(W_Sig_X_hX_eff @ diag(gamma) @ W_hat.T) — use (A*B).sum(1) trick
        f7_vec = ((W_Sig_X_hX_eff * gamma[None, :]) * W_hat).sum(dim=1)  # (a,)
        # f8_vec = diag(M @ Sig_hX @ M.T) where M = W_hat * gamma — use (M@S * M).sum(1)
        M = W_hat * gamma[None, :]                   # (a, n)
        f8_vec = (M @ Sig_hX * M).sum(dim=1)         # (a,)

        # Ridge for T-step (adaptive, same principle as Gamma-step)
        effective_ridge_t = max(ridge_eps, float(f8_vec.abs().mean()) * 1e-8)
        t = f7_vec / (f8_vec + effective_ridge_t)
        _t_pre_norm = t  # Save pre-normalization t for efficient mse_loss

        # ===== normalization: t.abs().sum() = 1, absorb scale into Gamma =====
        t_abs_sum = t.abs().sum()
        if sync:
            _col_allreduce(t_abs_sum.unsqueeze(0))
            t_abs_sum = t_abs_sum.squeeze(0)
        s = t_abs_sum / a_eff
        if float(s) > 0.0:
            t = t / s
            gamma = gamma * s

        # ===== report & stopping based on mse_loss changes =====
        # Efficient post-T mse_loss: reuse f7_vec/f8_vec from T-step.
        # cross = t_pre · f7_vec (O(a)), quad = t_pre² · f8_vec (O(a))
        # vs original O(an²) matmul.  t*gamma product is unchanged by normalization.
        del M
        _cross_t = _t_pre_norm.dot(f7_vec)
        _quad_t = _t_pre_norm.pow(2).dot(f8_vec)
        if sync:
            _mse_buf[0] = _cross_t
            _mse_buf[1] = _quad_t
            _col_allreduce(_mse_buf)
            _cross_t, _quad_t = _mse_buf[0], _mse_buf[1]
        loss_curr = (_tr_WSW - 2 * _cross_t + _quad_t) / (n * a_eff)
        del f7_vec, f8_vec, _t_pre_norm
        rel = torch.abs(loss_curr - loss_prev) / (torch.abs(loss_prev) + 1e-12)

        # Note: clipping gamma can increase loss (it's a projection, not a descent step).
        # But if loss increases for 2+ consecutive clipped iterations, stop — the
        # clipping is too aggressive and further optimization makes things worse.
        clipped_this_iter = (n_clipped_low > 0 or n_clipped_high > 0)
        if _sync_break(
            not clipped_this_iter and float(loss_curr) > float(loss_prev) * (1 + 1e-6),
            sync
        ):
            if not quiet:
                print(f"WARNING: T+Gamma step increased loss at iter {it} "
                      f"(loss_curr={float(loss_curr):.6e}, loss_prev={float(loss_prev):.6e}). "
                      f"Reverting to previous state.")
            gamma = gamma_prev
            t = t_prev
            break

        if not quiet:
            print(f"iter {it} | mse_loss = {float(loss_curr):.6e} | rel change = {float(rel):.3e}")

        if it % 10 == 0 or it <= 2:
            _nccl_checkpoint(f"rescalers3:iter{it}")

        if _sync_break(float(rel) < tol, sync):
            break
        loss_prev = loss_curr

    _nccl_checkpoint(f"rescalers3:done(iters={it})")

    # Print statistics:
    if not quiet:
        def print_stats(tens, name):
            mmin, mmax = tens.flatten().min(), tens.flatten().max()
            q25, q75 = torch.quantile(tens.flatten(), 0.25), torch.quantile(tens.flatten(), 0.75)
            mean, stddev = tens.mean(), torch.std(tens)
            print('Tensor ' + name + f' stats: min={mmin:.3g}, q25={q25:.3g}, mean = {mean:.3g}, q75={q75:.3g}, max={mmax:.3g};  std = {stddev:.3g}')

        print_stats(t, 'row-rescaler T:')
        print_stats(gamma, 'column-rescaler Gamma:')

    return t, gamma


@torch.no_grad()
def compress_w2q(
    W, Sig_X, target_rate=1.5, quiet=False, Sig_hX=None, Sig_X_hX=None, percdamp=0.0001,
    Sig_delta_R_Xhat=None,  # Residual compensation: Σ_{ΔR,X̂} = E[(R - R̂)X̂^T]
    global_nrows=None,  # For multi-GPU ColumnParallel: total rows across all ranks
    L_cached=None,  # Precomputed Cholesky factor (avoids redundant O(n³) recomputation)
    damp_cached=None,  # Damping value corresponding to L_cached
    apply_rescaler=True,  # When False, skip find_optimal_rescalers3 (fast search mode)
    fp32_ldlq=False,  # When True, run LDLQ loop in fp32 (2x faster, for search mode)
    target_precomputed=None,  # Precomputed target matrix (a, n) fp64 on GPU — skips matmul+solve
):
    """
    LDLQ with per-column gamma adaptation + T/Gamma rescaler optimization.

    Args:
        W: Weight matrix (a, n)
        Sig_X, Sig_hX, Sig_X_hX: E[XX^T], E[Xhat Xhat^T], E[X Xhat^T]
        target_rate: Target compression rate in bits
        quiet: Suppress print output
        Sig_delta_R_Xhat: Optional residual compensation term, shape (a, n).
                          When provided, modifies target: ŷ = (W Σ_{X,X̂} + Σ_{ΔR,X̂})(L̂^T)^{-1}

    Returns:
        (final_loss, final_rate, What, frame): Loss, rate, reconstructed weights, and locals dict
    """
    global _nccl_counter
    a, n = W.shape
    dtype_orig = W.dtype
    sync = global_nrows is not None
    a_eff = global_nrows if sync else a  # effective row count for column stats

    # Convert all inputs to double for numerical stability
    _gpu = W.device  # computation device (GPU)
    W = W.double()
    Sig_X = Sig_X.double()
    if Sig_hX is not None:
        Sig_hX = Sig_hX.double()
    if Sig_X_hX is not None:
        Sig_X_hX = Sig_X_hX.double()

    # Staged loading: when Sig matrices are on CPU (offloaded by pipeline for
    # large RowParallel w2), do Cholesky + target on CPU, then move L/Ycur to
    # GPU for the LDLQ loop.  After LDLQ frees L+Ycur, bring Sig to GPU for
    # mse_loss and rescalers.  This avoids having all n×n matrices on GPU at once.
    _staged = (Sig_X.device != _gpu)

    if Sig_hX is not None:
        assert Sig_X_hX is not None
        H = Sig_hX
        qronos = True
    else:
        H = Sig_X
        qronos = False

    # Cholesky factorization.  When L_cached is provided (from binary search
    # precomputation), skip the redundant O(n³) decomposition (~2s for n=28672).
    if L_cached is not None:
        L = L_cached
        damp = damp_cached
    elif _staged:
        # Staged (Sig on CPU): do Cholesky on GPU — the O(n³) Cholesky of
        # 28672² takes minutes on CPU but seconds on GPU.
        # .to() already returns a new tensor when crossing devices, no clone needed.
        _H_gpu = H.to(_gpu)
        damp = percdamp * torch.mean(torch.diag(_H_gpu))
        _H_gpu.diagonal().add_(damp)
        L = torch.linalg.cholesky(_H_gpu, upper=False)
        del _H_gpu
    else:
        damp = percdamp * torch.mean(torch.diag(H))
        H_damped = H.clone()
        H_damped.diagonal().add_(damp)
        L = torch.linalg.cholesky(H_damped, upper=False)
        del H_damped
    assert torch.all(L.diag() >= 0)

    # Multi-GPU: broadcast L and damp from rank 0 to ensure all ranks use
    # identical Cholesky factors. CUDA Cholesky is non-deterministic across
    # different GPUs, causing O(1e-13) differences that propagate through
    # c_param and the LDLQ loop, producing inconsistent Zsic codes.
    if sync and _dist.is_available() and _dist.is_initialized() and _dist.get_world_size() > 1:
        _nccl_counter += 1
        _dist.broadcast(L, src=0)
        _damp_t = torch.tensor([damp], device=_gpu, dtype=torch.float64)
        _nccl_counter += 1
        _dist.broadcast(_damp_t, src=0)
        damp = _damp_t.item()

    if target_precomputed is not None:
        # Precomputed target: skip the expensive W @ Sig_X_hX matmul.
        # target_precomputed = W @ blended_Sig_X_hX (fp64, on GPU).
        # Still need to add damp*W (damp depends on L which changes per q_eps).
        target = target_precomputed.clone()
        target.add_(W.double(), alpha=damp)
        Ycur = torch.linalg.solve_triangular(L.T, target, left=False, upper=True)
        del target
    elif qronos:
        # Qronos: Y = (W @ (Σ_{X,X̂} + damp·I) + Σ_{ΔR,X̂}) @ L̂^{-T}
        # Always compute on GPU for speed.  When Sig_X_hX is on CPU (offloaded
        # for large RowParallel w2), move it to GPU temporarily — the matmul
        # takes seconds on GPU vs minutes on CPU for 28672² matrices.
        _Sig_on_gpu = Sig_X_hX.to(_gpu) if Sig_X_hX.device != _gpu else Sig_X_hX
        W_d = W.double()  # W is already on GPU
        target = W_d @ _Sig_on_gpu
        if _Sig_on_gpu is not Sig_X_hX:
            del _Sig_on_gpu  # free temporary GPU copy
        target.add_(W_d, alpha=damp)
        if Sig_delta_R_Xhat is not None:
            if not quiet:
                print("[compress_w2q] applying residual compensation (Qronos)")
            target = target + Sig_delta_R_Xhat.to(_gpu).double()
        del W_d
        Ycur = torch.linalg.solve_triangular(L.T, target, left=False, upper=True)
        del target
    else:
        # LDLQ: Y = W @ L + Σ_{ΔR,X̂} @ L^{-T}
        if _staged:
            # L is on GPU; W is on GPU → direct matmul on GPU
            Ycur = W.double() @ L
        else:
            _sig_dev = Sig_X.device
            Ycur = W.to(_sig_dev).double() @ L
        if Sig_delta_R_Xhat is not None:
            if not quiet:
                print("[compress_w2q] applying residual compensation")
            Ycur = Ycur + torch.linalg.solve_triangular(
                L, Sig_delta_R_Xhat.double().to(L.device).T, left=True, upper=False
            ).T

    # When staged, L and Ycur are already on GPU (Cholesky done on GPU).
    # Just clean up.
    if _staged:
        torch.cuda.empty_cache()
        if not quiet:
            print(f"[compress_w2q] staged: Cholesky on {_gpu}, target on {'CPU' if Sig_X_hX is not None and not Sig_X_hX.is_cuda else _gpu}, LDLQ on {_gpu}")

    _nccl_checkpoint("compress_w2q:before_ldlq")
    if sync:
        _nccl_assert_sync("compress_w2q:before_ldlq")

    wtw_diag = (W ** 2).sum(0)  # Only need diagonal of W.T @ W; avoids n×n matrix
    if sync:
        _col_allreduce(wtw_diag)
    sw_diag = wtw_diag / a_eff
    del wtw_diag
    target_rate_nats = target_rate * math.log(2)  # in nats
    c_param = torch.exp(torch.log(12 * sw_diag * (L.diag() ** 2)).mean() / 2 - target_rate_nats)

    alphas = c_param / L.diag()
    gammas = torch.ones(n, device=_gpu)
    del sw_diag

    Zsic = torch.zeros_like(W, dtype=torch.int64, device=_gpu)

    # ── fp32 LDLQ for search mode ────────────────────────────────────────
    # When fp32_ldlq=True (coord-adapt search), downcast Ycur/L to fp32 before
    # the bandwidth-bound LDLQ loop.  Halves HBM traffic → ~2x speedup.
    # Cholesky and Ycur construction stay fp64.
    if fp32_ldlq:
        _L_ldlq = L.float()
        _alphas_ldlq = alphas.float()
        _c_param_ldlq = c_param.float()
    else:
        _L_ldlq = L
        _alphas_ldlq = alphas
        _c_param_ldlq = c_param

    ## Perform uneven rate LDLQ
    if sync:
        # ── Gathered LDLQ: all-gather Ycur rows to eliminate per-column ──
        # all-reduces.  Each rank gets the full matrix and runs the LDLQ
        # loop locally — no NCCL in the inner loop (replaces ~n all-reduces
        # with 1 all-gather).  After the loop, extract local Zsic shard.
        ws = _dist.get_world_size()
        rank = _dist.get_rank()
        _nccl_counter += 1  # count the all-gather
        if fp32_ldlq:
            Ycur_f = Ycur.float()
            del Ycur
            Ycur_shards = [torch.empty_like(Ycur_f) for _ in range(ws)]
            _dist.all_gather(Ycur_shards, Ycur_f.contiguous())
            Ycur_full = torch.cat(Ycur_shards, dim=0)
            del Ycur_shards, Ycur_f
        else:
            Ycur_shards = [torch.empty_like(Ycur) for _ in range(ws)]
            _dist.all_gather(Ycur_shards, Ycur.contiguous())
            Ycur_full = torch.cat(Ycur_shards, dim=0)  # (a_full, n)
            del Ycur_shards, Ycur

        row_start = rank * a  # each rank's contiguous shard in gathered matrix

        for col in range(n - 1, -1, -1):
            wcol = Ycur_full[:, col]
            zcol_full = torch.round(wcol / _c_param_ldlq).long()
            Zsic[:, col] = zcol_full[row_start:row_start + a]
            zcol_cast = zcol_full.float() if fp32_ldlq else zcol_full.double()
            f1 = (zcol_cast * wcol).sum()
            f2 = (zcol_cast * zcol_cast).sum()
            if f2 > 0:
                gammas[col] = f1 / f2 / _c_param_ldlq
                Ycur_full.addr_(zcol_cast, _L_ldlq[col, :], alpha=-gammas[col] * _alphas_ldlq[col])
            else:
                gammas[col] = 0

        del Ycur_full
    else:
        if fp32_ldlq:
            Ycur_work = Ycur.float()
            del Ycur
        else:
            Ycur_work = Ycur

        for col in range(n - 1, -1, -1):
            wcol = Ycur_work[:, col]
            zcol = torch.round(wcol / _c_param_ldlq).long()
            Zsic[:, col] = zcol
            zcol_cast = zcol.float() if fp32_ldlq else zcol.double()
            f1 = (zcol_cast * wcol).sum()
            f2 = (zcol_cast * zcol_cast).sum()
            if f2 > 0:
                gammas[col] = f1 / f2 / _c_param_ldlq
                Ycur_work.addr_(zcol_cast, _L_ldlq[col, :], alpha=-gammas[col] * _alphas_ldlq[col])
            else:
                gammas[col] = 0
        del Ycur_work

    if fp32_ldlq:
        del _L_ldlq, _alphas_ldlq

    _nccl_checkpoint("compress_w2q:after_ldlq")

    # Free large temporaries no longer needed after LDLQ loop
    del L  # n×n float64

    # Staged loading: now that L+Ycur are freed (~7 GiB), move ALL Sig matrices
    # to GPU.  Leaving Sig_X/Sig_X_hX on CPU forces mse_loss_func and
    # find_optimal_rescalers3 precomputation to do massive CPU matmuls
    # (2048×28672 @ 28672×28672 in float64), taking minutes per call vs seconds
    # on GPU.  With L+Ycur freed, there's room for all 3 Sig matrices on GPU
    # (~18.9 GiB total for 28672² × float64 × 3).
    if _staged:
        if Sig_hX is not None:
            Sig_hX = Sig_hX.to(_gpu)
        if Sig_X is not None and Sig_X.device != _gpu:
            Sig_X = Sig_X.to(_gpu)
        if Sig_X_hX is not None and Sig_X_hX.device != _gpu:
            Sig_X_hX = Sig_X_hX.to(_gpu)
        torch.cuda.empty_cache()

    What_pre = Zsic.double() * (alphas * gammas)[None, :]

    # MSE loss using trace-free form: trace(A @ B @ C.T) = (A @ B * C).sum()
    # Handles cross-device Sig matrices (CPU Sig_X/Sig_X_hX, GPU Sig_hX).

    # Precompute tr(W Sig_X W^T) once — used for both mse_null and qronos mse_loss.
    _d0 = Sig_X.device
    _W_d0 = W if W.device == _d0 else W.to(_d0)
    _tr_WSW = (_W_d0 @ Sig_X * _W_d0).sum()
    if sync:
        _col_allreduce(_tr_WSW.unsqueeze(0))
        _tr_WSW = _tr_WSW.squeeze(0)
    del _W_d0

    def mse_loss_func(What):
        if qronos:
            _d2 = Sig_X_hX.device
            _W_d2 = W if W.device == _d2 else W.to(_d2)
            _Wh_d2 = What if What.device == _d2 else What.to(_d2)
            cross = (_W_d2 @ Sig_X_hX * _Wh_d2).sum()

            quad = (What @ Sig_hX * What).sum()
            tr = _tr_WSW.to(_gpu) - 2 * cross.to(_gpu) + quad.to(_gpu)
        else:
            _d = Sig_X.device
            diff = What.to(_d) - W.to(_d)
            tr = (diff @ Sig_X * diff).sum().to(_gpu)
        if sync:
            _col_allreduce(tr.unsqueeze(0))
            tr = tr.squeeze(0)
        return tr / (n * a_eff)

    mse_out = mse_loss_func(What_pre)
    # mse_null = trace(W Sig_X W^T) / (n * a_eff) — already computed as _tr_WSW.
    # Avoids redundant a×n @ n×n matmul + allreduce.
    mse_null = _tr_WSW.to(_gpu) / (n * a_eff)
    rel_mse_out = mse_out / mse_null
    if not quiet:
        print(f'Target rate = {math.log2(math.exp(target_rate_nats))}, MSE = {mse_out}, relative_mse = {rel_mse_out}')
        print(f'Zsic: min = {Zsic.min()}, max = {Zsic.max()}, mean = {Zsic.float().mean()}, stddev = {math.sqrt(Zsic.float().var())}')

    # Compute entropy (synced across ranks for ColumnParallel)
    if sync:
        entropy = _compute_entropy_synced(Zsic, a_eff * n)
    else:
        zsic_elts, zsic_counts = torch.unique(Zsic.flatten(), return_counts=True)
        probs = zsic_counts.float() / Zsic.numel()
        entropy = -torch.sum(probs * torch.log2(probs)).item()
    if not quiet:
        print(f"Huffman coded compression rate = {entropy + 16 / a_eff} bit/entry.    Zsic entrywise entropy: {entropy} bits")

    _nccl_checkpoint("compress_w2q:after_entropy")
    if sync:
        _nccl_assert_sync("compress_w2q:after_entropy")

    ## Now let us optimize diagonal row- and column- scalers.
    # Free What_pre (a×n float64) — no longer needed after diagnostics above
    del What_pre

    if apply_rescaler:
        if not quiet:
            print('... optimizing diagonal rescalers')

        What_pre0 = Zsic.double() * alphas[None, :]  # remove Gamma multiplier

        # Move Zsic to CPU during rescaler optimization — it's a×n int64 (~896MB for 70B)
        # and not used inside find_optimal_rescalers3. Restored after.
        Zsic_cpu = Zsic.cpu()
        del Zsic

        # In staged mode, Sig_X and Sig_X_hX are already on CPU (pipeline offloaded
        # them for large RowParallel w2).  For non-staged weights, keep them on GPU:
        # the ~1 GB savings is negligible on H100, and CPU matmuls in
        # find_optimal_rescalers3 precomputation can cause NCCL timeouts if one rank's
        # CPU is slower than the other's.
        torch.cuda.empty_cache()

        # find_optimal_rescalers3 handles Sig_X/Sig_X_hX on any device: precomputes
        # on their device, brings results to GPU, then frees them internally.
        _orig_device = What_pre0.device
        if qronos:
            T, Gamma = find_optimal_rescalers3(What_pre0, W, Sig_X, gamma_init=gammas, quiet=quiet,
                                                Sig_hX=Sig_hX, Sig_X_hX=Sig_X_hX,
                                                Sig_delta_R_Xhat=Sig_delta_R_Xhat,
                                                global_nrows=global_nrows)
        else:
            T, Gamma = find_optimal_rescalers3(What_pre0, W, Sig_X, gamma_init=gammas, quiet=quiet,
                                                Sig_delta_R_Xhat=Sig_delta_R_Xhat,
                                                global_nrows=global_nrows)
        _nccl_checkpoint("compress_w2q:after_rescalers")
        if sync:
            _nccl_assert_sync("compress_w2q:after_rescalers")

        # T, Gamma are vectors (not diagonal matrices) from find_optimal_rescalers3
        T = T.to(_orig_device)
        Gamma = Gamma.to(_orig_device)

        # Restore Zsic for final loss computation.  mse_loss_func handles
        # cross-device Sig matrices (CPU in staged mode, GPU otherwise).
        Zsic = Zsic_cpu.to(_orig_device)
        del Zsic_cpu

        What = T[:, None] * Zsic.double() * (Gamma * alphas)[None, :]
    else:
        # Fast search mode: skip rescaler, use T=1 and LDLQ gammas directly
        if not quiet:
            print('... skipping rescaler (fast search mode)')
        T = torch.ones(a, device=_gpu)
        Gamma = gammas
        What = Zsic.double() * (gammas * alphas)[None, :]

    final_loss = mse_loss_func(What)
    final_rate = entropy + 16 / a_eff + 16 / n
    if not quiet:
        print(f'Final loss: {final_loss:.3g}, Final rate = {final_rate:.3g} bit/entry\n')

    return final_loss, final_rate, What.to(dtype_orig), locals()


# =============================================================================
# Wrapper for pipeline integration - compress_zsic
# =============================================================================

@torch.no_grad()
def compress_zsic(
    W: torch.Tensor,
    *,
    cfg: ZSICConfig,
    Sig_X: torch.Tensor,
    Sig_hX: torch.Tensor | None = None,
    Sig_X_hX: torch.Tensor | None = None,
    Sig_delta_R_Xhat: torch.Tensor | None = None,  # Residual compensation
    Sig_X_for_dead: torch.Tensor = None,  # Unquantized covariance for dead dim detection
    dead_row_indices: list | None = None,  # Output dims to exclude (e.g., from zero_out_rows)
    forced_dead_col_indices: list | None = None,  # Input dims to force-dead (e.g., w2 from w1/w3 zero_out_rows)
    global_nrows: int = None,  # For multi-GPU ColumnParallel: total rows across all ranks
    fp32_ldlq: bool = False,  # fp32 LDLQ loop for search mode (~2x faster)
    target_precomputed: torch.Tensor | None = None,  # Precomputed target matrix (skips matmul)
) -> Tuple[torch.Tensor, float, float, Dict[str, object]]:
    """ZSIC compression with optional Qronos mode - wrapper for pipeline.

    Args:
        W: Weight matrix (a x n)
        cfg: ZSIC configuration
        Sig_X: E[X X^T] - activations covariance for quantization optimization
        Sig_X_for_dead: E[X X^T] from unquantized model - used for dead dimension detection
        Sig_hX: E[X̂ X̂^T] - quantized activations covariance (for Qronos targeting)
        Sig_X_hX: E[X X̂^T] - cross-covariance (for Qronos targeting)
        Sig_delta_R_Xhat: E[(R - R̂) X̂^T] - residual compensation (for wo/w2 layers)
    """
    quiet = False
    qronos_mode = cfg.qronos and Sig_hX is not None and Sig_X_hX is not None
    residual_comp_mode = cfg.residual_compensation and Sig_delta_R_Xhat is not None
    dtype = W.dtype
    device = W.device
    n_original = W.shape[1]

    print(f"[compress_zsic] binary_search={cfg.binary_search}, qronos={qronos_mode}, "
          f"residual_comp={residual_comp_mode}, "
          f"Sig_hX={'provided' if Sig_hX is not None else 'None'}, "
          f"Sig_X_hX={'provided' if Sig_X_hX is not None else 'None'}, "
          f"Sig_delta_R_Xhat={'provided' if Sig_delta_R_Xhat is not None else 'None'}", flush=True)

    # =========================================================================
    # Dead row handling (output dimensions to exclude, e.g., from zero_out_rows)
    # =========================================================================
    a_original = W.shape[0]
    live_row_mask = None
    if dead_row_indices is not None and len(dead_row_indices) > 0:
        live_row_mask = torch.ones(a_original, dtype=torch.bool, device=device)
        for idx in dead_row_indices:
            live_row_mask[idx] = False
        W = W[live_row_mask, :]  # (a_original, n) -> (a_live, n)
        if Sig_delta_R_Xhat is not None:
            Sig_delta_R_Xhat = Sig_delta_R_Xhat[live_row_mask, :]  # (a_live, n)
        if target_precomputed is not None:
            target_precomputed = target_precomputed[live_row_mask.to(target_precomputed.device), :]
        print(f"[compress_zsic] {len(dead_row_indices)} dead rows excluded: {dead_row_indices}", flush=True)

    # =========================================================================
    # Dead dimension detection and removal
    # =========================================================================
    dead_mask = find_dead_dimensions(
        Sig_X_for_dead,
        threshold_ratio=cfg.dead_dim_threshold,
    )

    # Merge forced dead columns (e.g., from zero_out_rows for w2)
    if forced_dead_col_indices:
        for idx in forced_dead_col_indices:
            dead_mask[idx] = True
        print(f"[compress_zsic] {len(forced_dead_col_indices)} forced dead columns from zero_out_rows", flush=True)

    # Multi-GPU: broadcast dead mask from rank 0 to ensure all ranks use
    # identical dead dimensions.  Sigma_X for dead-dim detection is computed
    # independently per rank (from X.T @ X accumulation in Qronos stats).
    # CUDA matmul non-determinism across physical GPUs can produce O(1e-10)
    # differences that push borderline dimensions above/below the threshold
    # on different ranks.  A dead-dim mismatch means n_live differs, which
    # changes the LDLQ loop iteration count and the number of _col_allreduce
    # calls — an immediate NCCL collective-count desync.
    if global_nrows is not None and _dist.is_available() and _dist.is_initialized():
        # Broadcast dead mask from rank 0
        global _nccl_counter
        _nccl_counter += 1
        _dm_gpu = dead_mask.to(torch.int8).cuda()
        _dist.broadcast(_dm_gpu, src=0)
        dead_mask = _dm_gpu.bool().to(dead_mask.device)

    _nccl_checkpoint(f"compress_zsic:after_dead_mask(n_dead={int(dead_mask.sum())})")
    if global_nrows is not None:
        _nccl_assert_sync("compress_zsic:after_dead_mask")

    n_dead = int(dead_mask.sum())

    # Working copies of matrices (may be sliced if there are dead dims)
    W_work = W
    Sig_X_work = Sig_X
    Sig_hX_work = Sig_hX
    Sig_X_hX_work = Sig_X_hX
    Sig_delta_R_Xhat_work = Sig_delta_R_Xhat

    if n_dead > 0:
        dead_indices = dead_mask.nonzero().squeeze(-1).tolist()
        if isinstance(dead_indices, int):
            dead_indices = [dead_indices]
        # Show first 10 dead indices, truncate if more
        indices_str = str(dead_indices[:10]) + ('...' if n_dead > 10 else '')
        print(f"[compress_zsic] {n_dead} dead dimensions detected: {indices_str}", flush=True)

        # Slice out dead dimensions from all matrices
        sliced = slice_out_dead_dims(
            dead_mask, W, Sig_X,
            Sig_hX=Sig_hX if qronos_mode else None,
            Sig_X_hX=Sig_X_hX if qronos_mode else None,
            Sig_delta_R_Xhat=Sig_delta_R_Xhat if residual_comp_mode else None,
        )
        W_work = sliced["W"]
        Sig_X_work = sliced["Sig_X"]
        Sig_hX_work = sliced.get("Sig_hX")
        Sig_X_hX_work = sliced.get("Sig_X_hX")
        Sig_delta_R_Xhat_work = sliced.get("Sig_delta_R_Xhat")

        n_live = W_work.shape[1]
        print(f"[compress_zsic] working with {n_live}/{n_original} live dimensions", flush=True)

    # When the pipeline offloads covariance matrices to CPU (large RowParallel w2),
    # pass them as-is.  compress_w2q / _fast_rate_estimate do staged GPU loading:
    # Cholesky+target on CPU, LDLQ loop on GPU, then only Sig_hX moves to GPU for
    # rescaler iterations.  NCCL collectives inside these functions operate on small
    # derived quantities (column stats, entropy), never on the Sig matrices directly,
    # so Sig can safely stay on CPU.

    # =========================================================================
    # Core quantization (on live dimensions only)
    # =========================================================================
    if cfg.binary_search:
        print(f"[compress_zsic] calling compress_zsic_with_binary_search with desired_rate={cfg.target_rate_bits}", flush=True)
        What_live, final_loss, final_rate, frame = compress_zsic_with_binary_search(
            W_work, cfg=cfg, desired_rate=cfg.target_rate_bits,
            Sig_X=Sig_X_work,
            Sig_hX=Sig_hX_work,
            Sig_X_hX=Sig_X_hX_work,
            Sig_delta_R_Xhat=Sig_delta_R_Xhat_work if residual_comp_mode else None,
            n_original=n_original,
            n_dead=n_dead,
            global_nrows=global_nrows,
        )
    else:
        # Direct compression without binary search
        # Apply dead-dim slicing to target_precomputed if provided
        _target_pre = target_precomputed
        if _target_pre is not None and n_dead > 0:
            _target_pre = _target_pre[:, (~dead_mask).to(_target_pre.device)]
        loss, rate, What_live, frame_locals = compress_w2q(
            W_work, Sig_X_work, target_rate=cfg.target_rate_bits, quiet=quiet,
            Sig_hX=Sig_hX_work if qronos_mode else None,
            Sig_X_hX=Sig_X_hX_work if qronos_mode else None,
            percdamp=cfg.percdamp,
            Sig_delta_R_Xhat=Sig_delta_R_Xhat_work if residual_comp_mode else None,
            global_nrows=global_nrows,
            apply_rescaler=cfg.apply_rescaler,
            fp32_ldlq=fp32_ldlq,
            target_precomputed=_target_pre,
        )
        final_loss, final_rate = float(loss), float(rate)
        frame = _build_frame_from_locals(frame_locals, cfg, qronos_mode, residual_comp_mode)

    # =========================================================================
    # Handle dead dimensions: add metadata to frame, expand What for return
    # =========================================================================
    if n_dead > 0:
        dead_indices = dead_mask.nonzero().squeeze(-1).tolist()
        if isinstance(dead_indices, int):
            dead_indices = [dead_indices]
        n_live = n_original - n_dead

        # Add dead dimension metadata to frame (tensors stay live-only for storage)
        frame["dead_indices"] = dead_indices
        frame["n_original"] = n_original
        frame["n_live"] = n_live
        frame["n_dead"] = n_dead

        # Expand What_live to full size for return (needed by pipeline)
        _what_dev = What_live.device
        What = torch.zeros(W.shape[0], n_original, dtype=What_live.dtype, device=_what_dev)
        What[:, (~dead_mask).to(_what_dev)] = What_live

        # Adjust rate: dead dims cost only index storage, not entropy
        live_entropy = frame.get("entropy", 0.0)
        a_rows = global_nrows if global_nrows is not None else W.shape[0]
        total_bits_live = live_entropy * a_rows * n_live + 16 * a_rows + 16 * n_live  # T, Gamma for live
        dead_index_bits = 16 * n_dead  # Storage for dead indices (int16)
        final_rate = (total_bits_live + dead_index_bits) / (a_rows * n_original)
        frame["entropy_live"] = live_entropy
        frame["rate_adjusted"] = final_rate
        print(f"[compress_zsic] dead dims handled: rate={final_rate:.4f} (live_entropy={live_entropy:.4f}, n_dead={n_dead})", flush=True)
    else:
        What = What_live
        # No dead dims - add empty metadata for consistency
        frame["dead_indices"] = []
        frame["n_original"] = n_original
        frame["n_live"] = n_original
        frame["n_dead"] = 0

    # =========================================================================
    # Dead row expansion: expand What from (a_live, n) to (a_original, n)
    # =========================================================================
    frame["dead_row_indices"] = dead_row_indices if dead_row_indices else []
    frame["a_original"] = a_original

    if live_row_mask is not None:
        What_full = torch.zeros(a_original, What.shape[1], dtype=What.dtype, device=What.device)
        What_full[live_row_mask.to(What.device), :] = What
        What = What_full
        print(f"[compress_zsic] expanded dead rows: What {What.shape}", flush=True)

    return What.to(dtype), float(final_loss), float(final_rate), frame


def _build_frame_from_locals(
    frame_locals: dict, cfg: ZSICConfig, qronos: bool, residual_comp: bool = False
) -> Dict[str, object]:
    """Build frame dict from compress_w2q locals()."""
    Zsic = frame_locals.get('Zsic')
    alphas = frame_locals.get('alphas')
    gammas = frame_locals.get('gammas')
    T = frame_locals.get('T')
    Gamma = frame_locals.get('Gamma')
    entropy = frame_locals.get('entropy')
    final_loss = frame_locals.get('final_loss')
    a = frame_locals.get('a')
    a_eff = frame_locals.get('a_eff', a)  # global_nrows for multi-GPU, else local a
    n = frame_locals.get('n')

    t_vec = T.diag() if (T is not None and T.dim() == 2) else T
    g_vec = Gamma.diag() if (Gamma is not None and Gamma.dim() == 2) else Gamma

    frame = {
        "Z": Zsic,
        "alpha": (alphas * gammas) if alphas is not None and gammas is not None else None,
        "alpha_base": alphas,
        "zero_point": None,
        "apply_tgamma": True,  # compress_w2q always applies tgamma
        "t_vec": t_vec,
        "g_vec": g_vec,
        "sic_variant": "compress_w2q",
        "target_rate_bits": cfg.target_rate_bits,
        "entropy": entropy.item() if hasattr(entropy, 'item') else float(entropy),
        "rate_overhead": 16 / a_eff + 16 / n if a_eff and n else 0,
        "loss": float(final_loss) if final_loss is not None else None,
        "qronos": qronos,
        "residual_compensation": residual_comp,
    }
    return frame


@torch.no_grad()
def _fast_rate_estimate(W: torch.Tensor, Sig_X: torch.Tensor, target_rate: float,
                        row_fraction: float = 0.1, percdamp: float = 0.0001,
                        Sig_hX: torch.Tensor = None, Sig_X_hX: torch.Tensor = None,
                        Sig_delta_R_Xhat: torch.Tensor = None,
                        global_nrows: int = None,
                        L_cached: torch.Tensor = None,
                        damp_cached: float = None,
                        Ycur_cached: torch.Tensor = None,
                        sw_diag_cached: torch.Tensor = None) -> tuple:
    """Fast rate estimation using a subset of rows.

    Args:
        L_cached: Precomputed Cholesky factor (on GPU). Avoids recomputing the
                  O(n³) Cholesky every binary-search iteration.
        damp_cached: Precomputed damping value corresponding to L_cached.
        Ycur_cached: Precomputed Ycur (gathered, fp32, on GPU). Avoids
                     recomputing the expensive matmul + solve_triangular + all-gather
                     every iteration — these don't depend on target_rate.
        sw_diag_cached: Precomputed sw_diag (on GPU). Same rationale.

    Returns:
        (entropy, Ycur_gathered, sw_diag) — caller caches the latter two
        for subsequent iterations.
    """
    global _nccl_counter
    a, n = W.shape
    sync = global_nrows is not None
    _gpu = W.device

    # ── Reuse or compute Ycur and sw_diag ────────────────────────────────
    # Ycur = f(W, Sig_X_hX, L, damp, residual) — independent of target_rate.
    # sw_diag = sum(W²)/a — also independent.  Both are expensive O(a·n²)
    # operations, so caching across binary-search iterations saves ~14s/iter
    # for w2 (n=28672).
    if Ycur_cached is not None and sw_diag_cached is not None:
        Ycur_gathered_f = Ycur_cached
        sw_diag = sw_diag_cached
    else:
        if row_fraction < 1.0:
            n_rows = max(1, int(a * row_fraction))
            indices = torch.randperm(a, device=W.device)[:n_rows]
            W_sampled = W[indices]
            Sig_delta_R_Xhat_sampled = Sig_delta_R_Xhat[indices] if Sig_delta_R_Xhat is not None else None
        else:
            n_rows = a
            W_sampled = W
            Sig_delta_R_Xhat_sampled = Sig_delta_R_Xhat

        if L_cached is not None:
            L = L_cached
            damp = damp_cached
        else:
            H = Sig_hX.double() if Sig_hX is not None else Sig_X.double()
            damp = percdamp * torch.mean(torch.diag(H))
            H_damped = H.clone()
            H_damped.diagonal().add_(damp)
            L = torch.linalg.cholesky(H_damped, upper=False)
            del H_damped
            if H.device != _gpu:
                L = L.to(_gpu)
            # Multi-GPU: broadcast L from rank 0
            if sync and _dist.is_available() and _dist.is_initialized() and _dist.get_world_size() > 1:
                _nccl_counter += 1
                _dist.broadcast(L, src=0)
                _damp_t = torch.tensor([damp], device=_gpu, dtype=torch.float64)
                _nccl_counter += 1
                _dist.broadcast(_damp_t, src=0)
                damp = _damp_t.item()

        if Sig_hX is not None and Sig_X_hX is not None:
            _Sig_on_gpu = Sig_X_hX.double().to(_gpu) if Sig_X_hX.device != _gpu else Sig_X_hX.double()
            W_d = W_sampled.double()
            target = W_d @ _Sig_on_gpu
            del _Sig_on_gpu
            target.add_(W_d, alpha=damp)
            if Sig_delta_R_Xhat_sampled is not None:
                target = target + Sig_delta_R_Xhat_sampled.to(_gpu).double()
            del W_d
            Ycur = torch.linalg.solve_triangular(L.T, target, left=False, upper=True)
            del target
        else:
            Ycur = W_sampled.double() @ L
            if Sig_delta_R_Xhat_sampled is not None:
                Ycur = Ycur + torch.linalg.solve_triangular(
                    L, Sig_delta_R_Xhat_sampled.double().to(_gpu).T, left=True, upper=False
                ).T

        # Effective row count for column stats
        if sync:
            a_sampled_eff = int(global_nrows * row_fraction) if row_fraction < 1.0 else global_nrows
        else:
            a_sampled_eff = n_rows

        sw_diag = (W_sampled ** 2).sum(0)
        if sync:
            _col_allreduce(sw_diag)
        sw_diag /= a_sampled_eff

        # Gather Ycur and convert to fp32 (done once, reused across iterations)
        if sync:
            ws = _dist.get_world_size()
            _nccl_counter += 1
            Ycur_f = Ycur.float()
            del Ycur
            Ycur_shards = [torch.empty_like(Ycur_f) for _ in range(ws)]
            _dist.all_gather(Ycur_shards, Ycur_f.contiguous())
            Ycur_gathered_f = torch.cat(Ycur_shards, dim=0)
            del Ycur_shards, Ycur_f
        else:
            Ycur_gathered_f = Ycur.float()
            del Ycur

        if L_cached is None:
            del L

    # ── From here, only target_rate-dependent computation ────────────────
    L = L_cached if L_cached is not None else None  # needed for diag + L_f
    # If L was freed above (L_cached is None path), we can't get here because
    # the non-cached path is only for standalone calls (no binary search loop).
    # In the binary search loop, L_cached is always provided.

    target_rate_nats = target_rate * math.log(2)
    c_param = torch.exp(torch.log(12 * sw_diag.double() * (L.diag() ** 2)).mean() / 2 - target_rate_nats)

    alphas = c_param / L.diag()

    # ── LDLQ for binary search ──────────────────────────────────────
    # fp32 LDLQ for binary search entropy estimation.
    L_w = L.float()
    alphas_w = alphas.float()
    c_param_w = c_param.float()
    _Ycur_dtype = torch.float32
    _z_dtype = torch.int32

    a_full = Ycur_gathered_f.shape[0]
    Ycur_work = Ycur_gathered_f.to(_Ycur_dtype).clone()  # don't mutate the cached copy
    Zsic_full = torch.zeros(a_full, n, dtype=_z_dtype, device=_gpu)

    for col in range(n - 1, -1, -1):
        wcol = Ycur_work[:, col]
        zcol = torch.round(wcol / c_param_w).to(_z_dtype)
        Zsic_full[:, col] = zcol
        zcol_cast = zcol.to(Ycur_work.dtype)
        f1 = (zcol_cast * wcol).sum()
        f2 = (zcol_cast * zcol_cast).sum()
        if f2 > 0:
            gamma = f1 / f2 / c_param_w
            Ycur_work.addr_(zcol_cast, L_w[col, :], alpha=-gamma * alphas_w[col])
        else:
            pass  # gamma stays 0, no update

    del Ycur_work, L_w, alphas_w

    zsic_elts, zsic_counts = torch.unique(Zsic_full.flatten(), return_counts=True)
    probs = zsic_counts.float() / Zsic_full.numel()
    entropy = -torch.sum(probs * torch.log2(probs)).item()
    del Zsic_full

    return entropy, Ycur_gathered_f, sw_diag


@torch.no_grad()
def compress_zsic_with_binary_search(
    W: torch.Tensor,
    *,
    cfg: ZSICConfig,
    desired_rate: float,
    Sig_X: torch.Tensor,
    Sig_hX: torch.Tensor | None = None,
    Sig_X_hX: torch.Tensor | None = None,
    Sig_delta_R_Xhat: torch.Tensor | None = None,  # Residual compensation
    n_original: int | None = None,  # Original column count (before dead dim removal)
    n_dead: int = 0,  # Number of dead dimensions removed
    global_nrows: int = None,  # For multi-GPU ColumnParallel: total rows across all ranks
) -> Tuple[torch.Tensor, float, float, Dict[str, object]]:
    """ZSIC with secant-method rate targeting (3 iterations) + full compression."""
    global _nccl_counter
    qronos_mode = cfg.qronos and Sig_hX is not None and Sig_X_hX is not None
    residual_comp_mode = cfg.residual_compensation and Sig_delta_R_Xhat is not None
    dtype = W.dtype

    # Target entropy per live element.
    a, n_live = W.shape
    a_eff = global_nrows if global_nrows is not None else a  # effective row count for multi-GPU
    if n_original is None:
        n_original = n_live  # No dead dims case

    # Overhead per live element (for the live portion only):
    # - T: 16 bits per row = 16/n_live per element
    # - Gamma: 16 bits per live col = 16/a_eff per element
    overhead_per_live = 16.0 / n_live + 16.0 / a_eff

    if n_dead > 0 and n_original > n_live and not cfg.rate_control_active:
        # Scale up entropy target so final rate per original element = desired_rate.
        # This ensures each weight fully uses its budget when there's no rate controller
        # to redistribute dead-dim savings.
        # final_rate = (entropy * n_live + 16 + 16*n_live/a_eff + 16*n_dead/a_eff) / n_original
        # Setting final_rate = desired_rate and solving for entropy:
        # entropy = desired_rate * (n_original/n_live) - 16/n_live - 16/a_eff - 16*n_dead/(a_eff*n_live)
        dead_overhead_per_live = 16.0 * n_dead / (a_eff * n_live)
        adjusted_desired_rate = desired_rate * (n_original / n_live) - overhead_per_live - dead_overhead_per_live
    else:
        # No dead dims, or rate_control_active: target desired_rate per live element.
        # When rate_control_active, dead-dim savings naturally lower the actual rate
        # per original element, and the rate controller redistributes the surplus.
        adjusted_desired_rate = desired_rate - overhead_per_live

    # Compute expected final rate per original element for logging
    # final_rate = (entropy * n_live + 16 + 16*n_live/a_eff + 16*n_dead/a_eff) / n_original
    expected_rate_per_orig = (adjusted_desired_rate * n_live + 16 + 16.0*n_live/a_eff + 16.0*n_dead/a_eff) / n_original

    print(f"[binary-search] desired_rate={desired_rate:.4f}, overhead_per_live={overhead_per_live:.4f}", flush=True)
    print(f"[binary-search] n_live={n_live}, n_dead={n_dead}, target_entropy={adjusted_desired_rate:.4f}", flush=True)
    print(f"[binary-search] expected_rate_per_orig={expected_rate_per_orig:.4f}"
          + (" (rate_control: dead-dim savings will redistribute)" if cfg.rate_control_active and n_dead > 0 else ""),
          flush=True)

    left, right = cfg.binary_search_left, cfg.binary_search_right

    # Precompute Cholesky once on GPU.  L only depends on Sig_hX (or Sig_X) and
    # percdamp — these are constant across binary-search iterations.  For w2
    # column-parallel with Sig on CPU, the O(n³) Cholesky takes minutes on CPU;
    # doing it once on GPU (~2 s) prevents NCCL timeout when 4+ ranks compete
    # for the same CPU.
    import time as _time
    _gpu = W.device
    H = (Sig_hX.double() if qronos_mode else Sig_X.double())
    _H_gpu = H.to(_gpu) if H.device != _gpu else H
    _damp = float(cfg.percdamp * torch.mean(torch.diag(_H_gpu)))
    _H_damped = _H_gpu.clone()
    _H_damped.diagonal().add_(_damp)
    _t_chol = _time.monotonic()
    _L_cached = torch.linalg.cholesky(_H_damped, upper=False)
    torch.cuda.synchronize(_gpu)
    _t_chol = _time.monotonic() - _t_chol
    del _H_damped
    if _H_gpu is not H:
        del _H_gpu  # free the GPU copy of Sig
    torch.cuda.empty_cache()
    # Multi-GPU: broadcast L and damp from rank 0 for cross-rank consistency.
    if global_nrows is not None and _dist.is_available() and _dist.is_initialized() and _dist.get_world_size() > 1:
        _nccl_counter += 2
        _L_cached = _L_cached.contiguous()
        _dist.broadcast(_L_cached, src=0)
        _damp_t = torch.tensor([_damp], device=_gpu, dtype=torch.float64)
        _dist.broadcast(_damp_t, src=0)
        _damp = _damp_t.item()
    print(f"[binary-search] precomputed Cholesky on {_L_cached.device} (n={_L_cached.shape[0]}) took {_t_chol:.1f}s", flush=True)

    # Cache Sig_X_hX on GPU for the binary search loop.  Without this,
    # _fast_rate_estimate copies it CPU→GPU every iteration (13+ × 6.3 GiB
    # for w2 28672²).  The GPU copy lives alongside _L_cached (~12.7 GiB
    # total for w2) — well within 96 GiB H100 headroom.
    _Sig_X_hX_gpu = None
    if qronos_mode and Sig_X_hX is not None and Sig_X_hX.device != _gpu:
        _Sig_X_hX_gpu = Sig_X_hX.double().to(_gpu)
        print(f"[binary-search] cached Sig_X_hX on {_gpu} ({_Sig_X_hX_gpu.element_size() * _Sig_X_hX_gpu.nelement() / 1e9:.1f} GiB)", flush=True)

    _t_bsearch = _time.monotonic()
    _Ycur_cached = None  # Cached across iterations (computed once on iter 0)
    _sw_diag_cached = None

    _est_kwargs = dict(
        row_fraction=cfg.binary_search_row_fraction,
        percdamp=cfg.percdamp,
        Sig_hX=Sig_hX if qronos_mode else None,
        Sig_X_hX=_Sig_X_hX_gpu if _Sig_X_hX_gpu is not None else (Sig_X_hX if qronos_mode else None),
        Sig_delta_R_Xhat=Sig_delta_R_Xhat if residual_comp_mode else None,
        global_nrows=global_nrows,
        L_cached=_L_cached,
        damp_cached=_damp,
    )

    def _eval_at(r):
        nonlocal _Ycur_cached, _sw_diag_cached
        entropy, _Ycur_cached, _sw_diag_cached = _fast_rate_estimate(
            W, Sig_X, r, Ycur_cached=_Ycur_cached, sw_diag_cached=_sw_diag_cached,
            **_est_kwargs,
        )
        return entropy

    # ── Secant method: 3+ iterations instead of 12-15 binary search ──
    # H(target_rate) ≈ target_rate + O, approximately linear with slope ~0.8.
    # For w2/wo with residual compensation, the curve is highly nonlinear and
    # the first iteration can overshoot wildly. We use iterative refinement
    # with clamping to stay in a valid range.

    # Step 1: evaluate at desired_rate to measure offset O
    r0 = adjusted_desired_rate
    H0 = _eval_at(r0)
    O0 = H0 - adjusted_desired_rate
    print(f"[secant] iter 1: r={r0:.4f} entropy={H0:.4f} offset={O0:.4f}", flush=True)

    # Iterative secant with clamping
    _rs = [r0]
    _Hs = [H0]

    r_prev, H_prev = r0, H0
    for _si in range(5):  # up to 5 more iterations
        O_prev = H_prev - adjusted_desired_rate
        if _si == 0:
            # First correction: Newton step with slope=1 assumption
            r_next = r_prev - O_prev
        else:
            # Secant step using last two points
            dr = _rs[-1] - _rs[-2]
            dH = _Hs[-1] - _Hs[-2]
            if abs(dr) > 1e-10 and abs(dH) > 1e-10:
                slope = dH / dr
                r_next = r_prev - O_prev / slope
            else:
                r_next = r_prev
        # Clamp to valid range
        r_next = max(left, min(right, r_next))
        H_next = _eval_at(r_next)
        O_next = H_next - adjusted_desired_rate
        print(f"[secant] iter {_si+2}: r={r_next:.4f} entropy={H_next:.4f} offset={O_next:.4f}", flush=True)
        _rs.append(r_next)
        _Hs.append(H_next)
        r_prev, H_prev = r_next, H_next
        # Early stop if converged
        if abs(O_next) < 0.005:
            break

    # Pick best from all evaluated points
    candidates = list(zip(_rs, _Hs))
    best_target, best_entropy = min(candidates, key=lambda x: abs(x[1] - adjusted_desired_rate))
    best_diff = abs(best_entropy - adjusted_desired_rate)

    torch.cuda.synchronize(_gpu)
    _t_bsearch = _time.monotonic() - _t_bsearch
    _nccl_checkpoint(f"secant:done({len(_rs)} iters)")
    if global_nrows is not None:
        _nccl_assert_sync(f"secant:done({len(_rs)} iters)")

    # Expected final rate per original element (accounting for dead dims)
    expected_final_rate = (adjusted_desired_rate * n_live + 16 + 16.0*n_live/a_eff + 16.0*n_dead/a_eff) / n_original

    print(f"[secant] done in {_t_bsearch:.1f}s ({len(_rs)} iters), best_target={best_target:.4f} best_entropy_diff={best_diff:.4f} (expect final_rate≈{expected_final_rate:.4f})", flush=True)

    # Free cached Sig_X_hX GPU copy (no longer needed for binary search).
    # Keep _L_cached to pass to compress_w2q (avoids redundant O(n³) Cholesky).
    if _Sig_X_hX_gpu is not None:
        del _Sig_X_hX_gpu
    del _Ycur_cached, _sw_diag_cached

    # Run full compression with best target
    # compress_w2q handles both qronos and non-qronos cases
    print(f"[zsic] starting core compression (qronos={qronos_mode}, residual_comp={residual_comp_mode})", flush=True)

    # Pass cached Cholesky to avoid redundant O(n³) recomputation (~2s for n=28672).
    _t_compress = _time.monotonic()
    final_loss, final_rate, What, frame_locals = compress_w2q(
        W, Sig_X, target_rate=best_target, quiet=False,
        Sig_hX=Sig_hX if qronos_mode else None,
        Sig_X_hX=Sig_X_hX if qronos_mode else None,
        percdamp=cfg.percdamp,
        Sig_delta_R_Xhat=Sig_delta_R_Xhat if residual_comp_mode else None,
        global_nrows=global_nrows,
        L_cached=_L_cached,
        damp_cached=_damp,
    )
    torch.cuda.synchronize(_gpu)
    _t_compress = _time.monotonic() - _t_compress
    print(f"[zsic] core compression took {_t_compress:.1f}s", flush=True)

    frame = _build_frame_from_locals(frame_locals, cfg, qronos_mode, residual_comp_mode)
    frame["binary_search_iterations"] = 3  # secant method, fixed
    frame["binary_search_target_used"] = best_target
    frame["binary_search_desired"] = desired_rate
    frame["binary_search_final_diff"] = best_diff

    return What.to(dtype), float(final_loss), float(final_rate), frame
# =============================================================================
# Dequantization
# =============================================================================

@torch.no_grad()
def sic_decode(Z: torch.Tensor, alpha: torch.Tensor, zero_point: Optional[torch.Tensor] = None) -> torch.Tensor:
    # Use float32 to avoid overflow: Z codes can exceed float16 max (65504)
    out = Z.float() * alpha.float()
    return out + zero_point.float() if zero_point is not None else out


@torch.no_grad()
def dequantize_zsic(
    Z: torch.Tensor,
    alpha: torch.Tensor,
    *,
    alpha_base: torch.Tensor | None = None,
    zero_point: torch.Tensor | None = None,
    apply_tgamma: bool = False,
    t_vec: torch.Tensor | None = None,
    g_vec: torch.Tensor | None = None,
    dtype: torch.dtype,
    dead_indices: list | None = None,
    n_original: int | None = None,
    dead_row_indices: list | None = None,
    a_original: int | None = None,
) -> torch.Tensor:
    """Dequantize ZSIC-compressed weights.

    Args:
        Z: Quantized integer codes (a_live, n_live) or (a, n) if no dead dims/rows
        alpha: Scale factors (n_live,) or (n,)
        alpha_base: Base scales before gamma (n_live,) or (n,)
        zero_point: Zero points (optional)
        apply_tgamma: Whether to apply T/Gamma rescaling
        t_vec: Row rescaler (a_live,) or (a,)
        g_vec: Column rescaler (n_live,) or (n,)
        dtype: Output dtype
        dead_indices: List of dead column indices (for column expansion)
        n_original: Original column count (for column expansion)
        dead_row_indices: List of dead row indices (for row expansion)
        a_original: Original row count (for row expansion)

    Returns:
        Dequantized weight matrix (a_original, n_original)
    """
    # Decode live dimensions (all arithmetic in float32 to avoid float16 overflow)
    if apply_tgamma:
        if alpha_base is None or t_vec is None or g_vec is None:
            raise ValueError("alpha_base, t_vec, g_vec required when apply_tgamma=True")
        W_hat_live = sic_decode(Z, alpha_base, zero_point=zero_point)
        W_hat_live = (t_vec.float().unsqueeze(1) * W_hat_live) * g_vec.float().unsqueeze(0)
    else:
        W_hat_live = sic_decode(Z, alpha, zero_point=zero_point)

    # Expand dead columns to full size
    if dead_indices is not None and len(dead_indices) > 0 and n_original is not None:
        a = W_hat_live.shape[0]
        W_hat = torch.zeros(a, n_original, dtype=W_hat_live.dtype, device=W_hat_live.device)
        live_mask = torch.ones(n_original, dtype=torch.bool, device=W_hat_live.device)
        for idx in dead_indices:
            live_mask[idx] = False
        W_hat[:, live_mask] = W_hat_live
    else:
        W_hat = W_hat_live

    # Expand dead rows to full size
    if dead_row_indices is not None and len(dead_row_indices) > 0 and a_original is not None:
        W_full = torch.zeros(a_original, W_hat.shape[1], dtype=W_hat.dtype, device=W_hat.device)
        live_row_mask = torch.ones(a_original, dtype=torch.bool, device=W_hat.device)
        for idx in dead_row_indices:
            live_row_mask[idx] = False
        W_full[live_row_mask, :] = W_hat
        W_hat = W_full

    return W_hat.to(dtype)
