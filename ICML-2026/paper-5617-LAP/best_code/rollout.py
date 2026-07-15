# rollout.py

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Callable, Literal, Union, Dict
import torch.nn as nn
from mechanics import leapfrog_auto, x_verlet_auto, resolve_gamma
from losses import compute_loss  # assumed to exist
import numpy as np

# ============================================================
# 1. Utility: Marginal extractor
# ============================================================
from typing import Optional, Tuple, List
import torch
from torch import Tensor


def get_margs(
        X_em_torch: Tensor,
        time_grid: Tensor,
        p0_idx: int,
        t_idx: int,
        n_sub: Optional[int] = None,
        n_ahead: int = 1,
        subsample_targets: bool = True,
) -> Tuple[Tensor, Union[Tensor, List[Tensor]], Tensor, int, Tensor]:
    """
    Returns (x_t, Y_future, t0, L, x_idx) where

    x_t          : current marginal at time t_idx. n_eff particles each of dimension d. NaN-filtered and optionally subsampled.
    Y_future     : target marginals for next L steps
    t0           : scalar tensor, time at t_idx.
    L            : effective horizon (<= n_ahead).
    x_idx        : (n_eff,) indices for the NaN-filtered x at time t_idx.
                   `x_t == x_filtered[x_idx]`. Used to subset a paired tensor
                   (e.g. velocities V_em at the same time, NaN-filtered the same way)
                   so that particle correspondence between x_t and velocity is preserved. See BundleVelocityProvider.

    subsample_targets: whether to subsample the target marginals to match population size of x_t
    """
    device = X_em_torch.device
    T_plus_1 = X_em_torch.shape[2]
    T = T_plus_1 - 1

    if not (0 <= int(t_idx) <= T - 1):
        raise ValueError(f"t_idx out of range: {t_idx} (valid: 0..{T - 1})")

    L = int(min(int(n_ahead), T - int(t_idx)))
    t0 = time_grid[t_idx]

    # No future steps requested/available
    if L <= 0:
        x_t_full = X_em_torch[p0_idx, :, t_idx, :]
        x = x_t_full[torch.isfinite(x_t_full).all(dim=1)]
        if n_sub is not None and 0 < int(n_sub) < x.shape[0]:
            x_idx = torch.randperm(x.shape[0], device=device)[: int(n_sub)]
            x = x[x_idx]
        else:
            x_idx = torch.arange(x.shape[0], device=device)
        # Empty Y_future, type-matched to subsample_targets so callers can branch on it
        if subsample_targets:
            Y_future: Union[Tensor, List[Tensor]] = x.new_empty((0, x.shape[0], x.shape[1]))
        else:
            Y_future = []
        return x, Y_future, t0, 0, x_idx

    # --- slice current + future (raw, may contain NaNs) ---
    x_t_full = X_em_torch[p0_idx, :, t_idx, :]  # (N,d)
    Ys_full = [X_em_torch[p0_idx, :, t_idx + j, :] for j in range(1, L + 1)]

    # --- filter each time independently (NO cross-time identity) ---
    x = x_t_full[torch.isfinite(x_t_full).all(dim=1)]  # (Nx,d)
    Ys = [yj[torch.isfinite(yj).all(dim=1)] for yj in Ys_full]  # list of (Ny_j, d)

    # --- determine n_eff for x ---
    if subsample_targets:
        # Need a single n_eff that's valid for x AND every target step (since we'll stack).
        counts = [int(x.shape[0])] + [int(yj.shape[0]) for yj in Ys]
        n_eff = min(counts) if n_sub is None else min(int(n_sub), *counts)
    else:
        # x and targets are independent. n_eff only constrains x.
        n_eff_x = int(x.shape[0]) if n_sub is None else min(int(n_sub), int(x.shape[0]))
        n_eff = n_eff_x

    # --- empty-batch guard ---
    if n_eff <= 0 or x.shape[0] == 0 or any(int(yj.shape[0]) == 0 for yj in Ys):
        empty_x = x.new_empty((0, x.shape[1]))
        empty_idx = torch.empty(0, dtype=torch.long, device=device)
        if subsample_targets:
            return empty_x, x.new_empty((0, 0, x.shape[1])), t0, 0, empty_idx
        return empty_x, [], t0, 0, empty_idx

    # --- select x rows: subsample if needed, otherwise keep ORIGINAL ORDER (no shuffle) ---
    if n_eff < x.shape[0]:
        x_idx = torch.randperm(x.shape[0], device=device)[:n_eff]
    else:
        x_idx = torch.arange(x.shape[0], device=device)
    x_t = x[x_idx]

    # --- possibly subsample target marginals to match size fo rollout population ---
    if subsample_targets:
        Y_future = torch.stack(
            [
                Yj[torch.randperm(Yj.shape[0], device=device)[:n_eff]] if n_eff < Yj.shape[0] else Yj
                for Yj in Ys
            ],
            dim=0,
        )
    else:
        # Full-target path: keep each Y_j at its full NaN-filtered size
        Y_future = list(Ys)
        # Y_future = [Yj[torch.randperm(Yj.shape[0], device=device)] for Yj in Ys]
    return x_t, Y_future, t0, L, x_idx


class LearnableFriction(nn.Module):
    """
    Learn theta = log(gamma) with gamma = exp(theta). No clipping.
    """
    def __init__(self, init_gamma: float, gamma_init_floor: float = 1e-2):
        super().__init__()
        init_gamma = float(init_gamma)
        gamma0 = init_gamma if init_gamma > 0.0 else float(gamma_init_floor)
        self.theta = nn.Parameter(torch.tensor(np.log(gamma0), dtype=torch.float32))

    def friction_tensor(self) -> torch.Tensor:
        # gamma = exp(theta).
        return torch.exp(self.theta)

    def value(self) -> float:
        return float(self.friction_tensor().detach().cpu().item())

def _microsteps_between(t0, t1, dt_base: float, substeps_per_dt: int) -> int:
    # number of micro-steps to advance from physical time t0 -> t1
    delta = float((t1 - t0).detach().cpu().item())
    if delta < 0:
        raise ValueError(f"Non-monotone time_grid: {t0} -> {t1}")
    return int(round(delta / float(dt_base) * int(substeps_per_dt)))


from typing import Optional, Callable, Dict, Literal, Union
import torch
from torch import Tensor

def train_rollout_anchor_p0_randk(
        X_em_torch: Tensor,
        time_grid: Tensor,
        accel_train: Callable[[Tensor, float], Tensor],
        optimizer: torch.optim.Optimizer,
        dt_base: float,
        *,
        num_epochs: int,
        max_train_steps: int,
        substeps_per_dt: int,
        kernel_blur: float,
        loss_type: Literal["geom_sinkhorn", "geom_gaussian"] = "geom_sinkhorn",
        particles_per_batch: Optional[int] = None,
        vel_provider: Optional[Callable[[int, Tensor, float], Tensor]] = None,
        friction: Union[float, Tensor] = 0.0,
        debug: bool = False,
        name: str = "",
        verlet: str = "v",
        geom_p: int = 2,
        geom_scaling: float = 0.9,
        geom_debias: bool = True,
        geom_backend: Optional[str] = None,
        epoch_callback: Optional[Callable[[int, int, float, float], None]] = None,
        fixed_k: bool = False,
        subsample_targets: bool = True,
        k_ramp_fraction: float = 0.0,
):
    """
    Random-horizon rollout training, which always begins rollout from t=0.

    Each epoch:
      - sample k uniformly from {1, ..., K_max}
      - sample a population p0_idx
      - roll out from t=0 to t=k
      - compute mean loss across marginals m=1..k

    Returns dict with average and last loss.
    """
    device = X_em_torch.device
    num_p0, N, T_plus_1, d = X_em_torch.shape
    steps = T_plus_1 - 1  # macro steps in the bundle
    dt_train = float(dt_base) / int(substeps_per_dt)

    # horizon cap
    K_max = int(min(max_train_steps, steps))
    if K_max <= 0:
        raise ValueError(f"max_train_steps too small: max_train_steps={max_train_steps}, steps={steps}")

    friction_use = friction
    gamma0 = resolve_gamma(friction_use)
    is_learnable = bool(isinstance(gamma0, torch.Tensor) and gamma0.requires_grad)

    # choose integrator
    integrator = x_verlet_auto if verlet == "x" else leapfrog_auto

    init_val = float(gamma0.detach().item()) if isinstance(gamma0, torch.Tensor) else float(gamma0)
    print(
        f"[{name}] dt={dt_train:.6f}, "
        f"Friction={init_val:.4g} (Learnable={is_learnable}), verlet={verlet}"
    )

    loss_sum = 0.0
    loss_count = 0
    last_loss_val = None

    t0_val = float(time_grid[0].item())


    use_ramp = float(k_ramp_fraction) > 0.0
    if use_ramp:
        ramp_epochs = max(1, int(k_ramp_fraction * num_epochs))
        k_ramp_step = float(ramp_epochs) / float(K_max)

    for epoch in range(num_epochs):
        optimizer.zero_grad(set_to_none=True)
        if fixed_k:
            k = K_max
        elif use_ramp:
            k_max_cur = min(int(epoch / k_ramp_step) + 1, K_max)
            k = int(torch.randint(1, k_max_cur + 1, (1,), device=device).item())
        else:
            k = int(torch.randint(1, K_max + 1, (1,), device=device).item())

        # choose population index
        p0_idx = int(torch.randint(0, num_p0, (1,), device=device).item())

        # robustly extract x0 and future Y_1..Y_k with consistent NaN filtering
        x0, Y_future, t0, L, x_idx = get_margs(
            X_em_torch, time_grid,
            p0_idx=p0_idx,
            t_idx=0,
            n_sub=particles_per_batch,
            n_ahead=k,
            subsample_targets=subsample_targets,
        )
        # L should equal k unless you’re near the end, but at t_idx=0 it should be k.
        if L <= 0 or x0.numel() == 0:
            # nothing valid this epoch (all NaN for this pop / times); skip cleanly
            continue

        # initial velocity (pass x_idx so paired bundle velocities get subset identically)
        if vel_provider is not None:
            v0 = vel_provider(p0_idx, x0, t0_val, x_idx=x_idx)
        else:
            v0 = torch.zeros_like(x0)

        # rollout from t=0 -> t=k (microsteps)
        total_micro_steps = _microsteps_between(time_grid[0], time_grid[L], dt_base, substeps_per_dt)

        X_all_pred = integrator(
            x0=x0,
            v0=v0,
            accel=accel_train,
            dt=dt_train,
            steps=total_micro_steps,
            friction=resolve_gamma(friction_use),
            return_all=True,
            t_start=t0_val,
        )

        # loss across marginals m=1..L (mean so scale is comparable across varying k)
        per_step_losses = []
        for m in range(1, L + 1):
            t_m = time_grid[m]
            micro_idx = _microsteps_between(time_grid[0], t_m, dt_base, substeps_per_dt)
            micro_idx = min(micro_idx, total_micro_steps)

            x_pred_m = X_all_pred[:, micro_idx, :]
            x_gt_m = Y_future[m - 1]

            loss_m = compute_loss(
                x_pred_m, x_gt_m,
                loss_type=loss_type,
                blur=float(kernel_blur),
                p=geom_p,
                scaling=geom_scaling,
                debias=geom_debias,
                backend=geom_backend,
            )
            per_step_losses.append(loss_m)

        loss = torch.stack(per_step_losses, dim=0).mean()

        last_loss_val = float(loss.detach().item())
        loss_sum += last_loss_val
        loss_count += 1

        loss.backward()
        optimizer.step()

        if epoch_callback is not None:
            gamma_now = resolve_gamma(friction_use)
            fric_val = float(gamma_now.detach().item()) if isinstance(gamma_now, torch.Tensor) else float(gamma_now)
            epoch_callback(epoch, int(L), last_loss_val, fric_val)

        if debug or ((epoch + 1) % max(1, num_epochs // 10) == 0):
            gamma_now = resolve_gamma(friction_use)
            fric_val = float(gamma_now.detach().item()) if isinstance(gamma_now, torch.Tensor) else float(gamma_now)
            print(f"[{name}] Ep {epoch+1:04d}/{num_epochs} | k={int(L):02d} | loss={last_loss_val:.4e} | fric={fric_val:.4g}")

        del X_all_pred
        if torch.cuda.is_available() and (epoch + 1) % 25 == 0:
            torch.cuda.empty_cache()

    return {
        "train_loss_avg": loss_sum / max(1, loss_count),
        "train_loss_last": float(last_loss_val) if last_loss_val is not None else float("nan"),
        "num_effective_epochs": int(loss_count),
    }