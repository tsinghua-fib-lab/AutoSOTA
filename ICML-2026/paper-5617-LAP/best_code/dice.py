import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd


class ScalarScoreMLP(nn.Module):
    """
    Approximates the velocity potential s(x, t).
    v_t(x) = ∇_x s(x, t). Output is scalar.
    """

    def __init__(self, d: int = 2, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        # CRITICAL STABILITY FIX
        # Initialize last layer to near zero to prevent massive gradients at step 0.
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=1e-5)
        nn.init.constant_(self.net[-1].bias, 0.0)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        if isinstance(t, float):
            t = torch.full((x.shape[0], 1), t, device=x.device, dtype=x.dtype)
        elif t.ndim == 1:
            t = t.unsqueeze(1)
        inp = torch.cat([x, t.to(dtype=x.dtype)], dim=1)
        return self.net(inp)


def get_s_derivatives(model: nn.Module, x: torch.Tensor, t: torch.Tensor):
    """Computes s, ∇_x s, and ∂_t s via autograd."""
    x = x.requires_grad_(True)
    t = t.requires_grad_(True)

    s = model(x, t)  # (B,1)
    grad_outputs = torch.ones_like(s)
    grad_x, grad_t = autograd.grad(s, [x, t], grad_outputs=grad_outputs, create_graph=True)

    return s, grad_x, grad_t


def masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """
    Computes mean of tensor along dim, ignoring elements where mask is False.
    tensor: (A, B)
    mask:   (A, B) (bool or float 0/1)
    Returns: (A,)
    """
    mask_f = mask.float()
    # Avoid division by zero by clamping count to 1.0; if count is 0, sum is 0 anyway.
    return (tensor * mask_f).sum(dim=dim) / mask_f.sum(dim=dim).clamp(min=1.0)


def dice_loss(model, X_data, time_grid, batch_size_t=32, batch_size_x=256):
    """
    PyTorch implementation of the DICE loss, robust to NaN padding in X_data.
    X_data: (N_samples, T_steps, d) - may contain NaNs
    time_grid: (T_steps,)
    """
    device = X_data.device
    N, T, d = X_data.shape

    # 1. Sample Random Time Grid (t_q)
    # Pick random indices, always include 0 and T-1
    t_indices = torch.randperm(T - 2, device=device)[:batch_size_t] + 1
    t_indices = torch.cat([torch.tensor([0], device=device), t_indices, torch.tensor([T - 1], device=device)])
    t_indices, _ = torch.sort(t_indices)

    t_vals = time_grid[t_indices]  # (K,)

    # Weights w_q for the integral \int ||grad s||^2 dt
    dt_vec = t_vals[1:] - t_vals[:-1]
    w_q = 0.5 * torch.cat([dt_vec[:1], dt_vec[:-1] + dt_vec[1:], dt_vec[-1:]])

    # 2. Sample Particles
    x_indices = torch.randint(0, N, (batch_size_x,), device=device)
    X_batch = X_data[:, t_indices, :][x_indices, :, :].transpose(0, 1)  # -> (K, B_x, d)

    # 3. Create Validity Mask & Safe Input
    # EB data has NaN padding. We must:
    #   a) Identify valid particles.
    #   b) Replace NaNs with 0.0 for the forward pass (to avoid NaN propagation).
    #   c) Mask the loss averaging.
    mask = torch.isfinite(X_batch).all(dim=-1)  # (K, B_x)
    X_batch_safe = torch.nan_to_num(X_batch, nan=0.0)

    # Prepare batch sizes
    K = len(t_vals)
    B_x = batch_size_x

    # --- Term 1 (Transport) ---
    # 0.5 * sum [ E_{t_i}[s(t_{i+1})] - E_{t_{i+1}}[s(t_i)] + E_{t_i}[s(t_i)] - E_{t_{i+1}}[s(t_{i+1})] ]

    t_curr = t_vals[:-1]
    t_next = t_vals[1:]

    # Particles at t_i (curr) and t_{i+1} (next)
    x_curr = X_batch_safe[:-1].reshape(-1, d)  # ( (K-1)*B_x, d )
    x_next = X_batch_safe[1:].reshape(-1, d)  # ( (K-1)*B_x, d )

    # Corresponding masks
    mask_curr = mask[:-1].reshape(-1, B_x)  # (K-1, B_x)
    mask_next = mask[1:].reshape(-1, B_x)  # (K-1, B_x)

    # Expand times
    t_curr_expanded = t_curr.repeat_interleave(B_x).unsqueeze(1)
    t_next_expanded = t_next.repeat_interleave(B_x).unsqueeze(1)

    # Evaluate s four ways.
    # Note: reshape back to (K-1, B_x) to apply masked_mean per time step.

    # 1. s(x_i, t_{i+1}) -> Depends on x_i existence (mask_curr)
    s_xi_ti1 = model(x_curr, t_next_expanded).view(-1, B_x)
    E_s_xi_ti1 = masked_mean(s_xi_ti1, mask_curr, dim=1)

    # 2. s(x_{i+1}, t_i) -> Depends on x_{i+1} existence (mask_next)
    s_xi1_ti = model(x_next, t_curr_expanded).view(-1, B_x)
    E_s_xi1_ti = masked_mean(s_xi1_ti, mask_next, dim=1)

    # 3. s(x_i, t_i) -> Depends on x_i existence (mask_curr)
    s_xi_ti = model(x_curr, t_curr_expanded).view(-1, B_x)
    E_s_xi_ti = masked_mean(s_xi_ti, mask_curr, dim=1)

    # 4. s(x_{i+1}, t_{i+1}) -> Depends on x_{i+1} existence (mask_next)
    s_xi1_ti1 = model(x_next, t_next_expanded).view(-1, B_x)
    E_s_xi1_ti1 = masked_mean(s_xi1_ti1, mask_next, dim=1)

    loss_transport = torch.sum(0.5 * E_s_xi_ti1 - 0.5 * E_s_xi1_ti + 0.5 * E_s_xi_ti - 0.5 * E_s_xi1_ti1)

    # --- Term 2 (Regularization/Action) ---
    # 0.5 * \int E[ ||\nabla s||^2 ] dt

    flat_t_all = t_vals.repeat_interleave(B_x).unsqueeze(1)
    flat_x_all = X_batch_safe.reshape(-1, d)
    flat_mask_all = mask.reshape(-1, B_x)  # (K, B_x) for final averaging

    # Enable grad
    flat_x_all.requires_grad_(True)
    flat_t_all.requires_grad_(True)

    s_all = model(flat_x_all, flat_t_all)

    # We sum s_all to compute gradients. Even if inputs were dummy 0.0s,
    # we get valid gradients (finite numbers). We will mask them out later.
    grad_s_all = autograd.grad(s_all.sum(), flat_x_all, create_graph=True)[0]

    # (K, B_x)
    grad_norm_sq = (grad_s_all ** 2).sum(dim=1).view(K, B_x)

    # Average over valid particles only
    E_grad_norm_sq = masked_mean(grad_norm_sq, flat_mask_all, dim=1)

    loss_action = 0.5 * torch.sum(w_q * E_grad_norm_sq)

    return loss_transport + loss_action


@dataclass
class DiceConfig:
    steps: int = 10000
    lr: float = 1e-3
    lr_end: float = 1e-5
    clip_norm: float = 1.0
    hidden: int = 128
    batch_size_t: int = 32
    batch_size_x: int = 256
    log_every: int = 500


def _now_str() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def _atomic_save(obj: Any, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def _euler_ode_rollout(
        x0: torch.Tensor,
        drift_fn,
        dt: float,
        steps: int,
        t0: float = 0.0,
) -> torch.Tensor:
    """Deterministic Euler rollout."""
    xs = [x0]
    x = x0
    t = float(t0)
    for _ in range(int(steps)):
        x = x + float(dt) * drift_fn(x, t)
        t += float(dt)
        xs.append(x)
    return torch.stack(xs, dim=1)


def get_ensemble_velocity(dice_models, pop_idx: int, x: torch.Tensor, t: float) -> torch.Tensor:
    m = dice_models[int(pop_idx)]
    m.eval()
    with torch.enable_grad():
        x_req = x.detach().requires_grad_(True)
        t_req = torch.full((x.shape[0], 1), float(t), device=x.device, dtype=x.dtype).requires_grad_(True)
        s = m(x_req, t_req)
        grad_x = autograd.grad(outputs=s.sum(), inputs=x_req, create_graph=False, retain_graph=False)[0]
    return grad_x.detach()


def train_dice_models(
        *,
        X_em_torch: torch.Tensor,
        time_grid: torch.Tensor,
        d: int,
        device: torch.device,
        cfg: Optional[DiceConfig] = None,
        wandb_run=None,
        log_prefix: str = "",
) -> Tuple[List[nn.Module], Dict[str, Any]]:
    if cfg is None:
        cfg = DiceConfig()

    X = X_em_torch.to(device=device, dtype=torch.float32)
    tg = time_grid.to(device=device, dtype=torch.float32)

    num_p0, N, T, dd = X.shape
    assert dd == d, f"Expected d={d}, got {dd}"

    models: List[nn.Module] = []
    info: Dict[str, Any] = {"dice_cfg": asdict(cfg), "num_p0": int(num_p0), "N": int(N), "T": int(T)}

    for p in range(num_p0):
        model = ScalarScoreMLP(d=d, hidden=int(cfg.hidden)).to(device)
        opt = optim.Adam(model.parameters(), lr=float(cfg.lr))
        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(cfg.steps), eta_min=float(cfg.lr_end))

        model.train()
        t0 = time.time()

        for step in range(int(cfg.steps)):
            opt.zero_grad(set_to_none=True)
            loss = dice_loss(
                model=model,
                X_data=X[p],
                time_grid=tg,
                batch_size_t=int(cfg.batch_size_t),
                batch_size_x=int(cfg.batch_size_x),
            )

            # Safety check for NaNs in loss before backward
            if not torch.isfinite(loss):
                print(f"[DICE{log_prefix}] Warning: Loss is NaN/Inf at step {step}. Skipping update.")
                opt.zero_grad()  # Clear grads
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.clip_norm))
            opt.step()
            scheduler.step()

            if (step % int(cfg.log_every) == 0) or (step == int(cfg.steps) - 1):
                wall = time.time() - t0
                current_lr = scheduler.get_last_lr()[0]
                msg = f"[DICE{log_prefix}] pop={p} step={step:06d}/{int(cfg.steps) - 1:06d} loss={loss.item():.6e} lr={current_lr:.2e} wall_s={wall:.1f}"
                print(msg)
                if wandb_run is not None:
                    try:
                        import wandb
                        wandb.log({
                            f"dice/loss_pop{p}": float(loss.item()),
                            f"dice/lr_pop{p}": float(current_lr)
                        }, step=int(step))
                    except Exception:
                        pass

        model.eval()
        models.append(model)

    return models, info


def save_dice_bundle(*, path: str, dice_models: List[nn.Module], cfg: DiceConfig,
                     extra: Optional[Dict[str, Any]] = None) -> None:
    out = {
        "dice_state_dicts": [m.state_dict() for m in dice_models],
        "dice_cfg": asdict(cfg),
        "extra": (extra or {}),
    }
    _atomic_save(out, Path(path))


def load_dice_bundle(*, path: str, device: torch.device, d: int, hidden: int) -> Tuple[List[nn.Module], Dict[str, Any]]:
    payload = torch.load(path, map_location="cpu")
    sds = payload["dice_state_dicts"]
    cfg = payload.get("dice_cfg", {})
    extra = payload.get("extra", {})

    models: List[nn.Module] = []
    for sd in sds:
        m = ScalarScoreMLP(d=d, hidden=int(hidden)).to(device)
        m.load_state_dict(sd)
        m.eval()
        models.append(m)

    info = {"dice_cfg": cfg, "extra": extra, "num_models": len(models)}
    return models, info


def train_or_load_dice_bundle(
        *,
        bundle_path: Path,
        X_em_torch: torch.Tensor,
        time_grid: torch.Tensor,
        d: int,
        hidden: int,
        device: torch.device,
        wandb_run=None,
        log_prefix: str = "",
        lr: float = 1e-3,
        lr_end: float = 1e-5,
        clip_norm: float = 1.0,
        steps: int = 10000,
        batch_size_t: int = 0,
        batch_size_x: int = 128,
) -> Tuple[List[nn.Module], Dict[str, Any]]:
    bundle_path.parent.mkdir(parents=True, exist_ok=True)

    if bundle_path.exists():
        models, info = load_dice_bundle(path=str(bundle_path), device=device, d=d, hidden=hidden)
        info["source"] = "cache"
        print(f"[DICE{log_prefix}] loaded cached bundle: {bundle_path}")
        return models, info

    cfg = DiceConfig(hidden=int(hidden), lr=lr, lr_end=lr_end, clip_norm=clip_norm, steps=steps)
    cfg.batch_size_t = int(batch_size_t) if int(batch_size_t) > 0 else int(time_grid.numel())
    cfg.batch_size_x = int(batch_size_x)

    models, info = train_dice_models(
        X_em_torch=X_em_torch,
        time_grid=time_grid,
        d=d,
        device=device,
        cfg=cfg,
        wandb_run=wandb_run,
        log_prefix=log_prefix,
    )
    save_dice_bundle(path=str(bundle_path), dice_models=models, cfg=cfg, extra=info)
    info["source"] = "trained"
    info["bundle_path"] = str(bundle_path)
    print(f"[DICE{log_prefix}] saved bundle: {bundle_path}")
    return models, info


def maybe_make_dice_diagnostic_gif(
        *,
        save_path: str,
        X_em: torch.Tensor,
        time_grid: torch.Tensor,
        dice_models: List[nn.Module],
        pop_idx: int = 0,
        wandb_run=None,
        wandb_step: int = 0,
) -> None:
    try:
        from plot_utils import make_compare_gif
        has_gif = True
    except Exception:
        has_gif = False
        make_compare_gif = None

    if not has_gif:
        return

    device = X_em.device
    X_ref = X_em[int(pop_idx)].detach()
    tg = time_grid.detach()
    dt = float((tg[1] - tg[0]).item()) if tg.numel() >= 2 else 1.0
    steps = int(X_ref.shape[1] - 1)

    # 1. Get initial positions
    x0 = X_ref[:, 0, :]

    # 2. Filter out NaNs from the start
    #    (EB data often has varying particle counts, padded with NaN)
    mask_start = torch.isfinite(x0).all(dim=1)
    if not mask_start.any():
        return

    # Apply mask to both x0 AND X_ref so shapes match
    x0 = x0[mask_start]
    X_ref_filtered = X_ref[mask_start]

    def drift_fn(x, t):
        return get_ensemble_velocity(dice_models, pop_idx=int(pop_idx), x=x, t=float(t))

    # 3. Rollout using filtered valid particles
    X_pred = _euler_ode_rollout(x0=x0, drift_fn=drift_fn, dt=dt, steps=steps, t0=float(tg[0].item()))

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # 4. Pass filtered reference to plotter
    make_compare_gif(
        X_true=X_ref_filtered.detach().cpu(),
        X_learned=X_pred.detach().cpu(),
        dt=float(dt),
        save_path=str(save_path),
        frame_skip=1,
        always_show=False,
        subsample=1000
    )

    if wandb_run is not None:
        try:
            import wandb
            wandb.log({"dice/diagnostic_gif": wandb.Video(str(save_path), fps=5, format="gif")}, step=int(wandb_step))
        except Exception:
            pass