# losses.py
import torch
from typing import Literal, Optional

Tensor = torch.Tensor


from geomloss import SamplesLoss
import torch.nn as nn
from typing import Any, Dict, Optional, Callable, List, Union, Tuple
from mechanics import pick_integrator
from potential_energy_models import make_accel_from_potential


_SINKHORN_CACHE = {}
_GAUSSIAN_CACHE = {}
_ENERGY_CACHE = {}


def _get_geomloss_sinkhorn(
    *,
    p: int,
    blur: float,
    scaling: float,
    debias: bool,
    backend: str,
):
    key = ("sinkhorn", p, float(blur), float(scaling), bool(debias), str(backend))
    obj = _SINKHORN_CACHE.get(key, None)
    if obj is None:
        obj = SamplesLoss(
            loss="sinkhorn",
            p=p,
            blur=blur,
            scaling=scaling,
            debias=debias,
            backend=backend,
        )
        _SINKHORN_CACHE[key] = obj
    return obj


def _get_geomloss_gaussian(*, blur: float, backend: str):
    key = ("gaussian", float(blur), str(backend))
    obj = _GAUSSIAN_CACHE.get(key, None)
    if obj is None:
        obj = SamplesLoss(
            loss="gaussian",
            blur=blur,
            backend=backend,
        )
        _GAUSSIAN_CACHE[key] = obj
    return obj


def _get_geomloss_energy(*, p: int, backend: str):
    """
    GeomLoss 'energy' corresponds to an energy distance / MMD-like loss.
    It does NOT use blur; it depends on p (typically 1 or 2).
    """
    key = ("energy", int(p), str(backend))
    obj = _ENERGY_CACHE.get(key, None)
    if obj is None:
        obj = SamplesLoss(
            loss="energy",
            p=int(p),
            backend=backend,
        )
        _ENERGY_CACHE[key] = obj
    return obj


def _default_geomloss_backend(x: Tensor, y: Tensor) -> str:
    d = x.shape[1]
    nmax = max(x.shape[0], y.shape[0])

    # RELAXED LIMIT: Allow d <= 10 or even higher
    if d <= 20 and nmax <= 6000:
        return "tensorized"

    return "online"


# =============================================================
# 3) Unified interface (ONE compute_loss)
# =============================================================
LossType = Literal[
    "mmd", "sw2", "sinkhorn",
    "geom_sinkhorn", "geom_gaussian", "geom_energy"
]




def compute_loss(
    x: Tensor,
    y: Tensor,
    loss_type: str = "geom_sinkhorn",
    *,
    p: int = 2,
    blur: float = 0.2,
    scaling: float = 0.9,
    debias: bool = True,
    backend: Optional[str] = None,
    eps: float = 1e-12,
) -> Tensor:
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError(f"x,y must be (N,d) and (M,d). Got {x.shape=} {y.shape=}")
    if x.shape[1] != y.shape[1]:
        raise ValueError(f"Dim mismatch: {x.shape[1]} vs {y.shape[1]}")

    if not loss_type.startswith("geom_"):
        raise ValueError(f"Unknown or unsupported loss_type for this patch: {loss_type}")

    if backend is None:
        backend = _default_geomloss_backend(x, y)

    if loss_type == "geom_sinkhorn":
        loss_fn = _get_geomloss_sinkhorn(p=p, blur=blur, scaling=scaling, debias=debias, backend=backend)
    elif loss_type == "geom_gaussian":
        loss_fn = _get_geomloss_gaussian(blur=blur, backend=backend)
    elif loss_type == "geom_energy":
        loss_fn = _get_geomloss_energy(p=p, backend=backend)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    # Valid rows
    x_valid = torch.isfinite(x).all(dim=1)
    y_valid = torch.isfinite(y).all(dim=1)

    if not x_valid.any():
        raise ValueError("compute_loss: all rows in x are NaN/Inf.")
    if not y_valid.any():
        raise ValueError("compute_loss: all rows in y are NaN/Inf.")

    x = x[x_valid]
    y = y[y_valid]

    # Uniform weights on filtered supports
    a = torch.full((x.shape[0],), 1.0 / (x.shape[0] + eps), dtype=x.dtype, device=x.device)
    b = torch.full((y.shape[0],), 1.0 / (y.shape[0] + eps), dtype=y.dtype, device=y.device)

    return loss_fn(a, x, b, y)

import ot
import numpy as np


def get_w1(M, w_x=None, w_y=None):
    def get_w(w, n):
        if w is None:
            w = np.ones(n)
        if isinstance(w, torch.Tensor):
            w = w.detach().cpu()
        w = np.array(w).astype(np.float64)
        w /= w.sum()
        return w

    if isinstance(M, torch.Tensor):
        M = M.detach().cpu()

    M = np.array(M).astype(np.float64)
    w_x, w_y = get_w(w_x, M.shape[0]), get_w(w_y, M.shape[1])
    return ot.emd2(w_x, w_y, M, numItermax=1e7)

from sklearn.metrics import pairwise_distances


@torch.no_grad()
def _resolve_v0(
        *,
        p0_idx: int,
        x0: torch.Tensor,
        t0: float,
        vel_provider,
        vel_mode: str,
        V_em: Optional[torch.Tensor],
        time_grid: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        idx: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Resolve initial velocity v0 for an integration segment.

    For vel_mode='bundle': pulls v from V_em[p0_idx, :, m, :] where m matches t0,
    optionally pre-filtered by `valid_mask` (NaN mask of x at this t) and then
    `idx` (subsample indices).
    """
    vel_mode_l = str(vel_mode).lower()
    if vel_provider is None or vel_mode_l == "zero":
        return torch.zeros_like(x0)

    if vel_mode_l == "bundle":
        if V_em is None:
            raise ValueError("vel_mode='bundle' requires V_em")
        m = int(torch.argmin((time_grid - float(t0)).abs()).item())
        m = max(0, min(m, time_grid.numel() - 1))
        v_full = V_em[int(p0_idx), :, m, :]
        if valid_mask is not None:
            v_full = v_full[valid_mask]
        if idx is not None:
            v_full = v_full[idx]
        return v_full.to(device=x0.device, dtype=x0.dtype)

    return vel_provider(p0_idx, x0, t0).detach()


def _resolve_dt_micro(
        dt_base: float,
        substeps_per_dt: int,
        dt_integration: Optional[float] = None,
) -> Tuple[float, int]:
    """
    Returns (dt_micro, steps_per_macro). If dt_integration is given and < dt_base,
    uses that; otherwise falls back to dt_base / substeps_per_dt.
    """
    if dt_integration is not None and 0 < dt_integration < dt_base:
        dt_micro = float(dt_integration)
        ratio = float(dt_base) / dt_micro
        steps_per_macro = int(round(ratio))
        if abs(ratio - steps_per_macro) > 1e-3:
            print(f"  [warn] dt_base ({dt_base}) is not an integer multiple of dt ({dt_micro}); drift possible.")
    else:
        dt_micro = float(dt_base) / int(substeps_per_dt)
        steps_per_macro = int(substeps_per_dt)
    return dt_micro, steps_per_macro



@torch.no_grad()
def _rollout_segment(
        *,
        model: torch.nn.Module,
        X_em: torch.Tensor,
        time_grid: torch.Tensor,
        p0_idx: int,
        start_idx: int,
        end_idx: int,
        integrator_name: str,
        dt_micro: float,
        friction: Any,
        vel_provider,
        vel_mode: str,
        V_em: Optional[torch.Tensor],
        max_force: Optional[float],
        t_start_zero: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Roll out the model from time_grid[start_idx] to time_grid[end_idx].

    Both endpoints are NaN-filtered independently (no cross-time particle identity).
    Returns (x_pred_clean, y_true_clean) — both (N, d) tensors after NaN filtering.
    """
    from mechanics import pick_integrator
    from potential_energy_models import make_accel_from_potential

    device = X_em.device

    # Start state (NaN-filtered)
    x_start_full = X_em[int(p0_idx), :, int(start_idx), :]
    valid_s = torch.isfinite(x_start_full).all(dim=-1)
    x0 = x_start_full[valid_s].to(device)

    # Initial velocity (filtered the same way for bundle mode)
    t_start_val = float(time_grid[int(start_idx)].item())
    v0 = _resolve_v0(
        p0_idx=p0_idx, x0=x0, t0=t_start_val,
        vel_provider=vel_provider, vel_mode=vel_mode, V_em=V_em,
        time_grid=time_grid, valid_mask=valid_s, idx=None,
    )

    # Ground truth at end (NaN-filtered)
    y_full = X_em[int(p0_idx), :, int(end_idx), :]
    valid_e = torch.isfinite(y_full).all(dim=-1)
    y_true = y_full[valid_e].to(device)

    # Integrate
    t_end_val = float(time_grid[int(end_idx)].item())
    n_steps = int(round((t_end_val - t_start_val) / dt_micro))
    if n_steps <= 0 or x0.numel() == 0:
        return x0, y_true

    integrator, _ = pick_integrator(str(integrator_name))
    accel_eval = make_accel_from_potential(model, create_graph=False, max_force=max_force)

    t_start_arg = 0.0 if t_start_zero else t_start_val
    out = integrator(
        x0=x0, v0=v0, accel=accel_eval, dt=dt_micro, steps=n_steps,
        friction=friction, return_all=False, t_start=t_start_arg,
    )
    x_pred = out[0] if isinstance(out, tuple) else out

    valid_p = torch.isfinite(x_pred).all(dim=-1)
    return x_pred[valid_p], y_true


def _w1_or_nan(x_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """W1 between two (N,d) point clouds, or NaN if either is empty."""
    if x_pred.shape[0] == 0 or y_true.shape[0] == 0:
        return float("nan")
    M = pairwise_distances(x_pred.detach().cpu().numpy(),
                           y_true.detach().cpu().numpy(),
                           metric='euclidean')
    return float(get_w1(M))


@torch.no_grad()
def evaluate_model_w1(
        *,
        model: torch.nn.Module,
        X_em: torch.Tensor,
        time_grid: torch.Tensor,
        dt_base: float,
        substeps_per_dt: int,
        integrator_name: str,
        friction: Any,
        vel_provider,
        vel_mode: str,
        V_em: Optional[torch.Tensor],
        max_force: Optional[float],
        particles_eval: Optional[int],  # currently informational; per-segment NaN filtering controls counts
        eval_mode: str = "forecast",
        n_train_marginals: Optional[int] = None,  # required for forecast
        holdout_indices: Optional[List[int]] = None,  # required for interpolate
        train_time_idx: Optional[List[int]] = None,  # required for interpolate
        dt_integration: Optional[float] = None,
        t_start_zero: bool = False,
        return_clouds: bool = False,
) -> Dict[str, Any]:
    """
    Compute W1 metrics for predicted vs ground-truth marginals.

    Modes
    -----
    forecast (n_train_marginals required):
        Roll out from time_grid[0] for the full horizon; compute W1 against every
        marginal m in [1, T_total). Marginals m < n_train_marginals are reported
        under 'train_*' keys, the rest under 'test_*' keys.

    interpolate (holdout_indices and train_time_idx required):
        For each h in holdout_indices, roll out from the previous training time
        (max(t < h for t in train_time_idx), default 0) to h, and compute W1.
        No 'train_*' metrics are produced.

    Returns
    -------
    Dict with at minimum:
        - eval_mode
        - test_w1_avg, test_w1_se, test_w1_std, test_count
        - eval_w1_h{idx}: per-holdout W1 (interpolate only)
        - eval_w1_mean, eval_w1_std: mean/std across holdouts (interpolate only)
        - train_w1_avg, train_w1_se, train_w1_std, train_count (forecast only)
        - clouds: list of (h_idx, x_pred_cpu, y_true_cpu) if return_clouds=True (interpolate)
    """
    num_p0, N, T_total_plus_1, _ = X_em.shape
    if num_p0 != 1:
        # All current experiments use num_p0=1; if you ever revisit multi-pop, expand here.
        print(f"  [warn] evaluate_model_w1: num_p0={num_p0}, evaluating only p0_idx=0.")
    p0_idx = 0

    dt_micro, steps_per_macro = _resolve_dt_micro(dt_base, substeps_per_dt, dt_integration)

    out: Dict[str, Any] = {"eval_mode": eval_mode, "eval_dt_micro": dt_micro}

    if eval_mode == "forecast":
        if n_train_marginals is None:
            raise ValueError("forecast mode requires n_train_marginals")

        train_ms = list(range(1, int(n_train_marginals)))
        test_ms = list(range(int(n_train_marginals), int(T_total_plus_1)))

        print(f"\n[W1 Eval - Forecast]")
        print(f"  Train marginals: {len(train_ms)}, Test marginals: {len(test_ms)}")
        print(f"  dt_micro: {dt_micro:.4f} (steps_per_macro={steps_per_macro})")

        # One long rollout from t=0 spanning the full horizon, then index by macro step.
        from mechanics import pick_integrator
        from potential_energy_models import make_accel_from_potential

        # Initial conditions at t=0 (NaN-filtered)
        x_start_full = X_em[p0_idx, :, 0, :]
        valid_s = torch.isfinite(x_start_full).all(dim=-1)
        x0 = x_start_full[valid_s].to(X_em.device)
        t0_val = float(time_grid[0].item())
        v0 = _resolve_v0(
            p0_idx=p0_idx, x0=x0, t0=t0_val,
            vel_provider=vel_provider, vel_mode=vel_mode, V_em=V_em,
            time_grid=time_grid, valid_mask=valid_s, idx=None,
        )

        steps_macro = int(T_total_plus_1) - 1
        total_micro = steps_macro * steps_per_macro
        integrator, _ = pick_integrator(str(integrator_name))
        accel_eval = make_accel_from_potential(model, create_graph=False, max_force=max_force)
        t_start_arg = 0.0 if t_start_zero else t0_val
        X_pred = integrator(
            x0=x0, v0=v0, accel=accel_eval, dt=dt_micro, steps=total_micro,
            friction=friction, return_all=True, t_start=t_start_arg,
        )
        # Index back to macro grid: position at marginal m is X_pred[:, m * steps_per_macro, :]
        macro_idx = (torch.arange(0, steps_macro + 1, device=x0.device) * steps_per_macro).long()
        macro_idx = torch.clamp(macro_idx, max=X_pred.shape[1] - 1)
        X_pred_macro = X_pred[:, macro_idx, :]

        # W1 per marginal — y_true is independently NaN-filtered at each m
        train_vals: List[float] = []
        test_vals: List[float] = []
        for m in (train_ms + test_ms):
            xp = X_pred_macro[:, m, :]
            xp = xp[torch.isfinite(xp).all(dim=-1)]

            y_full = X_em[p0_idx, :, m, :]
            yt = y_full[torch.isfinite(y_full).all(dim=-1)].to(X_em.device)

            w1 = _w1_or_nan(xp, yt)
            (train_vals if m in train_ms else test_vals).append(w1)

        _put_stats(out, "train", train_vals)
        _put_stats(out, "test", test_vals)

        print(f"  → Train W1: {out['train_w1_avg']:.6f} ± {out['train_w1_se']:.6f} (n={out['train_count']})")
        print(f"  → Test  W1: {out['test_w1_avg']:.6f} ± {out['test_w1_se']:.6f} (n={out['test_count']})")
        return out

    if eval_mode == "interpolate":
        if not holdout_indices:
            raise ValueError("interpolate mode requires non-empty holdout_indices")
        if not train_time_idx:
            raise ValueError("interpolate mode requires non-empty train_time_idx")

        print(f"\n[W1 Eval - Interpolate] holdouts={holdout_indices}")
        print(f"  dt_micro: {dt_micro:.4f}")

        test_vals: List[float] = []
        clouds: List[Tuple[int, torch.Tensor, torch.Tensor]] = []
        for h_idx in holdout_indices:
            prev = max([t for t in train_time_idx if t < int(h_idx)], default=0)
            x_pred, y_true = _rollout_segment(
                model=model, X_em=X_em, time_grid=time_grid,
                p0_idx=p0_idx, start_idx=prev, end_idx=int(h_idx),
                integrator_name=integrator_name, dt_micro=dt_micro,
                friction=friction, vel_provider=vel_provider,
                vel_mode=vel_mode, V_em=V_em, max_force=max_force,
                t_start_zero=t_start_zero,
            )
            w1 = _w1_or_nan(x_pred, y_true)
            out[f"eval_w1_h{int(h_idx)}"] = w1
            test_vals.append(w1)
            if return_clouds:
                clouds.append((int(h_idx), x_pred.detach().cpu(), y_true.detach().cpu()))
            print(f"  Holdout {int(h_idx)} (t={time_grid[int(h_idx)].item():.2f}): "
                  f"W1={w1:.4f}, start_idx={prev}")

        _put_stats(out, "test", test_vals)
        valid = [w for w in test_vals if not np.isnan(w)]
        out["eval_w1_mean"] = float(np.mean(valid)) if valid else float("nan")
        out["eval_w1_std"] = float(np.std(valid)) if valid else float("nan")
        print(f"  → Mean W1: {out['eval_w1_mean']:.4f} ± {out['eval_w1_std']:.4f}")

        if return_clouds:
            out["clouds"] = clouds
        return out

    raise ValueError(f"Unknown eval_mode={eval_mode!r}; expected 'forecast' or 'interpolate'.")


def _put_stats(out: Dict[str, Any], prefix: str, vals: List[float]) -> None:
    """Populate {prefix}_w1_avg/se/std/count in `out`."""
    if vals:
        arr = np.asarray(vals, dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size > 0:
            out[f"{prefix}_w1_avg"] = float(np.mean(finite))
            out[f"{prefix}_w1_std"] = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
            out[f"{prefix}_w1_se"] = (out[f"{prefix}_w1_std"] / float(np.sqrt(finite.size))) if finite.size > 1 else 0.0
            out[f"{prefix}_count"] = int(finite.size)
            return
    out[f"{prefix}_w1_avg"] = float("nan")
    out[f"{prefix}_w1_std"] = float("nan")
    out[f"{prefix}_w1_se"] = float("nan")
    out[f"{prefix}_count"] = 0


