#!/usr/bin/env python3
"""
train.py

Train model to learn WLM from population dynamics
Data for training and inference is generated or loaded from data_generator.py.

Evaluation modes:
  - forecast: Train on first portion of data, evaluate on held-out future (default)
  - interpolate: Hold out a set of marginals, train on rest, evaluate on held-out indices

Runs will save in outdir/run_name/:
  - config.json (the parameters used for the run)
  - metrics.jsonl (W1 metrics for unseen marginals)
  - Optionally save model weights as ckpt_stepXXXXXXX.pt (periodic save) and final.pt
  - Optionally save visualization of model predictions vs true data as compare_stepXXXXXXX.gif
"""
import torch
import torch.nn as nn
import argparse
import matplotlib.pyplot as plt
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, List
from functools import partial


import wandb

from rollout import train_rollout_anchor_p0_randk, LearnableFriction
from mechanics import build_vel_provider
from potential_energy_models import build_model_and_kwargs, make_accel_from_potential

from dice import maybe_make_dice_diagnostic_gif, train_or_load_dice_bundle
from parse_save_helpers import (
    _parse_args_with_config, get_device, set_seed, ensure_dir, print_velocity_diagnostics,
    dump_json, append_jsonl, _find_complex, _wandb_sanitize, load_dice_models, now_str,
    TrainCallbackState, make_epoch_callback,
    save_ckpt as save_ckpt_impl,
    do_eval as do_eval_impl,
)
from plot_utils import plot_single_holdout_scatter, plot_multi_holdout_scatter, maybe_gif as maybe_gif_impl



# ============================================================
# EMA (Exponential Moving Average)
# ============================================================
class EMA:
    """
    Lightweight EMA over a potential energy model's state_dict (and optionally a friction module).
    """

    def __init__(
            self,
            model: nn.Module,
            friction_module: Optional[nn.Module] = None,
            decay: float = 0.999,
    ):
        self.decay = float(decay)
        self.step = 0
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}
        # Empty shadow if friction_module is None — tracks_friction is False then
        self.friction_shadow: Dict[str, torch.Tensor] = (
            {k: v.clone().detach() for k, v in friction_module.state_dict().items()}
            if friction_module is not None else {}
        )

    @property
    def tracks_friction(self) -> bool:
        return len(self.friction_shadow) > 0

    @staticmethod
    @torch.no_grad()
    def _ema_update_dict(shadow: Dict[str, torch.Tensor], src_state: Dict[str, torch.Tensor], decay: float) -> None:
        for k, v in src_state.items():
            if k not in shadow:
                continue
            if v.dtype.is_floating_point:
                shadow[k].mul_(decay).add_(v, alpha=1.0 - decay)
            else:
                shadow[k].copy_(v)

    @torch.no_grad()
    def update(self, model: nn.Module, friction_module: Optional[nn.Module] = None):
        """Update EMA shadows. Call after optimizer.step()."""
        self.step += 1
        self._ema_update_dict(self.shadow, model.state_dict(), self.decay)
        if self.tracks_friction and friction_module is not None:
            self._ema_update_dict(self.friction_shadow, friction_module.state_dict(), self.decay)

    def apply(self, model: nn.Module, friction_module: Optional[nn.Module] = None):
        """Context manager: temporarily swap in EMA weights for eval."""
        return _EMAApplyContext(self, model, friction_module)

    def state_dict(self) -> dict:
        return {
            'step': self.step,
            'decay': self.decay,
            'shadow': self.shadow,
            'friction_shadow': self.friction_shadow,
        }

    def load_state_dict(self, state_dict: dict):
        self.step = state_dict['step']
        self.decay = state_dict.get('decay', self.decay)
        self.shadow = state_dict['shadow']
        self.friction_shadow = state_dict.get('friction_shadow', {})


class _EMAApplyContext:
    """Restore-on-exit context manager for EMA.apply()."""

    def __init__(self, ema: EMA, model: nn.Module, friction_module: Optional[nn.Module]):
        self.ema = ema
        self.model = model
        self.friction_module = friction_module
        self._model_backup: Dict[str, torch.Tensor] = {}
        self._friction_backup: Dict[str, torch.Tensor] = {}

    def __enter__(self):
        self._model_backup = {k: v.clone() for k, v in self.model.state_dict().items()}
        self.model.load_state_dict(self.ema.shadow)
        if self.ema.tracks_friction and self.friction_module is not None:
            self._friction_backup = {k: v.clone() for k, v in self.friction_module.state_dict().items()}
            self.friction_module.load_state_dict(self.ema.friction_shadow)
        return self

    def __exit__(self, *exc):
        self.model.load_state_dict(self._model_backup)
        if self._friction_backup and self.friction_module is not None:
            self.friction_module.load_state_dict(self._friction_backup)
        return False

# ============================================================
# Train: test splits for forecast and interpolation
# ============================================================

def partition_data_forecast(
        X_em: torch.Tensor,
        V_em: Optional[torch.Tensor],
        time_grid: torch.Tensor,
        train_fraction: float,
) -> Dict[str, Any]:
    """
    Partition data for forecast mode: first portion for training, rest for eval.

    Returns dict with:
        X_train, V_train, t_train: Training data (contiguous first portion)
        X_full, V_full, time_grid: Full data for evaluation
        n_train_marginals: Number of training time steps
        max_train_steps: Maximum training horizon
        mode_info: Dict with partition details
    """
    num_p0, N, T_plus_1, d = X_em.shape

    # Compute training horizon
    n_train_marginals = max(2, int(round(T_plus_1 * train_fraction))) # need at least 2 marginals for training
    n_train_marginals = min(n_train_marginals, T_plus_1)  # Cap at full data
    max_train_steps = n_train_marginals - 1

    # Slice training data
    X_train = X_em[:, :, :n_train_marginals, :].contiguous().clone()
    t_train = time_grid[:n_train_marginals].contiguous().clone()

    V_train = None
    if V_em is not None:
        V_train = V_em[:, :, :n_train_marginals, :].contiguous().clone()

    mode_info = {
        "eval_mode": "forecast",
        "train_fraction": train_fraction,
        "n_train_marginals": n_train_marginals,
        "T_total": T_plus_1,
        "train_times": t_train.cpu().tolist(),
    }

    print(f"[Forecast] Train on t=[0..{n_train_marginals - 1}] ({train_fraction * 100:.0f}%), "
          f"eval on t=[{n_train_marginals - 1}..{T_plus_1 - 1}]")

    return {
        "X_train": X_train,
        "V_train": V_train,
        "t_train": t_train,
        "X_full": X_em,
        "V_full": V_em,
        "time_grid": time_grid,
        "n_train_marginals": n_train_marginals,
        "max_train_steps": max_train_steps,
        "mode_info": mode_info,
    }


def partition_data_interpolate(
        X_em: torch.Tensor,
        V_em: Optional[torch.Tensor],
        time_grid: torch.Tensor,
        holdout_indices: List[int],  # Changed from single int to list
) -> Dict[str, Any]:
    """
    Partition data for interpolate mode: hold out marginals at holdout indices.

    Args:
        holdout_indices: List of indices to hold out (e.g., [1, 3, 5, 7])

    Returns dict with:
        X_train, V_train, t_train: Training data (excluding holdouts)
        X_full, V_full, time_grid: Full data for evaluation
        holdout_indices: List of held-out marginal indices
        n_train_marginals: Number of training time steps
        max_train_steps: Maximum training horizon
        mode_info: Dict with partition details
    """
    num_p0, N, T_plus_1, d = X_em.shape

    holdout_indices = sorted(set(int(h) for h in holdout_indices))
    for h in holdout_indices:
        if h <= 0 or h >= T_plus_1:
            raise ValueError(f"holdout index must be in [1, {T_plus_1 - 1}], got {h}")

    # Training indices: all except holdouts
    train_time_idx = [i for i in range(T_plus_1) if i not in holdout_indices]

    # Slice training data
    X_train = X_em[:, :, train_time_idx, :].contiguous().clone()
    t_train = time_grid[train_time_idx].contiguous().clone()

    V_train = None
    if V_em is not None:
        V_train = V_em[:, :, train_time_idx, :].contiguous().clone()

    n_train_marginals = len(train_time_idx)
    max_train_steps = n_train_marginals - 1

    holdout_times = [float(time_grid[h].item()) for h in holdout_indices]

    mode_info = {
        "eval_mode": "interpolate",
        "holdout_indices": holdout_indices,
        "holdout_times": holdout_times,
        "n_train_marginals": n_train_marginals,
        "T_total": T_plus_1,
        "train_time_idx": train_time_idx,
        "train_times": t_train.cpu().tolist(),
    }

    print(f"[Interpolate] Holdout indices: {holdout_indices}")
    print(f"[Interpolate] Holdout times: {holdout_times}")
    print(f"[Interpolate] Train on {n_train_marginals} marginals: {train_time_idx}")

    return {
        "X_train": X_train,
        "V_train": V_train,
        "t_train": t_train,
        "X_full": X_em,
        "V_full": V_em,
        "time_grid": time_grid,
        "holdout_indices": holdout_indices,
        "train_time_idx": train_time_idx,
        "n_train_marginals": n_train_marginals,
        "max_train_steps": max_train_steps,
        "mode_info": mode_info,
    }


# ============================================================
# CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("train.py")

    # Config for WLM model
    p.add_argument("--config", type=str, default=None, help="YAML/JSON config file")
    p.add_argument("--set", action="append", default=None,
                   help="Override config with dot-keys, e.g. --set arch.attn_heads=2")

    # IO
    p.add_argument("--data", type=str, required=False, default=None, help="Path to .pt bundle from data_generator.py")
    p.add_argument("--outdir", type=str, required=False, default=None, help="Base output directory")
    p.add_argument("--run-name", type=str, default="", help="Optional run name; default timestamp")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])

    # Evaluation mode: forecast vs interpolate
    p.add_argument("--eval-mode", type=str, default="forecast",
                   choices=["forecast", "interpolate"],
                   help="forecast: train on first portion, eval on future. "
                        "interpolate: hold out middle marginal, train on rest.")
    p.add_argument("--train-fraction", type=float, default=0.5,
                   help="Fraction of data for training in forecast mode (default: 0.5)")
    p.add_argument("--holdout-marginals", type=int, nargs="+", default=None,
                   help="Marginal indices to hold out in interpolate mode (e.g., 1 3 5 7)")

    # Rollout knobs
    p.add_argument("--num-epochs", type=int, default=10000)
    p.add_argument("--k-ramp-fraction", type=float, default=0.0,
                   help="Fraction of epochs over which k_max ramps 1->K_max. 0 disables (default).")

    p.add_argument("--dt-base", type=float, default=None, help="Overrides dt from bundle meta if set")
    p.add_argument("--substeps-per-dt", type=int, default=1) # for rollouts during training
    p.add_argument("--max-train-steps", type=int, default=None,
                   help="Max training steps. If None, auto-calibrates based on eval mode.")
    p.add_argument("--integrator", type=str, default="v", choices=["v", "x", "euler"])
    p.add_argument("--particles-per-batch", type=int, default=None) # if not None, this is the size of mini batches
    p.add_argument("--subsample-targets", action=argparse.BooleanOptionalAction, default=True,
                   help="If set, target marginals are subsampled to match particles_per_batch "
                        "rows each (matched to x0).")
    p.add_argument("--dt-eval", type=float, default=0.2, help="dt used for evaluating model on test")


    # Loss
    p.add_argument("--loss-type", type=str, default="geom_sinkhorn",
                   choices=["mmd", "sw2", "sinkhorn", "geom_sinkhorn", "geom_gaussian", "geom_energy"])
    p.add_argument("--kernel-blur", type=float, default=None,
                   help="GeomLoss: blur. If omitted, uses bundle blur.")
    p.add_argument("--geom-p", type=int, default=2)
    p.add_argument("--geom-scaling", type=float, default=0.9)
    p.add_argument("--geom-debias", action="store_true", default=True)
    p.add_argument("--geom-backend", type=str, default=None)

    # Velocity
    p.add_argument("--vel", type=str, default='bundle',
                   choices=["bundle", "zero", "dice"],
                   help="Initial velocity mode.")
    # optional toggle to always set the model's internal clock at 0 when starting any rollout (always False for main experiments)
    p.add_argument("--eval-t-start-zero", action="store_true", default=False,
                   help="Force t_start=0.0 in eval leapfrog (ignore physical start time)")

    # if estimating the initial velocity with DICE, initialize the model architecture
    p.add_argument("--dice-hidden", type=int, default=128)


    # Friction
    p.add_argument("--friction", type=float, default=0.0, help="Used if not learnable (or as init if learnable)")
    p.add_argument("--learnable-friction", action="store_true")
    p.add_argument("--friction-lr", type=float, default=1e-2)
    p.add_argument("--friction-freeze-epochs", type=int, default=0,
                   help="Freeze friction for first N epochs (0=disabled)")

    # Force clamp
    p.add_argument("--max-force", type=float, default=None)

    # Architecture for potential energy functional
    p.add_argument("--arch", type=str, default="attn_flow",
                   choices=["attn_flow"])
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4,
                   help="Weight decay coefficient for AdamW (default: 1e-4)")
    p.add_argument("--lr-schedule", type=str, default="constant",
                   choices=["constant", "cosine", "wsd"],
                   help="Learning rate schedule")
    p.add_argument("--lr-min", type=float, default=1e-6,
                   help="Minimum LR for cosine schedule")
    p.add_argument("--lr-warmup", type=int, default=0,
                   help="Linear warmup steps")
    p.add_argument("--lr-plateau-end", type=int, default=6000,
                   help="End epoch for LR plateau in WSD schedule")

    # hparams for attn_flow model
    p.add_argument("--attn-hidden-dim", type=int, default=32)
    p.add_argument("--attn-layers", type=int, default=4)
    p.add_argument("--attn-heads", type=int, default=1)
    p.add_argument("--use-time", action="store_true", default=False)
    p.add_argument("--d-time", type=int, default=16)
    p.add_argument("--ff-dim", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--use-spectral-norm", action="store_true", default=False,
                   help="Apply spectral normalization to FFN layers")

    # Save/eval/gif frequencies for logging
    p.add_argument("--ckpt-every", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--gif-every", type=int, default=0)
    # for visualizing rollouts of WLM
    p.add_argument("--particles-gif", type=int, default=1000)
    p.add_argument("--gif-frame-skip", type=int, default=5)
    p.add_argument("--gif-fps", type=int, default=5)

    p.add_argument("--particles-eval", type=int, default=None)
    p.add_argument("--gif-p0-idx", type=int, default=0)

    # W&B
    p.set_defaults(wandb=False)
    p.add_argument("--no-wandb", dest="wandb", action="store_false")
    p.add_argument("--wandb-project", type=str, default="WLM")
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb-name", type=str, default="")
    p.add_argument("--wandb-tags", type=str, default="")

    # EMA
    p.add_argument("--use-ema", action="store_true", default=False,
                   help="Use exponential moving average for model weights")
    p.add_argument("--ema-decay", type=float, default=0.999,
                   help="EMA decay rate (higher = slower/more stable)")
    p.add_argument("--ema-friction", action="store_true", default=False,
                   help="Also apply EMA to friction parameter (default: False)")

    return p


def main() -> None:
    parser = build_parser()
    args = _parse_args_with_config(parser)

    if args.data is None or args.outdir is None:
        raise SystemExit("train.py: --data and --outdir are required.")

    device = get_device(args.device)
    set_seed(int(args.seed))

    # ---- Load data ----
    bundle = torch.load(args.data, map_location="cpu", weights_only=False)
    X_em = bundle["X_em_torch"].to(device=device, dtype=torch.float32)
    time_grid = bundle["time_grid"].to(device=device, dtype=torch.float32)
    V_em = bundle.get("V_em_torch", None)

    # Sanitize position and velocity for consistency
    if V_em is not None:
        V_em = V_em.to(device=device, dtype=torch.float32)
        valid_x = torch.isfinite(X_em).all(dim=-1)
        valid_v = torch.isfinite(V_em).all(dim=-1)
        valid_common = valid_x & valid_v
        nan_t = torch.tensor(float('nan'), device=device, dtype=X_em.dtype)
        mask_expand = valid_common.unsqueeze(-1)
        X_em = torch.where(mask_expand, X_em, nan_t)
        V_em = torch.where(mask_expand, V_em, nan_t)
        print(f"[Data] Enforced X/V consistency. Kept {valid_common.sum().item()} valid pairs.")

    meta = bundle.get("meta", {})
    blur = float(bundle.get("blur", meta.get("blur", 0.2)))
    num_p0, N, T_plus_1, d = X_em.shape
    print(f"Loaded: {num_p0} populations, {T_plus_1} marginals, {N} samples each, d={d}")

    # ---- Partition training/test data based on whether the task is forecast or interpolation
    eval_mode = str(args.eval_mode).lower()
    if eval_mode == "interpolate":
        if args.holdout_marginals is None:
            raise SystemExit("--eval-mode=interpolate requires --holdout-marginals")
        holdout_indices = [int(h) for h in args.holdout_marginals]
        partition = partition_data_interpolate(X_em, V_em, time_grid, holdout_indices)
    else:  # forecast
        train_fraction = float(args.train_fraction)
        partition = partition_data_forecast(X_em, V_em, time_grid, train_fraction)

    X_train = partition["X_train"]
    V_train = partition["V_train"]
    t_train = partition["t_train"]
    n_train_marginals = partition["n_train_marginals"]
    max_train_steps = partition["max_train_steps"]
    mode_info = partition["mode_info"]

    # Override max_train_steps if specified
    if args.max_train_steps is not None:
        max_train_steps = min(int(args.max_train_steps), max_train_steps)

    # ---- Resolved dt / kernel bw ----
    if args.dt_base is not None:
        dt_base = float(args.dt_base)
    else:
        dt_base = float(meta.get("dt", (time_grid[1] - time_grid[0]).item()))
    print(f"dt_base: {dt_base}")
    kernel_blur = float(args.kernel_blur) if args.kernel_blur is not None else float(blur)
    vel_mode = args.vel

    # ---- Output directory ----
    outdir = Path(args.outdir)
    ensure_dir(outdir)
    run_name = args.run_name.strip() or outdir.name or f"{args.arch}_{now_str()}_seed{args.seed}"

    # SDPA math kernels
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

    # ---- Friction ----
    friction_obj: Optional[LearnableFriction] = None
    if bool(args.learnable_friction):
        friction_obj = LearnableFriction(init_gamma=float(args.friction)).to(device)
        friction_param = friction_obj
    else:
        friction_param = float(args.friction)
    if friction_obj is not None:
        print("[friction] init gamma =", float(friction_obj.friction_tensor().detach().cpu().item()))

    def friction_value() -> float:
        return friction_obj.value() if friction_obj is not None else float(friction_param)

    # ---- Build model ----
    model, model_kwargs = build_model_and_kwargs(args, d=d)
    model = model.to(device)

    # ---- EMA ----
    ema: Optional[EMA] = None
    if bool(getattr(args, 'use_ema', False)):
        ema = EMA(
            model,
            friction_module=friction_obj if bool(getattr(args, 'ema_friction', False)) else None,
            decay=float(getattr(args, 'ema_decay', 0.999))
        )
        print(f"[EMA] Initialized with decay={ema.decay}")

    wd = float(getattr(args, 'weight_decay', 1e-4))  # Safety getattr if you didn't update parser yet

    friction_freeze_epochs = int(getattr(args, 'friction_freeze_epochs', 0))
    if friction_obj is not None:
        friction_lr_initial = 0.0 if friction_freeze_epochs > 0 else float(args.friction_lr)
        optimizer = torch.optim.AdamW([
            {
                "params": model.parameters(),
                "lr": float(args.lr),
                "weight_decay": wd
            },
            {
                "params": [friction_obj.theta],
                "lr": friction_lr_initial,
                "weight_decay": 0.0
            },
        ])
        if friction_freeze_epochs > 0:
            print(f"[friction] Frozen for first {friction_freeze_epochs} epochs")
    else: # no friction in learnable WLM model
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(args.lr),
            weight_decay=wd
        )
    # ---- LR Scheduler ----
    scheduler = None
    if str(getattr(args, 'lr_schedule', 'constant')) == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
        warmup_steps = int(getattr(args, 'lr_warmup', 0))
        total_steps = int(args.num_epochs)
        lr_min = float(getattr(args, 'lr_min', 1e-6))

        if warmup_steps > 0:
            warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
            cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=lr_min)
            scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=lr_min)
        print(f"[LR] Cosine schedule: {args.lr} -> {lr_min} over {total_steps} steps, warmup={warmup_steps}")
    elif str(getattr(args, 'lr_schedule', 'constant')) == 'wsd':
        from torch.optim.lr_scheduler import LambdaLR
        warmup_steps = int(getattr(args, 'lr_warmup', 500))
        plateau_end = int(getattr(args, 'lr_plateau_end', 6000))
        total_steps = int(args.num_epochs)
        lr_min = float(getattr(args, 'lr_min', 1e-6))
        lr_max = float(args.lr)

        def wsd_lambda(step):
            if step < warmup_steps:
                # Linear warmup from 0 to lr_max
                return step / max(1, warmup_steps)
            elif step < plateau_end:
                # Plateau at lr_max
                return 1.0
            else:
                # Linear decay from lr_max to lr_min
                decay_steps = total_steps - plateau_end
                if decay_steps <= 0:
                    return 1.0
                progress = (step - plateau_end) / decay_steps
                return max(lr_min / lr_max, 1.0 - progress * (1.0 - lr_min / lr_max))

        scheduler = LambdaLR(optimizer, lr_lambda=wsd_lambda)
        print(f"[LR] WSD schedule: warmup={warmup_steps}, plateau_end={plateau_end}, total={total_steps}, lr_max={lr_max}, lr_min={lr_min}")

    # ---- Config ----
    config: Dict[str, Any] = vars(args).copy()
    config.update({
        "resolved": {
            "dt_base": float(dt_base),
            "kernel_blur": float(kernel_blur),
            "bundle_blur": float(blur),
            "meta": meta,
            "num_p0": int(num_p0),
            "N": int(N),
            "steps": int(max_train_steps),
            "d": int(d),
            "device": str(device),
            "vel_mode": str(vel_mode),
        },
        "model_kwargs": model_kwargs,
        "mode_info": mode_info,
    })
    _find_complex(config)
    dump_json(outdir / "config.json", config)

    # ---- W&B init ----
    wb_run = None
    if bool(args.wandb):
        tags = [t.strip() for t in str(args.wandb_tags).split(",") if t.strip()]
        tags.append(eval_mode)  # Tag with eval mode
        wb_name = args.wandb_name.strip() or run_name
        os.environ.setdefault("WANDB_SILENT", "true")
        wb_config = _wandb_sanitize(config)
        wb_run = wandb.init(
            project=str(args.wandb_project),
            entity=(None if args.wandb_entity in (None, "", "none") else str(args.wandb_entity)),
            name=wb_name,
            dir=os.environ.get("WANDB_DIR", str(outdir)),
            config=wb_config,
            tags=tags,
        )

    metrics_path = outdir / "metrics.jsonl"

    # ---- DICE models (if estimating velocity) ----
    data_path = Path(args.data)
    master_dir = outdir
    for p in outdir.parents:
        if p.name == "train":
            master_dir = p.parent
            break

    data_tag = data_path.parent.name
    dice_models: Optional[List[nn.Module]] = None
    if vel_mode == "dice":
        dice_dir = master_dir / "dice" / data_tag
        ensure_dir(dice_dir)
        dice_models_pt = dice_dir / "dice_models.pt"
        dice_bundle_path = dice_dir / "dice_bundle.pt"

        if dice_models_pt.exists():
            print(f"[dice] load teacher: {dice_models_pt}")
            dice_models = load_dice_models(
                str(dice_models_pt), device=device, d=int(d),
                hidden=int(args.dice_hidden), num_p0=int(num_p0),
            )
        else:
            print(f"[dice] train/load teacher bundle: {dice_bundle_path}")
            dice_models, _ = train_or_load_dice_bundle(
                bundle_path=dice_bundle_path, X_em_torch=X_em, time_grid=time_grid,
                d=int(d), hidden=int(args.dice_hidden), device=device,
                wandb_run=wb_run, log_prefix=f"_{run_name}",
                steps=10000, lr=1e-3, lr_end=1e-5, clip_norm=1.0,
                batch_size_t=int(time_grid.numel()), batch_size_x=128,
            )

        dice_gif_path = dice_dir / "dice_diagnostic.gif"
        maybe_make_dice_diagnostic_gif(
            save_path=str(dice_gif_path), X_em=X_em, time_grid=time_grid,
            dice_models=dice_models, pop_idx=int(args.gif_p0_idx),
            wandb_run=wb_run, wandb_step=0,
        )

    # ---- Loaded or estimated velocity  ----
    vel_provider = build_vel_provider(vel_mode, meta, dice_models, V_em=V_train, time_grid=t_train)
    print_velocity_diagnostics(V_em, V_train, vel_mode, eval_mode)

    # the WLM model determined acceleration, which will get fed into the integrator (ex. leapfrog)
    accel_train = make_accel_from_potential(model, create_graph=True, max_force=args.max_force)

    def save_ckpt(*, tag: str, step_idx: int, friction_value: float, friction_raw=None):
        save_ckpt_impl(
            outdir=outdir, tag=tag, step_idx=step_idx, model=model,
            optimizer=optimizer, model_kwargs=model_kwargs, meta=meta,
            config=config,
            learnable_friction=bool(args.learnable_friction),
            friction_value=friction_value, friction_raw=friction_raw,
            run_name=run_name, wb_run=None, ema=ema,
        )

    # ---- Evaluation ----
    # Build the plot hook once. Only fires for interpolate mode (forecast mode
    # passes plot_hook=None implicitly when clouds aren't requested).
    eval_plot_dir = outdir / "eval_plots"

    def _plot_hook(*, step_idx: int, mode: str, clouds, time_grid, metrics) -> None:
        if mode != "interpolate" or not clouds:
            return
        ensure_dir(eval_plot_dir)
        if len(clouds) == 1:
            h_idx, x_pred_cpu, y_true_cpu = clouds[0]
            t_val = float(time_grid[h_idx].item())
            w1_h = metrics.get(f"eval_w1_h{h_idx}", float("nan"))
            fig = plot_single_holdout_scatter(
                x_pred_cpu, y_true_cpu,
                holdout_idx=h_idx, time_val=t_val, w1_value=w1_h,
            )
            fig.savefig(eval_plot_dir / f"scatter_h{h_idx}_step{step_idx:07d}.png",
                        dpi=150, bbox_inches="tight")
            plt.close(fig)
        else:
            all_true = [(y, h) for (h, _, y) in clouds]
            all_pred = [(x, h) for (h, x, _) in clouds]
            fig = plot_multi_holdout_scatter(
                all_true, all_pred, time_grid,
                title=f"Interpolation @ step {step_idx}",
            )
            fig.savefig(eval_plot_dir / f"scatter_multi_step{step_idx:07d}.png",
                        dpi=150, bbox_inches="tight")
            plt.close(fig)

    # Mode-specific eval args
    if eval_mode == "interpolate":
        mode_eval_kwargs: Dict[str, Any] = {
            "holdout_indices": list(partition["holdout_indices"]),
            "train_time_idx": list(partition["train_time_idx"]),
            "n_train_marginals": None,
        }
    else:
        mode_eval_kwargs = {
            "n_train_marginals": int(n_train_marginals),
            "holdout_indices": None,
            "train_time_idx": None,
        }

    do_eval = partial(
        do_eval_impl,
        model=model,
        friction_value_fn=friction_value,
        X_em=X_em,  # full data (interpolate needs all marginals; forecast needs full horizon)
        time_grid=time_grid,
        dt_base=float(dt_base),
        substeps_per_dt=int(args.substeps_per_dt),
        vel_provider=vel_provider,
        friction=friction_param,
        integrator_name=str(args.integrator),
        max_force=args.max_force,
        particles_eval=(None if args.particles_eval is None else int(args.particles_eval)),
        vel_mode=str(vel_mode),
        V_em=V_em,
        dt_eval=getattr(args, "dt_eval", None),
        ema=ema,
        friction_obj=friction_obj,
        eval_mode=eval_mode,
        t_start_zero=bool(getattr(args, "eval_t_start_zero", False)),
        plot_hook=_plot_hook,
        **mode_eval_kwargs,
    )

    maybe_gif = partial(
        maybe_gif_impl,
        gif_every=int(args.gif_every),
        gif_p0_idx=int(args.gif_p0_idx),
        particles_gif=(None if args.particles_gif is None else int(args.particles_gif)),
        gif_frame_skip=int(args.gif_frame_skip),
        gif_fps=int(args.gif_fps),
        substeps_per_dt=int(args.substeps_per_dt),
        integrator_name=str(args.integrator),
        max_force=args.max_force,
        model=model,
        X_em=X_em,
        time_grid=time_grid,
        dt_base=float(dt_base),
        vel_provider=vel_provider,
        vel_mode=str(vel_mode),
        V_em=V_em,
        friction=friction_param,
        outdir=outdir,
        device=device,
        wb_run=wb_run,
    )

    # ---- Training ----
    t0_wall = time.time()
    cb_state = TrainCallbackState()
    step_idx = 0

    remaining = int(args.num_epochs)
    chunk = max(1, int(args.eval_every))

    friction_raw_fn = (
        (lambda: float(friction_obj.theta.detach().cpu().item()))
        if friction_obj is not None else (lambda: None)
    )

    while remaining > 0:
        cur = min(chunk, remaining)
        cb_state.chunk_step_base = int(step_idx)

        epoch_cb = make_epoch_callback(
            state=cb_state, args=args, wb_run=wb_run,
            save_ckpt=save_ckpt, maybe_gif=maybe_gif,
            friction_value_fn=friction_value,
            friction_raw_fn=friction_raw_fn,
            model=model, friction_obj=friction_obj,
            ema=ema, scheduler=scheduler,
        )

        # main training loop; returns dict with summary losses
        tr = train_rollout_anchor_p0_randk(
            X_em_torch=X_train, time_grid=t_train, accel_train=accel_train,
            optimizer=optimizer, dt_base=float(dt_base), num_epochs=int(cur),
            max_train_steps=max_train_steps, substeps_per_dt=int(args.substeps_per_dt),
            kernel_blur=float(kernel_blur), loss_type=args.loss_type,
            particles_per_batch=args.particles_per_batch, vel_provider=vel_provider,
            friction=friction_param, debug=False, name=run_name,
            verlet=str(args.integrator), geom_p=int(args.geom_p),
            geom_scaling=float(args.geom_scaling), geom_debias=bool(args.geom_debias),
            geom_backend=args.geom_backend, epoch_callback=epoch_cb,
            subsample_targets=bool(args.subsample_targets),
            k_ramp_fraction=float(args.k_ramp_fraction),
        )

        step_idx += int(cur)
        # Unfreeze friction after freeze period
        if friction_obj is not None and friction_freeze_epochs > 0 and step_idx >= friction_freeze_epochs:
            for pg in optimizer.param_groups:
                if len(pg['params']) == 1 and pg['params'][0] is friction_obj.theta:
                    if pg['lr'] == 0.0:
                        pg['lr'] = float(args.friction_lr)
                        print(f"[friction] Unfrozen at step {step_idx}, lr={args.friction_lr}")
        remaining -= int(cur)

        if wb_run is not None and isinstance(tr, dict):
            wb_run.log({
                "train_summary/loss_avg": float(tr.get("train_loss_avg", 0.0)),
                "train_summary/loss_last": float(tr.get("train_loss_last", 0.0)),
            }, step=int(cb_state.global_step))

        ev = do_eval(step_idx=int(step_idx))
        ev["wall_s"] = float(time.time() - t0_wall)
        append_jsonl(metrics_path, {"type": "eval", **ev})

        if wb_run is not None:
            log_dict = {"eval/friction": float(ev["friction_value"])}
            # Forecast: train/test on contiguous split
            if "test_w1_avg" in ev and ev.get("test_count", 0) > 0:
                log_dict["eval/test_w1"] = float(ev["test_w1_avg"])
                log_dict["eval/test_w1_se"] = float(ev.get("test_w1_se", 0.0))
            if "train_w1_avg" in ev and ev.get("train_count", 0) > 0:
                log_dict["eval/train_w1"] = float(ev["train_w1_avg"])
                log_dict["eval/train_w1_se"] = float(ev.get("train_w1_se", 0.0))
            # Interpolate: per-holdout + mean
            for k, v in ev.items():
                if k.startswith("eval_w1_h"):
                    log_dict[f"eval/w1_h{k.removeprefix('eval_w1_h')}"] = float(v)
            if "eval_w1_mean" in ev:
                log_dict["eval/w1_mean"] = float(ev["eval_w1_mean"])
            wb_run.log(log_dict, step=int(cb_state.global_step))

        # Console summary
        if eval_mode == "forecast":
            print(f"[eval] step={step_idx} "
                  f"train_w1={ev.get('train_w1_avg', float('nan')):.4f}±{ev.get('train_w1_se', 0.0):.4f} "
                  f"test_w1={ev.get('test_w1_avg', float('nan')):.4f}±{ev.get('test_w1_se', 0.0):.4f}")
        else:
            print(f"[eval] step={step_idx} w1_mean={ev.get('eval_w1_mean', float('nan')):.4f}")

    # ---- Final save ----
    save_ckpt(
        tag="final",
        step_idx=int(step_idx),
        friction_value=float(friction_value()),
        friction_raw=(float(friction_obj.theta.detach().cpu().item()) if friction_obj is not None else None),
    )

    if wb_run is not None:
        wb_run.finish()


if __name__ == "__main__":
    main()