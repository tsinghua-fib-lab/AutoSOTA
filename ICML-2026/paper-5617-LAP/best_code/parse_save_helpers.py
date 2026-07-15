from __future__ import annotations
import torch
import argparse
import json
import os
import time
import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional, Callable, List, Union, Tuple
import torch.nn as nn

# ============================================================
# Config loading helpers (YAML/JSON) + CLI overrides
# ============================================================
from typing import Any, Dict, Optional, Sequence


try:
    import yaml

    _HAS_YAML = True
except Exception:
    yaml = None
    _HAS_YAML = False


# ============================================================
# Utilities
# ============================================================

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_device(device_str: str) -> torch.device:
    if device_str == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_str == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unknown device={device_str}. Use cpu|cuda.")


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_bundle(out_path, X_em_torch, time_grid, meta, *, blur, V_em_torch=None):
    payload = {
        "X_em_torch": X_em_torch.detach().cpu(),
        "time_grid": time_grid.detach().cpu(),
        "meta": meta,
        "blur": float(blur),
    }
    if V_em_torch is not None:
        payload["V_em_torch"] = V_em_torch.detach().cpu()
    torch.save(payload, out_path)


def _load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith((".yml", ".yaml")):
            if not _HAS_YAML:
                raise RuntimeError("YAML config requested but pyyaml is not installed.")
            return yaml.safe_load(f) or {}
        if path.endswith(".json"):
            return json.load(f) or {}
        # default: try YAML then JSON
        if _HAS_YAML:
            try:
                f.seek(0)
                return yaml.safe_load(f) or {}
            except Exception:
                pass
        f.seek(0)
        return json.load(f) or {}


def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def _coerce_scalar(s: str) -> Any:
    sl = s.lower()
    if sl in {"true", "false"}:
        return sl == "true"
    if sl in {"none", "null"}:
        return None
    try:
        if "." in s or "e" in sl:
            return float(s)
        return int(s)
    except Exception:
        return s


def _apply_dotset(cfg: Dict[str, Any], dotkey: str, value: Any) -> None:
    cur = cfg
    parts = dotkey.split(".")
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def _parse_set_kv(items: Optional[Sequence[str]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not items:
        return out
    for it in items:
        if "=" not in it:
            raise ValueError(f"--set expects key=value, got: {it!r}")
        k, v = it.split("=", 1)
        _apply_dotset(out, k, _coerce_scalar(v))
    return out


def _flatten_for_argparse(cfg: Dict[str, Any]) -> Dict[str, Any]:
    def norm(k: str) -> str:
        return str(k).replace("-", "_")

    flat: Dict[str, Any] = {}

    # top-level scalars
    for k, v in cfg.items():
        if not isinstance(v, dict):
            flat[norm(k)] = v

    # one-level nested scalars (your current behavior)
    for _, sub in cfg.items():
        if isinstance(sub, dict):
            for k, v in sub.items():
                if not isinstance(v, dict):
                    flat[norm(k)] = v

    return flat


def _parse_args_with_config(
        parser: argparse.ArgumentParser,
        argv: Optional[Sequence[str]] = None
) -> argparse.Namespace:
    """
    Parse CLI args with optional config file support.

    Supports config schema:
      - mode: <subcommand name>   (optional)
      - argv: [ ... ]             (optional list of tokens)
    plus dot-key overrides via --set.

    Precedence (highest last):
      1) config file values (including config.argv tokens)
      2) --set dot overrides applied to config (as defaults)
      3) explicit CLI flags (excluding --config/--set), which override config.argv
    """
    if argv is None:
        import sys
        argv = sys.argv[1:]

    # Pre-parse only --config and --set
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None)
    pre.add_argument("--set", action="append", default=None)
    ns, unknown_cli = pre.parse_known_args(list(argv))

    cfg: Dict[str, Any] = {}
    if ns.config:
        cfg = _load_config(ns.config)

    # allow dot overrides on top of file
    overrides = _parse_set_kv(ns.set)
    _deep_update(cfg, overrides)

    # Build config argv tokens (if provided)
    cfg_argv = list(cfg.get("argv", []) or [])
    cfg_mode = cfg.get("mode", None)

    # If using subparsers, ensure subcommand token is present when mode is specified.
    if cfg_mode is not None:
        if len(cfg_argv) == 0 or cfg_argv[0] != cfg_mode:
            cfg_argv = [str(cfg_mode)] + cfg_argv

    # Apply flattened defaults to the main parser (covers non-argv config keys)
    if cfg:
        parser.set_defaults(**_flatten_for_argparse(cfg))

    # NEW: also apply defaults to the selected subparser.
    # argparse parent defaults do NOT override subparser arg defaults.
    sp_action = None
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            sp_action = action
            break

    # Determine active mode (prefer cfg.mode; fallback to first argv token)
    mode_key = None
    if cfg_mode is not None:
        mode_key = str(cfg_mode)
    elif len(cfg_argv) > 0:
        mode_key = str(cfg_argv[0])

    if sp_action is not None and mode_key is not None and mode_key in sp_action.choices:
        # Start with top-level defaults (these include N/steps/dt in your current YAML)
        sub_defaults = _flatten_for_argparse(cfg)
        sub_defaults.pop("mode", None)
        sub_defaults.pop("argv", None)

        # If you *also* have a nested mode block (optional), let it override top-level
        mode_block = cfg.get(mode_key, {})
        if isinstance(mode_block, dict) and mode_block:
            sub_defaults.update(_flatten_for_argparse(mode_block))

        sp_action.choices[mode_key].set_defaults(**sub_defaults)

    # Final argv = config-provided tokens + CLI tokens other than --config/--set
    final_argv = cfg_argv + list(unknown_cli)

    args = parser.parse_args(final_argv)

    # Attach config path for logging if caller wants it
    if not hasattr(args, "config"):
        setattr(args, "config", ns.config)
    return args


# ============================================================
# IO / Repro helpers
# ============================================================

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_device(device_str: str) -> torch.device:
    if device_str == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_str == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unknown device={device_str}. Use cpu|cuda.")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _json_default(o):
    import numpy as np
    # Complex numbers
    if isinstance(o, (complex, np.complexfloating)):
        return {"__type__": "complex", "real": float(np.real(o)), "imag": float(np.imag(o))}
    # Numpy scalars/arrays (often show up too)
    if isinstance(o, (np.integer, np.floating, np.bool_)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    # Paths, etc.
    if isinstance(o, Path):
        return str(o)
    # Last resort: string-ify
    return str(o)


def dump_json(path, obj):
    path = Path(path)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=_json_default))


def _yaml_sanitize(o):
    import numpy as np
    if isinstance(o, (complex, np.complexfloating)):
        return {"__type__": "complex", "real": float(np.real(o)), "imag": float(np.imag(o))}
    if isinstance(o, dict):
        return {k: _yaml_sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_yaml_sanitize(v) for v in o]
    # numpy scalars / arrays frequently appear too
    if hasattr(o, "item") and type(o).__module__.startswith("numpy"):
        try:
            return o.item()
        except Exception:
            pass
    if hasattr(o, "tolist") and type(o).__module__.startswith("numpy"):
        try:
            return o.tolist()
        except Exception:
            pass
    return o


def dump_yaml(path, obj):
    path = Path(path)
    obj2 = _yaml_sanitize(obj)
    path.write_text(yaml.safe_dump(obj2, sort_keys=True))


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    with path.open("a") as f:
        f.write(json.dumps(obj) + "\n")


def atomic_torch_save(obj: Any, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def now_str() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def _find_complex(obj, path="config"):
    if isinstance(obj, (complex, np.complexfloating)):
        print(f"[CONFIG COMPLEX] {path} = {obj!r}")
        return True
    if isinstance(obj, dict):
        hit = False
        for k, v in obj.items():
            hit |= _find_complex(v, f"{path}.{k}")
        return hit
    if isinstance(obj, (list, tuple)):
        hit = False
        for i, v in enumerate(obj):
            hit |= _find_complex(v, f"{path}[{i}]")
        return hit
    return False


def _wandb_sanitize(o):
    import numpy as np
    # JSON-safe representation
    if isinstance(o, (complex, np.complexfloating)):
        return {"__type__": "complex", "real": float(np.real(o)), "imag": float(np.imag(o))}
    if isinstance(o, dict):
        return {k: _wandb_sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_wandb_sanitize(v) for v in o]
    if isinstance(o, (np.integer, np.floating, np.bool_)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o

def print_velocity_diagnostics(
        V_em: Optional[torch.Tensor],
        V_train: Optional[torch.Tensor],
        vel_mode: str,
        eval_mode: str,
):
    """Print velocity loading diagnostics."""
    print("\n" + "=" * 60)
    print(f"VELOCITY DIAGNOSTICS ({eval_mode.upper()} MODE)")
    print("=" * 60)
    print(f"  vel_mode: {vel_mode}")
    print(f"  V_em in bundle: {V_em is not None}")

    if V_em is not None:
        print(f"  V_em shape: {V_em.shape}")
        print(f"  V_em device: {V_em.device}, dtype: {V_em.dtype}")

        num_nan = torch.isnan(V_em).sum().item()
        num_finite = torch.isfinite(V_em).sum().item()
        total = V_em.numel()
        print(f"  V_em NaN: {num_nan}/{total} ({100 * num_nan / total:.2f}%)")

        finite_mask = torch.isfinite(V_em)
        if finite_mask.any():
            finite_vals = V_em[finite_mask]
            print(f"  V_em stats: min={finite_vals.min():.4f}, max={finite_vals.max():.4f}, "
                  f"mean={finite_vals.mean():.4f}, std={finite_vals.std():.4f}")

        if V_train is not None:
            print(f"\n  V_train (for training):")
            print(f"    shape: {V_train.shape}")
            num_nan_t = torch.isnan(V_train).sum().item()
            total_t = V_train.numel()
            print(f"    NaN: {num_nan_t}/{total_t} ({100 * num_nan_t / total_t:.2f}%)")

            # Sample velocities
            v0_sample = V_train[0, :5, 0, :]
            print(f"    Sample v0 (pop=0, t=0, first 5 particles):")
            for i, v in enumerate(v0_sample):
                if torch.isfinite(v).all():
                    vlist = v.tolist()
                    print(f"      particle {i}: {[f'{x:.4f}' for x in vlist]}")
    else:
        print("  WARNING: V_em not in bundle!")
    print("=" * 60 + "\n")


# ============================================================
# DICE loader
# ============================================================
from dice import (
    ScalarScoreMLP,
    get_s_derivatives,
    train_or_load_dice_bundle,  # NEW
    maybe_make_dice_diagnostic_gif,  # NEW
)


def load_dice_models(
        dice_path: str,
        *,
        device: torch.device,
        d: int,
        hidden: int,
        num_p0: int,
) -> List[nn.Module]:
    payload = torch.load(dice_path, map_location="cpu")
    if isinstance(payload, dict):
        if "dice_state_dicts" in payload:
            sds = payload["dice_state_dicts"]
        else:
            raise ValueError(f"Unrecognized DICE payload keys: {list(payload.keys())}")
    elif isinstance(payload, list):
        sds = payload
    else:
        raise ValueError("Unrecognized DICE payload format.")

    if len(sds) != num_p0:
        raise ValueError(f"DICE models count {len(sds)} != num_p0 {num_p0}")

    models: List[nn.Module] = []
    for p in range(num_p0):
        m = ScalarScoreMLP(d=d, hidden=hidden).to(device)
        m.load_state_dict(sds[p])
        m.eval()
        models.append(m)
    return models


def save_ckpt(
        *,
        outdir: Path,
        tag: str,
        step_idx: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        model_kwargs: Dict[str, Any],
        meta: Dict[str, Any],
        config: Dict[str, Any],
        learnable_friction: bool,
        friction_value: float,
        friction_raw: Optional[Union[torch.Tensor, float]],
        run_name: str,
        wb_run: Optional[Any] = None,
        ema: Optional[Any] = None,  # EMA object
) -> None:
    ckpt = {
        "step": int(step_idx),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "model_kwargs": model_kwargs,
        "meta": meta,
        "config": config,
        "friction_learnable": bool(learnable_friction),
        "friction_value": float(friction_value),
    }
    if friction_raw is not None:
        ckpt["friction_raw"] = (friction_raw.detach().cpu() if torch.is_tensor(friction_raw) else friction_raw)

    # Save EMA state if available
    if ema is not None:
        ckpt["ema_state"] = ema.state_dict()

    atomic_torch_save(ckpt, outdir / f"{tag}.pt")


def do_eval(
        *,
        model: nn.Module,
        step_idx: int,
        friction_value_fn: Callable[[], float],
        X_em: torch.Tensor,
        time_grid: torch.Tensor,
        dt_base: float,
        substeps_per_dt: int,
        vel_provider,
        friction: Any,
        integrator_name: str,
        max_force: Optional[float],
        particles_eval: Optional[int],
        vel_mode: str,
        V_em: Optional[torch.Tensor],
        dt_eval: Optional[float] = None,
        ema: Optional[Any] = None,
        friction_obj: Optional[nn.Module] = None,
        eval_mode: str = "forecast",
        # Forecast-only:
        n_train_marginals: Optional[int] = None,
        # Interpolate-only:
        holdout_indices: Optional[List[int]] = None,
        train_time_idx: Optional[List[int]] = None,
        t_start_zero: bool = False,
        plot_hook: Optional[Callable[..., None]] = None,
) -> Dict[str, Any]:
    """
    Evaluate WLM model via W1 of predicted vs true marginals.

    Mode dispatch:
      - "forecast":    integrate from t=0; needs `n_train_marginals`.
      - "interpolate": integrate from previous training time per holdout;
                       needs `holdout_indices` and `train_time_idx`.

    EMA: if `ema` is supplied, EMA weights are temporarily swapped into `model`
    (and `friction_obj`, if provided) for the duration of the eval. EMA tracks
    friction only if it was constructed with a non-None friction_module.

    plot_hook(step_idx, mode, clouds, time_grid) is called after eval if provided.
    `clouds` is a list of (h_idx, x_pred_cpu, y_true_cpu) for interpolate mode,
    or None for forecast mode.
    """
    from losses import evaluate_model_w1

    friction_val = friction_value_fn()

    # dt for integration
    dt_integration = None
    if dt_eval is not None and 0 < float(dt_eval) < float(dt_base):
        dt_integration = float(dt_eval)

    # Whether to capture the predicted/true clouds (only useful for interpolate plotting)
    return_clouds = (eval_mode == "interpolate") and (plot_hook is not None)

    def _run() -> Dict[str, Any]:
        model.eval()
        try:
            metrics = evaluate_model_w1(
                model=model,
                X_em=X_em,
                time_grid=time_grid,
                dt_base=float(dt_base),
                substeps_per_dt=int(substeps_per_dt),
                integrator_name=str(integrator_name),
                friction=friction,
                vel_provider=vel_provider,
                vel_mode=str(vel_mode),
                V_em=V_em,
                max_force=max_force,
                particles_eval=particles_eval,
                eval_mode=str(eval_mode),
                n_train_marginals=n_train_marginals,
                holdout_indices=holdout_indices,
                train_time_idx=train_time_idx,
                dt_integration=dt_integration,
                t_start_zero=bool(t_start_zero),
                return_clouds=return_clouds,
            )
        finally:
            model.train()
        return metrics

    if ema is not None:
        print(f"[Eval] Using EMA weights (step={ema.step})")
        with ema.apply(model, friction_module=friction_obj):
            metrics = _run()
    else:
        metrics = _run()

    # Hand off to caller for plotting; pop clouds out before persisting (tensors
    # don't belong in metrics.jsonl).
    clouds = metrics.pop("clouds", None) if return_clouds else None
    if plot_hook is not None:
        try:
            plot_hook(step_idx=int(step_idx), mode=str(eval_mode),
                      clouds=clouds, time_grid=time_grid, metrics=metrics)
        except Exception as e:
            print(f"[warn] plot_hook failed: {e}")

    return {
        "step": int(step_idx),
        "friction_value": float(friction_val),
        **metrics,
    }


from dataclasses import dataclass


@dataclass
class TrainCallbackState:
    """
    Mutable state shared across callbacks.
    - global_step: increments per epoch (for wandb step)
    - chunk_step_base: set before each chunk so epoch_in_chunk -> absolute epoch
    """
    global_step: int = 0
    chunk_step_base: int = 0


def make_epoch_callback(
        *,
        state: TrainCallbackState,
        args: Any,
        wb_run: Optional[Any],
        save_ckpt: Callable[..., None],
        maybe_gif: Callable[[int], None],
        friction_value_fn: Callable[[], float],
        friction_raw_fn: Callable[[], Optional[Any]],
        model: nn.Module,
        friction_obj: Optional[nn.Module] = None,
        ema: Optional[Any] = None,
        scheduler: Optional[Any] = None,
) -> Callable[[int, int, float, float], None]:
    """
    Returns epoch_callback(epoch_in_chunk, k, loss, friction_val).

    Per epoch this:
      1. Steps EMA shadows (if ema given) and the LR scheduler (if any).
      2. Logs train/loss + train/friction to wandb every LOG_FREQ epochs.
      3. Saves a checkpoint every args.ckpt_every epochs (if > 0).
      4. Emits a GIF every args.gif_every epochs (if > 0).
    """
    LOG_FREQ = 1000
    ckpt_freq = int(getattr(args, 'ckpt_every', 0))
    gif_freq = int(getattr(args, 'gif_every', 0))

    def epoch_callback(epoch_in_chunk: int, k: int, loss: float, friction_val: float) -> None:
        del k  # not used in this callback; horizon length is logged elsewhere if desired

        # 1. EMA + LR scheduler updates (after optimizer.step has already happened)
        if ema is not None:
            ema.update(model, friction_module=friction_obj)
        if scheduler is not None:
            scheduler.step()

        abs_epoch = int(state.chunk_step_base) + int(epoch_in_chunk) + 1  # 1-based absolute epoch
        state.global_step += 1

        # 2. Periodic train/loss + friction logging
        if wb_run is not None and (abs_epoch % LOG_FREQ == 0):
            import wandb
            wandb.log(
                {
                    "train/loss": float(loss),
                    "train/friction": float(friction_val),
                    "train/epoch": int(abs_epoch),
                },
                step=int(state.global_step),
            )

        # 3. Periodic checkpoint
        if ckpt_freq > 0 and (abs_epoch % ckpt_freq == 0):
            save_ckpt(
                tag=f"ckpt_step{abs_epoch:07d}",
                step_idx=int(abs_epoch),
                friction_value=float(friction_value_fn()),
                friction_raw=friction_raw_fn(),
            )
            if wb_run is not None:
                import wandb
                wandb.log({"ckpt/epoch": int(abs_epoch)}, step=int(state.global_step))

        # 4. Periodic GIF
        if gif_freq > 0 and (abs_epoch % gif_freq == 0):
            maybe_gif(int(abs_epoch))

    return epoch_callback


