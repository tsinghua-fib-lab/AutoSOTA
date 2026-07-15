"""
Unified training utilities.

This module consolidates the former:
- `train.config_utils` (Hydra/OmegaConf config helpers)
- `train.diffusion.train_utils` (logger/checkpoint/trainer/checkpoint-loading helpers)

Prefer importing from `train.utils` going forward.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from hydra.utils import instantiate
import lightning.pytorch as pl
from lightning.pytorch import Callback, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, ListConfig, OmegaConf


class StripMetadataModule(pl.LightningModule):
    def on_load_checkpoint(self, checkpoint):
        sd = checkpoint.get("state_dict")
        if sd:
            sd.pop("_metadata", None)
            for k in list(sd.keys()):
                if k.endswith("._metadata"):
                    sd.pop(k, None)


_NAME_KEYS = {"model_name", "mode_name"}


def strip_name_keys(
    cfg: DictConfig | ListConfig | None,
) -> DictConfig | ListConfig | None:
    """Return a deep-copied config with `model_name`/`mode_name` removed everywhere.

    We keep these keys in YAML for logging/run naming, but they must NOT be passed
    into constructors via `hydra.utils.instantiate`.
    """
    if cfg is None:
        return None
    cfg_copy = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    _strip_name_keys_inplace(cfg_copy)
    return cfg_copy


def _strip_name_keys_inplace(node: Any) -> None:
    if isinstance(node, DictConfig):
        for key in list(node.keys()):
            if key in _NAME_KEYS:
                del node[key]
            else:
                _strip_name_keys_inplace(node[key])
    elif isinstance(node, ListConfig):
        for item in node:
            _strip_name_keys_inplace(item)


def split_dataset_result(
    dataset_result: Any,
) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    """Unpack dataset instantiation results with optional extras dict.

    Supported forms:
    - (train_set, val_set, test_set)
    - ((train_set, val_set, test_set), extras_dict)   where extras_dict is a dict

    Returns:
        (train_set, val_set, test_set, val_trjs, test_trjs, val_stochastic, test_stochastic)

    Notes:
        `extras_dict` is an optional dict that may contain:
          - "val_trjs", "test_trjs": full trajectories for rollout eval
          - "val_stochastic", "test_stochastic": stochastic eval payloads for operator learning
            (recommended structure: {"x": x_orig, "y": y_samples_orig})
    """
    if isinstance(dataset_result, tuple) and len(dataset_result) == 2:
        (train_set, val_set, test_set), extras_dict = dataset_result
        if isinstance(extras_dict, dict):
            val_trjs = extras_dict.get("val_trjs")
            test_trjs = extras_dict.get("test_trjs")
            val_stochastic = extras_dict.get("val_stochastic")
            test_stochastic = extras_dict.get("test_stochastic")
        else:
            val_trjs = None
            test_trjs = None
            val_stochastic = None
            test_stochastic = None
        return (
            train_set,
            val_set,
            test_set,
            val_trjs,
            test_trjs,
            val_stochastic,
            test_stochastic,
        )

    train_set, val_set, test_set = dataset_result
    return train_set, val_set, test_set, None, None, None, None


def get_dataset_data_name_lower(cfg: DictConfig) -> str:
    """Best-effort dataset name (lowercased) used for small conventions."""
    return str(getattr(getattr(cfg, "dataset", None), "data_name", "")).lower()


def should_share_xy_normalizer(*, data_name_lower: str, training_cfg: Any) -> bool:
    """Heuristic convention: share x/y normalizer for KS-like next-step datasets."""
    return bool(getattr(training_cfg, "share_xy_normalizer", False)) or str(
        data_name_lower
    ).startswith("ks")


def _steps_from_percentiles(
    total_steps: int,
    percentiles: Iterable[float] = (20, 40, 60, 80),
) -> dict[int, int]:
    """
    Map percentile -> rollout step index (1-based) within [1, total_steps].

    Example: total_steps=100 -> p20->20, p40->40, ...
    """
    if total_steps <= 0:
        return {}

    out: dict[int, int] = {}
    for p in percentiles:
        p_f = float(p)
        p_f = max(0.0, min(100.0, p_f))
        step = int(round((p_f / 100.0) * float(total_steps)))
        step = max(1, min(total_steps, step))
        out[int(round(p_f))] = step
    seen: set[int] = set()
    deduped: dict[int, int] = {}
    for p in sorted(out.keys()):
        s = out[p]
        if s in seen:
            continue
        seen.add(s)
        deduped[p] = s
    return deduped


def optional_ckpt_path(value: Any) -> Optional[Path]:
    """Normalize optional checkpoint paths coming from Hydra config.

    Accepts None, empty string, "null", "none" as missing.
    """
    if value is None:
        return None
    s = str(value).strip()
    if s == "" or s.lower() in {"none", "null"}:
        return None
    return Path(s).expanduser()


def make_run_name(model_name: str, task_name: str) -> str:
    """Generate a timestamped run name."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{model_name}_{task_name}_{timestamp}"


def make_wandb_logger(
    cfg: DictConfig,
    *,
    project_name: str,
    run_name: str,
    log_model_default: str = "best_and_last",
    log_model: Any = None,
    save_dir: Optional[str] = None,
) -> WandbLogger:
    """Construct a WandbLogger with fallbacks to the Hydra config."""
    logging_cfg = getattr(cfg, "logging", None)

    if log_model is None:
        log_model = (
            getattr(logging_cfg, "log_model", log_model_default)
            if logging_cfg
            else log_model_default
        )

    if save_dir is None:
        save_dir = getattr(logging_cfg, "save_dir", None) if logging_cfg else None

    if save_dir is not None:
        save_dir = str(Path(save_dir).expanduser())

    return WandbLogger(
        project=project_name,
        name=run_name,
        log_model=log_model,
        save_dir=save_dir,
    )


def make_checkpoint_callback(
    *,
    project_name: str,
    run_name: str,
    filename_base: str,
    filename: Optional[str] = None,
    monitor: str = "val_loss",
    mode: str = "min",
    save_top_k: int = 1,
    save_last: bool = True,
) -> ModelCheckpoint:
    """Construct a ModelCheckpoint callback with standardized naming conventions."""
    checkpoint_dir = Path("checkpoints") / project_name / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    filename_template = (
        str(filename)
        if filename is not None
        else f"{filename_base}-{{epoch:02d}}-{{{monitor}:.4f}}"
    )

    return ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename=filename_template,
        save_top_k=int(save_top_k),
        monitor=str(monitor),
        mode=str(mode),
        save_last=bool(save_last),
    )


def make_trainer(
    *,
    training_cfg: DictConfig,
    max_epochs: int,
    train_loader_len: int,
    logger: WandbLogger,
    callbacks: Iterable[Callback],
) -> Trainer:
    """Construct the Lightning Trainer."""
    check_val_every_n_epoch = int(getattr(training_cfg, "check_val_every_n_epoch", 1))

    default_log_every = train_loader_len if train_loader_len > 0 else 1
    log_every_n_steps = int(
        getattr(training_cfg, "log_every_n_steps", default_log_every)
    )
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = 1

    gradient_clip_val = float(getattr(training_cfg, "gradient_clip_val", 0.0))
    gradient_clip_algorithm = str(
        getattr(training_cfg, "gradient_clip_algorithm", "norm")
    )
    if gradient_clip_val <= 0.0:
        gradient_clip_val = 0.0
        gradient_clip_algorithm = "norm"
    limit_train_batches = getattr(training_cfg, "limit_train_batches", None)
    limit_val_batches = getattr(training_cfg, "limit_val_batches", None)
    limit_test_batches = getattr(training_cfg, "limit_test_batches", None)
    num_sanity_val_steps = getattr(training_cfg, "num_sanity_val_steps", None)

    return Trainer(
        max_epochs=int(max_epochs),
        logger=logger,
        callbacks=list(callbacks),
        accelerator=accelerator,
        devices=devices,
        enable_progress_bar=True,
        check_val_every_n_epoch=check_val_every_n_epoch,
        log_every_n_steps=log_every_n_steps,
        gradient_clip_val=gradient_clip_val,
        gradient_clip_algorithm=gradient_clip_algorithm,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        limit_test_batches=limit_test_batches,
        num_sanity_val_steps=num_sanity_val_steps,
    )


def finish_wandb(logger: Any) -> None:
    """Close W&B run safely if the Lightning logger is a WandbLogger."""
    if (
        isinstance(logger, WandbLogger)
        and getattr(logger, "experiment", None) is not None
    ):
        logger.experiment.finish()


def load_lightning_checkpoint_state_dict(
    ckpt_path: Union[str, Path],
    *,
    map_location: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """Load a Lightning/W&B checkpoint and return the contained `state_dict`.

    This function is robust to:
    - Lightning `.ckpt` structure (top-level dict with `state_dict`).
    - Raw `state_dict` files (a dict of parameter tensors).
    - PyTorch 2.4+ `weights_only=True` security defaults (handling OmegaConf objects safely).
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    try:
        from torch.serialization import (
            safe_globals,
        )  # pyright: ignore[reportMissingImports]
        from omegaconf import DictConfig as _DictConfig  # noqa: N811
        from omegaconf import ListConfig as _ListConfig  # noqa: N811
        from omegaconf.base import ContainerMetadata as _ContainerMetadata  # noqa: N811

        safe_ctx = safe_globals([_DictConfig, _ListConfig, _ContainerMetadata])
    except Exception:
        from contextlib import nullcontext

        safe_ctx = nullcontext()

    with safe_ctx:
        try:
            ckpt = torch.load(
                str(ckpt_path), map_location=map_location, weights_only=True
            )
        except TypeError:
            ckpt = torch.load(str(ckpt_path), map_location=map_location)
        except Exception:
            ckpt = torch.load(
                str(ckpt_path), map_location=map_location, weights_only=False
            )

    if (
        isinstance(ckpt, dict)
        and "state_dict" in ckpt
        and isinstance(ckpt["state_dict"], dict)
    ):
        sd = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        sd = ckpt
    else:
        raise ValueError(f"Unexpected checkpoint type at {ckpt_path}: {type(ckpt)}")

    sd = dict(sd)
    if any(k.startswith("module.") for k in sd.keys()):
        sd = {
            (k[len("module.") :] if k.startswith("module.") else k): v
            for k, v in sd.items()
        }

    return {k: v for k, v in sd.items() if isinstance(v, torch.Tensor)}


def load_submodule_state_dict_from_lightning_ckpt(
    module: nn.Module,
    ckpt_path: Union[str, Path],
    *,
    prefixes: List[str],
    map_location: str = "cpu",
    strict: bool = True,
) -> Tuple[List[str], List[str], str]:
    """Load weights into `module` from a Lightning checkpoint by prefix matching.

    Returns:
        (missing_keys, unexpected_keys, used_prefix)
    """
    sd = load_lightning_checkpoint_state_dict(ckpt_path, map_location=map_location)

    used_prefix = ""
    candidate: Optional[Dict[str, torch.Tensor]] = None

    for prefix in prefixes:
        if prefix == "":
            candidate = sd
            used_prefix = ""
            break

        if any(k.startswith(prefix) for k in sd.keys()):
            candidate = {
                k[len(prefix) :]: v for k, v in sd.items() if k.startswith(prefix)
            }
            used_prefix = prefix
            break

    if candidate is None:
        candidate = sd
        used_prefix = ""

    incompatible = module.load_state_dict(candidate, strict=strict)
    return (
        list(incompatible.missing_keys),
        list(incompatible.unexpected_keys),
        used_prefix,
    )
