"""Config runner for scale."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from omegaconf import OmegaConf

from conformal_model.config_utils import ensure_run_fields
from experiments.run_scale import run_experiment


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _to_dict(obj: Any) -> dict:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "__dict__"):
        return vars(obj)
    raise TypeError(f"Unsupported config type: {type(obj)}")


def _require_src_dir(cfg: Mapping[str, Any]) -> None:
    val = cfg.get("src_dir")
    if val in (None, "???"):
        raise ValueError("src_dir is required; set it in the config file or override it.")


def run(cfg_or_args=None, **kwargs):
    cfg_dict = _to_dict(cfg_or_args)
    if kwargs:
        cfg_dict.update(kwargs)
    if not cfg_dict:
        raise ValueError("Empty config; expected args in CONFIG['args'] or overrides.")

    _require_src_dir(cfg_dict)
    cfg_dict = ensure_run_fields(cfg_dict, root=PROJECT_ROOT, default_dir=cfg_dict.get("run", {}).get("dir"))
    cfg = OmegaConf.create(cfg_dict)
    return run_experiment(cfg)
