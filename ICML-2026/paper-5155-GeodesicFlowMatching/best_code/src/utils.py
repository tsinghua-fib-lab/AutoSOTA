"""
YAML loading and path resolution for the research pipeline.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

REQUIRED_TOP_LEVEL = (
    "project",
    "wandb",
    "paths",
    "experiment",
    "ssp",
    "data",
    "trainer",
    "eval",
    "schedule",
)

REQUIRED_PATH_KEYS = ("data_root", "checkpoint_dir", "figures_dir")


def get_project_root(start: Path | None = None) -> Path:
    if start is None:
        start = Path(__file__).resolve().parent
    if start.name == "src":
        return start.parent
    return start


def validate_config(cfg: dict[str, Any]) -> None:
    if not isinstance(cfg, dict):
        raise ValueError("Config root must be a mapping")
    missing = [k for k in REQUIRED_TOP_LEVEL if k not in cfg]
    if missing:
        raise ValueError(f"Missing config section(s): {missing}")
    paths = cfg.get("paths")
    if not isinstance(paths, dict):
        raise ValueError("paths must be a mapping")
    missing_p = [k for k in REQUIRED_PATH_KEYS if k not in paths or paths[k] is None]
    if missing_p:
        raise ValueError(f"Missing paths key(s): {missing_p}")


def load_config(
    path: Path | str | None = None,
    *,
    project_root: Path | None = None,
) -> dict[str, Any]:
    root = project_root or get_project_root()
    if path is None:
        cfg_path = root / "configs" / "config.yaml"
    else:
        cfg_path = Path(path)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if raw is None:
        raise ValueError(f"Config is empty: {cfg_path}")
    validate_config(raw)
    return raw


def resolve_config_paths(cfg: dict[str, Any], project_root: Path) -> dict[str, Any]:
    out = deepcopy(cfg)
    paths = out.setdefault("paths", {})
    for key in REQUIRED_PATH_KEYS:
        val = paths.get(key)
        if val is None:
            continue
        p = Path(val)
        if not p.is_absolute():
            p = (project_root / p).resolve()
        else:
            p = p.resolve()
        paths[key] = p
    return out

