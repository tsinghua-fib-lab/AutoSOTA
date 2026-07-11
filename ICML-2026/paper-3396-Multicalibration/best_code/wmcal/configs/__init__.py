# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from ..experiments import ExperimentConfig

_PACKAGE_DIR = Path(__file__).parent
_REGISTRY: Dict[str, "ConfigEntry"] | None = None


@dataclass
class ConfigEntry:
    experiments: List[ExperimentConfig]
    workers: int = 1


def _load_registry() -> Dict[str, ConfigEntry]:
    """Walk subdirectories and register any ``.py`` file that exports ``experiments``."""

    registry: Dict[str, ConfigEntry] = {}
    for sub_dir in _PACKAGE_DIR.iterdir():
        if not sub_dir.is_dir() or sub_dir.name.startswith("_"):
            continue
        for py_file in sub_dir.glob("*.py"):
            if py_file.name.startswith("_"):
                continue
            module_name = f"wmcal.configs.{sub_dir.name}.{py_file.stem}"
            spec = importlib.util.spec_from_file_location(module_name, py_file)
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = mod
            try:
                spec.loader.exec_module(mod)
            except Exception:
                sys.modules.pop(module_name, None)
                continue
            experiments = getattr(mod, "experiments", None)
            if isinstance(experiments, list) and all(
                isinstance(e, ExperimentConfig) for e in experiments
            ):
                key = f"{sub_dir.name}/{py_file.stem}"
                workers = getattr(mod, "WORKERS", 1)
                registry[key] = ConfigEntry(experiments=experiments, workers=workers)
    return registry


def __getattr__(name: str) -> Any:
    if name == "CONFIG_REGISTRY":
        global _REGISTRY
        if _REGISTRY is None:
            _REGISTRY = _load_registry()
        return _REGISTRY
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["CONFIG_REGISTRY", "ConfigEntry"]
