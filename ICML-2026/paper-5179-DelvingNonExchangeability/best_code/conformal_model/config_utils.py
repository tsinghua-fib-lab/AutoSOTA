"""Config helpers for run dir/name resolution."""

from __future__ import annotations

import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


_NOW_PATTERN = re.compile(r"\$\{now:([^}]+)\}")


def _lookup(cfg: Mapping[str, Any], path: str) -> Any:
    cur: Any = cfg
    for key in path.split("."):
        if not isinstance(cur, Mapping) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _replace_now(text: str, now: datetime) -> str:
    def _sub(match: re.Match[str]) -> str:
        fmt = match.group(1)
        return now.strftime(fmt)

    return _NOW_PATTERN.sub(_sub, text)


def resolve_template(text: str, cfg: Mapping[str, Any], root: Path) -> str:
    now = datetime.now()
    out = text
    out = out.replace("${hydra:runtime.cwd}", str(root))
    out = out.replace("${dataset.name}", str(_lookup(cfg, "dataset.name") or "dataset"))
    out = out.replace("${model.name}", str(_lookup(cfg, "model.name") or "model"))
    out = out.replace("${method.name}", str(_lookup(cfg, "method.name") or "method"))
    out = out.replace("${run.seed}", str(_lookup(cfg, "run.seed") or "0"))
    out = _replace_now(out, now)
    return out


def ensure_run_fields(cfg: dict, *, root: Path, default_dir: str | None = None) -> dict:
    cfg = dict(cfg)
    run_cfg = dict(cfg.get("run", {}) or {})

    seed = run_cfg.get("seed")
    if seed is None:
        seed = random.randint(0, 999_999_999)
        run_cfg["seed"] = seed

    dir_template = run_cfg.get("dir") or default_dir or "logs"
    resolved_dir = resolve_template(str(dir_template), {**cfg, "run": run_cfg}, root)
    if not Path(resolved_dir).is_absolute():
        resolved_dir = str((root / resolved_dir).resolve())
    if Path(resolved_dir).exists():
        try:
            has_items = any(Path(resolved_dir).iterdir())
        except Exception:
            has_items = False
        if has_items:
            stamp = now.strftime("%Y-%m-%d_%H-%M-%S_%f")
            resolved_dir = str(Path(resolved_dir).with_name(f"{Path(resolved_dir).name}_{stamp}"))
    run_cfg["dir"] = resolved_dir

    name_template = run_cfg.get("name") or "${now:%Y-%m-%d_%H-%M-%S}_${run.seed}"
    run_cfg["name"] = resolve_template(str(name_template), {**cfg, "run": run_cfg}, root)

    cfg["run"] = run_cfg
    return cfg
