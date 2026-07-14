#!/usr/bin/env python3
"""
Dispatch k-fold macaque runs per-day for CEBRA, MARBLE, VDW, or LFADS.

Design:
- Expands day ranges (e.g., "0-3,7,10-12") into a sorted, deduplicated list.
- For each day, writes a temporary config that overrides:
  * training.root_dir -> --root_dir
  * training.save_dir -> <save_dir_base>/day_<day>/<model>
  * dataset.macaque_day_index -> <day>
- Invokes the appropriate runner sequentially on the current CUDA_VISIBLE_DEVICES
  (caller is expected to set it per-GPU).

Default configs:
- CEBRA / MARBLE / LFADS: config/yaml_files/macaque/experiment.yaml
- VDW: merge experiment.yaml with model config (default vdw_supcon_2.yaml)

Example usage:
python3 run_macaque_multiday_cv.py \
    --model=lfads \
    --days=3 \
    --root_dir=.../vdw
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

import yaml


def parse_days(days_str: str) -> List[int]:
    """
    Expand a day string like "0-3,7,10-12" into a sorted unique list of ints.
    """
    days: set[int] = set()
    if not days_str:
        return []
    parts = [p.strip() for p in days_str.split(",") if p.strip()]
    for part in parts:
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            try:
                start = int(start_str)
                end = int(end_str)
            except ValueError as exc:
                raise ValueError(f"Invalid day range segment '{part}'") from exc
            if start > end:
                start, end = end, start
            days.update(range(start, end + 1))
        else:
            try:
                days.add(int(part))
            except ValueError as exc:
                raise ValueError(f"Invalid day value '{part}'") from exc
    return sorted(days)


def deep_merge(base: Dict, override: Dict) -> Dict:
    """
    Recursively merge override into base (override wins).
    """
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def load_yaml(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(cfg: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)


def build_day_config(
    *,
    base_cfg_path: Path,
    model_cfg_path: Path | None,
    root_dir: Path,
    save_dir_base: Path,
    model_key: str,
    day: int,
) -> Path:
    base_cfg = load_yaml(base_cfg_path)
    merged_cfg = dict(base_cfg)
    if model_cfg_path is not None and model_cfg_path.exists():
        model_cfg = load_yaml(model_cfg_path)
        merged_cfg = deep_merge(merged_cfg, model_cfg)

    # Override root_dir, save_dir, and day index
    merged_cfg.setdefault("training", {})
    merged_cfg.setdefault("dataset", {})
    merged_cfg["training"]["root_dir"] = str(root_dir)
    # Keep save_dir relative to root_dir and use experiment_id per day to avoid nested duplicates
    rel_save_dir = Path(save_dir_base).relative_to(root_dir) if save_dir_base.is_absolute() else save_dir_base
    merged_cfg["training"]["save_dir"] = str(rel_save_dir)
    merged_cfg["training"]["experiment_id"] = f"day_{day}"
    merged_cfg["dataset"]["macaque_day_index"] = int(day)

    tmp_dir = Path(tempfile.mkdtemp(prefix=f"macaque_day{day}_{model_key}_"))
    cfg_path = tmp_dir / "config.yaml"
    write_yaml(merged_cfg, cfg_path)
    return cfg_path


def run_cmd(cmd: List[str]) -> int:
    print(f"[INFO] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-day macaque CV per GPU.")
    parser.add_argument("--model", required=True, choices=["cebra", "marble", "vdw", "lfads"])
    parser.add_argument("--days", required=True, help="Day list like '0-3,7,10-12'.")
    parser.add_argument(
        "--root_dir",
        type=Path,
        required=True,
        help="Project root (e.g., /bsuscratch/.../vdw).",
    )
    parser.add_argument(
        "--base_config",
        type=Path,
        help="Base experiment YAML. Defaults per model.",
    )
    parser.add_argument(
        "--model_config",
        type=Path,
        help="Optional model YAML to merge (used for VDW).",
    )
    parser.add_argument(
        "--save_dir_base",
        type=Path,
        default=None,
        help="Base relative or absolute save dir (default: <root_dir>/experiments/macaque_reaching).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print planned commands without executing.",
    )
    parser.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="Additional args passed through to the model runner.",
    )
    args = parser.parse_args()

    days = parse_days(args.days)
    if not days:
        raise ValueError("No days specified after parsing.")

    model = args.model.lower()
    root_dir = args.root_dir.expanduser().resolve()
    code_dir = root_dir / "code"

    default_base = (
        code_dir / "config" / "yaml_files" / "macaque" / "marble.yaml"
        if model == "marble"
        else code_dir / "config" / "yaml_files" / "macaque" / "experiment.yaml"
    )
    base_cfg_path = args.base_config or default_base
    if not base_cfg_path.is_absolute():
        base_cfg_path = code_dir / base_cfg_path
    base_cfg_path = base_cfg_path.resolve()

    model_cfg_path: Path | None = None
    if model == "vdw":
        default_model_cfg = code_dir / "config" / "yaml_files" / "macaque" / "vdw_supcon_2.yaml"
        if args.model_config:
            model_cfg_path = args.model_config
            if not model_cfg_path.is_absolute():
                model_cfg_path = code_dir / model_cfg_path
            model_cfg_path = model_cfg_path.resolve()
        else:
            model_cfg_path = default_model_cfg.resolve()

    save_dir_base = args.save_dir_base
    if save_dir_base is None:
        save_dir_base = Path("experiments") / "macaque_reaching"
    else:
        save_dir_base = save_dir_base if save_dir_base.is_absolute() else Path(save_dir_base)

    python_executable = sys.executable
    runlist: List[List[str]] = []
    for day in days:
        day_cfg = build_day_config(
            base_cfg_path=base_cfg_path,
            model_cfg_path=model_cfg_path,
            root_dir=root_dir,
            save_dir_base=save_dir_base,
            model_key=model,
            day=day,
        )
        if model == "cebra":
            runner = code_dir / "scripts" / "python" / "run_cebra_macaque_inductive.py"
            cmd = [python_executable, str(runner), "--config", str(day_cfg), "--day_idx", str(day)]
        elif model == "marble":
            runner = code_dir / "scripts" / "python" / "run_marble_macaque_inductive.py"
            cmd = [python_executable, str(runner), "--config", str(day_cfg)]
        elif model == "lfads":
            runner = code_dir / "scripts" / "python" / "run_lfads_macaque_inductive.py"
            cmd = [python_executable, str(runner), "--config", str(day_cfg), "--day_idx", str(day)]
        else:
            runner = code_dir / "scripts" / "python" / "main_training.py"
            cmd = [
                python_executable,
                str(runner),
                "--config",
                str(day_cfg),
                "--dataset",
                "macaque",
            ]
        if args.extra:
            cmd.extend(args.extra)
        runlist.append(cmd)
        print(f"[PLAN] day {day} -> {cmd}")

    if args.dry_run:
        print("[DRY RUN] Skipping execution.")
        return

    for cmd in runlist:
        code = run_cmd(cmd)
        if code != 0:
            raise RuntimeError(f"Command failed with exit code {code}: {' '.join(cmd)}")

    # Cleanup temp directories best-effort
    for cmd in runlist:
        try:
            for token in cmd:
                if token.endswith("config.yaml"):
                    tmp_path = Path(token)
                    tmp_dir = tmp_path.parent
                    if tmp_dir.exists() and "macaque_day" in tmp_dir.name:
                        shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception as exc:  # print warning only
            print(f"[WARN] Cleanup failed: {exc}")


if __name__ == "__main__":
    main()

