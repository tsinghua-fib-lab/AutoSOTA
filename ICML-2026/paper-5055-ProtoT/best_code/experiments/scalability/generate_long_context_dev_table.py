#!/usr/bin/env python3
"""
Generate long-context scalability table from EasyTuna logs.

By default, this scans:
  - logs/ablation_*
  - logs/scalability_*

It extracts final dev perplexity (best_trial.value from results.json),
maps studies into a fixed table layout, and prints a LaTeX table.
Missing cells are shown as "-".
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SMALL_TRAIN_MAX = 50_000
LARGE_TRAIN_MIN = 300_000
COLS = ["Default", "512", "1024", "2048", "Large-scale"]
SMALL_COL_TO_SEQ = {
    "Default": 256,
    "512": 512,
    "1024": 1024,
    "2048": 2048,
}


@dataclass
class Study:
    path: Path
    value: float
    model: str
    seq_len: int
    bottleneck: int
    layers: int
    train_size: int
    r: int
    dropout: float
    attn_dropout: float
    proto_alpha_init: float
    proto_disable_mass_norm: bool
    proto_disable_value_lowrank: bool


@dataclass
class RowSpec:
    label: str
    section: str
    model: str
    small_bottleneck: int
    small_layers: int
    large_bottleneck: int
    large_layers: int
    r: int | None = None
    require_proto_defaults: bool = False


def _as_bool(v: Any, default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return default
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "y", "on"}
    return default


def _as_int(v: Any, default: int) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _as_float(v: Any, default: float) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _discover_study_dirs(logs_root: Path, patterns: list[str]) -> list[Path]:
    out: list[Path] = []
    for pattern in patterns:
        for exp_dir in sorted(logs_root.glob(pattern)):
            if not exp_dir.is_dir():
                continue
            for candidate in sorted(exp_dir.iterdir()):
                if not candidate.is_dir():
                    continue
                if (candidate / "results.json").is_file() or (candidate / "trial_runs").is_dir():
                    out.append(candidate)
    return out


def _find_trial_dir(study_dir: Path, trial_number: int | None) -> Path | None:
    trial_runs = study_dir / "trial_runs"
    if not trial_runs.is_dir():
        return None
    if trial_number is not None:
        p = trial_runs / f"trial{trial_number:03d}"
        if p.is_dir():
            return p
    trials = sorted([p for p in trial_runs.iterdir() if p.is_dir() and p.name.startswith("trial")])
    return trials[0] if trials else None


def _load_args_for_trial(trial_dir: Path | None) -> dict[str, Any] | None:
    if trial_dir is None:
        return None

    direct = trial_dir / "args.json"
    data = _load_json(direct)
    if data is not None:
        return data

    for seed_dir in sorted(trial_dir.glob("seed_*")):
        data = _load_json(seed_dir / "args.json")
        if data is not None:
            return data
    return None


def _metric_from_trial_files(trial_dir: Path | None) -> float | None:
    if trial_dir is None:
        return None

    vals: list[float] = []
    direct = trial_dir / "final_val_ppl.txt"
    if direct.is_file():
        try:
            vals.append(float(direct.read_text().strip()))
        except (OSError, ValueError):
            pass

    for seed_dir in sorted(trial_dir.glob("seed_*")):
        p = seed_dir / "final_val_ppl.txt"
        if not p.is_file():
            continue
        try:
            vals.append(float(p.read_text().strip()))
        except (OSError, ValueError):
            continue

    if not vals:
        return None
    return sum(vals) / len(vals)


def _load_study(study_dir: Path) -> Study | None:
    results = _load_json(study_dir / "results.json") or {}
    best_trial = results.get("best_trial", {}) if isinstance(results.get("best_trial"), dict) else {}
    trial_number = best_trial.get("number")
    trial_number = int(trial_number) if isinstance(trial_number, int) else None

    trial_dir = _find_trial_dir(study_dir, trial_number)
    args = _load_args_for_trial(trial_dir)
    if args is None:
        return None

    value = None
    if isinstance(best_trial.get("value"), (int, float)):
        value = float(best_trial["value"])
    elif isinstance(results.get("best_value"), (int, float)):
        value = float(results["best_value"])
    else:
        value = _metric_from_trial_files(trial_dir)
    if value is None:
        return None

    model = str(args.get("MODEL", "")).strip().lower()

    return Study(
        path=study_dir,
        value=value,
        model=model,
        seq_len=_as_int(args.get("SEQ_LEN"), 256),
        bottleneck=_as_int(args.get("BOTTLENECK"), 256),
        layers=_as_int(args.get("LAYERS"), 6),
        train_size=_as_int(args.get("TRAIN_SIZE"), 18_000),
        r=_as_int(args.get("R"), 32),
        dropout=_as_float(args.get("DROPOUT"), 0.1),
        attn_dropout=_as_float(args.get("ATTN_DROPOUT"), 0.1),
        proto_alpha_init=_as_float(args.get("PROTO_ALPHA_INIT"), 1.0),
        proto_disable_mass_norm=_as_bool(args.get("PROTO_DISABLE_MASS_NORM"), False),
        proto_disable_value_lowrank=_as_bool(args.get("PROTO_DISABLE_VALUE_LOWRANK"), False),
    )


def _is_default_dropout(study: Study) -> bool:
    return abs(study.dropout - 0.1) < 1e-9 and abs(study.attn_dropout - 0.1) < 1e-9


def _is_default_proto(study: Study) -> bool:
    return (
        abs(study.proto_alpha_init - 1.0) < 1e-9
        and not study.proto_disable_mass_norm
        and not study.proto_disable_value_lowrank
    )


def _row_matches_col(study: Study, row: RowSpec, col: str) -> bool:
    if study.model != row.model:
        return False
    if not _is_default_dropout(study):
        return False
    if row.require_proto_defaults and not _is_default_proto(study):
        return False
    if row.r is not None and study.r != row.r:
        return False

    if col == "Large-scale":
        return (
            study.train_size >= LARGE_TRAIN_MIN
            and study.seq_len == 512
            and study.bottleneck == row.large_bottleneck
            and study.layers == row.large_layers
        )

    seq = SMALL_COL_TO_SEQ[col]
    return (
        study.train_size <= SMALL_TRAIN_MAX
        and study.seq_len == seq
        and study.bottleneck == row.small_bottleneck
        and study.layers == row.small_layers
    )


def _fmt(v: float | None, bold: bool = False) -> str:
    if v is None:
        return "--"
    txt = f"{v:.1f}"
    return f"\\textbf{{{txt}}}" if bold else txt


def _build_rows() -> list[RowSpec]:
    return [
        RowSpec(
            label="LLaMA",
            section="Main models",
            model="llama",
            small_bottleneck=256,
            small_layers=6,
            large_bottleneck=512,
            large_layers=12,
        ),
        RowSpec(
            label="Mamba",
            section="Main models",
            model="mamba",
            small_bottleneck=256,
            small_layers=6,
            large_bottleneck=512,
            large_layers=12,
        ),
        RowSpec(
            label="DeltaNet",
            section="Main models",
            model="deltanet",
            small_bottleneck=256,
            small_layers=6,
            large_bottleneck=512,
            large_layers=12,
        ),
        RowSpec(
            label="ProtoT",
            section="Main models",
            model="prototypeattn",
            small_bottleneck=256,
            small_layers=6,
            large_bottleneck=512,
            large_layers=12,
            r=32,
            require_proto_defaults=True,
        ),
        RowSpec(
            label="ProtoT (h=512)",
            section="ProtoT variants",
            model="prototypeattn",
            small_bottleneck=512,
            small_layers=6,
            large_bottleneck=512,
            large_layers=6,
            r=32,
            require_proto_defaults=True,
        ),
        RowSpec(
            label="ProtoT (L=12)",
            section="ProtoT variants",
            model="prototypeattn",
            small_bottleneck=256,
            small_layers=12,
            large_bottleneck=256,
            large_layers=12,
            r=32,
            require_proto_defaults=True,
        ),
        RowSpec(
            label="ProtoT (R=64)",
            section="ProtoT variants",
            model="prototypeattn",
            small_bottleneck=256,
            small_layers=6,
            large_bottleneck=256,
            large_layers=6,
            r=64,
            require_proto_defaults=True,
        ),
    ]


def _arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate long-context scalability table from logs.")
    p.add_argument(
        "--logs-root",
        type=Path,
        default=Path("logs"),
        help="Root logs directory (default: logs).",
    )
    p.add_argument(
        "--patterns",
        nargs="+",
        default=["ablation_*", "scalability_*"],
        help='Experiment directory glob patterns inside logs root (default: "ablation_* scalability_*").',
    )
    p.add_argument(
        "--show-sources",
        action="store_true",
        help="Print selected study path for each non-empty table cell.",
    )
    return p


def main() -> None:
    args = _arg_parser().parse_args()
    rows = _build_rows()
    studies = []
    for d in _discover_study_dirs(args.logs_root, args.patterns):
        study = _load_study(d)
        if study is not None:
            studies.append(study)

    # cells[(row_label, col)] = (best_value, best_path)
    cells: dict[tuple[str, str], tuple[float, Path] | None] = {}
    for row in rows:
        for col in COLS:
            best: tuple[float, Path] | None = None
            for study in studies:
                if not _row_matches_col(study, row, col):
                    continue
                if best is None or study.value < best[0]:
                    best = (study.value, study.path)
            cells[(row.label, col)] = best

    # Bold minima per section+column.
    min_per_section_col: dict[tuple[str, str], float] = {}
    for row in rows:
        for col in COLS:
            cell = cells[(row.label, col)]
            if cell is None:
                continue
            key = (row.section, col)
            if key not in min_per_section_col or cell[0] < min_per_section_col[key]:
                min_per_section_col[key] = cell[0]

    print("\\begin{table}[t]")
    print("\\centering")
    print("\\begin{tabular}{lccccc}")
    print("\\hline")
    print("Model & Default & 512 & 1024 & 2048 & Large-scale \\\\")
    print("\\hline")
    for row in rows:
        vals = []
        for col in COLS:
            cell = cells[(row.label, col)]
            value = None if cell is None else cell[0]
            is_bold = value is not None and min_per_section_col.get((row.section, col)) == value
            vals.append(_fmt(value, bold=is_bold))
        print(f"{row.label} & {vals[0]} & {vals[1]} & {vals[2]} & {vals[3]} & {vals[4]} \\\\")
    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")

    if args.show_sources:
        print("\nSelected sources:")
        for row in rows:
            for col in COLS:
                cell = cells[(row.label, col)]
                if cell is None:
                    continue
                print(f"- {row.label} / {col}: {cell[0]:.4f} <- {cell[1]}")


if __name__ == "__main__":
    main()
