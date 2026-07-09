#!/usr/bin/env python3


from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_int_suffix(name: str, prefix: str) -> int | None:
    if not name.startswith(prefix):
        return None
    rest = name[len(prefix):]
    return int(rest) if rest.isdigit() else None


def read_single_float(p: Path) -> float:
    return float(p.read_text(encoding="utf-8").strip())


def build_iteration_list(k_dir: Path) -> list[float | None]:
    vals: dict[int, float] = {}
    for child in k_dir.iterdir():
        if not child.is_file():
            continue
        idx = parse_int_suffix(child.name, "itr-")
        if idx is None:
            continue
        try:
            vals[idx] = read_single_float(child)
        except:
            pass

    if not vals:
        return []

    max_idx = max(vals)
    out: list[float | None] = [None] * (max_idx + 1)
    for i, v in vals.items():
        out[i] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert results/<run>/<dataset>/<algo>/k-*/itr-* files into structured JSON."
    )
    ap.add_argument(
        "results_root",
        nargs="?",
        type=Path,
        default=Path("results"),
        help="Path to the results/ directory (default: results).",
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("results.json"),
        help="Output JSON file path (default: results.json).",
    )
    args = ap.parse_args()

    root: Path = args.results_root
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    data: dict[str, Any] = {}

    for run_dir in sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name):
        run_obj: dict[str, Any] = {}
        for dataset_dir in sorted((p for p in run_dir.iterdir() if p.is_dir()), key=lambda p: p.name):
            dataset_obj: dict[str, Any] = {}
            for algo_dir in sorted((p for p in dataset_dir.iterdir() if p.is_dir()), key=lambda p: p.name):
                algo_obj: dict[str, Any] = {}
                for k_dir in sorted((p for p in algo_dir.iterdir() if p.is_dir()), key=lambda p: p.name):
                    algo_obj[k_dir.name] = build_iteration_list(k_dir)
                dataset_obj[algo_dir.name] = algo_obj
            run_obj[dataset_dir.name] = dataset_obj
        data[run_dir.name] = run_obj

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
