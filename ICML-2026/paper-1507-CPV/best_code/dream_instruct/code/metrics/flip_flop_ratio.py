#!/usr/bin/env python3
"""Compute paper-style flip-flop revision ratio from lm-eval sample JSONL files."""

import argparse
import json
from pathlib import Path
from typing import Any


def _sum_value(value: Any) -> int:
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, list):
        return sum(int(v) for v in value if isinstance(v, (int, float)))
    return 0


def _iter_samples(path: Path):
    files = [path] if path.is_file() else sorted(path.rglob("samples_*.jsonl"))
    for file_path in files:
        with file_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def _infer_strategy(path: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    text = str(path).lower()
    if "saber" in text:
        return "saber"
    if "wino" in text:
        return "wino"
    return "cover"


def compute(path: Path, strategy: str) -> dict[str, Any]:
    totals = {
        "n_samples": 0,
        "flip_flop_count": 0,
        "total_remask_count": 0,
        "total_unmask_count": 0,
        "replace_count": 0,
        "changed_after_remask_count": 0,
        "keep_count": 0,
    }
    for sample in _iter_samples(path):
        totals["n_samples"] += 1
        totals["flip_flop_count"] += _sum_value(sample.get("flip_flop_count"))
        totals["total_remask_count"] += _sum_value(
            sample.get("flip_flop_remask_count", sample.get("total_remask_count"))
        )
        totals["total_unmask_count"] += _sum_value(
            sample.get("flip_flop_unmask_count", sample.get("total_unmask_count"))
        )
        totals["replace_count"] += _sum_value(sample.get("replace_count"))
        totals["changed_after_remask_count"] += _sum_value(
            sample.get("changed_after_remask_count")
        )
        totals["keep_count"] += _sum_value(sample.get("keep_count"))

    if strategy == "cover":
        total_revisions = totals["total_remask_count"] + totals["replace_count"]
        effective_revisions = totals["changed_after_remask_count"] + totals["replace_count"]
        # Old COVER files may not have changed_after_remask_count.
        if totals["changed_after_remask_count"] == 0 and totals["total_remask_count"] > 0:
            effective_revisions = (
                totals["total_remask_count"] - totals["flip_flop_count"]
            ) + totals["replace_count"]
    elif strategy in {"wino", "saber"}:
        total_revisions = totals["total_remask_count"]
        effective_revisions = totals["total_remask_count"] - totals["flip_flop_count"]
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    effective_ratio = effective_revisions / total_revisions if total_revisions > 0 else None
    return {
        "strategy": strategy,
        **totals,
        "effective_revisions": effective_revisions,
        "total_revisions": total_revisions,
        "effective_ratio": effective_ratio,
    }


def format_summary(stats: dict[str, Any]) -> str:
    ratio = stats["effective_ratio"]
    ratio_text = "N/A" if ratio is None else f"{ratio * 100:.2f}%"
    return (
        "[flip-flop revision] "
        f"strategy={stats['strategy']} "
        f"samples={stats['n_samples']} "
        f"effective={stats['effective_revisions']} "
        f"total={stats['total_revisions']} "
        f"ratio={ratio_text} "
        f"remask={stats['total_remask_count']} "
        f"replace={stats['replace_count']} "
        f"changed_after_remask={stats['changed_after_remask_count']} "
        f"flip_flop={stats['flip_flop_count']} "
        f"keep={stats['keep_count']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--res_path", required=True, help="Result directory or sample JSONL file")
    parser.add_argument(
        "--strategy",
        choices=["auto", "cover", "wino", "saber"],
        default="auto",
        help="Revision formula to use. auto infers from path and defaults to cover.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output text file. Defaults to <res_path>/flip_flop_ratio.txt for directories.",
    )
    args = parser.parse_args()

    res_path = Path(args.res_path)
    strategy = _infer_strategy(res_path, args.strategy)
    stats = compute(res_path, strategy)
    summary = format_summary(stats)
    print(summary)

    out_path = Path(args.out) if args.out else None
    if out_path is None and res_path.is_dir():
        out_path = res_path / "flip_flop_ratio.txt"
    if out_path is not None:
        with out_path.open("w", encoding="utf-8") as f:
            f.write(summary + "\n")
            f.write(json.dumps(stats, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
