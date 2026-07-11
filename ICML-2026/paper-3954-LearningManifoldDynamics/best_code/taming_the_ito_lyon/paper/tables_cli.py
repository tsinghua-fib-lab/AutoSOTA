from __future__ import annotations

import argparse
from pathlib import Path

from taming_the_ito_lyon.paper.tables_rough_vol import render_rough_vol_table
from taming_the_ito_lyon.paper.tables_so3 import render_so3_table


def _parse_times(raw: list[str] | None) -> list[int]:
    if raw is None:
        return [128, 256, 384, 512]
    times: list[int] = []
    for token in raw:
        for chunk in token.split(","):
            chunk_stripped = chunk.strip()
            if chunk_stripped:
                times.append(int(chunk_stripped))
    if len(times) == 0:
        raise ValueError("At least one time point is required.")
    return times


def _parse_seeds(raw: list[str] | None) -> list[int] | None:
    if raw is None:
        return None
    seeds: list[int] = []
    for token in raw:
        for chunk in token.split(","):
            chunk_stripped = chunk.strip()
            if chunk_stripped:
                seeds.append(int(chunk_stripped))
    return seeds if len(seeds) > 0 else None


def _is_metrics_dir(path: Path) -> bool:
    return (path / "test_metrics.json").exists() or (path / "metrics.json").exists()


def _expand_run_dirs(paths: list[Path]) -> list[Path]:
    expanded: list[Path] = []
    for path in paths:
        if _is_metrics_dir(path):
            expanded.append(path)
            continue
        if path.is_dir():
            subdirs = [p for p in path.iterdir() if p.is_dir()]
            for subdir in sorted(subdirs):
                if _is_metrics_dir(subdir):
                    expanded.append(subdir)
    if len(expanded) == 0:
        raise ValueError("No run directories with metrics files were found.")
    return expanded


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper tables (LaTeX).")
    parser.add_argument(
        "--table",
        type=str,
        required=True,
        choices=["rough_vol", "so3"],
        help="Which table to generate.",
    )
    parser.add_argument(
        "--run_dir",
        nargs="+",
        type=str,
        required=True,
        help="Run directories to include (e.g. saved_models/run_1 saved_models/run_2).",
    )
    parser.add_argument(
        "--metrics-file",
        type=str,
        default=None,
        help="Metrics JSON filename to read (default: test_metrics.json if present, else metrics.json).",
    )
    parser.add_argument(
        "--times",
        nargs="+",
        type=str,
        default=None,
        help="KS time points (e.g. --times 128 256 384 512 or --times 128,256,384,512).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional output path to write the LaTeX table.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=str,
        default=None,
        help="Seeds to aggregate (e.g. --seeds 1 2 3 or --seeds 1,2,3).",
    )
    args = parser.parse_args()

    run_dirs = _expand_run_dirs([Path(p) for p in args.run_dir])
    times = _parse_times(args.times)
    seeds = _parse_seeds(args.seeds)

    if args.table == "rough_vol":
        table = render_rough_vol_table(
            run_dirs=run_dirs,
            times=times,
            metrics_file=args.metrics_file,
            seeds=seeds,
        )
    elif args.table == "so3":
        table = render_so3_table(
            run_dirs=run_dirs,
            metrics_file=args.metrics_file,
            seeds=seeds,
        )
    else:
        raise ValueError(f"Unknown table '{args.table}'")

    if args.out is None:
        print(table)
    else:
        out_path = Path(args.out)
        out_path.write_text(table, encoding="utf-8")


if __name__ == "__main__":
    main()
