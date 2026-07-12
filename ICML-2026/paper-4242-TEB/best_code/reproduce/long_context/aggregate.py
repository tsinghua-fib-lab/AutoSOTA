"""
Aggregate per-context-length bench JSONs into long_context_summary.csv.

`run_long_context_comparison.sh` writes one directory per (model, context-len,
concurrency) of the form
    reproduce/outputs/long_context_<MODEL>_i<INPUT_LEN>_o<OUTPUT_LEN>_c<C>_<ts>/
each containing bench_v1.json, bench_eb.json, bench_ebplus.json.

`plot_long_context.py` expects a CSV with one row per (context_len, scheduler):
    context_len, scheduler, throughput, mean_ttft_ms, mean_tpot_ms, mean_itl_ms

Usage
-----
    python aggregate.py reproduce/outputs/
    # → reproduce/outputs/long_context_summary.csv

Or restrict to a specific model:
    python aggregate.py reproduce/outputs/ --model Qwen3-8B

Or pass explicit dirs:
    python aggregate.py --dirs reproduce/outputs/long_context_Qwen3-8B_i131072_*
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

SCHEDULERS = ("v1", "eb", "ebplus")
# Legacy filenames from older revisions of run_long_context_comparison.sh.
LEGACY_NAMES = {"v1": "cp", "eb": "theta_eb"}

DIR_RE = re.compile(
    r"^long_context_(?P<model>.+?)_i(?P<input_len>\d+)_o(?P<output_len>\d+)"
    r"(?:_c(?P<concurrency>\d+))?_(?P<ts>\d{8}_\d{6})$"
)


def parse_dir(d: Path) -> dict | None:
    m = DIR_RE.match(d.name)
    if not m:
        return None
    return {
        "model": m.group("model"),
        "input_len": int(m.group("input_len")),
        "output_len": int(m.group("output_len")),
        "concurrency": int(m.group("concurrency")) if m.group("concurrency") else None,
        "ts": m.group("ts"),
    }


def load_metrics(json_path: Path) -> dict | None:
    if not json_path.exists():
        return None
    with open(json_path) as f:
        d = json.load(f)
    return {
        "throughput": d.get("request_throughput", d.get("total_token_throughput", 0)),
        "mean_ttft_ms": d.get("mean_ttft_ms", 0),
        "mean_tpot_ms": d.get("mean_tpot_ms", 0),
        "mean_itl_ms": d.get("mean_itl_ms", d.get("median_itl_ms", 0)),
    }


def aggregate(dirs: list[Path], model_filter: str | None) -> list[dict]:
    rows: list[dict] = []
    for d in dirs:
        meta = parse_dir(d)
        if meta is None:
            continue
        if model_filter and meta["model"] != model_filter:
            continue
        for sched in SCHEDULERS:
            jf = d / f"bench_{sched}.json"
            if not jf.exists() and sched in LEGACY_NAMES:
                jf = d / f"bench_{LEGACY_NAMES[sched]}.json"
            m = load_metrics(jf)
            if m is None:
                continue
            rows.append({
                "model": meta["model"],
                "context_len": meta["input_len"],
                "concurrency": meta["concurrency"] or "",
                "scheduler": sched,
                **m,
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parent", nargs="?", type=Path,
                    help="Parent dir containing long_context_*_<ts> subdirs")
    ap.add_argument("--dirs", nargs="+", type=Path, default=None,
                    help="Explicit list of long_context dirs")
    ap.add_argument("--model", default=None,
                    help="Only aggregate dirs matching this model short name")
    ap.add_argument("--output", type=Path, default=None,
                    help="Output CSV path (default: <parent>/long_context_summary.csv)")
    args = ap.parse_args()

    if args.dirs:
        dirs = list(args.dirs)
        default_parent = dirs[0].parent
    elif args.parent is not None:
        dirs = sorted(p for p in args.parent.iterdir()
                      if p.is_dir() and p.name.startswith("long_context_"))
        default_parent = args.parent
    else:
        ap.error("provide either `parent` positional or `--dirs`")
    if not dirs:
        ap.error("no long_context_* dirs found")

    output = args.output or (default_parent / "long_context_summary.csv")
    rows = aggregate(dirs, args.model)

    if not rows:
        print("No matching bench JSONs found; nothing written.")
        return

    fieldnames = ["model", "context_len", "concurrency", "scheduler",
                  "throughput", "mean_ttft_ms", "mean_tpot_ms", "mean_itl_ms"]
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
