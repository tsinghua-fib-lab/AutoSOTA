"""
Aggregate per-workload grid_summary.json files into optimal_per_scheduler.json.

The per-workload grid-search outputs from `run_grid_search.sh` end up in
separate directories named e.g.
    reproduce/outputs/grid_search_sharegpt_prompts_Qwen3-8B_Con_2048_Prompts_4000/
each containing `grid_summary.json` produced by `analyze_grid_search.py`.

The downstream plotting scripts (`plot_real_workload_latency.py`,
`plot_scalmodel.py`) expect a single combined file with this shape:

    {
      "ShareGPT":   {"v0": {...}, "v1": {...}, "eb": {...}},
      "LongBench":  {...},
      "WildChat":   {...},
      "NuminaMath": {...}
    }

This script walks the per-workload dirs, picks the best (TB, BS) per scheduler
from each, and writes the combined `optimal_per_scheduler.json`.

Usage
-----
Auto-discover all per-workload dirs under a parent and write the combined
output into that parent (the standard reproduction flow):

    python build_optimal_json.py reproduce/outputs/

Or pass per-workload dirs explicitly (useful for scalability sweeps where the
dirs span multiple parents):

    python build_optimal_json.py \\
        --dirs reproduce/outputs/grid_search_sharegpt_prompts_Qwen3-8B_Con_2048_Prompts_4000 \\
               reproduce/outputs/grid_search_longbench_prefill_Qwen3-8B_Con_2048_Prompts_4000 \\
               ... \\
        --output reproduce/outputs/optimal_per_scheduler.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


# Map dataset-basename prefixes (as they appear in the grid_search dir name)
# to the canonical workload labels used by the plotting scripts. The dir name
# format is `grid_search_<DATASET_NAME>_<MODEL>_Con_<c>_Prompts_<n>...`, and
# DATASET_NAME is the basename of the JSONL passed to run_grid_search.sh.
WORKLOAD_PREFIXES = [
    ("sharegpt", "ShareGPT"),
    ("longbench", "LongBench"),
    ("wildchat", "WildChat"),
    ("numina", "NuminaMath"),
]

# The plotting scripts read this flat set of fields from each scheduler entry.
METRIC_KEYS = ("throughput", "output_throughput", "mean_ttft_ms",
               "mean_tpot_ms", "median_ttft_ms", "p99_ttft_ms",
               "median_tpot_ms", "p99_tpot_ms")


def classify_workload(dir_name: str) -> str | None:
    """Return canonical workload label, or None if not recognised.

    Recognises both single-turn grid_search dirs (`grid_search_<prefix>_...`)
    and the multi-turn WildChat output dir (`multiturn_wildchat_*`).
    """
    lower = dir_name.lower()
    if re.search(r"multiturn_wildchat[^/]*", lower):
        return "WildChat"
    for prefix, label in WORKLOAD_PREFIXES:
        # The dir name embeds the dataset basename right after `grid_search_`;
        # match against that token to avoid false positives if a model name
        # happens to contain one of these substrings.
        if re.search(rf"grid_search_{prefix}[^/]*", lower):
            return label
    return None


def load_optimal_for_dir(grid_dir: Path) -> dict | None:
    """Return {scheduler: flat-metrics-dict} from one grid_summary.json."""
    summary = grid_dir / "grid_summary.json"
    if not summary.exists():
        print(f"  [skip] {grid_dir}: no grid_summary.json "
              f"(did you forget `analyze_grid_search.py {grid_dir}`?)")
        return None

    with open(summary) as f:
        data = json.load(f)

    optimal = data.get("optimal", {})
    flat = {}
    for sched, entry in optimal.items():
        metrics = entry.get("metrics", {})
        flat[sched] = {k: metrics.get(k) for k in METRIC_KEYS if k in metrics}
        # Also surface the chosen (tb, bs) so the plot scripts can annotate.
        flat[sched]["tb"] = entry.get("tb")
        flat[sched]["bs"] = entry.get("bs")
    return flat


def discover(parent: Path) -> list[Path]:
    """List per-workload result dirs under a parent (both grid_search_* and
    multi-turn multiturn_wildchat_*)."""
    return sorted(p for p in parent.iterdir() if p.is_dir() and (
        p.name.startswith("grid_search_") or p.name.startswith("multiturn_")
    ))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parent", nargs="?", type=Path,
                    help="Parent dir containing grid_search_<workload>_... subdirs")
    ap.add_argument("--dirs", nargs="+", type=Path, default=None,
                    help="Explicit list of per-workload grid-search dirs")
    ap.add_argument("--output", type=Path, default=None,
                    help="Output path (default: <parent>/optimal_per_scheduler.json)")
    args = ap.parse_args()

    if args.dirs:
        grid_dirs = list(args.dirs)
        default_output_parent = grid_dirs[0].parent
    elif args.parent is not None:
        grid_dirs = discover(args.parent)
        default_output_parent = args.parent
    else:
        ap.error("provide either `parent` positional or `--dirs`")
    if not grid_dirs:
        ap.error("no per-workload grid-search dirs found")

    output = args.output or (default_output_parent / "optimal_per_scheduler.json")

    combined: dict[str, dict] = {}
    for d in grid_dirs:
        label = classify_workload(d.name)
        if label is None:
            print(f"  [skip] {d.name}: workload prefix not recognised "
                  f"(expected one of {[p for p,_ in WORKLOAD_PREFIXES]})")
            continue
        flat = load_optimal_for_dir(d)
        if flat is None:
            continue
        if label in combined:
            print(f"  [warn] duplicate workload {label}: overwriting "
                  f"({combined[label].get('_source')} → {d.name})")
        combined[label] = flat
        combined[label]["_source"] = d.name
        print(f"  [ok]   {label:<10} ← {d.name}")

    if not combined:
        print("No usable per-workload summaries; nothing written.")
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nWrote {output}  (workloads: {sorted(combined.keys())})")


if __name__ == "__main__":
    main()
