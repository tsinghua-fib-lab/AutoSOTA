#!/usr/bin/env python3
"""
Run bbob-noisy experiments in parallel by splitting across algorithms.

Why:
- COCO suites are naturally "embarrassingly parallel" across algorithm datasets.
- The single-process runner is simple/reliable but can be slow when evaluating
  many algorithms/instances/budgets.

This script:
1) launches `run_coco_bbob_noisy.py` once per algorithm (subprocesses),
2) merges `bbob_summary.csv` + `trace_index.csv` into a single folder,
3) generates quick rank-based plots/metrics via `plot_bbob_results.py`,
4) writes a helper file listing matching COCO `exdata/` folders for cocopp.

Note:
- For cocopp (ERT/ECDF) you should use the COCO observer output under `exdata/`.
  This script does NOT merge COCO observer data; it just makes it easy to find.
"""

import argparse
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from _project import BASE_DIR, repo_relpath


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def sanitize(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower())
    s = s.strip("_")
    return s or "unnamed"


def run_one(cmd: list[str]) -> int:
    proc = subprocess.run(cmd, cwd=BASE_DIR)
    return int(proc.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        default=os.path.join(BASE_DIR, "Results", "noisy_parallel"),
        help="Directory to write per-algorithm parts and merged outputs.",
    )
    parser.add_argument("--dims", default="10,20,40")
    parser.add_argument("--budgets", default="200,500", help="Budget multipliers (xD)")
    parser.add_argument("--functions", default="1-30", help="Suite function indices (1–30 for bbob-noisy)")
    parser.add_argument("--instances", default="1-5", help="Instance indices, e.g. '1-5'")
    parser.add_argument(
        "--algorithms",
        required=True,
        help="Comma-separated algorithms to run (each runs in its own subprocess).",
    )
    parser.add_argument("--tag", default="", help="Tag passed through to run_coco_bbob_noisy.py (recommended).")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel subprocesses.")
    args = parser.parse_args()

    results_dir = os.path.abspath(args.results_dir)
    os.makedirs(results_dir, exist_ok=True)

    algos = [a.strip() for a in str(args.algorithms).split(",") if a.strip()]
    if not algos:
        raise SystemExit("No algorithms specified.")

    parts = []
    cmds = []
    for algo in algos:
        part_dir = os.path.join(results_dir, f"part_{sanitize(algo)}")
        os.makedirs(part_dir, exist_ok=True)
        parts.append(part_dir)
        cmd = [
            sys.executable,
            os.path.join(SCRIPT_DIR, "run_coco_bbob_noisy.py"),
            "--results-dir",
            part_dir,
            "--dims",
            str(args.dims),
            "--budgets",
            str(args.budgets),
            "--functions",
            str(args.functions),
            "--instances",
            str(args.instances),
            "--algorithms",
            algo,
            "--tag",
            str(args.tag),
        ]
        cmds.append(cmd)

    failures = 0
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
        futures = {ex.submit(run_one, cmd): cmd for cmd in cmds}
        for fut in as_completed(futures):
            cmd = futures[fut]
            rc = int(fut.result())
            if rc != 0:
                failures += 1
                print("FAILED:", " ".join(cmd), "rc=", rc)

    if failures:
        raise SystemExit(f"{failures} subprocess(es) failed.")

    merged_dir = os.path.join(results_dir, "merged")
    merge_cmd = [
        sys.executable,
        os.path.join(SCRIPT_DIR, "merge_results.py"),
        "--output-dir",
        merged_dir,
        "--input-dirs",
        *parts,
    ]
    subprocess.check_call(merge_cmd, cwd=BASE_DIR)

    plot_cmd = [
        sys.executable,
        os.path.join(SCRIPT_DIR, "plot_bbob_results.py"),
        "--results-dir",
        merged_dir,
    ]
    subprocess.check_call(plot_cmd, cwd=BASE_DIR)

    # Quick statistic (exact sign test on final best_f).
    sign_cmd = [
        sys.executable,
        os.path.join(SCRIPT_DIR, "pairwise_sign_test.py"),
        "--results-dir",
        merged_dir,
    ]
    subprocess.check_call(sign_cmd, cwd=BASE_DIR)

    # Help cocopp discovery by listing matching `exdata/noisy_<tag>_*` folders.
    tag = str(args.tag).strip()
    exdata_dir = os.path.join(BASE_DIR, "exdata")
    if tag and os.path.isdir(exdata_dir):
        prefix = f"noisy_{sanitize(tag)}_"
        matches = []
        for name in sorted(os.listdir(exdata_dir)):
            if name.startswith(prefix):
                path = os.path.join(exdata_dir, name)
                if os.path.isdir(path):
                    matches.append(repo_relpath(path))
        out_list = os.path.join(results_dir, "exdata_dirs.txt")
        with open(out_list, "w", encoding="utf-8") as f:
            for p in matches:
                f.write(p + "\n")

        print("Wrote:", repo_relpath(out_list))

    print("Done. Merged results:", repo_relpath(merged_dir))


if __name__ == "__main__":
    main()
