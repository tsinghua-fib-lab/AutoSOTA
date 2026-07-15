#!/usr/bin/env python3
"""
Build a rank-metrics-friendly `bbob_summary.csv` from COCO `exdata/` folders,
using the *noise-free* best-so-far values stored in `.dat` files.

Why:
- For bbob-noisy, COCO logs include both measured (noisy) and noise-free values.
- Our local runners' `bbob_summary.csv` uses `problem.best_observed_fvalue1`, which
  corresponds to measured values, and can misrepresent algorithms under noisy objectives.

This script parses:
- `bbobexp_f*.info` to map (function, dimension) -> .dat file and instance/run ids,
- `.dat` segments to extract the final "best noise-free fitness - Fopt" (delta).

It writes a synthesized `bbob_summary.csv` with `best_f` = final noise-free delta.
The output can be fed directly into `tools/plot_bbob_results.py`.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from dataclasses import dataclass

from _project import BASE_DIR, repo_relpath

@dataclass(frozen=True)
class RunRef:
    dat_relpath: str
    run_ids: list[int]
    run_evals: list[int]
    func_id: int
    dim: int
    alg_id: str


def read_lines(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return [line.rstrip("\n") for line in f]


def parse_info_file(path: str) -> list[RunRef]:
    lines = read_lines(path)
    out: list[RunRef] = []

    header_re = re.compile(r"funcId\s*=\s*(\d+).*DIM\s*=\s*(\d+).*algId\s*=\s*'([^']+)'")
    mapping_re = re.compile(r"^(data_f\d+/[^,]+)\s*,\s*(.+)$")

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        m = header_re.search(line)
        if not m:
            i += 1
            continue
        func_id = int(m.group(1))
        dim = int(m.group(2))
        alg_id = str(m.group(3))

        # Skip ahead to mapping line "data_fNNN/...dat, 1:..., 2:..."
        dat_relpath = ""
        run_ids: list[int] = []
        run_evals: list[int] = []

        j = i + 1
        while j < len(lines):
            m2 = mapping_re.match(lines[j].strip())
            if m2:
                dat_relpath = m2.group(1).strip()
                rest = m2.group(2).strip()
                parts = [p.strip() for p in rest.split(",") if p.strip()]
                for p in parts:
                    m3 = re.match(r"(\d+):(\d+)\|", p)
                    if not m3:
                        continue
                    run_ids.append(int(m3.group(1)))
                    run_evals.append(int(m3.group(2)))
                break
            # Next header begins -> stop.
            if header_re.search(lines[j]):
                break
            j += 1

        if dat_relpath and run_ids and len(run_ids) == len(run_evals):
            out.append(
                RunRef(
                    dat_relpath=dat_relpath,
                    run_ids=run_ids,
                    run_evals=run_evals,
                    func_id=func_id,
                    dim=dim,
                    alg_id=alg_id,
                )
            )
        i = j + 1

    return out


def parse_dat_segments(path: str) -> list[tuple[float, int, float]]:
    """
    Return a list of segments as (fopt, final_evals, final_best_delta).
    """

    segments: list[tuple[float, int, float]] = []
    fopt = float("nan")
    last_eval = None
    last_best = None

    header_re = re.compile(r"Fopt\s*\\(([-+0-9.eE]+)\\)")

    def flush():
        nonlocal last_eval, last_best, fopt
        if last_eval is not None and last_best is not None:
            segments.append((float(fopt), int(last_eval), float(last_best)))
        last_eval = None
        last_best = None

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("%"):
                # new segment
                flush()
                m = header_re.search(line)
                fopt = float(m.group(1)) if m else float("nan")
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                fe = int(parts[0])
                best_delta = float(parts[2])
            except ValueError:
                continue
            last_eval = fe
            last_best = best_delta

    flush()
    return segments


def read_exdata_list(path: str) -> list[str]:
    out = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            p = raw.strip()
            if not p or p.startswith("#"):
                continue
            out.append(p)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exdata-dirs", nargs="*", default=[], help="COCO exdata directories to include.")
    parser.add_argument(
        "--exdata-list",
        default="",
        help="Text file containing exdata directories (one per line), e.g. Results/.../exdata_dirs.txt",
    )
    parser.add_argument("--output-dir", required=True, help="Directory to write synthesized bbob_summary.csv")
    args = parser.parse_args()

    os.chdir(BASE_DIR)

    exdata_dirs = [os.path.abspath(p) for p in args.exdata_dirs if str(p).strip()]
    if str(args.exdata_list).strip():
        exdata_dirs.extend([os.path.abspath(p) for p in read_exdata_list(args.exdata_list)])
    exdata_dirs = [p for p in exdata_dirs if os.path.isdir(p)]
    exdata_dirs = sorted(set(exdata_dirs))
    if not exdata_dirs:
        raise SystemExit("No valid exdata directories provided.")

    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for exdir in exdata_dirs:
        info_files = [p for p in os.listdir(exdir) if p.startswith("bbobexp_f") and p.endswith(".info")]
        for info_name in sorted(info_files):
            info_path = os.path.join(exdir, info_name)
            refs = parse_info_file(info_path)
            for ref in refs:
                dat_path = os.path.join(exdir, ref.dat_relpath)
                if not os.path.isfile(dat_path):
                    continue
                segs = parse_dat_segments(dat_path)
                if not segs:
                    continue
                if len(segs) != len(ref.run_ids):
                    # Be conservative: only match the common prefix.
                    n = min(len(segs), len(ref.run_ids))
                    segs = segs[:n]
                    run_ids = ref.run_ids[:n]
                    run_evals = ref.run_evals[:n]
                else:
                    run_ids = ref.run_ids
                    run_evals = ref.run_evals

                for k, (fopt, fe_last, best_delta) in enumerate(segs):
                    inst = int(run_ids[k])
                    fe_target = int(run_evals[k])
                    # If COCO logged fewer evals than expected (early stop), keep the last logged eval count.
                    fe = int(fe_last)
                    if fe_target > 0:
                        # derive budget multiplier when possible
                        budget_mult = int(round(float(fe_target) / float(max(1, ref.dim))))
                    else:
                        budget_mult = int(round(float(fe) / float(max(1, ref.dim))))

                    rows.append(
                        {
                            "algorithm": str(ref.alg_id),
                            "budget_multiplier": budget_mult,
                            "function": int(ref.func_id),
                            "dimension": int(ref.dim),
                            "instance": inst,
                            "evaluations": fe,
                            # noise-free best delta is comparable within each (f,dim,inst).
                            "best_f": float(best_delta),
                            "final_target_hit": 0,
                            "elapsed_sec": 0.0,
                        }
                    )

    if not rows:
        raise SystemExit("No runs parsed from exdata.")

    out_path = os.path.join(out_dir, "bbob_summary.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "algorithm",
                "budget_multiplier",
                "function",
                "dimension",
                "instance",
                "evaluations",
                "best_f",
                "final_target_hit",
                "elapsed_sec",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    # Provide an empty trace_index.csv so downstream tooling can run without special-casing.
    trace_index_path = os.path.join(out_dir, "trace_index.csv")
    with open(trace_index_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "trace_id",
                "algorithm",
                "budget_multiplier",
                "function",
                "dimension",
                "instance",
                "trace_file",
            ]
        )

    print("Wrote:", repo_relpath(out_path))
    print("Wrote:", repo_relpath(trace_index_path))
    print("Rows:", len(rows))


if __name__ == "__main__":
    main()
