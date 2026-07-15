#!/usr/bin/env python3
"""
Summarize misranking CSVs produced by `tools/measure_misranking_severity.py`.

Typical use (bbob-noisy D=40):
  python3 tools/summarize_misranking_by_function.py \
    --input-csv Results/misranking_bbob_noisy_d40_es_f1-30_i1.csv \
    --output-csv Results/misranking_bbob_noisy_d40_es_f1-30_i1_by_func.csv
"""

import argparse
import csv
import os
from collections import defaultdict

import numpy as np

from _project import repo_relpath

def q(a: np.ndarray, p: float) -> float:
    if a.size <= 0:
        return float("nan")
    return float(np.quantile(a, float(p)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument(
        "--group-by-instance",
        action="store_true",
        help="Include instance in grouping key (default: aggregate over instances).",
    )
    args = parser.parse_args()

    in_path = os.path.abspath(str(args.input_csv))
    out_path = os.path.abspath(str(args.output_csv))

    groups = defaultdict(list)
    with open(in_path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            suite = str(row.get("suite", ""))
            sampling = str(row.get("sampling", ""))
            noise_model = str(row.get("noise_model", ""))
            dim = int(float(row.get("dimension", "nan")))
            func = int(float(row.get("function", "nan")))
            inst = int(float(row.get("instance", "nan")))

            key = (suite, sampling, noise_model, dim, func, inst) if args.group_by_instance else (suite, sampling, noise_model, dim, func)
            groups[key].append(row)

    rows_out = []
    for key, rows in sorted(groups.items()):
        suite = key[0]
        sampling = key[1]
        noise_model = key[2]
        dim = int(key[3])
        func = int(key[4])
        inst = int(key[5]) if args.group_by_instance else None

        rds = np.asarray([float(r["rank_disagreement"]) for r in rows], dtype=float)
        ovs = np.asarray([float(r["topmu_overlap"]) for r in rows], dtype=float)

        out = {
            "suite": suite,
            "sampling": sampling,
            "noise_model": noise_model,
            "dimension": dim,
            "function": func,
            "n_sets": int(len(rows)),
            "rank_disagreement_mean": float(np.mean(rds)) if rds.size else float("nan"),
            "rank_disagreement_median": float(np.median(rds)) if rds.size else float("nan"),
            "rank_disagreement_q90": q(rds, 0.9),
            "topmu_overlap_mean": float(np.mean(ovs)) if ovs.size else float("nan"),
            "topmu_overlap_median": float(np.median(ovs)) if ovs.size else float("nan"),
            "topmu_overlap_q10": q(ovs, 0.1),
        }
        if inst is not None:
            out["instance"] = int(inst)
        # Convenience for bbob-noisy: COCO reports function ids 101..130.
        if suite == "bbob-noisy" and func >= 101:
            out["function_index"] = int(func - 100)
        rows_out.append(out)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        fieldnames = list(rows_out[0].keys()) if rows_out else []
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if rows_out:
            w.writeheader()
            for row in rows_out:
                w.writerow(row)

    print("Wrote:", repo_relpath(out_path))


if __name__ == "__main__":
    main()
