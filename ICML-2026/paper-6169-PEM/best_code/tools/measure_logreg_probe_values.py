#!/usr/bin/env python3
"""
Measure probe values on the mini-batch logistic regression benchmark.

This script exists to make the "probe -> regime -> decision" story reproducible
outside COCO:
- misranking probe (rank_disagreement on a candidate set),
- tail-ratio probe (heavy-tail proxy on |Δf|),
- variance probe at x0 (baseline proxy).

Outputs a CSV suitable for evidence snapshots.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

from _project import BASE_DIR, repo_relpath
from algorithms import probe_switch as ms

import run_logreg_minibatch_sweep as logreg  # local tools/ module


def parse_int_list(spec: str) -> list[int]:
    out: list[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a_i = int(a.strip())
            b_i = int(b.strip())
            if b_i < a_i:
                a_i, b_i = b_i, a_i
            out.extend(range(a_i, b_i + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-csv", required=True, help="Output CSV path.")

    parser.add_argument(
        "--dataset",
        default="synthetic",
        help="Dataset for the per-seed instance: synthetic | breast_cancer | digits0.",
    )
    parser.add_argument("--dim", type=int, default=40)
    parser.add_argument("--n-samples", type=int, default=256)
    parser.add_argument("--batch-sizes", default="8,32")
    parser.add_argument("--seeds", default="1-12")
    parser.add_argument("--w-max", type=float, default=5.0)
    parser.add_argument("--weight-sigma", type=float, default=0.0)
    parser.add_argument(
        "--weight-sigma-stochastic-only",
        action="store_true",
        help="Match run_logreg_minibatch_sweep.py: apply lognormal weights only when batch_size < N.",
    )
    parser.add_argument("--l2-reg", type=float, default=0.0)
    parser.add_argument("--label-noise", type=float, default=0.0)
    parser.add_argument("--eval-independent-noise", action="store_true")

    parser.add_argument("--misranking-threshold", type=float, default=0.12)
    parser.add_argument("--tail-threshold", type=float, default=4.0)
    parser.add_argument("--variance-threshold", type=float, default=0.05)
    parser.add_argument("--variance-reps", type=int, default=10)
    args = parser.parse_args()

    os.chdir(BASE_DIR)

    dataset = str(args.dataset).strip().lower()
    if dataset == "synthetic":
        dim = int(args.dim)
    else:
        X0, _y0 = logreg._load_base_dataset(dataset)
        dim = int(X0.shape[1]) + 1  # +1 bias
    n_samples = int(args.n_samples)
    batch_sizes = parse_int_list(str(args.batch_sizes))
    seeds = parse_int_list(str(args.seeds))

    rows: list[dict[str, object]] = []
    for seed in seeds:
        for bs in batch_sizes:
            problem = logreg.NoisyMiniBatchLogRegProblem(
                seed=int(seed),
                dim=int(dim),
                n_samples=int(n_samples),
                batch_size=int(bs),
                w_max=float(args.w_max),
                weight_sigma=float(args.weight_sigma),
                weight_sigma_stochastic_only=bool(args.weight_sigma_stochastic_only),
                l2_reg=float(args.l2_reg),
                label_noise=float(args.label_noise),
                eval_independent_noise=bool(args.eval_independent_noise),
                dataset=str(args.dataset),
                add_bias=(str(args.dataset).strip().lower() != "synthetic"),
                standardize=True,
            )

            # NOTE: probes consume evaluations on the problem instance; that's fine
            # because we only use them for diagnosis here.
            rd = ms._misranking_probe(problem, max_evals=10**9)
            rd2, tail_ratio = ms._tail_ratio_probe(problem, max_evals=10**9, reps=2)
            rel_sd = ms._variance_probe(
                problem,
                max_evals=10**9,
                reps=int(args.variance_reps),
            )

            pred_noise_switch = "cma"
            if rd is not None and float(rd) >= float(args.misranking_threshold):
                pred_noise_switch = "hetero"
                if tail_ratio is not None and float(tail_ratio) >= float(args.tail_threshold):
                    pred_noise_switch = "robust"

            rows.append(
                {
                    "seed": int(seed),
                    "batch_size": int(bs),
                    "dim": int(dim),
                    "n_samples": int(n_samples),
                    "weight_sigma": float(args.weight_sigma),
                    "misranking_rd": "" if rd is None else float(rd),
                    "tail_probe_rd": "" if rd2 is None else float(rd2),
                    "tail_ratio": "" if tail_ratio is None else float(tail_ratio),
                    "variance_rel_sd": "" if rel_sd is None else float(rel_sd),
                    "pred_noise_probe_switch": str(pred_noise_switch),
                }
            )

    out_csv = os.path.abspath(str(args.out_csv))
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        fieldnames = list(rows[0].keys()) if rows else []
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if rows:
            w.writeheader()
            for r in rows:
                w.writerow(r)

    # Print a tiny aggregate for quick sanity checks.
    by_bs: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in rows:
        bs = int(r["batch_size"])
        pred = str(r["pred_noise_probe_switch"])
        by_bs[bs][pred] += 1
    for bs in sorted(by_bs.keys()):
        counts = dict(by_bs[bs])
        print(f"batch_size={bs} decision_counts={counts}")

    print("Wrote:", repo_relpath(out_csv))


if __name__ == "__main__":
    main()
