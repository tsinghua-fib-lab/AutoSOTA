#!/usr/bin/env python3
"""
Measure how different online probes behave under (potentially) state-dependent noise.

This script is meant to provide *mechanistic evidence* that:
- a pointwise variance proxy at x0 can miss misranking when noise is x-dependent, and
- a candidate-set misranking proxy captures this regime.

It wraps deterministic bbob-largescale problems with the same noise models used in
`utils.noise.NoisyProblem` and computes:
- misranking probe: mean |Δrank| / λ on a CMA-style initial population (2 draws),
- variance probe: relative std at x0 from repeated evaluations.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import cocoex
import numpy as np

from _project import BASE_DIR, repo_relpath

from algorithms import probe_switch as ms
from utils.noise import NoisyProblem


@dataclass(frozen=True)
class Row:
    function: int
    dimension: int
    instance: int
    noise_model: str
    noise_sigma: float
    misranking_rd: float | None
    variance_rel_sd: float | None
    misranking_trigger: int
    variance_trigger: int


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


def format_filter(dims: list[int], funcs: list[int], instances: list[int]) -> str:
    dims_s = ",".join(str(d) for d in dims)
    funcs_s = ",".join(str(f) for f in funcs)
    inst_s = ",".join(str(i) for i in instances)
    return f"dimensions:{dims_s} function_indices:{funcs_s} instance_indices:{inst_s}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dims", default="80,160")
    parser.add_argument("--functions", default="1,2,6,10,15,20")
    parser.add_argument("--instances", default="1-3")
    parser.add_argument(
        "--noise-model",
        default="radial_additive_rel",
        help="Noise model name (must be supported by run_noisy_wrapper_largescale.NoisyProblem).",
    )
    parser.add_argument("--noise-sigma", type=float, default=0.5)
    parser.add_argument("--misranking-threshold", type=float, default=0.12)
    parser.add_argument("--variance-threshold", type=float, default=0.05)
    parser.add_argument("--variance-reps", type=int, default=10)
    parser.add_argument(
        "--out-dir",
        default="",
        help="Output directory (writes a stable snapshot as <out-dir>/probe_values.csv).",
    )
    parser.add_argument(
        "--output-csv",
        default="",
        help="Output CSV path. If empty, defaults to Results/probe_decoupling_<tag>.csv. "
        "If --out-dir is set, this flag is ignored unless explicitly provided.",
    )
    args = parser.parse_args()

    dims = parse_int_list(args.dims)
    funcs = parse_int_list(args.functions)
    inst = parse_int_list(args.instances)
    suite_filter = format_filter(dims, funcs, inst)

    suite = cocoex.Suite("bbob-largescale", "", suite_filter)

    rows: list[Row] = []
    for problem in suite:
        sigma = float(args.noise_sigma)
        noise_seed = (
            int(problem.id_function) * 1000003
            + int(problem.id_instance) * 1009
            + int(problem.dimension) * 7
            + int(round(1000.0 * sigma)) * 13
        ) & 0xFFFFFFFF

        p = NoisyProblem(problem, noise_model=str(args.noise_model), noise_sigma=sigma, seed=noise_seed)

        # Use an effectively-unbounded eval cap; each probe is tiny relative to typical budgets.
        max_evals = int(10**9)
        rd = ms._misranking_probe(p, max_evals=max_evals)
        rel_sd = ms._variance_probe(p, max_evals=max_evals, reps=int(args.variance_reps))

        mis_trig = int(rd is not None and float(rd) >= float(args.misranking_threshold))
        var_trig = int(rel_sd is not None and float(rel_sd) >= float(args.variance_threshold))

        rows.append(
            Row(
                function=int(problem.id_function),
                dimension=int(problem.dimension),
                instance=int(problem.id_instance),
                noise_model=str(args.noise_model),
                noise_sigma=float(args.noise_sigma),
                misranking_rd=None if rd is None else float(rd),
                variance_rel_sd=None if rel_sd is None else float(rel_sd),
                misranking_trigger=mis_trig,
                variance_trigger=var_trig,
            )
        )

    out_path = str(args.output_csv).strip()
    out_dir = str(args.out_dir).strip()

    if not out_path and out_dir:
        out_dir_path = Path(out_dir)
        if not out_dir_path.is_absolute():
            out_dir_path = Path(BASE_DIR) / out_dir_path
        out_dir_path.mkdir(parents=True, exist_ok=True)
        out_path = str(out_dir_path / "probe_values.csv")

    if not out_path:
        results_dir = Path(BASE_DIR) / "Results"
        results_dir.mkdir(parents=True, exist_ok=True)
        tag = f"{str(args.noise_model)}_sigma{str(args.noise_sigma).replace('.', 'p')}"
        out_path = str(results_dir / f"probe_decoupling_{tag}.csv")

    out_path_p = Path(out_path)
    if not out_path_p.is_absolute():
        out_path_p = Path(BASE_DIR) / out_path_p
    out_path_p.parent.mkdir(parents=True, exist_ok=True)
    with out_path_p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "function",
                "dimension",
                "instance",
                "noise_model",
                "noise_sigma",
                "misranking_rd",
                "variance_rel_sd",
                "misranking_trigger",
                "variance_trigger",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r.function,
                    r.dimension,
                    r.instance,
                    r.noise_model,
                    f"{r.noise_sigma:.12g}",
                    "" if r.misranking_rd is None else f"{r.misranking_rd:.12g}",
                    "" if r.variance_rel_sd is None else f"{r.variance_rel_sd:.12g}",
                    r.misranking_trigger,
                    r.variance_trigger,
                ]
            )

    rds = [r.misranking_rd for r in rows if r.misranking_rd is not None and np.isfinite(r.misranking_rd)]
    vs = [r.variance_rel_sd for r in rows if r.variance_rel_sd is not None and np.isfinite(r.variance_rel_sd)]
    print("Wrote:", repo_relpath(str(out_path_p)))
    if rds:
        print("misranking_rd mean:", float(np.mean(rds)), "median:", float(np.median(rds)))
    if vs:
        print("variance_rel_sd mean:", float(np.mean(vs)), "median:", float(np.median(vs)))
    print(
        "trigger rates:",
        "misranking",
        sum(r.misranking_trigger for r in rows),
        "/",
        len(rows),
        "variance",
        sum(r.variance_trigger for r in rows),
        "/",
        len(rows),
    )


if __name__ == "__main__":
    main()
