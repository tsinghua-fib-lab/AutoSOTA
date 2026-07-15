#!/usr/bin/env python3
"""
Measure UH-CMA-ES per-generation evaluation cost on bbob-noisy (fixed budget).

This produces an evidence pack used by the depth–fidelity characterization:
- per-run measurements: `uh_cmaes_cost_measurements.csv`
- aggregated summary:  `uh_cmaes_cost_summary.csv`

The measurement uses pycma's NoiseHandler + cma.fmin2 and records:
- total evaluations consumed (budget usage),
- total generations (es.countiter),
- average evals per candidate.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass

import numpy as np

from _project import BASE_DIR, repo_relpath

try:
    import cma
except ImportError:  # pragma: no cover
    cma = None

try:
    import cocoex
except ImportError:  # pragma: no cover
    cocoex = None


@dataclass(frozen=True)
class RunMeasurement:
    algorithm: str
    function_id: int
    function_index: int
    dimension: int
    instance: int
    budget: int
    population_size: int
    total_evaluations: int
    total_generations: int
    avg_evals_per_generation: float
    avg_evals_per_candidate: float
    final_best_f: float
    converged: bool


def _parse_int_list(spec: str) -> list[int]:
    out: list[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            lo = int(a.strip())
            hi = int(b.strip())
            if hi < lo:
                lo, hi = hi, lo
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def _make_noise_handler(dim: int, maxevals: int, *, epsilon: float = 0.0):
    if cma is None:  # pragma: no cover
        raise RuntimeError("pycma not installed")
    start_evals = max(1, maxevals // 3)
    return cma.optimization_tools.NoiseHandler(
        dim,
        maxevals=[1, start_evals, maxevals],
        aggregate=np.median,
        reevals=None,
        epsilon=float(epsilon),
        parallel=False,
    )


def _run_uh_cmaes(
    problem,
    *,
    budget: int,
    maxevals_param: int,
    epsilon: float,
    seed: int,
) -> tuple[int, int, int, float, bool]:
    if cma is None:  # pragma: no cover
        raise RuntimeError("pycma not installed")

    dim = int(problem.dimension)
    lower = np.asarray(problem.lower_bounds, dtype=float)
    upper = np.asarray(problem.upper_bounds, dtype=float)
    x0 = np.clip(problem.initial_solution, lower, upper)
    sigma0 = 0.3 * float(np.min(upper - lower))

    eval_count = [0]

    def objective(x):
        eval_count[0] += 1
        return float(problem(np.clip(x, lower, upper)))

    popsize = max(4, 4 + int(3 * math.log(dim)))
    opts = {
        "bounds": [lower, upper],
        "maxfevals": int(budget),
        "seed": int(seed),
        "verbose": -9,
        "verb_log": 0,
        "verb_time": 0,
        "popsize": int(popsize),
        # Disable early stopping to use (nearly) the full budget.
        "tolfun": 0.0,
        "tolfunhist": 0.0,
        "tolx": 0.0,
        "tolstagnation": int(1e9),
        "tolxstagnation": False,
        "tolflatfitness": int(1e9),
        "tolconditioncov": 1e30,
        "tolupsigma": 1e30,
    }

    nh = _make_noise_handler(dim, int(maxevals_param), epsilon=float(epsilon))
    try:
        _, es = cma.fmin2(objective, x0, sigma0, options=opts, noise_handler=nh)
        converged = True
    except Exception:
        return int(eval_count[0]), 0, int(popsize), float("inf"), False

    total_evals = int(eval_count[0])
    total_gens = int(es.countiter)
    best_f = float(es.result.fbest) if hasattr(es, "result") and hasattr(es.result, "fbest") else float("inf")
    return total_evals, total_gens, int(popsize), best_f, converged


def _measure_problem(problem, *, algorithm: str, maxevals_param: int, epsilon: float, budget_mult: int) -> RunMeasurement:
    dim = int(problem.dimension)
    budget = int(budget_mult) * dim
    seed = (
        int(problem.id_function) * 1000003
        + int(problem.id_instance) * 1009
        + dim * 7
        + 4242
    ) & 0xFFFFFFFF

    total_evals, total_gens, popsize, best_f, converged = _run_uh_cmaes(
        problem,
        budget=budget,
        maxevals_param=maxevals_param,
        epsilon=float(epsilon),
        seed=int(seed),
    )

    if total_gens > 0:
        avg_evals_per_gen = float(total_evals) / float(total_gens)
        avg_evals_per_candidate = float(total_evals) / float(total_gens * popsize)
    else:
        avg_evals_per_gen = float("nan")
        avg_evals_per_candidate = float("nan")

    return RunMeasurement(
        algorithm=str(algorithm),
        function_id=int(problem.id_function),
        function_index=int(problem.id_function) - 100,
        dimension=int(dim),
        instance=int(problem.id_instance),
        budget=int(budget),
        population_size=int(popsize),
        total_evaluations=int(total_evals),
        total_generations=int(total_gens),
        avg_evals_per_generation=float(avg_evals_per_gen),
        avg_evals_per_candidate=float(avg_evals_per_candidate),
        final_best_f=float(best_f),
        converged=bool(converged),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure UH-CMA-ES evaluation cost (fixed budget)")
    parser.add_argument(
        "--out-dir",
        default=os.path.join(BASE_DIR, "evidence/uh_cmaes_cost_measurement"),
        help="Output directory",
    )
    parser.add_argument("--dims", default="40", help="Dimensions (comma-separated or ranges)")
    parser.add_argument(
        "--functions",
        default="8,10,11,13,14,16,17,19,20,22,23,25,26,28,29",
        help="bbob-noisy function indices (1-30) to measure",
    )
    parser.add_argument("--instances", default="1-15", help="Instance indices")
    parser.add_argument("--budget-mult", type=int, default=100, help="Budget multiplier (budget = mult*dim)")
    args = parser.parse_args()

    os.chdir(BASE_DIR)

    if cma is None:  # pragma: no cover
        raise SystemExit("pycma is required (pip install cma)")
    if cocoex is None:  # pragma: no cover
        raise SystemExit("cocoex is required (see docs/INSTALL.md)")

    dims = _parse_int_list(args.dims)
    funcs = _parse_int_list(args.functions)
    instances = _parse_int_list(args.instances)

    out_dir = os.path.abspath(str(args.out_dir))
    os.makedirs(out_dir, exist_ok=True)

    algos = [
        ("UH-CMA-ES(maxevals=10)", 10, 0.0),
        ("UH-CMA-ES(maxevals=30)", 30, 0.0),
    ]

    suite_filter = (
        f"dimensions:{','.join(map(str, dims))} "
        f"function_indices:{','.join(map(str, funcs))} "
        f"instance_indices:{','.join(map(str, instances))}"
    )

    measurements: list[RunMeasurement] = []

    for algo_name, maxevals_param, epsilon in algos:
        suite = cocoex.Suite("bbob-noisy", "", suite_filter)
        total = len(suite)
        for idx, problem in enumerate(suite):
            m = _measure_problem(
                problem,
                algorithm=algo_name,
                maxevals_param=int(maxevals_param),
                epsilon=float(epsilon),
                budget_mult=int(args.budget_mult),
            )
            measurements.append(m)
            status = "OK" if m.converged else "FAIL"
            print(
                f"[{idx+1:3d}/{total}] {algo_name} f{m.function_index:02d} d{m.dimension:02d} i{m.instance:02d}: "
                f"gens={m.total_generations:3d}, evals={m.total_evaluations:4d}, "
                f"cost/cand={m.avg_evals_per_candidate:.2f} [{status}]"
            )

    # Write per-run measurements
    meas_path = os.path.join(out_dir, "uh_cmaes_cost_measurements.csv")
    with open(meas_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "algorithm",
                "function",
                "function_index",
                "dimension",
                "instance",
                "budget",
                "population_size",
                "total_evaluations",
                "total_generations",
                "avg_evals_per_generation",
                "avg_evals_per_candidate",
                "final_best_f",
                "converged",
            ]
        )
        for m in measurements:
            w.writerow(
                [
                    m.algorithm,
                    m.function_id,
                    m.function_index,
                    m.dimension,
                    m.instance,
                    m.budget,
                    m.population_size,
                    m.total_evaluations,
                    m.total_generations,
                    f"{m.avg_evals_per_generation:.6f}" if np.isfinite(m.avg_evals_per_generation) else "",
                    f"{m.avg_evals_per_candidate:.6f}" if np.isfinite(m.avg_evals_per_candidate) else "",
                    f"{m.final_best_f:.12g}",
                    "1" if m.converged else "0",
                ]
            )

    # Aggregate summary
    from collections import defaultdict

    grouped: dict[tuple[str, int], list[RunMeasurement]] = defaultdict(list)
    for m in measurements:
        if m.converged and np.isfinite(m.avg_evals_per_candidate):
            grouped[(m.algorithm, m.dimension)].append(m)

    summary_path = os.path.join(out_dir, "uh_cmaes_cost_summary.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "algorithm",
                "dimension",
                "n_runs",
                "mean_evals_per_candidate",
                "median_evals_per_candidate",
                "std_evals_per_candidate",
                "min_evals_per_candidate",
                "max_evals_per_candidate",
                "mean_generations",
                "median_generations",
                "mean_depth_at_budget",
            ]
        )
        for (algo, dim), group in sorted(grouped.items()):
            costs = np.asarray([m.avg_evals_per_candidate for m in group], dtype=float)
            gens = np.asarray([m.total_generations for m in group], dtype=float)
            budget = float(group[0].budget)
            popsize = float(group[0].population_size)
            depth = budget / (costs * popsize)
            w.writerow(
                [
                    algo,
                    dim,
                    len(group),
                    f"{float(np.mean(costs)):.4f}",
                    f"{float(np.median(costs)):.4f}",
                    f"{float(np.std(costs)):.4f}",
                    f"{float(np.min(costs)):.4f}",
                    f"{float(np.max(costs)):.4f}",
                    f"{float(np.mean(gens)):.1f}",
                    f"{float(np.median(gens)):.1f}",
                    f"{float(np.mean(depth)):.1f}",
                ]
            )

    # README (portable, repo-relative)
    readme_path = os.path.join(out_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("# UH-CMA-ES Cost Measurement (fixed budget)\n\n")
        f.write("This evidence pack contains measurements of UH-CMA-ES evaluation overhead.\n\n")
        f.write("## Files\n\n")
        f.write(f"- `{repo_relpath(meas_path)}`: per-run measurements\n")
        f.write(f"- `{repo_relpath(summary_path)}`: aggregated summary\n\n")
        f.write("## Reproduce\n\n")
        f.write("```bash\n")
        f.write("python3 tools/run_uh_cmaes_cost_measurement.py \\\n")
        f.write(f"  --out-dir {repo_relpath(out_dir)} \\\n")
        f.write(f"  --dims {args.dims} \\\n")
        f.write(f"  --functions {args.functions} \\\n")
        f.write(f"  --instances {args.instances} \\\n")
        f.write(f"  --budget-mult {args.budget_mult}\n")
        f.write("```\n")

    print("Wrote:", repo_relpath(meas_path))
    print("Wrote:", repo_relpath(summary_path))
    print("Wrote:", repo_relpath(readme_path))


if __name__ == "__main__":
    main()

