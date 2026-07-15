#!/usr/bin/env python3
"""
Run the COCO BBOB noisy suite (bbob-noisy) for BERW and baselines.

Goal: test whether selection-stage uncertainty integration (BERW / ProbeSwitch) helps under noisy objectives.

Outputs under `Results/`:
- bbob_summary.csv
- trace_index.csv
- traces/*.csv (downsampled)
"""

import argparse
import csv
import os
import re
import time

import cocoex

from _project import BASE_DIR, repo_relpath

from algorithms.berw_es import (
    my_optimizer_noise_adaptive_sel_bootstrap_weights_hetero as berw_hetero_optimizer,
)
from algorithms.berw_es import (
    my_optimizer_noise_adaptive_sel_bootstrap_weights_hetero_bs16 as berw_hetero_bs16_optimizer,
)
from algorithms.berw_es import (
    my_optimizer_noise_adaptive_sel_bootstrap_weights_hetero_bs64 as berw_hetero_bs64_optimizer,
)
from algorithms.berw_es import (
    my_optimizer_noise_adaptive_sel_bootstrap_weights_hetero_reeval0 as berw_hetero_reeval0_optimizer,
)
from algorithms.berw_es import (
    my_optimizer_noise_adaptive_sel_bootstrap_weights_hetero_reeval3 as berw_hetero_reeval3_optimizer,
)
from algorithms.berw_es import (
    my_optimizer_noise_adaptive_sel_bootstrap_weights_hetero_robust as berw_hetero_robust_optimizer,
)
from algorithms.berw_es import (
    my_optimizer_noise_adaptive_sel_bootstrap_weights_hetero_tmatch as berw_hetero_tmatch_optimizer,
)
from algorithms.berw_es import (
    my_optimizer_noise_adaptive_sel_bootstrap_weights_heterovar as berw_hetero_var_optimizer,
)
from algorithms.probe_switch import (
    my_optimizer_misranking_probe_switch_t012 as probeswitch_mr_t012_optimizer,
)
from algorithms.probe_switch import (
    my_optimizer_misranking_probe_switch_t022 as probeswitch_mr_t022_optimizer,
)
from algorithms.probe_switch import (
    my_optimizer_misranking_probe_switch_t019 as probeswitch_mr_t019_optimizer,
)
from algorithms.probe_switch import (
    my_optimizer_misranking_probe_switch_t021 as probeswitch_mr_t021_optimizer,
)
from algorithms.probe_switch import (
    my_optimizer_misranking_probe_switch_t026 as probeswitch_mr_t026_optimizer,
)
from algorithms.probe_switch import (
    my_optimizer_misranking_probe_switch_t038 as probeswitch_mr_t038_optimizer,
)
from algorithms.probe_switch import (
    my_optimizer_misranking_probe_switch_t046 as probeswitch_mr_t046_optimizer,
)
from algorithms.probe_switch import (
    my_optimizer_misranking_probe_switch_robust_t012 as probeswitch_mr_robust_t012_optimizer,
)
from algorithms.probe_switch import (
    my_optimizer_misranking_probe_switch_robust_t022 as probeswitch_mr_robust_t022_optimizer,
)
from algorithms.probe_switch import (
    my_optimizer_misranking_probe_switch_robust_t019 as probeswitch_mr_robust_t019_optimizer,
)
from algorithms.probe_switch import (
    my_optimizer_misranking_probe_switch_robust_t021 as probeswitch_mr_robust_t021_optimizer,
)
from algorithms.probe_switch import (
    my_optimizer_misranking_probe_switch_robust_t026 as probeswitch_mr_robust_t026_optimizer,
)
from baselines.cmaes_noise import (
    my_optimizer as uh_cmaes_default_optimizer,
)
from baselines.cmaes_noise import (
    my_optimizer_uh_maxevals10 as uh_cmaes_maxevals10_optimizer,
)
from baselines.cmaes_noise import (
    my_optimizer_uh_maxevals30 as uh_cmaes_maxevals30_optimizer,
)
from baselines.cmaes_full import my_optimizer as cmaes_full_optimizer
from baselines.cmaes_sep import my_optimizer as cmaes_sep_optimizer
from baselines.cmaes_sep_resample import (
    my_optimizer_resample2 as cmaes_resample2_optimizer,
)
from baselines.cmaes_sep_resample import (
    my_optimizer_resample3 as cmaes_resample3_optimizer,
)
from baselines.cmaes_sep_resample import (
    my_optimizer_resample5 as cmaes_resample5_optimizer,
)
from baselines.cmaes_sep_resample import (
    my_optimizer_resample10 as cmaes_resample10_optimizer,
)
from baselines.cmaes_lra import my_optimizer as lra_cmaes_optimizer
from algorithms.berw_es import (
    my_optimizer_noise_adaptive_sel_bootstrap_weights_hetero_logw as berw_hetero_logw_optimizer,
)
from algorithms.probe_switch import (
    my_optimizer_misranking_probe_switch_logw_t012 as probeswitch_mr_logw_t012_optimizer,
)


ALGORITHMS = {
    "CMA-ES": cmaes_full_optimizer,
    "CMA-ES-sep": cmaes_sep_optimizer,
    "CMA-ES-Resample(k=2)": cmaes_resample2_optimizer,
    "CMA-ES-Resample(k=3)": cmaes_resample3_optimizer,
    "CMA-ES-Resample(k=5)": cmaes_resample5_optimizer,
    "CMA-ES-Resample(k=10)": cmaes_resample10_optimizer,
    "UH-CMA-ES": uh_cmaes_default_optimizer,
    "UH-CMA-ES(maxevals=10)": uh_cmaes_maxevals10_optimizer,
    "UH-CMA-ES(maxevals=30)": uh_cmaes_maxevals30_optimizer,
    "BERW-Hetero": berw_hetero_optimizer,
    "BERW-Hetero(reeval=0)": berw_hetero_reeval0_optimizer,
    "BERW-Hetero(reeval=3)": berw_hetero_reeval3_optimizer,
    "BERW-Hetero(bs=16)": berw_hetero_bs16_optimizer,
    "BERW-Hetero(bs=64)": berw_hetero_bs64_optimizer,
    "BERW-HeteroRobust": berw_hetero_robust_optimizer,
    "BERW-HeteroTMatch": berw_hetero_tmatch_optimizer,
    "BERW-HeteroVar": berw_hetero_var_optimizer,
    "ProbeSwitch-MR(t=0.12)": probeswitch_mr_t012_optimizer,
    "ProbeSwitch-MR(t=0.22)": probeswitch_mr_t022_optimizer,
    "ProbeSwitch-MR(t=0.19)": probeswitch_mr_t019_optimizer,
    "ProbeSwitch-MR(t=0.21)": probeswitch_mr_t021_optimizer,
    "ProbeSwitch-MR(t=0.26)": probeswitch_mr_t026_optimizer,
    "ProbeSwitch-MR(t=0.38)": probeswitch_mr_t038_optimizer,
    "ProbeSwitch-MR(t=0.46)": probeswitch_mr_t046_optimizer,
    "ProbeSwitch-MR-Robust(t=0.12)": probeswitch_mr_robust_t012_optimizer,
    "ProbeSwitch-MR-Robust(t=0.22)": probeswitch_mr_robust_t022_optimizer,
    "ProbeSwitch-MR-Robust(t=0.19)": probeswitch_mr_robust_t019_optimizer,
    "ProbeSwitch-MR-Robust(t=0.21)": probeswitch_mr_robust_t021_optimizer,
    "ProbeSwitch-MR-Robust(t=0.26)": probeswitch_mr_robust_t026_optimizer,
    "BERW-Hetero-LogW": berw_hetero_logw_optimizer,
    "ProbeSwitch-MR-LogW(t=0.12)": probeswitch_mr_logw_t012_optimizer,
    "LRA-CMA-ES": lra_cmaes_optimizer,
}

DIMS = [10, 20, 40]
BUDGET_MULTS = [200, 500]

# Note: bbob-noisy reports problem.id_function in 101–130, but the suite filter
# expects indices 1–30 (mapping to 101–130).
FUNCTIONS = [1, 5, 10, 15, 20, 25, 30]
INSTANCES = [1]

TRACE_FUNCTIONS = {101, 110, 120, 130}
TRACE_DIMS = {20, 40}
TRACE_BUDGETS = {500}
TRACE_LOG_EVERY = 20


class LoggingProblem:
    """Wraps a COCO problem to log best-so-far (downsampled)."""

    def __init__(self, problem, trace, log_every=20):
        self._problem = problem
        self._trace = trace
        self._log_every = int(max(1, log_every))
        self.best_f = float("inf")

    def __call__(self, x):
        value = self._problem(x)
        improved = False
        if value < self.best_f:
            self.best_f = value
            improved = True
        ev = int(self._problem.evaluations)
        if improved or (ev % self._log_every == 0):
            self._trace.append((ev, float(self.best_f)))
        return value

    def __getattr__(self, name):
        return getattr(self._problem, name)


def sanitize_name(name):
    s = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower())
    s = s.strip("_")
    return s or "unnamed"


def parse_int_list(spec):
    out = []
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


def format_filter(dims, funcs, instances):
    dims_s = ",".join(str(d) for d in dims)
    funcs_s = ",".join(str(f) for f in funcs)
    inst_s = ",".join(str(i) for i in instances)
    return f"dimensions:{dims_s} function_indices:{funcs_s} instance_indices:{inst_s}"


def run_suite(*, results_dir, instances, dims, budgets, functions, algorithm_names, tag):
    results_dir = os.path.abspath(results_dir)
    os.chdir(BASE_DIR)
    traces_dir = os.path.join(results_dir, "traces")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(traces_dir, exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "exdata"), exist_ok=True)

    run_id = time.strftime("%Y%m%d_%H%M%S")

    summary_path = os.path.join(results_dir, "bbob_summary.csv")
    trace_index_path = os.path.join(results_dir, "trace_index.csv")

    with open(summary_path, "w", newline="") as summary_file, open(
        trace_index_path, "w", newline=""
    ) as trace_index_file:
        summary_writer = csv.writer(summary_file)
        trace_index_writer = csv.writer(trace_index_file)

        summary_writer.writerow(
            [
                "algorithm",
                "budget_multiplier",
                "function",
                "dimension",
                "instance",
                "evaluations",
                "best_f",
                "final_target_hit",
                "elapsed_sec",
            ]
        )
        trace_index_writer.writerow(
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

        algorithms = ALGORITHMS
        if algorithm_names:
            want = set(str(a).strip() for a in algorithm_names if str(a).strip())
            missing = [a for a in want if a not in ALGORITHMS]
            if missing:
                raise ValueError("Unknown algorithms: " + ", ".join(sorted(missing)))
            algorithms = {k: v for k, v in ALGORITHMS.items() if k in want}

        suite_filter = format_filter(dims, functions, instances)

        for algo_name, optimizer in algorithms.items():
            algo_id = sanitize_name(algo_name)
            for budget_mult in budgets:
                suite = cocoex.Suite("bbob-noisy", "", suite_filter)
                prefix = f"noisy_{sanitize_name(tag)}_" if tag else "noisy_"
                output_folder = f"{prefix}{algo_id}_B{budget_mult}_{run_id}"
                observer = cocoex.Observer(
                    "bbob",
                    f"result_folder: {output_folder} algorithm_name: {algo_name}",
                )

                print(f"Running {algo_name} | budget {budget_mult}x | filter: {suite_filter}")
                start_time = time.time()
                count = 0

                for problem in suite:
                    problem.observe_with(observer)
                    budget = int(budget_mult * problem.dimension)

                    use_trace = (
                        problem.id_function in TRACE_FUNCTIONS
                        and problem.dimension in TRACE_DIMS
                        and budget_mult in TRACE_BUDGETS
                    )

                    trace = []
                    wrapped_problem = (
                        LoggingProblem(problem, trace, log_every=TRACE_LOG_EVERY)
                        if use_trace
                        else problem
                    )

                    t0 = time.time()
                    optimizer(wrapped_problem, budget)
                    elapsed = time.time() - t0

                    count += 1
                    summary_writer.writerow(
                        [
                            algo_name,
                            budget_mult,
                            problem.id_function,
                            problem.dimension,
                            problem.id_instance,
                            problem.evaluations,
                            problem.best_observed_fvalue1,
                            int(problem.final_target_hit),
                            f"{elapsed:.6f}",
                        ]
                    )

                    if use_trace:
                        trace_id = (
                            f"{algo_id}_B{budget_mult}_f{problem.id_function}_"
                            f"d{problem.dimension}_i{problem.id_instance}"
                        )
                        trace_file = os.path.join(traces_dir, f"{trace_id}.csv")
                        with open(trace_file, "w", newline="") as tf:
                            writer = csv.writer(tf)
                            writer.writerow(["evals", "best_f"])
                            writer.writerows(trace)
                        trace_index_writer.writerow(
                            [
                                trace_id,
                                algo_name,
                                budget_mult,
                                problem.id_function,
                                problem.dimension,
                                problem.id_instance,
                                trace_file,
                            ]
                        )

                    if count % 5 == 0:
                        print(
                            f"  [{count:3d}/{len(suite)}] f{problem.id_function:03d} "
                            f"d{problem.dimension:02d} evals={problem.evaluations:5d} "
                            f"best={problem.best_observed_fvalue1:.3e}"
                        )

                total_elapsed = time.time() - start_time
                print(f"Finished {algo_name} | budget {budget_mult}x in {total_elapsed:.1f}s")

    print("All runs completed.")
    print(f"Summary: {repo_relpath(summary_path)}")
    print(f"Traces: {repo_relpath(traces_dir)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        default=os.path.join(BASE_DIR, "Results", "noisy"),
        help="Directory to write bbob_summary.csv, trace_index.csv and plots/",
    )
    parser.add_argument("--dims", default="10,20,40")
    parser.add_argument("--budgets", default="200,500", help="Budget multipliers (xD)")
    parser.add_argument("--functions", default="1,5,10,15,20,25,30")
    parser.add_argument(
        "--instances",
        default="1",
        help="Instance indices, e.g. '1' or '1-5' or '1,2,3'",
    )
    parser.add_argument(
        "--algorithms",
        default="",
        help="Comma-separated subset of algorithms to run (default: all).",
    )
    parser.add_argument("--tag", default="", help="Optional tag to include in COCO result_folder names.")
    args = parser.parse_args()
    algo_names = [s.strip() for s in args.algorithms.split(",") if s.strip()]
    run_suite(
        results_dir=args.results_dir,
        instances=parse_int_list(args.instances),
        dims=parse_int_list(args.dims),
        budgets=parse_int_list(args.budgets),
        functions=parse_int_list(args.functions),
        algorithm_names=algo_names,
        tag=str(args.tag).strip(),
    )
