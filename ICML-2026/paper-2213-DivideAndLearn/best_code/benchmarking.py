import os
import sys
import json
import time
import yaml
import argparse
import multiprocessing
from typing import Any, Dict

import torch

from MOCO.problems import (
    BiObjectiveTSP,
    MultiObjectiveKnapsack,
    TriObjectiveTSP,
    BiObjectiveCVRP,
)
from MOCO.evaluation import MOCOEvaluator

from src.divide_n_learn.our_method_UCB_combinatorial_bandit import (
    CachedAdvancedBiKPWrapper as UCBWrapper,
)
from src.divide_n_learn.our_method_TS_combinatorial_bandit import (
    CachedAdvancedBiKPWrapper as TSWrapper,
)

try:
    multiprocessing.set_start_method("spawn")
except RuntimeError:
    pass


ALGORITHM_REGISTRY = {
    "ucb": UCBWrapper,
    "ts":  TSWrapper,
}

PROBLEM_MAP = {
    "BiObjectiveTSP":         {"class": BiObjectiveTSP,         "ref_type": "BiTSP"},
    "MultiObjectiveKnapsack": {"class": MultiObjectiveKnapsack, "ref_type": "BiKP"},
    "TriObjectiveTSP":        {"class": TriObjectiveTSP,        "ref_type": "TriTSP"},
    "BiObjectiveCVRP":        {"class": BiObjectiveCVRP,        "ref_type": "BiCVRP"},
}


def _ref_size(problem_type: str, problem_params: Dict[str, Any]) -> int:
    if problem_type in ("BiObjectiveTSP", "TriObjectiveTSP"):
        return problem_params["n_cities"]
    if problem_type == "MultiObjectiveKnapsack":
        return problem_params["n_items"]
    if problem_type == "BiObjectiveCVRP":
        return problem_params["n_customers"]
    raise ValueError(f"Unknown problem_type: {problem_type}")


def _resolve_reference_point(temp_evaluator, problem_type, ref_type, problem_params):
    ref_size = _ref_size(problem_type, problem_params)
    try:
        standard_points = temp_evaluator.get_standard_points(
            problem_type=ref_type, problem_size=ref_size
        )
        if standard_points and "reference" in standard_points:
            return standard_points["reference"], standard_points.get("ideal", (0, 0))
    except Exception as e:
        print(f"Warning: could not get standard points ({e}); using defaults")

    # Defaults
    if problem_type == "BiObjectiveTSP":
        return (35, 35), (0, 0)
    return (20, 20), (50, 50)


def run_benchmark(
    method_name: str,
    algorithm_class,
    params: Dict[str, Any],
    problems: Dict[str, Dict[str, Dict[str, Any]]],
    num_runs: int,
    cuda_device: int = 0,
):
    device = torch.device(f"cuda:{cuda_device}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join("results", f"{method_name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    results: Dict[str, Any] = {}
    config_out = {
        "metadata": {"timestamp": timestamp, "num_runs": num_runs, "device": device.type, "method": method_name},
        "results": {},
    }

    overall_start = time.time()
    print(f"\n{'-'*60}\nBenchmarking: {method_name}\n{'-'*60}")

    results[method_name] = {}
    config_out["results"][method_name] = {}
    temp_evaluator = MOCOEvaluator(reference_point=(1.0, 1.0), results_dir=run_dir)

    for problem_type, sizes in problems.items():
        if problem_type not in PROBLEM_MAP:
            print(f"Problem {problem_type} not defined. Skipping.")
            continue

        problem_class = PROBLEM_MAP[problem_type]["class"]
        ref_type = PROBLEM_MAP[problem_type]["ref_type"]
        results[method_name][problem_type] = {}
        config_out["results"][method_name][problem_type] = {}

        for size_name, problem_params in sizes.items():
            print(f"\nBenchmarking {problem_type} ({size_name})...")

            reference_point, ideal_point = _resolve_reference_point(
                temp_evaluator, problem_type, ref_type, problem_params
            )
            print(f"Reference Point: {reference_point}")
            print(f"Ideal Point: {ideal_point}")

            evaluator = MOCOEvaluator(
                reference_point=reference_point,
                results_dir=run_dir,
            )
            evaluator.parallel = False

            algorithm_params = dict(params)
            algorithm_params["reference_point"] = reference_point

            t0 = time.time()
            try:
                print(f"Running {method_name} on {problem_type} ({size_name}) with {num_runs} runs...")
                result = evaluator.evaluate_algorithm(
                    algorithm_class=algorithm_class,
                    problem_class=problem_class,
                    algorithm_name=method_name,
                    parameters=algorithm_params,
                    problem_params=problem_params,
                    num_runs=num_runs,
                )
                runtime_local = time.time() - t0
                bench = {
                    "status": "success",
                    "runtime": float(result.runtime),
                    "local_runtime": float(runtime_local),
                    "hypervolume": float(result.hypervolume),
                    "num_nondominated": int(result.num_nondominated),
                }
                print(
                    f"  Runtime: {result.runtime:.2f}s  Local: {runtime_local:.2f}s  "
                    f"HV: {result.hypervolume:.4f}  Nondom: {result.num_nondominated}"
                )
            except Exception as e:
                import traceback
                traceback.print_exc()
                bench = {"status": "error", "error": str(e)}

            results[method_name][problem_type][size_name] = bench
            config_out["results"][method_name][problem_type][size_name] = bench

    overall_time = time.time() - overall_start
    config_out["metadata"]["overall_time"] = overall_time

    out_path = os.path.join(run_dir, "summary.yaml")
    try:
        with open(out_path, "w") as f:
            yaml.dump(config_out, f, default_flow_style=False)
        print(f"\nRun artefacts saved to: {run_dir}/")
    except Exception as e:
        print(f"Error saving results: {e}")

    print("\n" + "=" * 50 + "\nBENCHMARK SUMMARY\n" + "=" * 50)
    print(f"Overall time: {overall_time:.1f}s")
    return results, config_out


def _load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _parse_kv_overrides(items):
    """Parse `--param key=value` overrides; values are coerced via YAML scalar parsing."""
    out = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"--param expects key=value, got: {item}")
        k, v = item.split("=", 1)
        out[k.strip()] = yaml.safe_load(v)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run divide-n-learn benchmarks.")
    parser.add_argument("--config", type=str, help="Path to YAML config (e.g. configs/ucb.yaml).")
    parser.add_argument("--algorithm", choices=list(ALGORITHM_REGISTRY), help="Algorithm: ucb or ts (overrides config).")
    parser.add_argument("--method-name", type=str, help="Display/method name (overrides config).")
    parser.add_argument("--num-runs", type=int, help="Number of runs per (problem, size) (overrides config).")
    parser.add_argument("--cuda-device", type=int, help="CUDA device index (overrides config).")
    parser.add_argument("--problems", type=str, help="Inline JSON for problem sizes (overrides config).")
    parser.add_argument("--params", type=str, help="Inline JSON for algorithm params (merged onto config params).")
    parser.add_argument("--param", action="append", help="Single param override key=value (repeatable).")
    args = parser.parse_args(argv)

    cfg: Dict[str, Any] = _load_config(args.config) if args.config else {}

    algorithm_key = args.algorithm or cfg.get("algorithm")
    if algorithm_key not in ALGORITHM_REGISTRY:
        parser.error(f"Missing/invalid algorithm. Got {algorithm_key!r}; valid: {list(ALGORITHM_REGISTRY)}")

    method_name = args.method_name or cfg.get("method_name") or algorithm_key.upper()
    num_runs    = args.num_runs    if args.num_runs    is not None else cfg.get("num_runs", 1)
    cuda_device = args.cuda_device if args.cuda_device is not None else cfg.get("cuda_device", 0)

    problems = cfg.get("problems", {}) or {}
    if args.problems:
        problems = json.loads(args.problems)
    if not problems:
        parser.error("No problems specified. Provide --config with `problems:` or --problems JSON.")

    params = dict(cfg.get("params", {}) or {})
    if args.params:
        params.update(json.loads(args.params))
    params.update(_parse_kv_overrides(args.param))

    return run_benchmark(
        method_name=method_name,
        algorithm_class=ALGORITHM_REGISTRY[algorithm_key],
        params=params,
        problems=problems,
        num_runs=num_runs,
        cuda_device=cuda_device,
    )


if __name__ == "__main__":
    main()
