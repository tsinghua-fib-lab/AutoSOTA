"""
Generic benchmarking helper.

Works with any problem class that follows the MOCOProblem protocol (constructor takes problem-size kwargs,
exposes `num_objectives` / `n_cities` / `n_items` / `n_customers` etc.),
and any algorithm class whose constructor takes a `problem` argument and
exposes a parameter-less `.run()` method that returns a list of solutions.

Add new problems / algorithms by registering them once at import time:

    from benchmarking_helper import register_problem, register_algorithm

    @register_problem()
    class MyProblem(MOCOProblem): ...

    @register_algorithm("my_algo")
    class MyAlgo: ...

Then list them in `configs/benchmark_ws_config.yaml`.
"""

import os
import time
import yaml
import numpy as np
from inspect import signature
from typing import Any, Dict, Optional, Tuple, Type

from MOCO.problems import (
    BiObjectiveTSP,
    TriObjectiveTSP,
    MultiObjectiveKnapsack,
    BiObjectiveCVRP,
)
from MOCO.evaluation import MOCOEvaluator


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

PROBLEM_REGISTRY: Dict[str, Type] = {}
ALGORITHM_REGISTRY: Dict[str, Type] = {}


def register_problem(name: Optional[str] = None):
    def deco(cls):
        PROBLEM_REGISTRY[name or cls.__name__] = cls
        return cls
    return deco


def register_algorithm(name: Optional[str] = None):
    def deco(cls):
        ALGORITHM_REGISTRY[name or cls.__name__] = cls
        return cls
    return deco


# Auto-register the built-in problems
for _cls in (BiObjectiveTSP, TriObjectiveTSP, MultiObjectiveKnapsack, BiObjectiveCVRP):
    PROBLEM_REGISTRY.setdefault(_cls.__name__, _cls)


# ---------------------------------------------------------------------------
# Problem introspection
# ---------------------------------------------------------------------------

def infer_num_objectives(problem: Any) -> int:
    """Best-effort detection — mirrors the logic in UniversalValidatedWrapper."""
    for attr in ("num_objectives", "_n_objectives", "_m_objectives", "n_objectives"):
        v = getattr(problem, attr, None)
        if v is not None:
            return int(v() if callable(v) else v)
    name = problem.__class__.__name__
    if "Tri" in name:
        return 3
    if "Bi" in name:
        return 2
    return 2


def infer_problem_size(problem: Any) -> Optional[int]:
    for attr in ("n_cities", "n_items", "n_customers"):
        v = getattr(problem, attr, None)
        if v is not None:
            return int(v)
    return None


_BASE_FROM_CLASS = {
    "BiObjectiveTSP": "TSP",
    "TriObjectiveTSP": "TSP",
    "MultiObjectiveKnapsack": "KP",
    "BiObjectiveCVRP": "CVRP",
}

_PREFIX_FROM_N = {2: "Bi", 3: "Tri"}


def infer_problem_type(problem: Any) -> str:
    """
    Return the standard-points key used by MOCOEvaluator
    (e.g. 'BiTSP', 'TriTSP', 'BiKP', 'TriKP', 'BiCVRP').

    Falls back to the class name for fully custom problems; the evaluator
    handles unknown keys with safe defaults.
    """
    name = problem.__class__.__name__
    base = _BASE_FROM_CLASS.get(name)
    if base is None:
        if "TSP" in name:
            base = "TSP"
        elif "Knapsack" in name or "KP" in name:
            base = "KP"
        elif "CVRP" in name or "VRP" in name:
            base = "CVRP"
        else:
            return name  # let evaluator's default kick in
    n_obj = infer_num_objectives(problem)
    prefix = _PREFIX_FROM_N.get(n_obj, f"{n_obj}Obj")
    return f"{prefix}{base}"


def resolve_reference_points(
    evaluator: MOCOEvaluator, problem: Any
) -> Tuple[tuple, tuple, str, int]:
    """
    Return (reference_point, ideal_point, problem_type, problem_size),
    using the evaluator's standard-points table where possible and
    falling back to dimensionally-correct defaults otherwise.
    """
    ptype = infer_problem_type(problem)
    psize = infer_problem_size(problem) or 0
    n_obj = infer_num_objectives(problem)

    pts: Dict[str, Any] = {}
    try:
        pts = evaluator.get_standard_points(ptype, psize) or {}
    except Exception as e:  # pragma: no cover — diagnostic only
        print(f"  [warn] standard-points lookup failed: {e}")

    ref = pts.get("reference")
    ideal = pts.get("ideal")

    if ref is None or len(ref) != n_obj:
        ref = tuple([10.0] * n_obj)
    if ideal is None or len(ideal) != n_obj:
        ideal = tuple([0.0] * n_obj)

    return tuple(ref), tuple(ideal), ptype, psize


# ---------------------------------------------------------------------------
# Algorithm invocation
# ---------------------------------------------------------------------------

def _instantiate(algorithm_class: Type, problem: Any, params: Dict[str, Any]):
    """Pass `problem` however the algorithm constructor expects it."""
    try:
        init_params = signature(algorithm_class.__init__).parameters
        if "problem" in init_params:
            return algorithm_class(problem=problem, **params)
    except (TypeError, ValueError):
        pass
    return algorithm_class(problem, **params)


def run_algorithm_with_timing(
    algorithm_class: Type,
    problem_class: Type,
    num_runs: int = 30,
    problem_params: Optional[Dict[str, Any]] = None,
    algorithm_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run `algorithm_class` on a fresh `problem_class` instance `num_runs` times."""
    problem_params = problem_params or {}
    algorithm_params = algorithm_params or {}

    run_times: list = []
    solutions_list: list = []

    for run in range(num_runs):
        np.random.seed(run + 1)  # match MOCOEvaluator's seed convention
        problem = problem_class(**problem_params)
        algorithm = _instantiate(algorithm_class, problem, algorithm_params)

        t0 = time.time()
        try:
            solutions = algorithm.run()
            solutions_list.append(solutions)
        except Exception as e:
            print(f"  [run {run + 1}] error: {e}")
            solutions_list.append(None)
        run_times.append(time.time() - t0)

    return {
        "timing_stats": {
            "total_time": float(sum(run_times)),
            "mean_time": float(np.mean(run_times)),
            "std_time": float(np.std(run_times)),
            "min_time": float(np.min(run_times)),
            "max_time": float(np.max(run_times)),
            "runs": num_runs,
        },
        "solutions": solutions_list,
        "run_times": run_times,
    }


def print_timing_details(
    timing_results: Dict[str, Any],
    algorithm_name: str,
    problem_name: str,
    size_label: str,
) -> None:
    s = timing_results["timing_stats"]
    rt = timing_results["run_times"]
    print(f"\nTiming — {algorithm_name} on {problem_name} ({size_label}):")
    print(f"  Runs              : {s['runs']}")
    print(f"  Total / Mean      : {s['total_time']:.4f}s / {s['mean_time']:.4f}s")
    print(f"  Std  / Min / Max  : {s['std_time']:.4f}s / {s['min_time']:.4f}s / {s['max_time']:.4f}s")
    for p in (25, 50, 75, 90):
        print(f"  P{p:<3}              : {np.percentile(rt, p):.4f}s")


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

def benchmark(
    config_path: str = "configs/benchmark_ws_config.yaml",
    results_dir: str = "benchmark_results",
) -> Dict[str, Dict[str, list]]:
    """Run every (problem, size, algorithm) combination listed in the config."""
    if not os.path.exists(config_path):
        generate_default_config(config_path)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    bench_cfg = config.get("benchmark", {})
    num_runs = bench_cfg.get("num_runs", 5)
    problem_entries = bench_cfg.get("problems", [])

    if not problem_entries:
        print("No problems listed in config; nothing to do.")
        return {}

    results: Dict[str, Dict[str, list]] = {}

    for prob_cfg in problem_entries:
        prob_name = prob_cfg["name"]
        if prob_name not in PROBLEM_REGISTRY:
            print(f"Skipping {prob_name!r}: not in PROBLEM_REGISTRY. "
                  f"Register it with @register_problem.")
            continue

        problem_class = PROBLEM_REGISTRY[prob_name]
        sizes = prob_cfg.get("sizes", {})  # {size_label: {**problem_params}}
        if not sizes:
            print(f"  {prob_name}: no sizes defined; skipping.")
            continue

        for size_label, problem_params in sizes.items():
            for algo_cfg in prob_cfg.get("algorithms", []):
                algo_name = algo_cfg["name"]
                if algo_name not in ALGORITHM_REGISTRY:
                    print(f"  Skipping algorithm {algo_name!r}: "
                          f"not in ALGORITHM_REGISTRY.")
                    continue

                algo_class = ALGORITHM_REGISTRY[algo_name]
                algo_params = algo_cfg.get("parameters", {})

                # Probe the problem once to set up the evaluator with the
                # right reference / ideal points.
                probe = problem_class(**problem_params)
                n_obj = infer_num_objectives(probe)
                evaluator = MOCOEvaluator(
                    reference_point=tuple([10.0] * n_obj),
                    results_dir=results_dir,
                )
                ref, ideal, ptype, psize = resolve_reference_points(evaluator, probe)
                evaluator.reference_point = ref

                print(f"\n=== {algo_name} on {prob_name} / {size_label} ===")
                print(f"  problem_params : {problem_params}")
                print(f"  inferred       : type={ptype}, size={psize}, n_obj={n_obj}")
                print(f"  reference      : {ref}")
                print(f"  ideal          : {ideal}")

                try:
                    timing = run_algorithm_with_timing(
                        algorithm_class=algo_class,
                        problem_class=problem_class,
                        problem_params=problem_params,
                        algorithm_params=algo_params,
                        num_runs=num_runs,
                    )
                    print_timing_details(timing, algo_class.__name__, prob_name, size_label)

                    result = evaluator.evaluate_algorithm(
                        algorithm_class=algo_class,
                        problem_class=problem_class,
                        algorithm_name=algo_name,
                        parameters=algo_params,
                        problem_params=problem_params,
                        num_runs=num_runs,
                    )

                    # Attach the timing + normalization metadata to the result.
                    result.timing_details = timing["timing_stats"]
                    result.reference_point = ref
                    result.ideal_point = ideal

                    ref_volume = float(np.prod(np.array(ref) - np.array(ideal)))
                    if hasattr(result, "hypervolume") and ref_volume > 0:
                        result.normalized_hypervolume = result.hypervolume / ref_volume
                    else:
                        result.normalized_hypervolume = 0.0

                    results.setdefault(prob_name, {}).setdefault(size_label, []).append(result)

                    print(f"\n  Runtime              : {result.runtime:.4f}s")
                    if hasattr(result, "hypervolume"):
                        print(f"  Raw HV               : {result.hypervolume:.4f}")
                    print(f"  Normalized HV        : {result.normalized_hypervolume:.4f}")
                    if hasattr(result, "num_nondominated"):
                        print(f"  Non-dominated        : {result.num_nondominated}")

                except Exception as e:
                    print(f"  ERROR: {e}")
                    import traceback
                    traceback.print_exc()

    analyze_results(results)
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def analyze_results(results: Dict[str, Dict[str, list]]) -> None:
    if not results:
        print("\nNo results to analyze.")
        return

    print("\n" + "=" * 60)
    print("Benchmark summary")
    print("=" * 60)

    for prob_name, by_size in results.items():
        print(f"\n{prob_name}")
        for size_label, run_results in by_size.items():
            print(f"  {size_label}:")
            for r in run_results:
                print(f"    {r.algorithm_name}")
                if hasattr(r, "timing_details"):
                    td = r.timing_details
                    print(f"      runtime (total / mean) : "
                          f"{td['total_time']:.4f}s / {td['mean_time']:.4f}s")
                if hasattr(r, "hypervolume"):
                    print(f"      raw HV                 : {r.hypervolume:.4f}")
                if hasattr(r, "normalized_hypervolume"):
                    print(f"      normalized HV          : {r.normalized_hypervolume:.4f}")
                if hasattr(r, "num_nondominated"):
                    print(f"      non-dominated          : {r.num_nondominated}")


# ---------------------------------------------------------------------------
# Default config generator
# ---------------------------------------------------------------------------

def generate_default_config(path: str = "configs/benchmark_ws_config.yaml") -> None:
    """Generate a starter config exercising every registered problem."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    default = {
        "benchmark": {
            "num_runs": 5,
            "problems": [
                {
                    "name": "BiObjectiveTSP",
                    "sizes": {
                        "small": {"n_cities": 20},
                        "medium": {"n_cities": 50},
                    },
                    "algorithms": [
                        {"name": "WSLKH", "parameters": {"num_weights": 40}},
                    ],
                },
                {
                    "name": "TriObjectiveTSP",
                    "sizes": {
                        "small": {"n_cities": 20},
                    },
                    "algorithms": [
                        {"name": "WSLKH", "parameters": {"num_weights": 40}},
                    ],
                },
                {
                    "name": "MultiObjectiveKnapsack",
                    "sizes": {
                        "small": {"n_items": 50, "n_objectives": 2, "capacity": 10.0},
                        "medium": {"n_items": 100, "n_objectives": 3, "capacity": 20.0},
                    },
                    "algorithms": [
                        {"name": "WSDP", "parameters": {"num_weights": 40}},
                    ],
                },
            ],
        }
    }

    with open(path, "w") as f:
        yaml.dump(default, f, default_flow_style=False, sort_keys=False)
    print(f"Wrote default config to {path}")


# ---------------------------------------------------------------------------
# auto-register the WS algorithms if available
# ---------------------------------------------------------------------------

try:
    from WS_LKH_DP import WSLKH, WSDP  # type: ignore
    register_algorithm("WSLKH")(WSLKH)
    register_algorithm("WSDP")(WSDP)
except ImportError:
    pass  # User can still register their own algorithms manually.


if __name__ == "__main__":
    benchmark()