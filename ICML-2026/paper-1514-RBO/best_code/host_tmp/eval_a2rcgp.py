#!/usr/bin/env python3
"""Reproduction eval script for paper-1514: A2-RCGP-UCB on Forrester benchmark.

Rubric: n_iterations=105 (5 initial + 100 BO), n_seeds=10, corruption_magnitude=20,
input_standardisation=True, setting=corrupted, model=A2-RCGP-UCB.

Target: Cumulative Regret per Table 7 (Appendix H.4).
Paper reports: 185.84 +/- 26.01 for best config (LR=1.5, LA=0.5).

Usage: python3 reproduce_a2rcgp_forrester.py
Output: prints final cumulative regret and saves JSON to artifacts/reproduction/
"""
import os, sys, torch, json, warnings
from datetime import datetime

warnings.filterwarnings("ignore")

from bo_framework import SearchSpace, Dimension
from bo_framework.base.schedulers import TheoryGuidedScheduler
from bo_framework.synthetic.evaluators import SyntheticEvaluator
from bo_framework.corruption.composable import (
    ComposableCorruptor, TimeBudgetDecider, AdversarialStrategy,
)
from experiments.synthetic.functions import ForresterFunction
from bo_framework.models.factory import create_a2rcgp_model
from utilities.multiseed_experiments import (
    run_model_across_seeds, aggregate_results_across_seeds,
)
from utilities.regret_analysis import (
    compare_experiments_multiseed, print_comparison_table_multiseed,
)

# --- Parameters matching rubric and paper Table 3/7 defaults ---
N_INITIAL = 5
N_BO_ITERATIONS = 100  # total = 105 evaluations
N_SEEDS = 10
HIGH_CORRUPTION_VALUE = 20.0
LOW_CORRUPTION_VALUE = -20.0
STANDARDIZE = True
NOISE_STD = 1.0

# Corruption: T^1/3 frequency-constrained budget (paper Section 5.3)
TIME_BUDGET_ALPHA = 1 / 3

# A2RCGP: (LR=1.5, LA=0.5) -- best per Table 3 (Appendix H.1)
a2rcgp_kwargs = {
    "inner_param_handling_dict": {
        "plateau_width": {"method": "manual", "value": 0.5},  # LA (anchor)
        "c": {"method": "empirical_std"},
        "sigma": {"method": "fit"},
        "mean": {"method": "fit"},
    },
    "outer_param_handling_dict": {
        "plateau_width": {"method": "manual", "value": 1.5},  # LR (adaptive)
        "c": {"method": "empirical_std"},
        "sigma": {"method": "fit"},
        "mean": {"method": "fit"},
    },
    "fitting_objective_type": "wloo-cv",
    "optimizer_type": "lbfgs",
    "standardize": STANDARDIZE,
    "verbose": False,
}


def create_corruptor_factory():
    def factory(optimal_point, budget, high_value, low_value):
        decider = TimeBudgetDecider(alpha=TIME_BUDGET_ALPHA, skip_initial=True, n_initial=N_INITIAL)
        strategy = AdversarialStrategy(
            optimal_points=optimal_point,
            near_threshold=0.2, far_threshold=0.5,
            high_value=high_value, low_value=low_value,
        )
        return ComposableCorruptor(decider=decider, strategy=strategy, skip_initial=True)
    return factory


def main():
    print("=" * 80)
    print("A2-RCGP-UCB Forrester Reproduction (Paper 1514)")
    print(f"Total evaluations: {N_INITIAL + N_BO_ITERATIONS}  Seeds: {N_SEEDS}")
    print(f"Corruption: +-{HIGH_CORRUPTION_VALUE}  Standardize: {STANDARDIZE}")
    print("=" * 80 + "\n")

    forrester_func = ForresterFunction()
    optimal_point = torch.tensor([1.0], dtype=torch.double)
    optimal_value = forrester_func.optimal_value

    search_space = SearchSpace(
        (Dimension(name="x0", type="continuous", bounds=(0.0, 1.0), normalize=True),)
    )
    clean_evaluator = SyntheticEvaluator(forrester_func)
    corruptor_factory = create_corruptor_factory()
    scheduler = TheoryGuidedScheduler(scale=1.7, offset=2, min_beta=1.0)

    print("Running A2-RCGP-UCB across 10 seeds...")
    all_results = run_model_across_seeds(
        "A2RCGP", create_a2rcgp_model, a2rcgp_kwargs, scheduler,
        clean_evaluator, optimal_point, search_space,
        N_SEEDS, N_BO_ITERATIONS, N_INITIAL,
        2, HIGH_CORRUPTION_VALUE, LOW_CORRUPTION_VALUE,
        corruptor_factory=corruptor_factory, noise_std=NOISE_STD,
    )

    results = aggregate_results_across_seeds(all_results)
    multiseed_metrics = compare_experiments_multiseed(
        results_dict={"A2RCGP": results["all_results"]},
        optimal_value=optimal_value,
    )
    print_comparison_table_multiseed(multiseed_metrics, show_regret=True, show_corruption=True, show_std=True)

    a2 = multiseed_metrics["A2RCGP"]
    cum_regret_mean = a2["final_cumulative_regret_mean"]
    cum_regret_std = a2["final_cumulative_regret_std"]

    print(f"\nCUMULATIVE_REGRET_MEAN={cum_regret_mean:.2f}")
    print(f"CUMULATIVE_REGRET_STD={cum_regret_std:.2f}")
    print(f"PAPER_TARGET=185.84  REPRO_CI=[159.83, 211.85]")

    # Save results
    out_dir = "/repo/artifacts/reproduction"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "a2rcgp_forrester_results.json")
    serializable = {
        "paper_id": 1514, "model": "A2-RCGP-UCB", "benchmark": "Forrester",
        "timestamp": datetime.now().isoformat(),
        "parameters": {
            "n_initial": N_INITIAL, "n_bo_iterations": N_BO_ITERATIONS,
            "n_total": N_INITIAL + N_BO_ITERATIONS, "n_seeds": N_SEEDS,
            "corruption_magnitude": HIGH_CORRUPTION_VALUE,
            "corruption_alpha": TIME_BUDGET_ALPHA,
            "standardize": STANDARDIZE, "noise_std": NOISE_STD,
            "plateau_widths": {"LA": 0.5, "LR": 1.5},
        },
        "cumulative_regret": {
            "mean": float(cum_regret_mean) if cum_regret_mean else None,
            "std": float(cum_regret_std) if cum_regret_std else None,
            "paper_value": 185.84, "paper_ci": [159.83, 211.85],
        },
    }
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Results saved: {out_path}")


if __name__ == "__main__":
    main()
