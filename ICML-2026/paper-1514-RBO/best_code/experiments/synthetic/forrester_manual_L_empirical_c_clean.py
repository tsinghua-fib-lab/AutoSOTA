"""Forrester experiment sweeping manual L (plateau_width) with empirical std for c — no corruption."""

import os
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import json
from datetime import datetime
from typing import Dict, Any, List
from bo_framework import SearchSpace, Dimension, ExperimentRunner
from bo_framework.base.schedulers import (
    ConstantBetaScheduler,
    TheoryGuidedScheduler,
    RCGPScheduler,
)
from bo_framework.base.acquisition import UCBAcquisition
from bo_framework.synthetic.evaluators import SyntheticEvaluator
from bo_framework.wrappers.noisy import NoisyEvaluator
from experiments.synthetic.functions import ForresterFunction
from utilities.plotting import PlotConfig
from utilities.io import save_experiment_results, save_comparison_table
from utilities.regret_analysis import (
    compare_experiments_multiseed,
    print_comparison_table_multiseed,
)
from utilities.multiseed_experiments import (
    aggregate_results_across_seeds,
    save_multiseed_summary,
    plot_regret_comparison_multiseed,
)
from bo_framework.models.factory import (
    create_rcgp_model,
    create_a2rcgp_model,
)

# Suppress standardization warnings from BoTorch/GPyTorch
import warnings

warnings.filterwarnings("ignore", message=".*outcome_transform.*")
warnings.filterwarnings("ignore", message=".*standardized.*")
warnings.filterwarnings("ignore", message=".*InputDataWarning.*")

# Experiment parameters
N_ITERATIONS = 30
N_INITIAL = 5
N_SEEDS = 10
STANDARDIZE = False
NOISE_STD = 1.0

# L (plateau_width) values to sweep
L_VALUES = [0, 2, 4, 8, 16]

# Beta scheduler configuration
RCGP_SCHEDULER_TYPE = "theory"
A2RCGP_SCHEDULER_TYPE = "theory"

# Beta scheduling parameters
CONSTANT_BETA = 2.0
RCGP_SCALE = 1.0
THEORY_SCALE = 1.7
THEORY_OFFSET = 2


def set_global_seed(seed: int):
    """Set global seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def run_model_across_seeds_clean(
    model_name: str,
    model_factory,
    model_kwargs: Dict[str, Any],
    scheduler,
    clean_evaluator,
    search_space,
    n_seeds: int,
    n_iterations: int,
    n_initial: int,
    noise_std: float = 1.0,
) -> List[Dict[str, Any]]:
    """Run a model across multiple seeds without corruption."""
    all_results = []

    for seed in range(n_seeds):
        print(f"\nRunning {model_name} with seed {seed + 1}/{n_seeds}...")

        set_global_seed(seed)

        noisy_evaluator = NoisyEvaluator(
            clean_evaluator, noise_std=noise_std, seed=seed
        )

        runner = ExperimentRunner(search_space, noisy_evaluator)

        results = runner.run(
            n_iterations=n_iterations,
            n_initial=n_initial,
            model_factory=model_factory,
            acquisition_factory=UCBAcquisition.create,
            model_kwargs=model_kwargs,
            beta_scheduler=scheduler,
            seed=seed,
            verbose=(seed == 0),
        )

        all_results.append(results)

    return all_results


def create_rcgp_kwargs(L_value):
    """Create RCGP kwargs with manual plateau_width (L) and empirical std for c."""
    return {
        "param_handling_dict": {
            "plateau_width": {"method": "manual", "value": float(L_value)},
            "c": {"method": "empirical_std"},
            "sigma": {"method": "fit"},
            "mean": {"method": "fit"},
        },
        "fitting_objective_type": "wloo-cv",
        "optimizer_type": "lbfgs",
        "standardize": STANDARDIZE,
        "verbose": False,
    }


def create_a2rcgp_kwargs(L_value):
    """Create A2RCGP kwargs with manual plateau_width (L) and empirical std for c."""
    return {
        "inner_param_handling_dict": {
            "plateau_width": {"method": "manual", "value": float(L_value)},
            "c": {"method": "empirical_std"},
            "sigma": {"method": "fit"},
            "mean": {"method": "fit"},
        },
        "outer_param_handling_dict": {
            "plateau_width": {"method": "manual", "value": float(L_value)},
            "c": {"method": "empirical_std"},
            "sigma": {"method": "fit"},
            "mean": {"method": "fit"},
        },
        "fitting_objective_type": "wloo-cv",
        "optimizer_type": "lbfgs",
        "standardize": STANDARDIZE,
        "verbose": False,
    }


def create_scheduler(scheduler_type):
    """Create beta scheduler based on configuration."""
    if scheduler_type == "constant":
        return ConstantBetaScheduler(beta=CONSTANT_BETA)
    elif scheduler_type == "theory":
        return TheoryGuidedScheduler(
            scale=THEORY_SCALE, offset=THEORY_OFFSET, min_beta=1.0
        )
    elif scheduler_type == "rcgp-constant":
        return RCGPScheduler(
            scale=RCGP_SCALE,
            base_scheduler=ConstantBetaScheduler(beta=CONSTANT_BETA),
        )
    elif scheduler_type == "rcgp-theory":
        return RCGPScheduler(
            scale=RCGP_SCALE,
            base_scheduler=TheoryGuidedScheduler(
                scale=THEORY_SCALE, offset=THEORY_OFFSET, min_beta=1.0
            ),
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def create_timestamped_folder(base_dir="artifacts"):
    """Create a timestamped folder for experiment results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"forrester_manual_L_empirical_c_clean_{timestamp}"
    folder_path = os.path.join(base_dir, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path


def save_experiment_config(config_dict, folder_path):
    """Save experiment configuration to JSON file."""
    config_path = os.path.join(folder_path, "experiment_config.json")
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2, default=str)
    print(f"Experiment configuration saved to: {config_path}")


def main():
    """Sweep manual L with empirical std for c on Forrester — no corruption."""

    artifacts_dir = create_timestamped_folder()
    print(f"Created experiment folder: {artifacts_dir}")

    config_dict = {
        "experiment_info": {
            "name": "Forrester Manual L Sweep (Empirical c, Clean)",
            "timestamp": datetime.now().isoformat(),
            "script": "forrester_manual_L_empirical_c_clean.py",
        },
        "experiment_parameters": {
            "N_ITERATIONS": N_ITERATIONS,
            "N_INITIAL": N_INITIAL,
            "N_SEEDS": N_SEEDS,
            "STANDARDIZE": STANDARDIZE,
            "NOISE_STD": NOISE_STD,
            "L_VALUES": L_VALUES,
            "c_method": "empirical_std",
            "corruption": False,
        },
        "scheduler_config": {
            "RCGP_SCHEDULER_TYPE": RCGP_SCHEDULER_TYPE,
            "A2RCGP_SCHEDULER_TYPE": A2RCGP_SCHEDULER_TYPE,
            "CONSTANT_BETA": CONSTANT_BETA,
            "RCGP_SCALE": RCGP_SCALE,
            "THEORY_SCALE": THEORY_SCALE,
            "THEORY_OFFSET": THEORY_OFFSET,
        },
    }

    save_experiment_config(config_dict, artifacts_dir)

    print("\n" + "=" * 80)
    print("EXPERIMENT CONFIGURATION")
    print("=" * 80)
    print(f"Iterations: {N_ITERATIONS}, Initial points: {N_INITIAL}")
    print(f"Number of seeds: {N_SEEDS}")
    print(f"L values: {L_VALUES}")
    print(f"c method: empirical_std (standard deviation of standardized Y)")
    print(f"Corruption: NONE (clean observations with noise_std={NOISE_STD})")
    print(f"RCGP scheduler: {RCGP_SCHEDULER_TYPE}")
    print(f"A2RCGP scheduler: {A2RCGP_SCHEDULER_TYPE}")
    print("=" * 80 + "\n")

    forrester_func = ForresterFunction()
    optimal_point = torch.tensor([1.0], dtype=torch.double)
    optimal_value = forrester_func.optimal_value

    search_space = SearchSpace(
        (Dimension(name="x0", type="continuous", bounds=(0.0, 1.0), normalize=True),)
    )

    clean_evaluator = SyntheticEvaluator(forrester_func)

    # Collect all results
    all_raw_results = {}
    all_agg_results = {}

    for L in L_VALUES:
        # --- RCGP ---
        rcgp_name = f"RCGP_L={L}"
        rcgp_kwargs = create_rcgp_kwargs(L)
        rcgp_scheduler = create_scheduler(RCGP_SCHEDULER_TYPE)
        print(f"\nRunning {rcgp_name} ...")
        rcgp_all = run_model_across_seeds_clean(
            rcgp_name,
            create_rcgp_model,
            rcgp_kwargs,
            rcgp_scheduler,
            clean_evaluator,
            search_space,
            N_SEEDS,
            N_ITERATIONS,
            N_INITIAL,
            noise_std=NOISE_STD,
        )
        all_raw_results[rcgp_name] = rcgp_all
        all_agg_results[rcgp_name] = aggregate_results_across_seeds(rcgp_all)

        # --- A2RCGP ---
        a2rcgp_name = f"A2RCGP_L={L}"
        a2rcgp_kwargs = create_a2rcgp_kwargs(L)
        a2rcgp_scheduler = create_scheduler(A2RCGP_SCHEDULER_TYPE)
        print(f"\nRunning {a2rcgp_name} ...")
        a2rcgp_all = run_model_across_seeds_clean(
            a2rcgp_name,
            create_a2rcgp_model,
            a2rcgp_kwargs,
            a2rcgp_scheduler,
            clean_evaluator,
            search_space,
            N_SEEDS,
            N_ITERATIONS,
            N_INITIAL,
            noise_std=NOISE_STD,
        )
        all_raw_results[a2rcgp_name] = a2rcgp_all
        all_agg_results[a2rcgp_name] = aggregate_results_across_seeds(a2rcgp_all)

    # Compare results
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS (AGGREGATED ACROSS SEEDS)")
    print("=" * 80)

    multiseed_metrics_dict = compare_experiments_multiseed(
        results_dict={
            name: agg["all_results"] for name, agg in all_agg_results.items()
        },
        optimal_value=optimal_value,
    )

    print_comparison_table_multiseed(
        multiseed_metrics_dict,
        show_regret=True,
        show_corruption=False,
        show_std=True,
    )

    # Save results
    print("\n" + "=" * 80)
    print("SAVING RESULTS AND GENERATING PLOTS")
    print("=" * 80)

    multiseed_results_dict = {
        f"{model_name}_seed_{seed}": all_raw_results[model_name][seed]
        for model_name in all_raw_results
        for seed in range(len(all_raw_results[model_name]))
    }

    save_experiment_results(
        results=multiseed_results_dict,
        experiment_name="forrester_manual_L_empirical_c_clean",
        artifacts_dir=artifacts_dir,
        save_pickle=True,
        save_json=True,
        optimal_value=optimal_value,
        verbose=True,
    )

    save_comparison_table(
        results_dict={
            name: agg["all_results_flat"] for name, agg in all_agg_results.items()
        },
        experiment_name="forrester_manual_L_empirical_c_clean",
        artifacts_dir=artifacts_dir,
        optimal_value=optimal_value,
    )

    save_multiseed_summary(
        results_dict=all_agg_results,
        optimal_value=optimal_value,
        artifacts_dir=artifacts_dir,
    )

    # Regret comparison plots
    config = PlotConfig(figsize=(15, 10))
    regret_save_path = os.path.join(artifacts_dir, "regret")

    regret_fig, simple_regret_fig = plot_regret_comparison_multiseed(
        results_dict=all_agg_results,
        optimal_value=optimal_value,
        n_seeds=N_SEEDS,
        save_path=regret_save_path,
        config=config,
    )

    plt.close(regret_fig)
    plt.close(simple_regret_fig)

    print(f"\nAll artifacts saved to: {artifacts_dir}/")


if __name__ == "__main__":
    main()
