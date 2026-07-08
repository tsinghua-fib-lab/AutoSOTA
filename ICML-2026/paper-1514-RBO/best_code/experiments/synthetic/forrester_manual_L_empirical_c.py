"""Forrester experiment sweeping manual L (plateau_width) with c set to empirical std."""

import os
import torch
import matplotlib.pyplot as plt
import json
from datetime import datetime
from bo_framework import SearchSpace, Dimension
from bo_framework.base.schedulers import (
    ConstantBetaScheduler,
    TheoryGuidedScheduler,
    RCGPScheduler,
)
from bo_framework.synthetic.evaluators import SyntheticEvaluator
from bo_framework.corruption.composable import (
    ComposableCorruptor,
    TimeBudgetDecider,
    PeriodicDecider,
    AdversarialStrategy,
    RandomStrategy,
    ConstantStrategy,
)
from experiments.synthetic.forrester_manual_L import L_VALUES
from experiments.synthetic.functions import ForresterFunction
from utilities.plotting import plot_experiment_summary, PlotConfig
from utilities.io import save_experiment_results, save_comparison_table
from utilities.regret_analysis import (
    compare_experiments_multiseed,
    print_comparison_table_multiseed,
)
from utilities.multiseed_experiments import (
    run_model_across_seeds,
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
SCRIPT_NAME = "forrester_manual_L_c_1.py"
N_ITERATIONS = 100
N_INITIAL = 5
N_SEEDS = 10
HIGH_CORRUPTION_VALUE = 30.0
LOW_CORRUPTION_VALUE = -30.0
ADVERSARIAL_BUDGET = 2
STANDARDIZE = True
NOISE_STD = 1.0

# L (plateau_width) values to sweep
#L_VALUES = [0.0 , 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
L_VALUES = [0.0, 0.75, 0.5, 1.0, 1.5, 2.0]

# Corruption configuration
CORRUPTION_TYPE = "time_budget"
TIME_BUDGET_ALPHA = 1 / 3
PERIODIC_INTERVAL = 10
CORRUPTION_STRATEGY = "adversarial"

# Beta scheduler configuration
RCGP_SCHEDULER_TYPE = "theory"
A2RCGP_SCHEDULER_TYPE = "theory"

# Beta scheduling parameters
CONSTANT_BETA = 2.0
RCGP_SCALE = 1.0
THEORY_SCALE = 1.7
THEORY_OFFSET = 2


def create_rcgp_kwargs(L_value):
    """Create RCGP kwargs with manual plateau_width (L) and empirical std for c."""
    return {
        "param_handling_dict": {
            "plateau_width": {"method": "manual", "value": float(L_value)},
            "c": {"method": "manual", "value": 1.0},
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
            "c": {"method": "manual", "value": 1.0},
            "sigma": {"method": "fit"},
            "mean": {"method": "fit"},
        },
        "outer_param_handling_dict": {
            "plateau_width": {"method": "manual", "value": float(L_value)},
            "c": {"method": "manual", "value": 1.0},
            "sigma": {"method": "fit"},
            "mean": {"method": "fit"},
        },
        "fitting_objective_type": "wloo-cv",
        "optimizer_type": "lbfgs",
        "standardize": STANDARDIZE,
        "verbose": False,
    }


def create_scheduler(scheduler_type, model_type="rcgp"):
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
    folder_name = f"forrester_manual_L_empirical_c_{timestamp}"
    folder_path = os.path.join(base_dir, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path


def save_experiment_config(config_dict, folder_path):
    """Save experiment configuration to JSON file."""
    config_path = os.path.join(folder_path, "experiment_config.json")
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2, default=str)
    print(f"Experiment configuration saved to: {config_path}")


def create_corruptor_factory(corruption_type: str, strategy_type: str):
    """Create a factory function for the specified corruptor configuration."""

    def factory(optimal_point, budget, high_value, low_value):
        if corruption_type == "time_budget":
            decider = TimeBudgetDecider(
                alpha=TIME_BUDGET_ALPHA, skip_initial=True, n_initial=N_INITIAL
            )
        elif corruption_type == "periodic":
            decider = PeriodicDecider(
                period=PERIODIC_INTERVAL, skip_initial=True, n_initial=N_INITIAL
            )
        elif corruption_type == "original":
            from bo_framework.corruption.adversarial import AdversarialCorruptor

            return AdversarialCorruptor(
                optimal_point=optimal_point,
                budget=budget,
                near_threshold=0.1,
                far_threshold=0.4,
                high_value=high_value,
                low_value=low_value,
            )
        else:
            raise ValueError(f"Unknown corruption type: {corruption_type}")

        if strategy_type == "adversarial":
            strategy = AdversarialStrategy(
                optimal_points=optimal_point,
                near_threshold=0.1,
                far_threshold=0.4,
                high_value=high_value,
                low_value=low_value,
            )
        elif strategy_type == "random":
            strategy = RandomStrategy(
                corruption_range=(low_value, high_value),
                distribution="uniform",
                seed=42,
            )
        elif strategy_type == "constant":
            strategy = ConstantStrategy(corruption_value=high_value)
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")

        return ComposableCorruptor(
            decider=decider, strategy=strategy, skip_initial=True
        )

    return factory


def main():
    """Sweep manual L (plateau_width) with empirical std for c on Forrester function."""

    artifacts_dir = create_timestamped_folder()
    print(f"Created experiment folder: {artifacts_dir}")

    config_dict = {
        "experiment_info": {
            "name": "Forrester Manual L Sweep (Empirical c)",
            "timestamp": datetime.now().isoformat(),
            "script": SCRIPT_NAME,
        },
        "experiment_parameters": {
            "N_ITERATIONS": N_ITERATIONS,
            "N_INITIAL": N_INITIAL,
            "N_SEEDS": N_SEEDS,
            "HIGH_CORRUPTION_VALUE": HIGH_CORRUPTION_VALUE,
            "LOW_CORRUPTION_VALUE": LOW_CORRUPTION_VALUE,
            "ADVERSARIAL_BUDGET": ADVERSARIAL_BUDGET,
            "STANDARDIZE": STANDARDIZE,
            "NOISE_STD": NOISE_STD,
            "L_VALUES": L_VALUES,
            "c_method": "empirical_std",
        },
        "corruption_config": {
            "CORRUPTION_TYPE": CORRUPTION_TYPE,
            "TIME_BUDGET_ALPHA": TIME_BUDGET_ALPHA,
            "PERIODIC_INTERVAL": PERIODIC_INTERVAL,
            "CORRUPTION_STRATEGY": CORRUPTION_STRATEGY,
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
    print("c method: empirical_std (standard deviation of standardized Y)")
    print(f"Corruption type: {CORRUPTION_TYPE}")
    if CORRUPTION_TYPE == "time_budget":
        print(f"  Time budget alpha: {TIME_BUDGET_ALPHA} (T^{TIME_BUDGET_ALPHA})")
    elif CORRUPTION_TYPE == "periodic":
        print(f"  Periodic interval: every {PERIODIC_INTERVAL} observations")
    print(f"Corruption strategy: {CORRUPTION_STRATEGY}")
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
    corruptor_factory = create_corruptor_factory(CORRUPTION_TYPE, CORRUPTION_STRATEGY)

    # Collect all results
    all_raw_results = {}
    all_agg_results = {}

    for L in L_VALUES:
        # --- RCGP ---
        rcgp_name = f"RCGP_L={L}"
        rcgp_kwargs = create_rcgp_kwargs(L)
        rcgp_scheduler = create_scheduler(RCGP_SCHEDULER_TYPE, "rcgp")
        print(f"\nRunning {rcgp_name} ...")
        rcgp_all = run_model_across_seeds(
            rcgp_name,
            create_rcgp_model,
            rcgp_kwargs,
            rcgp_scheduler,
            clean_evaluator,
            optimal_point,
            search_space,
            N_SEEDS,
            N_ITERATIONS,
            N_INITIAL,
            ADVERSARIAL_BUDGET,
            HIGH_CORRUPTION_VALUE,
            LOW_CORRUPTION_VALUE,
            corruptor_factory=corruptor_factory,
            noise_std=NOISE_STD,
        )
        all_raw_results[rcgp_name] = rcgp_all
        all_agg_results[rcgp_name] = aggregate_results_across_seeds(rcgp_all)

        # --- A2RCGP ---
        a2rcgp_name = f"A2RCGP_L={L}"
        a2rcgp_kwargs = create_a2rcgp_kwargs(L)
        a2rcgp_scheduler = create_scheduler(A2RCGP_SCHEDULER_TYPE, "a2rcgp")
        print(f"\nRunning {a2rcgp_name} ...")
        a2rcgp_all = run_model_across_seeds(
            a2rcgp_name,
            create_a2rcgp_model,
            a2rcgp_kwargs,
            a2rcgp_scheduler,
            clean_evaluator,
            optimal_point,
            search_space,
            N_SEEDS,
            N_ITERATIONS,
            N_INITIAL,
            ADVERSARIAL_BUDGET,
            HIGH_CORRUPTION_VALUE,
            LOW_CORRUPTION_VALUE,
            corruptor_factory=corruptor_factory,
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
        show_corruption=True,
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
        experiment_name="forrester_manual_L_empirical_c",
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
        experiment_name="forrester_manual_L_empirical_c",
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
