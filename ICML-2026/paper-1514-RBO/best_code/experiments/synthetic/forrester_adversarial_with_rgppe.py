"""Forrester adversarial experiment with RGP-PE baseline.

Compares RCGP, GP, A2RCGP, and the Robust GP Phased Elimination (RGP-PE)
algorithm from Bogunovic et al. (2022) under adversarial corruption.

DiagnosticGP and Student-t baselines are removed relative to the original
``forrester_adversarial.py`` script.
"""

import os
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import json
from datetime import datetime

from bo_framework import SearchSpace, Dimension, ExperimentRunner
from bo_framework.base.schedulers import (
    ConstantBetaScheduler,
    TheoryGuidedScheduler,
    RCGPScheduler,
)
from bo_framework.base.acquisition import UCBAcquisition
from bo_framework.synthetic.evaluators import SyntheticEvaluator
from bo_framework.corruption.composable import (
    ComposableCorruptor,
    TimeBudgetDecider,
    PeriodicDecider,
    AdversarialStrategy,
    RandomStrategy,
    ConstantStrategy,
)
from bo_framework.corruption.adversarial import AdversarialCorruptor
from bo_framework.wrappers.noisy import NoisyEvaluator
from bo_framework.wrappers.corrupted import CorruptedEvaluator
from experiments.synthetic.functions import ForresterFunction
from utilities.plotting import plot_experiment_summary, PlotConfig
from utilities.io import save_experiment_results, save_comparison_table
from utilities.regret_analysis import (
    compare_experiments,
    print_comparison_table,
    compare_experiments_multiseed,
    print_comparison_table_multiseed,
)
from utilities.multiseed_experiments import (
    run_model_across_seeds,
    set_global_seed,
    aggregate_results_across_seeds,
    save_individual_seed_results,
    save_multiseed_summary,
    plot_regret_comparison_multiseed,
)
from bo_framework.models.factory import (
    create_rcgp_model,
    create_gp_model,
    create_a2rcgp_model,
)
from rcgp.algorithms.rgp_pe import RobustGPPhasedElimination
from torch.quasirandom import SobolEngine

# Suppress standardization warnings from BoTorch/GPyTorch
import warnings

warnings.filterwarnings("ignore", message=".*outcome_transform.*")
warnings.filterwarnings("ignore", message=".*standardized.*")
warnings.filterwarnings("ignore", message=".*InputDataWarning.*")

# ======================================================================
# Experiment parameters
# ======================================================================
N_ITERATIONS = 100
N_INITIAL = 5
N_SEEDS = 10
HIGH_CORRUPTION_VALUE = 20.0
LOW_CORRUPTION_VALUE = -20.0
ADVERSARIAL_BUDGET = 2
FIT_STANDARD_GP = True
STANDARDIZE = False
CUSTOM_GP_MODEL = False
NOISE_STD = 1.0

# Corruption configuration
CORRUPTION_TYPE = "time_budget"
TIME_BUDGET_ALPHA = 1 / 3
PERIODIC_INTERVAL = 10
CORRUPTION_STRATEGY = "adversarial"

# Beta scheduler configuration
RCGP_SCHEDULER_TYPE = "theory"
GP_SCHEDULER_TYPE = "theory"
A2RCGP_SCHEDULER_TYPE = "theory"

CONSTANT_BETA = 2.0
RCGP_SCALE = 1.0
THEORY_SCALE = 1.7
THEORY_OFFSET = 2

# ======================================================================
# RGP-PE parameters (from Bogunovic et al., 2022, Section 4.4)
# ======================================================================
RGPPE_ETA = 2.0           # switching parameter
RGPPE_PSI = 0.5           # truncation parameter
RGPPE_BETA = 4.0          # constant confidence bound
RGPPE_LAMBDA = 1.0        # regularisation
RGPPE_B = 0.1             # practical confidence scaling
RGPPE_N_GRID = 100        # domain discretisation (100 points in [0,1])
RGPPE_STEPS_PER_EPOCH = 5  # exploration evaluations per epoch

# ======================================================================
# Model configurations
# ======================================================================
rcgp_kwargs = {
    "param_handling_dict": {
        "plateau_width": {"method": "heuristics", "value": 2.0},
        "c": {"method": "manual", "value": 1.0},
        "sigma": {"method": "fit"},
        "mean": {"method": "fit"},
    },
    "fitting_objective_type": "wloo-cv",
    "optimizer_type": "lbfgs",
    "standardize": STANDARDIZE,
    "verbose": False,
}

a2rcgp_kwargs = {
    "inner_param_handling_dict": {
        "plateau_width": {"method": "heuristics", "value": 2.0},
        "c": {"method": "manual", "value": 1.0},
        "sigma": {"method": "fit"},
        "mean": {"method": "fit"},
    },
    "outer_param_handling_dict": {
        "plateau_width": {"method": "heuristics", "value": 1.5},
        "c": {"method": "manual", "value": 0.8},
        "sigma": {"method": "fit"},
        "mean": {"method": "fit"},
    },
    "fitting_objective_type": "wloo-cv",
    "optimizer_type": "lbfgs",
    "standardize": STANDARDIZE,
    "verbose": False,
}


# ======================================================================
# Scheduler factory
# ======================================================================
def create_scheduler(scheduler_type, model_type="gp"):
    if model_type in ("rcgp", "a2rcgp"):
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
    else:
        if scheduler_type == "constant":
            return ConstantBetaScheduler(beta=CONSTANT_BETA)
        elif scheduler_type == "theory":
            return TheoryGuidedScheduler(
                scale=THEORY_SCALE, offset=THEORY_OFFSET, min_beta=1.0
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")


# ======================================================================
# Helpers
# ======================================================================
def create_timestamped_folder(base_dir="artifacts"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"forrester_rgppe_experiment_{timestamp}"
    folder_path = os.path.join(base_dir, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path


def save_experiment_config(config_dict, folder_path):
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


# ======================================================================
# RGP-PE runner
# ======================================================================
def _generate_initial_points(search_space, n_initial, seed):
    """Replicate the same Sobol initial-point generation as ExperimentRunner."""
    sobol = SobolEngine(search_space.n_dims, scramble=True, seed=seed)
    pts_01 = sobol.draw(n_initial).double()
    bounds = search_space.bounds
    return pts_01 * (bounds[1] - bounds[0]) + bounds[0]


def run_rgppe_across_seeds(
    corruptor_factory,
    clean_evaluator,
    optimal_point,
    search_space,
    n_seeds,
    n_iterations,
    n_initial,
    adversarial_budget,
    high_corruption_value,
    low_corruption_value,
    noise_std=1.0,
    rgppe_kwargs=None,
):
    """Run RGP-PE across multiple seeds, returning the same structure
    as ``run_model_across_seeds``."""
    rgppe_kwargs = rgppe_kwargs or {}
    all_results = []

    for seed in range(n_seeds):
        print(f"\nRunning RGP-PE with seed {seed + 1}/{n_seeds}...")
        set_global_seed(seed)

        # --- build evaluator chain (same as other models) ---
        noisy_eval = NoisyEvaluator(clean_evaluator, noise_std=noise_std, seed=seed)
        if corruptor_factory:
            corruptor = corruptor_factory(
                optimal_point,
                adversarial_budget,
                high_corruption_value,
                low_corruption_value,
            )
        else:
            corruptor = AdversarialCorruptor(
                optimal_point=optimal_point,
                budget=adversarial_budget,
                near_threshold=0.1,
                far_threshold=0.4,
                high_value=high_corruption_value,
                low_value=low_corruption_value,
            )
        corrupted_eval = CorruptedEvaluator(
            base_evaluator=noisy_eval, corruptor=corruptor, n_initial=n_initial
        )

        # --- generate and evaluate initial points (same Sobol sequence) ---
        X_init = _generate_initial_points(search_space, n_initial, seed)
        initial_results = []
        for x in X_init:
            params = search_space.decode_point(x)
            result = corrupted_eval.evaluate(params)
            initial_results.append(result)

        # --- build discretised domain for RGP-PE ---
        grid = torch.linspace(0.0, 1.0, RGPPE_N_GRID, dtype=torch.double).unsqueeze(-1)

        rgp_pe = RobustGPPhasedElimination(
            domain_points=grid,
            eta=rgppe_kwargs.get("eta", RGPPE_ETA),
            psi=rgppe_kwargs.get("psi", RGPPE_PSI),
            beta=rgppe_kwargs.get("beta", RGPPE_BETA),
            lambda_reg=rgppe_kwargs.get("lambda_reg", RGPPE_LAMBDA),
            b=rgppe_kwargs.get("b", RGPPE_B),
            corruption_budget=rgppe_kwargs.get("corruption_budget", high_corruption_value),
            steps_per_epoch=rgppe_kwargs.get("steps_per_epoch", RGPPE_STEPS_PER_EPOCH),
        )

        results = rgp_pe.run(
            evaluator=corrupted_eval,
            search_space=search_space,
            total_budget=n_iterations,
            initial_results=initial_results,
            verbose=(seed == 0),
        )

        # Attach seed for bookkeeping
        results["seed"] = seed
        all_results.append(results)

    return all_results


# ======================================================================
# Main
# ======================================================================
def main():
    artifacts_dir = create_timestamped_folder()
    print(f"Created experiment folder: {artifacts_dir}")

    config_dict = {
        "experiment_info": {
            "name": "Forrester Adversarial + RGP-PE Experiment",
            "timestamp": datetime.now().isoformat(),
            "script": "forrester_adversarial_with_rgppe.py",
        },
        "experiment_parameters": {
            "N_ITERATIONS": N_ITERATIONS,
            "N_INITIAL": N_INITIAL,
            "N_SEEDS": N_SEEDS,
            "HIGH_CORRUPTION_VALUE": HIGH_CORRUPTION_VALUE,
            "LOW_CORRUPTION_VALUE": LOW_CORRUPTION_VALUE,
            "ADVERSARIAL_BUDGET": ADVERSARIAL_BUDGET,
            "STANDARDIZE": STANDARDIZE,
            "FIT_STANDARD_GP": FIT_STANDARD_GP,
            "NOISE_STD": NOISE_STD,
        },
        "corruption_config": {
            "CORRUPTION_TYPE": CORRUPTION_TYPE,
            "TIME_BUDGET_ALPHA": TIME_BUDGET_ALPHA,
            "PERIODIC_INTERVAL": PERIODIC_INTERVAL,
            "CORRUPTION_STRATEGY": CORRUPTION_STRATEGY,
        },
        "scheduler_config": {
            "RCGP_SCHEDULER_TYPE": RCGP_SCHEDULER_TYPE,
            "GP_SCHEDULER_TYPE": GP_SCHEDULER_TYPE,
            "A2RCGP_SCHEDULER_TYPE": A2RCGP_SCHEDULER_TYPE,
            "CONSTANT_BETA": CONSTANT_BETA,
            "RCGP_SCALE": RCGP_SCALE,
            "THEORY_SCALE": THEORY_SCALE,
            "THEORY_OFFSET": THEORY_OFFSET,
        },
        "rgppe_config": {
            "RGPPE_ETA": RGPPE_ETA,
            "RGPPE_PSI": RGPPE_PSI,
            "RGPPE_BETA": RGPPE_BETA,
            "RGPPE_LAMBDA": RGPPE_LAMBDA,
            "RGPPE_B": RGPPE_B,
            "RGPPE_N_GRID": RGPPE_N_GRID,
            "RGPPE_STEPS_PER_EPOCH": RGPPE_STEPS_PER_EPOCH,
        },
        "model_configs": {
            "rcgp_kwargs": rcgp_kwargs,
            "a2rcgp_kwargs": a2rcgp_kwargs,
        },
    }
    save_experiment_config(config_dict, artifacts_dir)

    # Print configuration
    print("\n" + "=" * 80)
    print("EXPERIMENT CONFIGURATION")
    print("=" * 80)
    print(f"Iterations: {N_ITERATIONS}, Initial points: {N_INITIAL}")
    print(f"Number of seeds: {N_SEEDS}")
    print(f"Corruption type: {CORRUPTION_TYPE}")
    if CORRUPTION_TYPE == "time_budget":
        print(f"  Time budget alpha: {TIME_BUDGET_ALPHA} (T^{TIME_BUDGET_ALPHA})")
    elif CORRUPTION_TYPE == "periodic":
        print(f"  Periodic interval: every {PERIODIC_INTERVAL} observations")
    elif CORRUPTION_TYPE == "original":
        print(f"  Adversarial budget: {ADVERSARIAL_BUDGET}")
    print(f"Corruption strategy: {CORRUPTION_STRATEGY}")
    print(f"RCGP scheduler: {RCGP_SCHEDULER_TYPE}")
    print(f"GP scheduler: {GP_SCHEDULER_TYPE}")
    print(f"A2RCGP scheduler: {A2RCGP_SCHEDULER_TYPE}")
    print(
        f"RGP-PE: eta={RGPPE_ETA}, psi={RGPPE_PSI}, beta={RGPPE_BETA}, "
        f"steps_per_epoch={RGPPE_STEPS_PER_EPOCH}, grid={RGPPE_N_GRID}"
    )
    print("=" * 80 + "\n")

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    forrester_func = ForresterFunction()
    optimal_point = torch.tensor([1.0], dtype=torch.double)
    optimal_value = forrester_func.optimal_value

    search_space = SearchSpace(
        (Dimension(name="x0", type="continuous", bounds=(0.0, 1.0), normalize=True),)
    )
    clean_evaluator = SyntheticEvaluator(forrester_func)
    corruptor_factory = create_corruptor_factory(CORRUPTION_TYPE, CORRUPTION_STRATEGY)

    # ------------------------------------------------------------------
    # Run baselines: RCGP, GP, A2RCGP
    # ------------------------------------------------------------------
    rcgp_scheduler = create_scheduler(RCGP_SCHEDULER_TYPE, "rcgp")
    print(f"RCGP using scheduler: {rcgp_scheduler.__class__.__name__}")
    rcgp_all_results = run_model_across_seeds(
        "RCGP",
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

    gp_model_kwargs = {
        "fit_hyperparameters": FIT_STANDARD_GP,
        "standardize": STANDARDIZE,
        "use_botorch_model": not CUSTOM_GP_MODEL,
    }
    gp_scheduler = create_scheduler(GP_SCHEDULER_TYPE, "gp")
    print(f"GP using scheduler: {gp_scheduler.__class__.__name__}")
    gp_all_results = run_model_across_seeds(
        "GP",
        create_gp_model,
        gp_model_kwargs,
        gp_scheduler,
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

    a2rcgp_scheduler = create_scheduler(A2RCGP_SCHEDULER_TYPE, "a2rcgp")
    print(f"A2RCGP using scheduler: {a2rcgp_scheduler.__class__.__name__}")
    a2rcgp_all_results = run_model_across_seeds(
        "A2RCGP",
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

    # ------------------------------------------------------------------
    # Run RGP-PE
    # ------------------------------------------------------------------
    print(f"\nRGP-PE using eta={RGPPE_ETA}, psi={RGPPE_PSI}, beta={RGPPE_BETA}")
    rgppe_all_results = run_rgppe_across_seeds(
        corruptor_factory=corruptor_factory,
        clean_evaluator=clean_evaluator,
        optimal_point=optimal_point,
        search_space=search_space,
        n_seeds=N_SEEDS,
        n_iterations=N_ITERATIONS,
        n_initial=N_INITIAL,
        adversarial_budget=ADVERSARIAL_BUDGET,
        high_corruption_value=HIGH_CORRUPTION_VALUE,
        low_corruption_value=LOW_CORRUPTION_VALUE,
        noise_std=NOISE_STD,
        rgppe_kwargs={
            "eta": RGPPE_ETA,
            "psi": RGPPE_PSI,
            "beta": RGPPE_BETA,
            "lambda_reg": RGPPE_LAMBDA,
            "b": RGPPE_B,
            "corruption_budget": HIGH_CORRUPTION_VALUE,
            "steps_per_epoch": RGPPE_STEPS_PER_EPOCH,
        },
    )

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------
    rcgp_results = aggregate_results_across_seeds(rcgp_all_results)
    gp_results = aggregate_results_across_seeds(gp_all_results)
    a2rcgp_results = aggregate_results_across_seeds(a2rcgp_all_results)
    rgppe_results = aggregate_results_across_seeds(rgppe_all_results)

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS (AGGREGATED ACROSS SEEDS)")
    print("=" * 80)

    scenarios = [
        ("RCGP", rcgp_results),
        ("GP", gp_results),
        ("A2RCGP", a2rcgp_results),
        ("RGP-PE", rgppe_results),
    ]

    multiseed_metrics_dict = compare_experiments_multiseed(
        results_dict={name: results["all_results"] for name, results in scenarios},
        optimal_value=optimal_value,
    )
    print_comparison_table_multiseed(
        multiseed_metrics_dict,
        show_regret=True,
        show_corruption=True,
        show_std=True,
    )

    # First-seed reference
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS (FIRST SEED ONLY)")
    print("=" * 80)
    metrics_dict = compare_experiments(
        results_dict={name: results["all_results_flat"] for name, results in scenarios},
        optimal_value=optimal_value,
    )
    print_comparison_table(metrics_dict, show_regret=True, show_corruption=True)

    # ------------------------------------------------------------------
    # Save results & plots
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SAVING RESULTS AND GENERATING PLOTS")
    print("=" * 80)

    multiseed_results_dict = {
        f"{model_name}_seed_{seed}": all_results[seed]
        for model_name, all_results in [
            ("RCGP", rcgp_all_results),
            ("GP", gp_all_results),
            ("A2RCGP", a2rcgp_all_results),
            ("RGP-PE", rgppe_all_results),
        ]
        for seed in range(len(all_results))
    }

    save_experiment_results(
        results=multiseed_results_dict,
        experiment_name="forrester_rgppe_experiment",
        artifacts_dir=artifacts_dir,
        save_pickle=True,
        save_json=True,
        optimal_value=optimal_value,
        verbose=True,
    )

    print("Saving individual seed JSON files...")
    for model_name, all_results in [
        ("RCGP", rcgp_all_results),
        ("GP", gp_all_results),
        ("A2RCGP", a2rcgp_all_results),
        ("RGP-PE", rgppe_all_results),
    ]:
        for seed, results in enumerate(all_results):
            seed_path = save_individual_seed_results(
                model_name=model_name,
                seed=seed,
                results=results,
                optimal_value=optimal_value,
                artifacts_dir=artifacts_dir,
            )
            if seed == 0:
                print(f"  {model_name} individual seeds saved (e.g., {seed_path})")

    save_comparison_table(
        results_dict={name: results["all_results_flat"] for name, results in scenarios},
        experiment_name="forrester_rgppe_experiment",
        artifacts_dir=artifacts_dir,
        optimal_value=optimal_value,
    )

    save_multiseed_summary(
        results_dict={name: results for name, results in scenarios},
        optimal_value=optimal_value,
        artifacts_dir=artifacts_dir,
    )

    # --- Plots ---
    config = PlotConfig(figsize=(15, 10))

    # Summary plots for each model (first seed only, skip RGP-PE since it has no model)
    first_seed_scenarios = [
        ("RCGP", rcgp_all_results[0]),
        ("GP", gp_all_results[0]),
        ("A2RCGP", a2rcgp_all_results[0]),
    ]
    for name, results in first_seed_scenarios:
        print(f"Creating summary plot for: {name} (seed 0)")
        plot_path = os.path.join(
            artifacts_dir, f"forrester_{name.lower().replace(' ', '_')}_seed0.png"
        )
        fig = plot_experiment_summary(
            results=results,
            objective_func=lambda x: forrester_func.evaluate(x),
            optimal_value=optimal_value,
            save_path=plot_path,
            config=config,
        )
        plt.close(fig)

    # Regret comparison plots
    print("\n" + "=" * 80)
    print("CREATING REGRET COMPARISON PLOTS")
    print("=" * 80)

    results_dict_for_plot = {name: results for name, results in scenarios}
    colors = {
        "RCGP": "blue",
        "GP": "orange",
        "A2RCGP": "red",
        "RGP-PE": "purple",
    }

    regret_save_path = os.path.join(artifacts_dir, "regret")
    regret_fig, simple_regret_fig = plot_regret_comparison_multiseed(
        results_dict=results_dict_for_plot,
        optimal_value=optimal_value,
        n_seeds=N_SEEDS,
        save_path=regret_save_path,
        config=config,
        colors=colors,
    )
    plt.close(regret_fig)
    plt.close(simple_regret_fig)

    # ------------------------------------------------------------------
    # Print hyperparameters
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("MODEL HYPERPARAMETERS (Seed 0)")
    print("=" * 80)

    # RCGP
    rcgp_model = rcgp_results["final_model"]
    print(f"\nRCGP:")
    print(f"  Noise std: {torch.sqrt(rcgp_model.likelihood.noise).item():.4f}")
    covar = rcgp_model.covar_module
    if hasattr(covar, "base_kernel"):
        print(f"  Lengthscale: {covar.base_kernel.lengthscale.item():.4f}")
        print(f"  Output scale: {covar.outputscale.item():.4f}")
    if hasattr(rcgp_model.mean_module, "constant"):
        print(f"  Mean: {rcgp_model.mean_module.constant.item():.4f}")
    print(f"  Plateau width: {rcgp_model.weighting_function.plateau_width:.4f}")
    print(f"  C param: {rcgp_model.weighting_function.c:.4f}")

    # GP
    gp_model = gp_results["final_model"]
    print(f"\nGP:")
    print(f"  Noise std: {torch.sqrt(gp_model.likelihood.noise).item():.4f}")
    covar = gp_model.covar_module
    if hasattr(covar, "base_kernel"):
        print(f"  Lengthscale: {covar.base_kernel.lengthscale.item():.4f}")
        print(f"  Output scale: {covar.outputscale.item():.4f}")

    # A2RCGP
    a2rcgp_model = a2rcgp_results["final_model"]
    print(f"\nA2RCGP:")
    print(
        f"  Inner noise std: {torch.sqrt(a2rcgp_model.inner_rcgp.likelihood.noise).item():.4f}"
    )
    inner_covar = a2rcgp_model.inner_rcgp.covar_module
    if hasattr(inner_covar, "base_kernel"):
        print(f"  Inner lengthscale: {inner_covar.base_kernel.lengthscale.item():.4f}")
    print(
        f"  Inner plateau width: {a2rcgp_model.inner_rcgp.weighting_function.plateau_width:.4f}"
    )
    print(
        f"  Outer noise std: {torch.sqrt(a2rcgp_model.likelihood.noise).item():.4f}"
    )
    outer_covar = a2rcgp_model.covar_module
    if hasattr(outer_covar, "base_kernel"):
        print(f"  Outer lengthscale: {outer_covar.base_kernel.lengthscale.item():.4f}")
    print(
        f"  Outer plateau width: {a2rcgp_model.weighting_function.plateau_width:.4f}"
    )

    # RGP-PE
    print(f"\nRGP-PE:")
    print(f"  eta={RGPPE_ETA}, psi={RGPPE_PSI}, beta={RGPPE_BETA}")
    print(f"  lambda={RGPPE_LAMBDA}, b={RGPPE_B}")
    print(f"  Grid size: {RGPPE_N_GRID}, Steps/epoch: {RGPPE_STEPS_PER_EPOCH}")

    print(f"\nAll artifacts saved to: {artifacts_dir}/")


if __name__ == "__main__":
    main()
