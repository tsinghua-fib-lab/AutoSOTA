"""Test the new clean API with CorruptedEvaluator wrapper."""

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
    aggregate_results_across_seeds,
    save_individual_seed_results,
    save_multiseed_summary,
    plot_regret_comparison_multiseed,
)
from bo_framework.models.factory import (
    create_rcgp_model,
    create_gp_model,
    create_student_t_model,
    create_a2rcgp_model,
    create_diagnostic_gp_model,
)

# Suppress standardization warnings from BoTorch/GPyTorch
import warnings

warnings.filterwarnings("ignore", message=".*outcome_transform.*")
warnings.filterwarnings("ignore", message=".*standardized.*")
warnings.filterwarnings("ignore", message=".*InputDataWarning.*")

# Experiment parameters
N_ITERATIONS = 100
N_INITIAL = 5
N_SEEDS = 10
HIGH_CORRUPTION_VALUE = 20.0
LOW_CORRUPTION_VALUE = -20.0
ADVERSARIAL_BUDGET = 2  # Used for original adversarial corruptor
STANDARDIZE = True
FIT_STANDARD_GP = True
CUSTOM_GP_MODEL = False
NOISE_STD = 1.0

# Corruption configuration
# Choose: 'time_budget', 'periodic', 'original'
CORRUPTION_TYPE = "time_budget"  # Default to original for backward compatibility

# Time budget parameters
TIME_BUDGET_ALPHA = 1 / 3  # T^alpha budget (0.5 = sqrt(T))

# Periodic parameters
PERIODIC_INTERVAL =  10 # Corrupt every Nth observations

# Corruption strategy
# Choose: 'adversarial', 'random', 'constant'
CORRUPTION_STRATEGY = "adversarial"

# Beta scheduler configuration
# For RCGP: Choose from 'constant', 'theory', 'rcgp-constant', 'rcgp-theory'
RCGP_SCHEDULER_TYPE = "theory"  # 'theory' works better than 'rcgp-theory' the sqrt{t_c} term is too pessimistic

# For GP/Student-t: Choose from 'constant', 'theory'
GP_SCHEDULER_TYPE = "theory"  # Standard constant beta

# For A2RCGP: Choose from 'constant', 'theory', 'rcgp-constant', 'rcgp-theory'
A2RCGP_SCHEDULER_TYPE = "theory"  # Standard theory-guided scheduler

# Beta scheduling parameters
CONSTANT_BETA = 2.0
RCGP_SCALE = 1.0  # Scale factor for RCGP adaptive term

# Theory scheduler parameters
THEORY_SCALE = 1.7  # Scale for theory-guided beta schedule
THEORY_OFFSET = 2  # Offset to handle early iterations

# Student-specific scheduler parameters (original working values)
STUDENT_THEORY_SCALE = 1.0  # Lower scale for student-t models
STUDENT_THEORY_OFFSET = 1  # Lower offset for student-t models
STUDENT_MIN_BETA = 0.1  # Lower minimum beta for student-t models


rcgp_kwargs = {
    "param_handling_dict": {
        "plateau_width": {"method": "manual", "value": 0.75},
        "c": {"method": "empirical_std"},
        "sigma": {"method": "fit"},
        "mean": {"method": "fit"},
    },
    "fitting_objective_type": "wloo-cv",  # options 'mll', 'loo-cv' or 'wloo-cv'
    "optimizer_type": "lbfgs",
    # "optimizer_kwargs": {"learning_rate": 0.001, "max_iter": 500},
    "standardize": STANDARDIZE,
    "verbose": False,
}

# A2RCGP configuration with inner and outer model parameters
a2rcgp_kwargs = {
    "inner_param_handling_dict": {
        "plateau_width": {"method": "manual", "value": 1.5},
        "c": {"method": "empirical_std"},
        "sigma": {"method": "fit"},
        "mean": {"method": "fit"},  # Inner RCGP uses fitted constant mean
    },
    "outer_param_handling_dict": {
        "plateau_width": {"method": "manual", "value": 1.5},
        "c": {"method": "empirical_std"},
        "sigma": {"method": "fit"},
        "mean": {"method": "fit"},  # Outer RCGP also uses fitted mean
    },
    "fitting_objective_type": "wloo-cv",
    "optimizer_type": "lbfgs",
    "standardize": STANDARDIZE,
    "verbose": False,  # Set to True for detailed fitting output
}

# Diagnostic GP (OD-BO) configuration
diagnostic_kwargs = {
    "n_init": 5,  # Start diagnosis after 8 points
    "n_schedule": 1,  # Run diagnosis every 2 iterations
    "nu": 4.0,  # Student-t degrees of freedom
    "alpha": 0.05,  # Outlier threshold
    "fitting_kwargs": {
        # "learning_rate": 0.01,  # Reduced for better convergence
        "num_iterations": 200,  # Increased for better optimization
        "verbose": False,
    },
    "model_kwargs": {
        "standardize": STANDARDIZE,
        "fit_hyperparameters": FIT_STANDARD_GP,
        "use_botorch_model": not CUSTOM_GP_MODEL,
    },
}


def create_scheduler(scheduler_type, model_type="gp"):
    """Create beta scheduler based on configuration.

    Args:
        scheduler_type: Type of scheduler ('constant', 'theory', 'rcgp-constant', 'rcgp-theory')
        model_type: Model type ('rcgp', 'gp', 'student', 'a2rcgp', 'diagnostic')

    Returns:
        BetaScheduler instance
    """
    if model_type == "rcgp":
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
            raise ValueError(f"Unknown RCGP scheduler type: {scheduler_type}")
    elif model_type == "a2rcgp":
        # A2RCGP can use any scheduler type like RCGP
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
            raise ValueError(f"Unknown A2RCGP scheduler type: {scheduler_type}")
    else:  # GP, Student-t, or Diagnostic
        if scheduler_type == "constant":
            return ConstantBetaScheduler(beta=CONSTANT_BETA)
        elif scheduler_type == "theory":
            if model_type == "student":
                # Student-t models use different parameters (original working values)
                return TheoryGuidedScheduler(
                    scale=STUDENT_THEORY_SCALE,
                    offset=STUDENT_THEORY_OFFSET,
                    min_beta=STUDENT_MIN_BETA,
                )
            elif model_type == "diagnostic":
                # Diagnostic models use standard GP scheduler since they use GP for acquisition
                return TheoryGuidedScheduler(
                    scale=THEORY_SCALE, offset=THEORY_OFFSET, min_beta=1.0
                )
            else:
                # Standard GP uses current parameters
                return TheoryGuidedScheduler(
                    scale=THEORY_SCALE, offset=THEORY_OFFSET, min_beta=1.0
                )
        else:
            raise ValueError(f"Unknown GP/Student scheduler type: {scheduler_type}")


def create_timestamped_folder(base_dir="artifacts"):
    """Create a timestamped folder for experiment results.

    Args:
        base_dir: Base directory to create the timestamped folder in

    Returns:
        Path to the created timestamped folder
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"forrester_experiment_{timestamp}"
    folder_path = os.path.join(base_dir, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path


def save_experiment_config(config_dict, folder_path):
    """Save experiment configuration to JSON file.

    Args:
        config_dict: Dictionary containing all experiment parameters
        folder_path: Path to the experiment folder
    """
    config_path = os.path.join(folder_path, "experiment_config.json")
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2, default=str)
    print(f"Experiment configuration saved to: {config_path}")


def create_corruptor_factory(corruption_type: str, strategy_type: str):
    """Create a factory function for the specified corruptor configuration.

    Args:
        corruption_type: Type of corruption decider ('time_budget', 'periodic', 'original')
        strategy_type: Type of corruption strategy ('adversarial', 'random', 'constant')

    Returns:
        Factory function that creates configured corruptors
    """

    def factory(optimal_point, budget, high_value, low_value):
        # Create decider based on type
        if corruption_type == "time_budget":
            decider = TimeBudgetDecider(
                alpha=TIME_BUDGET_ALPHA, skip_initial=True, n_initial=N_INITIAL
            )
            # Note: budget parameter is ignored for time-based decider
        elif corruption_type == "periodic":
            decider = PeriodicDecider(
                period=PERIODIC_INTERVAL, skip_initial=True, n_initial=N_INITIAL
            )
        elif corruption_type == "original":
            # Return original adversarial corruptor for backward compatibility
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

        # Create strategy based on type
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

        # Return composable corruptor
        return ComposableCorruptor(
            decider=decider, strategy=strategy, skip_initial=True
        )

    return factory


def main():
    """Compare RCGP against standard GP for Forrester function with adversarial corruption."""

    # Create timestamped folder for results
    artifacts_dir = create_timestamped_folder()
    print(f"Created experiment folder: {artifacts_dir}")

    # Create configuration dictionary
    config_dict = {
        "experiment_info": {
            "name": "Forrester Adversarial Experiment",
            "timestamp": datetime.now().isoformat(),
            "script": "forrester_adversarial.py",
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
            "CUSTOM_GP_MODEL": CUSTOM_GP_MODEL,
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
            "STUDENT_THEORY_SCALE": STUDENT_THEORY_SCALE,
            "STUDENT_THEORY_OFFSET": STUDENT_THEORY_OFFSET,
            "STUDENT_MIN_BETA": STUDENT_MIN_BETA,
        },
        "model_configs": {
            "rcgp_kwargs": rcgp_kwargs,
            "a2rcgp_kwargs": a2rcgp_kwargs,
            "diagnostic_kwargs": diagnostic_kwargs,
        },
    }

    # Save configuration to JSON file
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
    print(f"GP/Student-t scheduler: {GP_SCHEDULER_TYPE}")
    print(f"A2RCGP scheduler: {A2RCGP_SCHEDULER_TYPE}")
    print(f"Constant beta: {CONSTANT_BETA}, RCGP scale: {RCGP_SCALE}")
    print("=" * 80 + "\n")

    # Forrester optimal point and value (in normalized space)
    forrester_func = ForresterFunction()
    optimal_point = torch.tensor([1.0], dtype=torch.double)
    optimal_value = forrester_func.optimal_value

    # Create search space
    search_space = SearchSpace(
        (Dimension(name="x0", type="continuous", bounds=(0.0, 1.0), normalize=True),)
    )

    # Create evaluator
    clean_evaluator = SyntheticEvaluator(forrester_func)

    # Create corruptor factory based on configuration
    corruptor_factory = create_corruptor_factory(CORRUPTION_TYPE, CORRUPTION_STRATEGY)

    # Run RCGP experiments across all seeds
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

    # Run GP experiments across all seeds
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

    # Run Student-t experiments across all seeds
    stp_model_kwargs = {
        "nu": 3.0,  # Degrees of freedom (lower = heavier tails)
        "standardize": STANDARDIZE,
        "fit_hyperparameters": FIT_STANDARD_GP,
        "optimizer_type": "lbfgs",
    }
    stp_scheduler = create_scheduler(GP_SCHEDULER_TYPE, "student")
    print(f"Student-t using scheduler: {stp_scheduler.__class__.__name__}")
    stp_all_results = run_model_across_seeds(
        "Student-t",
        create_student_t_model,
        stp_model_kwargs,
        stp_scheduler,
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

    # Run A2RCGP experiments across all seeds
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

    # Run Diagnostic GP experiments across all seeds
    diagnostic_scheduler = create_scheduler(GP_SCHEDULER_TYPE, "diagnostic")
    print(f"Diagnostic GP using scheduler: {diagnostic_scheduler.__class__.__name__}")
    diagnostic_all_results = run_model_across_seeds(
        "DiagnosticGP",
        create_diagnostic_gp_model,
        diagnostic_kwargs,
        diagnostic_scheduler,
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
    )

    # Aggregate results across seeds
    rcgp_results = aggregate_results_across_seeds(rcgp_all_results)
    gp_results = aggregate_results_across_seeds(gp_all_results)
    stp_results = aggregate_results_across_seeds(stp_all_results)
    a2rcgp_results = aggregate_results_across_seeds(a2rcgp_all_results)
    diagnostic_results = aggregate_results_across_seeds(diagnostic_all_results)

    # Compare the results using regret_analysis utilities
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS (AGGREGATED ACROSS SEEDS)")
    print("=" * 80)

    scenarios = [
        ("RCGP", rcgp_results),
        ("GP", gp_results),
        ("Student-t", stp_results),
        ("A2RCGP", a2rcgp_results),
        ("DiagnosticGP", diagnostic_results),
    ]

    # Create results dictionary for comparison
    results_dict = {name: results for name, results in scenarios}

    # Use multiseed comparison for proper aggregation across all seeds
    multiseed_metrics_dict = compare_experiments_multiseed(
        results_dict={name: results["all_results"] for name, results in scenarios},
        optimal_value=optimal_value,
    )

    # Print detailed comparison using the multiseed comparison utility
    print_comparison_table_multiseed(
        multiseed_metrics_dict,
        show_regret=True,
        show_corruption=True,
        show_std=True,  # Show standard deviations
    )

    # Also print single-seed comparison (first seed only) for reference
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS (FIRST SEED ONLY - FOR REFERENCE)")
    print("=" * 80)

    metrics_dict = compare_experiments(
        results_dict={name: results["all_results_flat"] for name, results in scenarios},
        optimal_value=optimal_value,
    )

    print_comparison_table(metrics_dict, show_regret=True, show_corruption=True)

    # Save results and create plots
    print("\n" + "=" * 80)
    print("SAVING RESULTS AND GENERATING PLOTS")
    print("=" * 80)

    # Save multi-seed results properly
    # Create a dictionary with the raw results from all seeds for pickle/summary.json
    multiseed_results_dict = {
        f"{model_name}_seed_{seed}": all_results[seed]
        for model_name, all_results in [
            ("RCGP", rcgp_all_results),
            ("GP", gp_all_results),
            ("Student-t", stp_all_results),
            ("A2RCGP", a2rcgp_all_results),
            ("DiagnosticGP", diagnostic_all_results),
        ]
        for seed in range(len(all_results))
    }

    # Save all individual seed results (pickle + combined JSON)
    save_experiment_results(
        results=multiseed_results_dict,
        experiment_name="forrester_experiment",
        artifacts_dir=artifacts_dir,
        save_pickle=True,
        save_json=True,
        optimal_value=optimal_value,
        verbose=True,
    )

    # Save individual JSON files for each seed
    print("Saving individual seed JSON files...")
    for model_name, all_results in [
        ("RCGP", rcgp_all_results),
        ("GP", gp_all_results),
        ("Student-t", stp_all_results),
        ("A2RCGP", a2rcgp_all_results),
        ("DiagnosticGP", diagnostic_all_results),
    ]:
        for seed, results in enumerate(all_results):
            seed_path = save_individual_seed_results(
                model_name=model_name,
                seed=seed,
                results=results,
                optimal_value=optimal_value,
                artifacts_dir=artifacts_dir,
            )
            if seed == 0:  # Only print first seed path for each model to avoid clutter
                print(f"  {model_name} individual seeds saved (e.g., {seed_path})")

    # Save comparison table using flattened results for proper analysis
    save_comparison_table(
        results_dict={name: results["all_results_flat"] for name, results in scenarios},
        experiment_name="forrester_experiment",
        artifacts_dir=artifacts_dir,
        optimal_value=optimal_value,
    )

    # Save aggregated multi-seed statistics
    save_multiseed_summary(
        results_dict=results_dict,
        optimal_value=optimal_value,
        artifacts_dir=artifacts_dir,
    )

    # Create plots for each scenario (using first seed only for summary plots)
    config = PlotConfig(figsize=(15, 10))

    # Only create detailed model summary plots for first seed
    first_seed_scenarios = [
        ("RCGP", rcgp_all_results[0]),
        ("GP", gp_all_results[0]),
        ("Student-t", stp_all_results[0]),
        ("A2RCGP", a2rcgp_all_results[0]),
        ("DiagnosticGP", diagnostic_all_results[0]),
    ]

    for name, results in first_seed_scenarios:
        print(f"Creating summary plot for: {name} (seed 0)")

        # Create the summary plot using first seed only
        plot_filename = f"forrester_{name.lower().replace(' ', '_')}_seed0.png"
        plot_path = os.path.join(artifacts_dir, plot_filename)

        fig = plot_experiment_summary(
            results=results,
            objective_func=lambda x: forrester_func.evaluate(x),
            optimal_value=optimal_value,
            save_path=plot_path,
            config=config,
        )
        plt.close(fig)  # Close to free memory

    print(f"\nAll artifacts saved to: {artifacts_dir}/")

    # Create regret comparison plots
    print("\n" + "=" * 80)
    print("CREATING REGRET COMPARISON PLOTS")
    print("=" * 80)

    regret_save_path = os.path.join(artifacts_dir, "regret")
    colors = {
        "RCGP": "blue",
        "GP": "orange",
        "Student-t": "green",
        "A2RCGP": "red",
        "DiagnosticGP": "purple",
    }

    regret_fig, simple_regret_fig = plot_regret_comparison_multiseed(
        results_dict=results_dict,
        optimal_value=optimal_value,
        n_seeds=N_SEEDS,
        save_path=regret_save_path,
        config=config,
        colors=colors,
    )

    # Close figures to free memory
    plt.close(regret_fig)
    plt.close(simple_regret_fig)

    # Print model hyperparameters (from first seed)
    print("\n" + "=" * 80)
    print("MODEL HYPERPARAMETERS (From First Seed)")
    print("=" * 80)

    # RCGP model parameters
    rcgp_model = rcgp_results["final_model"]
    print(
        f"\nRCGP Model (averaged across {N_SEEDS} seeds in regret plots, showing seed 0 hyperparameters):"
    )
    print(f"  Noise std (sigma): {torch.sqrt(rcgp_model.likelihood.noise).item():.4f}")

    # Handle different covariance module structures
    covar = rcgp_model.covar_module
    if hasattr(covar, "base_kernel"):
        # ScaleKernel case
        print(f"  Lengthscale: {covar.base_kernel.lengthscale.item():.4f}")
        print(f"  Output scale: {covar.outputscale.item():.4f}")
    else:
        # Direct kernel case
        if hasattr(covar, "lengthscale"):
            print(f"  Lengthscale: {covar.lengthscale.item():.4f}")
        if hasattr(covar, "outputscale"):
            print(f"  Output scale: {covar.outputscale.item():.4f}")

    if hasattr(rcgp_model.mean_module, "constant"):
        print(f"  Mean constant: {rcgp_model.mean_module.constant.item():.4f}")
    print(f"  Plateau width: {rcgp_model.weighting_function.plateau_width:.4f}")
    print(f"  C parameter: {rcgp_model.weighting_function.c:.4f}")

    # GP model parameters
    gp_model = gp_results["final_model"]
    print("\nGP Model (seed 0 hyperparameters):")
    print(f"  Noise std (sigma): {torch.sqrt(gp_model.likelihood.noise).item():.4f}")

    # Handle different covariance module structures
    covar = gp_model.covar_module
    if hasattr(covar, "base_kernel"):
        # ScaleKernel case
        print(f"  Lengthscale: {covar.base_kernel.lengthscale.item():.4f}")
        print(f"  Output scale: {covar.outputscale.item():.4f}")
    else:
        # Direct kernel case
        if hasattr(covar, "lengthscale"):
            print(f"  Lengthscale: {covar.lengthscale.item():.4f}")
        if hasattr(covar, "outputscale"):
            print(f"  Output scale: {covar.outputscale.item():.4f}")

    if hasattr(gp_model.mean_module, "constant"):
        print(f"  Mean constant: {gp_model.mean_module.constant.item():.4f}")

    # Student-t Process model parameters
    stp_model = stp_results["final_model"]
    print("\nStudent-t Process Model (seed 0 hyperparameters):")
    print(f"  Degrees of freedom (nu): {stp_model.nu.item():.2f}")
    print(f"  Noise std (sigma): {torch.sqrt(stp_model.likelihood.noise).item():.4f}")

    # Handle different covariance module structures
    covar = stp_model.covar_module
    if hasattr(covar, "base_kernel"):
        # ScaleKernel case
        print(f"  Lengthscale: {covar.base_kernel.lengthscale.item():.4f}")
        print(f"  Output scale: {covar.outputscale.item():.4f}")
    else:
        # Direct kernel case
        if hasattr(covar, "lengthscale"):
            print(f"  Lengthscale: {covar.lengthscale.item():.4f}")
        if hasattr(covar, "outputscale"):
            print(f"  Output scale: {covar.outputscale.item():.4f}")

    if hasattr(stp_model.mean_module, "constant"):
        print(f"  Mean constant: {stp_model.mean_module.constant.item():.4f}")

    # A2RCGP model parameters
    a2rcgp_model = a2rcgp_results["final_model"]
    print("\nA2RCGP Model (seed 0 hyperparameters):")

    # Inner RCGP parameters
    print("  Inner RCGP:")
    print(
        f"    Noise std (sigma): {torch.sqrt(a2rcgp_model.inner_rcgp.likelihood.noise).item():.4f}"
    )

    inner_covar = a2rcgp_model.inner_rcgp.covar_module
    if hasattr(inner_covar, "base_kernel"):
        print(f"    Lengthscale: {inner_covar.base_kernel.lengthscale.item():.4f}")
        print(f"    Output scale: {inner_covar.outputscale.item():.4f}")
    else:
        if hasattr(inner_covar, "lengthscale"):
            print(f"    Lengthscale: {inner_covar.lengthscale.item():.4f}")
        if hasattr(inner_covar, "outputscale"):
            print(f"    Output scale: {inner_covar.outputscale.item():.4f}")

    if hasattr(a2rcgp_model.inner_rcgp.mean_module, "constant"):
        print(
            f"    Mean constant: {a2rcgp_model.inner_rcgp.mean_module.constant.item():.4f}"
        )
    print(
        f"    Plateau width: {a2rcgp_model.inner_rcgp.weighting_function.plateau_width:.4f}"
    )
    print(f"    C parameter: {a2rcgp_model.inner_rcgp.weighting_function.c:.4f}")

    # Outer RCGP parameters
    print("  Outer RCGP:")
    print(
        f"    Noise std (sigma): {torch.sqrt(a2rcgp_model.likelihood.noise).item():.4f}"
    )

    outer_covar = a2rcgp_model.covar_module
    if hasattr(outer_covar, "base_kernel"):
        print(f"    Lengthscale: {outer_covar.base_kernel.lengthscale.item():.4f}")
        print(f"    Output scale: {outer_covar.outputscale.item():.4f}")
    else:
        if hasattr(outer_covar, "lengthscale"):
            print(f"    Lengthscale: {outer_covar.lengthscale.item():.4f}")
        if hasattr(outer_covar, "outputscale"):
            print(f"    Output scale: {outer_covar.outputscale.item():.4f}")

    if hasattr(a2rcgp_model.mean_module, "constant"):
        print(f"    Mean constant: {a2rcgp_model.mean_module.constant.item():.4f}")
    print(f"    Plateau width: {a2rcgp_model.weighting_function.plateau_width:.4f}")
    print(f"    C parameter: {a2rcgp_model.weighting_function.c:.4f}")

    # Corruption detection
    print("  Corruption Detection:")
    corruption_results = a2rcgp_model.detect_corruptions()
    inner_corruptions = corruption_results["inner"].sum().item()
    outer_corruptions = corruption_results["outer"].sum().item()
    print(f"    Inner RCGP detected corruptions: {inner_corruptions}")
    print(f"    Outer RCGP detected corruptions: {outer_corruptions}")

    # Diagnostic GP model info
    diagnostic_model = diagnostic_results["final_model"]
    print("\nDiagnostic GP Model (seed 0):")
    diagnostic_info = diagnostic_model.get_diagnostic_info()
    print(f"  Total points: {diagnostic_info['total_points']}")
    print(f"  Inliers: {diagnostic_info['num_inliers']}")
    print(f"  Outliers detected: {diagnostic_info['num_outliers']}")
    if diagnostic_info["outlier_indices"]:
        print(f"  Outlier indices: {diagnostic_info['outlier_indices']}")

    # Show underlying model hyperparameters
    underlying_model = diagnostic_model.model
    print("  Underlying Model:")
    print(
        f"    Noise std (sigma): {torch.sqrt(underlying_model.likelihood.noise).item():.4f}"
    )

    covar = underlying_model.covar_module
    if hasattr(covar, "base_kernel"):
        print(f"    Lengthscale: {covar.base_kernel.lengthscale.item():.4f}")
        print(f"    Output scale: {covar.outputscale.item():.4f}")
    else:
        if hasattr(covar, "lengthscale"):
            print(f"    Lengthscale: {covar.lengthscale.item():.4f}")
        if hasattr(covar, "outputscale"):
            print(f"    Output scale: {covar.outputscale.item():.4f}")

    if hasattr(underlying_model.mean_module, "constant"):
        print(f"    Mean constant: {underlying_model.mean_module.constant.item():.4f}")


if __name__ == "__main__":
    main()
