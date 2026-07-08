"""Test the new clean API with HPT CIFAR mixed variable optimization."""

import torch
import os
import json
import matplotlib.pyplot as plt
from datetime import datetime
from bo_framework import SearchSpace, Dimension, ExperimentRunner
from bo_framework.base.acquisition import UCBAcquisition
from bo_framework.base.schedulers import ConstantBetaScheduler, TheoryGuidedScheduler, RCGPScheduler
from bo_framework.models.factory import create_mixed_gp_model, create_mixed_rcgp_model, create_diagnostic_gp_model, create_student_t_model, create_a2rcgp_model
from bo_framework.corruption.composable import (
    ComposableCorruptor,
    PeriodicDecider,
    TimeBudgetDecider,
    ConstantStrategy,
    CorruptionDecider
)
from bo_framework.wrappers.corrupted import CorruptedEvaluator
from experiments.hpt_cifar.evaluator import HPTCIFAREvaluator
from utilities.io import save_experiment_results, save_comparison_table


# Experiment parameters
N_ITERATIONS = 150
N_INITIAL = 2
SEED = 42
STANDARDIZE = True
FIT_STANDARD_GP = True
MAX_EPOCHS = 2

# Corruption configuration
# Choose: 'periodic', 'time_budget', 'budget', 'none'
CORRUPTION_TYPE = 'time_budget'  # Test budget-based corruption (original Forrester method)

# Periodic parameters
PERIODIC_INTERVAL = 5  # Corrupt every Nth observation (simulate training crash)

# Time budget parameters
TIME_BUDGET_ALPHA = 1/3  # T^alpha budget (0.5 = sqrt(T))

# Budget-based parameters (original Forrester method)
CORRUPTION_BUDGET = 7  # Total number of corruptions allowed

# Training crash parameters
TRAINING_CRASH_VALUE = -5.0  # Value returned when training crashes

# Beta scheduler configuration
# For RCGP: Choose from 'constant', 'theory', 'rcgp-constant', 'rcgp-theory'
RCGP_SCHEDULER_TYPE = 'theory'

# For GP: Choose from 'constant', 'theory'
GP_SCHEDULER_TYPE = 'theory'

# For A2RCGP: Choose from 'constant', 'theory', 'rcgp-constant', 'rcgp-theory'
A2RCGP_SCHEDULER_TYPE = 'theory'

# Beta scheduling parameters
CONSTANT_BETA = 2.0
RCGP_SCALE = 1.0  # Scale factor for RCGP adaptive term

# Theory scheduler parameters
THEORY_SCALE = 1.7  # Scale for theory-guided beta schedule
THEORY_OFFSET = 2   # Offset to handle early iterations

# RCGP configuration following Forrester pattern
rcgp_kwargs = {
    "param_handling_dict": {
        "plateau_width": {"method": "heuristics", "value": 2.0},
        "c": {"method": "manual", "value": 1.0},
        "sigma": {"method": "fit"},
        "mean": {"method": "fit"}
    },
    "fitting_objective_type": "wloo-cv",  # options 'mll', 'loo-cv' or 'wloo-cv'
    "optimizer_type": "lbfgs",
    "verbose": False
}

# Student-t Process configuration
student_t_kwargs = {
    'nu': 3.0,  # Degrees of freedom (lower = heavier tails)
    'standardize': STANDARDIZE,
    'fit_hyperparameters': FIT_STANDARD_GP,
    'optimizer_type': 'lbfgs'
}

# Diagnostic GP (OD-BO) configuration
diagnostic_kwargs = {
    "n_init": 3,  # Start diagnosis after 3 points
    "n_schedule": 1,  # Run diagnosis every iteration
    "nu": 4.0,  # Student-t degrees of freedom
    "alpha": 0.05,  # Outlier threshold
    "fitting_kwargs": {
        "num_iterations": 200,  # Increased for better optimization
        "verbose": False
    },
    "model_kwargs": {
        "standardize": STANDARDIZE,
        "fit_hyperparameters": FIT_STANDARD_GP,
        "use_botorch_model": True
    }
}

# A2RCGP configuration with inner and outer model parameters
a2rcgp_kwargs = {
    "inner_param_handling_dict": {
        "plateau_width": {"method": "heuristics", "value": 2.0},
        "c": {"method": "manual", "value": 1.0},
        "sigma": {"method": "fit"},
        "mean": {"method": "fit"}  # Inner RCGP uses fitted constant mean
    },
    "outer_param_handling_dict": {
        "plateau_width": {"method": "heuristics", "value": 1.5},
        "c": {"method": "manual", "value": 0.8},
        "sigma": {"method": "fit"},
        "mean": {"method": "fit"}  # Outer RCGP also uses fitted mean
    },
    "fitting_objective_type": "wloo-cv",
    "optimizer_type": "lbfgs",
    "standardize": STANDARDIZE,
    "verbose": False  # Set to True for detailed fitting output
}


def create_scheduler(scheduler_type, model_type='gp'):
    """Create beta scheduler based on configuration.

    Args:
        scheduler_type: Type of scheduler ('constant', 'theory', 'rcgp-constant', 'rcgp-theory')
        model_type: Model type ('rcgp', 'gp', 'a2rcgp', 'student', 'diagnostic')

    Returns:
        BetaScheduler instance
    """
    if model_type == 'rcgp' or model_type == 'a2rcgp':
        if scheduler_type == 'constant':
            return ConstantBetaScheduler(beta=CONSTANT_BETA)
        elif scheduler_type == 'theory':
            return TheoryGuidedScheduler(
                scale=THEORY_SCALE, 
                offset=THEORY_OFFSET, 
                min_beta=1.0
            )
        elif scheduler_type == 'rcgp-constant':
            return RCGPScheduler(
                scale=RCGP_SCALE,
                base_scheduler=ConstantBetaScheduler(beta=CONSTANT_BETA)
            )
        elif scheduler_type == 'rcgp-theory':
            return RCGPScheduler(
                scale=RCGP_SCALE,
                base_scheduler=TheoryGuidedScheduler(
                    scale=THEORY_SCALE, 
                    offset=THEORY_OFFSET, 
                    min_beta=1.0
                )
            )
        else:
            raise ValueError(f"Unknown RCGP scheduler type: {scheduler_type}")
    else:  # GP
        if scheduler_type == 'constant':
            return ConstantBetaScheduler(beta=CONSTANT_BETA)
        elif scheduler_type == 'theory':
            return TheoryGuidedScheduler(
                scale=THEORY_SCALE, 
                offset=THEORY_OFFSET, 
                min_beta=1.0
            )
        else:
            raise ValueError(f"Unknown GP scheduler type: {scheduler_type}")


def create_training_crash_corruptor(corruption_type: str = 'periodic'):
    """Create a training crash corruptor that simulates training failures.
    
    Args:
        corruption_type: Type of corruption decider ('periodic', 'time_budget', 'budget', 'none')
        
    Returns:
        ComposableCorruptor or AdversarialCorruptor instance or None if corruption_type is 'none'
    """
    if corruption_type == 'none':
        return None
    
    # Handle budget-based corruption (original Forrester method)
    if corruption_type == 'budget':
        # Create a simple count-based decider that corrupts up to the budget
        class CountBudgetDecider(CorruptionDecider):
            def __init__(self, budget, skip_initial=True):
                self.budget = budget
                self.corruptions_used = 0
                self.skip_initial = skip_initial
                
            def should_corrupt(self, iteration, total_iterations, is_initial, history):
                if is_initial and self.skip_initial:
                    return False
                return self.corruptions_used < self.budget
                
            def reset(self):
                self.corruptions_used = 0
                
            def update_corruption(self):
                self.corruptions_used += 1
                
            @property
            def info(self):
                return f"Budget: {self.corruptions_used}/{self.budget}"
        
        decider = CountBudgetDecider(
            budget=CORRUPTION_BUDGET,
            skip_initial=True
        )
        strategy = ConstantStrategy(corruption_value=TRAINING_CRASH_VALUE)
        return ComposableCorruptor(
            decider=decider,
            strategy=strategy,
            skip_initial=True
        )
    
    # Create decider based on type
    if corruption_type == 'periodic':
        decider = PeriodicDecider(
            period=PERIODIC_INTERVAL, 
            skip_initial=True, 
            n_initial=N_INITIAL
        )
    elif corruption_type == 'time_budget':
        decider = TimeBudgetDecider(
            alpha=TIME_BUDGET_ALPHA, 
            skip_initial=True, 
            n_initial=N_INITIAL
        )
    else:
        raise ValueError(f"Unknown corruption type: {corruption_type}")
    
    # Create constant strategy that returns the crash value
    strategy = ConstantStrategy(corruption_value=TRAINING_CRASH_VALUE)
    
    # Return composable corruptor
    return ComposableCorruptor(
        decider=decider, 
        strategy=strategy, 
        skip_initial=True
    )


def create_fresh_evaluator():
    """Create a fresh evaluator with a new corruptor for each experiment run.

    This ensures each experiment starts with clean state and is not affected
    by previous experiments.

    Returns:
        Fresh evaluator instance (CorruptedEvaluator or base evaluator)
    """

    # Create fresh base evaluator
    base_evaluator = HPTCIFAREvaluator(
        max_epochs=MAX_EPOCHS,
        device="cuda" if torch.cuda.is_available() else "cpu",
        penalty_value=0.0
    )

    # Create fresh corruptor
    corruptor = create_training_crash_corruptor(CORRUPTION_TYPE)

    # Wrap in CorruptedEvaluator if corruption is enabled
    if corruptor is not None:
        return CorruptedEvaluator(
            base_evaluator=base_evaluator,
            corruptor=corruptor,
            n_initial=N_INITIAL
        )
    else:
        return base_evaluator


def create_timestamped_folder(base_dir="artifacts"):
    """Create a timestamped folder for experiment results.
    
    Args:
        base_dir: Base directory to create the timestamped folder in
        
    Returns:
        Path to the created timestamped folder
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"hpt_cifar_experiment_{timestamp}"
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
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)
    print(f"Experiment configuration saved to: {config_path}")


def calculate_cumulative_regret(results, optimal_value):
    """Calculate cumulative regret for a set of results.

    Cumulative regret is the sum of instantaneous regrets:
    Σ(optimal_value - observed_value_i) for i = 1 to t

    Args:
        results: List of EvaluationResult objects
        optimal_value: The optimal value to compare against

    Returns:
        List of cumulative regret values
    """
    if not results:
        return []

    # Get all observed values
    observed_values = [r.y_observed for r in results]

    # Calculate instantaneous regrets
    instantaneous_regrets = [optimal_value - value for value in observed_values]

    # Calculate cumulative sum of regrets
    cumulative_regret = []
    cumulative_sum = 0
    for regret in instantaneous_regrets:
        cumulative_sum += regret
        cumulative_regret.append(cumulative_sum)

    return cumulative_regret


def calculate_simple_regret(results, optimal_value):
    """Calculate simple regret for a set of results.

    Simple regret is the regret of the best point found so far:
    optimal_value - max(observed_values[1:t])

    Args:
        results: List of EvaluationResult objects
        optimal_value: The optimal value to compare against

    Returns:
        List of simple regret values
    """
    if not results:
        return []

    # Get all observed values
    observed_values = [r.y_observed for r in results]

    # Calculate cumulative maximum
    cumulative_max = []
    current_max = observed_values[0]
    for value in observed_values:
        current_max = max(current_max, value)
        cumulative_max.append(current_max)

    # Calculate simple regret (optimal - cumulative_max)
    simple_regret = [optimal_value - max_val for max_val in cumulative_max]
    return simple_regret


def plot_cumulative_regret_comparison(results_dict, optimal_value, save_path, title="Cumulative Regret Comparison"):
    """Plot cumulative regret comparison for multiple models.

    Args:
        results_dict: Dictionary mapping model names to their results
        optimal_value: The optimal value to compare against
        save_path: Path to save the plot
        title: Plot title
    """
    plt.figure(figsize=(12, 8))

    colors = ['blue', 'orange', 'green', 'red', 'purple']
    for i, (model_name, results) in enumerate(results_dict.items()):
        cumulative_regret = calculate_cumulative_regret(results, optimal_value)
        iterations = list(range(1, len(cumulative_regret) + 1))

        plt.plot(iterations, cumulative_regret,
                label=model_name,
                color=colors[i % len(colors)],
                linewidth=2,
                marker='o',
                markersize=4)

    plt.xlabel('Iteration')
    plt.ylabel('Cumulative Regret')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Cumulative regret plot saved to: {save_path}")


def plot_simple_regret_comparison(results_dict, optimal_value, save_path, title="Simple Regret Comparison"):
    """Plot simple regret comparison for multiple models.

    Args:
        results_dict: Dictionary mapping model names to their results
        optimal_value: The optimal value to compare against
        save_path: Path to save the plot
        title: Plot title
    """
    plt.figure(figsize=(12, 8))

    colors = ['blue', 'orange', 'green', 'red', 'purple']
    for i, (model_name, results) in enumerate(results_dict.items()):
        simple_regret = calculate_simple_regret(results, optimal_value)
        iterations = list(range(1, len(simple_regret) + 1))

        plt.plot(iterations, simple_regret,
                label=model_name,
                color=colors[i % len(colors)],
                linewidth=2,
                marker='o',
                markersize=4)

    plt.xlabel('Iteration')
    plt.ylabel('Simple Regret')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Simple regret plot saved to: {save_path}")


def save_chosen_points(results_dict, save_path):
    """Save chosen points and observed values for all models.
    
    Args:
        results_dict: Dictionary mapping model names to their results
        save_path: Path to save the data
    """
    chosen_points_data = {}
    
    for model_name, results in results_dict.items():
        points_data = []
        for i, result in enumerate(results):
            point_data = {
                'iteration': i + 1,
                'parameters': result.x,
                'observed_value': result.y_observed,
                'true_value': result.y_true,
                'is_corrupted': result.y_observed != result.y_true
            }
            points_data.append(point_data)
        
        chosen_points_data[model_name] = points_data
    
    # Save as JSON
    json_path = os.path.join(save_path, "chosen_points.json")
    with open(json_path, 'w') as f:
        json.dump(chosen_points_data, f, indent=2, default=str)
    print(f"Chosen points data saved to: {json_path}")
    
    # Also save as CSV for easy analysis
    csv_path = os.path.join(save_path, "chosen_points_summary.csv")
    with open(csv_path, 'w') as f:
        f.write("model,iteration,learning_rate,optimizer,architecture,batch_size,num_layers,observed_value,true_value,is_corrupted\n")
        
        for model_name, points_data in chosen_points_data.items():
            for point in points_data:
                params = point['parameters']
                f.write(f"{model_name},{point['iteration']},{params['learning_rate']:.6f},{params['optimizer']},{params['architecture']},{params['batch_size']},{params['num_layers']},{point['observed_value']:.6f},{point['true_value']:.6f},{point['is_corrupted']}\n")
    
    print(f"Chosen points summary saved to: {csv_path}")


def save_model_hyperparameters(results_dict, save_path):
    """Save model hyperparameters for all models.
    
    Args:
        results_dict: Dictionary mapping model names to their results
        save_path: Path to save the data
    """
    hyperparams_data = {}
    
    for model_name, results in results_dict.items():
        if hasattr(results, 'final_model'):
            model = results.final_model
            hyperparams = extract_model_hyperparameters_dict(model, model_name)
            hyperparams_data[model_name] = hyperparams
    
    # Save as JSON
    json_path = os.path.join(save_path, "model_hyperparameters.json")
    with open(json_path, 'w') as f:
        json.dump(hyperparams_data, f, indent=2, default=str)
    print(f"Model hyperparameters saved to: {json_path}")


def extract_model_hyperparameters_dict(model, model_name):
    """Extract hyperparameters as a dictionary.
    
    Args:
        model: The trained model
        model_name: Name of the model
        
    Returns:
        Dictionary of hyperparameters
    """
    hyperparams = {}
    
    try:
        # Handle mixed variable model structure
        if hasattr(model, 'model'):
            # Mixed variable wrapper
            underlying_model = model.model
            covar = underlying_model.covar_module
            mean_module = underlying_model.mean_module
        else:
            # Direct model
            covar = model.covar_module
            mean_module = model.mean_module
        
        # Noise
        hyperparams['noise_std'] = float(torch.sqrt(model.likelihood.noise).item())
        
        # Covariance parameters
        if hasattr(covar, 'base_kernel'):
            # ScaleKernel case
            if covar.base_kernel.lengthscale is not None:
                hyperparams['lengthscale'] = float(covar.base_kernel.lengthscale.item())
            if hasattr(covar, 'outputscale'):
                hyperparams['output_scale'] = float(covar.outputscale.item())
        else:
            # Direct kernel case
            if hasattr(covar, 'lengthscale') and covar.lengthscale is not None:
                hyperparams['lengthscale'] = float(covar.lengthscale.item())
            if hasattr(covar, 'outputscale'):
                hyperparams['output_scale'] = float(covar.outputscale.item())
        
        # Mean parameters
        if hasattr(mean_module, 'constant'):
            hyperparams['mean_constant'] = float(mean_module.constant.item())
        
        # RCGP-specific parameters
        if hasattr(model, 'weighting_function'):
            hyperparams['plateau_width'] = float(model.weighting_function.plateau_width)
            hyperparams['c_parameter'] = float(model.weighting_function.c)
        
        # Student-t specific parameters
        if hasattr(model, 'nu'):
            hyperparams['degrees_of_freedom'] = float(model.nu.item())
            
    except Exception as e:
        hyperparams['error'] = str(e)
    
    return hyperparams


def main():
    """Demonstrate the new clean API with CIFAR hyperparameter optimization."""
    
    # Create timestamped folder for results
    artifacts_dir = create_timestamped_folder()
    print(f"Created experiment folder: {artifacts_dir}")
    
    # Create configuration dictionary
    config_dict = {
        "experiment_info": {
            "name": "HPT CIFAR Experiment",
            "timestamp": datetime.now().isoformat(),
            "script": "test_hpt_cifar_clean_api.py"
        },
        "experiment_parameters": {
            "N_ITERATIONS": N_ITERATIONS,
            "N_INITIAL": N_INITIAL,
            "SEED": SEED,
            "STANDARDIZE": STANDARDIZE,
            "FIT_STANDARD_GP": FIT_STANDARD_GP,
            "MAX_EPOCHS": MAX_EPOCHS
        },
        "corruption_config": {
            "CORRUPTION_TYPE": CORRUPTION_TYPE,
            "PERIODIC_INTERVAL": PERIODIC_INTERVAL,
            "TIME_BUDGET_ALPHA": TIME_BUDGET_ALPHA,
            "CORRUPTION_BUDGET": CORRUPTION_BUDGET,
            "TRAINING_CRASH_VALUE": TRAINING_CRASH_VALUE
        },
        "scheduler_config": {
            "RCGP_SCHEDULER_TYPE": RCGP_SCHEDULER_TYPE,
            "GP_SCHEDULER_TYPE": GP_SCHEDULER_TYPE,
            "A2RCGP_SCHEDULER_TYPE": A2RCGP_SCHEDULER_TYPE,
            "CONSTANT_BETA": CONSTANT_BETA,
            "RCGP_SCALE": RCGP_SCALE,
            "THEORY_SCALE": THEORY_SCALE,
            "THEORY_OFFSET": THEORY_OFFSET
        },
        "model_configs": {
            "rcgp_kwargs": rcgp_kwargs,
            "student_t_kwargs": student_t_kwargs,
            "diagnostic_kwargs": diagnostic_kwargs,
            "a2rcgp_kwargs": a2rcgp_kwargs
        }
    }
    
    # Save configuration to JSON file
    save_experiment_config(config_dict, artifacts_dir)
    
    # Print configuration
    print("\n" + "=" * 80)
    print("HPT CIFAR EXPERIMENT CONFIGURATION")
    print("=" * 80)
    print(f"Iterations: {N_ITERATIONS}, Initial points: {N_INITIAL}")
    print(f"Seed: {SEED}")
    print(f"Corruption type: {CORRUPTION_TYPE}")
    if CORRUPTION_TYPE == 'periodic':
        print(f"  Periodic interval: every {PERIODIC_INTERVAL} observations")
    elif CORRUPTION_TYPE == 'time_budget':
        print(f"  Time budget alpha: {TIME_BUDGET_ALPHA} (T^{TIME_BUDGET_ALPHA})")
    elif CORRUPTION_TYPE == 'budget':
        print(f"  Corruption budget: {CORRUPTION_BUDGET}")
    print(f"Training crash value: {TRAINING_CRASH_VALUE}")
    print(f"RCGP scheduler: {RCGP_SCHEDULER_TYPE}")
    print(f"GP scheduler: {GP_SCHEDULER_TYPE}")
    print(f"A2RCGP scheduler: {A2RCGP_SCHEDULER_TYPE}")
    print(f"Constant beta: {CONSTANT_BETA}, RCGP scale: {RCGP_SCALE}")
    print("=" * 80 + "\n")
    
    # Create mixed variable search space matching existing HPT experiment
    search_space = SearchSpace((
        Dimension(name="learning_rate", type="continuous", bounds=(1e-4, 1e-1), 
                 log_scale=True, normalize=True),
        Dimension(name="optimizer", type="categorical", choices=["sgd", "adam"]),
        Dimension(name="architecture", type="categorical", choices=["resnet", "vgg"]),
        Dimension(name="batch_size", type="ordinal", choices=[8, 16, 32, 64], normalize=True),
        Dimension(name="num_layers", type="ordinal", choices=[3, 4, 5], normalize=True),
    ))
    
    print("HPT CIFAR Search Space:")
    print("  Learning rate: [1e-4, 1e-1] (log scale)")
    print("  Optimizer: ['sgd', 'adam']")
    print("  Architecture: ['resnet', 'vgg']")
    print("  Batch size: [8, 16, 32, 64] (ordinal)")
    print("  Num layers: [3, 4, 5] (ordinal)")
    print()

    # Show corruption configuration (without creating actual corruptor yet)
    print(f"Corruption type: {CORRUPTION_TYPE}")
    if CORRUPTION_TYPE != 'none':
        print(f"  Crash value: {TRAINING_CRASH_VALUE}")
        if CORRUPTION_TYPE == 'periodic':
            print(f"  Periodic interval: every {PERIODIC_INTERVAL} observations")
        elif CORRUPTION_TYPE == 'time_budget':
            print(f"  Time budget alpha: {TIME_BUDGET_ALPHA} (T^{TIME_BUDGET_ALPHA})")
        elif CORRUPTION_TYPE == 'budget':
            print(f"  Corruption budget: {CORRUPTION_BUDGET}")
    else:
        print("  No corruption applied")
    print()
    
    # Create model factories following Forrester pattern
    def mixed_gp_model_factory(X, Y, **kwargs):
        """Factory for mixed variable GP models."""
        # Categorical dimensions are optimizer and architecture (indices 1 and 2)
        cat_dims = search_space.categorical_dims
        # Merge default kwargs with any additional kwargs
        merged_kwargs = {
            'standardize': STANDARDIZE,
            'fit_hyperparameters': FIT_STANDARD_GP,
            **kwargs
        }
        return create_mixed_gp_model(X, Y, cat_dims=cat_dims, **merged_kwargs)
    
    def mixed_rcgp_model_factory(X, Y, **kwargs):
        """Factory for mixed variable RCGP models."""
        # Categorical dimensions are optimizer and architecture (indices 1 and 2)
        cat_dims = search_space.categorical_dims
        # Merge rcgp_kwargs with any additional kwargs
        merged_kwargs = {**rcgp_kwargs, **kwargs}
        return create_mixed_rcgp_model(X, Y, cat_dims=cat_dims, **merged_kwargs)
    
    def mixed_student_t_model_factory(X, Y, **kwargs):
        """Factory for mixed variable Student-t Process models."""
        # Categorical dimensions are optimizer and architecture (indices 1 and 2)
        cat_dims = search_space.categorical_dims
        # Merge student_t_kwargs with any additional kwargs
        merged_kwargs = {**student_t_kwargs, **kwargs}
        return create_student_t_model(X, Y, cat_dims=cat_dims, **merged_kwargs)
    
    def mixed_diagnostic_gp_model_factory(X, Y, **kwargs):
        """Factory for mixed variable Diagnostic GP models."""
        # Categorical dimensions are optimizer and architecture (indices 1 and 2)
        cat_dims = search_space.categorical_dims
        # Merge diagnostic_kwargs with any additional kwargs
        merged_kwargs = {**diagnostic_kwargs, **kwargs}
        return create_diagnostic_gp_model(X, Y, cat_dims=cat_dims, **merged_kwargs)

    def mixed_a2rcgp_model_factory(X, Y, **kwargs):
        """Factory for mixed variable A2RCGP models."""
        # Categorical dimensions are optimizer and architecture (indices 1 and 2)
        cat_dims = search_space.categorical_dims
        # Merge a2rcgp_kwargs with any additional kwargs
        merged_kwargs = {**a2rcgp_kwargs, **kwargs}
        return create_a2rcgp_model(X, Y, cat_dims=cat_dims, **merged_kwargs)

    # Create schedulers
    gp_scheduler = create_scheduler(GP_SCHEDULER_TYPE, 'gp')
    rcgp_scheduler = create_scheduler(RCGP_SCHEDULER_TYPE, 'rcgp')
    student_scheduler = create_scheduler(GP_SCHEDULER_TYPE, 'student')
    diagnostic_scheduler = create_scheduler(GP_SCHEDULER_TYPE, 'diagnostic')
    a2rcgp_scheduler = create_scheduler(A2RCGP_SCHEDULER_TYPE, 'a2rcgp')
    
    print(f"GP using scheduler: {gp_scheduler.__class__.__name__}")
    print(f"RCGP using scheduler: {rcgp_scheduler.__class__.__name__}")
    print(f"Student-t using scheduler: {student_scheduler.__class__.__name__}")
    print(f"Diagnostic GP using scheduler: {diagnostic_scheduler.__class__.__name__}")
    print(f"A2RCGP using scheduler: {a2rcgp_scheduler.__class__.__name__}")
    print()
    
    # Test: Mixed variable hyperparameter optimization with standard GP
    print("=" * 80)
    print("HPT CIFAR with Mixed Variable GP (New Framework)")
    print("=" * 80)

    gp_evaluator = create_fresh_evaluator()
    runner = ExperimentRunner(search_space, gp_evaluator)
    gp_results = runner.run(
        n_iterations=N_ITERATIONS,
        n_initial=N_INITIAL,
        model_factory=mixed_gp_model_factory,
        acquisition_factory=UCBAcquisition.create,
        seed=SEED,
        model_kwargs={},  # Default kwargs already merged in factory
        verbose=True
    )
    
    # Test: Mixed variable hyperparameter optimization with RCGP
    print("\n" + "=" * 80)
    print("HPT CIFAR with Mixed Variable RCGP (New Framework)")
    print("=" * 80)

    rcgp_evaluator = create_fresh_evaluator()
    rcgp_runner = ExperimentRunner(search_space, rcgp_evaluator)
    rcgp_results = rcgp_runner.run(
        n_iterations=N_ITERATIONS,
        n_initial=N_INITIAL,
        model_factory=mixed_rcgp_model_factory,
        acquisition_factory=UCBAcquisition.create,
        seed=SEED,
        model_kwargs={},  # rcgp_kwargs already merged in factory
        verbose=True
    )
    
    # Test: Mixed variable hyperparameter optimization with Student-t Process
    print("\n" + "=" * 80)
    print("HPT CIFAR with Mixed Variable Student-t Process (New Framework)")
    print("=" * 80)

    student_evaluator = create_fresh_evaluator()
    student_runner = ExperimentRunner(search_space, student_evaluator)
    student_results = student_runner.run(
        n_iterations=N_ITERATIONS,
        n_initial=N_INITIAL,
        model_factory=mixed_student_t_model_factory,
        acquisition_factory=UCBAcquisition.create,
        seed=SEED,
        model_kwargs={},  # student_t_kwargs already merged in factory
        verbose=True
    )
    
    # Test: Mixed variable hyperparameter optimization with Diagnostic GP
    print("\n" + "=" * 80)
    print("HPT CIFAR with Mixed Variable Diagnostic GP (New Framework)")
    print("=" * 80)

    diagnostic_evaluator = create_fresh_evaluator()
    diagnostic_runner = ExperimentRunner(search_space, diagnostic_evaluator)
    diagnostic_results = diagnostic_runner.run(
        n_iterations=N_ITERATIONS,
        n_initial=N_INITIAL,
        model_factory=mixed_diagnostic_gp_model_factory,
        acquisition_factory=UCBAcquisition.create,
        seed=SEED,
        model_kwargs={},  # diagnostic_kwargs already merged in factory
        verbose=True
    )

    # Test: Mixed variable hyperparameter optimization with A2RCGP
    print("\n" + "=" * 80)
    print("HPT CIFAR with Mixed Variable A2RCGP (New Framework)")
    print("=" * 80)

    a2rcgp_evaluator = create_fresh_evaluator()
    a2rcgp_runner = ExperimentRunner(search_space, a2rcgp_evaluator)
    a2rcgp_results = a2rcgp_runner.run(
        n_iterations=N_ITERATIONS,
        n_initial=N_INITIAL,
        model_factory=mixed_a2rcgp_model_factory,
        acquisition_factory=UCBAcquisition.create,
        seed=SEED,
        model_kwargs={},  # a2rcgp_kwargs already merged in factory
        verbose=True
    )

    # Analysis following Forrester pattern
    print("\n" + "=" * 80)
    print("EXPERIMENT RESULTS COMPARISON")
    print("=" * 80)
    
    print("GP Model Results:")
    print(f"  Best observed accuracy: {gp_results['best_observed_value']:.4f}")
    print(f"  Best observed params: {gp_results['best_observed_params']}")
    print(f"  Best true accuracy: {gp_results['best_true_value']:.4f}")
    print(f"  Best true params: {gp_results['best_true_params']}")
    
    print("\nRCGP Model Results:")
    print(f"  Best observed accuracy: {rcgp_results['best_observed_value']:.4f}")
    print(f"  Best observed params: {rcgp_results['best_observed_params']}")
    print(f"  Best true accuracy: {rcgp_results['best_true_value']:.4f}")
    print(f"  Best true params: {rcgp_results['best_true_params']}")
    
    print("\nStudent-t Process Model Results:")
    print(f"  Best observed accuracy: {student_results['best_observed_value']:.4f}")
    print(f"  Best observed params: {student_results['best_observed_params']}")
    print(f"  Best true accuracy: {student_results['best_true_value']:.4f}")
    print(f"  Best true params: {student_results['best_true_params']}")
    
    print("\nDiagnostic GP Model Results:")
    print(f"  Best observed accuracy: {diagnostic_results['best_observed_value']:.4f}")
    print(f"  Best observed params: {diagnostic_results['best_observed_params']}")
    print(f"  Best true accuracy: {diagnostic_results['best_true_value']:.4f}")
    print(f"  Best true params: {diagnostic_results['best_true_params']}")

    print("\nA2RCGP Model Results:")
    print(f"  Best observed accuracy: {a2rcgp_results['best_observed_value']:.4f}")
    print(f"  Best observed params: {a2rcgp_results['best_observed_params']}")
    print(f"  Best true accuracy: {a2rcgp_results['best_true_value']:.4f}")
    print(f"  Best true params: {a2rcgp_results['best_true_params']}")
    
    # Show all evaluated configurations for GP
    print(f"\nGP - All evaluated configurations ({len(gp_results['all_results'])}):")
    print("-" * 60)
    for i, result in enumerate(gp_results['all_results']):
        lr = result.x['learning_rate']
        opt = result.x['optimizer']
        arch = result.x['architecture']
        batch = result.x['batch_size']
        layers = result.x['num_layers']
        acc = result.y_true
        print(f"{i+1:2d}. lr={lr:.1e} {opt:4s} {arch:6s} batch={batch:2d} layers={layers} → {acc:.4f}")
    
    # Show all evaluated configurations for RCGP
    print(f"\nRCGP - All evaluated configurations ({len(rcgp_results['all_results'])}):")
    print("-" * 60)
    for i, result in enumerate(rcgp_results['all_results']):
        lr = result.x['learning_rate']
        opt = result.x['optimizer']
        arch = result.x['architecture']
        batch = result.x['batch_size']
        layers = result.x['num_layers']
        acc = result.y_true
        print(f"{i+1:2d}. lr={lr:.1e} {opt:4s} {arch:6s} batch={batch:2d} layers={layers} → {acc:.4f}")
    
    # Show all evaluated configurations for Student-t
    print(f"\nStudent-t - All evaluated configurations ({len(student_results['all_results'])}):")
    print("-" * 60)
    for i, result in enumerate(student_results['all_results']):
        lr = result.x['learning_rate']
        opt = result.x['optimizer']
        arch = result.x['architecture']
        batch = result.x['batch_size']
        layers = result.x['num_layers']
        acc = result.y_true
        print(f"{i+1:2d}. lr={lr:.1e} {opt:4s} {arch:6s} batch={batch:2d} layers={layers} → {acc:.4f}")
    
    # Show all evaluated configurations for Diagnostic GP
    print(f"\nDiagnostic GP - All evaluated configurations ({len(diagnostic_results['all_results'])}):")
    print("-" * 60)
    for i, result in enumerate(diagnostic_results['all_results']):
        lr = result.x['learning_rate']
        opt = result.x['optimizer']
        arch = result.x['architecture']
        batch = result.x['batch_size']
        layers = result.x['num_layers']
        acc = result.y_true
        print(f"{i+1:2d}. lr={lr:.1e} {opt:4s} {arch:6s} batch={batch:2d} layers={layers} → {acc:.4f}")

    # Show all evaluated configurations for A2RCGP
    print(f"\nA2RCGP - All evaluated configurations ({len(a2rcgp_results['all_results'])}):")
    print("-" * 60)
    for i, result in enumerate(a2rcgp_results['all_results']):
        lr = result.x['learning_rate']
        opt = result.x['optimizer']
        arch = result.x['architecture']
        batch = result.x['batch_size']
        layers = result.x['num_layers']
        acc = result.y_true
        print(f"{i+1:2d}. lr={lr:.1e} {opt:4s} {arch:6s} batch={batch:2d} layers={layers} → {acc:.4f}")

    # Search space utilization summary
    models = [
        ("GP", gp_results['all_results']),
        ("RCGP", rcgp_results['all_results']),
        ("Student-t", student_results['all_results']),
        ("Diagnostic GP", diagnostic_results['all_results']),
        ("A2RCGP", a2rcgp_results['all_results'])
    ]
    
    for model_name, results in models:
        print(f"\nSearch space utilization ({model_name}):")
        print(f"  Optimizers tried: {set(r.x['optimizer'] for r in results)}")
        print(f"  Architectures tried: {set(r.x['architecture'] for r in results)}")
        print(f"  Batch sizes tried: {sorted(set(r.x['batch_size'] for r in results))}")
        print(f"  Layer counts tried: {sorted(set(r.x['num_layers'] for r in results))}")
    
    # Model hyperparameters (following Forrester pattern)
    print("\n" + "=" * 80)
    print("MODEL HYPERPARAMETERS")
    print("=" * 80)
    
    # Helper function to extract hyperparameters
    def extract_model_hyperparameters(model, model_name):
        print(f"\n{model_name} Model:")
        print(f"  Noise std (sigma): {torch.sqrt(model.likelihood.noise).item():.4f}")
        
        try:
            # Handle mixed variable model structure
            if hasattr(model, 'model'):
                # Mixed variable wrapper
                underlying_model = model.model
                covar = underlying_model.covar_module
            else:
                # Direct model
                covar = model.covar_module
            
            if hasattr(covar, 'base_kernel'):
                # ScaleKernel case
                if covar.base_kernel.lengthscale is not None:
                    print(f"  Lengthscale: {covar.base_kernel.lengthscale.item():.4f}")
                if hasattr(covar, 'outputscale'):
                    print(f"  Output scale: {covar.outputscale.item():.4f}")
            else:
                # Direct kernel case
                if hasattr(covar, 'lengthscale') and covar.lengthscale is not None:
                    print(f"  Lengthscale: {covar.lengthscale.item():.4f}")
                if hasattr(covar, 'outputscale'):
                    print(f"  Output scale: {covar.outputscale.item():.4f}")
            
            if hasattr(model, 'model'):
                mean_module = underlying_model.mean_module
            else:
                mean_module = model.mean_module
                
            if hasattr(mean_module, 'constant'):
                print(f"  Mean constant: {mean_module.constant.item():.4f}")
                
            # RCGP-specific parameters
            if hasattr(model, 'weighting_function'):
                print(f"  Plateau width: {model.weighting_function.plateau_width:.4f}")
                print(f"  C parameter: {model.weighting_function.c:.4f}")
                
            # Student-t specific parameters
            if hasattr(model, 'nu'):
                print(f"  Degrees of freedom (nu): {model.nu.item():.2f}")
                
        except Exception as e:
            print(f"  Could not extract {model_name} hyperparameters: {e}")
    
    # Extract hyperparameters for all models
    extract_model_hyperparameters(gp_results['final_model'], "GP")
    extract_model_hyperparameters(rcgp_results['final_model'], "RCGP")
    extract_model_hyperparameters(student_results['final_model'], "Student-t")
    extract_model_hyperparameters(a2rcgp_results['final_model'], "A2RCGP")
    
    # Diagnostic GP model info
    diagnostic_model = diagnostic_results['final_model']
    print("\nDiagnostic GP Model:")
    try:
        diagnostic_info = diagnostic_model.get_diagnostic_info()
        print(f"  Total points: {diagnostic_info['total_points']}")
        print(f"  Inliers: {diagnostic_info['num_inliers']}")
        print(f"  Outliers detected: {diagnostic_info['num_outliers']}")
        if diagnostic_info['outlier_indices']:
            print(f"  Outlier indices: {diagnostic_info['outlier_indices']}")
        
        # Show underlying model hyperparameters
        underlying_model = diagnostic_model.model
        print("  Underlying Model:")
        print(f"    Noise std (sigma): {torch.sqrt(underlying_model.likelihood.noise).item():.4f}")
        
        covar = underlying_model.covar_module
        if hasattr(covar, 'base_kernel'):
            if covar.base_kernel.lengthscale is not None:
                print(f"    Lengthscale: {covar.base_kernel.lengthscale.item():.4f}")
            if hasattr(covar, 'outputscale'):
                print(f"    Output scale: {covar.outputscale.item():.4f}")
        else:
            if hasattr(covar, 'lengthscale') and covar.lengthscale is not None:
                print(f"    Lengthscale: {covar.lengthscale.item():.4f}")
            if hasattr(covar, 'outputscale'):
                print(f"    Output scale: {covar.outputscale.item():.4f}")
        
        if hasattr(underlying_model.mean_module, 'constant'):
            print(f"    Mean constant: {underlying_model.mean_module.constant.item():.4f}")
    except Exception as e:
        print(f"  Could not extract Diagnostic GP info: {e}")
    
    # Comprehensive result saving and analysis
    print("\n" + "=" * 80)
    print("SAVING RESULTS AND GENERATING PLOTS")
    print("=" * 80)
    
    # Create results dictionary for analysis
    results_dict = {
        "GP": gp_results['all_results'],
        "RCGP": rcgp_results['all_results'],
        "Student-t": student_results['all_results'],
        "Diagnostic GP": diagnostic_results['all_results'],
        "A2RCGP": a2rcgp_results['all_results']
    }
    
    # Calculate optimal value (maximum observed across all models)
    all_values = []
    for results in results_dict.values():
        all_values.extend([r.y_true for r in results])
    optimal_value = max(all_values) if all_values else 0.0
    print(f"Optimal value (max observed): {optimal_value:.4f}")
    
    # Save chosen points and observed values
    save_chosen_points(results_dict, artifacts_dir)
    
    # Save model hyperparameters
    save_model_hyperparameters(results_dict, artifacts_dir)
    
    # Create cumulative regret plot
    cumulative_regret_plot_path = os.path.join(artifacts_dir, "cumulative_regret_comparison.png")
    plot_cumulative_regret_comparison(
        results_dict,
        optimal_value,
        cumulative_regret_plot_path,
        title="HPT CIFAR: Cumulative Regret Comparison"
    )

    # Create simple regret plot
    simple_regret_plot_path = os.path.join(artifacts_dir, "simple_regret_comparison.png")
    plot_simple_regret_comparison(
        results_dict,
        optimal_value,
        simple_regret_plot_path,
        title="HPT CIFAR: Simple Regret Comparison"
    )
    
    # Create individual model summary plots
    for model_name, results in results_dict.items():
        print(f"Creating summary plot for: {model_name}")
        
        # Create the summary plot
        plot_filename = f"hpt_cifar_{model_name.lower().replace(' ', '_').replace('-', '_')}_summary.png"
        plot_path = os.path.join(artifacts_dir, plot_filename)
        
        # For HPT CIFAR, we don't have a true objective function, so we'll create a simple plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Observed values over iterations
        iterations = list(range(1, len(results) + 1))
        observed_values = [r.y_observed for r in results]
        true_values = [r.y_true for r in results]
        
        ax1.plot(iterations, observed_values, 'o-', label='Observed', linewidth=2, markersize=6)
        ax1.plot(iterations, true_values, 's-', label='True', linewidth=2, markersize=6)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Accuracy')
        ax1.set_title(f'{model_name}: Observed vs True Values')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Cumulative maximum
        cumulative_max = []
        current_max = observed_values[0]
        for value in observed_values:
            current_max = max(current_max, value)
            cumulative_max.append(current_max)
        
        ax2.plot(iterations, cumulative_max, 'o-', color='green', linewidth=2, markersize=6)
        ax2.axhline(y=optimal_value, color='red', linestyle='--', label=f'Optimal ({optimal_value:.4f})')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Cumulative Maximum Accuracy')
        ax2.set_title(f'{model_name}: Cumulative Maximum')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Summary plot saved to: {plot_path}")
    
    # Save comprehensive results using utility functions
    print("\nSaving comprehensive results...")
    
    # Create a combined results dictionary for the utility functions
    combined_results = {}
    for model_name, results in results_dict.items():
        combined_results[model_name] = results
    
    # Save experiment results
    save_experiment_results(
        results=combined_results,
        experiment_name="hpt_cifar_experiment",
        artifacts_dir=artifacts_dir,
        save_pickle=True,
        save_json=True,
        optimal_value=optimal_value,
        verbose=True
    )
    
    # Save comparison table
    save_comparison_table(
        results_dict=combined_results,
        experiment_name="hpt_cifar_experiment",
        artifacts_dir=artifacts_dir,
        optimal_value=optimal_value
    )
    
    # Print final summary
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    print(f"Optimal value found: {optimal_value:.4f}")
    print(f"Total evaluations per model: {len(gp_results['all_results'])}")
    print(f"Corruption type: {CORRUPTION_TYPE}")
    if CORRUPTION_TYPE == 'budget':
        print(f"Corruption budget: {CORRUPTION_BUDGET}")
    
    # Count corruptions
    for model_name, results in results_dict.items():
        corruptions = sum(1 for r in results if r.y_observed != r.y_true)
        print(f"{model_name} corruptions: {corruptions}/{len(results)}")
    
    print("\nExperiment completed successfully!")
    print(f"Total evaluations: GP={len(gp_results['all_results'])}, RCGP={len(rcgp_results['all_results'])}, Student-t={len(student_results['all_results'])}, Diagnostic GP={len(diagnostic_results['all_results'])}, A2RCGP={len(a2rcgp_results['all_results'])}")
    print(f"\nAll artifacts saved to: {artifacts_dir}/")
    print("Files saved:")
    print("  - experiment_config.json (configuration)")
    print("  - chosen_points.json (detailed point data)")
    print("  - chosen_points_summary.csv (point data in CSV format)")
    print("  - model_hyperparameters.json (model hyperparameters)")
    print("  - cumulative_regret_comparison.png (cumulative regret plot)")
    print("  - simple_regret_comparison.png (simple regret plot)")
    print("  - hpt_cifar_*_summary.png (individual model plots)")
    print("  - experiment_results.pkl (pickle format)")
    print("  - experiment_results.json (JSON format)")
    print("  - comparison_table.json (comparison metrics)")


if __name__ == "__main__":
    main()