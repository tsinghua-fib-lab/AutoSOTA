"""Main experiment script for Twitter Flash Crash robust optimization experiment.

This script demonstrates RCGP-UCB's ability to find robust trading strategy parameters
by optimizing on corrupted data (containing the Twitter flash crash) while evaluating
true performance on clean out-of-sample data.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any, Optional
import warnings
import logging

# BO Framework imports
from bo_framework import SearchSpace, Dimension, ExperimentRunner
from bo_framework.base.acquisition import UCBAcquisition
from bo_framework.models.factory import create_mixed_gp_model, create_mixed_rcgp_model

# Experiment-specific imports
from experiments.twitter_crash.data_utils import load_and_prepare_data, check_for_crash
from experiments.twitter_crash.evaluator import TwitterCrashEvaluator

# Import our new robust acquisition function
try:
    from bo_framework.base.acquisition import RobustUCBAcquisition
    ROBUST_AVAILABLE = True
except ImportError:
    warnings.warn("RobustUCBAcquisition not available, will skip robust acquisition tests")
    ROBUST_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_search_space(reduced: bool = False) -> SearchSpace:
    """Create the search space for EMA crossover strategy parameters.
    
    Args:
        reduced: If True, create a smaller search space for faster testing
    
    Returns:
        SearchSpace with fast window, slow window, and stop-loss dimensions
    """
    if reduced:
        # Smaller search space for testing/debugging
        return SearchSpace([
            # Fast EMA window: reduced range
            Dimension(name="W_Fast", type="ordinal", 
                     choices=[10, 15, 20, 25, 30], normalize=True),
            # Slow EMA window: reduced range
            Dimension(name="W_Slow", type="ordinal",
                     choices=[40, 50, 60, 70, 80], normalize=True),
            # Stop-Loss: smaller range
            Dimension(name="S_L", type="continuous", 
                     bounds=(0.01, 0.02), normalize=True),
        ])
    else:
        # Full search space
        return SearchSpace([
            # Fast EMA window: 5 to 100 minutes (ordinal for integer values)
            Dimension(name="W_Fast", type="ordinal", 
                     choices=list(range(5, 101)), normalize=True),
            # Slow EMA window: 101 to 390 minutes (ordinal for integer values)  
            Dimension(name="W_Slow", type="ordinal",
                     choices=list(range(101, 391)), normalize=True),
            # Stop-Loss: 0.5% to 3.0% (continuous)
            Dimension(name="S_L", type="continuous", 
                     bounds=(0.005, 0.03), normalize=True),
        ])


def create_model_factories(search_space: SearchSpace):
    """Create model factory functions for GP and RCGP models.
    
    Args:
        search_space: SearchSpace instance
        
    Returns:
        Tuple of (gp_factory, rcgp_factory) functions
    """
    # Identify categorical dimensions (ordinal dimensions are treated like categorical)
    cat_dims = search_space.ordinal_dims
    
    def gp_model_factory(X, Y, **kwargs):
        """Factory for standard GP models."""
        return create_mixed_gp_model(
            X, Y, 
            cat_dims=cat_dims,
            standardize=True,
            fit_hyperparameters=True,
            **kwargs
        )
    
    def rcgp_model_factory(X, Y, **kwargs):
        """Factory for robust RCGP models."""
        return create_mixed_rcgp_model(X, Y, cat_dims=cat_dims, **kwargs)
    
    return gp_model_factory, rcgp_model_factory


def run_single_trial(
    runner: ExperimentRunner, 
    gp_factory, 
    rcgp_factory,
    trial_seed: int,
    n_iterations: int = 100,
    n_initial: int = 20) -> Dict[str, Any]:
    """Run a single trial comparing GP-UCB vs RCGP-UCB.
    
    Args:
        runner: ExperimentRunner instance
        gp_factory: GP model factory function
        rcgp_factory: RCGP model factory function  
        trial_seed: Random seed for this trial
        n_iterations: Number of BO iterations
        n_initial: Number of initial random samples
        
    Returns:
        Dictionary with results from both methods
    """
    results = {}
    
    logger.info(f"Running trial with seed {trial_seed}")
    
    # Run GP-UCB (Standard Method)
    logger.info("  Running GP-UCB...")
    runner.evaluator.reset()  # Reset state for a clean run
    results_gp = runner.run(
        n_iterations=n_iterations,
        n_initial=n_initial,
        model_factory=gp_factory,
        acquisition_factory=UCBAcquisition.create,
        seed=trial_seed,
        verbose=True  # Enable verbose output from BO
    )
    results["GP-UCB"] = results_gp
    
    # Run RCGP-UCB (Robust Method)
    logger.info("  Running RCGP-UCB...")
    
    # RCGP model kwargs following working examples
    rcgp_kwargs = {
        "param_handling_dict": {
            "plateau_width": {"method": "heuristics"},
            "c": {"method": "manual", "value": 1.0},
            "sigma": {"method": "fit"},
            "mean": {"method": "fit"}
        },
        "fitting_objective_type": "wloo-cv",
        "optimizer_type": "lbfgs",
        "verbose": False
    }
    
    # Use RCGP model with standard UCB acquisition (no custom acquisition yet)
    logger.info("  Using RCGP model with standard UCB acquisition")
    runner.evaluator.reset()  # Reset state for a clean run
    results_rcgp = runner.run(
        n_iterations=n_iterations,
        n_initial=n_initial,
        model_factory=rcgp_factory,
        acquisition_factory=UCBAcquisition.create,  # Standard UCB
        seed=trial_seed,
        model_kwargs=rcgp_kwargs,  # Pass as model_kwargs
        verbose=True
    )
    results["RCGP-Standard"] = results_rcgp
    
    return results


def analyze_convergence(all_results: Dict[str, List[Dict]]) -> Dict[str, Any]:
    """Analyze convergence of true performance (OOS Sharpe) across trials.
    
    Args:
        all_results: Dictionary mapping method names to lists of trial results
        
    Returns:
        Dictionary with convergence statistics
    """
    convergence_stats = {}
    
    for method_name, trials in all_results.items():
        if not trials:
            continue
            
        # Extract y_true (OOS Sharpe) convergence for each trial
        y_true_trials = []
        
        for trial_results in trials:
            # Extract y_true values from evaluation results
            y_true_values = [res.y_true for res in trial_results['all_results']]
            
            # Calculate cumulative maximum (best found so far)
            best_so_far = np.maximum.accumulate(y_true_values)
            y_true_trials.append(best_so_far)
        
        if y_true_trials:
            # Calculate statistics across trials
            mean_convergence = np.mean(y_true_trials, axis=0)
            std_convergence = np.std(y_true_trials, axis=0)
            final_values = [trial[-1] for trial in y_true_trials]
            
            convergence_stats[method_name] = {
                'mean_convergence': mean_convergence,
                'std_convergence': std_convergence,
                'final_mean': np.mean(final_values),
                'final_std': np.std(final_values),
                'n_trials': len(y_true_trials)
            }
    
    return convergence_stats


def plot_convergence(convergence_stats: Dict[str, Any], save_path: Optional[str] = None):
    """Plot convergence of true performance across methods.
    
    Args:
        convergence_stats: Statistics from analyze_convergence
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    for method_name, stats in convergence_stats.items():
        mean_conv = stats['mean_convergence']
        std_conv = stats['std_convergence']
        n_trials = stats['n_trials']
        
        # Standard error
        stderr = std_conv / np.sqrt(n_trials)
        
        # Plot mean with error bars
        iterations = range(len(mean_conv))
        plt.plot(iterations, mean_conv, label=f"{method_name} (n={n_trials})", linewidth=2)
        plt.fill_between(iterations, 
                        mean_conv - stderr, 
                        mean_conv + stderr, 
                        alpha=0.2)
    
    plt.title("Convergence of Out-of-Sample (True) Performance", fontsize=14)
    plt.xlabel("BO Iteration", fontsize=12)
    plt.ylabel("Best OOS Sharpe Ratio Found", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Convergence plot saved to {save_path}")
    
    plt.show()


def analyze_robustness(evaluator: TwitterCrashEvaluator) -> Dict[str, Any]:
    """Analyze robustness characteristics of the experiment.
    
    Args:
        evaluator: Evaluator with evaluation history
        
    Returns:
        Dictionary with robustness analysis
    """
    # Get summary statistics
    summary = evaluator.get_summary_statistics()
    crash_impact = evaluator.analyze_crash_impact()
    
    # Calculate correlation between IS and OOS performance
    correlation = summary.get('correlation_is_oos', 0)
    
    # Best parameters analysis
    best_analysis = evaluator.get_detailed_analysis()
    
    return {
        'summary': summary,
        'crash_impact': crash_impact,
        'correlation_is_oos': correlation,
        'best_analysis': best_analysis
    }


def main():
    """Main experiment execution function."""
    logger.info("Starting Twitter Flash Crash Robust Optimization Experiment")
    
    # Experiment configuration
    N_TRIALS = 1
    N_ITERATIONS = 40 
    N_INITIAL = 5 
    DATA_PATH = None  # Will generate synthetic data
    
    # Create output directory
    output_dir = Path("experiments/twitter_crash/results")
    output_dir.mkdir(exist_ok=True)
    
    try:
        # 1. Load and prepare data
        logger.info("Loading and preparing data...")
        data_is, data_oos = load_and_prepare_data(
            data_path=DATA_PATH,
            generate_synthetic=True
        )
        
        # Check for crash events in the data
        logger.info("Analyzing crash events...")
        crash_events = check_for_crash(data_is)
        logger.info(f"Found {len(crash_events)} potential crash events in IS data")
        
        # 2. Create search space and evaluator
        search_space = create_search_space(reduced=True)  # Use reduced space for testing
        
        # Extract the ranges from the search space definition for pre-computation
        # This ensures EMA cache covers the entire optimization domain
        try:
            # Access the Dimension objects
            w_fast_dim = next(d for d in search_space.dimensions if d.name == "W_Fast")
            w_slow_dim = next(d for d in search_space.dimensions if d.name == "W_Slow")

            # Extract choices and determine ranges
            w_fast_choices = w_fast_dim.choices
            w_slow_choices = w_slow_dim.choices
            
            w_fast_range = (min(w_fast_choices), max(w_fast_choices))
            w_slow_range = (min(w_slow_choices), max(w_slow_choices))

            logger.info(f"Initializing Evaluator with pre-computation ranges: Fast={w_fast_range}, Slow={w_slow_range}")
            
            # Initialize evaluator (This will trigger pre-computation)
            evaluator = TwitterCrashEvaluator(
                data_is, 
                data_oos, 
                w_fast_range=w_fast_range,
                w_slow_range=w_slow_range,
                verbose=True  # Show pre-computation progress
            )
        except (AttributeError, TypeError, StopIteration, ValueError) as e:
            logger.error(f"Could not extract ranges from SearchSpace for pre-computation: {e}")
            logger.warning("Falling back to standard Evaluator initialization without full pre-computation.")
            # Initialize without specific ranges (will use defaults)
            evaluator = TwitterCrashEvaluator(data_is, data_oos, verbose=False)

        runner = ExperimentRunner(search_space, evaluator)
        
        # 3. Create model factories
        gp_factory, rcgp_factory = create_model_factories(search_space)
        
        # 4. Run multiple trials
        logger.info(f"Running {N_TRIALS} trials with {N_ITERATIONS} iterations each")
        all_results = {"GP-UCB": [], "RCGP-Standard": []}
        
        for trial in range(N_TRIALS):
            trial_results = run_single_trial(
                runner=runner,
                gp_factory=gp_factory,
                rcgp_factory=rcgp_factory,
                trial_seed=trial,
                n_iterations=N_ITERATIONS,
                n_initial=N_INITIAL
            )
            
            # Store results
            for method_name, results in trial_results.items():
                all_results[method_name].append(results)
        
        # 5. Analyze results
        logger.info("Analyzing results...")
        
        # Convergence analysis
        convergence_stats = analyze_convergence(all_results)
        
        # Print summary
        logger.info("Final Results Summary:")
        for method_name, stats in convergence_stats.items():
            logger.info(f"  {method_name}: Final OOS Sharpe = "
                       f"{stats['final_mean']:.3f} ± {stats['final_std']:.3f}")
        
        # Plot convergence
        plot_convergence(convergence_stats, 
                        save_path=output_dir / "convergence_plot.png")
        
        # Robustness analysis
        robustness_analysis = analyze_robustness(evaluator)
        logger.info(f"IS-OOS Correlation: {robustness_analysis['correlation_is_oos']:.3f}")
        
        # 6. Save detailed results
        results_file = output_dir / "experiment_results.txt"
        with open(results_file, 'w') as f:
            f.write("Twitter Flash Crash Experiment Results\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Configuration:\n")
            f.write(f"  Trials: {N_TRIALS}\n")
            f.write(f"  Iterations: {N_ITERATIONS}\n")
            f.write(f"  Initial samples: {N_INITIAL}\n")
            f.write(f"  IS data: {len(data_is)} points\n")
            f.write(f"  OOS data: {len(data_oos)} points\n\n")
            
            f.write("Final Performance:\n")
            for method_name, stats in convergence_stats.items():
                f.write(f"  {method_name}: {stats['final_mean']:.4f} ± {stats['final_std']:.4f}\n")
            
            f.write(f"\nRobustness Analysis:\n")
            f.write(f"  IS-OOS Correlation: {robustness_analysis['correlation_is_oos']:.4f}\n")
            
            if robustness_analysis['best_analysis']:
                best = robustness_analysis['best_analysis']
                f.write(f"\nBest Configuration:\n")
                f.write(f"  Parameters: {best['parameters']}\n")
                f.write(f"  IS Sharpe: {best['in_sample']['sharpe_ratio']:.4f}\n")
                f.write(f"  OOS Sharpe: {best['out_of_sample']['sharpe_ratio']:.4f}\n")
                f.write(f"  Overfitting Ratio: {best['overfitting_ratio']:.4f}\n")
        
        logger.info(f"Detailed results saved to {results_file}")
        logger.info("Experiment completed successfully!")
        
        return all_results, convergence_stats, robustness_analysis
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        raise


if __name__ == "__main__":
    # Run the experiment
    results = main()