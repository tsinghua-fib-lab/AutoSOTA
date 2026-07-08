"""Test trading strategy optimization with RCGP vs Standard GP - following Forrester example pattern."""

import os
import torch
from bo_framework import SearchSpace, Dimension, ExperimentRunner
from bo_framework.base.acquisition import UCBAcquisition
from bo_framework.models.factory import create_botorch_model, create_rcgp_model
from experiments.financial_trading.evaluator import TradingStrategyEvaluator
from experiments.financial_trading.grid_search import GridSearchOptimizer


def main():
    """Test trading strategy optimization following the Forrester example pattern."""
    
    # Step 1: Establish optimal baseline with grid search
    print("=" * 80)
    print("STEP 1: ESTABLISHING OPTIMAL BASELINE")
    print("=" * 80)
    
    evaluator = TradingStrategyEvaluator(cache_data=True)
    
    # Quick grid search for optimal baseline (small for testing)
    grid_search = GridSearchOptimizer(evaluator)
    grid_results = grid_search.search(
        short_window_range=(2, 4),
        long_window_range=(15, 20), 
        short_step=2,
        long_step=5,
        verbose=True
    )
    
    optimal_pnl = grid_results['optimal_pnl']
    optimal_params = grid_results['optimal_params']
    
    print(f"\nOptimal PnL from grid search: {optimal_pnl:.4f}")
    print(f"Optimal parameters: {optimal_params}")
    
    # Create search space
    search_space = SearchSpace((
        Dimension(name="short_window", type="ordinal", 
                 choices=list(range(1, 31)), normalize=True),
        Dimension(name="long_window", type="ordinal", 
                 choices=list(range(10, 121)), normalize=True),
    ))
    
    # Step 2: Test Standard GP
    print("\n" + "=" * 80)
    print("STEP 2: Standard GP Optimization")
    print("=" * 80)
    
    runner_standard = ExperimentRunner(search_space, evaluator)
    results_standard = runner_standard.run(
        n_iterations=30,
        n_initial=8,
        model_factory=create_botorch_model,
        acquisition_factory=UCBAcquisition.create,
        seed=42,
        model_kwargs={'standardize': True, 'fit_hyperparameters': False},
        verbose=True
    )
    
    # Step 3: Test RCGP
    print("\n" + "=" * 80)
    print("STEP 3: RCGP Optimization")
    print("=" * 80)
    
    runner_rcgp = ExperimentRunner(search_space, evaluator)
    results_rcgp = runner_rcgp.run(
        n_iterations=30,
        n_initial=8,
        model_factory=create_rcgp_model,
        acquisition_factory=UCBAcquisition.create,
        seed=42,
        model_kwargs={
            'standardize': True, 
            'fit_hyperparameters': False,
            'plateau_width': 50.0,
            'sigma': 25.0,
            'c': 1.0
        },
        verbose=True
    )
    
    # Step 4: Compare Results
    print("\n" + "=" * 80)
    print("STEP 4: RESULTS COMPARISON")
    print("=" * 80)
    
    scenarios = [
        ("Standard GP", results_standard),
        ("RCGP", results_rcgp)
    ]
    
    print(f"{'Method':<15} {'Best PnL':<12} {'Best Params':<30} {'Regret':<12}")
    print("-" * 80)
    
    for name, results in scenarios:
        best_pnl = results['best_observed_value']
        best_params = results['best_observed_params']
        regret = optimal_pnl - best_pnl
        
        params_str = f"short={best_params['short_window']}, long={best_params['long_window']}"
        print(f"{name:<15} {best_pnl:<12.4f} {params_str:<30} {regret:<12.4f}")
    
    # Step 5: Show detailed analysis
    print(f"\nOptimal baseline: {optimal_pnl:.4f} at {optimal_params}")
    
    # Get crash statistics
    crash_stats = evaluator.get_crash_statistics()
    if crash_stats.get('crash_detected', False):
        print(f"\nFlash crash detected:")
        print(f"  Time: {crash_stats['crash_timestamp']}")
        print(f"  Magnitude: {crash_stats['crash_percentage']:.2f}%")
        print(f"  Recovery: {crash_stats['recovery_percentage']:.2f}%")
    
    # Show strategy details for best found strategies
    for name, results in scenarios:
        print(f"\n{name} - Best Strategy Details:")
        try:
            best_params = results['best_observed_params']
            details = evaluator.evaluate_strategy_details(
                best_params['short_window'], 
                best_params['long_window']
            )
            print(f"  Total trades: {details['total_trades']}")
            print(f"  Final PnL: {details['final_pnl']:.4f}")
            if details['trades']:
                print(f"  First few trades:")
                for trade in details['trades'][:3]:
                    print(f"    {trade}")
        except Exception as e:
            print(f"  Error getting details: {e}")
    
    # Save results
    print(f"\nSaving results...")
    os.makedirs("artifacts", exist_ok=True)
    
    # Save grid search results
    grid_search.save_results("artifacts/trading_grid_search.csv")
    
    # Save BO comparison results
    from utilities.io import save_experiment_results
    
    comparison_results = {
        "Standard_GP": results_standard,
        "RCGP": results_rcgp
    }
    
    saved_paths = save_experiment_results(
        results=comparison_results,
        experiment_name="trading_strategy_comparison",
        artifacts_dir="artifacts",
        save_pickle=True,
        save_json=True,
        optimal_value=optimal_pnl,
        verbose=True
    )
    
    print(f"Results saved to: {saved_paths['directory']}")
    
    return results_standard, results_rcgp, optimal_pnl


if __name__ == "__main__":
    main()