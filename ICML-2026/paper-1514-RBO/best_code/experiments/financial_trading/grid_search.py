"""Grid search functionality to establish optimal PnL baseline."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from itertools import product
import time
from .evaluator import TradingStrategyEvaluator


class GridSearchOptimizer:
    """
    Exhaustive grid search over trading strategy parameters to find optimal baseline.
    """
    
    def __init__(self, evaluator: TradingStrategyEvaluator):
        """
        Initialize grid search optimizer.
        
        Args:
            evaluator: TradingStrategyEvaluator instance
        """
        self.evaluator = evaluator
        self.results = None
        self.optimal_params = None
        self.optimal_pnl = None
    
    def search(self, 
               short_window_range: Tuple[int, int] = (1, 30),
               long_window_range: Tuple[int, int] = (10, 120),
               short_step: int = 1,
               long_step: int = 5,
               verbose: bool = True) -> Dict:
        """
        Perform exhaustive grid search over parameter space.
        
        Args:
            short_window_range: (min, max) for short window
            long_window_range: (min, max) for long window  
            short_step: Step size for short window
            long_step: Step size for long window
            verbose: Whether to print progress
            
        Returns:
            Dictionary with results
        """
        start_time = time.time()
        
        # Generate parameter combinations
        short_windows = range(short_window_range[0], short_window_range[1] + 1, short_step)
        long_windows = range(long_window_range[0], long_window_range[1] + 1, long_step)
        
        # Filter to ensure short < long constraint
        valid_combinations = [
            (s, l) for s, l in product(short_windows, long_windows) 
            if s < l
        ]
        
        total_combinations = len(valid_combinations)
        
        if verbose:
            print(f"Grid search over {total_combinations} parameter combinations")
            print(f"Short window range: {short_window_range} (step {short_step})")
            print(f"Long window range: {long_window_range} (step {long_step})")
            print("-" * 60)
        
        # Evaluate all combinations
        results = []
        best_pnl = float('-inf')
        best_params = None
        
        for i, (short_win, long_win) in enumerate(valid_combinations):
            params = {'short_window': short_win, 'long_window': long_win}
            
            # Evaluate
            eval_result = self.evaluator.evaluate(params)
            pnl = eval_result.y_true
            
            # Store result
            result_entry = {
                'short_window': short_win,
                'long_window': long_win, 
                'pnl': pnl
            }
            results.append(result_entry)
            
            # Update best
            if pnl > best_pnl:
                best_pnl = pnl
                best_params = params.copy()
            
            # Progress reporting
            if verbose and (i + 1) % 50 == 0:
                elapsed = time.time() - start_time
                progress = (i + 1) / total_combinations * 100
                print(f"Progress: {i+1}/{total_combinations} ({progress:.1f}%) - "
                      f"Best PnL: {best_pnl:.4f} - Elapsed: {elapsed:.1f}s")
        
        elapsed_time = time.time() - start_time
        
        # Store results
        self.results = pd.DataFrame(results)
        self.optimal_params = best_params
        self.optimal_pnl = best_pnl
        
        # Summary statistics
        summary = {
            'optimal_params': best_params,
            'optimal_pnl': best_pnl,
            'total_combinations': total_combinations,
            'elapsed_time': elapsed_time,
            'results_df': self.results,
            'mean_pnl': self.results['pnl'].mean(),
            'std_pnl': self.results['pnl'].std(),
            'min_pnl': self.results['pnl'].min(),
            'max_pnl': self.results['pnl'].max(),
            'median_pnl': self.results['pnl'].median()
        }
        
        if verbose:
            print(f"\nGrid search completed in {elapsed_time:.2f} seconds")
            print(f"Optimal parameters: {best_params}")
            print(f"Optimal PnL: {best_pnl:.4f}")
            print(f"PnL statistics:")
            print(f"  Mean: {summary['mean_pnl']:.4f}")
            print(f"  Std:  {summary['std_pnl']:.4f}")
            print(f"  Min:  {summary['min_pnl']:.4f}")
            print(f"  Max:  {summary['max_pnl']:.4f}")
            print(f"  Med:  {summary['median_pnl']:.4f}")
        
        return summary
    
    def get_top_strategies(self, n: int = 10) -> pd.DataFrame:
        """
        Get top N strategies by PnL.
        
        Args:
            n: Number of top strategies to return
            
        Returns:
            DataFrame with top strategies
        """
        if self.results is None:
            raise ValueError("Must run search() first")
        
        return self.results.nlargest(n, 'pnl')
    
    def get_parameter_analysis(self) -> Dict:
        """
        Analyze parameter sensitivities.
        
        Returns:
            Dictionary with parameter analysis
        """
        if self.results is None:
            raise ValueError("Must run search() first")
        
        df = self.results
        
        # Group by parameters to analyze sensitivity
        short_win_analysis = df.groupby('short_window')['pnl'].agg(['mean', 'std', 'max', 'min', 'count'])
        long_win_analysis = df.groupby('long_window')['pnl'].agg(['mean', 'std', 'max', 'min', 'count'])
        
        return {
            'short_window_analysis': short_win_analysis,
            'long_window_analysis': long_win_analysis,
            'correlation_matrix': df[['short_window', 'long_window', 'pnl']].corr()
        }
    
    def save_results(self, filepath: str):
        """
        Save grid search results to CSV.
        
        Args:
            filepath: Path to save CSV file
        """
        if self.results is None:
            raise ValueError("Must run search() first")
        
        self.results.to_csv(filepath, index=False)
        print(f"Grid search results saved to: {filepath}")


def run_grid_search_analysis():
    """Run comprehensive grid search analysis."""
    print("=" * 80)
    print("TRADING STRATEGY GRID SEARCH ANALYSIS")
    print("=" * 80)
    
    # Initialize evaluator
    evaluator = TradingStrategyEvaluator()
    
    # Initialize grid search
    grid_search = GridSearchOptimizer(evaluator)
    
    # Run coarse grid search first
    print("\n1. COARSE GRID SEARCH")
    print("-" * 40)
    
    coarse_results = grid_search.search(
        short_window_range=(1, 30),
        long_window_range=(10, 120),
        short_step=2,    # Every 2 minutes
        long_step=10,    # Every 10 minutes
        verbose=True
    )
    
    # Analyze results
    print("\n2. TOP 10 STRATEGIES")
    print("-" * 40)
    top_strategies = grid_search.get_top_strategies(10)
    print(top_strategies.to_string(index=False))
    
    # Parameter analysis
    print("\n3. PARAMETER SENSITIVITY ANALYSIS") 
    print("-" * 40)
    param_analysis = grid_search.get_parameter_analysis()
    
    print("Short Window Analysis (top 5 by mean PnL):")
    short_analysis = param_analysis['short_window_analysis'].sort_values('mean', ascending=False)
    print(short_analysis.head().round(4))
    
    print("\nLong Window Analysis (top 5 by mean PnL):")
    long_analysis = param_analysis['long_window_analysis'].sort_values('mean', ascending=False)
    print(long_analysis.head().round(4))
    
    print("\nParameter Correlations:")
    print(param_analysis['correlation_matrix'].round(4))
    
    # Fine grid search around optimal region
    print("\n4. FINE GRID SEARCH AROUND OPTIMAL REGION")
    print("-" * 40)
    
    opt_short = coarse_results['optimal_params']['short_window']
    opt_long = coarse_results['optimal_params']['long_window']
    
    # Search in neighborhood of optimal
    fine_short_range = (max(1, opt_short - 5), opt_short + 6)
    fine_long_range = (max(opt_short + 1, opt_long - 15), opt_long + 16)
    
    fine_grid_search = GridSearchOptimizer(evaluator)
    fine_results = fine_grid_search.search(
        short_window_range=fine_short_range,
        long_window_range=fine_long_range,
        short_step=1,    # Every minute
        long_step=2,     # Every 2 minutes
        verbose=True
    )
    
    print("\nFine search top strategies:")
    fine_top = fine_grid_search.get_top_strategies(5)
    print(fine_top.to_string(index=False))
    
    # Final optimal strategy
    final_optimal = fine_results['optimal_params']
    final_pnl = fine_results['optimal_pnl']
    
    print(f"\n5. FINAL OPTIMAL STRATEGY")
    print("-" * 40)
    print(f"Parameters: {final_optimal}")
    print(f"Optimal PnL: {final_pnl:.4f}")
    
    # Get detailed analysis of optimal strategy
    try:
        details = evaluator.evaluate_strategy_details(
            final_optimal['short_window'], 
            final_optimal['long_window']
        )
        
        print(f"Total trades: {details['total_trades']}")
        print(f"Trade details:")
        for trade in details['trades'][:5]:  # Show first 5 trades
            print(f"  {trade}")
        if len(details['trades']) > 5:
            print(f"  ... and {len(details['trades']) - 5} more trades")
    except Exception as e:
        print(f"Error getting strategy details: {e}")
    
    # Save results
    grid_search.save_results("artifacts/financial_trading_grid_search_coarse.csv")
    fine_grid_search.save_results("artifacts/financial_trading_grid_search_fine.csv")
    
    return {
        'coarse_results': coarse_results,
        'fine_results': fine_results,
        'final_optimal_params': final_optimal,
        'final_optimal_pnl': final_pnl
    }


if __name__ == "__main__":
    # Ensure artifacts directory exists
    import os
    os.makedirs("artifacts", exist_ok=True)
    
    # Run analysis
    results = run_grid_search_analysis()