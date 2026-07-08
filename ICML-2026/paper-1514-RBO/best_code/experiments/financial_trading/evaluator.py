"""Trading strategy evaluator for Bayesian Optimization."""

import torch
import numpy as np
from typing import Dict, Any, Union
from bo_framework.base.evaluator import BaseEvaluator
from bo_framework.base.evaluation_result import EvaluationResult
from .data_loader import DJIADataLoader
from .trading_strategy import MovingAverageCrossoverStrategy


class TradingStrategyEvaluator(BaseEvaluator):
    """
    Evaluator for trading strategy optimization using historical DJIA data.
    
    This evaluator takes moving average parameters and returns the PnL from backtesting
    the strategy on the April 17, 2013 flash crash data.
    """
    
    def __init__(self, data_path: str = None, cache_data: bool = True):
        """
        Initialize the trading strategy evaluator.
        
        Args:
            data_path: Path to DJIA data CSV file
            cache_data: Whether to cache loaded data
        """
        super().__init__()
        
        self.data_loader = DJIADataLoader(data_path)
        self.cache_data = cache_data
        self._cached_data = None
        self._crash_stats = None
        
        # Load data once if caching enabled
        if self.cache_data:
            self._load_and_cache_data()
    
    def _load_and_cache_data(self):
        """Load and cache the flash crash data."""
        try:
            self._cached_data = self.data_loader.load_flash_crash_data()
            self._crash_stats = self.data_loader.get_crash_statistics(self._cached_data)
            print(f"Cached DJIA data: {len(self._cached_data)} points")
            if self._crash_stats.get('crash_detected', False):
                print(f"Flash crash detected at {self._crash_stats['crash_timestamp']}")
                print(f"Crash magnitude: {self._crash_stats['crash_percentage']:.2f}%")
        except Exception as e:
            print(f"Warning: Failed to load data during initialization: {e}")
            self._cached_data = None
            self._crash_stats = None
    
    def evaluate(self, params: Union[Dict[str, Any], torch.Tensor]) -> EvaluationResult:
        """
        Evaluate trading strategy with given parameters.
        
        Args:
            params: Dictionary with 'short_window' and 'long_window' keys,
                   or tensor with [short_window, long_window]
                   
        Returns:
            EvaluationResult with PnL as objective value
        """
        # Handle both dict and tensor inputs
        if isinstance(params, torch.Tensor):
            # Assume tensor is [short_window, long_window]
            params_dict = {
                'short_window': int(params[0].item()),
                'long_window': int(params[1].item())
            }
        else:
            params_dict = params.copy()
        
        # Extract parameters
        short_window = int(params_dict['short_window'])
        long_window = int(params_dict['long_window'])
        
        # Validate parameters
        if short_window >= long_window:
            # Invalid parameters - return penalty
            return EvaluationResult.from_true_value(params, -1000.0)
        
        if short_window < 1 or long_window < 2:
            # Invalid parameters - return penalty
            return EvaluationResult.from_true_value(params, -1000.0)
        
        try:
            # Get data
            data = self._get_data()
            
            # Create and run strategy
            strategy = MovingAverageCrossoverStrategy(short_window, long_window)
            results = strategy.backtest(
                prices=data['price'].values,
                timestamps=data['timestamp'].values
            )
            
            # Return PnL as objective value (BO maximizes)
            pnl = results['final_pnl']
            
            return EvaluationResult.from_true_value(params, pnl)
            
        except Exception as e:
            print(f"Error evaluating strategy {params_dict}: {e}")
            # Return penalty for failed evaluations
            return EvaluationResult.from_true_value(params, -1000.0)
    
    def _get_data(self):
        """Get the data (from cache or load fresh)."""
        if self.cache_data and self._cached_data is not None:
            return self._cached_data
        else:
            return self.data_loader.load_flash_crash_data()
    
    def get_crash_statistics(self) -> dict:
        """Get flash crash statistics."""
        if self._crash_stats is not None:
            return self._crash_stats
        else:
            data = self._get_data()
            return self.data_loader.get_crash_statistics(data)
    
    def evaluate_strategy_details(self, short_window: int, long_window: int) -> dict:
        """
        Get detailed strategy evaluation results for analysis.
        
        Args:
            short_window: Short moving average window
            long_window: Long moving average window
            
        Returns:
            Detailed results dictionary with trades, signals, etc.
        """
        if short_window >= long_window:
            raise ValueError(f"short_window ({short_window}) must be < long_window ({long_window})")
        
        data = self._get_data()
        strategy = MovingAverageCrossoverStrategy(short_window, long_window)
        
        return strategy.backtest(
            prices=data['price'].values,
            timestamps=data['timestamp'].values
        )
    
    @property
    def is_deterministic(self) -> bool:
        """Trading strategy evaluation is deterministic given fixed data."""
        return True
    
    def get_search_space_bounds(self) -> Dict[str, tuple]:
        """
        Get reasonable bounds for the search space.
        
        Returns:
            Dictionary with parameter bounds
        """
        return {
            'short_window': (1, 60),    # 1 to 60 minutes
            'long_window': (10, 240)    # 10 to 240 minutes (4 hours)
        }
    
    def __repr__(self) -> str:
        data_info = f"{len(self._cached_data)} points" if self._cached_data is not None else "not loaded"
        return f"TradingStrategyEvaluator(data={data_info})"


def test_evaluator():
    """Test the evaluator with various parameter combinations."""
    evaluator = TradingStrategyEvaluator()
    
    # Test cases
    test_cases = [
        {'short_window': 5, 'long_window': 20},   # Responsive strategy
        {'short_window': 15, 'long_window': 50},  # Moderate strategy  
        {'short_window': 30, 'long_window': 120}, # Conservative strategy
        {'short_window': 1, 'long_window': 10},   # Very responsive (likely to get whipsawed)
        {'short_window': 50, 'long_window': 200}, # Very conservative
    ]
    
    print("Testing TradingStrategyEvaluator:")
    print("=" * 60)
    
    results = []
    for params in test_cases:
        result = evaluator.evaluate(params)
        results.append((params, result.y_true))
        
        print(f"Strategy {params}: PnL = {result.y_true:.4f}")
        
        # Get detailed results for analysis
        try:
            details = evaluator.evaluate_strategy_details(
                params['short_window'], params['long_window']
            )
            print(f"  Total trades: {details['total_trades']}")
            if details['trades']:
                print(f"  First trade: {details['trades'][0]}")
                if len(details['trades']) > 1:
                    print(f"  Last trade: {details['trades'][-1]}")
        except Exception as e:
            print(f"  Error getting details: {e}")
        
        print()
    
    # Find best strategy
    best_params, best_pnl = max(results, key=lambda x: x[1])
    print(f"Best strategy: {best_params} with PnL = {best_pnl:.4f}")
    
    # Show crash statistics
    crash_stats = evaluator.get_crash_statistics()
    if crash_stats.get('crash_detected', False):
        print(f"\nFlash crash detected:")
        print(f"  Time: {crash_stats['crash_timestamp']}")
        print(f"  Magnitude: {crash_stats['crash_percentage']:.2f}%")
        print(f"  Recovery: {crash_stats['recovery_percentage']:.2f}%")


if __name__ == "__main__":
    test_evaluator()