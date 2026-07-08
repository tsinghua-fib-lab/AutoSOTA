"""Custom evaluator for Twitter Flash Crash experiment.

This evaluator implements the core experimental logic: optimizing on in-sample (corrupted)
data while tracking out-of-sample (true) performance. The in-sample data contains the
Twitter flash crash, while out-of-sample data represents clean market conditions.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from functools import lru_cache
from bo_framework.base.evaluator import BaseEvaluator
from bo_framework.base.evaluation_result import EvaluationResult
from experiments.twitter_crash.backtester import run_backtest, analyze_backtest_results, _calculate_ema
import logging

logger = logging.getLogger(__name__)


class TwitterCrashEvaluator(BaseEvaluator):
    """Evaluator for Twitter crash financial optimization experiment.
    
    This evaluator:
    1. Evaluates strategy parameters on in-sample data (April 2013 with crash)
    2. Simultaneously evaluates on out-of-sample data (May-Dec 2013, clean)
    3. Returns IS performance as y_observed (for optimization)
    4. Returns OOS performance as y_true (for evaluation)
    
    The key insight is that robust methods should find parameters that perform
    well out-of-sample despite being optimized on corrupted in-sample data.
    """
    
    def __init__(
        self, 
        data_in_sample: pd.DataFrame, 
        data_out_of_sample: pd.DataFrame,
        w_fast_range: Tuple[int, int] = (5, 100),
        w_slow_range: Tuple[int, int] = (101, 390),
        verbose: bool = False
    ):
        """Initialize the Twitter crash evaluator.
        
        Args:
            data_in_sample: In-sample data (April 2013, contains crash)
            data_out_of_sample: Out-of-sample data (May-Dec 2013, clean)
            w_fast_range: Range of fast EMA windows for pre-computation
            w_slow_range: Range of slow EMA windows for pre-computation
            verbose: If True, print evaluation details
        """
        self.data_in_sample = data_in_sample
        self.data_out_of_sample = data_out_of_sample
        self.verbose = verbose
        
        # Optimization 1: Pre-compute EMAs
        if verbose:
            print("Pre-computing EMAs...")
        self.ema_cache_is = self._precompute_emas(data_in_sample, w_fast_range, w_slow_range)
        self.ema_cache_oos = self._precompute_emas(data_out_of_sample, w_fast_range, w_slow_range)
        if verbose:
            print(f"EMA Pre-computation complete. Cached {len(self.ema_cache_is)} EMA series.")
        
        # Initialize internal state
        self.reset()
    
    @property
    def is_deterministic(self) -> bool:
        """Backtesting is deterministic."""
        return True
    
    def _precompute_emas(self, data: pd.DataFrame, w_fast_range: Tuple[int, int], w_slow_range: Tuple[int, int]) -> Dict[int, pd.Series]:
        """Pre-computes all required EMA windows for the given data.
        
        Args:
            data: Price data DataFrame
            w_fast_range: Range of fast EMA windows (min, max)
            w_slow_range: Range of slow EMA windows (min, max)
            
        Returns:
            Dictionary mapping window sizes to EMA Series
        """
        ema_cache = {}
        # Combine ranges and find unique windows
        windows = list(range(w_fast_range[0], w_fast_range[1] + 1)) + \
                  list(range(w_slow_range[0], w_slow_range[1] + 1))
        
        prices = data['close']
        for window in set(windows):
            # Use the optimized Pandas function from backtester
            ema_cache[window] = _calculate_ema(prices, window)
        return ema_cache
    
    @lru_cache(maxsize=None)  # Unlimited cache size for deterministic function
    def _cached_evaluate(self, w_fast: int, w_slow: int, s_l: float) -> Tuple[float, float]:
        """Cached evaluation of the backtest for specific parameters.
        
        This method is cached using lru_cache to avoid redundant backtesting
        for the same parameter combinations.
        
        Args:
            w_fast: Fast EMA window (integer)
            w_slow: Slow EMA window (integer) 
            s_l: Stop-loss percentage (float, rounded for consistent cache keys)
            
        Returns:
            Tuple of (in_sample_sharpe, out_of_sample_sharpe)
        """
        params = {"W_Fast": w_fast, "W_Slow": w_slow, "S_L": s_l}

        # Evaluate on in-sample data (passing the pre-computed cache)
        # run_backtest now uses Numba internally for stop-loss calculation
        sharpe_in = run_backtest(self.data_in_sample, params, self.ema_cache_is)
        
        # Evaluate on out-of-sample data (passing the pre-computed cache)
        sharpe_out = run_backtest(self.data_out_of_sample, params, self.ema_cache_oos)
        
        return sharpe_in, sharpe_out
        
    def evaluate(self, params: Dict[str, Any]) -> EvaluationResult:
        """Evaluate trading strategy parameters on both IS and OOS data.
        
        Args:
            params: Dictionary with strategy parameters:
                - W_Fast: Fast EMA window (will be converted to integer)
                - W_Slow: Slow EMA window (will be converted to integer)
                - S_L: Stop-loss percentage (float)
                
        Returns:
            EvaluationResult with:
                - y_observed: In-sample Sharpe ratio (what BO optimizes)
                - y_true: Out-of-sample Sharpe ratio (true performance)
                - y_noisy: Same as y_observed (no additional noise)
                - noise: 0.0 (deterministic evaluation)
                - corruption: Implicit in IS/OOS difference
        """
        self.evaluation_count += 1
        
        # Type conversion and rounding for the caching key
        w_fast = int(round(params["W_Fast"]))
        w_slow = int(round(params["W_Slow"]))
        # Round S_L to ensure consistent caching keys (5 decimal places)
        s_l = round(params["S_L"], 5)

        # Call the cached evaluation logic (this is where the magic happens!)
        sharpe_in, sharpe_out = self._cached_evaluate(w_fast, w_slow, s_l)
        
        # Create typed_params for history tracking
        typed_params = {"W_Fast": w_fast, "W_Slow": w_slow, "S_L": s_l}
        
        # Track best configurations
        if sharpe_in > self.best_is_sharpe:
            self.best_is_sharpe = sharpe_in
            self.best_is_params = typed_params.copy()
            
        if sharpe_out > self.best_oos_sharpe:
            self.best_oos_sharpe = sharpe_out
            self.best_oos_params = typed_params.copy()
        
        # Store in history
        self.history.append({
            'iteration': self.evaluation_count,
            'params': typed_params.copy(),
            'sharpe_is': sharpe_in,
            'sharpe_oos': sharpe_out,
            'gap': sharpe_in - sharpe_out  # Overfitting indicator
        })
        
        if self.verbose:
            print(f"Eval #{self.evaluation_count}: "
                  f"W_Fast={typed_params['W_Fast']}, "
                  f"W_Slow={typed_params['W_Slow']}, "
                  f"S_L={typed_params['S_L']:.3f} | "
                  f"IS Sharpe={sharpe_in:.3f}, "
                  f"OOS Sharpe={sharpe_out:.3f}, "
                  f"Gap={sharpe_in - sharpe_out:.3f}")
        
        # Construct evaluation result
        # CRITICAL: Map OOS -> y_true, IS -> y_observed
        return EvaluationResult(
            x=params,  # Use original params (not typed)
            y_true=sharpe_out,  # True performance (OOS)
            y_noisy=sharpe_in,  # Same as observed (no additional noise)
            y_observed=sharpe_in,  # What the optimizer sees (IS)
            noise=0.0,  # No observation noise
            corruption=sharpe_in - sharpe_out  # Explicitly define corruption as the IS/OOS performance gap
        )
    
    def reset(self):
        """Resets the evaluator's history and statistics for a new run."""
        if self.verbose:
            logger.info("Resetting evaluator state (history and counters).")
        self.evaluation_count = 0
        self.history = []
        self.best_is_params = None
        self.best_is_sharpe = -np.inf
        self.best_oos_params = None
        self.best_oos_sharpe = -np.inf
        # Note: The lru_cache on _cached_evaluate is NOT cleared.
        # This is a deliberate choice to speed up trials, as function
        # evaluations are deterministic and can be shared between methods.
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics of the evaluation history.
        
        Returns:
            Dictionary with evaluation statistics
        """
        if not self.history:
            return {}
        
        gaps = [h['gap'] for h in self.history]
        is_sharpes = [h['sharpe_is'] for h in self.history]
        oos_sharpes = [h['sharpe_oos'] for h in self.history]
        
        return {
            'n_evaluations': self.evaluation_count,
            'best_is_sharpe': self.best_is_sharpe,
            'best_oos_sharpe': self.best_oos_sharpe,
            'best_is_params': self.best_is_params,
            'best_oos_params': self.best_oos_params,
            'mean_gap': np.mean(gaps),
            'std_gap': np.std(gaps),
            'correlation_is_oos': np.corrcoef(is_sharpes, oos_sharpes)[0, 1] if len(is_sharpes) > 1 else 0
        }
    
    def analyze_crash_impact(self) -> Dict[str, Any]:
        """Analyze how the crash affects different parameter configurations.
        
        Returns:
            Dictionary with crash impact analysis
        """
        if not self.history:
            return {}
        
        # Group by stop-loss levels
        sl_impact = {}
        for h in self.history:
            sl = h['params']['S_L']
            sl_bucket = round(sl, 3)  # Round to nearest 0.1%
            
            if sl_bucket not in sl_impact:
                sl_impact[sl_bucket] = {
                    'is_sharpes': [],
                    'oos_sharpes': [],
                    'gaps': []
                }
            
            sl_impact[sl_bucket]['is_sharpes'].append(h['sharpe_is'])
            sl_impact[sl_bucket]['oos_sharpes'].append(h['sharpe_oos'])
            sl_impact[sl_bucket]['gaps'].append(h['gap'])
        
        # Calculate averages
        sl_analysis = {}
        for sl, data in sl_impact.items():
            sl_analysis[sl] = {
                'mean_is_sharpe': np.mean(data['is_sharpes']),
                'mean_oos_sharpe': np.mean(data['oos_sharpes']),
                'mean_gap': np.mean(data['gaps']),
                'n_samples': len(data['is_sharpes'])
            }
        
        return {
            'stop_loss_impact': sl_analysis,
            'most_robust_sl': min(sl_analysis, key=lambda x: abs(sl_analysis[x]['mean_gap']))
            if sl_analysis else None
        }
    
    def get_detailed_analysis(
        self, 
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Get detailed backtest analysis for specific parameters.
        
        Args:
            params: Strategy parameters to analyze. If None, uses best OOS params.
            
        Returns:
            Dictionary with detailed metrics for both IS and OOS periods
        """
        if params is None:
            params = self.best_oos_params
            if params is None:
                return {}
        
        # Ensure proper typing
        typed_params = params.copy()
        typed_params["W_Fast"] = int(round(typed_params["W_Fast"]))
        typed_params["W_Slow"] = int(round(typed_params["W_Slow"]))
        
        # Get detailed analysis for both periods
        is_analysis = analyze_backtest_results(self.data_in_sample, typed_params)
        oos_analysis = analyze_backtest_results(self.data_out_of_sample, typed_params)
        
        return {
            'parameters': typed_params,
            'in_sample': is_analysis,
            'out_of_sample': oos_analysis,
            'overfitting_ratio': (is_analysis['sharpe_ratio'] / oos_analysis['sharpe_ratio'] 
                                 if oos_analysis['sharpe_ratio'] != 0 else np.inf)
        }