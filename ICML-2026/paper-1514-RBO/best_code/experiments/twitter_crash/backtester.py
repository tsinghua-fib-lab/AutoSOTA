"""Backtesting engine for EMA Crossover trading strategy.

This module implements a vectorized backtesting engine for the Exponential Moving Average
(EMA) crossover strategy with stop-loss. The strategy generates trading signals based on
fast and slow EMA crossovers and includes risk management via stop-loss orders.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
import warnings
import numba as nb


@nb.jit(nopython=True)
def _apply_stop_loss_numba(close_prices: np.ndarray, low_prices: np.ndarray, signals: np.ndarray, stop_loss: float) -> np.ndarray:
    """Apply stop-loss logic using Numba acceleration.
    
    This function replaces the slow Python loop with compiled machine code.
    
    Args:
        close_prices: Array of closing prices
        low_prices: Array of low prices  
        signals: Array of initial trading signals (1=long, 0=flat)
        stop_loss: Stop-loss percentage (e.g., 0.01 for 1%)
        
    Returns:
        Modified signals array with stop-loss exits applied
    """
    signals_modified = signals.copy()
    position = 0
    entry_price = 0.0
    n = len(signals)
    
    for i in range(n):
        current_signal = signals[i]
        
        if position == 0:
            if current_signal == 1:
                # Enter long position
                position = 1
                # Assume entry at the close of the signal bar
                entry_price = close_prices[i]
        
        elif position == 1:
            # Check stop-loss condition using the low price of the current bar
            current_low = low_prices[i]
            stop_price = entry_price * (1 - stop_loss)
            
            if current_low <= stop_price:
                # Stop-loss triggered
                signals_modified[i] = 0
                position = 0
                # Reset entry_price (optimization: not strictly needed but clear)
                entry_price = 0.0
            elif current_signal == 0:
                # Regular exit signal (EMA crossover reversal)
                position = 0
                entry_price = 0.0
            # else: Maintain position (signals_modified[i] is already correct from copy)

    return signals_modified


def run_backtest(data: pd.DataFrame, params: Dict[str, Any], ema_cache: Optional[Dict[int, pd.Series]] = None) -> float:
    """Run backtest of EMA Crossover strategy with stop-loss (Optimized).
    
    The strategy:
    - Goes long when fast EMA crosses above slow EMA
    - Goes flat (exits) when fast EMA crosses below slow EMA
    - Exits immediately if price drops below stop-loss threshold
    
    Args:
        data: DataFrame with OHLCV data (columns: open, high, low, close, volume)
        params: Dictionary with strategy parameters:
            - W_Fast: Fast EMA window (integer, minutes)
            - W_Slow: Slow EMA window (integer, minutes)  
            - S_L: Stop-loss percentage (float, e.g., 0.01 for 1%)
        ema_cache: Optional cache of pre-computed EMAs for faster lookup
            
    Returns:
        Sharpe ratio of the strategy over the data period
        Returns -10.0 if parameters are invalid (W_Slow <= W_Fast)
    """
    # Extract and validate parameters
    w_fast = int(round(params.get('W_Fast', 20)))
    w_slow = int(round(params.get('W_Slow', 50)))
    stop_loss = float(params.get('S_L', 0.01))
    
    # Validate parameter constraint
    if w_slow <= w_fast:
        return -10.0  # Heavy penalty for invalid parameters
    
    # Check if we have enough data
    if len(data) < w_slow + 1:
        warnings.warn(f"Insufficient data: {len(data)} points, need at least {w_slow + 1}")
        return -10.0
    
    # Calculate EMAs (Optimized: Use Cache if available)
    if ema_cache:
        try:
            # Ensure the index aligns (important if data subsetting occurred)
            ema_fast = ema_cache[w_fast].loc[data.index]
            ema_slow = ema_cache[w_slow].loc[data.index]
        except (KeyError, KeyError):
            warnings.warn(f"Required EMAs ({w_fast}, {w_slow}) not found in cache. Calculating on the fly.")
            ema_fast = _calculate_ema(data['close'], w_fast)
            ema_slow = _calculate_ema(data['close'], w_slow)
    else:
        # Fallback to on-the-fly calculation if no cache provided
        ema_fast = _calculate_ema(data['close'], w_fast)
        ema_slow = _calculate_ema(data['close'], w_slow)
    
    # Generate trading signals (Vectorized Pandas - already efficient)
    signals = _generate_signals(ema_fast, ema_slow)
    
    # Apply stop-loss logic (Optimized: Numba)
    # Convert to NumPy arrays for Numba
    close_np = data['close'].to_numpy(dtype=np.float64)
    low_np = data['low'].to_numpy(dtype=np.float64)
    # Use int32 for signals for slight memory saving
    signals_np = signals.to_numpy(dtype=np.int32)

    signals_modified_np = _apply_stop_loss_numba(close_np, low_np, signals_np, stop_loss)
    
    # Convert back to Pandas Series
    signals_modified = pd.Series(signals_modified_np, index=data.index, dtype=signals.dtype)

    # Calculate returns and Sharpe ratio
    returns = _calculate_returns(data, signals_modified)
    sharpe = _calculate_sharpe_ratio(returns)
    
    return sharpe


def _calculate_ema(prices: pd.Series, window: int) -> pd.Series:
    """Calculate Exponential Moving Average.
    
    Args:
        prices: Price series
        window: EMA window period
        
    Returns:
        EMA series
    """
    return prices.ewm(span=window, adjust=False, min_periods=window).mean()


def _generate_signals(ema_fast: pd.Series, ema_slow: pd.Series) -> pd.Series:
    """Generate trading signals based on EMA crossovers.
    
    Args:
        ema_fast: Fast EMA series
        ema_slow: Slow EMA series
        
    Returns:
        Signal series: 1 for long, 0 for flat
    """
    # Initialize signals
    signals = pd.Series(0, index=ema_fast.index)
    
    # Skip NaN values at the beginning
    valid_idx = ~(ema_fast.isna() | ema_slow.isna())
    
    # Calculate crossovers on valid data
    fast_above_slow = ema_fast > ema_slow
    
    # Generate signals: 1 when fast > slow, 0 otherwise
    signals[valid_idx] = fast_above_slow[valid_idx].astype(int)
    
    return signals



def _calculate_returns(data: pd.DataFrame, signals: pd.Series) -> pd.Series:
    """Calculate strategy returns with transaction costs.
    
    Args:
        data: OHLCV data
        signals: Trading signals (1 for long, 0 for flat)
        
    Returns:
        Strategy returns series
    """
    # Calculate price returns
    price_returns = data['close'].pct_change().fillna(0)
    
    # Calculate position changes (for transaction cost calculation)
    position_changes = signals.diff().abs().fillna(0)
    
    # Transaction cost (10 basis points per trade)
    transaction_cost = 0.001
    costs = position_changes * transaction_cost
    
    # Strategy returns: price returns when in position, minus transaction costs
    strategy_returns = (signals.shift(1) * price_returns - costs).fillna(0)
    
    return strategy_returns


def _calculate_sharpe_ratio(returns: pd.Series, periods_per_year: int = 252 * 390) -> float:
    """Calculate annualized Sharpe ratio.
    
    Args:
        returns: Returns series
        periods_per_year: Number of periods in a year (252 days * 390 minutes)
        
    Returns:
        Annualized Sharpe ratio
    """
    if len(returns) < 2:
        return -10.0
    
    # Remove any NaN or infinite values
    clean_returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(clean_returns) < 2:
        return -10.0
    
    # Calculate statistics
    mean_return = clean_returns.mean()
    std_return = clean_returns.std()
    
    if std_return == 0 or np.isnan(std_return):
        return 0.0 if mean_return >= 0 else -10.0
    
    # Annualize Sharpe ratio
    # Assuming 252 trading days and 390 minutes per day
    sharpe = mean_return / std_return * np.sqrt(periods_per_year)
    
    # Cap Sharpe ratio to reasonable bounds
    sharpe = np.clip(sharpe, -10.0, 10.0)
    
    return float(sharpe)


def analyze_backtest_results(
    data: pd.DataFrame, 
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """Run detailed backtest analysis and return comprehensive metrics.
    
    Args:
        data: OHLCV data
        params: Strategy parameters
        
    Returns:
        Dictionary with detailed backtest metrics
    """
    # Run basic backtest
    sharpe = run_backtest(data, params)
    
    # Get additional metrics
    w_fast = int(round(params.get('W_Fast', 20)))
    w_slow = int(round(params.get('W_Slow', 50)))
    stop_loss = float(params.get('S_L', 0.01))
    
    # Calculate EMAs and signals for detailed analysis
    ema_fast = _calculate_ema(data['close'], w_fast)
    ema_slow = _calculate_ema(data['close'], w_slow)
    signals = _generate_signals(ema_fast, ema_slow)
    
    # Apply stop-loss using Numba version
    close_np = data['close'].to_numpy(dtype=np.float64)
    low_np = data['low'].to_numpy(dtype=np.float64)
    signals_np = signals.to_numpy(dtype=np.int32)
    signals_modified_np = _apply_stop_loss_numba(close_np, low_np, signals_np, stop_loss)
    signals = pd.Series(signals_modified_np, index=data.index, dtype=signals.dtype)
    returns = _calculate_returns(data, signals)
    
    # Calculate additional metrics
    total_return = (1 + returns).prod() - 1
    n_trades = signals.diff().abs().sum() / 2  # Number of round trips
    win_rate = (returns[returns > 0].count() / returns[returns != 0].count() 
                if returns[returns != 0].count() > 0 else 0)
    
    # Check if strategy was affected by crash (if present)
    min_return = returns.min()
    max_drawdown = _calculate_max_drawdown(returns)
    
    return {
        'sharpe_ratio': sharpe,
        'total_return': total_return,
        'n_trades': int(n_trades),
        'win_rate': win_rate,
        'min_return': min_return,
        'max_drawdown': max_drawdown,
        'params': params
    }


def _calculate_max_drawdown(returns: pd.Series) -> float:
    """Calculate maximum drawdown from returns series.
    
    Args:
        returns: Returns series
        
    Returns:
        Maximum drawdown (negative value)
    """
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return float(drawdown.min())