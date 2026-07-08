"""Moving Average Crossover trading strategy implementation."""

import torch
import numpy as np
from typing import Dict, List, Tuple
import pandas as pd


class MovingAverageCrossoverStrategy:
    """
    Moving Average Crossover trading strategy.
    
    Generates buy/sell signals based on crossovers between short and long moving averages.
    """
    
    def __init__(self, short_window: int, long_window: int):
        """
        Initialize the strategy with moving average parameters.
        
        Args:
            short_window: Fast moving average window (minutes)
            long_window: Slow moving average window (minutes)
        """
        if short_window >= long_window:
            raise ValueError(f"Short window ({short_window}) must be less than long window ({long_window})")
        
        self.short_window = short_window
        self.long_window = long_window
    
    def compute_moving_averages(self, prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute short and long moving averages.
        
        Args:
            prices: Array of price data
            
        Returns:
            Tuple of (short_ma, long_ma) arrays
        """
        # Convert to pandas Series for rolling window calculation
        price_series = pd.Series(prices)
        
        short_ma = price_series.rolling(window=self.short_window, min_periods=self.short_window).mean().values
        long_ma = price_series.rolling(window=self.long_window, min_periods=self.long_window).mean().values
        
        return short_ma, long_ma
    
    def generate_signals(self, prices: np.ndarray) -> np.ndarray:
        """
        Generate trading signals based on moving average crossovers.
        
        Args:
            prices: Array of price data
            
        Returns:
            Array of signals: 1 for BUY, -1 for SELL, 0 for HOLD
        """
        short_ma, long_ma = self.compute_moving_averages(prices)
        
        signals = np.zeros_like(prices)
        
        # Find valid indices (where both MAs are computed)
        valid_start = self.long_window - 1
        if len(prices) <= valid_start:
            return signals
        
        # Detect crossovers
        for i in range(valid_start + 1, len(prices)):
            if np.isnan(short_ma[i]) or np.isnan(long_ma[i]):
                continue
            if np.isnan(short_ma[i-1]) or np.isnan(long_ma[i-1]):
                continue
                
            # Golden Cross: short MA crosses above long MA (BUY signal)
            if short_ma[i-1] <= long_ma[i-1] and short_ma[i] > long_ma[i]:
                signals[i] = 1
            # Death Cross: short MA crosses below long MA (SELL signal)  
            elif short_ma[i-1] >= long_ma[i-1] and short_ma[i] < long_ma[i]:
                signals[i] = -1
        
        return signals
    
    def backtest(self, prices: np.ndarray, timestamps=None) -> Dict:
        """
        Backtest the strategy on price data.
        
        Args:
            prices: Array of price data
            timestamps: Optional timestamps for tracking
            
        Returns:
            Dictionary with backtest results including PnL, trades, etc.
        """
        signals = self.generate_signals(prices)
        
        # Track trading state
        position = 0  # 0 = no position, 1 = long position
        entry_price = 0.0
        trades = []
        pnl = 0.0
        
        for i, (price, signal) in enumerate(zip(prices, signals)):
            if signal == 1 and position == 0:  # BUY signal and no position
                position = 1
                entry_price = price
                timestamp = timestamps[i] if timestamps is not None else i
                trades.append({
                    'type': 'BUY',
                    'price': price,
                    'timestamp': timestamp,
                    'index': i
                })
                
            elif signal == -1 and position == 1:  # SELL signal and have position
                position = 0
                exit_price = price
                trade_pnl = exit_price - entry_price
                pnl += trade_pnl
                
                timestamp = timestamps[i] if timestamps is not None else i
                trades.append({
                    'type': 'SELL',
                    'price': price,
                    'timestamp': timestamp,
                    'index': i,
                    'pnl': trade_pnl
                })
        
        # Close any open position at the end
        if position == 1:
            exit_price = prices[-1]
            trade_pnl = exit_price - entry_price
            pnl += trade_pnl
            
            final_timestamp = timestamps[-1] if timestamps is not None else len(prices) - 1
            trades.append({
                'type': 'SELL_EOD',
                'price': exit_price,
                'timestamp': final_timestamp,
                'index': len(prices) - 1,
                'pnl': trade_pnl
            })
        
        # Calculate additional metrics
        buy_trades = [t for t in trades if t['type'] == 'BUY']
        sell_trades = [t for t in trades if t['type'] in ['SELL', 'SELL_EOD']]
        
        return {
            'final_pnl': pnl,
            'total_trades': len(sell_trades),
            'trades': trades,
            'buy_signals': buy_trades,
            'sell_signals': sell_trades,
            'short_window': self.short_window,
            'long_window': self.long_window,
            'signals': signals
        }
    
    def __repr__(self) -> str:
        return f"MovingAverageCrossoverStrategy(short={self.short_window}, long={self.long_window})"


def test_strategy():
    """Test the strategy with synthetic data."""
    # Create synthetic price data with a trend
    np.random.seed(42)
    n_points = 100
    trend = np.linspace(100, 110, n_points)
    noise = np.random.normal(0, 1, n_points)
    prices = trend + noise
    
    # Add a flash crash in the middle
    crash_start = 45
    crash_end = 50
    prices[crash_start:crash_end] -= 10  # Sharp drop
    prices[crash_end:crash_end+3] += 10  # Quick recovery
    
    strategy = MovingAverageCrossoverStrategy(short_window=5, long_window=20)
    results = strategy.backtest(prices)
    
    print(f"Strategy: {strategy}")
    print(f"Final PnL: {results['final_pnl']:.4f}")
    print(f"Total trades: {results['total_trades']}")
    print(f"Trades: {len(results['trades'])}")
    
    for trade in results['trades']:
        print(f"  {trade}")


if __name__ == "__main__":
    test_strategy()