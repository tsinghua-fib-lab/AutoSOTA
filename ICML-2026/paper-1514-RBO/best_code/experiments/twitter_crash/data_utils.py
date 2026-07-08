"""Data loading and preprocessing utilities for Twitter Flash Crash experiment.

This module handles loading and splitting SPY financial data for the experiment.
The data split includes:
- In-Sample (IS): April 2013 (contains the Twitter flash crash on April 23)
- Out-of-Sample (OOS): May 2013 - December 2013 (clean data for true performance)
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from datetime import datetime
import warnings


def load_and_prepare_data(
    data_path: Optional[str] = None,
    generate_synthetic: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and prepare financial data for the Twitter crash experiment.
    
    Args:
        data_path: Path to CSV file with SPY 1-minute OHLCV data. If None and
                  generate_synthetic is True, will generate synthetic data.
        generate_synthetic: If True and data_path is None, generate synthetic
                           data with a simulated crash event.
    
    Returns:
        Tuple of (in_sample_data, out_of_sample_data) DataFrames with columns:
        - datetime: Timestamp index
        - open, high, low, close: Price data
        - volume: Trading volume
    
    Raises:
        ValueError: If neither data_path is provided nor generate_synthetic is True
    """
    if data_path is not None:
        # Load real data from CSV
        df = pd.read_csv(data_path, parse_dates=['datetime'])
        df.set_index('datetime', inplace=True)
        
        # Ensure timezone is ET (Eastern Time)
        if df.index.tz is None:
            df.index = df.index.tz_localize('America/New_York')
        else:
            df.index = df.index.tz_convert('America/New_York')
            
        # Sort by datetime
        df.sort_index(inplace=True)
        
    elif generate_synthetic:
        # Generate synthetic data for testing purposes
        df = _generate_synthetic_spy_data()
    else:
        raise ValueError("Either provide data_path or set generate_synthetic=True")
    
    # Split data into in-sample and out-of-sample periods
    is_data, oos_data = _split_data(df)
    
    # Validate data quality
    _validate_data(is_data, "in-sample")
    _validate_data(oos_data, "out-of-sample")
    
    return is_data, oos_data


def _generate_synthetic_spy_data() -> pd.DataFrame:
    """Generate synthetic SPY data with a simulated flash crash event.
    
    Creates realistic 1-minute OHLCV data for SPY from April 2013 to December 2013,
    including a flash crash event on April 23, 2013 around 1:07 PM ET.
    
    Returns:
        DataFrame with synthetic SPY data
    """
    # Generate date range (market hours only: 9:30 AM - 4:00 PM ET)
    dates = pd.date_range(
        start='2013-04-01 09:30:00',
        end='2013-12-31 16:00:00',
        freq='1min',
        tz='America/New_York'
    )
    
    # Filter to market hours only
    dates = dates[(dates.hour >= 9) & 
                  ((dates.hour < 16) | ((dates.hour == 16) & (dates.minute == 0)))]
    dates = dates[dates.weekday < 5]  # Weekdays only
    
    # Initialize with base price around 156 (SPY level in April 2013)
    n_points = len(dates)
    base_price = 156.0
    
    # Generate base returns with slight upward drift
    returns = np.random.normal(0.00001, 0.0008, n_points)  # 1-minute returns
    
    # Add the flash crash on April 23, 2013 around 1:07 PM
    crash_datetime = pd.Timestamp('2013-04-23 13:07:00', tz='America/New_York')
    crash_idx = dates.get_indexer([crash_datetime], method='nearest')[0]
    
    # Simulate the crash: -1.5% drop in 2 minutes, then recovery
    if crash_idx < n_points:
        returns[crash_idx] = -0.008  # First minute: -0.8% drop
        returns[crash_idx + 1] = -0.007  # Second minute: -0.7% drop
        # Recovery over next 5 minutes
        for i in range(2, 7):
            if crash_idx + i < n_points:
                returns[crash_idx + i] = 0.003  # Partial recovery
    
    # Calculate prices
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Generate OHLCV data
    df = pd.DataFrame(index=dates)
    df['close'] = prices
    
    # Generate realistic OHLC from close prices
    noise_scale = 0.0003
    df['open'] = df['close'] * (1 + np.random.normal(0, noise_scale, n_points))
    df['high'] = df[['open', 'close']].max(axis=1) * (1 + np.abs(np.random.normal(0, noise_scale, n_points)))
    df['low'] = df[['open', 'close']].min(axis=1) * (1 - np.abs(np.random.normal(0, noise_scale, n_points)))
    
    # Generate volume (higher during crash)
    base_volume = 1_000_000
    df['volume'] = base_volume * (1 + np.abs(np.random.normal(0, 0.3, n_points)))
    
    # Spike volume during crash
    if crash_idx < n_points:
        for i in range(10):  # High volume for 10 minutes around crash
            if crash_idx - 5 + i >= 0 and crash_idx - 5 + i < n_points:
                df.iloc[crash_idx - 5 + i, df.columns.get_loc('volume')] *= 3
    
    return df


def _split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into in-sample (April 2013) and out-of-sample (May-Dec 2013) periods.
    
    Args:
        df: Full dataset with datetime index
        
    Returns:
        Tuple of (in_sample_data, out_of_sample_data)
    """
    # Define split dates
    is_start = pd.Timestamp('2013-04-01', tz=df.index.tz)
    is_end = pd.Timestamp('2013-04-30 23:59:59', tz=df.index.tz)
    oos_start = pd.Timestamp('2013-05-01', tz=df.index.tz)
    oos_end = pd.Timestamp('2013-12-31 23:59:59', tz=df.index.tz)
    
    # Split data
    is_data = df[(df.index >= is_start) & (df.index <= is_end)].copy()
    oos_data = df[(df.index >= oos_start) & (df.index <= oos_end)].copy()
    
    return is_data, oos_data


def _validate_data(df: pd.DataFrame, label: str) -> None:
    """Validate data quality and raise warnings for any issues.
    
    Args:
        df: DataFrame to validate
        label: Label for the data (for error messages)
        
    Raises:
        ValueError: If critical data issues are found
    """
    if df.empty:
        raise ValueError(f"No {label} data found")
    
    # Check for required columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in {label} data: {missing_cols}")
    
    # Check for NaN values
    if df[required_cols].isna().any().any():
        warnings.warn(f"{label} data contains NaN values, will forward-fill")
        df[required_cols] = df[required_cols].fillna(method='ffill')
    
    # Check for negative prices
    price_cols = ['open', 'high', 'low', 'close']
    if (df[price_cols] <= 0).any().any():
        raise ValueError(f"{label} data contains non-positive prices")
    
    # Check OHLC consistency
    invalid_ohlc = (df['high'] < df['low']).any() or \
                   (df['high'] < df['close']).any() or \
                   (df['high'] < df['open']).any() or \
                   (df['low'] > df['close']).any() or \
                   (df['low'] > df['open']).any()
    
    if invalid_ohlc:
        warnings.warn(f"{label} data has inconsistent OHLC values")
    
    print(f"✓ {label} data validated: {len(df)} records from {df.index[0]} to {df.index[-1]}")


def check_for_crash(df: pd.DataFrame, threshold: float = -0.005) -> pd.DataFrame:
    """Identify potential crash events in the data.
    
    Args:
        df: DataFrame with OHLC data
        threshold: Return threshold to identify crashes (default -1%)
        
    Returns:
        DataFrame with crash events and their characteristics
    """
    # Calculate 1-minute returns
    returns = df['close'].pct_change()
    
    # Find large negative returns
    crash_mask = returns < threshold
    crash_events = df[crash_mask].copy()
    crash_events['return'] = returns[crash_mask]
    
    if not crash_events.empty:
        print(f"Found {len(crash_events)} potential crash events:")
        print(crash_events[['close', 'return', 'volume']].head(10))
    
    return crash_events