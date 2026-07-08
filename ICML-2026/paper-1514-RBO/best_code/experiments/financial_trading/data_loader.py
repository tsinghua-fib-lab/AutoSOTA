"""Data loader for historical DJIA data with focus on April 17, 2013 flash crash."""

import pandas as pd
import numpy as np
from datetime import datetime, time
from typing import Optional, Tuple
import os


class DJIADataLoader:
    """
    Loads and processes DJIA historical data for the flash crash experiment.
    """
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize data loader.
        
        Args:
            data_path: Path to CSV file with DJIA data. If None, will look for default locations.
        """
        self.data_path = data_path
        self.data = None
        
    def load_flash_crash_data(self, date_str: str = "2013-04-17") -> pd.DataFrame:
        """
        Load DJIA data for the flash crash day.
        
        Args:
            date_str: Date string in YYYY-MM-DD format
            
        Returns:
            DataFrame with timestamp and price columns
        """
        if self.data_path and os.path.exists(self.data_path):
            # Load from actual data file
            df = pd.read_csv(self.data_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Filter for specific date
            target_date = pd.to_datetime(date_str).date()
            df = df[df['timestamp'].dt.date == target_date].copy()
            
            if len(df) == 0:
                raise ValueError(f"No data found for date {date_str}")
                
            return df.reset_index(drop=True)
        else:
            # Generate synthetic data that mimics the flash crash pattern
            print(f"Warning: No real data file found. Generating synthetic flash crash data for {date_str}")
            return self._generate_synthetic_flash_crash_data()
    
    def _generate_synthetic_flash_crash_data(self) -> pd.DataFrame:
        """
        Generate synthetic DJIA data that mimics the April 17, 2013 flash crash pattern.
        
        Returns:
            DataFrame with timestamp and price columns
        """
        # Trading hours: 9:30 AM to 4:00 PM EST = 390 minutes
        trading_minutes = 390
        
        # Generate timestamps (minute-by-minute)
        base_date = datetime(2013, 4, 17, 9, 30)  # 9:30 AM
        timestamps = pd.date_range(base_date, periods=trading_minutes, freq='1min')
        
        # Generate base price movement (slight upward trend with noise)
        np.random.seed(42)  # For reproducibility
        base_price = 14700  # DJIA was around this level in April 2013
        
        # Create gradual trend with noise
        trend = np.linspace(0, 50, trading_minutes)  # Slight upward trend
        noise = np.random.normal(0, 10, trading_minutes)  # Market noise
        prices = base_price + trend + noise
        
        # Add the flash crash around 1:07 PM (timing based on historical event)
        crash_time_minutes = int(3.5 * 60)  # About 3.5 hours after market open
        flash_crash_start = crash_time_minutes
        flash_crash_duration = 5  # 5-minute crash
        recovery_duration = 3    # 3-minute recovery
        
        # Create the flash crash pattern
        crash_magnitude = 150  # Points drop
        for i in range(flash_crash_start, flash_crash_start + flash_crash_duration):
            if i < len(prices):
                # Exponential decay for crash
                crash_factor = np.exp(-2 * (i - flash_crash_start) / flash_crash_duration)
                prices[i] -= crash_magnitude * crash_factor
        
        # Quick recovery
        for i in range(flash_crash_start + flash_crash_duration, 
                      min(flash_crash_start + flash_crash_duration + recovery_duration, len(prices))):
            recovery_factor = (i - flash_crash_start - flash_crash_duration) / recovery_duration
            prices[i] += crash_magnitude * 0.7 * recovery_factor  # Partial recovery
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': timestamps,
            'price': prices
        })
        
        self.data = df
        return df
    
    def get_trading_hours_data(self, df: pd.DataFrame, 
                              start_time: time = time(9, 30),
                              end_time: time = time(16, 0)) -> pd.DataFrame:
        """
        Filter data to trading hours only.
        
        Args:
            df: DataFrame with timestamp column
            start_time: Market open time
            end_time: Market close time
            
        Returns:
            Filtered DataFrame
        """
        mask = (df['timestamp'].dt.time >= start_time) & (df['timestamp'].dt.time <= end_time)
        return df[mask].copy()
    
    def detect_flash_crash(self, df: pd.DataFrame, 
                          threshold_pct: float = 0.5) -> Tuple[Optional[int], Optional[int]]:
        """
        Detect the flash crash period in the data.
        
        Args:
            df: DataFrame with price column
            threshold_pct: Percentage drop threshold to identify crash
            
        Returns:
            Tuple of (crash_start_index, crash_end_index) or (None, None) if not found
        """
        prices = df['price'].values
        
        # Look for rapid drops
        returns = np.diff(prices) / prices[:-1] * 100  # Percentage returns
        
        # Find periods of significant drops
        drop_threshold = -threshold_pct  # Negative for drops
        significant_drops = np.where(returns < drop_threshold)[0]
        
        if len(significant_drops) == 0:
            return None, None
        
        # Find the most significant continuous drop period
        crash_start = significant_drops[0]
        crash_end = crash_start
        
        # Extend to find the full crash period
        min_price_idx = np.argmin(prices[crash_start:crash_start + 20]) + crash_start
        crash_end = min(min_price_idx + 5, len(prices) - 1)  # Include recovery
        
        return int(crash_start), int(crash_end)
    
    def get_crash_statistics(self, df: pd.DataFrame) -> dict:
        """
        Calculate statistics about the flash crash.
        
        Args:
            df: DataFrame with price and timestamp columns
            
        Returns:
            Dictionary with crash statistics
        """
        crash_start, crash_end = self.detect_flash_crash(df)
        
        if crash_start is None:
            return {"crash_detected": False}
        
        prices = df['price'].values
        pre_crash_price = prices[crash_start]
        min_crash_price = np.min(prices[crash_start:crash_end+1])
        post_crash_price = prices[min(crash_end + 5, len(prices) - 1)]
        
        crash_magnitude = pre_crash_price - min_crash_price
        crash_pct = (crash_magnitude / pre_crash_price) * 100
        
        recovery_amount = post_crash_price - min_crash_price
        recovery_pct = (recovery_amount / crash_magnitude) * 100 if crash_magnitude > 0 else 0
        
        return {
            "crash_detected": True,
            "crash_start_idx": crash_start,
            "crash_end_idx": crash_end,
            "pre_crash_price": pre_crash_price,
            "min_crash_price": min_crash_price,
            "post_crash_price": post_crash_price,
            "crash_magnitude": crash_magnitude,
            "crash_percentage": crash_pct,
            "recovery_amount": recovery_amount,
            "recovery_percentage": recovery_pct,
            "crash_timestamp": df.iloc[crash_start]['timestamp'],
            "min_price_timestamp": df.iloc[np.argmin(prices[crash_start:crash_end+1]) + crash_start]['timestamp']
        }


def test_data_loader():
    """Test the data loader with synthetic data."""
    loader = DJIADataLoader()
    
    # Load flash crash data
    df = loader.load_flash_crash_data()
    print(f"Loaded {len(df)} data points")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Price range: ${df['price'].min():.2f} to ${df['price'].max():.2f}")
    
    # Get crash statistics
    stats = loader.get_crash_statistics(df)
    print(f"\nFlash crash statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    return df


if __name__ == "__main__":
    test_data_loader()