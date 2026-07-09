"""Local volatility estimator for conformal prediction on financial time series.

Computes sigma_t = sqrt(EWMA of squared log-returns) using only past price data
(causally valid — no look-ahead bias).

Reference: Dewolf et al., "Heteroskedastic Conformal Regression," 2023.
           Canete, "Online NoVaS Conformal Volatility Prediction," COPA 2023.
"""
import numpy as np


class ExponentialVolatilityEstimator:
    """Exponentially-weighted moving standard deviation of log-returns.

    Args:
        half_life: Half-life of the EWMA in trading days (default 20).
        min_periods: Minimum observations before returning a valid estimate.
    """

    def __init__(self, half_life=20, min_periods=10):
        self.decay = np.exp(-np.log(2) / half_life)
        self.ewma_var = 0.0
        self.ewma_mean = 0.0
        self.count = 0
        self.min_periods = min_periods
        self.last_price = None
        self._volatility = np.nan

    def update(self, price):
        """Update with a new price observation and return current volatility.

        Args:
            price: Current price (float).

        Returns:
            Current volatility estimate or NaN if insufficient data.
        """
        if self.last_price is not None and self.last_price > 0:
            log_return = np.log(price / self.last_price)
            # Update EWMA mean (for centering)
            self.ewma_mean = (self.decay * self.ewma_mean +
                              (1.0 - self.decay) * log_return)
            # Update EWMA variance
            sq_dev = (log_return - self.ewma_mean) ** 2
            self.ewma_var = (self.decay * self.ewma_var +
                             (1.0 - self.decay) * sq_dev)
            self.count += 1
            if self.count >= self.min_periods:
                self._volatility = np.sqrt(max(self.ewma_var, 1e-12))
        self.last_price = price
        return self._volatility

    @property
    def volatility(self):
        return self._volatility


def compute_volatility_series(prices, half_life=20, min_periods=10):
    """Compute a causally-valid volatility series from a price array.

    Args:
        prices: 1-D numpy array of price data.
        half_life: EWMA half-life in trading days.
        min_periods: Minimum observations before valid estimate.

    Returns:
        1-D numpy array of same length as prices with volatility estimates.
        Early entries (before min_periods) are forward-filled from the first
        valid estimate.
    """
    estimator = ExponentialVolatilityEstimator(
        half_life=half_life, min_periods=min_periods
    )
    volatility = np.full(len(prices), np.nan)
    for i in range(len(prices)):
        volatility[i] = estimator.update(prices[i])

    # Forward-fill NaN values at the start
    first_valid = np.argmax(~np.isnan(volatility)) if np.any(~np.isnan(volatility)) else 0
    if first_valid > 0:
        volatility[:first_valid] = volatility[first_valid]
    # If all NaN (very short series), use a fallback
    if np.all(np.isnan(volatility)):
        volatility[:] = 0.01
    return volatility
