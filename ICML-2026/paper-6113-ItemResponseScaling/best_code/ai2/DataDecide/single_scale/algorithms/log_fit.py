import numpy as np
from utils import RANDOM_BASELINES
from scipy.optimize import curve_fit

class LogFit:
    """
    Ian's fit method that inputs values and computes a mean fit and std around the mean.
    """
    def __init__(self, threshold=1.0):
        self.threshold = threshold

    def fit(self, values, baseline_agg=None):
        """Fit the data and return mean_fit and std arrays."""
        if baseline_agg is None:
            baseline_agg = (sum(RANDOM_BASELINES.values())/len(RANDOM_BASELINES))

        mean_fit = self._apply_log_fit(values, initial_guest=baseline_agg)
        std = self._apply_mse_around_mean_with_ordinary_mean(values)
        return mean_fit, std

    def target_keep_fn(self, primary_mean1_latest, primary_mean2_latest,
                       primary_std1_latest, primary_std2_latest, threshold=None) -> bool:
        """Determines whether two series are well-differentiated."""
        if threshold is None:
            threshold = self.threshold
        mean_diff = abs(primary_mean1_latest - primary_mean2_latest)
        sum_std = primary_std1_latest + primary_std2_latest
        ratio = mean_diff / sum_std
        return ratio > threshold

    def _apply_log_fit(self, values, initial_guest):
        def log_f(x, a, b, c, baseline):
            # Model definition
            _d = baseline - a * np.log2(c)
            return a * np.log2(b * x + c) + _d

        x = np.arange(1, len(values) + 1)
        y = np.array(values)

        p0_log = [1, 1, 1, initial_guest]
        bounds = ([0, 0, 0, 0], [np.inf, np.inf, np.inf, 1])

        popt, _ = curve_fit(log_f, x, y, p0=p0_log, bounds=bounds, maxfev=100000)
        a, b, c, baseline = popt
        transformed_values = log_f(x, a, b, c, baseline)
        return transformed_values

    def _apply_mse_around_mean_with_ordinary_mean(self, data, window=10):
        data = np.array(data)
        n = len(data)
        mse_values = np.zeros(n)
        # Compute a simple moving average and MSE around it
        for i in range(n):
            start_index = max(0, i - window // 2)
            end_index = min(n, i + window // 2 + 1)
            window_data = data[start_index:end_index]
            mean_value = np.mean(window_data)
            mse = np.mean((window_data - mean_value) ** 2)
            mse_values[i] = mse
        std = np.sqrt(mse_values)
        return std
