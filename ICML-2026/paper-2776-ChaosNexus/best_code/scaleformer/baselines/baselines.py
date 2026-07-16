from dataclasses import dataclass

import numpy as np
from scaleformer.utils import safe_standardize
from statsmodels.tsa.arima.model import ARIMA


@dataclass
class MeanBaseline:
    prediction_length: int

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forecast the mean of the context

        Args:
            x: (batch_size, num_timesteps, num_features) numpy array

        Returns:
            (batch_size, prediction_length, num_features) numpy array
        """
        return np.mean(x, axis=1, keepdims=True) * np.ones(
            (x.shape[0], self.prediction_length, x.shape[2])
        )


@dataclass
class FourierBaseline:
    prediction_length: int

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forecast the Fourier series of the context

        Args:
            x: (batch_size, num_timesteps, num_features) numpy array

        Returns:
            (batch_size, prediction_length, num_features) numpy array
        """
        batch_size, context_length, num_features = x.shape
        rfft_vals = np.fft.rfft(safe_standardize(x, axis=1), axis=1)
        ntotal = context_length + self.prediction_length
        reconstructed = np.fft.irfft(rfft_vals, n=ntotal, axis=1)
        return safe_standardize(
            reconstructed[:, context_length:], context=x, axis=1, denormalize=True
        )


@dataclass
class FourierARIMABaseline:
    prediction_length: int
    order: tuple[int, int, int] = (4, 1, 4)
    num_fourier_terms: int = 8
    min_period: int = 20
    max_period: int = 200

    def _estimate_period(self, signal: np.ndarray) -> np.ndarray:
        """Estimate the dominant period from the signal using autocorrelation.

        Args:
            signal: (batch_size, n_points) numpy array

        Returns:
            Array of shape (batch_size) containing estimated periods
        """
        b, _ = signal.shape
        fft_signal = np.fft.fft(signal, axis=1)
        autocorr = np.fft.ifft(fft_signal * np.conj(fft_signal), axis=1).real

        # Find peaks in batch for each feature
        peaks = np.zeros_like(autocorr, dtype=bool)
        peaks[:, 1:-1] = (autocorr[:, 1:-1] > autocorr[:, :-2]) & (
            autocorr[:, 1:-1] > autocorr[:, 2:]
        )

        periods = self.max_period * np.ones(b)
        x, y = np.where(peaks)
        _, inds = np.unique(x, return_index=True)
        periods[x[inds]] = y[inds] + 1  # the periods are the lag values
        return periods

    def _create_fourier_features(
        self,
        signal: np.ndarray,
        n_points: int,
        period: np.ndarray | float | None = None,
    ) -> np.ndarray:
        """Create Fourier features for seasonal decomposition.

        Args:
            signal: (batch_size, n_points) numpy array
            n_points: Number of time points
            period: Fundamental period for Fourier terms. If None, estimates from data

        Returns:
            Array of shape (batch_size, n_points, 2 * num_fourier_terms) containing sin/cos features
        """
        batch_size, _ = signal.shape

        if period is None:
            period = self._estimate_period(signal)
        elif isinstance(period, (int, float)):
            period = np.ones((batch_size)) * period

        t = np.arange(n_points)
        fourier_features = np.zeros((batch_size, n_points, 2 * self.num_fourier_terms))

        # Add harmonics with phase alignment
        for i in range(self.num_fourier_terms):
            freq = 2 * np.pi * (i + 1) / period
            fourier_features[:, :, 2 * i] = np.sin(freq[..., None] * t)
            fourier_features[:, :, 2 * i + 1] = np.cos(freq[..., None] * t)

        return fourier_features

    def _deseasonalize(
        self, x: np.ndarray, fourier_features: np.ndarray, eps: float = 1e-6
    ) -> tuple[np.ndarray, np.ndarray]:
        """Remove seasonal components using Fourier features.

        Args:
            x: (batch_size, num_timesteps) time series
            fourier_features: (batch_size, num_timesteps, 2 * num_fourier_terms) Fourier features

        Returns:
            Deseasonalized time series and coefficients
        """
        ATA = np.matmul(fourier_features.transpose(0, 2, 1), fourier_features)
        ATb = np.matmul(fourier_features.transpose(0, 2, 1), x[..., None]).squeeze()
        reg = np.eye(fourier_features.shape[2]) * eps
        coeffs = np.linalg.solve(ATA + reg[None, ...], ATb)
        seasonal = (fourier_features @ coeffs[..., None]).squeeze(-1)
        return x - seasonal, coeffs

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forecast using Fourier-ARIMA decomposition with pattern continuation.

        Args:
            x: (batch_size, num_timesteps, num_features) numpy array

        Returns:
            (batch_size, prediction_length, num_features) forecasts
        """
        batch_size, context_length, num_features = x.shape
        forecasts = np.zeros((batch_size, self.prediction_length, num_features))
        total_length = context_length + self.prediction_length

        batched = x.transpose(0, 2, 1).reshape(-1, context_length)
        standardized = safe_standardize(batched, axis=1)
        periods = self._estimate_period(standardized)
        fourier_features = self._create_fourier_features(
            standardized, total_length, period=periods
        )
        context_features = fourier_features[:, :context_length]
        forecast_features = fourier_features[:, context_length:]
        deseasonalized, coeffs = self._deseasonalize(standardized, context_features)
        seasonal_forecast = (forecast_features @ coeffs[..., None]).squeeze(-1)

        forecasts = np.zeros((batch_size * num_features, self.prediction_length))
        for i in range(batch_size * num_features):
            try:
                # Fit ARIMA on deseasonalized data
                model = ARIMA(deseasonalized[i], order=self.order)
                results = model.fit()
                forecasts[i] = results.forecast(self.prediction_length)
            except Exception as e:
                print(f"Error fitting ARIMA on {i}: {e}")

        # Add back seasonal component and denormalize
        forecasts += seasonal_forecast
        forecasts = safe_standardize(
            forecasts, axis=1, context=batched, denormalize=True
        )

        return forecasts.reshape(
            batch_size, num_features, self.prediction_length
        ).transpose(0, 2, 1)
