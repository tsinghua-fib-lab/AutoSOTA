from __future__ import annotations

from typing import Sequence, Union

import numpy as np
from sklearn.linear_model import LogisticRegression
from etl.model import Model

ArrayLike = Union[Sequence[float], np.ndarray]

class BinormalGenerator(Model):
    """Generate simple synthetic binormal data and provide basic utilities.

    This implementation generates synthetic data from normal distributions
    and fits a logistic regression model to produce predictions.

    Parameters
    ----------
    mean_0 : float, default=0.0
        Mean of the normal distribution for class 0.
    mean_1 : float, default=1.0
        Mean of the normal distribution for class 1.
    std_0 : float, default=1.0
        Standard deviation of the normal distribution for class 0.
    std_1 : float, default=1.0
        Standard deviation of the normal distribution for class 1.
    num_samples : int, default=10_000
        Number of samples to generate for each class.
    prevalence : float, default=0.5
        Prevalence of class 1 in the training data.
    cost : float, default=0.5
        Cost parameter for the logistic regression model.
    """

    def __init__(
        self,
        mean_0: float = 0.0,
        mean_1: float = 1.0,
        std_0: float = 1.0,
        std_1: float = 1.0,
        num_samples: int = 10_000,
        prevalence: float = 0.5,
        cost: float = 0.5,
    ) -> None:
        self.mean_0 = mean_0
        self.mean_1 = mean_1
        self.std_0 = std_0
        self.std_1 = std_1
        self.num_samples = num_samples
        self._train_prevalence = prevalence
        self.train_cost = cost

        # Generate samples for each class.
        self.original_samples_0 = np.random.normal(mean_0, std_0, num_samples)
        self.original_samples_1 = np.random.normal(mean_1, std_1, num_samples)

        # Initialise training arrays
        self._train_0 = None
        self._train_1 = None

        # Fit a simple logistic regression model to map raw samples → scores.
        self.fit_logistic(prevalence, cost)

    @property
    def train_0(self) -> np.ndarray:
        return self._train_0

    @property
    def train_1(self) -> np.ndarray:
        return self._train_1

    @property
    def train_prevalence(self) -> float:
        return self._train_prevalence

    def fit_logistic(self, prevalence: float = 0.5, cost: float = 0.5):
        """Fit a simple logistic regression on the generated samples.

        Parameters
        ----------
        prevalence : float, default=0.5
            Prevalence of class 1 for weighting the training data.
        cost : float, default=0.5
            Cost parameter for weighting the training data.

        Returns
        -------
        LogisticRegression
            Fitted logistic regression model.
        """
        self._train_prevalence = prevalence
        self.train_cost = cost

        # Build weighted training data.
        w0 = (1 - prevalence) * cost
        w1 = prevalence * (1 - cost)
        w_norm = w1 / (w0 + w1)

        X = np.concatenate((self.original_samples_0, self.original_samples_1))
        y = np.concatenate((np.zeros(self.num_samples), np.ones(self.num_samples)))

        # Sample weights.
        W = np.concatenate(
            ((1 - w_norm) * np.ones(self.num_samples), w_norm * np.ones(self.num_samples))
        )

        # Add a quadratic term so the logistic model can capture non-linearity
        # (the basic Gaussian likelihood ratio is quadratic in x).
        X_design = np.stack([X, X ** 2], axis=1)

        model = LogisticRegression()
        model.fit(X_design, y, sample_weight=W)

        y_hat = model.predict_proba(X_design)[:, 1]
        self._train_0 = y_hat[: self.num_samples]
        self._train_1 = y_hat[self.num_samples :]

        return model
