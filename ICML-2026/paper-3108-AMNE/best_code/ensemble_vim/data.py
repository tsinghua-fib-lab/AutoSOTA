"""
This module provides the code for generating the benchmark datasets:
 - Friedman1
 - Ishigami function
 - G-function
"""

import numpy as np
from scipy.linalg import toeplitz
from sklearn.datasets import make_friedman1, make_friedman2, make_friedman3
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler


class GFunction:
    """
    Class to generate samples from the G-function.
    y = (prod_{i=1}^d (|4*x_i - 2| + a_i)) / (prod_{i=1}^d (1 + a_i))

    Parameters
    ----------
    a_i_values : list or array-like, shape (d,)
        Coefficients for each feature. Higher values reduce the influence of that
        feature.
    correlation : float, optional
        Correlation coefficient for the Toeplitz covariance matrix. If None,
        features are uncorrelated.
    snr : float, optional
        Signal-to-noise ratio for the output variable.
    """

    def __init__(self, a_i_values, correlation=None, snr=1.0):
        self.a_i_values = a_i_values
        self.ai_arr = np.array(a_i_values)
        self.d = len(a_i_values)
        self.correlation = correlation
        self.snr = snr
        if correlation is not None:
            self.cov = toeplitz(correlation ** np.arange(0, self.d))
            self.cov_chol = np.linalg.cholesky(self.cov)
        else:
            self.cov = np.eye(self.d)
            self.cov_chol = np.eye(self.d)

    def sample(self, n_samples, random_state=None):
        rng = np.random.default_rng(random_state)
        X = rng.uniform(0, 1, size=(n_samples, self.d))
        X = X.dot(self.cov_chol.T)

        y = self.g_function(X, self.ai_arr)
        noise = rng.normal(0, np.std(y) / self.snr, size=n_samples)
        y = y + noise

        y = StandardScaler().fit_transform(y.reshape(-1, 1)).ravel()
        X = StandardScaler().fit_transform(X)
        return X, y

    @staticmethod
    def g_function(X, ai_arr):
        """
        Parameters
        ----------
        X : array-like, shape (n_samples, d)
            Input samples.
        ai_arr : array-like, shape (d,)
            Coefficients for each dimension.
        """
        numerator = np.prod(np.abs(4 * X - 2) + ai_arr, axis=1)
        denominator = np.prod(1 + ai_arr)
        return numerator / denominator


class IshigamiFunction:
    """
    Class to generate samples from the Ishigami function.
    y = sin(x1) + 7*sin(x2)^2 + 0.1*x3^4*sin(x1)

    Parameters
    ----------
    n_features : int, optional
        Number of input features. Default is 3.
    correlation : float, optional
        Correlation coefficient for the Toeplitz covariance matrix. If None,
        features are uncorrelated.
    snr : float, optional
        Signal-to-noise ratio for the output variable.
    classification : bool, optional
        If True, the output variable is converted to a binary classification
        problem based on the median value.
    """

    def __init__(self, n_features=3, correlation=None, snr=1.0, classification=False):
        self.n_features = n_features
        self.correlation = correlation
        self.snr = snr
        self.classification = classification
        if correlation is not None:
            self.cov = toeplitz(correlation ** np.arange(0, self.n_features))
            self.cov_chol = np.linalg.cholesky(self.cov)
        else:
            self.cov = np.eye(self.n_features)
            self.cov_chol = np.eye(self.n_features)

    def sample(self, n_samples, random_state=None):
        rng = np.random.default_rng(random_state)
        X = rng.uniform(-np.pi, np.pi, size=(n_samples, self.n_features))
        X = X.dot(self.cov_chol.T)

        y = self.ishigami_function(X)
        noise = rng.normal(0, np.std(y) / self.snr, size=n_samples)
        y = y + noise

        y = StandardScaler().fit_transform(y.reshape(-1, 1)).ravel()
        X = StandardScaler().fit_transform(X)
        if self.classification:
            median = np.median(y)
            y = (y > median).astype(int)
        return X, y

    @staticmethod
    def ishigami_function(X):
        """
        Parameters
        ----------
        X : array-like, shape (n_samples, d)
            Input samples.
        ai_arr : array-like, shape (d,)
            Coefficients for each dimension.
        """
        y = (
            np.sin(X[:, 0])
            + 7 * np.sin(X[:, 1]) ** 2
            + 0.1 * X[:, 2] ** 4 * np.sin(X[:, 0])
        )
        return y


class NonLinearDataset:
    """
    Features X have a linear correlation structure (Multivariate Normal),
    but the target Y is generated using non-linear transformations of the
    active features. Supports larger support sizes than Friedman/Ishigami/G.

    The data-generating process (support, coefficients, transformations) is
    fixed via ``dgp_seed`` so that the ground truth is identical across
    repetitions.  Only the sampling of X and noise varies with the
    ``random_state`` passed to :meth:`sample`.

    Notes
    -----
    The non-linear dataset was used for the experiment with d=100 in order
    to allow increasing the support size beyond the previous datasets.

    Parameters
    ----------
    n_features : int
        Total number of features.
    sparsity : float
        Fraction of features that are active (support_size = n_features * sparsity).
    correlation : float
        Toeplitz correlation coefficient for X.
    dgp_seed : int
        Fixed seed for the data-generating process (support, coefficients,
        transformations).  Identical across all repetitions.
    coeff_range : tuple of float
        Range for uniform coefficient magnitudes.
    snr : float
        Signal-to-noise ratio.
    """

    def __init__(
        self,
        n_features,
        sparsity=0.5,
        correlation=0.5,
        dgp_seed=0,
        coeff_range=(9.0, 10.0),
        snr=100,
    ):
        self.n_features = n_features
        self.sparsity = sparsity
        self.correlation = correlation
        self.dgp_seed = dgp_seed
        self.coeff_range = coeff_range
        self.snr = snr

    def _initialize_simulation(self):
        if hasattr(self, "_is_initialized_"):
            return self

        rng = np.random.default_rng(self.dgp_seed)
        self.support_size_ = int(self.n_features * self.sparsity)
        self.support_ = rng.choice(self.n_features, self.support_size_, replace=False)

        self.support_mask_ = np.zeros(self.n_features, dtype=int)
        self.support_mask_[self.support_] = 1

        self.coefficients_ = np.zeros(self.n_features)
        self.coefficients_[self.support_] = rng.uniform(
            self.coeff_range[0], self.coeff_range[1], size=self.support_size_
        )
        self.coefficients_[self.support_] *= rng.choice(
            [-1, 1], size=self.support_size_
        )

        # Assign non-linear functions to the active features
        available_funcs = ["sin", "cos", "square", "abs", "tanh"]
        self.transformations_ = rng.choice(available_funcs, size=self.support_size_)

        # Linear correlation structure for X
        self.covariance_ = toeplitz(self.correlation ** np.arange(self.n_features))
        self._is_initialized_ = True
        return self

    def sample(self, n_samples, random_state=None):
        self._initialize_simulation()
        rng = np.random.default_rng(random_state)

        X = rng.multivariate_normal(
            mean=np.zeros(self.n_features), cov=self.covariance_, size=n_samples
        )

        signal = np.zeros(n_samples)
        for i, feature_idx in enumerate(self.support_):
            x_col = X[:, feature_idx]
            func_type = self.transformations_[i]
            c = self.coefficients_[feature_idx]

            if func_type == "sin":
                term = np.sin(x_col * 2.0)
            elif func_type == "cos":
                term = np.cos(x_col * 2.0)
            elif func_type == "square":
                term = x_col**2
            elif func_type == "abs":
                term = np.abs(x_col)
            elif func_type == "tanh":
                term = np.tanh(x_col)

            signal += c * term

        signal_variance = np.var(signal)
        noise_std = np.sqrt(signal_variance / self.snr) if signal_variance > 0 else 1e-9
        noise = rng.normal(loc=0, scale=noise_std, size=n_samples)

        y = signal + noise
        return X, y


def get_dataset(
    dataset_name, n_samples, snr, random_state, n_features=None, sparsity=0.25
):
    """
    Wrapper function to generate datasets. For each dataset, the target variable is
    normalized to have zero mean and unit variance.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset to generate. Options are "friedman1", "friedman2",
        "friedman3", "g_function", "ishigami".
    n_samples : int
        Number of samples to generate.
    snr : float
        Signal-to-noise ratio for the output variable.
    random_state : int
        Random seed for reproducibility.
    n_features : int, optional
        Number of features.

    Returns
    -------
    X : array-like, shape (n_samples, n_features)
        Generated input features.
    y : array-like, shape (n_samples,)
        Generated target variable, normalized.
    support : array-like, shape (n_relevant_features,)
        Indices of the features that are part of the true support.
    support_bis : array-like, shape (n_relevant_features,)
        Indices of the features that are part of the true support (duplicate unused).
    """
    if dataset_name == "friedman1":
        X, y = make_friedman1(
            n_samples=n_samples,
            n_features=n_features,
            noise=2 / snr,
            random_state=random_state,
        )
        minfo = mutual_info_regression(X, y, random_state=random_state)
        support = np.argsort(minfo)[-5:]
        support_bis = support
    elif dataset_name == "friedman2":
        X, y = make_friedman2(
            n_samples=n_samples,
            noise=2 / snr,
            random_state=random_state,
        )
        minfo = mutual_info_regression(X, y, random_state=random_state)
        support = np.arange(X.shape[1])
        support_bis = support
    elif dataset_name == "friedman3":
        X, y = make_friedman3(
            n_samples=n_samples,
            noise=2 / snr,
            random_state=random_state,
        )
        minfo = mutual_info_regression(X, y, random_state=random_state)
        support = np.arange(X.shape[1])
        support_bis = support

    elif dataset_name == "g_function":
        a_i_values = [0, 1, 2, 3, 4] + [100] * (n_features - 5)
        g_func = GFunction(a_i_values, correlation=0.3, snr=snr)
        X, y = g_func.sample(n_samples=n_samples, random_state=random_state)
        support = np.arange(5)
        support_bis = support
    elif dataset_name == "ishigami":
        ishigami = IshigamiFunction(n_features=n_features, correlation=0.3, snr=snr)
        X, y = ishigami.sample(n_samples=n_samples, random_state=random_state)
        support = np.array([0, 1, 2])
        support_bis = support
    elif dataset_name == "nonlinear":
        ds = NonLinearDataset(
            n_features=n_features,
            sparsity=sparsity,
            correlation=0.5,
            dgp_seed=0,  # same support, coefficients, transformations across repetitions
            snr=snr,
        )
        X, y = ds.sample(n_samples=n_samples, random_state=random_state)
        support = np.sort(ds.support_)
        support_bis = support
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    y_norm = StandardScaler().fit_transform(y.reshape(-1, 1)).ravel()
    return X, y_norm, support, support_bis
