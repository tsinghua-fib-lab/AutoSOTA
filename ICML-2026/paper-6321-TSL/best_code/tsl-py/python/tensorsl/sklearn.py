"""
Scikit-learn compatible wrappers for TSL models.
"""

from typing import Optional

import numpy as np
from sklearn.utils.validation import check_array, check_X_y

try:
    from sklearn.base import BaseEstimator, RegressorMixin
    from sklearn.metrics import r2_score
    from sklearn.utils.validation import check_is_fitted

    _SKLEARN_AVAILABLE = True
except ImportError:
    # Define dummy classes if scikit-learn is not available
    # This allows the module to be imported, but the classes will raise errors if used.
    class BaseEstimator:
        pass

    class RegressorMixin:
        pass

    def check_is_fitted(estimator, attributes=None):
        if not hasattr(
            estimator, attributes if isinstance(attributes, str) else attributes[0]
        ):
            raise RuntimeError(
                "Scikit-learn is required to use this estimator. Please install it."
            )

    def r2_score(*args, **kwargs):
        raise RuntimeError(
            "Scikit-learn is required to use this estimator. Please install it."
        )

    _SKLEARN_AVAILABLE = False

from tensorsl._tensorsl import TSL


class TSLRegressor(BaseEstimator, RegressorMixin):
    """Scikit-learn compatible wrapper for TSL.

    Parameters
    ----------
    epochs : int, default=10
        Number of boosting epochs.
    n_trees : int, default=10
        Number of trees per epoch.
    n_iter : int, default=10
        Number of iterations for fitting each GridTensor.
    decay : float, default=1.0
        The decay rate for the number of splits in each epoch.
    split_try : int, default=10
        Number of split points to try for each feature.
    colsample_bytree : float, default=0.8
        Subsample ratio of columns when constructing each tree.
    alpha: float, default=0.0
        Regularization parameter for the L2 norm.
    complexity_penalty : float, default=0.0
        Complexity penalty (lambda) for adaptive merge bonus. This is BIC-inspired and scale-invariant.
        Larger values encourage simpler models. Typical values: 0.5-2.0. Default: 0.0 (no complexity penalty).
    min_interval_samples : int, default=1
        Minimum number of samples in an interval.
    min_split_loss : float, default=0.0
        Minimum loss reduction required to split a node.
    split_strategy : str, default="random"
        Strategy for selecting split points ("random", "best_split", "top_k").
    refinement_strategy : str, default="l2"
        Strategy for refining grid values ("l2", "huber").
    prior_sample_size : float, default=0.0
        Prior sample size (tau_0) for parent anchoring. Interpreted as "how many samples worth of confidence
        we have that children should equal their parent". With tau_0=30, a child with 10 samples will be
        heavily shrunk toward the parent, while a child with 100 samples will mostly trust its own data.
        Default: 0.0 (no anchoring). Typical values: 10-50.
    update_clamp : float, default=float('inf')
        Clamping parameter for refinement updates. Limits magnitude of update multipliers to the range
        [exp(-update_clamp), exp(update_clamp)]. Use float('inf') for no clamping.
    tilt_tau : float, default=0.01
        Two-tensor L2 coupling between u_+ and u_- (objective τ). Controls the strength
        of the quadratic penalty on the difference between positive and negative components.
    tilt_rho : float, default=0.0
        Two-tensor L1 coupling on (u_+ - u_-) (objective ρ). When > 0, can drive the tilt
        exactly to zero on many sides, yielding pure "backbone-only" updates.
    top_k : int, default=10
        When `split_strategy == "top_k"`, number of top candidate splits to consider.
    must_fill_all_k : bool, default=True
        When `split_strategy == "top_k"`, require all K splits to be filled.
    similarity_threshold : float, default=0.0
        Threshold for merging similar components.
    bagged : bool, default=False
        Whether to use bagging.
    seed : int, default=42
        Random seed for reproducibility.
    verbosity : int, default=1
        Verbosity level for the rust backend. 0 = off, 1 = info, 2 = debug, 3 = trace.
    visualdb : str, optional, default=None
        Path to SQLite database for saving split events and run information.
        If specified, split events will be recorded during fitting and saved to the database.
        Requires the evo-logging feature to be enabled. If the feature is disabled,
        a warning will be logged.

    Attributes
    ----------
    core_estimator_ : TSL
        The underlying fitted TSL instance.
    fit_result_ : FitResult
        The FitResult object returned by the core fitting process.
    """

    def __init__(
        self,
        epochs: int = 10,
        n_trees: int = 10,
        n_iter: int = 10,
        decay: float = 1.0,
        split_try: int = 10,
        colsample_bytree: float = 0.8,
        alpha: float = 0.0,
        complexity_penalty: float = 0.0,
        min_split_loss: float = 0.0,
        min_interval_samples: int = 1,
        refinement_strategy: str = "l2",
        prior_sample_size: float = 0.0,
        update_clamp: float = float("inf"),
        tilt_tau: float = 0.01,
        tilt_rho: float = 0.0,
        split_strategy: str = "random",
        top_k: int = 10,
        must_fill_all_k: bool = True,
        similarity_threshold: float = 0.0,
        bagged: bool = False,
        seed: int = 42,
        verbosity: int = 1,
        visualdb: Optional[str] = None,
    ):
        if not _SKLEARN_AVAILABLE:
            raise ImportError(
                "Scikit-learn is required to use TSLRegressor. "
                "Please install it (`pip install scikit-learn`)."
            )

        self.epochs = epochs
        self.n_trees = n_trees
        self.n_iter = n_iter
        self.decay = decay
        self.split_try = split_try
        self.colsample_bytree = colsample_bytree
        self.alpha = alpha
        self.complexity_penalty = complexity_penalty
        self.min_split_loss = min_split_loss
        self.min_interval_samples = min_interval_samples
        self.refinement_strategy = refinement_strategy
        self.prior_sample_size = prior_sample_size
        self.update_clamp = update_clamp
        self.tilt_tau = tilt_tau
        self.tilt_rho = tilt_rho
        self.split_strategy = split_strategy
        self.top_k = top_k
        self.must_fill_all_k = must_fill_all_k
        self.similarity_threshold = similarity_threshold
        self.bagged = bagged
        self.seed = seed
        self.verbosity = verbosity
        self.visualdb = visualdb

    def fit(self, X, y):
        """Fit the TSL Boosted regressor to the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        self : object
            Returns self.
        """
        # Input validation (optional but recommended for robustness)
        X, y = check_X_y(X, y, accept_sparse=False, dtype=np.float64)

        # Use the classmethod `fit` from the original TSL.Boosted class
        # Pass hyperparameters stored in self
        fitted_instance, fit_result = TSL.fit(
            x=X,  # Renamed for clarity, assuming X is the feature matrix
            y=y,
            epochs=self.epochs,
            decay=self.decay,
            n_trees=self.n_trees,
            n_iter=self.n_iter,
            split_try=self.split_try,
            colsample_bytree=self.colsample_bytree,
            alpha=self.alpha,
            complexity_penalty=self.complexity_penalty,
            min_split_loss=self.min_split_loss,
            min_interval_samples=self.min_interval_samples,
            refinement_strategy=self.refinement_strategy,
            prior_sample_size=self.prior_sample_size,
            update_clamp=self.update_clamp,
            tilt_tau=self.tilt_tau,
            tilt_rho=self.tilt_rho,
            split_strategy=self.split_strategy,
            top_k=self.top_k,
            must_fill_all_k=self.must_fill_all_k,
            similarity_threshold=self.similarity_threshold,
            bagged=self.bagged,
            seed=self.seed,
            verbosity=self.verbosity,
            visualdb=self.visualdb,
        )

        # Store the fitted core estimator and the fit result
        self.core_estimator_ = fitted_instance
        self.fit_result_ = fit_result

        return self

    def predict(self, X):
        """Predict regression target for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted values.
        """
        # Check if fit has been called
        check_is_fitted(self, "core_estimator_")

        # Input validation (optional)
        X = check_array(X, accept_sparse=False, dtype=np.float64)

        # Delegate prediction to the core estimator
        return self.core_estimator_.predict(X)

    def score(self, X, y):
        """Return the coefficient of determination R^2 of the prediction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True values for X.

        Returns
        -------
        score : float
            R^2 of self.predict(X) wrt. y.
        """
        # Check if fit has been called
        check_is_fitted(self, "core_estimator_")
        y_pred = self.predict(X)
        return r2_score(y, y_pred)

    @property
    def stage_predictors(self):
        check_is_fitted(self, "core_estimator_")
        return self.core_estimator_.stage_predictors

    def save(self, path: str) -> None:
        """Save the TSL model to a binary file (preserves exact floating point values).

        Parameters
        ----------
        path : str
            Path to the file where the model will be saved.
        """
        check_is_fitted(self, "core_estimator_")
        self.core_estimator_.save(path)

    @classmethod
    def load(cls, path: str) -> "TSLRegressor":
        """Load an TSL model from a binary file.

        Parameters
        ----------
        path : str
            Path to the binary file containing the saved model.

        Returns
        -------
        TSLRegressor
            The loaded TSLRegressor instance.
        """
        # Load the core TSL model
        core_estimator = TSL.load(path)

        # Create a new TSLRegressor instance
        instance = cls.__new__(cls)
        instance.core_estimator_ = core_estimator
        instance.fit_result_ = None  # Fit result is not saved

        return instance
