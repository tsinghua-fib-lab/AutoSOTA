from sklearn.base import clone, check_is_fitted
from joblib import Parallel, delayed
import numpy as np
import pandas as pd

# This class is just the conditional sampler from the CPI adapted to the HRT

class RegressionSamplerSingle:
    def __init__(self, estimator, X_train, X_test, covariate_index, n_jobs=1, random_state=None):
        """
        Parameters
        ----------
        estimator : scikit-learn estimator
            Regression model to predict the covariate.
        X_train : np.ndarray or pd.DataFrame
            Training data.
        X_test : np.ndarray or pd.DataFrame
            Test data.
        covariate_index : int
            Index of the covariate to model.
        n_jobs : int
            Number of parallel jobs.
        n_permutations : int
            Number of residual permutations.
        random_state : int or None
            Random seed.
        """
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.X_train = X_train
        self.X_test = X_test
        self.j = covariate_index
        self.rng = np.random.default_rng(random_state)
        self._imputer_model = None
        self.residuals_test = None

    def fit(self):
        """Fit the imputer for the selected covariate."""
        X_j = self.X_train[:, self.j] if isinstance(self.X_train, np.ndarray) else self.X_train.iloc[:, self.j].values
        X_minus_j = np.delete(self.X_train, self.j, axis=1) if isinstance(self.X_train, np.ndarray) else self.X_train.drop(self.X_train.columns[self.j], axis=1).values
        
        self._imputer_model = clone(self.estimator)
        self._imputer_model.fit(X_minus_j, X_j)
        return self

    def sampler(self):
        X_j = self.X_test[:, self.j].copy()
        X_minus_j = np.delete(self.X_test, self.j, axis=1)
        X_j_hat = self._imputer_model.predict(X_minus_j)
        residuals = X_j - X_j_hat
        sample = X_j_hat + self.rng.permutation(residuals)
        return sample.ravel()  # ensure shape (n_test,)





class RegressionSampler:
    def __init__(self, estimator, X_train, X_test, n_jobs=1, n_permutations=1, random_state=None):
        """
        Parameters
        ----------
        estimator : scikit-learn estimator
            The regression model to use for imputing covariates.
        n_jobs : int
            Number of parallel jobs.
        n_permutations : int
            Number of residual permutations for later sampling.
        random_state : int or None
            Random seed.
        """
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.X_train = X_train
        self.X_test = X_test
        self.n_permutations = n_permutations
        self.rng = np.random.default_rng(random_state)
        self._imputer_models = []
        self.groups = None
        self.residuals_test = None

    def fit(self, groups=None):
        """
        Fit an imputer for each group of covariates on training data.
        """
        self.groups = groups
        n_features = self.X_train.shape[1]
        if self.groups is None:
            self.groups = {j: [j] for j in range(n_features)}
        self.n_groups = len(self.groups)

        # create one imputer per group
        self._imputer_models = [clone(self.estimator) for _ in range(self.n_groups)]

        def _fit_one_group(model, X, j):
            # X_j is the group to predict
            if isinstance(X, pd.DataFrame):
                X_j = X[self.groups[j]].copy().values
                X_minus_j = X.drop(columns=self.groups[j]).values
            else:
                X_j = X[:, self.groups[j]].copy()
                if X_j.shape[1] == 1:
                    X_j = X_j.ravel()
                X_minus_j = np.delete(X, self.groups[j], axis=1)
            model.fit(X_minus_j, X_j)
            return model

        self._imputer_models = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_one_group)(model, self.X_train, j)
            for j, model in zip(self.groups.keys(), self._imputer_models)
        )

        return self

    def sampler(self):
        """
        Compute residuals on test data using fitted imputers.
        Returns residuals for each group.
        """
        if len(self._imputer_models) == 0:
            raise ValueError("Call fit first!")
        for m in self._imputer_models:
            check_is_fitted(m)

        def _compute_residual(model, X, j):
            if isinstance(X, pd.DataFrame):
                X_j = X[self.groups[j]].copy().values
                X_minus_j = X.drop(columns=self.groups[j]).values
            else:
                X_j = X[:, self.groups[j]].copy()
                if X_j.ndim == 1:
                    X_j = X_j.reshape(-1, 1)
                X_minus_j = np.delete(X, self.groups[j], axis=1)

            X_j_hat = model.predict(X_minus_j).reshape(X_j.shape)
            residuals = X_j - X_j_hat
            return X_j_hat+self.rng.permutation(residuals)

        residuals_list = Parallel(n_jobs=self.n_jobs)(
            delayed(_compute_residual)(model, self.X_test, j)
            for j, model in zip(self.groups.keys(), self._imputer_models)
        )

        # Stack residuals as (groups, n_samples, group_dim)
        self.residuals_test = np.stack(residuals_list, axis=0)
        return self.residuals_test
