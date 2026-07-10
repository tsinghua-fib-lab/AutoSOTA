from copy import copy, deepcopy

import numpy as np
import pandas as pd
import sage
from hidimstat import LOCO
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.utils import resample
from sklearn.utils.validation import check_is_fitted
from tqdm import tqdm


def select_k_best_features_intersection(
    X, y, y_stratif, cv, k=300, imputer=SimpleImputer(), method=f_regression
):
    """
    Select features using univariate statistical tests.
     1. For each fold in the cross-validation, select the top `k` features using
     SelectKBest.
     2. Count how many times each feature was selected across all folds.
     3. Retain features that were selected in all folds.

    Parameters
    ----------
    X : pd.DataFrame, shape (n_samples, n_features)
        The input features.
    y : array-like, shape (n_samples,)
        The target labels.
    cv : cross-validation generator
        The cross-validation strategy.
    k : int, default=300
        The number of top features to select in each fold.
    imputer : imputer object, default=SimpleImputer()
        The imputer to handle missing values.
    method : callable, default=f_regression
        The scoring function to use for univariate feature selection.

    Returns
    -------
    selected_feature_indices : np.ndarray
        Indices of the selected features.
    selected_feature_names : pd.Index
        Names of the selected features.
    """
    selector = SelectKBest(score_func=method, k=k)
    selected_features = []
    X_arr = np.array(X)
    y_arr = np.array(y)
    X_cols = X.columns

    for train_index, _ in tqdm(cv.split(X, y_stratif)):
        X_train, y_train = X_arr[train_index], y_arr[train_index]
        imputer.fit(X_train)
        X_train = imputer.transform(X_train)
        selector.fit(X_train, y_train)
        selected_features.append(selector.get_support(indices=True))

    selected_features = np.hstack(selected_features)
    most_selected_features, counts = np.unique(selected_features, return_counts=True)
    selected_feature_keep = most_selected_features[counts == cv.get_n_splits()]

    return np.sort(selected_feature_keep), X_cols[selected_feature_keep]


def get_sub_models(model):
    """
    Helper function to extract sub-models from an ensemble model.

    Parameters
    ----------
    model : sklearn estimator
        The ensemble model with attribute `estimators_`.

    Returns
    -------
    sub_models : list
        List of sub-models contained in the ensemble.
    """
    if hasattr(model, "estimators_"):
        return model.estimators_
    else:
        raise ValueError("Model type not supported for sub-model extraction.")


def _parallel_fit_with_indices(estimator, X, y, random_state):
    """Fits a model and returns both the model and the bootstrap indices."""
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    boot_indices = resample(
        indices, replace=True, n_samples=n_samples, random_state=random_state
    )

    X_boot, y_boot = X[boot_indices], y[boot_indices]
    fitted_est = clone(estimator).fit(X_boot, y_boot)

    return fitted_est, boot_indices


class BaggingVoting(BaseEstimator, RegressorMixin):
    """
    An ensemble regressor that fits multiple estimators, with possibly different
    hyperparameters, on bootstrap samples of the training data.

    Parameters
    ----------
    estimators : list of tuples
        List of (name, estimator) tuples to be used in the ensemble.
    n_jobs : int, optional (default=-1)
        The number of jobs to run in parallel for both `fit` and `predict`.
    random_state : int, optional (default=None)
        Random seed for reproducibility. Controls the random sampling of
        bootstrap samples for each estimator.
    """

    def __init__(self, estimators, n_jobs=-1, random_state=None):
        self.estimators = estimators
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, X, y):
        # Generate a seed for each estimator to ensure different bootstrap samples
        rng = np.random.RandomState(self.random_state)
        seeds = rng.randint(np.iinfo(np.int32).max, size=len(self.estimators))

        # Parallel execution returns a list of (fitted_model, indices) tuples
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(_parallel_fit_with_indices)(est, X, y, seed)
            for (_, est), seed in zip(self.estimators, seeds)
        )
        self.estimators_, self.estimators_samples_ = zip(*results)
        return self

    def predict(self, X):
        check_is_fitted(self)
        preds = np.column_stack([est.predict(X) for est in self.estimators_])
        return np.mean(preds, axis=1)


def loco_one(X, y, train_index, test_index, model, fold_id, n_jobs=1):
    """
    Helper function to compute LOCO importances for a given fold. This function
    is used for parallel computation of LOCO importances across CV folds.
    """
    n_features = X.shape[1]
    output_list = []
    imputer = SimpleImputer()
    X_train = imputer.fit_transform(X[train_index])
    X_test = imputer.transform(X[test_index])

    # 1. Ensure the ensemble model is fitted
    if not check_is_fitted(model):
        model_c = clone(model)
        model_c.fit(X_train, y[train_index])
    else:
        model_c = model

    if hasattr(model_c, "n_jobs"):
        model_c.n_jobs = 1

    # 2. Model-level ensembling (ensemble)
    print("LOCO ensemble ...")
    loco = LOCO(model_c, loss=mean_squared_error, n_jobs=n_jobs)

    loco.fit(X_train, y[train_index])
    importances_full = loco.importance(X_test, y[test_index])

    output_list.append(
        pd.DataFrame(
            {
                "feature": np.arange(n_features),
                "importance": importances_full,
                "fold": fold_id,
                "model": "ensemble",
            }
        )
    )

    # 3. Importance-level ensembling (sub-models)
    sub_model_list = get_sub_models(model_c)
    if hasattr(model_c, "estimators_samples_"):
        bootstrap_samples_indices = model_c.estimators_samples_
    else:
        bootstrap_samples_indices = [
            np.arange(len(train_index)) for _ in range(len(sub_model_list))
        ]

    print("LOCO sub-models ...")
    for i, sub_model in enumerate(sub_model_list):
        # Retrieve the specific indices for sub-model i
        # Note: These indices are relative to the X[train_index] passed to model_c.fit
        current_bootstrap_indices = bootstrap_samples_indices[i]

        X_train_bootstrap = X_train[current_bootstrap_indices]
        y_train_bootstrap = y[train_index][current_bootstrap_indices]

        sub_model_c = deepcopy(sub_model)
        # Calculate LOCO for the individual sub-model f_b
        loco_sub = LOCO(sub_model_c, loss=mean_squared_error, n_jobs=n_jobs)
        loco_sub.fit(X_train_bootstrap, y_train_bootstrap)
        importances_sub = loco_sub.importance(X_test, y[test_index])

        output_list.append(
            pd.DataFrame(
                {
                    "feature": np.arange(n_features),
                    "importance": importances_sub,
                    "fold": fold_id,
                    "model": f"sub_model_{i}",
                }
            )
        )

    return output_list


def sage_one(
    X, y, train_index, test_index, model, fold_id, seed, n_jobs=1, n_samples=512
):
    """
    Compute SAGE values for a given fold. This function is used for parallel
    computation of SAGE values across CV folds.

    Parameters
    ----------
    X : array-like
        Feature matrix.
    y : array-like
        Target vector.
    train_index : array-like
        Indices for training data.
    test_index : array-like
        Indices for test data.
    model : sklearn estimator
        Fitted prediction model.
    fold_id : int
        Fold identifier.
    seed : int
        Random seed for reproducibility.
    n_jobs : int, default=1
        Number of parallel jobs.
    n_samples : int, default=512
        Number of samples to use for SAGE computation.

    Returns
    -------
    output_list : list of pd.DataFrame
        List of dataframes containing SAGE values for the ensemble and sub-models.
    """

    X_train, y_train = X[train_index], y[train_index]
    X_test, y_test = X[test_index], y[test_index]
    n_features = X.shape[1]
    output_list = []
    if not check_is_fitted(model):
        model_c = clone(model)
        model_c.fit(X[train_index], y[train_index])
    else:
        model_c = model

    if X_test.shape[0] > n_samples:
        impute_ids = np.random.choice(X_test.shape[0], size=n_samples, replace=False)
        X_impute = X_test[impute_ids]
    else:
        X_impute = X_test

    if hasattr(model_c, "n_jobs"):
        model_c.n_jobs = 1

    imputer = sage.MarginalImputer(model_c, X_impute[:,])
    estimator = sage.PermutationEstimator(
        imputer, "mse", random_state=seed, n_jobs=n_jobs
    )

    # Subsample to make computation tractable
    sample_ids = np.random.choice(X_test.shape[0], size=n_samples, replace=False)
    X_subsample = X_test[sample_ids]
    y_subsample = y_test[sample_ids]
    print(X_subsample.shape)
    sage_values = estimator(X_subsample, y_subsample)
    output_list.append(
        pd.DataFrame(
            {
                "feature": np.arange(n_features),
                "importance": sage_values.values,
                "fold": fold_id,
                "model": "ensemble",
                "std": sage_values.std,
            }
        )
    )

    print("SAGE sub-models")
    sub_model_list = get_sub_models(model_c)
    for i, sub_model in enumerate(sub_model_list):
        sub_model_c = copy(sub_model)
        imputer = sage.MarginalImputer(sub_model_c, X_impute)
        estimator = sage.PermutationEstimator(
            imputer, "mse", random_state=seed, n_jobs=n_jobs
        )
        sage_values = estimator(X_subsample, y_subsample)
        output_list.append(
            pd.DataFrame(
                {
                    "feature": np.arange(n_features),
                    "importance": sage_values.values,
                    "fold": fold_id,
                    "model": f"sub_model_{i}",
                    "std": sage_values.std,
                }
            )
        )
    return output_list
