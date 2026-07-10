"""
Script to run simulation study for ensemble VIMs.
"""

import argparse
from copy import copy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sage
from hidimstat import CFI, LOCO, PFI
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.validation import check_is_fitted

from ensemble_vim.data import get_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Run ensemble VIM simulation")
    parser.add_argument(
        "--n_samples",
        type=int,
        nargs="+",
        default=[128, 256, 512, 1024, 2048],
        help="Number of samples per dataset",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--n_jobs", type=int, default=5, help="Number of parallel jobs")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of CV splits")
    parser.add_argument("--snr", type=float, default=4.0, help="Signal to noise ratio")
    parser.add_argument(
        "--n_ensemble", type=int, default=5, help="Number of ensemble members"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="./results",
        help="Results directory",
    )
    parser.add_argument(
        "--dataset_name", type=str, default="friedman1", help="Dataset name"
    )
    parser.add_argument("--n_features", type=int, default=20, help="Number of features")
    parser.add_argument(
        "--model_name",
        type=str,
        default="mlp",
        help="Model name (mlp or rf)",
    )
    parser.add_argument(
        "--ensemble",
        type=str,
        default="voting",
        help="Ensemble type (voting or bagging)",
    )
    parser.add_argument("--sage", action="store_true", help="Compute SAGE")
    args = parser.parse_args()
    return args


def get_sub_models(model):
    """
    Extract sub-models from an ensemble model to compute individual "sub-model"
    importances.

    Parameters
    ----------
    model : sklearn estimator
        Fitted ensemble model with `estimators_` attribute.

    Returns
    -------
    sub_models : list
        List of fitted sub-models.
    """
    if hasattr(model, "estimators_"):
        sub_models = model.estimators_
    else:
        raise ValueError("Model type not supported for sub-model extraction.")
    return sub_models


def get_model(model_name, n_ensemble, ensemble="voting", seed=None):
    """
    Wrapper function to get the prediction model. This can be either a voting or bagging
    ensemble which respectively ensemble estimators trained on bootstrap samples or
    with different random seeds.

    Parameters
    ----------
    model_name : str
        Name of the model to use. Options are "mlp", "rf", "linear".
    n_ensemble : int
        Number of estimators in the ensemble.
    ensemble : str
        Type of ensemble to use. Options are "voting" or "bagging".
    seed : int
        Random seed for reproducibility.

    Notes
    -----
    The mlp256 model was used for the experiment with d=100 only. It was found
    to achieve better performance and was therefore preferred.

    Returns
    -------
    model : sklearn estimator
        Ensemble prediction model.
    """
    if (model_name == "mlp") or (model_name == "mlp256"):
        if model_name == "mlp256":
            hidden_layer_sizes = (256, 256)
        else:
            hidden_layer_sizes = (64, 32, 8)
        if ensemble == "voting":
            model = VotingRegressor(
                estimators=[
                    (
                        f"mlp_{k}",
                        MLPRegressor(
                            hidden_layer_sizes=hidden_layer_sizes,
                            max_iter=2000, n_iter_no_change=50, validation_fraction=0.2,
                            random_state=seed + k,
                            early_stopping=True,
                        ),
                    )
                    for k in range(n_ensemble)
                ],
                n_jobs=1,
            )
        elif ensemble == "bagging":
            base_model = MLPRegressor(
                hidden_layer_sizes=hidden_layer_sizes,
                max_iter=2000, n_iter_no_change=50, validation_fraction=0.2,
                random_state=seed,
                early_stopping=True,
            )
            model = BaggingRegressor(
                estimator=base_model,
                n_estimators=n_ensemble,
                random_state=seed,
                n_jobs=1,
            )
    elif model_name == "rf":
        if ensemble == "voting":
            model = VotingRegressor(
                estimators=[
                    (
                        f"rf_{k}",
                        DecisionTreeRegressor(random_state=seed + k),
                    )
                    for k in range(n_ensemble)
                ],
                n_jobs=1,
            )
        elif ensemble == "bagging":
            model = RandomForestRegressor(
                n_estimators=n_ensemble, random_state=seed, n_jobs=1
            )
    elif model_name == "linear":
        if ensemble == "voting":
            model = VotingRegressor(
                estimators=[
                    (
                        f"ridge_{k}",
                        RidgeCV(),
                    )
                    for k in range(n_ensemble)
                ],
                n_jobs=1,
            )
        elif ensemble == "bagging":
            base_model = RidgeCV()
            model = BaggingRegressor(
                estimator=base_model,
                n_estimators=n_ensemble,
                random_state=seed,
                n_jobs=1,
            )
    elif model_name == "tabicl":
        from tabicl import TabICLRegressor

        model = BaggingRegressor(
            estimator=TabICLRegressor(n_estimators=1),
            n_estimators=n_ensemble,
            random_state=seed,
        )
    else:
        raise ValueError("Model name not recognized.")
    return model


def joblib_fit_one(
    X,
    y,
    train_index,
    model,
    fold_id,
    seed,
    cache_dir=None,
):
    """
    Utility function to fit a model and save it to cache using joblib. This is used
    for parallel fitting of models.

    Parameters
    ----------
    X : array-like
        Feature matrix.
    y : array-like
        Target vector.
    train_index : array-like
        Indices for training data.
    model : sklearn estimator
        Model to fit.
    fold_id : int
        Fold identifier.
    seed : int
        Random seed for reproducibility.
    cache_dir : str or None, optional
        Directory to save the fitted model. If None, model is not saved.

    Returns
    -------
    model_c : sklearn estimator
        Fitted model.
    """
    model_c = clone(model)
    model_c.fit(X[train_index], y[train_index])
    if cache_dir is not None:
        joblib.dump(model_c, Path(cache_dir) / f"model_{fold_id}_{seed}.pkl")
    return model_c


def loco_one(X, y, train_index, test_index, model, fold_id):
    """
    Compute LOCO importances for a given fold. This function is used for parallel
    computation of LOCO importances across CV folds.

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

    Returns
    -------
    output_list : list of pd.DataFrame
        List of dataframes containing LOCO importances for the ensemble and sub-models.
    """
    n_features = X.shape[1]
    output_list = []
    if not check_is_fitted(model):
        model_c = clone(model)
        model_c.fit(X[train_index], y[train_index])
    else:
        model_c = model
    loco = LOCO(model_c, loss=mean_squared_error)
    loco.fit(X[train_index], y[train_index])
    importances_full = loco.importance(X[test_index], y[test_index])
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
    sub_model_list = get_sub_models(model_c)
    for i, sub_model in enumerate(sub_model_list):
        sub_model_c = copy(sub_model)
        loco_sub = LOCO(sub_model_c, loss=mean_squared_error)
        loco_sub.fit(X[train_index], y[train_index])
        importances_sub = loco_sub.importance(X[test_index], y[test_index])
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


def pfi_one(X, y, train_index, test_index, model, fold_id, seed):
    """
    Compute PFI importances for a given fold. This function is used for parallel
    computation of PFI importances across CV folds.

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

    Returns
    -------
    output_list : list of pd.DataFrame
        List of dataframes containing PFI importances for the ensemble and sub-models.
    """
    n_features = X.shape[1]
    output_list = []
    if not check_is_fitted(model):
        model_c = clone(model)
        model_c.fit(X[train_index], y[train_index])
    else:
        model_c = model
    pfi = PFI(
        model_c,
        n_permutations=100,
        random_state=seed,
        loss=mean_squared_error,
    )
    pfi.fit(X[train_index], y[train_index])
    importances_full = pfi.importance(X[test_index], y[test_index])
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
    sub_model_list = get_sub_models(model_c)
    for i, sub_model in enumerate(sub_model_list):
        sub_model_c = copy(sub_model)
        pfi_sub = PFI(
            sub_model_c,
            n_permutations=100,
            random_state=seed,
            loss=mean_squared_error,
        )
        pfi_sub.fit(X[train_index], y[train_index])
        importances_sub = pfi_sub.importance(X[test_index], y[test_index])
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


def cfi_one(X, y, train_index, test_index, model, fold_id, seed):
    """
    Compute CFI importances for a given fold. This function is used for parallel
    computation of CFI importances across CV folds.

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

    Returns
    -------
    output_list : list of pd.DataFrame
        List of dataframes containing CFI importances for the ensemble and sub-models.
    """
    n_features = X.shape[1]
    output_list = []
    if not check_is_fitted(model):
        model_c = clone(model)
        model_c.fit(X[train_index], y[train_index])
    else:
        model_c = model
    cfi = CFI(
        model_c,
        imputation_model_continuous=RidgeCV(),
        n_permutations=100,
        random_state=seed,
        loss=mean_squared_error,
    )
    cfi.fit(X[train_index], y[train_index])
    importances_full = cfi.importance(X[test_index], y[test_index])
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
    sub_model_list = get_sub_models(model_c)
    for i, sub_model in enumerate(sub_model_list):
        sub_model_c = copy(sub_model)
        cfi_sub = CFI(
            sub_model_c,
            imputation_model_continuous=RidgeCV(),
            n_permutations=100,
            random_state=seed,
            loss=mean_squared_error,
        )
        cfi_sub.fit(X[train_index], y[train_index])
        importances_sub = cfi_sub.importance(X[test_index], y[test_index])
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


def sage_one(X, y, train_index, test_index, model, fold_id, seed):
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

    if X_test.shape[0] > 1024:
        impute_ids = np.random.choice(X_test.shape[0], size=1024, replace=False)
        X_impute = X_test[impute_ids]
    else:
        X_impute = X_test
    imputer = sage.MarginalImputer(model_c, X_impute)
    estimator = sage.PermutationEstimator(imputer, "mse", random_state=seed)
    sage_values = estimator(X_test, y_test)
    output_list.append(
        pd.DataFrame(
            {
                "feature": np.arange(n_features),
                "importance": sage_values.values,
                "fold": fold_id,
                "model": "ensemble",
            }
        )
    )

    sub_model_list = get_sub_models(model_c)
    for i, sub_model in enumerate(sub_model_list):
        sub_model_c = copy(sub_model)
        imputer = sage.MarginalImputer(sub_model_c, X_impute)
        estimator = sage.PermutationEstimator(imputer, "mse", random_state=seed)
        sage_values = estimator(X_test, y_test)
        output_list.append(
            pd.DataFrame(
                {
                    "feature": np.arange(n_features),
                    "importance": sage_values.values,
                    "fold": fold_id,
                    "model": f"sub_model_{i}",
                }
            )
        )
    return output_list


# %%


def main(args):

    n_samples_list = args.n_samples
    seed = args.seed
    n_jobs = args.n_jobs
    n_splits = args.n_splits
    snr = args.snr
    n_ensemble = args.n_ensemble
    n_features = args.n_features
    dataset_name = args.dataset_name
    model_name = args.model_name
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    compute_sage = args.sage
    ensemble = args.ensemble

    for n_samples in n_samples_list:
        snr_string = int(snr) if snr >= 1 else "0" + str(int(snr * 100))
        data_str = dataset_name
        cache_dir = (
            results_dir
            / f"{data_str}_{model_name}_n{n_samples}_p{n_features}_{ensemble}{n_ensemble}_snr{snr_string}"
        )
        cache_dir.mkdir(parents=True, exist_ok=True)

        # prediction model
        model = get_model(model_name, n_ensemble, ensemble, seed)

        # generate data
        X, y, support, support_bis = get_dataset(
            dataset_name=dataset_name,
            n_samples=n_samples,
            n_features=n_features,
            random_state=seed,
            snr=snr,
        )

        np.save(cache_dir / f"support_{data_str}_{seed}.npy", support)
        np.save(cache_dir / f"support_bis_{data_str}_{seed}.npy", support_bis)
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        metric_dict = {"r2": r2_score, "mse": mean_squared_error}

        models_dir = cache_dir / "models"
        model_files = [
            models_dir / f"model_{fold_id}_{seed}.pkl" for fold_id in range(n_splits)
        ]
        if all([model_file.exists() for model_file in model_files]):
            print("Models already fitted. Skipping fitting.")
            fitted_list = [joblib.load(model_file) for model_file in model_files]
        else:
            print("Fitting models...")
            models_dir.mkdir(parents=True, exist_ok=True)
            fitted_list = Parallel(n_jobs=n_jobs)(
                delayed(joblib_fit_one)(
                    X,
                    y,
                    train_index,
                    model,
                    fold_id,
                    seed,
                    cache_dir=models_dir,
                )
                for fold_id, (train_index, _) in enumerate(cv.split(X, y))
            )

        # compute CV score
        scores_path = cache_dir / f"scores_{data_str}_{seed}.csv"
        if scores_path.exists():
            print(f"Scores file {scores_path} already exists. Skipping computation.")
        else:
            print("Computing CV scores...")
            scores_list = []
            for i, (_, test_index) in enumerate(cv.split(X, y)):
                model_c = fitted_list[i]
                y_pred = model_c.predict(X[test_index])

                scores_list.append(
                    pd.DataFrame(
                        {
                            "score": metric(y[test_index], y_pred),
                            "model": "ensemble",
                            "metric": metric_name,
                            "fold": i,
                        }
                        for metric_name, metric in metric_dict.items()
                    )
                )
                sub_model_list = get_sub_models(model_c)
                for sub_model in sub_model_list:
                    sub_model_c = copy(sub_model)

                    y_pred_sub = sub_model_c.predict(X[test_index])
                    scores_list.append(
                        pd.DataFrame(
                            {
                                "score": metric(y[test_index], y_pred_sub),
                                "model": "sub_models",
                                "metric": metric_name,
                                "fold": i,
                            }
                            for metric_name, metric in metric_dict.items()
                        )
                    )

            df_scores = pd.concat(scores_list, axis=0).reset_index(drop=True)
            df_scores.to_csv(cache_dir / f"scores_{data_str}_{seed}.csv", index=False)

        # compute VIMs with LOCO
        print("Computing VIMs with LOCO...")
        loco_path = cache_dir / f"loco_{data_str}_{seed}.csv"
        if loco_path.exists():
            print(f"LOCO file {loco_path} already exists. Skipping computation.")
        else:
            loco_output = Parallel(n_jobs=n_jobs)(
                delayed(loco_one)(
                    X,
                    y,
                    train_index,
                    test_index,
                    fitted_list[fold_id],
                    fold_id,
                )
                for fold_id, (train_index, test_index) in enumerate(cv.split(X, y))
            )
            loco_df = pd.concat(
                [item for sublist in loco_output for item in sublist], axis=0
            )
            loco_df.to_csv(loco_path, index=False)

        # compute VIMs with CFI
        print("Computing VIMs with CFI...")
        cfi_path = cache_dir / f"cfi_{data_str}_{seed}.csv"
        if cfi_path.exists():
            print(f"CFI file {cfi_path} already exists. Skipping computation.")
        else:
            cfi_output = Parallel(n_jobs=n_jobs)(
                delayed(cfi_one)(
                    X,
                    y,
                    train_index,
                    test_index,
                    fitted_list[fold_id],
                    fold_id,
                    seed,
                )
                for fold_id, (train_index, test_index) in enumerate(cv.split(X, y))
            )
            cfi_df = pd.concat(
                [item for sublist in cfi_output for item in sublist], axis=0
            )
            cfi_df.to_csv(cfi_path, index=False)

        # compute VIMs with PFI
        print("Computing VIMs with PFI...")
        pfi_path = cache_dir / f"pfi_{data_str}_{seed}.csv"
        if pfi_path.exists():
            print(f"PFI file {pfi_path} already exists. Skipping computation.")
        else:
            pfi_output = Parallel(n_jobs=n_jobs)(
                delayed(pfi_one)(
                    X,
                    y,
                    train_index,
                    test_index,
                    fitted_list[fold_id],
                    fold_id,
                    seed,
                )
                for fold_id, (train_index, test_index) in enumerate(cv.split(X, y))
            )
            pfi_df = pd.concat(
                [item for sublist in pfi_output for item in sublist], axis=0
            )
            pfi_df.to_csv(pfi_path, index=False)

        # compute VIMs with SAGE
        if compute_sage:
            print("Computing VIMs with SAGE...")
            sage_path = cache_dir / f"sage_{data_str}_{seed}.csv"
            if sage_path.exists():
                print(f"SAGE file {sage_path} already exists. Skipping computation.")
            else:
                sage_output = Parallel(n_jobs=n_jobs)(
                    delayed(sage_one)(
                        X,
                        y,
                        train_index,
                        test_index,
                        fitted_list[fold_id],
                        fold_id,
                        seed,
                    )
                    for fold_id, (train_index, test_index) in enumerate(cv.split(X, y))
                )
                sage_df = pd.concat(
                    [item for sublist in sage_output for item in sublist], axis=0
                )
                sage_df.to_csv(sage_path, index=False)
        else:
            print("Skipping SAGE computation.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
    print("Done.")
