from copy import copy
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

from data import get_dataset
from simulation import (
    cfi_one,
    get_model,
    get_sub_models,
    joblib_fit_one,
    loco_one,
    parse_args,
    sage_one,
)


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
    ensemble_type = args.ensemble

    for n_samples in n_samples_list:
        data_str = dataset_name
        cache_dir = (
            results_dir
            / f"asympt_n{n_samples}_{data_str}_{model_name}_p{n_features}_{ensemble_type}{n_ensemble}"
        )
        cache_dir.mkdir(parents=True, exist_ok=True)

        # prediction model
        model = get_model(model_name, n_ensemble, ensemble_type, seed)

        # generate data
        X, y, support, support_bis = get_dataset(
            dataset_name=dataset_name,
            n_samples=n_samples,
            n_features=n_features,
            random_state=seed,
            snr=snr,
        )

        cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        metric_dict = {"r2": r2_score, "mse": mean_squared_error}

        # fit models
        print("Fitting models...")
        models_dir = cache_dir / "models"
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
        loco_path = cache_dir / f"asympt_n{int(X.shape[0])}_loco_{data_str}_{seed}.csv"
        if loco_path.exists():
            print(f"File {loco_path} already exists, skipping...")
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
            loco_df.to_csv(
                loco_path,
                index=False,
            )

        # compute VIMs with CFI
        print("Computing VIMs with CFI...")
        cfi_path = cache_dir / f"asympt_n{int(X.shape[0])}_cfi_{data_str}_{seed}.csv"
        if cfi_path.exists():
            print(f"File {cfi_path} already exists, skipping...")
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
            cfi_df.to_csv(
                cfi_path,
                index=False,
            )

        # compute VIMs with SAGE
        print("Computing VIMs with SAGE...")
        sage_path = cache_dir / f"asympt_n{int(X.shape[0])}_sage_{data_str}_{seed}.csv"
        if sage_path.exists():
            print(f"File {sage_path} already exists, skipping...")
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
            sage_df.to_csv(
                sage_path,
                index=False,
            )


if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)
    print("Done.")
