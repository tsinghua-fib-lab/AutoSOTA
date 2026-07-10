from pathlib import Path

import numpy as np
import polars as pl
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


def read_one(
    results_dir: Path,
    dataset_name: str,
    n: int,
    model_name: str,
    ensemble_name: str,
    method_name: str,
    seed: int,
    n_ensemble: int = 10,
    snr: int = 1,
    p: int = 20,
) -> pl.DataFrame:
    """
    Read a single importance csv file. Adds a columns indicating whether the feature is
    in the support or not. Aggregates the importance over sub-models.

    Parameters
    ----------
    results_dir : Path
        Directory where results are stored.
    dataset_name : str
        Name of the dataset (friedman1, ishigami, g_function)
    n : int
        Number of samples.
    model_name : str
        Name of the model (mlp, rf)
    ensemble_name : str
        Name of the ensemble method (voting, bagging)
    method_name : str
        Name of the importance method (loco, sage, cfi)
    seed : int
        Random seed.
    n_ensemble : int, optional
        Number of models in the ensemble, by default 10
    snr : int, optional
        Signal to noise ratio, by default 1
    p : int, optional
        Number of features, by default 20

    Returns
    -------
    pl.DataFrame
        DataFrame with importance values averaged over sub-models and support column.
    """
    sub_dir = (
        results_dir
        / f"{dataset_name}_{model_name}_n{n}_p{p}_{ensemble_name}{n_ensemble}_snr{snr}"
    )

    importance_df = pl.read_csv(sub_dir / f"{method_name}_{dataset_name}_{seed}.csv")
    support_bis = np.load(sub_dir / f"support_bis_{dataset_name}_{seed}.npy")

    importance_df = importance_df.with_columns(
        support=pl.col("feature").is_in(support_bis.tolist())
    )
    importance_df = importance_df.with_columns(
        strategy=pl.when(pl.col("model") == "ensemble")
        .then(pl.lit("ensemble"))
        .otherwise(pl.lit("sub-models")),
    ).drop("model")

    importance_df_mean = (
        importance_df.with_columns(
            dataset_name=pl.lit(dataset_name),
            n=pl.lit(n),
            model_name=pl.lit(model_name),
            ensemble_name=pl.lit(ensemble_name),
            seed=pl.lit(seed),
            method_name=pl.lit(method_name),
        )
        .group_by(
            [
                "feature",
                "fold",
                "support",
                "dataset_name",
                "n",
                "model_name",
                "ensemble_name",
                "strategy",
                "seed",
                "method_name",
            ]
        )
        .agg(pl.col("importance").mean().alias("importance"))
    )

    return importance_df_mean


def try_read_one(*args, **kwargs):
    """
    Utils function to read one file with error handling.
    """
    try:
        return read_one(*args, **kwargs)
    except Exception as e:
        print(f"Error reading with args {args}, kwargs {kwargs}: {e}")
        return None


def read_outputs(
    results_dir: Path,
    dataset_name: str,
    model_name: list[str],
    ensemble_name: list[str],
    method_name: list[str],
    n_list: list[int],
    seeds: list[int],
    n_jobs: int = 10,
    p=20,
    n_ensemble=10,
) -> pl.DataFrame:
    """
    Parallelized reading of multiple importance csv files. Also reads asymptotic results
    and merges them with the main results.

    Parameters
    ----------
    results_dir : Path
        Directory where results are stored.
    dataset_name : str
        Name of the dataset (friedman1, ishigami, g_function)
    model_name : list[str]
        List of model names (mlp, rf)
    ensemble_name : list[str]
        List of ensemble methods (voting, bagging)
    method_name : list[str]
        List of importance methods (loco, sage, cfi)
    n_list : list[int]
        List of number of samples.
    seeds : list[int]
        List of random seeds.
    n_jobs : int, optional
        Number of parallel jobs, by default 10
    p : int, optional
        Number of features, by default 20
    n_ensemble : int, optional
        Number of models in the ensemble, by default 10

    Returns
    -------
    pl.DataFrame
        DataFrame with all importance values and asymptotic importance values merged.
    """
    out_list = Parallel(n_jobs=n_jobs)(
        delayed(try_read_one)(
            results_dir,
            dataset_name,
            n,
            m_name,
            e_name,
            mt_name,
            seed,
        )
        for seed in tqdm(seeds)
        for n in n_list
        for m_name in model_name
        for e_name in ensemble_name
        for mt_name in method_name
    )
    list_df = pl.concat([x for x in out_list if x is not None])
    asymp_list = []
    for m_name in model_name:
        for e_name in ensemble_name:
            for mt_name in method_name:
                asymptotic_folder = (
                    results_dir
                    / f"asympt_n100000_{dataset_name}_{m_name}_p{p}_{e_name}{n_ensemble}"
                )
                asymptotic_df = (
                    pl.read_csv(
                        asymptotic_folder
                        / f"asympt_n100000_{mt_name}_{dataset_name}_1.csv"
                    )
                    .with_columns(
                        dataset_name=pl.lit(dataset_name),
                        n=pl.lit(100000),
                        model_name=pl.lit(m_name),
                        method_name=pl.lit(mt_name),
                        strategy=pl.when(pl.col("model") == "ensemble")
                        .then(pl.lit("ensemble"))
                        .otherwise(pl.lit("sub-models")),
                        ensemble_name=pl.lit(e_name),
                    )
                    .drop("model")
                    .group_by(
                        [
                            "feature",
                            "fold",
                            "dataset_name",
                            "n",
                            "model_name",
                            "ensemble_name",
                            "strategy",
                            "method_name",
                        ]
                    )
                    .agg(pl.col("importance").mean().alias("importance"))
                )
                join_cols = [
                    c
                    for c in list_df.columns
                    if c in asymptotic_df.columns and c not in ("importance", "n")
                ]
                asymp_list.append(
                    asymptotic_df.select(
                        [
                            *(pl.col(c) for c in join_cols),
                            pl.col("importance").alias("asymptotic_importance"),
                        ]
                    )
                )
    asymptotic_join = pl.concat(asymp_list).unique()
    df_merged = list_df.join(asymptotic_join, on=join_cols, how="left")
    return df_merged


def compute_mse_per_seed(df):
    return (
        df.with_columns(
            squarred_error=(pl.col("importance") - pl.col("asymptotic_importance")) ** 2
        )
        .group_by(
            [
                "dataset_name",
                "n",
                "model_name",
                "ensemble_name",
                "strategy",
                "method_name",
                "support",
                "seed",
            ]
        )
        .agg(
            pl.col("squarred_error").mean().alias("mse"),
        )
        .with_columns(rmse=pl.col("mse").sqrt())
    )


def compute_auc_per_seed(df, threshold=1e-3):
    return (
        df
        # 1. Define Target (y) and Score (y_score)
        .with_columns(
            y=(pl.col("asymptotic_importance") > threshold).cast(pl.Int8),
            y_score=pl.col("importance"),
        )
        # 2. Group by SEED (and experiment settings)
        #    This collapses the 'feature' dimension into lists for AUC calculation
        .group_by(
            [
                "dataset_name",
                "n",
                "model_name",
                "ensemble_name",
                "strategy",
                "method_name",
                "seed",  # <--- Calculate one AUC per seed
            ]
        ).agg(
            [
                pl.col("y"),
                pl.col("y_score"),
            ]
        )
        # 3. Compute AUC for each seed using sklearn
        .with_columns(
            roc_auc=pl.struct(["y", "y_score"]).map_elements(
                lambda row: (
                    roc_auc_score(row["y"], row["y_score"])
                    if len(set(row["y"])) > 1
                    else float("nan")
                ),
                return_dtype=pl.Float64,
            )
        )
        # 4. Cleanup
        .drop(["y", "y_score"])
    )


def read_one_score(
    results_dir: Path,
    dataset_name: str,
    n: int,
    model_name: str,
    ensemble_name: str,
    seed: int,
    n_ensemble: int = 10,
    snr: int = 1,
    p: int = 20,
) -> pl.DataFrame:
    """
    Read a single importance csv file. Adds a columns indicating whether the feature is
    in the support or not. Aggregates the importance over sub-models.

    Parameters
    ----------
    results_dir : Path
        Directory where results are stored.
    dataset_name : str
        Name of the dataset (friedman1, ishigami, g_function)
    n : int
        Number of samples.
    model_name : str
        Name of the model (mlp, rf)
    ensemble_name : str
        Name of the ensemble method (voting, bagging)
    method_name : str
        Name of the importance method (loco, sage, cfi)
    seed : int
        Random seed.
    n_ensemble : int, optional
        Number of models in the ensemble, by default 10
    snr : int, optional
        Signal to noise ratio, by default 1
    p : int, optional
        Number of features, by default 20

    Returns
    -------
    pl.DataFrame
        DataFrame with importance values averaged over sub-models and support column.
    """
    sub_dir = (
        results_dir
        / f"{dataset_name}_{model_name}_n{n}_p{p}_{ensemble_name}{n_ensemble}_snr{snr}"
    )

    scores_df = pl.read_csv(sub_dir / f"scores_{dataset_name}_{seed}.csv")

    scores_df = scores_df.with_columns(
        dataset_name=pl.lit(dataset_name),
        n=pl.lit(n),
        model_name=pl.lit(model_name),
        ensemble_name=pl.lit(ensemble_name),
        seed=pl.lit(seed),
    )

    return scores_df


def try_read_one_score(*args, **kwargs):
    """
    Utils function to read one file with error handling.
    """
    try:
        return read_one_score(*args, **kwargs)
    except Exception as e:
        print(f"Error reading with args {args}, kwargs {kwargs}: {e}")
        return None


def read_scores(
    results_dir: Path,
    dataset_name: str,
    model_name: list[str],
    ensemble_name: list[str],
    n_list: list[int],
    seeds: list[int],
    n_jobs: int = 10,
    p=20,
    n_ensemble=10,
) -> pl.DataFrame:
    """
    Parallelized reading of multiple importance csv files. Also reads asymptotic results
    and merges them with the main results.

    Parameters
    ----------
    results_dir : Path
        Directory where results are stored.
    dataset_name : str
        Name of the dataset (friedman1, ishigami, g_function)
    model_name : list[str]
        List of model names (mlp, rf)
    ensemble_name : list[str]
        List of ensemble methods (voting, bagging)
    n_list : list[int]
        List of number of samples.
    seeds : list[int]
        List of random seeds.
    n_jobs : int, optional
        Number of parallel jobs, by default 10
    p : int, optional
        Number of features, by default 20
    n_ensemble : int, optional
        Number of models in the ensemble, by default 10

    Returns
    -------
    pl.DataFrame
        DataFrame with all importance values and asymptotic importance values merged.
    """
    out_list = Parallel(n_jobs=n_jobs)(
        delayed(try_read_one_score)(
            results_dir,
            dataset_name,
            n,
            m_name,
            e_name,
            seed,
        )
        for seed in tqdm(seeds)
        for n in n_list
        for m_name in model_name
        for e_name in ensemble_name
    )
    list_df = pl.concat([x for x in out_list if x is not None])
    return list_df
