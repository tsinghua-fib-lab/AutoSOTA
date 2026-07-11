"""
Sensitivity studies.

* :func:`run_lambda_sensitivity_experiment` sweeps the Fisher regularisation
  parameter for one (model, prompt) pair and reports label-propagation
  performance for each value.
* :func:`run_full_lambda_sensitivity_study` runs the above over the entire
  dataset, with optional per-pair caching.

Training-set-size sensitivity is handled inside the standard
:func:`~aporia.label_propagation.run_full_label_propagation_study` by
passing a list of ``train_fractions``; this module just provides a thin
wrapper for the corresponding figure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tqdm.auto import tqdm

from .caching import ensure_dir, per_pair_path, write_json
from .config import Config
from .data import extract_prompt_data, generate_fixed_test_sets
from .evaluation import LabelPropagationEvaluator
from .label_propagation import WassersteinLabelPropagator
from .projections import FisherProjection


# ============================================================
# ================ LAMBDA SENSITIVITY (PAIR) =================
# ============================================================

def run_lambda_sensitivity_experiment(
    df: pd.DataFrame,
    model_id: int,
    prompt_id: int,
    lambda_values: list[float] | np.ndarray,
    cfg: Config,
    test_fraction: float = 0.2,
    n_splits: int = 5,
    random_state: int | None = None,
    logskip: bool = False,
) -> pd.DataFrame | None:
    """Sweep ``lambda_reg`` for one (model, prompt) pair on full training data."""
    random_state = cfg.experiment.random_state if random_state is None else random_state

    X, y = extract_prompt_data(df, model_id, prompt_id, cfg)
    n_H = int(y.sum())
    n_G = len(y) - n_H
    if n_H < cfg.experiment.min_per_class or n_G < cfg.experiment.min_per_class:
        if not logskip:
            print(
                f"Skipping model {model_id}, prompt {prompt_id} "
                f"(n_G={n_G}, n_H={n_H})"
            )
        return None

    trn_sets, tst_sets = generate_fixed_test_sets(
        X, y,
        n_splits=n_splits,
        test_fraction=test_fraction,
        random_state=random_state,
    )

    results: list[dict] = []

    for split_id, ((X_train, y_train), (X_test, y_test)) in enumerate(
        zip(trn_sets, tst_sets)
    ):
        for lambda_reg in lambda_values:
            detector = WassersteinLabelPropagator(
                FisherProjection(lambda_reg=float(lambda_reg))
            ).fit(X_train, y_train)

            evaluator = LabelPropagationEvaluator(
                detector, X_test, y_test, fisher_ref=detector,
            )
            metrics = evaluator.evaluate()
            metrics.update({
                "lambda_reg": float(lambda_reg),
                "split_id":   split_id,
                "model_id":   model_id,
                "prompt_id":  prompt_id,
                "n_train":    len(X_train),
                "n_test":     len(X_test),
            })
            results.append(metrics)

    return pd.DataFrame(results) if results else None


# ============================================================
# ================ LAMBDA SENSITIVITY (FULL) =================
# ============================================================

def run_full_lambda_sensitivity_study(
    df: pd.DataFrame,
    cfg: Config,
    lambda_values: list[float] | np.ndarray,
    test_fraction: float = 0.2,
    n_splits: int = 5,
    random_state: int | None = None,
    use_cache: bool = False,
    cache_dir: str | None = None,
    overwrite_cache: bool = False,
    logskip: bool = False,
    model_ids: list[int] | None = None,
    prompt_ids_by_model: dict[int, list[int]] | None = None,
) -> pd.DataFrame:
    """Lambda-sensitivity sweep over the full dataset."""
    random_state = cfg.experiment.random_state if random_state is None else random_state

    if cache_dir is None:
        cache_dir = f"{cfg.cache.root}/lambda_sensitivity"
    if use_cache:
        ensure_dir(cache_dir)

    if prompt_ids_by_model is None:
        from .data import prompt_ids_by_model as _ids
        prompt_ids_by_model = _ids(df, cfg)
    if model_ids is None:
        model_ids = list(prompt_ids_by_model.keys())

    all_results: list[pd.DataFrame] = []

    for mid in tqdm(model_ids, desc="Model"):
        for pid in tqdm(prompt_ids_by_model[mid], desc="Prompt", leave=False):

            cache_path = None
            if use_cache:
                cache_path = per_pair_path(cache_dir, model=mid, prompt=pid)
                if cache_path.exists() and not overwrite_cache:
                    all_results.append(pd.read_parquet(cache_path))
                    continue

            res_df = run_lambda_sensitivity_experiment(
                df=df,
                model_id=mid,
                prompt_id=pid,
                lambda_values=lambda_values,
                cfg=cfg,
                test_fraction=test_fraction,
                n_splits=n_splits,
                random_state=random_state,
                logskip=logskip,
            )

            if res_df is None or len(res_df) == 0:
                continue

            all_results.append(res_df)

            if use_cache and cache_path is not None:
                res_df.to_parquet(cache_path, index=False)

    if not all_results:
        return pd.DataFrame()

    results = pd.concat(all_results, ignore_index=True)

    if use_cache:
        write_json(
            f"{cache_dir}/meta.json",
            {
                "model_ids":     list(model_ids),
                "lambda_values": [float(x) for x in lambda_values],
                "test_fraction": test_fraction,
                "n_splits":      n_splits,
                "random_state":  random_state,
                "n_cached_pairs": len(all_results),
                "dataset_name":  cfg.dataset.name,
            },
        )

    return results



# ============================================================
# ============== POST-EXPERIMENT AGGREGATION =================
# ============================================================
#
# The following helpers operate on the DataFrames returned by
# :func:`run_full_lambda_sensitivity_study`.  They were originally
# duplicated across :file:`LabelPropagation__LambdaSensitivity.ipynb`
# but are reusable analysis utilities that belong in the library.

def aggregate_metric_over_lambda(
    df: pd.DataFrame,
    metric: str        = "f1",
    score_metric: str  = "accuracy",
    agg_prompts: bool  = True,
    agg_models: bool   = False,
) -> pd.DataFrame:
    """Aggregate a metric and a score across runs as a function of ``lambda_reg``.

    Returns one row per ``(model_id, lambda_reg)`` unless ``agg_models=True``
    or ``agg_prompts=False`` are passed.
    """
    group_cols = ["lambda_reg"]
    if not agg_models:
        group_cols.insert(0, "model_id")
    if not agg_prompts:
        group_cols.insert(1, "prompt_id")

    return (
        df.groupby(group_cols)
          .agg(
              metric_mean=(metric, "mean"),
              metric_std=(metric, "std"),
              score_mean=(score_metric, "mean"),
              score_std=(score_metric, "std"),
              n_runs=(metric, "count"),
          )
          .reset_index()
          .sort_values("lambda_reg")
    )


def aggregate_over_runs(df: pd.DataFrame, metric: str = "f1") -> pd.DataFrame:
    """Average ``metric`` over splits/iterations, grouping by (model, prompt, lambda)."""
    return (
        df.groupby(["model_id", "prompt_id", "lambda_reg"])[metric]
          .mean()
          .reset_index(name=metric)
    )


def compute_best_scores(df_agg: pd.DataFrame, metric: str = "f1") -> pd.DataFrame:
    """Per-task best metric value: ``s*_{m,p} = max_lambda s(m, p, lambda)``."""
    return (
        df_agg.groupby(["model_id", "prompt_id"])[metric]
              .max()
              .reset_index(name="best_score")
    )


def compute_relative_loss(
    df_agg: pd.DataFrame,
    df_best: pd.DataFrame,
    metric: str = "f1",
) -> pd.DataFrame:
    """Add a ``relative_loss = (best_score - metric) / best_score`` column."""
    df = df_agg.merge(df_best, on=["model_id", "prompt_id"], how="left")
    df["relative_loss"] = (df["best_score"] - df[metric]) / df["best_score"]
    return df


def aggregate_relative_loss(df_rel: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the per-task relative loss into ``L(lambda)`` (mean/std/max)."""
    return (
        df_rel.groupby("lambda_reg")["relative_loss"]
              .agg(["mean", "std", "max"])
              .reset_index()
              .rename(columns={
                  "mean": "rel_loss_mean",
                  "std":  "rel_loss_std",
                  "max":  "rel_loss_max",
              })
    )


def select_lambda_min_regret(df_lambda_loss: pd.DataFrame) -> pd.Series:
    """The row of ``df_lambda_loss`` with smallest ``rel_loss_mean``."""
    return df_lambda_loss.loc[df_lambda_loss["rel_loss_mean"].idxmin()]


def compute_average_best_lambda(df_agg: pd.DataFrame, metric: str = "f1") -> dict:
    """Mean and median of the per-task argmax-lambda values."""
    best_per_task = (
        df_agg.sort_values(metric, ascending=False)
              .groupby(["model_id", "prompt_id"])
              .first()
              .reset_index()
    )
    return {
        "mean_best_lambda":   best_per_task["lambda_reg"].mean(),
        "median_best_lambda": best_per_task["lambda_reg"].median(),
    }


def global_stats(
    grp: pd.DataFrame,
    n_denom: int,
) -> pd.Series:
    """Per-``N`` summary used by the sample-size ablation tables.

    Operates on the per-(model, prompt, N) summary frame produced by
    aggregating :func:`run_structural_ablation_for_prompt` results.
    ``n_denom`` is the per-N denominator (typically the number of
    (model, prompt) pairs that were tested at that N).
    """
    n_tested = len(grp)
    n_sig_w  = grp["sig_wass_majority"].sum()
    n_sig_mw = grp["sig_mw_majority"].sum()
    n_above  = grp["above_null_mean"].sum()
    return pd.Series({
        "n_denom":           n_denom,
        "n_tested":          n_tested,
        # Wasserstein
        "n_sig_wass":        int(n_sig_w),
        "frac_sig_wass":     n_sig_w  / n_denom,
        # Mann-Whitney
        "n_sig_mw":          int(n_sig_mw),
        "frac_sig_mw":       n_sig_mw / n_denom,
        # W_obs > null mean (weaker, no significance threshold)
        "n_above":           int(n_above),
        "frac_above_fixed":  n_above / n_denom,
        "frac_above_tested": n_above / n_tested if n_tested > 0 else np.nan,
        "1-p_wass_mean":     1 - grp["p_wass_mean"].mean(),
        # W stats
        "W_obs_mean":        grp["W_obs_mean"].mean(),
        "W_obs_std":         grp["W_obs_mean"].std(),
        "W_null_mean":       grp["W_null_mean"].mean(),
        "W_null_std":        grp["W_null_mean"].std(),
        # separability
        "sep_mean":          grp["sep_mean"].mean(),
        "sep_std":           grp["sep_mean"].std(),
        "sep_z_mean":        grp["sep_z_mean"].mean(),
        "sep_z_std":         grp["sep_z_mean"].std(),
    })
