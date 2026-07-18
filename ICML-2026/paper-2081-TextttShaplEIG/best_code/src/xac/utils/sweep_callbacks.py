import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from hydra.experimental.callback import Callback

from xac.utils.plotting import get_plot_name, plot_trajectory

log = logging.getLogger(__name__)

SHAPLEIG_INDEX_SET_FULL = [
    "Regression MSR",
    "LeverageSHAP",
    "KernelSHAP",
    "Permutation Sampling", #"SVARM",
    "LeverageSHAP-GP",
    "EIG-EP",
    "Random",
    "EIG-FP",
]

SHAPLEIG_PLOT_SUBSETS = [
    ("full", SHAPLEIG_INDEX_SET_FULL),
    (
        "sv_accuracy",
        [
            "Regression MSR",
            "LeverageSHAP",
            "KernelSHAP",
            "Permutation Sampling", #"SVARM",
            "EIG-FP",
        ],
    ),
    ("sv_baselines", ["LeverageSHAP-GP", "EIG-EP", "Random", "EIG-FP"]),
]

MAE_MSE_QUANTILES = (0.95, 0.97, 0.99, 0.999)


def _coerce_metric_values(metric_values) -> torch.Tensor:
    parsed_values = []

    for value in metric_values:
        if isinstance(value, str):
            stripped_value = value.strip()

            if stripped_value.startswith("[") and stripped_value.endswith("]"):
                stripped_value = stripped_value[1:-1].strip()
                value = (
                    np.fromstring(stripped_value, sep=",", dtype=float).tolist()
                    if stripped_value
                    else []
                )
            else:
                value = float(stripped_value)

        parsed_values.append(value)

    return torch.as_tensor(parsed_values, dtype=torch.float64)


def _metric_has_observed_values(metric_values) -> bool:
    metric_tensor = _coerce_metric_values(metric_values)
    return (~torch.isnan(metric_tensor)).any().item()


def _get_metric_categories(grouped_df: pd.DataFrame, metric_name: str):
    category_names = []
    category_values = []

    for category_name, metric_values in grouped_df[metric_name].items():
        if _metric_has_observed_values(metric_values):
            category_names.append(category_name)
            category_values.append(_coerce_metric_values(metric_values))

    return category_names, category_values


def _iter_plot_subsets(category_names, category_values):
    values_by_name = dict(zip(category_names, category_values))

    for subset_name, subset_categories in SHAPLEIG_PLOT_SUBSETS:
        if not set(subset_categories).issubset(values_by_name):
            continue

        yield (
            subset_name,
            subset_categories,
            [values_by_name[category] for category in subset_categories],
        )


def _get_quantile_suffix(quantile: float) -> str:
    return str(quantile).replace("0.", "")


def _get_y_minimum(category_values) -> float:
    mean_values = torch.concat([values.mean(axis=0) for values in category_values])
    minimum_mean = mean_values.min()

    if any(values.shape[0] <= 1 for values in category_values):
        return minimum_mean.item()

    sem_values = torch.concat(
        [
            values.std(dim=0, unbiased=True) / np.sqrt(values.shape[0])
            for values in category_values
        ]
    )
    return (minimum_mean - 2 * sem_values[mean_values.argmin()]).item()


class ResultAggregator(Callback):
    """
    Collects each job's metrics.json and writes all_metrics.csv at sweep end.
    """

    def on_job_end(self, config, job_return, **kwargs) -> None:
        # # called after a child job finishes
        log.debug(f"""Finished sweep number {str(config.hydra.sweep.subdir)}""")

    def on_multirun_end(self, config, **kwargs) -> None:
        # called once after the *last* job


        sweep_dir = Path(config.hydra.sweep.dir)
        # Can also be set manually when debugging, e.g.: config.hydra.sweep.dir= 'multirun/2025-09-26/11-06-27'

        if config.meta.debug_mode:
            path_agg = Path(config.hydra.sweep.dir) / "aggregated/"
            path_agg.mkdir(parents=True, exist_ok=True)

            path_agg_csv= path_agg / "metrics_agg.csv"

            if (path_agg_csv).exists():
                results_df = pd.read_csv(path_agg_csv)

            else: 
                metrics_files = sweep_dir.rglob("metrics.json")
                rows = [json.loads(p.read_text()) for p in metrics_files if p.exists()]
                results_df = pd.DataFrame(rows)

                results_df.to_csv(path_agg_csv, index=False)

            # For each blackbox function, plot aggregated results
            unique_bbfs = list(results_df["blackbox"].unique())


            # Iterate over each unique value and restrict df
            for temp_bbf in unique_bbfs:
                bbf_path = path_agg / f"""{str(temp_bbf)}/"""
                bbf_path.mkdir(parents=True, exist_ok=True)

                bbf_df = results_df[results_df["blackbox"] == temp_bbf]

                # Filter the df to all seeds that have been evaluated on all acquisition functions
                n_acq_fns = bbf_df["acquisition"].nunique()
                seed_counts = bbf_df.groupby("seed")["acquisition"].nunique()
                valid_seeds = seed_counts[seed_counts == n_acq_fns].index
                bbf_df = bbf_df[bbf_df["seed"].isin(valid_seeds)]

                metrics = [
                    "mae",
                    "mse",
                    "mse_normalized",
                    "nlpd",
                    "nlpd_noisy",
                ]

                if (
                    config.application._target_
                    == "xac.applications.TabRepoBenchmarkApplication"
                ) and (
                    config.surrogate.fit_config._target_ != "xac.surrogates.NUTSConfig"
                ):
                    metrics.append("ce_loss")
                    metrics.append("ce_loss_noisy")
                    metrics.append("accuracy")
                    metrics.append("accuracy_noisy")


                if config.meta.time_ops:
                    metrics.append("hp_fit_duration")
                    metrics.append("acq_fun_duration")

    
                bbf_df_agg = bbf_df.groupby("acquisition").agg(list)

                if set(bbf_df_agg.index.to_list()) == set(SHAPLEIG_INDEX_SET_FULL):
                    bbf_df_agg = bbf_df_agg.reindex(SHAPLEIG_INDEX_SET_FULL)

                for temp_metric_name in metrics:
                    temp_cat_names, temp_cat_list = _get_metric_categories(
                        bbf_df_agg, temp_metric_name
                    )

                    if not temp_cat_names:
                        continue

                    for log_scaling in [True, False]:
                        if (not log_scaling) or (
                            log_scaling and (temp_metric_name in ["mae", "mse", "mse_normalized"])
                        ):
 
                            # Plot mean and std (with y maximum fixed)
                            if temp_metric_name in ["mae", "mse", "mse_normalized"]:
                                # Plot mean with y maximum fixed
                                for temp_quantile in MAE_MSE_QUANTILES:
                                    quantile_suffix = _get_quantile_suffix(
                                        temp_quantile
                                    )

                                    for legend in [True, False]:
                                        for (
                                            subset_name,
                                            subset_categories,
                                            temp_cat_list_subset,
                                        ) in _iter_plot_subsets(
                                            temp_cat_names, temp_cat_list
                                        ):
                                            y_minimum = _get_y_minimum(
                                                temp_cat_list_subset
                                            )

                                            plot_trajectory(
                                                main_data=[
                                                    temp_devs.mean(axis=0)
                                                    for temp_devs in temp_cat_list_subset
                                                ],
                                                granular_data=temp_cat_list_subset,
                                                plot_title=None,  # plot_title,  # Map DS name to more formal name via dictionary
                                                y_label=get_plot_name(temp_metric_name)
                                                + " (Mean + SEM)",  # Map to more formal names via dictionary (also mode if not in dictionary)
                                                categories=subset_categories,  # Map to more formal names via dictionary
                                                path=bbf_path
                                                / str(temp_metric_name)
                                                / f"""{temp_metric_name}_mean_w_std{("_log" if log_scaling else "")}{("_leg" if legend else "")}_ycap{quantile_suffix}_{subset_name}.png""",
                                                plot_std=True,
                                                plot_individual_runs=False,
                                                y_range_top=None,
                                                y_range_bottom=None,
                                                y_log_scale=log_scaling,
                                                y_maximum=torch.quantile(
                                                    torch.concat(
                                                        [
                                                            temp_devs
                                                            for temp_devs in temp_cat_list_subset
                                                        ]
                                                    ),
                                                    temp_quantile,
                                                ).item(),
                                                y_minimum=y_minimum,  # torch.concat([temp_devs.mean(axis=0) - 2*(temp_devs.numpy().std(axis=0, ddof=1) / np.sqrt(temp_devs.shape[0])) for temp_devs in temp_cat_list]).min().item(),
                                                legend=legend,
                                                legend_placement_bottom=False,
                                                size_X0=int(
                                                    results_df[
                                                        "initial_design_size"
                                                    ].iloc[0]
                                                ),
                                            )

                            if temp_metric_name in [
                                "acq_fun_duration",
                                "hp_fit_duration",
                            ]:
                                if "EIG-FP" not in temp_cat_names:
                                    continue

                                eig_fp_idx = temp_cat_names.index("EIG-FP")
                                plot_trajectory(
                                    main_data=[
                                        temp_cat_list[eig_fp_idx].mean(axis=0)
                                    ],
                                    granular_data=[temp_cat_list[eig_fp_idx]],
                                    plot_title=None,  # plot_title,  # Map DS name to more formal name via dictionary
                                    y_label=get_plot_name(
                                        temp_metric_name
                                    ),  # Map to more formal names via dictionary (also mode if not in dictionary)
                                    categories=["EIG-FP"],  # Map to more formal names via dictionary
                                    path=bbf_path
                                    / str(temp_metric_name)
                                    / f"""{temp_metric_name}_mean_w_std_cust.png""",
                                    plot_std=True,
                                    plot_individual_runs=False,
                                    y_range_top=None,
                                    y_range_bottom=None,
                                    y_log_scale=False,
                                    legend=False,
                                    legend_placement_bottom=False,
                                    size_X0=int(
                                        results_df["initial_design_size"].iloc[0]
                                    ),
                                )

                        else:
                            pass

        log.info(f"""Finished on_multirun_end.""")
