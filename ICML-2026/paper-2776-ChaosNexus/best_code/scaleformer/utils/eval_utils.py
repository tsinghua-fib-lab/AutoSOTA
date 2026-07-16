"""
utils for evaluation scripts
"""

import logging
import os
from collections import defaultdict
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from gluonts.dataset.common import FileDataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


def get_eval_data_dict(
    data_paths_lst: str,
    num_subdirs: int | None = None,
    num_samples_per_subdir: int | None = None,
    one_dim_target: bool = False,
) -> dict[str, list[FileDataset]]:  # type: ignore
    """
    data_paths_lst: list of test data directories paths
    num_subdirs: number of subdirectories to consider
    num_samples_per_subdir: number of samples per subdirectory to consider
    """
    # get test data paths
    test_data_dirs_dict = defaultdict(list)  # maps subdirectory name to test data path
    for test_data_dir in data_paths_lst:
        test_data_dir = os.path.expandvars(test_data_dir)
        system_dirs = [d for d in Path(test_data_dir).iterdir() if d.is_dir()]
        for subdir in system_dirs:
            # can be either name of skew pair or name of the unique system (parameter perturbation)
            subdir_name = subdir.name
            test_data_dirs_dict[subdir_name].append(subdir)

    test_data_dict = {}
    for subdir_name, subdirs in islice(test_data_dirs_dict.items(), num_subdirs):
        system_files = []
        for subdir in subdirs:
            system_files.extend(list(subdir.glob("*")))
        # sort system_files by sample_idx where the files in system_files are named like {sample_idx}_T-4096.arrow
        # also, take only the first cfg.eval.num_samples_per_subdir files in each subdirectory
        # ---> This means only the first 10 parameter perturbations per skew pair, if the subdirectory names are the skew pairs
        system_files = sorted(
            system_files,
            key=lambda x: int(x.stem.split("_")[0]),
        )[:num_samples_per_subdir]

        test_data_dict[subdir_name] = [
            FileDataset(path=Path(file_path), freq="h", one_dim_target=one_dim_target)
            for file_path in system_files
            if file_path.is_file()
        ]
    return test_data_dict


def left_pad_and_stack_1D(tensors: list[torch.Tensor]) -> torch.Tensor:
    """
    Left pad a list of 1D tensors to the same length and stack them.
    Used in pipeline, if given context is a list of tensors.
    """
    max_len = max(len(c) for c in tensors)
    padded = []
    for c in tensors:
        assert isinstance(c, torch.Tensor)
        assert c.ndim == 1
        padding = torch.full(
            size=(max_len - len(c),), fill_value=torch.nan, device=c.device
        )
        padded.append(torch.concat((padding, c), dim=-1))
    return torch.stack(padded)


def left_pad_and_stack_multivariate(tensors: list[torch.Tensor]) -> torch.Tensor:
    """
    Left pad a list of multivariate time series tensors to the same length and stack them.
    Used in pipeline, if given context is a list of tensors.
    """
    max_len = max(c.shape[0] for c in tensors)
    padded = []
    for c in tensors:
        assert isinstance(c, torch.Tensor)
        assert c.ndim == 2
        padding = torch.full(
            size=(max_len - len(c),), fill_value=torch.nan, device=c.device
        )
        padded.append(torch.concat((padding, c), dim=-1))
    return torch.stack(padded)


def save_evaluation_results(
    metrics: dict[int, dict[str, dict[str, float]]] | None = None,
    metrics_metadata: dict[str, dict[str, Any]] | None = None,
    metrics_save_dir: str = "results",
    metrics_fname: str | None = None,
    overwrite: bool = False,
) -> None:
    """
    Save prediction metrics and optionally forecast trajectories.

    Args:
        metrics: Nested dictionary containing computed metrics for each system.
        metrics_metadata: Dictionary containing metadata for the metrics.
            Keys are the quantity names, values are dictionaries containing metadata for each system.
                e.g. {"system_dims": {"system_name": 3}}
        metrics_save_dir: Directory to save metrics to.
        metrics_fname: Name of the metrics file to save.
        overwrite: Whether to overwrite an existing metrics file
    """
    if metrics is not None:
        for forecast_length, metric_dict in metrics.items():
            result_rows = [
                {"system": system, **metric_dict[system]} for system in metric_dict
            ]
            results_df = pd.DataFrame(result_rows)
            if metrics_metadata is not None:
                for quantity_name in metrics_metadata:
                    results_df[quantity_name] = results_df["system"].map(
                        metrics_metadata[quantity_name]
                    )
            curr_metrics_fname = (
                f"{metrics_fname or 'metrics'}_pred{forecast_length}.csv"
            )
            metrics_save_path = os.path.join(metrics_save_dir, curr_metrics_fname)
            logger.info(f"Saving metrics to: {metrics_save_path}")
            os.makedirs(os.path.dirname(metrics_save_path), exist_ok=True)

            if os.path.isfile(metrics_save_path) and not overwrite:
                existing_df = pd.read_csv(metrics_save_path)
                results_df = pd.concat([existing_df, results_df], ignore_index=True)
            results_df.to_csv(metrics_save_path, index=False)


# For processing the saved metrics csv files
def get_summary_metrics_dict(
    unrolled_metrics: dict, metric_name: str
) -> tuple[dict[str, dict[str, list[float]]], dict[str, bool]]:
    """
    Get the summary metrics for a given metric name.

    Returns a tuple of the summary metrics dictionary and a boolean indicating whether there are NaNs in the metrics.
    """
    summary_metrics_dict = defaultdict(dict)
    has_nans = defaultdict(bool)
    for model_name, metrics_dict in unrolled_metrics.items():
        prediction_lengths = list(metrics_dict.keys())
        summary_metrics_dict[model_name]["prediction_lengths"] = prediction_lengths
        num_vals = len(metrics_dict[prediction_lengths[0]][metric_name])
        summary_metrics_dict[model_name]["num_vals"] = num_vals
        means = []
        medians = []
        stds = []
        stes = []
        all_vals = []
        for prediction_length in tqdm(
            prediction_lengths, desc=f"Computing {metric_name} for {model_name}"
        ):
            metric_vals = np.array(metrics_dict[prediction_length][metric_name])
            if np.isnan(metric_vals).any():
                has_nans[model_name] = True
            mean = np.nanmean(metric_vals)
            median = np.nanmedian(metric_vals)
            std = np.nanstd(metric_vals)
            ste = std / np.sqrt(len(metric_vals))

            means.append(mean)
            medians.append(median)
            stds.append(std)
            stes.append(ste)
            all_vals.append(metric_vals)

        if has_nans:
            print(f"NaNs in {model_name} for {metric_name}")

        summary_metrics_dict[model_name]["means"] = means
        summary_metrics_dict[model_name]["medians"] = medians
        summary_metrics_dict[model_name]["stds"] = stds
        summary_metrics_dict[model_name]["stes"] = stes
        summary_metrics_dict[model_name]["all_vals"] = all_vals

    return summary_metrics_dict, has_nans
