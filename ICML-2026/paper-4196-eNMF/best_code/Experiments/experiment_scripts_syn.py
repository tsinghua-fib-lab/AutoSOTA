"""Run NMF algorithms on synthetic datasets from experiment config."""

import logging
import time
from pathlib import Path

from nmf_algos.registry import get_algorithm_class
from nmf_algos.utils.utils import (
    fetch_factors_from_result_path,
    load_data_basedon_proto,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def load_synthetic_dataset_configs(config_path):
    """Load synthetic-dataset experiment configuration."""
    return load_data_basedon_proto(config_path, mode="synDatasets").real_dataset


def load_synthetic_matrix(data_path):
    """Load the synthetic matrix from a saved result path."""
    return fetch_factors_from_result_path(data_path)


def run_algorithm(method_name, X, dataset_name, rank, target_run_time):
    """Run one NMF algorithm on one dataset with one latent dimension."""
    params = {
        "X": X.copy(),  # protect shared data from in-place modification
        "dataset_name": dataset_name,
        "r": rank,
    }

    algorithm_cls = get_algorithm_class(method_name)
    algorithm = algorithm_cls(method_name=method_name, params=params)

    start_time = time.time()
    algorithm.run_within_fixed_time(target_run_time=target_run_time)
    elapsed = time.time() - start_time

    logger.info(
        "Finished %s on %s with rank=%d in %.2f seconds.",
        method_name,
        dataset_name,
        rank,
        elapsed,
    )

    return algorithm


def main():
    project_dir = Path.cwd()
    config_path = project_dir / "Experiments" / "configs" / "syn_data_algo_exp.json"

    target_run_time = 1000
    dataset_configs = load_synthetic_dataset_configs(config_path)

    for dataset_config in dataset_configs:
        data_path = project_dir / dataset_config.data_dir / dataset_config.data_path
        X = load_synthetic_matrix(data_path)

        logger.info(
            "Loaded dataset %s from %s with shape %s.",
            dataset_config.name,
            dataset_config.data_path,
            X.shape,
        )

        for method_config in dataset_config.method_config:
            method_name = method_config.method_name

            for rank in dataset_config.latent_dim:
                run_algorithm(
                    method_name=method_name,
                    X=X,
                    dataset_name=dataset_config.name,
                    rank=rank,
                    target_run_time=target_run_time,
                )


if __name__ == "__main__":
    main()
