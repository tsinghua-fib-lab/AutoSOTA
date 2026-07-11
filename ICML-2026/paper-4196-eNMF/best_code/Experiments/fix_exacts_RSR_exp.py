"""Run eNMF on exact synthetic datasets for the RSR experiment."""

import logging
from pathlib import Path

from nmf_algos import NMF_ENMF
from nmf_algos.utils.utils import (
    fetch_factors_from_result_path,
    load_data_basedon_proto,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def load_experiment_config(config_path):
    """Load exact-dataset experiment configuration."""
    return load_data_basedon_proto(config_path, mode="exactDatasets").exact_dataset


def load_exact_matrix(project_dir, dataset_config):
    """Load the exact synthetic data matrix."""
    data_path = project_dir / dataset_config.data_dir / dataset_config.data_path
    return fetch_factors_from_result_path(
        data_path,
        f_type="exacts",
        key_list=["X"],
    )


def build_enmf_params(X, dataset_name, rank, rerun_times):
    """Build eNMF parameters."""
    return {
        "X": X.copy(),  # protect shared data from in-place modification
        "dataset_name": dataset_name,
        "r": rank,
        "rho": 5,
        "epsilon": 1e-4,
        "max_iter": 10000,
        "tau_inc": 1.1,
        "tau_dec": 1.1,
        "num_steps": 10,
        "hals_rounds": 1,
        "rerun_times": rerun_times,
    }


def run_enmf_on_dataset(project_dir, dataset_config, rerun_times):
    """Run eNMF on one exact synthetic dataset."""
    X = load_exact_matrix(project_dir, dataset_config)
    rank = int(dataset_config.method_config[0].latent_dim)

    logger.info(
        "Loaded dataset %s from %s with shape %s and rank=%d.",
        dataset_config.name,
        dataset_config.data_path,
        X.shape,
        rank,
    )

    params = build_enmf_params(
        X=X,
        dataset_name=dataset_config.name,
        rank=rank,
        rerun_times=rerun_times,
    )

    nmf_enmf = NMF_ENMF(params=params)
    nmf_enmf.basic_run()

    logger.info("Finished eNMF on dataset %s.", dataset_config.name)

    return nmf_enmf


def main():
    project_dir = Path.cwd()
    config_path = project_dir / "Experiments" / "configs" / "exact_data_algo_RSR.json"

    rerun_times = 1
    dataset_configs = load_experiment_config(config_path)

    for dataset_config in dataset_configs[:1]:
        run_enmf_on_dataset(
            project_dir=project_dir,
            dataset_config=dataset_config,
            rerun_times=rerun_times,
        )


if __name__ == "__main__":
    main()
