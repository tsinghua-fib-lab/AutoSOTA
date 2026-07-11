"""Run NMF baseline algorithms on exact synthetic datasets."""

import logging
import time
from pathlib import Path

from nmf_algos.registry import get_algorithm_class
from nmf_algos.utils.utils import load_data_basedon_proto, load_data_matrix

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def load_exact_dataset_config(config_path):
    """Load exact-dataset experiment configuration."""
    return load_data_basedon_proto(config_path, mode="exactDatasets")


def load_exact_matrix(project_dir, dataset_config):
    """Load one exact synthetic dataset matrix."""
    data_path = project_dir / dataset_config.data_dir / dataset_config.data_path
    data_dict = load_data_matrix(data_path)
    return data_dict["X"]


def run_algorithm(method_name, X, dataset_name, rank, target_run_time):
    """Run one NMF algorithm on one dataset."""
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
    config_path = project_dir / "Experiments" / "configs" / "exact_data_algo_exp.json"

    rank = 50
    target_run_time = 600
    method_names = ["HALS", "MUL", "AOADMM", "GRADMUL", "ALS"]

    config = load_exact_dataset_config(config_path)

    for dataset_config in config.exact_dataset:
        X = load_exact_matrix(project_dir, dataset_config)

        logger.info(
            "Loaded dataset %s from %s with shape %s.",
            dataset_config.name,
            dataset_config.data_path,
            X.shape,
        )

        for method_name in method_names:
            run_algorithm(
                method_name=method_name,
                X=X,
                dataset_name=dataset_config.name,
                rank=rank,
                target_run_time=target_run_time,
            )


if __name__ == "__main__":
    main()
