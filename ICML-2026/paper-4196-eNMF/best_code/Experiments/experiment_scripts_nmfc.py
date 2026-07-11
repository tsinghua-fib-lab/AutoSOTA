"""Run NMFC algorithms on matrix-completion datasets."""

import logging
import time
from pathlib import Path

from nmf_algos.registry import NMFC_METHOD_REGISTRY
from nmf_algos.utils.utils import (
    fetch_factors_from_result_path,
    load_data_basedon_proto,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def load_dataset_configs(config_path):
    """Load dataset configurations for NMFC experiments."""
    return load_data_basedon_proto(config_path, mode="synDatasets").real_dataset


def load_matrix(project_dir, dataset_config):
    """Load one data matrix from the configured path."""
    data_path = project_dir / dataset_config.data_dir / dataset_config.data_path
    X = fetch_factors_from_result_path(data_path)

    logger.info(
        "Loaded dataset %s from %s with shape %s.",
        dataset_config.name,
        data_path,
        X.shape,
    )

    return X


def run_algorithm(method_name, X, dataset_name, rank):
    """Run one NMFC algorithm."""
    if method_name not in NMFC_METHOD_REGISTRY:
        raise ValueError(
            f"Unsupported NMFC method: {method_name}. "
            f"Available methods: {list(NMFC_METHOD_REGISTRY)}"
        )

    params = {
        "X": X.copy(),
        "dataset_name": dataset_name,
        "r": rank,
        "known_mask": (X > 0).astype(int),
    }

    algorithm_cls = NMFC_METHOD_REGISTRY[method_name]
    algorithm = algorithm_cls(method_name=method_name, params=params)

    start_time = time.time()
    algorithm.basic_run()
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
    config_path = project_dir / "Experiments" / "configs" / "real_data_algo_NMFC.json"

    dataset_configs = load_dataset_configs(config_path)

    for dataset_config in dataset_configs:
        X = load_matrix(project_dir, dataset_config)

        for method_config in dataset_config.method_config:
            run_algorithm(
                method_name=method_config.method_name,
                X=X,
                dataset_name=dataset_config.name,
                rank=int(method_config.latent_dim),
            )


if __name__ == "__main__":
    main()
