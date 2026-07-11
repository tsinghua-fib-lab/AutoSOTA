"""Run NMF baseline algorithms on the Verb dataset."""

import logging
import time
from pathlib import Path

from nmf_algos.registry import get_algorithm_class
from nmf_algos.utils.utils import load_data_matrix

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def load_verb_dataset(data_path):
    """Load the Verb data matrix."""
    X = load_data_matrix(data_path)
    logger.info("Loaded Verb dataset with shape %s.", X.shape)
    return X


def run_algorithm(method_name, X, dataset_name, rank, target_run_time):
    """Run one NMF algorithm for one latent dimension."""
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
        "Finished %s with rank=%d in %.2f seconds.",
        method_name,
        rank,
        elapsed,
    )

    return algorithm


def main():
    project_dir = Path.cwd()
    data_path = project_dir / "Dataset" / "verb" / "right_matrix.npy"

    dataset_name = "Verb"
    method_names = ["HALS", "MUL", "AOADMM", "GRADMUL", "ALS"]
    rank = 20
    target_run_time = 20

    X = load_verb_dataset(data_path)

    for method_name in method_names:
        run_algorithm(
            method_name=method_name,
            X=X,
            dataset_name=dataset_name,
            rank=rank,
            target_run_time=target_run_time,
        )


if __name__ == "__main__":
    main()
