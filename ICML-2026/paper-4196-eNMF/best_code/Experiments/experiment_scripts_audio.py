"""Run NMF baseline algorithms on the AudioMNIST dataset."""

import logging
import time
from pathlib import Path

from nmf_algos.registry import NMF_METHOD_REGISTRY
from nmf_algos.utils.utils import audio_preprocess, load_audio_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def load_dataset(data_path):
    """Load and preprocess the AudioMNIST data matrix."""
    data_mat, _ = load_audio_data(data_path)
    data_mat = audio_preprocess(data_mat)

    logger.info("Loaded AudioMNIST data with shape %s.", data_mat.shape)
    return data_mat


def run_algorithm(method_name, X, dataset_name, rank, rerun_times, target_run_time):
    """Run one NMF algorithm with one latent dimension."""
    if method_name not in NMF_METHOD_REGISTRY:
        raise ValueError(
            f"Unsupported NMF method: {method_name}. "
            f"Available methods: {list(NMF_METHOD_REGISTRY)}"
        )

    params = {
        "X": X.copy(),  # protect shared data from in-place modification
        "dataset_name": dataset_name,
        "r": rank,
        "rerun_times": rerun_times,
    }

    algorithm_cls = NMF_METHOD_REGISTRY[method_name]
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
    data_path = project_dir / "Dataset" / "audiomnist.npy.npz"

    dataset_name = "Audiomnist"
    method_names = ["HALS", "MUL", "AOADMM", "GRADMUL", "ALS", "ADMM"]
    ranks = [20, 40, 80, 100]
    rerun_times = 5
    target_run_time = 600

    X = load_dataset(data_path)

    for rank in ranks:
        for method_name in method_names:
            run_algorithm(
                method_name=method_name,
                X=X,
                dataset_name=dataset_name,
                rank=rank,
                rerun_times=rerun_times,
                target_run_time=target_run_time,
            )


if __name__ == "__main__":
    main()
