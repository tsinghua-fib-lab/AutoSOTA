"""Run eNMF on the AudioMNIST dataset."""

import logging
import time
from pathlib import Path

import numpy as np

from nmf_algos.registry import get_algorithm_class
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

    logger.info(
        "Loaded data: shape=%s, min=%.6e, max=%.6e",
        data_mat.shape,
        np.min(data_mat),
        np.max(data_mat),
    )

    return data_mat


def run_algorithm(method_name, X, dataset_name, rank, target_run_time, rerun_times):
    """Run one NMF algorithm with one latent dimension."""
    params = {
        "X": X.copy(),  # protect the shared data from in-place modification
        "dataset_name": dataset_name,
        "r": rank,
        "rerun_times": rerun_times,
        # eNMF / ADMM-related parameters
        "rho": 5,
        "epsilon": 1e-4,
        "max_iter": 2000,
        "tau_inc": 1.1,
        "tau_dec": 1.1,
        "num_steps": 10,
        "hals_rounds": 1,
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


def main():
    project_dir = Path.cwd()
    data_path = project_dir / "Dataset" / "audiomnist.npy.npz"

    dataset_name = "Audiomnist"
    method_names = ["ENMF"]
    ranks = [10, 20, 40, 80, 100]
    target_run_time = 60
    rerun_times = 3

    X = load_dataset(data_path)

    for method_name in method_names:
        for rank in ranks:
            run_algorithm(
                method_name=method_name,
                X=X,
                dataset_name=dataset_name,
                rank=rank,
                target_run_time=target_run_time,
                rerun_times=rerun_times,
            )


if __name__ == "__main__":
    main()
