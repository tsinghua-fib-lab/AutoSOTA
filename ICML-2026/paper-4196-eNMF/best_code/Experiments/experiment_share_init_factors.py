"""Run NMF algorithms on the Verb dataset with different initialization methods."""

import logging
import pickle
import time
from pathlib import Path

from initialization_algos.init_algo import get_init_factors
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


def save_initial_factors(save_dir, U, V, rank, init_method):
    """Save initialized factors for reproducibility."""
    save_dir.mkdir(parents=True, exist_ok=True)

    data_dict = {
        "U": U,
        "V": V,
        "init_method": init_method,
        "r": rank,
    }

    save_path = save_dir / f"r_{rank}_{init_method}.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info("Saved initialization factors to %s.", save_path)


def run_algorithm(
    method_name, X, U, V, dataset_name, rank, init_method, result_dir, target_run_time
):
    """Run one NMF algorithm with one initialization."""
    result_dir.mkdir(parents=True, exist_ok=True)

    params = {
        "X": X.copy(),  # protect shared data from in-place modification
        "U": U.copy(),
        "V": V.T.copy(),
        "dataset_name": dataset_name,
        "r": rank,
        "save_dir": str(result_dir),
    }

    algorithm_cls = get_algorithm_class(method_name)
    algorithm = algorithm_cls(method_name=method_name, params=params)

    start_time = time.time()
    algorithm.run_within_fixed_time(target_run_time=target_run_time)
    elapsed = time.time() - start_time

    logger.info(
        "Finished %s with rank=%d and init=%s in %.2f seconds.",
        method_name,
        rank,
        init_method,
        elapsed,
    )

    return algorithm


def main():
    project_dir = Path.cwd()
    data_path = project_dir / "Dataset" / "verb" / "right_matrix.npy"

    dataset_name = "Verb"
    method_names = ["HALS", "MUL", "AOADMM", "GRADMUL", "ALS"]
    ranks = [20]
    init_methods = ["pso", "de", "fss", "random"]
    target_run_time = 20

    init_factor_save_dir = project_dir / "Results" / dataset_name / "Init"
    X = load_verb_dataset(data_path)

    for init_method in init_methods:
        for rank in ranks:
            U, V = get_init_factors(X, rank, init_method=init_method)

            logger.info(
                "Finished %s initialization with U shape %s and V shape %s.",
                init_method,
                U.shape,
                V.shape,
            )

            save_initial_factors(
                save_dir=init_factor_save_dir,
                U=U,
                V=V,
                rank=rank,
                init_method=init_method,
            )

            for method_name in method_names:
                result_dir = (
                    project_dir
                    / "Results"
                    / dataset_name
                    / method_name
                    / f"latent_dim_{rank}"
                    / f"init_{init_method}"
                )

                run_algorithm(
                    method_name=method_name,
                    X=X,
                    U=U,
                    V=V,
                    dataset_name=dataset_name,
                    rank=rank,
                    init_method=init_method,
                    result_dir=result_dir,
                    target_run_time=target_run_time,
                )


if __name__ == "__main__":
    main()
