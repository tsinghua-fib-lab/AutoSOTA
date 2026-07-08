import json
import os
from datetime import datetime

from src.common_utils import ExperimentTimer, convert_numpy_to_list
from src.experiments.ablation_study import EntropyAblationStudy


def ablation_study_entropy_regularization():
    env_params = {"gamma": 0.9, "eps": 0.5, "num_followers": 2, "beta": 5.0}

    common_params = {
        "max_iterations": 5,
        "lamda": 0.1,
        "reg": "ER",
        "gradient": True,
        "eta": 0.1,
        "sampling": False,
        "n_sample": 100,
        "policy_gradient": True,
        "nu": 0.1,
        "unregularized_obj": False,
        "lagrangian": False,
        "N": 10,
        "delta": 0.1,
        "B": 1.0,
        "feature_dim": 32,
        "nn_lr": 0.001,
        "warmup": 1,
        "num_trajectories": 30,
        "value_lr": 0.1,
    }

    entropy_lambdas = [0.01, 0.5, 1, 2]
    num_seeds = 20
    seeds = list(range(1, num_seeds + 1))
    n_jobs = min(4, len(seeds))

    study = EntropyAblationStudy(
        env_params=env_params,
        common_params=common_params,
        entropy_lambdas=entropy_lambdas,
        seeds=seeds,
        n_jobs=n_jobs,
    )

    return study.run()


if __name__ == "__main__":
    with ExperimentTimer("Entropy Ablation"):
        ablation_results = ablation_study_entropy_regularization()

    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_filename = f"results/regpepg_ablation_multiseed_{timestamp}.json"

    json_serializable_results = convert_numpy_to_list(ablation_results)

    with open(json_filename, "w") as f:
        json.dump(json_serializable_results, f, indent=4)

    print(f"Saved: {json_filename}")
