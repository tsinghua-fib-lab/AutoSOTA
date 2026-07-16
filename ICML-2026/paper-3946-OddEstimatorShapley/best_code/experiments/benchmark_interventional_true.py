from __future__ import annotations
import argparse
import numpy as np
from pathlib import Path
from benchmark_exhaustive_approx import BenchmarkFactory
import pandas as pd
from shapiq_benchmark.tree import InterventionalTreeBenchmark
import numpy as np
pd.set_option("future.no_silent_downcasting", True)

parser = argparse.ArgumentParser(
    description="Run benchmark approximations on explanation games."
)
parser.add_argument(
    "--config_approximators",
    type=int,
    default=37,
    help="Configuration ID for approximators: 40 (PAIRING=False, REPLACEMENT=True), "
    "39 (PAIRING=False, REPLACEMENT=False), "
    "38 (PAIRING=True, REPLACEMENT=True), "
    "37 (PAIRING=True, REPLACEMENT=False). Default is 37.",
)
parser.add_argument(
    "--n_games",
    type=int,
    default=30,
    help="Number of games to run for each benchmark. Default is 30.",
)
parser.add_argument(
    "--order",
    type=int,
    default=1,
    help="Order of the interaction index. Default is 2.",
)
args = parser.parse_args()

if __name__ == "__main__":
    """
    This script runs selected approximation algorithms on explanation games that use baseline
    imputatation, which were pre-computed in the shapiq library. The ground truth values
    are computed using exhaustive evaluation. Approximations are stored in
    /approximations/exhaustive/ and ground truth values in /ground_truth/exhaustive/.
    """
    RANDOM_STATE = 40  # random state for the games
    # ID_CONFIG_APPROXIMATORS = 40  # PAIRING=False, REPLACEMENT=True
    # ID_CONFIG_APPROXIMATORS = 39  # PAIRING=False, REPLACEMENT=False
    # ID_CONFIG_APPROXIMATORS = 38  # PAIRING=True, REPLACEMENT=True
    ID_CONFIG_APPROXIMATORS = (
        args.config_approximators
    )  # PAIRING=True, REPLACEMENT=False

    if ID_CONFIG_APPROXIMATORS == 40:
        REPLACEMENT = True
        PAIRING = False
    if ID_CONFIG_APPROXIMATORS == 39:
        REPLACEMENT = False
        PAIRING = False
    if ID_CONFIG_APPROXIMATORS == 38:
        REPLACEMENT = True
        PAIRING = True
    if ID_CONFIG_APPROXIMATORS == 37:
        REPLACEMENT = False
        PAIRING = True

    BENCHMARKS = BenchmarkFactory.load_benchmarks_from_json(
        config_path="shapiq-benchmark/benchmarks/configuration_interventional_sv.json"
    )
    for game_identifier, benchmark_info in BENCHMARKS.items():
        games = benchmark_info["games"]
        for id_explain, game_instance in enumerate(games):
            x_explain = game_instance.x.astype(np.float32)
            # Sample 5 reference instances from the training data
            random_state = np.random.RandomState(RANDOM_STATE+id_explain)
            reference_indices = random_state.choice(
                game_instance.setup.x_train.shape[0], size=50, replace=False
            )
            reference_data = game_instance.setup.x_train[reference_indices, :].astype(np.float32)
            # print(f"Reference indices: {reference_indices}")
            # print("Reference data: ", reference_data)
            tree_benchmark = InterventionalTreeBenchmark(
                tree_model=game_instance.setup.model,
                x_explain=x_explain,
                reference_data=reference_data,
            )
            # print("Original Tree Predict:", game_instance.setup.model.predict_proba(x_explain.reshape(1, -1)))
            # print("ORIGINAL GAME GRAND COALS:", game_instance.grand_coalition_value)
            # print("TREE GAME GRAND COALS:", tree_benchmark.game.grand_coalition_value)
            save_path = Path(
                "ground_truth/interventional/"
                + game_identifier
                + "_"
                + str(RANDOM_STATE)
                + "_"
                + str(id_explain)
                + "_"
                + benchmark_info["index"]
                + "_"
                + str(benchmark_info["order"])
                + "_exact_values.json"
            )
            ground_truth = tree_benchmark.exact_values(
                index=benchmark_info["index"], order=benchmark_info["order"]
            )
            ground_truth.save(save_path)
            print(f"Exact: {ground_truth} saved to {save_path}")
