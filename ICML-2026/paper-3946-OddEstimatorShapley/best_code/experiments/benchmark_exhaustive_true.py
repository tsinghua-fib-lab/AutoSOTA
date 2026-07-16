from __future__ import annotations
import argparse
from pathlib import Path
from benchmark_exhaustive_approx import BenchmarkFactory
import pandas as pd

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
    N_GAMES = args.n_games
    ORDER = args.order
    BENCHMARKS = BenchmarkFactory.load_benchmarks_from_json(
        config_path="shapiq-benchmark/benchmarks/configuration_exhaustive_sv.json"
    )
    for game_identifier, benchmark_info in BENCHMARKS.items():
        games = benchmark_info["games"]
        for id_explain, game_instance in enumerate(games):
            save_path = Path(
                "ground_truth/exhaustive/"
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
            ground_truth = game_instance.exact_values(
                index=benchmark_info["index"], order=benchmark_info["order"]
            )
            ground_truth.save(save_path)
            print(f"Exact: {ground_truth} saved to {save_path}")
