from __future__ import annotations
import argparse
import time
from typing import Any
import numpy as np
from oddshap.approx_utils import get_approximators
from pathlib import Path
from shapiq_benchmark.load import GameFactory
import pandas as pd
import os

pd.set_option("future.no_silent_downcasting", True)


class BenchmarkFactory:
    @staticmethod
    def create_benchmarks_interactive(
        game_config_names: list[str],
        *,
        n_games: list[int] | int = 30,
        approximation_methods: list[str] = [
            "SVARM",
            "KernelSHAP",
            "PermutationSamplingSV",
            "RegressionMSR",
        ],
        index: str = "SV",
        order: int = 1,
        config_path: str = "shapiq-benchmark/configurations_exhaustive/",
        config_save_name: str = "configuration",
    ) -> dict[str, Any]:
        """Create benchmarks from a list of game configuration names.
        Also saves a json file with the configuration details.
        The game_config_names should correspond to the json files in
        shapiq-benchmark/configurations_exhaustive/.

         Args:
            game_config_names (list[str]): List of game configuration names.
            n_games (list[int] | None): List of number of games to create for each configuration.
                If None, all games will be created. Default is None.
         Returns:
            dict: Dictionary of benchmarks with game configuration names as keys and
                game generators as values.
        """
        benchmarks = {}
        json_dict = {}
        for idx, game_config_name in enumerate(game_config_names):
            game_n_games: int | None = (
                n_games[idx] if isinstance(n_games, list) else n_games
            )
            game_generator, config_id = GameFactory.load_configuration_file_interactive(
                config_path=f"{config_path}{game_config_name}.json",
                n_games=game_n_games,
                check_pre_computed=True,
                only_pre_computed=True,
                return_config_id=True,
            )
            benchmarks[game_config_name + "_" + str(config_id)] = {
                "games": game_generator,
                "approximation_methods": approximation_methods,
                "index": index,
                "order": order,
            }
            json_dict[game_config_name] = {
                "game_config_name": f"{config_path}{game_config_name}.json",
                "n_games": game_n_games,
                "config_id": config_id,
                "approximation_methods": approximation_methods,
                "index": index,
                "order": order,
            }
        # Save json_dict to a json file
        import json

        save_path = Path("shapiq-benchmark/benchmarks/")
        save_path.mkdir(parents=True, exist_ok=True)
        with open(save_path / (config_save_name + ".json"), "w") as f:
            json.dump(json_dict, f, indent=4)
        return benchmarks

    @staticmethod
    def load_benchmarks_from_json(config_path: str):
        import json

        benchmarks = {}
        with open(config_path, "r") as f:
            json_dict = json.load(f)
        for benchmark_name, benchmark_info in json_dict.items():
            game_generator = GameFactory.load_configuration_file(
                config_path=benchmark_info["game_config_name"],
                config_id=benchmark_info["config_id"],
                n_games=benchmark_info["n_games"],
                check_pre_computed=True,
                only_pre_computed=True,
            )
            benchmarks[benchmark_name + "_" + str(benchmark_info["config_id"])] = {
                "games": game_generator,
                "approximation_methods": benchmark_info.get(
                    "approximation_methods", []
                ),
                "index": benchmark_info["index"],
                "order": benchmark_info["order"],
            }
        return benchmarks


if __name__ == "__main__":
    """
    This script runs selected approximation algorithms on explanation games that use baseline
    imputatation, which were pre-computed in the shapiq library. The ground truth values
    are computed using exhaustive evaluation. Approximations are stored in
    /approximations/exhaustive/ and ground truth values in /ground_truth/exhaustive/.
    """
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
        "--max_budget",
        type=int,
        default=20000,
        help="Maximum budget for approximations. Default is 20000.",
    )
    parser.add_argument(
        "--n_budget_steps",
        type=int,
        default=10,
        help="Number of budget steps for approximations. Default is 10.",
    )
    args = parser.parse_args()
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
        config_path="shapiq-benchmark/benchmarks/configuration_exhaustive_sv.json"
    )

    MAX_BUDGET = args.max_budget
    N_BUDGET_STEPS = args.n_budget_steps

    def explain_instance(args):
        game_id, id_explain, game_instance, approximators, index, order = args
        approximator_list = get_approximators(
            approximators,
            game_instance.n_players,
            RANDOM_STATE,
            PAIRING,
            REPLACEMENT,
        )

        min_budget = game_instance.n_players + 1
        max_budget = min(2**game_instance.n_players, MAX_BUDGET)
        budget_range = (
            np.ceil(
                np.logspace(np.log10(min_budget), np.log10(max_budget), N_BUDGET_STEPS)
            )
            .clip(min_budget, max_budget)
            .astype(int)
        )
        for approximator in approximator_list:
            print(
                "Computing approximations for",
                approximator.name,
                "on game",
                game_id,
                "explanation id",
                id_explain,
            )
            # FFD methods have a fixed budget — run once and save a single file.
            _ffd_names = {"FFD-RD", "FFD-RD-Corrected"}
            # FourierSHAP saves by actual budget to avoid collisions when two
            # requested budgets round to the same b.
            _actual_budget_names = _ffd_names | {"FourierSHAP"}
            _budget_iter = budget_range[:1] if approximator.name in _ffd_names else budget_range
            for budget in _budget_iter:
                # FourierSHAP needs at least T*(n+1)*2 evaluations (b=1).
                if approximator.name == "FourierSHAP":
                    _T = approximator.T
                    if budget < _T * (game_instance.n_players + 1) * 2:
                        continue
                a = time.time()
                shap_approx = approximator.approximate(
                    budget=int(budget),
                    game=game_instance,  # need to int(budget) to avoid error in json serialization
                )
                run_time = time.time() - a
                if approximator.name in ["RegressionMSR-LGBM",
                                         "Proxy-LGBM"]:
                    detailed_runtimes = approximator.runtime_last_approximate_run
                    run_time = {
                        "total_runtime": run_time,
                        "total_Runtime_dealing": detailed_runtimes["total"],
                        "adjustment": detailed_runtimes["adjustment"],
                        "proxy_fit": detailed_runtimes["proxy_fit"],
                        "extraction": detailed_runtimes["extraction"],
                        "evaluations": detailed_runtimes["evaluations"],
                        "sampling": detailed_runtimes["sampling"],
                    }
                elif approximator.name in ["OddSHAP-Fourier-Random",
                                           "OddSHAP-Fourier-ProxySPEX-NoCV",
                                           "OddSHAP-Fourier-ProxySPEX-NoCV-5",
                                           "OddSHAP-Fourier-ProxySPEX-NoCV-20"
                                           ]:
                    detailed_runtimes = approximator.runtime_last_approximate_run
                    run_time = {
                        "total_runtime": run_time,
                        "total_Runtime_dealing": detailed_runtimes["total"],
                        "proxy_fit": detailed_runtimes["proxy_fit"],
                        "extraction": detailed_runtimes["extraction"],
                        "regression": detailed_runtimes["regression"],
                        "evaluations": detailed_runtimes["evaluations"],
                        "sampling": detailed_runtimes["sampling"],
                    }
                else:
                    detailed_runtimes = approximator.runtime_last_approximate_run
                    run_time = {
                        "total_runtime": run_time,
                        "evaluations": detailed_runtimes["evaluations"],
                        "total_approximation": detailed_runtimes["total"],
                    }

                # For FFD/FourierSHAP use the actual evaluation count as the
                # budget label so the saved file sits at the correct x-axis position.
                save_budget = (
                    shap_approx.estimation_budget
                    if approximator.name in _actual_budget_names
                    else budget
                )
                save_path = Path(
                    "approximations/exhaustive/"
                    + game_id
                    + "_"
                    + str(ID_CONFIG_APPROXIMATORS)
                    + "_"
                    + str(id_explain)
                    + "_"
                    + approximator.name
                    + "_"
                    + str(save_budget)
                    + "_"
                    + str(index)
                    + "_"
                    + str(order)
                    + ".json"
                )
                shap_approx.save(save_path, **run_time)

    # Build flat list of all (game, instance) jobs
    all_jobs = []
    for game_identifier, benchmark_info in BENCHMARKS.items():
        for id_explain, game_instance in enumerate(benchmark_info["games"]):
            all_jobs.append((
                game_identifier,
                id_explain,
                game_instance,
                benchmark_info["approximation_methods"],
                benchmark_info["index"],
                benchmark_info["order"],
            ))

    # If running as a SLURM array, pick only this task's (game, instance) job
    if "SLURM_ARRAY_TASK_ID" in os.environ:
        task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
        all_jobs = [all_jobs[task_id]]
        print(f"SLURM task {task_id}: running job {all_jobs[0][0]} instance {all_jobs[0][1]}")

    for job in all_jobs:
        explain_instance(job)
