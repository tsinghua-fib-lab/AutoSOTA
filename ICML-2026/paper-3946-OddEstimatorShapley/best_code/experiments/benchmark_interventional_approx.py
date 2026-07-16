from __future__ import annotations
import argparse
import os
import numpy as np
from oddshap.approx_utils import get_approximators
from pathlib import Path
from benchmark_exhaustive_approx import BenchmarkFactory
import pandas as pd
from oddshap.interventional import InterventionalGame
import time

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
        config_path="shapiq-benchmark/benchmarks/configuration_interventional_rebuttal_oddshapvariants.json"
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
        # Subsample the approx list if we have a SLURM_ARRAY_TASK_ID
        if "SLURM_ARRAY_TASK_ID" in os.environ:
            task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
            approximator_list = [approximator_list[task_id]]
            print(
                "Subsampling approximator list to only run:", approximator_list[0].name
            )
        min_budget = game_instance.n_players + 1
        max_budget = min(2**game_instance.n_players, MAX_BUDGET)
        # This is all needed to the numerical instability which arises when using logspace and making sure that the budget is in [min_budget, max_budget]
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
            _budget_iter = budget_range[:1] if approximator.name in _ffd_names else budget_range
            for budget in _budget_iter:
                try:
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
                            "proxy_fit": detailed_runtimes["proxy_fit"],
                            "extraction": detailed_runtimes["extraction"],
                            "adjustment": detailed_runtimes["adjustment"],
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

                    # For FFD methods use the actual evaluation count as the
                    # budget label so the saved file sits at the correct x-axis position.
                    save_budget = (
                        shap_approx.estimation_budget
                        if approximator.name in _ffd_names
                        else budget
                    )
                    save_path = Path(
                        "approximations/interventional/"
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
                except Exception as e:
                    print(
                        f"Approximation failed for {approximator.name} on game {game_id} explanation id {id_explain} with budget {budget}. Error: {e}"
                    )

    for game_identifier, benchmark_info in BENCHMARKS.items():
        games = benchmark_info["games"]
        for id_explain, game_instance in enumerate(games):
            x_explain = game_instance.x.astype(np.float32)

            random_state = np.random.RandomState(RANDOM_STATE+id_explain)
            reference_indices = random_state.choice(
                game_instance.setup.x_train.shape[0], size=50, replace=False
            )

            reference_data = game_instance.setup.x_train[reference_indices, :].astype(
                np.float32
            )
            # print(f"Reference indices: {reference_indices}")
            # print("Reference data: ", reference_data)
            tree_game = InterventionalGame(
                model=game_instance.setup.model,
                reference_data=reference_data,
                target_instance=x_explain,
            )
            # print("ORIGINAL GAME GRAND COALS:", game_instance.grand_coalition_value)
            # print("TREE GAME GRAND COALS:", tree_game.grand_coalition_value)

            explain_instance(
                (
                    game_identifier,
                    id_explain,
                    tree_game,
                    benchmark_info["approximation_methods"],
                    benchmark_info["index"],
                    benchmark_info["order"],
                )
            )
