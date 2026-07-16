from __future__ import annotations
import argparse
import os
import numpy as np
import pandas as pd
import time
from pathlib import Path

# Import oddshap modules
from oddshap.interventional import InterventionalGame
from oddshap.oddshap import OddSHAP
from oddshap.polyshap import ExplanationFrontierGenerator, PolySHAP

# Import benchmark factory
from benchmark_exhaustive_approx import BenchmarkFactory

pd.set_option("future.no_silent_downcasting", True)

parser = argparse.ArgumentParser(
    description="Run OddSHAP ablation (varying interaction numbers) on explanation games."
)
args = parser.parse_args()

if __name__ == "__main__":
    """
    This script runs OddSHAP with varying assumed interaction counts (k) on interventional games.
    Results are stored in /approximations/interventional/.
    """
    RANDOM_STATE = 40

    # Load standardized benchmarks
    BENCHMARKS = BenchmarkFactory.load_benchmarks_from_json(
        config_path="shapiq-benchmark/benchmarks/configuration_interventional_sv.json"
    )


    def explain_instance(args):
        game_id, id_explain, game_instance, index, order = args

        # Fixed budgets as requested
        budget_range = [5000, 10000, 20000]

        tree_max_depths = [1,2,5,10,20]
        n_trees = [10,50,100,200]

        print(f"Starting explanation for {game_id}, id {id_explain}, budgets {budget_range}")

        # Iterate over budgets
        for budget in budget_range:
            n_players = game_instance.n_players
            sampling_weights_leverage = np.ones(n_players + 1)

            for max_depth in tree_max_depths:
                tree_params = {"max_depth":max_depth}
                approximator = OddSHAP(
                    n=n_players,
                    tree_params=tree_params,
                    random_state=RANDOM_STATE,
                    sampling_weights=sampling_weights_leverage  # leverage score weights
                )

                # Approximator name includes k for unique identification
                # Standard OddSHAP name is just "OddSHAP". We append k.
                approx_name = f"OddSHAP-depth{max_depth}"
                print(f"Computing {approx_name} on {game_id} (id {id_explain}) with budget {budget}")
                try:
                    start_time = time.time()
                    shap_approx = approximator.approximate(
                        budget=int(budget),
                        game=game_instance,
                    )
                    total_time = time.time() - start_time

                    run_time_meta = {
                        "total_runtime": total_time,
                        "max_depth": max_depth
                    }

                    # Construct save path
                    # Format: approximations/interventional/{game_id}_{CONFIG}_{id}_{approx}_{budget}_{index}_{order}.json
                    # We use a dummy config ID for this ablation, e.g. "ablation"
                    config_id = "depth"

                    save_path = Path(
                        f"approximations/interventional/{game_id}_{config_id}_{id_explain}_{approx_name}_{budget}_{index}_{order}.json"
                    )

                    # Create directory if needed? (approximator.save usually handles file creation, assume dir exists)
                    os.makedirs(save_path.parent, exist_ok=True)

                    shap_approx.save(save_path, **run_time_meta)
                except Exception as e:
                    print(f"Failed {approx_name} on {game_id} (id {id_explain}, b={budget}): {e}")


            for n_estimators in n_trees:
                tree_params = {"n_estimators": n_estimators, "max_depth":10}
                n_players = game_instance.n_players
                sampling_weights_leverage = np.ones(n_players + 1)

                approximator = OddSHAP(
                    n=n_players,
                    tree_params=tree_params,
                    random_state=RANDOM_STATE,
                    sampling_weights=sampling_weights_leverage  # leverage score weights
                )

                # Approximator name includes k for unique identification
                # Standard OddSHAP name is just "OddSHAP". We append k.
                approx_name = f"OddSHAP-trees{n_estimators}"
                print(f"Computing {approx_name} on {game_id} (id {id_explain}) with budget {budget}")
                try:
                    start_time = time.time()
                    shap_approx = approximator.approximate(
                        budget=int(budget),
                        game=game_instance,
                    )
                    total_time = time.time() - start_time

                    run_time_meta = {
                        "total_runtime": total_time,
                        "n_estimators": n_estimators
                    }

                    # Construct save path
                    # Format: approximations/interventional/{game_id}_{CONFIG}_{id}_{approx}_{budget}_{index}_{order}.json
                    # We use a dummy config ID for this ablation, e.g. "ablation"
                    config_id = "trees"

                    save_path = Path(
                        f"approximations/interventional/{game_id}_{config_id}_{id_explain}_{approx_name}_{budget}_{index}_{order}.json"
                    )

                    # Create directory if needed? (approximator.save usually handles file creation, assume dir exists)
                    os.makedirs(save_path.parent, exist_ok=True)

                    shap_approx.save(save_path, **run_time_meta)
                except Exception as e:
                    print(f"Failed {approx_name} on {game_id} (id {id_explain}, b={budget}): {e}")

    # Main Loop
    for game_identifier, benchmark_info in BENCHMARKS.items():
        games = benchmark_info["games"]
        # Limit games if needed? Default runs all loaded instances (usually 30).

        for id_explain, game_instance in enumerate(games):
            if id_explain < 7:
                continue  # Skip first 7 instances for testing purposes
            x_explain = game_instance.x.astype(np.float32)

            reference_indices = np.random.RandomState(RANDOM_STATE + id_explain).choice(
                game_instance.setup.x_train.shape[0], size=50, replace=False
            )
            reference_data = game_instance.setup.x_train[reference_indices, :].astype(np.float32)

            # Create InterventionalGame
            tree_game = InterventionalGame(
                model=game_instance.setup.model,
                reference_data=reference_data,
                target_instance=x_explain,
            )

            explain_instance(
                (
                    game_identifier,
                    id_explain,
                    tree_game,
                    benchmark_info["index"],
                    benchmark_info["order"],
                )
            )
