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
    This script runs OddSHAP with and without paired sampling and with and without even interactions on the interventional games.
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

        print(f"Starting explanation for {game_id}, id {id_explain}, budgets {budget_range}")

        PAIRING_TRICK_SETTINGS = [True,False]
        ODD_ONLY_SETTINGS = [True,False]

        n_players = game_instance.n_players
        sampling_weights_leverage = np.ones(n_players + 1)

        # Iterate over budgets
        for budget in budget_range:
            # Initialize different OddSHAP variants
            for pairing in PAIRING_TRICK_SETTINGS:
                for odd_only in ODD_ONLY_SETTINGS:

                    approximator = OddSHAP(
                        n=n_players,
                        random_state=RANDOM_STATE,
                        sampling_weights=sampling_weights_leverage,  # leverage score weights
                        pairing_trick=pairing,
                        odd_only=odd_only,
                    )

                    # Approximator name includes k for unique identification
                    # Standard OddSHAP name is just "OddSHAP". We append k.
                    approx_name = f"OddSHAP-pairing{int(pairing)}oddonly{int(odd_only)}"

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
                            "pairing": pairing,
                            "oddonly": odd_only
                        }

                        # Construct save path
                        # Format: approximations/interventional/{game_id}_{CONFIG}_{id}_{approx}_{budget}_{index}_{order}.json
                        # We use a dummy config ID for this ablation, e.g. "ablation"
                        config_id = "sampling"

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
