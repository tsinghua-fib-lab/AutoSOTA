
from __future__ import annotations
import argparse
import os
import numpy as np
import pandas as pd
import time
from pathlib import Path

# Import oddshap modules
from oddshap.oddshap import OddSHAP

# Import benchmark factory
from benchmark_exhaustive_approx import BenchmarkFactory
from oddshap.polyshap import ExplanationFrontierGenerator, PolySHAP

pd.set_option("future.no_silent_downcasting", True)

parser = argparse.ArgumentParser(
    description="Run OddSHAP ablation (varying interaction numbers) on exhaustive explanation games."
)
args = parser.parse_args()

if __name__ == "__main__":
    """
    This script runs OddSHAP with varying assumed interaction counts (k) on exhaustive games.
    Results are stored in /approximations/exhaustive/.
    """
    RANDOM_STATE = 40
    
    # Load standardized benchmarks
    BENCHMARKS = BenchmarkFactory.load_benchmarks_from_json(
        config_path="shapiq-benchmark/benchmarks/configuration_exhaustive_sv.json"
    )
    
    # Ablation values for k (interaction factor)
    FACTORS = [0, 1000, 100, 50, 10, 5, 2]

    def explain_instance(args):
        game_id, id_explain, game_instance, index, order = args
        
        # Fixed budgets as requested
        budget_range = [5000, 10000, 20000]
        
        print(f"Starting explanation for {game_id}, id {id_explain}, budgets {budget_range}")

        # Iterate over budgets
        for budget in budget_range:
            # Iterate over assumed interaction counts (k)
            for factor in FACTORS:
                try:
                    # Calculate interaction_factor
                    n_players = game_instance.n_players
                    sampling_weights_leverage = np.ones(n_players+1)
                    if factor == 0:
                        k=0
                    else:
                        k = max(0, budget // factor - n_players)

                    if k == 0:
                        # LeverageSHAP
                        frontier_generator = ExplanationFrontierGenerator(N=set(range(n_players)))
                        explanation_frontier = frontier_generator.generate_kadd(max_order=1)
                        approximator = PolySHAP(
                            n_players,
                            explanation_frontier=explanation_frontier,
                            random_state=RANDOM_STATE,
                            sampling_weights=sampling_weights_leverage,
                            pairing_trick=True,
                            replacement=False,
                        )
                    else:
                        approximator = OddSHAP(
                            n=n_players,
                            interaction_factor=factor,
                            random_state=RANDOM_STATE,
                            sampling_weights=sampling_weights_leverage, # leverage score weights
                            grid_search=False # Fixed factor
                        )
                    
                    approx_name = f"OddSHAP-k{k}"
                    
                    print(f"Computing {approx_name} on {game_id} (id {id_explain}) with budget {budget}")
                    
                    start_time = time.time()
                    shap_approx = approximator.approximate(
                        budget=int(budget),
                        game=game_instance,
                    )
                    total_time = time.time() - start_time
                    
                    run_time_meta = {
                        "total_runtime": total_time,
                        "k": k,
                        "interaction_factor": factor
                    }

                    # Construct save path
                    # Format: approximations/exhaustive/{game_id}_{CONFIG}_{id}_{approx}_{budget}_{index}_{order}.json
                    config_id = "ablation"
                    
                    save_path = Path(
                        f"approximations/exhaustive/{game_id}_{config_id}_{id_explain}_{approx_name}_{budget}_{index}_{order}.json"
                    )
                    
                    os.makedirs(save_path.parent, exist_ok=True)
                    
                    shap_approx.save(save_path, **run_time_meta)
                    
                except Exception as e:
                    print(f"Failed {approx_name} on {game_id} (id {id_explain}, b={budget}): {e}")

    # Main Loop
    for game_identifier, benchmark_info in BENCHMARKS.items():
        games = benchmark_info["games"]
        
        for id_explain, game_instance in enumerate(games):
            # Exhaustive games are ready to use
            explain_instance(
                (
                    game_identifier,
                    id_explain,
                    game_instance,
                    benchmark_info["index"],
                    benchmark_info["order"],
                )
            )
