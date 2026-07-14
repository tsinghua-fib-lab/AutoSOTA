# Copyright 2026 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any

# Canonical method ordering for output tables.
METHOD_ORDER = [
    # Baselines
    "Random", "RF+IS", "LR+IS", "RF+LURE", "LR+LURE",
    # Random selection + BQ
    "BQ-SF Rand", "BQ-RPF Rand", "BQ-TPF Rand",
    # Active selection + BQ
    "BQ-SF", "BQ-RPF", "BQ-TPF",
    "BQ-SF Rounded", "BQ-RPF Rounded", "BQ-TPF Rounded",
]


def save_experiment_results(
    dataset: str,
    target_model: str,
    setting: str,
    methods_results: Dict[str, List[np.ndarray]],
    true_mean: float,
    df_table: pd.DataFrame,
    results_dir: str = "results",
    gmm_abstained: bool = False,
    methods_variances: Dict[str, List[np.ndarray]] = None,
    budget: int = None,
):
    """
    Saves detailed experimental results including each step's MAE,
    estimates, and the final summary table.
    """
    os.makedirs(results_dir, exist_ok=True)
    
    # Create experiment signature for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    budget_tag = f"_budget{budget}" if budget is not None else ""
    exp_signature = f"{dataset}_{target_model}{budget_tag}_{setting}_{timestamp}"
    
    # 1. Save final table
    df_table["GMM_Abstained"] = gmm_abstained
    
    # Sort by canonical method order
    order_map = {name: i for i, name in enumerate(METHOD_ORDER)}
    df_table["_sort_key"] = df_table["Method"].map(
        lambda m: order_map.get(m, len(METHOD_ORDER))
    )
    df_table = df_table.sort_values("_sort_key").drop(columns=["_sort_key"])
    
    table_path = os.path.join(results_dir, f"summary_{exp_signature}.csv")
    df_table.to_csv(table_path, index=False)
    print(f"\nSaved summary table to {table_path}")
    
    # 2. Save detailed step-by-step estimates/MAE
    detailed_records = []
    for method, runs_estimates in methods_results.items():
        for run_idx, est_arr in enumerate(runs_estimates):
            for step_idx, est in enumerate(est_arr):
                mae = np.abs(est - true_mean)
                record = {
                    "method": method,
                    "run": run_idx,
                    "step": step_idx + 1,
                    "estimate": float(est),
                    "mae": float(mae),
                    "true_mean": float(true_mean)
                }
                if methods_variances and method in methods_variances:
                    var_arr = methods_variances[method][run_idx]
                    if var_arr is not None and len(var_arr) > step_idx:
                        record["integral_variance"] = float(var_arr[step_idx])
                detailed_records.append(record)
    
    df_detailed = pd.DataFrame(detailed_records)
    detailed_path = os.path.join(results_dir, f"detailed_{exp_signature}.csv")
    df_detailed.to_csv(detailed_path, index=False)
    print(f"Saved detailed step results to {detailed_path}")
