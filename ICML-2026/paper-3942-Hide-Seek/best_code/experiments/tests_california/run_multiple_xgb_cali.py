import itertools
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import repo_paths  # noqa: F401
from params import CONSTANT_PARAMS as PARAMS


def build_cmd(base_script, params):
    cmd = [sys.executable, base_script]
    for key, value in params.items():
        if value is None:
            continue

        flag = f"--{key}"
        if key == "xgb-params":
            cmd.extend([flag, json.dumps(value)])
        elif isinstance(value, list):
            cmd.append(flag)
            cmd.extend([str(v) for v in value])
        else:
            cmd.extend([flag, str(value)])
    return cmd

#trained on val with the grid, seed 10. got best params and ran with test on seeds 0-9.

best_params = {'max_depth': 7,
 'eta': 0.2,
 'colsample_bytree': 0.8,
 'objective': 'binary:logistic',
 'eval_metric': 'logloss',
 'seed': 10,
 'num_boost_round': 100}

# Grid to sweep.
max_depth_grid = [best_params['max_depth']]#[3, 5, 7]
eta_grid = [best_params['eta']]#[0.03, 0.1, 0.2]
colsample_bytree_grid = [best_params['colsample_bytree']]#[0.5, 0.8, 1.0]
num_boost_round_grid = [100]
seed_grid = [0,1,2,3,4,5,6,7,8,9]#[10]

# Fixed run settings.
model_type = "shap_xgboost"
folder_for_pickle = "ICML_experiments/california_tests/test"
lmbda = 0.06
num_important_features = 3
location_cols_to_use = ["longitude"]
eval_split = "test"

# Ensemble controls (optional).
n_ensemble = None
colsample = None
ensemble_parallel = None
ensemble_n_jobs = None
ensemble_backend = "loky"

# Optional overrides. Keep None to use defaults from hide_and_seek/params.py.
epochs_override = None
batch_size_override = None


if __name__ == "__main__":
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools_housing.py")

    if model_type not in PARAMS:
        raise ValueError(f"Unsupported model_type for PARAMS lookup: {model_type}")

    resolved_epochs = PARAMS[model_type]["epochs"] if epochs_override is None else epochs_override
    resolved_batch_size = PARAMS[model_type]["batch_size"] if batch_size_override is None else batch_size_override

    run_idx = 0
    for max_depth, eta, colsample_bytree, num_boost_round, seed in itertools.product(
        max_depth_grid,
        eta_grid,
        colsample_bytree_grid,
        num_boost_round_grid,
        seed_grid,
    ):
        run_idx += 1

        xgb_params = {
            "max_depth": max_depth,
            "eta": eta,
            "colsample_bytree": colsample_bytree,
            "num_boost_round": num_boost_round,
        }

        params = {
            "model-type": model_type,
            "folder-for-pickle": folder_for_pickle,
            "lmbda": lmbda,
            "seed": seed,
            "epochs": resolved_epochs,
            "batch-size": resolved_batch_size,
            "num-important-features": num_important_features,
            "location-cols-to-use": location_cols_to_use,
            "eval-split": eval_split,
            "n-ensemble": n_ensemble,
            "colsample": colsample,
            "ensemble-parallel": ensemble_parallel,
            "ensemble-n-jobs": ensemble_n_jobs,
            "ensemble-backend": ensemble_backend,
            "xgb-params": xgb_params,
        }

        cmd = build_cmd(script_path, params)
        print(f"\n[{run_idx}] RUNNING: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
