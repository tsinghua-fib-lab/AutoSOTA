import itertools
import os
import sys
import traceback
from datetime import datetime

from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import repo_paths  # noqa: F401
from tools import run_feature_selection_model

N_JOBS = 18

NUM_BOOST_ROUND = 100

BEST_PARAMS_11 = {'max_depth': 5, #these are the best params for 11 features
'eta': 0.1,
'colsample_bytree': 0.9,
'num_boost_round': NUM_BOOST_ROUND #didn't vary
}

BEST_PARAMS_100 = {'max_depth': 7, # these are teh best params for 100 features
'eta': 0.1,
'colsample_bytree': 1.0,
'num_boost_round': NUM_BOOST_ROUND}

def _run_one_job(data_set, params):
    p = dict(params)
    xgb_params = p.pop("xgb-params")
    try:
        results = run_feature_selection_model(
            data_type=data_set,
            model_type=p["model-type"],
            folder_for_pickle=p["folder-for-pickle"],
            seed=p["seed"],
            num_syn_features=p["num-syn-features"],
            num_important_features=p["num-important-features"],
            train_N=p["train-N"],
            test_N=p["test-N"],
            data_mode=p["data_mode"],
            rho=p["rho"],
            xgb_params=xgb_params,
            syn_switch_quantile=p["syn-switch-quantile"],
            save_experiment_data=True,
            return_results=True,
        )
        return {"success": True, "data_set": data_set,
                "TPR_mean": float(results["TPR_mean"]),
                "FDR_mean": float(results["FDR_mean"]),
                "f1": float(results["f1"])}
    except Exception as exc:
        return {"success": False, "data_set": data_set, "params": params,
                "error": f"{exc}\n{traceback.format_exc()}"}


if __name__ == "__main__":

    folder_for_pickle = "ICML_experiments/switch_quantile" #"ICML_experiments/correlated/tuning/xgboost" 

    seed_grid = [0,1,2,3,4]
    rho_grid = [None]#[0.1, 0.3, 0.5, 0.7, 0.9]
    num_important_features_grid = ['use_gtruth']
    syn_switch_quantile_s = [1/20, 1/10, 1/4, 1/3, 1/2]# [None]  # e.g. [1/20, 1/10, 1/4, 1/3, 1/2] for Syn4Q/Syn5Q/Syn6Q. or [None] for Syn1-Syn6
    data_sets = ['Syn4Q','Syn5Q','Syn6Q']#['Syn1', 'Syn2', 'Syn3', 'Syn4', 'Syn5', 'Syn6']  # e.g. ['Syn4Q','Syn5Q','Syn6Q'] or ['Syn1', 'Syn2', 'Syn3', 'Syn4', 'Syn5', 'Syn6']

    params_mode = 'best_11' # 'best_11' or 'best_100' or #'grid'


    if params_mode == 'best_11':
        max_depth_grid = [BEST_PARAMS_11['max_depth']]
        eta_grid = [BEST_PARAMS_11['eta']]
        colsample_bytree_grid = [BEST_PARAMS_11['colsample_bytree']]
    elif params_mode == 'best_100':
        max_depth_grid = [BEST_PARAMS_100['max_depth']]
        eta_grid = [BEST_PARAMS_100['eta']]
        colsample_bytree_grid = [BEST_PARAMS_100['colsample_bytree']]
    elif params_mode == 'grid':
        max_depth_grid = [3, 5, 7, 10]
        eta_grid = [0.01, 0.05, 0.1, 0.2]
        colsample_bytree_grid = [0.5, 0.7, 0.9, 1.0]

    num_boost_round = NUM_BOOST_ROUND

    combos = list(itertools.product(
        max_depth_grid,
        eta_grid,
        colsample_bytree_grid,
        seed_grid,
        num_important_features_grid,
        rho_grid,
        syn_switch_quantile_s,
    ))

    jobs = []
    job_info_for_idx = []
    for i, (max_depth, eta, colsample_bytree, seed, num_important_features, rho, syn_switch_quantile) in enumerate(combos):
        xgb_params = {
            "max_depth": max_depth,
            "eta": eta,
            "colsample_bytree": colsample_bytree,
            "num_boost_round": num_boost_round,
        }
        base_params = {
            "model-type": "shap_xgboost",
            "folder-for-pickle": folder_for_pickle,
            "seed": seed,
            "num-syn-features": 11,
            "train-N": 10_000,
            "test-N": 10_000,
            "data_mode": "synthetic",
            "num-important-features": num_important_features,
            "rho": rho,
            "syn-switch-quantile": syn_switch_quantile,
            "xgb-params": xgb_params,
        }
        for ds in data_sets:
            jobs.append((ds, base_params))
            job_info_for_idx.append((i, max_depth, eta, colsample_bytree, seed, rho, syn_switch_quantile, ds))

    print(f"Total combos: {len(combos)} | datasets: {len(data_sets)} | total jobs: {len(jobs)} | N_JOBS: {N_JOBS}")

    results_list = Parallel(n_jobs=N_JOBS, backend='loky')(
        delayed(_run_one_job)(ds, params) for ds, params in jobs
    )

    failed_jobs = []
    for idx, r in enumerate(results_list):
        if not r["success"]:
            failed_jobs.append((job_info_for_idx[idx], r["error"]))
            print(f"FAILED: {job_info_for_idx[idx]}: {r['error']}")

    if failed_jobs:
        print(f"\n{'='*60}")
        print(f"FAILED JOBS ({len(failed_jobs)} total):")
        for job_info, err in failed_jobs:
            print(f"  {job_info}")
            print(f"    ERR: {err}")

        log_dir = os.path.expanduser(f"~/Data/{folder_for_pickle}")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"failed_jobs_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
        with open(log_path, "w") as f:
            f.write(f"Failed jobs from run at {datetime.now().isoformat()}\n")
            f.write(f"Total failed: {len(failed_jobs)}\n\n")
            for job_info, err in failed_jobs:
                f.write(f"job: {job_info}\n")
                f.write(f"error: {err}\n\n")
        print(f"\nFailed jobs log written to: {log_path}")
    else:
        print("\nAll jobs completed successfully.")
