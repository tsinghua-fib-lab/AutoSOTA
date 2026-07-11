import itertools
import json
import os
import subprocess
import sys
import tempfile
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import pandas as pd
from joblib import Parallel, delayed
# from tabulate import tabulate

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import repo_paths  # noqa: F401
from tools import run_feature_selection_model
from params import CONSTANT_PARAMS as PARAMS

# True: fresh process per job via subprocess (required for INVASE — TF graph leak).
# False: joblib loky process pool calling run_feature_selection_model directly
#        (faster for LIME, anything non-INVASE — no per-job startup cost).
USE_SUBPROCESS = False

# Max number of concurrent workers. Applies to both modes (ThreadPoolExecutor
# pool size for USE_SUBPROCESS=True; joblib n_jobs for USE_SUBPROCESS=False).
N_JOBS = 18

DEFAULT_DATA_SETS = ['Syn1', 'Syn2', 'Syn3', 'Syn4', 'Syn5', 'Syn6']


def build_cmd(base_script, params):
    cmd = [sys.executable, base_script]
    for key, value in params.items():
        flag = f"--{key}"
        if key == "batchnorm-hs":
            if value is True:
                cmd.append(flag)  # including flag sets it to True because action="store_true".
        elif key == "return_losses_on_val":
            if value is True:
                cmd.append(flag)
        elif key == "ensemble-parallel":
            if value is not None:
                cmd.extend([flag, str(value)])
        elif key == "syn-switch-quantile":
            if value is not None:
                cmd.extend([flag, str(value)])
        elif key == "data-set":
            if value is not None:
                cmd.extend([flag, str(value)])
        elif key == "metrics-out":
            if value is not None:
                cmd.extend([flag, str(value)])
        else:
            cmd.extend([flag, str(value)])
    return cmd

seeds_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,17,18,19,20] #on seed 15 Hide&Seek Syn4 11 features collapsed. This is a 1/120 occurence (20 seeds, 6 datasets). Investigate
lmbda_exponent_s = [2] #[0.5, 1, 2, 3, 4, 5]
rho_s = [None]#[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1] #correlation between important and non-important features
num_important_features_s = ["use_gtruth"]
lmbda_s = [0.3]#[0.01, 0.05, 0.1, 0.2, 0.3, 0.4]
model_type_s = ["hide_and_seek"]
folder_for_pickle = "ICML_experiments/synthetic"
data_mode = "synthetic" #credit_data_val #synthetic
perturbation_method_s = ["draw_marginal"]#["draw_marginal","knock_off","conditional_rf"]
n_ensemble = None #10
colsample_s = [None] #0.8
syn_switch_quantile_s = [None]#[1/20, 1/10, 1/4, 1/3, 1/2]  # set to a float in (0,1) when using Syn4Q/Syn5Q/Syn6Q #note this experiment was not reported in the paper
data_sets = [None]#[None] gives (Syn1-Syn6); single dataset: e.g. ['Syn4']; These ['Syn4Q','Syn5Q','Syn6Q'] are another experiment not reported on in the paper
num_syn_features = 11

# Optionally restrict which jobs run. Each entry is (lmbda, rho, data_set, seed).
# Leave as [] to run everything in the product.
OVERRIDE_JOBS = []

ensemble_parallel = True #None, True, False
if ensemble_parallel:
    ensemble_n_jobs = -1 #None, int, -1
else:
    ensemble_n_jobs = None
ensemble_backend = 'loky'


SCRIPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_synthetic_tests.py")


def _resolve_active_data_sets():
    if data_sets == [None]:
        return list(DEFAULT_DATA_SETS)
    return list(data_sets)


def _run_one_subprocess(data_set, base_params, tmpdir, tag):
    """Run a single (param_combo, data_set) subprocess; return parsed metrics dict."""
    metrics_path = os.path.join(tmpdir, f"metrics_{tag}.json")
    params = dict(base_params)
    params["data-set"] = data_set
    params["metrics-out"] = metrics_path

    cmd = build_cmd(SCRIPT_PATH, params)
    print("RUNNING:", " ".join(cmd))
    subprocess.run(cmd, check=True, stderr=subprocess.PIPE)

    with open(metrics_path, "r") as f:
        return json.load(f)


def _run_one_direct(combo_key, ds, params):
    """In-process worker: call run_feature_selection_model directly.

    Used when USE_SUBPROCESS=False. Returns a dict with success flag, metrics
    on success, error traceback on failure — so a single failed job doesn't
    abort the whole Parallel(...) call.
    """
    model_type = params["model-type"]

    # LIME-only: collapse BLAS to single-threaded in this worker process to
    # avoid N_JOBS × BLAS-threads oversubscription. Backwards-compatible at
    # the metric level (only machine-epsilon coefficient differences).
    if model_type == 'lime':
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['OPENBLAS_NUM_THREADS'] = '1'

    epochs = PARAMS[model_type]['epochs'] if params["epochs"] is None else params["epochs"]
    batch_size = PARAMS[model_type]['batch_size'] if params["batch-size"] is None else params["batch-size"]

    try:
        results = run_feature_selection_model(
            data_type=ds,
            model_type=model_type,
            folder_for_pickle=params["folder-for-pickle"],
            seed=params["seed"],
            lmbda=params["lmbda"],
            rho=params["rho"],
            num_syn_features=params["num-syn-features"],
            num_important_features=params["num-important-features"],
            train_N=params["train-N"],
            test_N=params["test-N"],
            hide_hidden_dim=params["hide-hidden-dim"],
            seek_hidden_dim=params["seek-hidden-dim"],
            hide_num_hidden_layers=params["hide-num-hidden-layers"],
            seek_num_hidden_layers=params["seek-num-hidden-layers"],
            batchnorm_hs=params["batchnorm-hs"],
            return_losses_on_val=params["return_losses_on_val"],
            lmbda_exponent=params["lmbda-exponent"],
            data_mode=params["data_mode"],
            perturbation_method=params["perturbation-method"],
            n_ensemble=params["n-ensemble"],
            colsample=params["colsample"],
            ensemble_parallel=params["ensemble-parallel"],
            ensemble_n_jobs=params["ensemble-n-jobs"],
            ensemble_backend=params["ensemble-backend"],
            syn_switch_quantile=params["syn-switch-quantile"],
            epochs=epochs,
            batch_size=batch_size,
            save_experiment_data=True,
            return_results=True,
        )
        return {"success": True, "combo_key": combo_key, "ds": ds,
                "metrics": {"TPR_mean": float(results["TPR_mean"]),
                            "FDR_mean": float(results["FDR_mean"]),
                            "f1":       float(results["f1"])}}
    except Exception as exc:
        return {"success": False, "combo_key": combo_key, "ds": ds,
                "error": f"{exc}\n{traceback.format_exc()}"}


def _print_combo_table(combo_key, active_data_sets, metrics_by_job):
    rows = []
    for ds in active_data_sets:
        m = metrics_by_job.get((combo_key, ds))
        if m is None:
            rows.append({"syn": ds, "TPR": float("nan"), "FDR": float("nan"), "F1": float("nan")})
        else:
            rows.append({
                "syn": ds,
                "TPR": round(m["TPR_mean"], 4),
                "FDR": round(m["FDR_mean"], 4),
                "F1":  round(m["f1"],       4),
            })
    df = pd.DataFrame(rows)
    mean_row = {
        "syn": "mean",
        "TPR": round(df["TPR"].mean(), 4),
        "FDR": round(df["FDR"].mean(), 4),
        "F1":  round(df["F1"].mean(),  4),
    }
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

    print("\n===== COMBO =====")
    # combo_key fields, in the same order they're built below:
    (lmbda, lmbda_exponent, rho, num_important_features, model_type,
     perturbation_method, colsample, seed, syn_switch_quantile) = combo_key
    print(f"seed={seed} lmbda={lmbda} lmbda_exponent={lmbda_exponent} rho={rho} "
          f"num_important_features={num_important_features} model_type={model_type} "
          f"perturbation_method={perturbation_method} colsample={colsample} "
          f"syn_switch_quantile={syn_switch_quantile}")
    # print(tabulate(df, headers="keys", tablefmt="fancy_grid"))
    print(df)


if __name__ == "__main__":
    active_data_sets = _resolve_active_data_sets()

    combo_order = []           # preserves itertools.product ordering for printing
    combo_to_params = {}       # combo_key -> base params dict (no data-sets / metrics-out yet)

    for lmbda, lmbda_exponent, rho, num_important_features, model_type, perturbation_method, colsample, seed, syn_switch_quantile in itertools.product(
        lmbda_s,
        lmbda_exponent_s,
        rho_s,
        num_important_features_s,
        model_type_s,
        perturbation_method_s,
        colsample_s,
        seeds_list,
        syn_switch_quantile_s,
    ):
        combo_key = (lmbda, lmbda_exponent, rho, num_important_features, model_type,
                     perturbation_method, colsample, seed, syn_switch_quantile)
        params = {
            "lmbda": lmbda,
            "seed": seed,
            "rho": rho,
            "model-type": model_type,
            "folder-for-pickle": folder_for_pickle,
            "data_mode": data_mode,
            "perturbation-method": perturbation_method,
            "n-ensemble": n_ensemble,
            "colsample": colsample,
            "ensemble-parallel": ensemble_parallel,
            "ensemble-n-jobs": ensemble_n_jobs,
            "ensemble-backend": ensemble_backend,
            "batchnorm-hs": False,   # handled specially
            "return_losses_on_val": False,
            "epochs": None,
            "batch-size": None,
            "num-syn-features": num_syn_features,
            "num-important-features": num_important_features,
            "train-N": 10_000,
            "test-N": 10_000,
            "hide-hidden-dim": 32,
            "seek-hidden-dim": 32,
            "hide-num-hidden-layers": 2,
            "seek-num-hidden-layers": 2,
            "lmbda-exponent": lmbda_exponent,
            "syn-switch-quantile": syn_switch_quantile,
        }
        combo_order.append(combo_key)
        combo_to_params[combo_key] = params

    # Build the flat list of (combo_key, data_set) jobs to actually submit.
    if OVERRIDE_JOBS:
        override_set = {(l, r, ds, s) for l, r, ds, s in OVERRIDE_JOBS}
        jobs_to_run = [
            (combo_key, ds)
            for combo_key in combo_order
            for ds in active_data_sets
            if (combo_key[0], combo_key[2], ds, combo_key[7]) in override_set
        ]
    else:
        jobs_to_run = [(combo_key, ds) for combo_key in combo_order for ds in active_data_sets]

    print(f"Total param combos: {len(combo_order)} | data sets per combo: {len(active_data_sets)} "
          f"| jobs to run: {len(jobs_to_run)} | N_JOBS: {N_JOBS}"
          + (" (OVERRIDE active)" if OVERRIDE_JOBS else ""))

    metrics_by_job = {}  # (combo_key, data_set) -> metrics dict
    failed_jobs = []     # list of (combo_key, data_set, cmd_str, error_msg)

    print(f"Mode: {'SUBPROCESS (fresh process per job)' if USE_SUBPROCESS else 'DIRECT (joblib loky pool)'}")

    if USE_SUBPROCESS:
        with tempfile.TemporaryDirectory(prefix="run_multiple_") as tmpdir:
            with ThreadPoolExecutor(max_workers=N_JOBS) as ex:
                fut_to_job = {}
                for i, (combo_key, ds) in enumerate(jobs_to_run):
                    base_params = combo_to_params[combo_key]
                    tag = f"job{i}_{ds}"
                    fut = ex.submit(_run_one_subprocess, ds, base_params, tmpdir, tag)
                    fut_to_job[fut] = (combo_key, ds)

                for fut in as_completed(fut_to_job):
                    combo_key, ds = fut_to_job[fut]
                    try:
                        metrics_by_job[(combo_key, ds)] = fut.result()
                    except subprocess.CalledProcessError as exc:
                        cmd_str = " ".join(exc.cmd)
                        stderr_str = exc.stderr.decode() if exc.stderr else ""
                        err_full = f"{exc}\n{stderr_str}" if stderr_str else str(exc)
                        failed_jobs.append((combo_key, ds, cmd_str, err_full))
                    except Exception as exc:
                        failed_jobs.append((combo_key, ds, "", str(exc)))
                        print(f"FAILED: {ds} combo={combo_key}: {exc}")
    else:
        results_list = Parallel(n_jobs=N_JOBS, backend='loky')(
            delayed(_run_one_direct)(combo_key, ds, combo_to_params[combo_key])
            for combo_key, ds in jobs_to_run
        )
        for r in results_list:
            if r["success"]:
                metrics_by_job[(r["combo_key"], r["ds"])] = r["metrics"]
            else:
                failed_jobs.append((r["combo_key"], r["ds"], "", r["error"]))
                print(f"FAILED: {r['ds']} combo={r['combo_key']}: {r['error']}")

    # Print per-combo tables in original iteration order (mirrors prior output format).
    for combo_key in combo_order:
        _print_combo_table(combo_key, active_data_sets, metrics_by_job)

    if failed_jobs:
        print(f"\n{'='*60}")
        print(f"FAILED JOBS ({len(failed_jobs)} total):")
        for combo_key, ds, cmd_str, err in failed_jobs:
            print(f"  {ds} combo={combo_key}")
            if cmd_str:
                print(f"    CMD: {cmd_str}")
            print(f"    ERR: {err}")

        log_dir = os.path.expanduser(f"~/Data/{folder_for_pickle}")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"failed_jobs_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
        with open(log_path, "w") as f:
            f.write(f"Failed jobs from run at {datetime.now().isoformat()}\n")
            f.write(f"Total failed: {len(failed_jobs)}\n\n")
            for combo_key, ds, cmd_str, err in failed_jobs:
                f.write(f"dataset: {ds}\ncombo: {combo_key}\n")
                if cmd_str:
                    f.write(f"cmd: {cmd_str}\n")
                f.write(f"error: {err}\n\n")
        print(f"\nFailed jobs log written to: {log_path}")
