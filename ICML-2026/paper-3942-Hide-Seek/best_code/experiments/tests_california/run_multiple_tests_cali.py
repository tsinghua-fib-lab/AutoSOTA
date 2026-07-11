import itertools
import subprocess
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import repo_paths  # noqa: F401
from params import CONSTANT_PARAMS as PARAMS

N_JOBS = 12


def build_cmd(base_script, params):
    cmd = [sys.executable, base_script]
    for key, value in params.items():
        if value is None:
            continue

        flag = f"--{key}"

        if isinstance(value, list):
            cmd.append(flag)
            cmd.extend([str(v) for v in value])
        elif key == "ensemble-parallel":
            cmd.extend([flag, str(value)])
        else:
            cmd.extend([flag, str(value)])
    return cmd


# Sweep dimensions requested for v1.
seeds_list = [0,1,2,3,4,5,6,7,8,9] #[10]#
lmbda_values = [0.5]#[0,0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5]

# Single-run settings (kept fixed across sweep).
model_type = "hide_and_seek"
folder_for_pickle = "ICML_experiments/california_tests/test"

# If set to None, defaults come from hide_and_seek/params.py for the selected model.
epochs_override = None
batch_size_override = None

num_important_features = 3
location_cols_to_use = ["longitude"]
eval_split = "test"

# Ensemble controls (optional).
n_ensemble = None
colsample = None
ensemble_parallel = None
ensemble_n_jobs = None
ensemble_backend = "loky"


if __name__ == "__main__":
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools_housing.py")

    if model_type not in PARAMS:
        raise ValueError(f"Unsupported model_type for PARAMS lookup: {model_type}")

    if model_type == 'lime':
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['OPENBLAS_NUM_THREADS'] = '1'

    resolved_epochs = PARAMS[model_type]['epochs'] if epochs_override is None else epochs_override
    resolved_batch_size = PARAMS[model_type]['batch_size'] if batch_size_override is None else batch_size_override

    jobs = list(itertools.product(seeds_list, lmbda_values))

    def _run_one(seed, lmbda):
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
        }
        cmd = build_cmd(script_path, params)
        print("RUNNING:", " ".join(cmd))
        subprocess.run(cmd, check=True, stderr=subprocess.PIPE)

    failed_jobs = []
    with ThreadPoolExecutor(max_workers=N_JOBS) as ex:
        fut_to_job = {ex.submit(_run_one, seed, lmbda): (seed, lmbda) for seed, lmbda in jobs}
        for fut in as_completed(fut_to_job):
            seed, lmbda = fut_to_job[fut]
            try:
                fut.result()
            except Exception as exc:
                failed_jobs.append((seed, lmbda, str(exc)))
                print(f"FAILED: seed={seed} lmbda={lmbda}: {exc}")

    if failed_jobs:
        print(f"\nFAILED JOBS ({len(failed_jobs)} total):")
        for seed, lmbda, err in failed_jobs:
            print(f"  seed={seed} lmbda={lmbda}: {err}")
