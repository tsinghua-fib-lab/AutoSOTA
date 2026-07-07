import sys
import os
sys.path.append(os.path.abspath("../src"))

import numpy as np
import pandas as pd
import traceback

from algs.cox_tree import *
from algs.ddgroup import *
from algs.prim import *
from algs.random import *
from algs.survival_tree import *
from algs.ost import *
from utils.subgroup import *
from utils.metrics import *
from evaluation.run_experiment import run_experiment, get_job_list

import warnings
warnings.filterwarnings("ignore")
np.seterr(all="ignore")
warnings.filterwarnings("ignore", module="sksurv.*")


def log_failure(config_name, task_id, error_msg):
    os.makedirs(f"../results/{config_name}/logs", exist_ok=True)
    with open(f"../results/{config_name}/logs/{task_id}_err.txt", "w") as f:
        f.write(error_msg)


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_local.py <config_name> [start_idx] [end_idx]")
        print("  config_name: name of YAML config file (without .yaml extension)")
        print("  start_idx: optional starting task index (default: 0)")
        print("  end_idx: optional ending task index (default: run all tasks)")
        sys.exit(1)

    config_name = sys.argv[1]

    # Get the full job list from config
    job_list = get_job_list(config_name)

    # Allow running a subset of jobs
    start_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    end_idx = int(sys.argv[3]) if len(sys.argv) > 3 else len(job_list)

    print(f"Config: {config_name}")
    print(f"Total jobs in config: {len(job_list)}")
    print(f"Running jobs {start_idx} to {end_idx-1}")
    print("=" * 80)

    failed_jobs = []

    for task_id in range(start_idx, end_idx):
        job = job_list[task_id]
        method = job['method']
        dataset = job['dataset']
        subgp_cols = job['subgp_cols']
        adjust_cols = job['adjust_cols']
        seed = job['seed']
        dataset_hyper = job['dataset_hyper']
        R_star = job['R_star']
        kwargs = job['kwargs']

        print(f"\n[Job {task_id}/{len(job_list)-1}] Dataset: {dataset}, Method: {method}, Seed: {seed}")
        print(f"  subgp_cols: {subgp_cols}, adjust_cols: {adjust_cols}")

        try:
            run_experiment(method, dataset, seed, subgp_cols, adjust_cols, config_name, task_id, dataset_hyper, R_star, **kwargs)
            print(f"  ✓ Job {task_id} completed successfully")
        except Exception as e:
            # Capture traceback so you know what failed
            tb_str = traceback.format_exc()
            log_failure(config_name, task_id, tb_str)
            print(f"  ✗ Job {task_id} FAILED. Error logged to ../results/{config_name}/logs/{task_id}_err.txt")
            print(f"  Error: {str(e)}")
            failed_jobs.append(task_id)

    print("\n" + "=" * 80)
    print(f"Completed {end_idx - start_idx} jobs")
    if failed_jobs:
        print(f"Failed jobs ({len(failed_jobs)}): {failed_jobs}")
    else:
        print("All jobs completed successfully!")


if __name__ == "__main__":
    main()