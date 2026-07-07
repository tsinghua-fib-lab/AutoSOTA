import sys
import os
sys.path.append(os.path.abspath("../src"))

import numpy as np
import pandas as pd
import sys
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

config_name = sys.argv[1]

job_list = get_job_list(config_name)

def log_failure(task_id, error_msg):
    os.makedirs(f"../results/{config_name}/logs", exist_ok=True)
    with open(f"../results/{config_name}/logs/{task_id}_err.txt", "w") as f:
            f.write(error_msg)

def main():
    task_id = int(sys.argv[2])
    job = job_list[task_id]
    method = job['method']
    dataset = job['dataset']
    subgp_cols = job['subgp_cols']
    adjust_cols = job['adjust_cols']
    seed = job['seed']
    dataset_hyper = job['dataset_hyper']
    R_star = job['R_star']
    kwargs = job['kwargs']

    try:
        run_experiment(method, dataset, seed, subgp_cols, adjust_cols, config_name, task_id, dataset_hyper, R_star, **kwargs)
    except Exception as e:
        # capture traceback so you know what failed
        tb_str = traceback.format_exc()
        log_failure(task_id, tb_str)
        print(f"Config {config_name}, job {task_id} failed. Logged to results/{config_name}/logs/{task_id}_err.txt.")
        sys.exit(1)  # let SLURM know it failed

    

if __name__ == "__main__":
    main()
