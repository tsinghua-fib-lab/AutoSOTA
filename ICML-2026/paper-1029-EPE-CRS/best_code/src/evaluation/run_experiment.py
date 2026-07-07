import pandas as pd
import numpy as np
import os
import time
import yaml

from data.load import load_data
from utils.metrics import run_eval
from config.constants import METHOD_DICT, METHOD_HYPERS


def get_job_list(config_name, config_dir=None):
    if config_dir is None:
        with open(f"configs/{config_name}.yaml", "r") as f:
            config = yaml.safe_load(f)
    else:
        with open(f"{config_dir}/{config_name}.yaml", "r") as f:
            config = yaml.safe_load(f)

    job_datasets = config["datasets"]
    job_methods = config["methods"]

    job_list = []
    for dataset in job_datasets:
        for dataset_hyper in job_datasets[dataset]["hypers"]:
            for subgp_cols, adjust_cols in zip(job_datasets[dataset]["subgp_cols"], job_datasets[dataset]["adjust_cols"]):
                for seed in job_datasets[dataset]["seeds"]:
                    for method in job_methods:
                        for kwargs in METHOD_HYPERS[method]:
                            if 'R_star' in job_datasets[dataset]:
                                job_list.append({'method': method,
                                                 'dataset': dataset,
                                                 'subgp_cols': subgp_cols,
                                                 'adjust_cols': adjust_cols,
                                                 'seed': seed,
                                                 'dataset_hyper': dataset_hyper,
                                                 'R_star': np.array(job_datasets[dataset]['R_star']),
                                                 'kwargs': kwargs})
                            else:
                                job_list.append({'method': method,
                                                 'dataset': dataset,
                                                 'subgp_cols': subgp_cols,
                                                 'adjust_cols': adjust_cols,
                                                 'seed': seed,
                                                 'dataset_hyper': dataset_hyper,
                                                 'R_star': None,
                                                 'kwargs': kwargs})
    return job_list


def run_experiment(method, dataset, seed, subgp_cols, adjust_cols, config_name, task_id, dataset_hyper=None, R_star=None, **kwargs):
    start = time.time()
    alg_job = METHOD_DICT[method]

    print(f"Load data: dataset={dataset}, adjust_cols={adjust_cols}, subgp_cols={subgp_cols}, seed={seed}, n={dataset_hyper}", flush=True)
    X_adjust, X_subgp, Y, X_adjust_test, X_subgp_test, Y_test, B_subgp, scaler = load_data(
        dataset, adjust_cols, subgp_cols, seed, n=dataset_hyper
    )
    print("Finished loading", flush=True)
    print(f"  X_adjust shape: {X_adjust.shape}, X_subgp shape: {X_subgp.shape}", flush=True)

    # New approach: subgp_cols and adjust_cols are both absolute indices in original feature space
    # They are handled independently at the data loading level
    print("Running method", flush=True)
    results = alg_job(X_adjust, X_subgp, Y, B_subgp, **kwargs)
    print("Finished running", flush=True)

    print("Run eval", flush=True)
    eval_df = run_eval(X_adjust, X_subgp, Y, X_adjust_test, X_subgp_test, Y_test, results, R_star=R_star)
    print("Finished eval", flush=True)

    results_df = pd.DataFrame(results)
    full_df = pd.DataFrame(data=pd.concat([results_df, eval_df], axis=1))
    full_df['seed'] = seed
    full_df['subgp_cols'] = [subgp_cols for _ in range(len(full_df))]
    full_df['adjust_cols'] = [adjust_cols for _ in range(len(full_df))]

    runtime = time.time() - start

    print("Saving data", flush=True)
    os.makedirs(f"../results/{config_name}/raw", exist_ok=True)
    pd.to_pickle(full_df, f"../results/{config_name}/raw/{task_id}_results.pkl")
    print("Save completed", flush=True)

    print("Saving runtime", flush=True)
    runtime_df = pd.DataFrame({
                'method': [method],
                'dataset': [dataset],
                'subgp_cols': [subgp_cols],
                'adjust_cols': [adjust_cols],
                'seed': [seed],
                'kwargs': [kwargs],
                'runtime': [runtime]
                })
    pd.to_pickle(runtime_df, f"../results/{config_name}/raw/{task_id}_runtime.pkl")
    print("Save completed", flush=True)
