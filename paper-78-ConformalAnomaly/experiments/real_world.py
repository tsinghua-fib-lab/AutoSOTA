import sys
sys.path.append('.')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import random
import torch
import torch.nn as nn
import anomaly_tpp as tpp
import argparse

from tqdm.auto import tqdm, trange
from statsmodels.distributions.empirical_distribution import ECDF

sns.set_style("whitegrid")

### Parse args ###
parser = argparse.ArgumentParser()
args = parser.parse_args()
print(args, flush=True)

# Dataset-specific bandwidth parameters
STEAD_params = {
    'Anchorage, AK': {'h_int': 0.1, 'h_arr': 0.001},
    'Aleutian Islands, AK': {'h_int': 0.01, 'h_arr': 0.005},
    'Helmet, CA': {'h_int': 0.1, 'h_arr': 0.001}
}

ServerLogs_params = {
    'Packet delay (frontend)': {'h_int': 0.01, 'h_arr': 0.05},
    'Packet corruption (10%)': {'h_int': 0.01, 'h_arr': 0.001},
    'Packet corruption (1%)': {'h_int': 0.005, 'h_arr': 0.005},
    'Packet delay (all services)': {'h_int': 0.01, 'h_arr': 0.005},
    'Packet duplication(1%)': {'h_int': 0.1, 'h_arr': 0.001}
}

t_max = 100
batch_size = 64
num_seeds = 5

results = []

for seed in trange(num_seeds):
    print(f"Processing seed: {seed}")
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    # Set the scenario
    scenario = tpp.scenarios.real_world.ServerLogs()
    #scenario = tpp.scenarios.real_world.STEAD()
    

    # Select the appropriate parameter dictionary based on scenario
    if isinstance(scenario, tpp.scenarios.real_world.ServerLogs):
        params_dict = ServerLogs_params
    else:  # STEAD
        params_dict = STEAD_params
        
    print(f"Running scenario: {scenario.name}")

    id_train = scenario.id_train
    id_proper_train, id_cal, _ = id_train.train_val_test_split(train_size=0.5, val_size=0.5, test_size=0.0, seed=seed)
    id_test = scenario.id_test

    dl_train = id_proper_train.get_dataloader(batch_size=batch_size, shuffle=True)

    # Fit a neural TPP model on the id_proper_train
    ntpp = tpp.utils.fit_ntpp_model(dl_train, num_marks=id_proper_train.num_marks, max_epochs=500, patience=5) 
    
    
    test_statistics = [
        tpp.statistics.kl_int,
        tpp.statistics.kl_arr,
    ]

    # Get ID test sequences
    id_test_batch = tpp.data.Batch.from_list(id_test)
    id_test_poisson_times = tpp.utils.extract_poisson_arrival_times(ntpp, id_test_batch)

    # Process each OOD dataset with its own parameters
    for name, ood_test in scenario.ood_test_datasets.items():
        print(f"Processing OOD dataset: {name}")
        
        # Get the dataset-specific parameters
        h_int = params_dict[name]['h_int']
        h_arr = params_dict[name]['h_arr']
        
        print(f"Using h_int={h_int}, h_arr={h_arr} for {name}")
        
        # Estimate distribution of each test statistic under H_0 by using id_cal
        id_cal_batch = tpp.data.Batch.from_list(id_cal)
        id_cal_poisson_times = tpp.utils.extract_poisson_arrival_times(ntpp, id_cal_batch)
        
        ecdfs = {}
        for stat in test_statistics:
            stat_name = stat.__name__
            scores = stat(poisson_times_per_mark=id_cal_poisson_times, model=ntpp, batch=id_cal_batch, h_int=h_int, h_arr=h_arr)
            ecdfs[stat_name] = ECDF(scores)
        
        def twosided_cf_pval(stat_name: str, scores: np.ndarray):
            """Compute the two-sided conformal p-value for the given values of test statistic.
            
            Args:
                stat_name: Name of the test statistic, 
                    {"kl_int", "kl_arr"}
                scores: Value of the statistic for each sample in the test set,
                    shape [num_test_samples]
            
            Returns:
                p_vals: Two-sided conformal p-value for each sample in the test set,
                    shape [num_test_samples]
            """
            ecdf = ecdfs[stat_name](scores)
            n_count = len(ecdfs[stat_name].x)-1
            pvalue_right = ((1 - ecdf) * n_count + 1) / (n_count + 1)
            p_value_left = (ecdf * n_count + 1) / (n_count + 1)
            return np.minimum(np.minimum(2*pvalue_right, 2*p_value_left),1)
        
        # Compute ID test scores with the dataset-specific parameters
        id_test_scores = {}
        for stat in test_statistics:
            stat_name = stat.__name__
            id_test_scores[stat_name] = stat(poisson_times_per_mark=id_test_poisson_times, model=ntpp, batch=id_test_batch, h_int=h_int, h_arr=h_arr)
        
        # Process OOD data
        ood_test_batch = tpp.data.Batch.from_list(ood_test)
        ood_test_poisson_times = tpp.utils.extract_poisson_arrival_times(ntpp, ood_test_batch)

        for stat in test_statistics:
            stat_name = stat.__name__
            
            id_scores = id_test_scores[stat_name]
            id_pvals = twosided_cf_pval(stat_name, id_scores)

            ood_scores = stat(poisson_times_per_mark=ood_test_poisson_times, model=ntpp, batch=ood_test_batch, h_int=h_int, h_arr=h_arr)
            ood_pvals = twosided_cf_pval(stat_name, ood_scores)

            if stat_name == 'kl_int':
                id_pvals_int = id_pvals
                ood_pvals_int = ood_pvals
            elif stat_name == 'kl_arr':
                id_pvals_arr = id_pvals
                ood_pvals_arr = ood_pvals

            if 'id_pvals_int' in locals() and 'id_pvals_arr' in locals():
                id_pvals = np.minimum(np.minimum(2*id_pvals_int, 2*id_pvals_arr), 1)
                ood_pvals = np.minimum(np.minimum(2*ood_pvals_int, 2*ood_pvals_arr), 1)

                auc = tpp.utils.roc_auc_from_pvals(id_pvals, ood_pvals)

                res = {
                    "statistic": "CADES", 
                    "auc": auc,
                    "scenario": name,
                    "seed": seed
                }
                results.append(res)
                
                del id_pvals_int, id_pvals_arr, ood_pvals_int, ood_pvals_arr


df = pd.DataFrame(results)

# Create output directory if it doesn't exist
output_dir = "./experiments/output/output_real_world"
os.makedirs(output_dir, exist_ok=True)

# Group results and save to CSV
grouped_df = df.groupby(['scenario', 'statistic'])[['auc']].agg('mean').sort_values(by=['scenario', 'statistic']).reset_index()
grouped_df['auc'] = grouped_df['auc'].apply(lambda x: round(x * 100, 2))
grouped_df.to_csv(f"{output_dir}/{scenario.name}.csv", index=False)