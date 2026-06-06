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


t_max = 100
batch_size = 64
num_seeds = 5

alphas = np.round(np.linspace(0.05, 0.5, 10), 2) 
results = []  

for seed in trange(num_seeds):
    print(f"Processing seed: {seed}")
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    # Set the scenario
    scenario = tpp.scenarios.real_world.ServerLogs()
    #scenario = tpp.scenarios.real_world.STEAD()
    

    if isinstance(scenario, tpp.scenarios.real_world.ServerLogs):
        h_int = 0.01
        h_arr = 0.001
        ood_test_dataset_name = 'Packet corruption (1%)'
    else:  # STEAD
        h_int = 0.1
        h_arr = 0.001
        ood_test_dataset_name = 'Anchorage, AK'
        

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

    # Estimate distribution of each test statistic under H_0 by using id_cal
    id_cal_batch = tpp.data.Batch.from_list(id_cal)
    id_cal_poisson_times = tpp.utils.extract_poisson_arrival_times(ntpp, id_cal_batch)

    ecdfs = {}
    for stat in test_statistics:
        name = stat.__name__
        scores = stat(poisson_times_per_mark=id_cal_poisson_times, model=ntpp, batch=id_cal_batch, h_int=h_int, h_arr=h_arr)
        ecdfs[name] = ECDF(scores)

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

    # Compute test statistic for ID test sequences
    id_test_batch = tpp.data.Batch.from_list(id_test)
    id_test_poisson_times = tpp.utils.extract_poisson_arrival_times(ntpp, id_test_batch)

    id_test_scores = {}
    for stat in test_statistics:
        name = stat.__name__
        id_test_scores[name] = stat(poisson_times_per_mark=id_test_poisson_times, model=ntpp, batch=id_test_batch, h_int=h_int, h_arr=h_arr)

    # Compute test statistic for OOD test sequences & evaluate based on different alpha values
    for alpha in alphas:
        # Get the specific OOD dataset for this scenario
        ood_test = scenario.ood_test_datasets[ood_test_dataset_name]
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

                tpr, fpr = tpp.utils.tpr_fpr_from_pvals(id_pvals, ood_pvals, alpha=alpha)

                res = {
                    "alpha": alpha, 
                    "tpr": tpr, 
                    "fpr": fpr, 
                    "seed": seed, 
                    "statistic": "CADES", 
                    "scenario": scenario.name
                }
                results.append(res)
                
                del id_pvals_int, id_pvals_arr, ood_pvals_int, ood_pvals_arr


df = pd.DataFrame(results)

# Create output directory if it doesn't exist
output_dir = "./experiments/output/output_fpr_control"
os.makedirs(output_dir, exist_ok=True)


# Create plot
plt.figure(figsize=(6, 6))
data = [df[df['alpha'] == alpha]['fpr'].values for alpha in alphas]
plt.boxplot(data, positions=alphas, widths=0.03)

plt.plot([0.05, 0.5], [0.05, 0.5], linestyle="--", color="red", label=r"FPR=$\alpha$")

plt.xlim(0.02, 0.53)  # Set x-axis range
plt.ylim(0, 1.05)  # Set y-axis range
plt.xticks(ticks=np.arange(0.05, 0.55, 0.05), fontsize=12) 
plt.yticks(ticks=np.concatenate(([0], np.arange(0.05, 1, 0.15), [1])), fontsize=12) 

plt.xlabel(r"$\alpha$", fontsize=14)
plt.ylabel("FPR", fontsize=14)
plt.title(f"{scenario.name}", fontsize=16)
plt.grid(axis="both", linestyle="--", linewidth=0.5, alpha=0.7)
plt.legend(fontsize=14, loc="upper left", frameon=False)

# Save plot
output_path = f"{output_dir}/{scenario.name}_fpr_control.png"
plt.savefig(output_path, bbox_inches="tight")
plt.show()