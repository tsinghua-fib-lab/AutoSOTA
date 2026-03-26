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
num_sequences = 1000
batch_size = 64

seed = 123
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

# Scenarios with specific bandwidth parameters
scenarios = [
    {"scenario": tpp.scenarios.multivariate.ServerStop(t_max, num_sequences), "h_int": 0.01, "h_arr": 0.01},
    {"scenario": tpp.scenarios.multivariate.ServerOverload(t_max, num_sequences), "h_int": 0.01, "h_arr": 0.01},
    {"scenario": tpp.scenarios.multivariate.Latency(t_max, num_sequences), "h_int": 0.5, "h_arr": 0.5},
    {"scenario": tpp.scenarios.multivariate.Connectome(), "h_int": 0.001, "h_arr": 0.5},
]

# ========== Process each scenario separately ==========
for scenario_dict in scenarios:
    scenario = scenario_dict["scenario"]
    h_int = scenario_dict["h_int"]
    h_arr = scenario_dict["h_arr"]
    
    print(f"{scenario.name} with h_int={h_int}, h_arr={h_arr}", flush=True)
    id_train = scenario.get_id_train()
    id_proper_train, id_cal, _ = id_train.train_val_test_split(train_size=0.5, val_size=0.5, test_size=0.0, seed=42)
    dl_train = id_proper_train.get_dataloader(batch_size=batch_size, shuffle=True)

    # Fit a neural TPP model on the id_proper_train
    ntpp = tpp.utils.fit_ntpp_model(dl_train, num_marks=id_proper_train.num_marks, max_epochs=500, patience=5)

    test_statistics = [
        tpp.statistics.kl_int,
        tpp.statistics.kl_arr,
    ]

    ## Estimate distribution of each test statistic under H_0 by using id_cal
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

    ## Compute test statistic for ID test sequences
    # ID test sequences will be compared to OOD test sequences to evaluate our CADES method
    id_test = scenario.get_id_test()
    id_test_batch = tpp.data.Batch.from_list(id_test)
    id_test_poisson_times = tpp.utils.extract_poisson_arrival_times(ntpp, id_test_batch)

    # Compute the statistics for all ID test sequences with scenario-specific parameters
    id_test_scores = {}
    for stat in test_statistics:
        name = stat.__name__
        id_test_scores[name] = stat(poisson_times_per_mark=id_test_poisson_times, model=ntpp, batch=id_test_batch, h_int=h_int, h_arr=h_arr)

    ## Compute test statistic for OOD test sequences & evaluate AUC ROC based on the p-values

    ntpp.cuda()
    results = []
    detectability_values = np.arange(0, 1, step=0.05)
    num_seeds = 10

    for seed in trange(num_seeds):
        for det in tqdm(detectability_values):
            np.random.seed(seed)
            ood_test = scenario.sample_ood(detectability=det, seed=seed)
            ood_test_batch = tpp.data.Batch.from_list(ood_test).cuda()
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

            id_pvals = np.minimum(np.minimum(2 * id_pvals_int, 2 * id_pvals_arr), 1)
            ood_pvals = np.minimum(np.minimum(2 * ood_pvals_int, 2 * ood_pvals_arr), 1)

            
            auc = tpp.utils.roc_auc_from_pvals(id_pvals, ood_pvals)

            res = {"statistic": "CADES", "seed": seed, "detectability": det, 
            "auc": auc, "scenario": scenario.name}
            results.append(res)


    df = pd.DataFrame(results)
    
    # Create output directory if it doesn't exist
    output_dir = "./experiments/output/output_multivariate"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create plot for this scenario
    plt.figure(dpi=100)
    sns.pointplot(data=df, x="detectability", y="auc", hue="statistic", ci=None)
    ax = plt.gca()
    ax.set_xticks([1, 5, 10, 15, 18])
    ax.set_title(f"{scenario.name}")
    plt.legend(fontsize=8)
    plt.savefig(f"{output_dir}/{scenario.name}_1.png", bbox_inches="tight")
    plt.show()