import sys
sys.path.append('.')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import anomaly_tpp as tpp
import argparse

from tqdm.auto import tqdm, trange
from statsmodels.distributions.empirical_distribution import ECDF
sns.set_style('whitegrid')

### Parse args ###
parser = argparse.ArgumentParser()
args = parser.parse_args()
print(args, flush=True)

num_sequences = 1000
t_max = 100

# Scenarios with specific bandwidth parameters
scenario_params = [
    {"scenario": tpp.scenarios.spp.IncreasingRate(t_max), "h_int": 1, "h_arr": 0.005},
    {"scenario": tpp.scenarios.spp.DecreasingRate(t_max), "h_int": 1, "h_arr": 0.005},
    {"scenario": tpp.scenarios.spp.InhomogeneousPoisson(t_max), "h_int": 1, "h_arr": 0.1},
    {"scenario": tpp.scenarios.spp.RenewalUp(t_max), "h_int": 1, "h_arr": 0.05},
    {"scenario": tpp.scenarios.spp.RenewalDown(t_max), "h_int": 0.5, "h_arr": 0.1},
    {"scenario": tpp.scenarios.spp.Hawkes(t_max), "h_int": 1, "h_arr": 0.1},
    {"scenario": tpp.scenarios.spp.SelfCorrecting(t_max), "h_int": 0.5, "h_arr": 0.1},
    {"scenario": tpp.scenarios.spp.Stopping(t_max), "h_int": 1, "h_arr": 0.05},
    {"scenario": tpp.scenarios.spp.Uniform(t_max), "h_int": 0.1, "h_arr": 0.1},
]


model = tpp.models.StandardPoissonProcess()

# List of test statistics to evaluate
test_statistics = [
    tpp.statistics.ks_arrival,
    tpp.statistics.ks_interevent,
    tpp.statistics.chi_squared,
    tpp.statistics.sum_of_squared_spacings,
    tpp.statistics.Q_plus_statistic,
    tpp.statistics.Q_minus_statistic,
    tpp.statistics.kl_int,
    tpp.statistics.kl_arr,
]

# Create output directory if it doesn't exist
output_dir = "./experiments/output/output_spp"
os.makedirs(output_dir, exist_ok=True)

# ========== Process each scenario separately ==========
results = []

detectability_values = np.arange(0, 1, step=0.05)
num_seeds = 10

for scenario_dict in tqdm(scenario_params, desc="Processing scenarios"):
    scenario = scenario_dict["scenario"]
    h_int = scenario_dict["h_int"]
    h_arr = scenario_dict["h_arr"]
    
    print(f"Processing scenario: {scenario.name} with h_int={h_int}, h_arr={h_arr}")
    
    # Sample ID training sequences
    id_train = scenario.sample_id(num_sequences)
    id_train_batch = tpp.data.Batch.from_list(id_train)
    id_train_poisson_times = tpp.utils.extract_poisson_arrival_times(model, id_train_batch)
    
    # Empirical distribution of each test statistic on id_train
    # This approximates the CDF of the test statistic under H_0
    # and is used to compute the p-values
    ecdfs = {}
    for stat in test_statistics:
        name = stat.__name__
        scores = stat(poisson_times_per_mark=id_train_poisson_times, h_int=h_int, h_arr=h_arr)
        ecdfs[name] = ECDF(scores)


    def twosided_pval(stat_name: str, scores: np.ndarray):
        """Compute the two-sided baseline p-value for the given values of test statistic.
        
        Args:
            stat_name: Name of the test statistic, 
                {"ks_arrival", "ks_interevent", "chi_squared", "sum_of_squared_spacings", "Q_plus_statistic", "Q_minus_statistic"}
            scores: Value of the statistic for each sample in the test set,
                shape [num_test_samples]
        
        Returns:
            p_vals: Two-sided baseline p-value for each sample in the test set,
                shape [num_test_samples]
        """
        ecdf = ecdfs[stat_name](scores)
        return 2 * np.minimum(ecdf, 1 - ecdf)


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

    # Get ID test sequences
    id_test = scenario.sample_id(num_sequences)
    id_test_batch = tpp.data.Batch.from_list(id_test)
    id_test_poisson_times = tpp.utils.extract_poisson_arrival_times(model, id_test_batch)

    # Compute statistics for ID test sequences with scenario-specific parameters
    id_test_scores = {}
    for stat in test_statistics:
        name = stat.__name__
        id_test_scores[name] = stat(poisson_times_per_mark=id_test_poisson_times, h_int=h_int, h_arr=h_arr)
    
    # Run experiments for each seed and detectability value
    for seed in trange(num_seeds, desc=f"Seeds for {scenario.name}"):
        for det in detectability_values:
            np.random.seed(seed)
            ood_test = scenario.sample_ood(num_sequences=num_sequences, detectability=det)
            ood_test_batch = tpp.data.Batch.from_list(ood_test)
            ood_poisson_times_per_mark = tpp.utils.extract_poisson_arrival_times(model, ood_test_batch)

            for stat in test_statistics:
                stat_name = stat.__name__
                if stat_name in ['kl_int', 'kl_arr']:
                    id_scores = id_test_scores[stat_name]
                    id_pvals = twosided_cf_pval(stat_name, id_scores)

                    ood_scores = stat(poisson_times_per_mark=ood_poisson_times_per_mark, h_int=h_int, h_arr=h_arr)
                    ood_pvals = twosided_cf_pval(stat_name, ood_scores)

                    if stat_name == 'kl_int':
                        id_pvals_int = id_pvals
                        ood_pvals_int = ood_pvals
                    elif stat_name == 'kl_arr':
                        id_pvals_arr = id_pvals
                        ood_pvals_arr = ood_pvals

                    if 'id_pvals_int' in locals() and 'id_pvals_arr' in locals():
                        id_pvals = np.minimum(np.minimum(2 * id_pvals_int, 2 * id_pvals_arr), 1)
                        ood_pvals = np.minimum(np.minimum(2 * ood_pvals_int, 2 * ood_pvals_arr), 1)

                        auc = tpp.utils.roc_auc_from_pvals(id_pvals, ood_pvals)
                        res = {
                            "statistic": "CADES", 
                            "seed": seed, 
                            "detectability": det,
                            "auc": auc, 
                            "scenario": scenario.name
                        }
                        results.append(res)
                        
                        del id_pvals_int, id_pvals_arr, ood_pvals_int, ood_pvals_arr

                else:
                    id_scores = id_test_scores[stat_name]
                    id_pvals = twosided_pval(stat_name, id_scores)

                    ood_scores = stat(poisson_times_per_mark=ood_poisson_times_per_mark, h_int=h_int, h_arr=h_arr)
                    ood_pvals = twosided_pval(stat_name, ood_scores)

                    auc = tpp.utils.roc_auc_from_pvals(id_pvals, ood_pvals)

                    res = {
                        "statistic": stat_name, 
                        "seed": seed, 
                        "detectability": det,
                        "auc": auc, 
                        "scenario": scenario.name
                    }
                    results.append(res)


    scenario_df = pd.DataFrame([r for r in results if r["scenario"] == scenario.name])
    
    # Create plot for this scenario
    plt.figure(figsize=[4, 3], dpi=100)
    sns.pointplot(data=scenario_df, x="detectability", y="auc", hue="statistic")
    ax = plt.gca()
    ax.set_xticks([1, 5, 10, 15, 18])
    ax.set_title(f"{scenario.name}")
    plt.legend(fontsize=8)
    output_path = f"{output_dir}/{scenario.name}.png"
    plt.savefig(output_path, bbox_inches="tight")
    plt.show()

print("Processing complete. Results saved to:", output_dir)