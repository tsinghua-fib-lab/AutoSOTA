
import sys, os
sys.path.insert(0, ".")
import repo_paths
from tools import run_feature_selection_model
import numpy as np

seeds = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,17,18,19,20]
tprs = []
fdrs = []
f1s = []

for seed in seeds:
    results = run_feature_selection_model(
        data_type="Syn4",
        model_type="hide_and_seek",
        seed=seed,
        lmbda=0.28,
        epochs=500,
        num_syn_features=11,
        train_N=10000,
        test_N=10000,
        hide_hidden_dim=32,
        seek_hidden_dim=32,
        hide_num_hidden_layers=2,
        seek_num_hidden_layers=2,
        lmbda_exponent=2,
        data_mode="synthetic",
        task="classification",
        save_experiment_data=False,
        return_results=True,
        warmup_epochs=100,
    )
    tprs.append(results["TPR_mean"])
    fdrs.append(results["FDR_mean"])
    f1s.append(results["f1"])

tpr_median = np.median(tprs)
fdr_median = np.median(fdrs)
f1_median = np.median(f1s)

print("=== HideSeek Syn4 Results (median of 20 runs) ===")
print("TPR: {:.2f}".format(tpr_median))
print("FDR: {:.2f}".format(fdr_median))
print("F1:  {:.2f}".format(f1_median))
print("")
print("All TPR: {}".format(sorted([round(x,1) for x in tprs])))
print("All FDR: {}".format(sorted([round(x,1) for x in fdrs])))
