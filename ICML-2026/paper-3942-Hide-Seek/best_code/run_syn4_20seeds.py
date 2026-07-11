import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(".")))
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
        lmbda=0.3,
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
    )
    tpr = results["TPR_mean"]
    fdr = results["FDR_mean"]
    f1 = results["f1"]
    tprs.append(tpr)
    fdrs.append(fdr)
    f1s.append(f1)
    print("Seed %d: TPR=%.4f, FDR=%.4f, F1=%.4f" % (seed, tpr, fdr, f1))

tpr_median = np.median(tprs)
fdr_median = np.median(fdrs)
f1_median = np.median(f1s)
print("")
print("Median TPR: %.2f" % tpr_median)
print("Median FDR: %.2f" % fdr_median)
print("Median F1: %.2f" % f1_median)
print("")
print("All TPR: %s" % sorted([round(x,1) for x in tprs]))
print("All FDR: %s" % sorted([round(x,1) for x in fdrs]))
