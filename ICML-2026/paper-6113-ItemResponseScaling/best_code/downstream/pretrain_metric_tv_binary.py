import os
import numpy as np
import pickle
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error                
from scipy.stats import spearmanr
from tueplots import bundles
bundles.icml2024()

def backtest_krr(x, y, A):
    x = x.reshape(-1, 1)
    n = len(y)
    split = int(A * n)
    x_tr, y_tr_gt = x[:split], y[:split]
    x_te, y_te_gt = x[split:], y[split:]
    kr = KernelRidge(alpha=1.0, kernel='rbf', gamma=0.1)
    kr.fit(x_tr, y_tr_gt)
    y_tr_pred = kr.predict(x_tr)
    y_te_pred = kr.predict(x_te)
    train_mae = mean_absolute_error(y_tr_pred, y_tr_gt) / (y.max() - y.min())
    test_mae = mean_absolute_error(y_te_pred, y_te_gt) / (y.max() - y.min())
    return (train_mae, test_mae)

def total_variance(a):
    n = len(a)
    num = np.abs(np.diff(a)).sum()
    denom = abs(a[-1] - a[0])
    return (n / (n - 1.0)) * (num / denom)

if __name__ == "__main__":
    RESULT_PATH = "/lfs/skampere2/0/sttruong/irsl/result/pretrain_binarycat_helm_full_backup/result.pkl"
    DROP_FIRST_FRAC = 0.10  # drop the first 10% of checkpoints (same idea as your reference)
    BUDGET = 50
    
    with open(RESULT_PATH, "rb") as f:
        results_dict = pickle.load(f)
        
    acc_tvs   = []
    theta_tvs = []
    rows = []
    rng = np.random.default_rng(0)  # reproducible subset

    for scenario, model_dict in results_dict.items():
        for model, payload in model_dict.items():
            ys      = torch.tensor(payload["resmat"].values)               # torch.Tensor (n_checkpoints, n_items)
            thetass = payload["thetass"]           # torch.Tensor (n_checkpoints, budget)

            final_thetas = thetass[:, -1].cpu().numpy()
            means_sub = torch.nanmean(ys[:, torch.randperm(ys.shape[1])[:BUDGET]], dim=1).cpu().numpy()

            # optional warm-up drop, like in your reference
            drop = int(DROP_FIRST_FRAC * len(final_thetas))
            final_thetas = final_thetas[drop:]
            means_sub    = means_sub[drop:]

            acc_tv   = total_variance(means_sub)
            theta_tv = total_variance(final_thetas)

            acc_tvs.append(acc_tv)
            theta_tvs.append(theta_tv)

    # Averages across all (scenario, model) pairs
    acc_tv_avg   = float(np.mean(acc_tvs))
    theta_tv_avg = float(np.mean(theta_tvs))

    print(f"=== TV averages across all datasets & models ===")
    print(f"Accuracy TV avg: {acc_tv_avg:.4f}")
    print(f"Theta TV avg:    {theta_tv_avg:.4f}\n")
