#!/usr/bin/env python3
"""Evaluation script for NRDE on mammography.
Reproduces AUROC and AUPRC with best hyperparameters found via grid search.
"""
import os, sys, numpy as np, torch, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from NRDE import NRDE_run, read_data

def main():
    data_path = "/datasets/23_mammography.npz"
    # Best hyperparameters from grid search (n_runs=5 average)
    lr = 0.005
    grad_pun = 1.0  # lambda
    n_epochs = 100
    bs = 512
    mid_dim = 2048
    act = 2
    PNAL = "L_1sq"
    n_runs = 5
    seeds = [42, 52, 62, 72, 82]

    results = []
    for seed in seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        train_data, train_labels, test_data, test_labels, _, _ = read_data(
            data_path, normalization="z-score", seed=seed
        )
        auroc, auprc = NRDE_run(
            train_data, train_labels, test_data, test_labels,
            lr=lr, grad_pun=grad_pun, n_epochs=n_epochs, bs=bs, mid_dim=mid_dim,
            act=act, adam=True, PNAL=PNAL, verbose=False
        )
        results.append({"seed": seed, "auroc": float(auroc), "auprc": float(auprc)})

    auroc_mean = np.mean([r["auroc"] for r in results])
    auroc_std = np.std([r["auroc"] for r in results])
    auprc_mean = np.mean([r["auprc"] for r in results])
    auprc_std = np.std([r["auprc"] for r in results])

    print(json.dumps({
        "auroc_mean": round(auroc_mean, 4),
        "auroc_std": round(auroc_std, 4),
        "auprc_mean": round(auprc_mean, 4),
        "auprc_std": round(auprc_std, 4),
        "individual_runs": results,
    }, indent=2))

if __name__ == "__main__":
    main()
