#!/usr/bin/env python3
"""Diagnostic run: single seed with per-epoch logging to assess convergence.
Outputs per-epoch train loss, sldj, Jacobian penalty, test AUROC/AUPRC.
"""
import os, sys, numpy as np, torch, json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from NRDE import NRDE_run_verbose, read_data


def main():
    data_path = "/datasets/23_mammography.npz"
    lr = 0.005
    grad_pun = 1.0
    n_epochs = 150  # Extended to 150 to see if we benefit from more epochs
    bs = 512
    mid_dim = 2048
    act = 2
    PNAL = "L_1sq"
    seed = 42

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    train_data, train_labels, test_data, test_labels, _, _ = read_data(
        data_path, normalization="z-score", seed=seed
    )

    results = NRDE_run_verbose(
        train_data, train_labels, test_data, test_labels,
        lr=lr, grad_pun=grad_pun, n_epochs=n_epochs, bs=bs, mid_dim=mid_dim,
        act=act, adam=True, PNAL=PNAL, verbose=True
    )

    # results: (best_auc, best_auprc, epoch_log)
    best_auc, best_auprc, epoch_log = results
    print(json.dumps({
        "best_auroc": round(best_auc, 4),
        "best_auprc": round(best_auprc, 4),
        "n_epochs_run": n_epochs,
        "last_10_epochs_auroc": [round(e["auroc"], 4) for e in epoch_log[-10:]],
        "last_10_epochs_loss": [round(e["loss"], 4) for e in epoch_log[-10:]],
    }, indent=2))


if __name__ == "__main__":
    main()
