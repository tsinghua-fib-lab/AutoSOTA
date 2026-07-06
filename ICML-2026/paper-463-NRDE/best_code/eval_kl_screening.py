#!/usr/bin/env python3
"""KL regularization screening: single-seed test of kl_weight values."""
import os, sys, numpy as np, torch, json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from NRDE import NRDE_run_verbose, read_data


def main():
    data_path = "/datasets/23_mammography.npz"
    lr = 0.005
    grad_pun = 1.0
    n_epochs = 100
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

    results = {}
    for kl_weight in [0.0, 0.001, 0.01, 0.1]:
        torch.manual_seed(seed)  # Reset seed for fair comparison
        torch.cuda.manual_seed_all(seed)
        best_auc, best_auprc, epoch_log = NRDE_run_verbose(
            train_data, train_labels, test_data, test_labels,
            lr=lr, grad_pun=grad_pun, n_epochs=n_epochs, bs=bs, mid_dim=mid_dim,
            act=act, adam=True, PNAL=PNAL, kl_weight=kl_weight, verbose=False
        )
        results[f"kl_{kl_weight}"] = {
            "best_auroc": round(float(best_auc), 4),
            "best_auprc": round(float(best_auprc), 4),
            "best_epoch": max(range(len(epoch_log)), key=lambda i: epoch_log[i]["auroc"]),
            "last_5_losses": [round(e["loss"], 4) for e in epoch_log[-5:]],
        }
        print(f"kl_weight={kl_weight}: AUROC={float(best_auc):.4f}, AUPRC={float(best_auprc):.4f}")

    print("\n" + json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
