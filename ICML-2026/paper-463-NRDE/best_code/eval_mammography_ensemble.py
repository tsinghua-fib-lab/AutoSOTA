#!/usr/bin/env python3
"""Ensemble evaluation: average per-sample anomaly scores across 5 seeds
before computing AUROC/AUPRC. Also reports per-seed metrics for comparison.
"""
import os, sys, numpy as np, torch, json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from NRDE import NRDE_run, NRDE_run_ensemble, read_data


def main():
    data_path = "/datasets/23_mammography.npz"
    lr = 0.005
    grad_pun = 1.0
    n_epochs = 100
    bs = 512
    mid_dim = 2048
    act = 2
    PNAL = "L_1sq"
    n_runs = 5
    seeds = [42, 52, 62, 72, 82]

    per_seed_results = []
    all_preds = []
    all_targets = None

    # FIX: Use a fixed data split seed (42) for all models so we average
    # per-sample scores across the SAME test samples.
    # Model seeds differ for independent training.
    data_seed = 42
    np.random.seed(data_seed)
    torch.manual_seed(data_seed)
    torch.cuda.manual_seed_all(data_seed)
    train_data, train_labels, test_data, test_labels, _, _ = read_data(
        data_path, normalization="z-score", seed=data_seed
    )
    print(f"Fixed data split (seed={data_seed}): train={len(train_data)}, test={len(test_data)}")

    for model_seed in seeds:
        # Set model seed for reproducible initialization
        np.random.seed(model_seed)
        torch.manual_seed(model_seed)
        torch.cuda.manual_seed_all(model_seed)

        auroc, auprc, preds, targets = NRDE_run_ensemble(
            train_data, train_labels, test_data, test_labels,
            lr=lr, grad_pun=grad_pun, n_epochs=n_epochs, bs=bs, mid_dim=mid_dim,
            act=act, adam=True, PNAL=PNAL, verbose=False
        )

        per_seed_results.append({"seed": model_seed, "auroc": float(auroc), "auprc": float(auprc)})
        all_preds.append(preds.detach().cpu().numpy() if torch.is_tensor(preds) else preds)
        if all_targets is None:
            all_targets = targets.detach().cpu().numpy() if torch.is_tensor(targets) else targets

    # Compute ensemble: average per-sample scores across seeds
    ensemble_preds = np.mean(np.stack(all_preds, axis=0), axis=0)

    # Compute ensemble AUROC/AUPRC
    from torchmetrics import AUROC, AveragePrecision
    ensemble_preds_t = torch.tensor(ensemble_preds)
    ensemble_targets_t = torch.tensor(all_targets)
    auroc_ens = AUROC(task="binary")(ensemble_preds_t, ensemble_targets_t)
    auprc_ens = AveragePrecision(task="binary")(ensemble_preds_t, ensemble_targets_t)

    # Per-seed stats (using best-epoch per-seed results)
    auroc_mean = np.mean([r["auroc"] for r in per_seed_results])
    auroc_std = np.std([r["auroc"] for r in per_seed_results])
    auprc_mean = np.mean([r["auprc"] for r in per_seed_results])
    auprc_std = np.std([r["auprc"] for r in per_seed_results])

    print(json.dumps({
        "ensemble_auroc": round(float(auroc_ens), 4),
        "ensemble_auprc": round(float(auprc_ens), 4),
        "per_seed_auroc_mean": round(auroc_mean, 4),
        "per_seed_auroc_std": round(auroc_std, 4),
        "per_seed_auprc_mean": round(auprc_mean, 4),
        "per_seed_auprc_std": round(auprc_std, 4),
        "individual_runs": per_seed_results,
    }, indent=2))


if __name__ == "__main__":
    main()
