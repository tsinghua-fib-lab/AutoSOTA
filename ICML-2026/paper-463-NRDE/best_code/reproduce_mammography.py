#!/usr/bin/env python3
"""Reproduction script for Paper 463 NRDE on mammography dataset.

Paper settings:
- Dataset: mammography (from ADBench)
- Model: NRDE (RealNVP, 2 coupling layers, width=2048)
- Optimizer: Adam (amsgrad=True)
- Batch size: 512
- Epochs: 100
- n_runs: 5
- Grid search: lr ∈ {0.001, 0.005, 0.01}, λ (grad_pun) ∈ {0.01, 0.1, 1}
- Normalization: z-score
- 50% normal-split train/test protocol
"""

import sys
import os
import json
import time
import numpy as np
import torch
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from NRDE import NRDE_run, read_data, set_seed


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1")

    data_path = "/datasets/23_mammography.npz"
    n_epochs = 100
    bs = 512
    mid_dim = 2048
    act = 2  # LeakyReLU
    PNAL = "L_1sq"
    n_runs = 5

    # Grid search: lr and lambda (grad_pun)
    lr_list = [0.001, 0.005, 0.01]
    grad_pun_list = [0.01, 0.1, 1.0]

    # Phase 1: Grid search with 1 seed to find best hyperparams
    print("=" * 80)
    print("PHASE 1: Grid Search (1 seed per combination)")
    print("=" * 80)

    best_auroc = -1
    best_params = None
    grid_results = []

    grid_seed = 42

    for lr in lr_list:
        for grad_pun in grad_pun_list:
            print(f"\n--- lr={lr}, lambda={grad_pun} ---")
            t0 = time.time()

            np.random.seed(grid_seed)
            torch.manual_seed(grid_seed)
            torch.cuda.manual_seed_all(grid_seed)

            train_data, train_labels, test_data, test_labels, _, _ = read_data(
                data_path, normalization="z-score", seed=grid_seed
            )

            auroc, auprc = NRDE_run(
                train_data, train_labels, test_data, test_labels,
                lr=lr, grad_pun=grad_pun, n_epochs=n_epochs, bs=bs, mid_dim=mid_dim,
                act=act, adam=True, PNAL=PNAL, verbose=False
            )

            elapsed = time.time() - t0
            grid_results.append({
                "lr": lr,
                "grad_pun": grad_pun,
                "seed": grid_seed,
                "auroc": float(auroc),
                "auprc": float(auprc),
                "time_s": elapsed,
            })
            print(f"  AUROC={auroc:.4f}, AUPRC={auprc:.4f} ({elapsed:.1f}s)")

            if auroc > best_auroc:
                best_auroc = auroc
                best_params = (lr, grad_pun)

    print(f"\nBest grid params: lr={best_params[0]}, lambda={best_params[1]}, AUROC={best_auroc:.4f}")

    # Phase 2: Run 5 seeds with best params
    print("\n" + "=" * 80)
    print(f"PHASE 2: 5-seed run with lr={best_params[0]}, lambda={best_params[1]}")
    print("=" * 80)

    best_lr, best_grad_pun = best_params
    run_results = []
    all_aurocs = []
    all_auprcs = []

    for run_idx in range(n_runs):
        seed = 42 + run_idx * 10
        print(f"\n--- Run {run_idx+1}/{n_runs} (seed={seed}) ---")
        t0 = time.time()

        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        train_data, train_labels, test_data, test_labels, _, _ = read_data(
            data_path, normalization="z-score", seed=seed
        )

        auroc, auprc = NRDE_run(
            train_data, train_labels, test_data, test_labels,
            lr=best_lr, grad_pun=best_grad_pun, n_epochs=n_epochs, bs=bs, mid_dim=mid_dim,
            act=act, adam=True, PNAL=PNAL, verbose=True
        )

        elapsed = time.time() - t0
        all_aurocs.append(float(auroc))
        all_auprcs.append(float(auprc))
        run_results.append({
            "run": run_idx + 1,
            "seed": seed,
            "lr": best_lr,
            "grad_pun": best_grad_pun,
            "auroc": float(auroc),
            "auprc": float(auprc),
            "time_s": elapsed,
        })
        print(f"  AUROC={auroc:.4f}, AUPRC={auprc:.4f} ({elapsed:.1f}s)")

    # Summary statistics
    auroc_mean = np.mean(all_aurocs)
    auroc_std = np.std(all_aurocs)
    auprc_mean = np.mean(all_auprcs)
    auprc_std = np.std(all_auprcs)

    print("\n" + "=" * 80)
    print("REPRODUCTION RESULTS")
    print("=" * 80)
    print(f"AUROC: {auroc_mean:.4f} ± {auroc_std:.4f}")
    print(f"AUPRC: {auprc_mean:.4f} ± {auprc_std:.4f}")
    print(f"Paper AUROC: 91.7 ± 0.1")
    print(f"Paper AUPRC: 49.6 ± 6.8")
    print(f"Reference AUROC bounds: [91.6, 91.8]")
    print(f"Reference AUPRC bounds: [42.8, 56.4]")

    # Check against rubric bounds
    auroc_in_bounds = 91.6 <= auroc_mean <= 91.8
    auprc_in_bounds = 42.8 <= auprc_mean <= 56.4

    print(f"\nAUROC within CI bounds: {auroc_in_bounds}")
    print(f"AUPRC within CI bounds: {auprc_in_bounds}")

    # Save results
    output = {
        "paper_id": 463,
        "dataset": "mammography",
        "model": "NRDE (RealNVP)",
        "settings": {
            "n_epochs": n_epochs,
            "batch_size": bs,
            "width": mid_dim,
            "act": act,
            "PNAL": PNAL,
            "n_runs": n_runs,
            "best_lr": best_lr,
            "best_grad_pun": best_grad_pun,
        },
        "grid_search_results": grid_results,
        "run_results": run_results,
        "summary": {
            "auroc_mean": float(auroc_mean),
            "auroc_std": float(auroc_std),
            "auprc_mean": float(auprc_mean),
            "auprc_std": float(auprc_std),
        },
        "paper_reference": {
            "auroc": 91.7,
            "auroc_std": 0.1,
            "auprc": 49.6,
            "auprc_std": 6.8,
        },
        "timestamp": datetime.now().isoformat(),
    }

    with open("/repo/reproduction_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to /repo/reproduction_results.json")

    return output


if __name__ == "__main__":
    main()
