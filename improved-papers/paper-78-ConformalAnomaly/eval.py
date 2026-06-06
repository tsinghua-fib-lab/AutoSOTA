"""
Evaluation script for paper-3728: Conformal Anomaly Detection in Event Sequences (CADES)

Runs AUROC (real_world) and TPR@alpha=0.05 (fpr_control) experiments on STEAD dataset.
All dependencies and data are pre-installed in the Docker image.

Output format (machine-parseable):
    auroc_anchorage_ak: XX.XX
    auroc_aleutian_islands_ak: XX.XX
    auroc_helmet_ca: XX.XX
    tpr_at_alpha_005_anchorage_ak: XX.XX
"""
import os
import sys
import numpy as np
import pandas as pd
import random
import torch
import anomaly_tpp

from tqdm.auto import trange
from statsmodels.distributions.empirical_distribution import ECDF

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.insert(0, "/repo")

# ── Config ────────────────────────────────────────────────────────────────────
# Bandwidth parameters per OOD region (from paper / reproduction)
STEAD_auroc_params = {
    'Anchorage, AK':        {'h_int': 0.1,  'h_arr': 0.001},
    'Aleutian Islands, AK': {'h_int': 0.01, 'h_arr': 0.005},
    'Helmet, CA':           {'h_int': 0.1,  'h_arr': 0.001},
}
# fpr_control.py uses Anchorage AK with these params
FPR_REGION  = 'Anchorage, AK'
FPR_H_INT   = 0.1
FPR_H_ARR   = 0.001

NUM_SEEDS   = 5
BATCH_SIZE  = 64
MAX_EPOCHS  = 500
PATIENCE    = 5
ALPHAS      = np.round(np.linspace(0.05, 0.5, 10), 2)


# ── Helpers ───────────────────────────────────────────────────────────────────
def build_ecdfs(test_stats, ntpp, batch, pt, h_int, h_arr):
    ecdfs = {}
    for stat in test_stats:
        scores = stat(poisson_times_per_mark=pt, model=ntpp,
                      batch=batch, h_int=h_int, h_arr=h_arr)
        ecdfs[stat.__name__] = ECDF(scores)
    return ecdfs


def twosided_pval(ecdfs, stat_name, scores):
    ecdf = ecdfs[stat_name](scores)
    n    = len(ecdfs[stat_name].x) - 1
    pr   = ((1 - ecdf) * n + 1) / (n + 1)
    pl   = (ecdf * n + 1) / (n + 1)
    return np.minimum(np.minimum(2 * pr, 2 * pl), 1)


def combined_pvals(ecdfs, test_stats, ntpp, batch, pt, h_int, h_arr):
    """Fisher-min combination of KL-int and KL-arr conformal p-values."""
    int_p = arr_p = None
    for stat in test_stats:
        sn  = stat.__name__
        sc  = stat(poisson_times_per_mark=pt, model=ntpp,
                   batch=batch, h_int=h_int, h_arr=h_arr)
        pv  = twosided_pval(ecdfs, sn, sc)
        if sn == 'kl_int':
            int_p = pv
        elif sn == 'kl_arr':
            arr_p = pv
    return np.minimum(np.minimum(2 * int_p, 2 * arr_p), 1)


# ── Main ──────────────────────────────────────────────────────────────────────
auroc_results = []
tpr_results   = []
test_stats    = [anomaly_tpp.statistics.kl_int, anomaly_tpp.statistics.kl_arr]

for seed in trange(NUM_SEEDS, desc="Seeds"):
    print(f"\n--- Seed {seed} ---", flush=True)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    scenario = anomaly_tpp.scenarios.real_world.STEAD()
    id_train  = scenario.id_train
    id_proper_train, id_cal, _ = id_train.train_val_test_split(
        train_size=0.5, val_size=0.5, test_size=0.0, seed=seed)
    id_test = scenario.id_test

    dl_train = id_proper_train.get_dataloader(batch_size=BATCH_SIZE, shuffle=True)
    ntpp = anomaly_tpp.utils.fit_ntpp_model(
        dl_train, num_marks=id_proper_train.num_marks,
        max_epochs=MAX_EPOCHS, patience=PATIENCE)

    id_test_batch = anomaly_tpp.data.Batch.from_list(id_test)
    id_test_pt    = anomaly_tpp.utils.extract_poisson_arrival_times(ntpp, id_test_batch)
    id_cal_batch  = anomaly_tpp.data.Batch.from_list(id_cal)
    id_cal_pt     = anomaly_tpp.utils.extract_poisson_arrival_times(ntpp, id_cal_batch)

    # ── AUROC for all 3 STEAD OOD regions ────────────────────────────────
    for ood_name, params in STEAD_auroc_params.items():
        h_int, h_arr = params['h_int'], params['h_arr']

        ecdfs    = build_ecdfs(test_stats, ntpp, id_cal_batch, id_cal_pt, h_int, h_arr)
        id_pvs   = combined_pvals(ecdfs, test_stats, ntpp,
                                  id_test_batch, id_test_pt, h_int, h_arr)

        ood      = scenario.ood_test_datasets[ood_name]
        ood_b    = anomaly_tpp.data.Batch.from_list(ood)
        ood_pt   = anomaly_tpp.utils.extract_poisson_arrival_times(ntpp, ood_b)
        ood_pvs  = combined_pvals(ecdfs, test_stats, ntpp, ood_b, ood_pt, h_int, h_arr)

        auc = anomaly_tpp.utils.roc_auc_from_pvals(id_pvs, ood_pvs)
        auroc_results.append({'scenario': ood_name, 'seed': seed, 'auc': auc})
        print(f"  AUROC {ood_name}: {auc*100:.2f}%", flush=True)

    # ── TPR@alpha for Anchorage AK (mirrors fpr_control.py logic) ────────
    ecdfs_f  = build_ecdfs(test_stats, ntpp, id_cal_batch, id_cal_pt,
                           FPR_H_INT, FPR_H_ARR)
    id_pvs_f = combined_pvals(ecdfs_f, test_stats, ntpp,
                              id_test_batch, id_test_pt, FPR_H_INT, FPR_H_ARR)

    ood_fpr  = scenario.ood_test_datasets[FPR_REGION]
    ood_b_f  = anomaly_tpp.data.Batch.from_list(ood_fpr)
    ood_pt_f = anomaly_tpp.utils.extract_poisson_arrival_times(ntpp, ood_b_f)
    ood_pvs_f = combined_pvals(ecdfs_f, test_stats, ntpp,
                               ood_b_f, ood_pt_f, FPR_H_INT, FPR_H_ARR)

    for alpha in ALPHAS:
        tpr, fpr = anomaly_tpp.utils.tpr_fpr_from_pvals(id_pvs_f, ood_pvs_f, alpha=alpha)
        tpr_results.append({'alpha': alpha, 'tpr': tpr, 'fpr': fpr, 'seed': seed})


# ── Aggregate & print ─────────────────────────────────────────────────────────
auroc_df  = pd.DataFrame(auroc_results)
tpr_df    = pd.DataFrame(tpr_results)
auroc_pct = auroc_df.groupby('scenario')['auc'].mean() * 100
tpr_val   = tpr_df[np.isclose(tpr_df['alpha'], 0.05)]['tpr'].mean() * 100

print("\n=== RESULTS ===")
print(f"auroc_anchorage_ak: {auroc_pct.get('Anchorage, AK', float('nan')):.2f}")
print(f"auroc_aleutian_islands_ak: {auroc_pct.get('Aleutian Islands, AK', float('nan')):.2f}")
print(f"auroc_helmet_ca: {auroc_pct.get('Helmet, CA', float('nan')):.2f}")
print(f"tpr_at_alpha_005_anchorage_ak: {tpr_val:.2f}")

print("\n=== AUROC Table (%) ===")
print(auroc_pct.round(2).to_string())
print(f"\n=== TPR at alpha=0.05 — {FPR_REGION}: {tpr_val:.2f}% ===")
