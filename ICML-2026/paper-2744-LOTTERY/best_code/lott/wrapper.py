"""
Experiment wrapper for running LoTT on various datasets.

Handles data loading, optional embedding extraction, data splitting
(train/calibration/holdout), and repeated testing.

All parameters are exposed for systematic optimization via CLI (CODE-01).
"""

import numpy as np
import torch
import time
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from exp.dataloader import load_data, check_device
from lott.lott import LoTT, LoTTWithSelection


def run_lott(dataset, N_ref, N_query, rs, check, n_test=100, alpha=0.05,
             is_selection=True, model_arch=None, model=None, verbose=False,
             # --- CODE-01: exposed parameters with current defaults ---
             selection_method="precision_weight",
             n_select=2,
             variance_threshold=None,
             perturbation_scale=0.01,
             M=10,
             subset_size=10,
             n_permutations=500,
             train_frac=0.4,
             calib_frac=0.1,
             k_knn=5,
             k_lof=20,
             # --- ALGO-06 ---
             statistic_formulation="mean_of_squares",
             # --- ALGO-01 ---
             use_multiscale_me=False,
             multiscale_bw_scales=None):
    """
    Run LoTT two-sample test on the given dataset.

    Args:
        dataset: Dataset name ("blob", "cifar10", "higgs")
        N_ref: Number of reference samples (N in the paper)
        N_query: Number of query samples (M in the paper)
        rs: Random seed
        check: 1 = test power (P != Q), 0 = type I error (P = Q)
        n_test: Number of independent test repetitions
        alpha: Significance level
        is_selection: If True, use LoTTWithSelection; else use LoTT
        model_arch: Model architecture name for embedding extraction
        model: Pre-loaded model (optional, avoids reloading)
        verbose: Print selection details
        selection_method: RDR selection method
        n_select: Number of RDRs for "top_n"
        variance_threshold: Threshold for "threshold" method
        perturbation_scale: Scale for "sensitivity_weight" sensitivity computation
        M: Number of landmark RDRs
        subset_size: Subset size per landmark RDR
        n_permutations: Number of permutations for null distribution
        train_frac: Fraction of X for RDR training
        calib_frac: Fraction of X for calibration
        k_knn: k for KNN_RDR
        k_lof: k for LOF_RDR
        statistic_formulation: "mean_of_squares", "square_of_mean", or "hybrid"
        use_multiscale_me: Use MultiScaleME_RDR instead of ME_RDR (ALGO-01)
        multiscale_bw_scales: List of bandwidth scale factors (ALGO-01)

    Returns:
        H: Array of rejection decisions (0 or 1) over n_test trials
    """
    device = check_device()
    np.random.seed(rs)
    torch.manual_seed(rs)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if model is not None:
        model.to(device)
        model.eval()

    H = np.zeros(n_test)
    test_time = 0

    for k in range(n_test):
        start_time = time.time()

        X, Y_test, _ = load_data(dataset, N_ref, N_query, rs * 1000 + k, check,
                                  need_labels=True, model_arch=model_arch or "Res18")
        X = X.to(device, dtype=torch.float32)
        Y_test = Y_test.to(device, dtype=torch.float32)

        if model is not None:
            with torch.no_grad():
                X = model(X)
                Y_test = model(Y_test)

        n = len(X)
        perm = torch.randperm(n, device=device)
        n_train = int(n * train_frac)
        n_calib = int(n * calib_frac)
        X_train = X[perm[:n_train]]
        X_calib = X[perm[n_train:n_train + n_calib]]
        X_hold = X[perm[n_train + n_calib:]]

        if is_selection:
            lott = LoTTWithSelection(
                alpha=alpha, n_permutations=n_permutations,
                selection_method=selection_method,
                n_select=n_select,
                variance_threshold=variance_threshold,
                perturbation_scale=perturbation_scale,
                statistic_formulation=statistic_formulation,
                verbose=verbose
            )
        else:
            lott = LoTT(alpha=alpha, n_permutations=n_permutations)

        lott.fit(X_train, X_calib, X_hold,
                 M=M, subset_size=subset_size,
                 k_knn=k_knn, k_lof=k_lof,
                 use_multiscale_me=use_multiscale_me,
                 multiscale_bw_scales=multiscale_bw_scales)

        results = lott.test(Y_test)
        H[k] = int(results["reject"])

        test_time += time.time() - start_time

    torch.cuda.empty_cache()
    return H
