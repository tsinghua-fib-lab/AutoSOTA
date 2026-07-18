#!/usr/bin/env python3
"""Sweep PBGI stopping rule thresholds to find optimal configuration."""
import sys, builtins
sys.modules["__builtin__"] = builtins
import dill
import numpy as np

pkl_path = "/repo/notebooks/empirical_results/lcbench_known_cost_metrics_per_acq_updated.pkl"
with open(pkl_path, "rb") as f:
    metrics_per_acq = dill.load(f)

datasets = list(metrics_per_acq.keys())
lam = 1e-4

best_acc_per_dataset = {
    "Fashion-MNIST": 90.17316017316017,
    "adult": 83.00552211950115,
    "higgs": 71.86302385956238,
    "volkert": 62.765681026866915,
}

acq_order = ["LogEIPC", "PBGI(1e-3)", "PBGI(1e-4)", "PBGI(1e-5)", "LCB", "TS"]


def eval_pbgi_threshold(threshold_scale, use_best_obs_k=True, smooth_window=0):
    cost_adjusted_regrets = {}
    for dataset in datasets:
        best_error = 100.0 - best_acc_per_dataset[dataset]
        for acq in acq_order:
            data = metrics_per_acq[dataset].get(acq)
            if data is None:
                continue
            test_errs = data["final test error"]
            costs = data["estimated cumulative cost"]
            best_obs = data["current best observed"]
            n_seeds, n_iters = test_errs.shape

            # PBGI stopping with thresholds
            for sr_lambda in ["1e-3", "1e-4", "1e-5"]:
                pbgi_acq_sig = data.get(f"PBGI({sr_lambda}) acq")
                if pbgi_acq_sig is None:
                    continue
                stop_vals_pbgi = []
                for seed in range(n_seeds):
                    stop_idx = n_iters - 1
                    for k in range(20, n_iters):
                        if smooth_window > 0 and k >= smooth_window:
                            pbgi_val = np.mean(pbgi_acq_sig[seed, k - smooth_window + 1 : k + 1])
                        else:
                            pbgi_val = float(pbgi_acq_sig[seed, k])
                        ref_idx = k if use_best_obs_k else k - 1
                        if ref_idx >= n_iters:
                            ref_idx = n_iters - 1
                        if pbgi_val >= threshold_scale * float(best_obs[seed, ref_idx]):
                            stop_idx = k
                            break
                    regret = float(test_errs[seed, stop_idx] - best_error) + lam * float(costs[seed, stop_idx])
                    stop_vals_pbgi.append(regret)
                cost_adjusted_regrets[(dataset, acq, f"PBGI({sr_lambda})")] = float(np.mean(stop_vals_pbgi))

            # Other stopping rules (unchanged)
            logeipc_sig = data.get("LogEIPC acq")
            exp_min_regret = data.get("exp min regret gap")
            regret_ub = data.get("regret upper bound")
            prb_sig = data.get("PRB")

            for sr in ["LogEIPC-med", "SRGap-med", "UCB-LCB", "PRB", "GSS", "Convergence"]:
                if sr == "LogEIPC-med" and logeipc_sig is None:
                    continue
                if sr == "SRGap-med" and exp_min_regret is None:
                    continue
                if sr == "UCB-LCB" and regret_ub is None:
                    continue
                stop_vals = []
                for seed in range(n_seeds):
                    stop_idx = n_iters - 1
                    for k in range(20, n_iters):
                        stop_now = False
                        try:
                            if sr == "LogEIPC-med":
                                med = np.nanmedian(logeipc_sig[seed, 1:21])
                                stop_now = float(logeipc_sig[seed, k]) <= np.log(0.01) + med
                            elif sr == "SRGap-med":
                                med = np.nanmedian(exp_min_regret[seed, 1:21])
                                stop_now = float(exp_min_regret[seed, k]) <= 0.1 * med
                            elif sr == "UCB-LCB":
                                stop_now = float(regret_ub[seed, k]) <= 0.01
                            elif sr == "PRB" and prb_sig is not None:
                                stop_now = float(prb_sig[seed, k]) >= 0.95
                            elif sr == "GSS":
                                bh = best_obs[seed, : k + 1]
                                iqr_v = np.nanpercentile(bh, 75) - np.nanpercentile(bh, 25)
                                if iqr_v == 0:
                                    stop_now = True
                                elif k >= 5:
                                    stop_now = float(bh[k - 5] - bh[k]) / iqr_v <= 0.01
                            elif sr == "Convergence":
                                if k >= 5:
                                    stop_now = float(best_obs[seed, k]) == float(best_obs[seed, k - 5])
                        except Exception:
                            continue
                        if stop_now:
                            stop_idx = k
                            break
                    regret = float(test_errs[seed, stop_idx] - best_error) + lam * float(costs[seed, stop_idx])
                    stop_vals.append(regret)
                cost_adjusted_regrets[(dataset, acq, sr)] = float(np.mean(stop_vals))

    # Compute rankings
    n_datasets = len(datasets)
    target_acq = "PBGI(1e-4)"
    target_sr = "PBGI(1e-4)"
    all_srs = [f"PBGI(1e-3)", f"PBGI(1e-4)", f"PBGI(1e-5)",
               "LogEIPC-med", "SRGap-med", "UCB-LCB", "PRB", "GSS", "Convergence"]

    top1_count = 0
    top2_count = 0
    top3_count = 0

    for dataset in datasets:
        vals = []
        for sr in all_srs:
            v = cost_adjusted_regrets.get((dataset, target_acq, sr), np.inf)
            vals.append(v)
        vals = np.array(vals)
        ranks = vals.argsort().argsort()
        target_idx = all_srs.index(target_sr)
        rank = int(ranks[target_idx]) + 1
        if rank <= 1:
            top1_count += 1
        if rank <= 2:
            top2_count += 1
        if rank <= 3:
            top3_count += 1

    return top1_count / n_datasets * 100, top2_count / n_datasets * 100, top3_count / n_datasets * 100


if __name__ == "__main__":
    print("=== PBGI Threshold Sweep (use best_obs[k]) ===")
    print(f"{'scale':>8s} {'Top-1%':>8s} {'Top-2%':>8s} {'Top-3%':>8s}")
    for scale in [0.80, 0.85, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.00, 1.02, 1.05]:
        t1, t2, t3 = eval_pbgi_threshold(scale, use_best_obs_k=True)
        marker = " <-- IMPROVED" if t1 > 75.0 else ""
        print(f"{scale:>8.2f} {t1:>7.1f}% {t2:>7.1f}% {t3:>7.1f}%{marker}")

    print()
    print("=== PBGI Threshold Sweep (use best_obs[k-1], original) ===")
    for scale in [0.80, 0.85, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.00, 1.02, 1.05]:
        t1, t2, t3 = eval_pbgi_threshold(scale, use_best_obs_k=False)
        marker = " <-- IMPROVED" if t1 > 75.0 else ""
        print(f"{scale:>8.2f} {t1:>7.1f}% {t2:>7.1f}% {t3:>7.1f}%{marker}")

    print()
    print("=== PBGI with smoothing window (scale=1.0, best_obs[k-1]) ===")
    for w in [2, 3, 5, 7, 10]:
        t1, t2, t3 = eval_pbgi_threshold(1.0, use_best_obs_k=False, smooth_window=w)
        marker = " <-- IMPROVED" if t1 > 75.0 else ""
        print(f"window={w:>2d} {t1:>7.1f}% {t2:>7.1f}% {t3:>7.1f}%{marker}")

    print()
    print("=== Combined: scale + smoothing + best_obs[k] ===")
    best = (0, 0, 0, 0)
    for scale in [0.85, 0.90, 0.92, 0.95, 0.97, 0.98, 0.99, 1.00]:
        for w in [0, 3, 5]:
            for use_k in [True, False]:
                t1, t2, t3 = eval_pbgi_threshold(scale, use_best_obs_k=use_k, smooth_window=w)
                if t1 > best[1] or (t1 == best[1] and t2 >= best[2] and t3 >= best[3]):
                    best = (f"scale={scale:.2f},w={w},use_k={use_k}", t1, t2, t3)
                if t1 > 75.0:
                    print(f"  scale={scale:.2f}, w={w}, use_k={use_k}: Top-1={t1:.1f}%, Top-2={t2:.1f}%, Top-3={t3:.1f}%")

    print(f"\nBest: {best[0]} -> Top-1={best[1]:.1f}%, Top-2={best[2]:.1f}%, Top-3={best[3]:.1f}%")
