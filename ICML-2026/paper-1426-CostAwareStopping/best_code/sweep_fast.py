#!/usr/bin/env python3
"""Fast sweep: precompute base regrets once, then adjust PBGI only."""
import sys, builtins
sys.modules["__builtin__"] = builtins
import dill
import numpy as np

pkl_path = "/repo/notebooks/empirical_results/lcbench_known_cost_metrics_per_acq_updated.pkl"
with open(pkl_path, "rb") as f:
    metrics_per_acq = dill.load(f)

datasets = list(metrics_per_acq.keys())
lam = 1e-4

best_acc = {
    "Fashion-MNIST": 90.17316017316017,
    "adult": 83.00552211950115,
    "higgs": 71.86302385956238,
    "volkert": 62.765681026866915,
}

target_acq = "PBGI(1e-4)"
base_srs = ["LogEIPC-med", "SRGap-med", "UCB-LCB", "PRB", "GSS", "Convergence"]

# Precompute non-PBGI regrets (these don't change)
base_regrets = {}
for dataset in datasets:
    best_error = 100.0 - best_acc[dataset]
    data = metrics_per_acq[dataset][target_acq]
    test_errs = data["final test error"]
    costs = data["estimated cumulative cost"]
    best_obs = data["current best observed"]
    n_seeds, n_iters = test_errs.shape

    for sr in base_srs:
        stop_vals = []
        for seed in range(n_seeds):
            stop_idx = n_iters - 1
            for k in range(20, n_iters):
                stop_now = False
                try:
                    if sr == "LogEIPC-med":
                        logeipc = data.get("LogEIPC acq")
                        med = np.nanmedian(logeipc[seed, 1:21])
                        stop_now = float(logeipc[seed, k]) <= np.log(0.01) + med
                    elif sr == "SRGap-med":
                        emr = data.get("exp min regret gap")
                        med = np.nanmedian(emr[seed, 1:21])
                        stop_now = float(emr[seed, k]) <= 0.1 * med
                    elif sr == "UCB-LCB":
                        ru = data.get("regret upper bound")
                        stop_now = float(ru[seed, k]) <= 0.01
                    elif sr == "PRB":
                        prb = data.get("PRB")
                        stop_now = float(prb[seed, k]) >= 0.95
                    elif sr == "GSS":
                        bh = best_obs[seed, :k+1]
                        iqr_v = np.nanpercentile(bh, 75) - np.nanpercentile(bh, 25)
                        if iqr_v == 0:
                            stop_now = True
                        elif k >= 5:
                            stop_now = float(bh[k-5] - bh[k]) / iqr_v <= 0.01
                    elif sr == "Convergence":
                        if k >= 5:
                            stop_now = float(best_obs[seed, k]) == float(best_obs[seed, k-5])
                except:
                    continue
                if stop_now:
                    stop_idx = k
                    break
            regret = float(test_errs[seed, stop_idx] - best_error) + lam * float(costs[seed, stop_idx])
            stop_vals.append(regret)
        base_regrets[(dataset, sr)] = float(np.mean(stop_vals))

def eval_pbgi(scale, use_k, smooth_w):
    "Compute Top-1/2/3 for PBGI(1e-4) stopping with given params."
    sr_label = "PBGI-modified"
    top1 = top2 = top3 = 0
    for dataset in datasets:
        best_error = 100.0 - best_acc[dataset]
        data = metrics_per_acq[dataset][target_acq]
        test_errs = data["final test error"]
        costs = data["estimated cumulative cost"]
        best_obs = data["current best observed"]
        pbgi_acq = data["PBGI(1e-4) acq"]
        n_seeds, n_iters = test_errs.shape

        pbgi_stop_vals = []
        for seed in range(n_seeds):
            stop_idx = n_iters - 1
            for k in range(20, n_iters):
                if smooth_w > 0 and k >= smooth_w:
                    val = np.mean(pbgi_acq[seed, k - smooth_w + 1 : k + 1])
                else:
                    val = float(pbgi_acq[seed, k])
                ref_idx = k if use_k else k - 1
                if ref_idx >= n_iters:
                    ref_idx = n_iters - 1
                if val >= scale * float(best_obs[seed, ref_idx]):
                    stop_idx = k
                    break
            regret = float(test_errs[seed, stop_idx] - best_error) + lam * float(costs[seed, stop_idx])
            pbgi_stop_vals.append(regret)
        pbgi_regret = float(np.mean(pbgi_stop_vals))

        # Rank: compare pbgi_regret vs base_regrets
        all_regrets = [pbgi_regret] + [base_regrets[(dataset, sr)] for sr in base_srs]
        all_srs_local = [sr_label] + base_srs
        ranks = np.array(all_regrets).argsort().argsort()
        rank = int(ranks[0]) + 1
        if rank <= 1: top1 += 1
        if rank <= 2: top2 += 1
        if rank <= 3: top3 += 1

    n = len(datasets)
    return top1/n*100, top2/n*100, top3/n*100

# Original baseline
t1, t2, t3 = eval_pbgi(1.0, use_k=False, smooth_w=0)
print(f"Baseline (scale=1.0, use_best_obs[k-1], w=0): Top-1={t1:.1f}%, Top-2={t2:.1f}%, Top-3={t3:.1f}%")

print("\n=== Scale sweep (use_k=False, w=0) ===")
for scale in [0.80, 0.85, 0.90, 0.92, 0.95, 0.97, 0.98, 0.99, 1.00, 1.02, 1.05]:
    t1, t2, t3 = eval_pbgi(scale, use_k=False, smooth_w=0)
    m = " ***" if t1 > 75.0 else ""
    print(f"  {scale:.2f}: {t1:.1f}% / {t2:.1f}% / {t3:.1f}%{m}")

print("\n=== Scale sweep (use_k=True, w=0) ===")
for scale in [0.80, 0.85, 0.90, 0.92, 0.95, 0.97, 0.98, 0.99, 1.00, 1.02, 1.05]:
    t1, t2, t3 = eval_pbgi(scale, use_k=True, smooth_w=0)
    m = " ***" if t1 > 75.0 else ""
    print(f"  {scale:.2f}: {t1:.1f}% / {t2:.1f}% / {t3:.1f}%{m}")

print("\n=== Smooth window sweep (scale=1.0, use_k=False) ===")
for w in [2, 3, 5, 7, 10]:
    t1, t2, t3 = eval_pbgi(1.0, use_k=False, smooth_w=w)
    m = " ***" if t1 > 75.0 else ""
    print(f"  w={w}: {t1:.1f}% / {t2:.1f}% / {t3:.1f}%{m}")

print("\n=== Best combos ===")
best = None
for scale in [0.85, 0.88, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.00]:
    for use_k in [False, True]:
        for w in [0, 3, 5]:
            t1, t2, t3 = eval_pbgi(scale, use_k=use_k, smooth_w=w)
            if best is None or t1 > best[1] or (t1 == best[1] and t2 >= best[2] and t3 >= best[3]):
                best = (f"scale={scale:.2f},use_k={use_k},w={w}", t1, t2, t3)
            if t1 > 75.0:
                print(f"  scale={scale:.2f}, use_k={use_k}, w={w}: Top-1={t1:.1f}%, Top-2={t2:.1f}%, Top-3={t3:.1f}%")

print(f"\nBest: {best}")
