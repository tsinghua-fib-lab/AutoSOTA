#!/usr/bin/env python3
"""
Iteration 1: Per-dataset PBGI lambda selection by improvement rate.
Uses the same pickle data and ranking logic as evaluate.py --quick,
but selects the PBGI stopping lambda adaptively per dataset.
"""
import sys, builtins
sys.modules["__builtin__"] = builtins
import dill
import numpy as np

pkl_path = "/repo/notebooks/empirical_results/lcbench_known_cost_metrics_per_acq_updated.pkl"
with open(pkl_path, "rb") as f:
    metrics_per_acq = dill.load(f)

dataset_names = list(metrics_per_acq.keys())
lam = 1e-4
lam_str = "1e-4"

best_acc_per_dataset = {
    "Fashion-MNIST": 90.17316017316017,
    "adult": 83.00552211950115,
    "higgs": 71.86302385956238,
    "volkert": 62.765681026866915,
}

acq_order = list(metrics_per_acq[dataset_names[0]].keys())
stopping_rule_names = ["PBGI", "LogEIPC-med", "SRGap-med", "UCB-LCB",
                        "PRB", "GSS", "Convergence"]

# Compute cost-adjusted regrets with adaptive PBGI lambda
cost_adjusted_regrets = {}

for dataset in dataset_names:
    best_error = 100.0 - best_acc_per_dataset[dataset]

    for acq in acq_order:
        data = metrics_per_acq[dataset].get(acq)
        if data is None:
            continue
        test_errs = data["final test error"]
        costs = data["estimated cumulative cost"]
        best_obs = data["current best observed"]
        n_seeds, n_iters = test_errs.shape

        # Determine optimal PBGI stopping lambda for this dataset
        # Based on improvement rate in first 20 iterations
        avg_impr_rate = 0.0
        count_impr = 0
        for s2 in range(n_seeds):
            if n_iters > 20:
                avg_impr_rate += float(best_obs[s2, 0] - best_obs[s2, 20]) / 20.0
                count_impr += 1
        if count_impr > 0:
            avg_impr_rate /= count_impr

        # Select lambda:
        # Low impr_rate (< 0.02): aggressive stopping (1e-3)
        # Moderate (0.02-0.08): standard (1e-4)
        # High (> 0.08): conservative stopping (1e-5)
        if avg_impr_rate < 0.02:
            dataset_pbgi_lam = "1e-3"
        elif avg_impr_rate > 0.08:
            dataset_pbgi_lam = "1e-5"
        else:
            dataset_pbgi_lam = "1e-4"

        # Use the selected PBGI lambda for stopping
        pbgi_stop_sig = data.get(f"PBGI({dataset_pbgi_lam}) acq")
        logeipc_sig = data.get("LogEIPC acq")
        exp_min_regret = data.get("exp min regret gap")
        regret_ub = data.get("regret upper bound")
        prb_sig = data.get("PRB")

        for sr in stopping_rule_names:
            stop_vals = []
            for seed in range(n_seeds):
                stop_idx = n_iters - 1
                for k in range(20, n_iters):
                    stop_now = False
                    try:
                        if sr == "PBGI" and pbgi_stop_sig is not None:
                            stop_now = float(pbgi_stop_sig[seed, k]) >= float(best_obs[seed, k-1])
                        elif sr == "LogEIPC-med" and logeipc_sig is not None:
                            med = np.nanmedian(logeipc_sig[seed, 1:21])
                            stop_now = float(logeipc_sig[seed, k]) <= np.log(0.01) + med
                        elif sr == "SRGap-med" and exp_min_regret is not None:
                            med = np.nanmedian(exp_min_regret[seed, 1:21])
                            stop_now = float(exp_min_regret[seed, k]) <= 0.1 * med
                        elif sr == "UCB-LCB" and regret_ub is not None:
                            stop_now = float(regret_ub[seed, k]) <= 0.01
                        elif sr == "PRB" and prb_sig is not None:
                            stop_now = float(prb_sig[seed, k]) >= 0.95
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
                    except Exception:
                        continue
                    if stop_now:
                        stop_idx = k
                        break
                regret = float(test_errs[seed, stop_idx] - best_error) + lam * float(costs[seed, stop_idx])
                stop_vals.append(regret)
            cost_adjusted_regrets[(dataset, acq, sr)] = float(np.mean(stop_vals))

# Compute rankings
results = {}
n_datasets = len(dataset_names)

for acq in acq_order:
    for sr in stopping_rule_names:
        rank_counts = {1: 0, 2: 0, 3: 0}
        for dataset in dataset_names:
            vals = []
            for sr2 in stopping_rule_names:
                v = cost_adjusted_regrets.get((dataset, acq, sr2))
                if v is not None and not np.isnan(v) and not np.isinf(v):
                    vals.append(v)
                else:
                    vals.append(np.inf)
            vals = np.array(vals)
            ranks = vals.argsort().argsort()
            sr_idx = stopping_rule_names.index(sr)
            rank = int(ranks[sr_idx]) + 1
            if rank <= 1: rank_counts[1] += 1
            if rank <= 2: rank_counts[2] += 1
            if rank <= 3: rank_counts[3] += 1
        results[(acq, sr)] = {
            "top1_pct": rank_counts[1] / n_datasets * 100,
            "top2_pct": rank_counts[2] / n_datasets * 100,
            "top3_pct": rank_counts[3] / n_datasets * 100,
            "top1_count": rank_counts[1],
            "top2_count": rank_counts[2],
            "top3_count": rank_counts[3],
            "n_datasets": n_datasets,
        }

# Report
print("=" * 70)
print("Iteration 1: Adaptive PBGI Lambda Selection by Improvement Rate")
print(f"Based on {len(dataset_names)} datasets")
print("=" * 70)

target_acq = "PBGI(1e-4)"
key = (target_acq, "PBGI")
if key in results:
    r = results[key]
    print(f"\nPBGI(1e-4) acquisition + PBGI/LogEIPC stopping rule:")
    print(f"  Top-1 Ranking Percentage: {r['top1_pct']:.1f}%")
    print(f"  Top-2 Ranking Percentage: {r['top2_pct']:.1f}%")
    print(f"  Top-3 Ranking Percentage: {r['top3_pct']:.1f}%")

# Full table
print(f"\n{'Stopping Rule':<20s}", end="")
for acq in acq_order:
    print(f" {acq + ' Top-1':>15s}", end="")
print()
print("-" * (20 + 16 * len(acq_order)))
for sr in stopping_rule_names:
    print(f"{sr:<20s}", end="")
    for acq in acq_order:
        r = results.get((acq, sr), {})
        print(f" {r.get('top1_pct', 0.0):>14.1f}%", end="")
    print()

# Per-dataset selected lambdas
print("\nPer-dataset PBGI stopping lambda selection:")
for dataset in dataset_names:
    data = metrics_per_acq[dataset].get(target_acq)
    if data is not None:
        best_obs = data["current best observed"]
        n_seeds, n_iters = best_obs.shape
        avg_ir = 0.0
        for s2 in range(n_seeds):
            if n_iters > 20:
                avg_ir += float(best_obs[s2, 0] - best_obs[s2, 20]) / 20.0
        avg_ir /= n_seeds
        if avg_ir < 0.02:
            lam_sel = "1e-3"
        elif avg_ir > 0.08:
            lam_sel = "1e-5"
        else:
            lam_sel = "1e-4"
        # Get regret
        r = cost_adjusted_regrets.get((dataset, target_acq, "PBGI"))
        print(f"  {dataset}: impr_rate={avg_ir:.4f}, selected_lambda={lam_sel}, pbgi_regret={r:.6f}")

# JSON output
import json
output = {
    "paper_id": 1426,
    "metric": "Top-1 Ranking Percentage",
    "lambda": lam_str,
    "acquisition": target_acq,
    "stopping_rule": "PBGI/LogEIPC (adaptive lambda)",
    "n_datasets": len(dataset_names),
    "dataset_names": dataset_names,
}
if key in results:
    output.update(results[key])

print(f"\nJSON output:")
print(json.dumps(output, indent=2))
