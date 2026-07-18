#!/usr/bin/env python3
"""Compute ranking percentages from pre-computed LCBench experiment data."""

import sys
import builtins
sys.modules["__builtin__"] = builtins
import dill
import numpy as np


def compute_cost_adjusted_regrets(metrics_per_acq, lambdas, lam_strs, dataset_names,
                                   acq_order, best_acc_per_dataset, init=20):
    """
    Compute cost-adjusted regret for all (lam, dataset, acq, stop_rule) combinations.
    """
    cost_adjusted_regrets = {}
    stopping_rule_names = []

    for lam_idx, (lam, lam_str) in enumerate(zip(lambdas, lam_strs)):
        for dataset in dataset_names:
            best_acc = best_acc_per_dataset[dataset]
            best_error = 100.0 - best_acc

            # Collect Immediate (stop at iter 0) values across all acquisitions
            immediate_vals_all = []
            for acq in acq_order:
                data = metrics_per_acq[dataset][acq]
                test_errs = data["final test error"]
                costs = data["estimated cumulative cost"]
                n_seeds = test_errs.shape[0]
                for seed in range(n_seeds):
                    immediate_vals_all.append(
                        float(test_errs[seed, 0] - best_error) + lam * float(costs[seed, 0])
                    )

            # Define stopping rules
            rules = []

            # 1. PBGI (LogEIPC equivalent)
            rules.append(("PBGI", lam_str))

            # 2. LogEIPC-med
            rules.append(("LogEIPC-med", None))

            # 3. SRGap-med
            rules.append(("SRGap-med", None))

            # 4. UCB-LCB
            rules.append(("UCB-LCB", None))

            # 5. PRB
            rules.append(("PRB", None))

            # 6. GSS
            rules.append(("GSS", None))

            # 7. Convergence
            rules.append(("Convergence", None))

            if lam_idx == 0:
                stopping_rule_names = [r[0] for r in rules] + ["Immediate", "Hindsight"]

            # Compute for each (acq, stopping_rule)
            for acq in acq_order:
                data = metrics_per_acq[dataset][acq]
                test_errs = data["final test error"]  # (50, 201)
                costs = data["estimated cumulative cost"]  # (50, 201)
                best_obs = data["current best observed"]  # (50, 201)
                pbgi_acq_sig = data.get(f"PBGI({lam_str}) acq", None)
                logeipc_acq_sig = data.get("LogEIPC acq", None)
                exp_min_regret = data.get("exp min regret gap", None)
                regret_ub = data.get("regret upper bound", None)
                prb_sig = data.get("PRB", None)
                n_seeds, n_iters = test_errs.shape

                for rule_name, rule_param in rules:
                    stop_vals = []
                    for seed in range(n_seeds):
                        stop_idx = n_iters - 1  # default: last iter
                        for k in range(init, n_iters):
                            stop_now = False
                            try:
                                if rule_name == "PBGI":
                                    stop_now = (
                                        float(pbgi_acq_sig[seed, k]) >= float(best_obs[seed, k - 1])
                                    )
                                elif rule_name == "LogEIPC-med":
                                    median_ref = np.nanmedian(logeipc_acq_sig[seed, 1:21])
                                    stop_now = (
                                        float(logeipc_acq_sig[seed, k]) <= np.log(0.01) + median_ref
                                    )
                                elif rule_name == "SRGap-med":
                                    median_ref = np.nanmedian(exp_min_regret[seed, 1:21])
                                    stop_now = (
                                        float(exp_min_regret[seed, k]) <= 0.1 * median_ref
                                    )
                                elif rule_name == "UCB-LCB":
                                    stop_now = float(regret_ub[seed, k]) <= 0.01
                                elif rule_name == "PRB":
                                    stop_now = float(prb_sig[seed, k]) >= 0.95
                                elif rule_name == "GSS":
                                    best_hist = best_obs[seed, :k + 1]
                                    iqr_val = (np.nanpercentile(best_hist, 75) -
                                               np.nanpercentile(best_hist, 25))
                                    if iqr_val == 0:
                                        stop_now = True
                                    elif k >= 5:
                                        diff = float(best_hist[k - 5] - best_hist[k])
                                        stop_now = diff / iqr_val <= 0.01
                                elif rule_name == "Convergence":
                                    if k >= 5:
                                        stop_now = (
                                            float(best_obs[seed, k]) == float(best_obs[seed, k - 5])
                                        )
                            except Exception:
                                continue
                            if stop_now:
                                stop_idx = k
                                break
                        regret = (float(test_errs[seed, stop_idx]) - best_error
                                  + lam * float(costs[seed, stop_idx]))
                        stop_vals.append(regret)

                    cost_adjusted_regrets[(lam_str, dataset, acq, rule_name)] = float(np.mean(stop_vals))

                # Hindsight (oracle)
                hindsight_vals = []
                for seed in range(n_seeds):
                    regs = np.array(test_errs[seed]) - best_error + lam * np.array(costs[seed])
                    hindsight_vals.append(float(np.min(regs)))
                cost_adjusted_regrets[(lam_str, dataset, acq, "Hindsight")] = float(np.mean(hindsight_vals))

            # Immediate (same for all acquisitions)
            immediate_mean = float(np.mean(immediate_vals_all))
            for acq in acq_order:
                cost_adjusted_regrets[(lam_str, dataset, acq, "Immediate")] = immediate_mean

    return cost_adjusted_regrets, stopping_rule_names


def compute_rankings(cost_adjusted_regrets, datasets, stopping_rule_names, lambda_strs, acq_order):
    """Compute Top-1, Top-2, Top-3 ranking percentages."""
    results = {}
    n_datasets = len(datasets)

    for lam_str in lambda_strs:
        for acq in acq_order:
            for stop_rule in stopping_rule_names:
                rank_counts = {1: 0, 2: 0, 3: 0}

                for dataset in datasets:
                    # Collect regrets for all pairs on this dataset
                    pair_regrets = []
                    for a in acq_order:
                        for sr in stopping_rule_names:
                            val = cost_adjusted_regrets.get((lam_str, dataset, a, sr), None)
                            if val is not None and not np.isnan(val) and not np.isinf(val):
                                pair_regrets.append(((a, sr), val))

                    # Sort by regret (lower is better)
                    pair_regrets.sort(key=lambda x: x[1])

                    # Find rank of this pair
                    our_val = cost_adjusted_regrets.get((lam_str, dataset, acq, stop_rule), None)
                    if our_val is None or np.isnan(our_val) or np.isinf(our_val):
                        continue
                    try:
                        rank = next(i + 1 for i, (pr, val) in enumerate(pair_regrets)
                                    if pr == (acq, stop_rule))
                    except StopIteration:
                        continue

                    if rank <= 1:
                        rank_counts[1] += 1
                    if rank <= 2:
                        rank_counts[2] += 1
                    if rank <= 3:
                        rank_counts[3] += 1

                results[(lam_str, acq, stop_rule)] = {
                    "top1_pct": rank_counts[1] / n_datasets * 100,
                    "top2_pct": rank_counts[2] / n_datasets * 100,
                    "top3_pct": rank_counts[3] / n_datasets * 100,
                    "top1_count": rank_counts[1],
                    "top2_count": rank_counts[2],
                    "top3_count": rank_counts[3],
                    "n_datasets": n_datasets,
                }

    return results


def main():
    # Load pre-computed experiment data
    pkl_path = "/repo/notebooks/empirical_results/lcbench_known_cost_metrics_per_acq_updated.pkl"
    with open(pkl_path, "rb") as f:
        metrics_per_acq = dill.load(f)

    print("Datasets:", list(metrics_per_acq.keys()))

    # Best test accuracy per dataset (from notebook)
    best_acc_per_dataset = {
        "Fashion-MNIST": 90.17316017316017,
        "adult": 83.00552211950115,
        "higgs": 71.86302385956238,
        "volkert": 62.765681026866915,
    }

    dataset_names = list(metrics_per_acq.keys())
    lambdas = [1e-3, 1e-4, 1e-5]
    lam_strs = ["1e-3", "1e-4", "1e-5"]
    acq_order = ["LogEIPC", "PBGI(1e-3)", "PBGI(1e-4)", "PBGI(1e-5)", "LCB", "TS"]

    # Compute cost-adjusted regrets
    cost_adjusted_regrets, stopping_rule_names = compute_cost_adjusted_regrets(
        metrics_per_acq, lambdas, lam_strs, dataset_names,
        acq_order, best_acc_per_dataset, init=20
    )

    print(f"\nStopping rules: {stopping_rule_names}")
    print(f"Total regret entries: {len(cost_adjusted_regrets)}")

    # Compute rankings
    results = compute_rankings(
        cost_adjusted_regrets, dataset_names, stopping_rule_names, lam_strs, acq_order
    )

    # Display key results
    print("\n" + "=" * 80)
    print("RESULTS: Cost-adjusted regret ranking percentages")
    print(f"Based on {len(dataset_names)} datasets")
    print("=" * 80)

    target_lam = "1e-4"

    print(f"\n--- Lambda = {target_lam} ---")
    print(f"{'Acquisition+Stopping':<35s} {'Top-1':>8s} {'Top-2':>8s} {'Top-3':>8s}")
    print("-" * 60)

    for acq in acq_order:
        for sr in stopping_rule_names:
            key = (target_lam, acq, sr)
            if key in results:
                r = results[key]
                label = f"{acq}+{sr}"
                print(f"{label:<35s} {r['top1_pct']:>7.1f}% {r['top2_pct']:>7.1f}% {r['top3_pct']:>7.1f}%")

    print(f"\n--- Rubric target: PBGI(1e-4) + PBGI/LogEIPC at lambda=1e-4 ---")
    key = ("1e-4", "PBGI(1e-4)", "PBGI")
    if key in results:
        r = results[key]
        print(f"PBGI(1e-4)+PBGI (PBGI/LogEIPC):")
        print(f"  Top-1: {r['top1_pct']:.1f}%  (paper: 40.0%)")
        print(f"  Top-2: {r['top2_pct']:.1f}%  (paper: 70.0%)")
        print(f"  Top-3: {r['top3_pct']:.1f}%  (paper: 80.0%)")

    # Also show LogEIPC+PBGI
    key2 = ("1e-4", "LogEIPC", "PBGI")
    if key2 in results:
        r2 = results[key2]
        print(f"\nLogEIPC+PBGI (LogEIPC/LogEIPC):")
        print(f"  Top-1: {r2['top1_pct']:.1f}%  (paper: 30.0% estimate)")


if __name__ == "__main__":
    main()
