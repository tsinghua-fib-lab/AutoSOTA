#!/usr/bin/env python3
"""Test hybrid PBGI+LogEIPC stopping rules."""
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

base_srs = ["LogEIPC-med", "SRGap-med", "UCB-LCB", "PRB", "GSS", "Convergence"]

# Precompute base regrets
base_regrets = {}
for dataset in datasets:
    best_error = 100.0 - best_acc[dataset]
    for acq_key in ["PBGI(1e-3)", "PBGI(1e-4)", "PBGI(1e-5)", "LogEIPC", "LCB", "TS"]:
        data = metrics_per_acq[dataset].get(acq_key)
        if data is None:
            continue
        test_errs = data["final test error"]
        costs = data["estimated cumulative cost"]
        best_obs_data = data["current best observed"]
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
                            bh = best_obs_data[seed, :k+1]
                            iqr_v = np.nanpercentile(bh, 75) - np.nanpercentile(bh, 25)
                            if iqr_v == 0:
                                stop_now = True
                            elif k >= 5:
                                stop_now = float(bh[k-5] - bh[k]) / iqr_v <= 0.01
                        elif sr == "Convergence":
                            if k >= 5:
                                stop_now = float(best_obs_data[seed, k]) == float(best_obs_data[seed, k-5])
                    except:
                        continue
                    if stop_now:
                        stop_idx = k
                        break
                regret = float(test_errs[seed, stop_idx] - best_error) + lam * float(costs[seed, stop_idx])
                stop_vals.append(regret)
            base_regrets[(dataset, acq_key, sr)] = float(np.mean(stop_vals))


def eval_custom_pbgi(rule_fn, rule_name):
    """Evaluate a custom PBGI stopping rule."""
    target_acq = "PBGI(1e-4)"
    n_datasets = len(datasets)

    # Compute custom PBGI regret for each dataset
    custom_regrets = {}
    for dataset in datasets:
        best_error = 100.0 - best_acc[dataset]
        data = metrics_per_acq[dataset][target_acq]
        test_errs = data["final test error"]
        costs = data["estimated cumulative cost"]
        best_obs_data = data["current best observed"]
        n_seeds, n_iters = test_errs.shape

        stop_vals = []
        stop_idxs = []
        for seed in range(n_seeds):
            stop_idx = n_iters - 1
            for k in range(20, n_iters):
                stop_now = rule_fn(data, seed, k, best_obs_data)
                if stop_now:
                    stop_idx = k
                    break
            regret = float(test_errs[seed, stop_idx] - best_error) + lam * float(costs[seed, stop_idx])
            stop_vals.append(regret)
            stop_idxs.append(stop_idx)
        custom_regrets[dataset] = float(np.mean(stop_vals))

    # Rank within each dataset
    top1 = top2 = top3 = 0
    for dataset in datasets:
        other_regrets = [base_regrets[(dataset, target_acq, sr)] for sr in base_srs]
        all_regrets = [custom_regrets[dataset]] + other_regrets
        all_labels = [rule_name] + base_srs
        ranks = np.array(all_regrets).argsort().argsort()
        r = int(ranks[0]) + 1
        if r <= 1: top1 += 1
        if r <= 2: top2 += 1
        if r <= 3: top3 += 1

        # Per-dataset detail
        print(f"  {dataset}: regret={custom_regrets[dataset]:.6f}, rank={r}/{len(all_labels)}")

    t1 = top1 / n_datasets * 100
    t2 = top2 / n_datasets * 100
    t3 = top3 / n_datasets * 100
    print(f"  Result: Top-1={t1:.1f}%, Top-2={t2:.1f}%, Top-3={t3:.1f}%")
    return t1, t2, t3


# ======== Rule 1: PBGI with LogEIPC convergence-adaptive threshold ========
def rule_convergence_adaptive(data, seed, k, best_obs_data):
    pbgi_acq = data["PBGI(1e-4) acq"]
    logeipc_acq = data["LogEIPC acq"]

    pbgi_val = float(pbgi_acq[seed, k])
    best_val = float(best_obs_data[seed, k-1])

    # Check LogEIPC convergence: how far below its early median?
    early_vals = logeipc_acq[seed, min(1,k):min(21, k)]
    if len(early_vals) < 5:
        return pbgi_val >= best_val

    early_median = np.nanmedian(early_vals)
    early_std = np.nanstd(early_vals) + 1e-10
    current_logeipc = float(logeipc_acq[seed, k])

    # Z-score of current LogEIPC relative to early values
    z_score = (current_logeipc - early_median) / early_std

    # If LogEIPC has dropped significantly (strong convergence signal),
    # use a lower PBGI threshold
    if z_score < -2.0:
        threshold_scale = 0.92
    elif z_score < -1.0:
        threshold_scale = 0.96
    else:
        threshold_scale = 1.0

    return pbgi_val >= threshold_scale * best_val


# ======== Rule 2: PBGI with best-lambda-per-dataset (selected by LogEIPC trend) ========
def rule_lambda_by_logeipc(data, seed, k, best_obs_data):
    """Select PBGI lambda based on LogEIPC convergence speed."""
    pbgi_1e3 = data.get("PBGI(1e-3) acq")
    pbgi_1e4 = data.get("PBGI(1e-4) acq")
    pbgi_1e5 = data.get("PBGI(1e-5) acq")
    logeipc_acq = data.get("LogEIPC acq")

    best_val = float(best_obs_data[seed, k-1])

    # Measure LogEIPC convergence speed
    if k >= 30:
        recent_eipc = logeipc_acq[seed, k-10:k+1]
        early_eipc = logeipc_acq[seed, 1:21]
        drop_ratio = np.nanmedian(recent_eipc) - np.nanmedian(early_eipc)

        # Fast convergence -> use aggressive lambda
        if drop_ratio < -2.0:
            pbgi_val = float(pbgi_1e3[seed, k])
        elif drop_ratio < -0.5:
            pbgi_val = float(pbgi_1e4[seed, k])
        else:
            pbgi_val = float(pbgi_1e5[seed, k])
    else:
        pbgi_val = float(pbgi_1e4[seed, k])

    return pbgi_val >= best_val


# ======== Rule 3: PBGI with per-dataset optimal lambda (oracle, for comparison) ========
optimal_per_dataset = {
    "Fashion-MNIST": "1e-3",
    "adult": "1e-3",
    "higgs": "1e-4",
    "volkert": "1e-5",
}
def rule_oracle_lambda(data, seed, k, best_obs_data, dataset_name):
    lam_key = optimal_per_dataset[dataset_name]
    pbgi_acq = data.get(f"PBGI({lam_key}) acq")
    pbgi_val = float(pbgi_acq[seed, k])
    best_val = float(best_obs_data[seed, k-1])
    return pbgi_val >= best_val


# ======== Rule 4: PBGI with majority vote across lambdas ========
def rule_majority_vote(data, seed, k, best_obs_data):
    pbgi_1e3 = data.get("PBGI(1e-3) acq")
    pbgi_1e4 = data.get("PBGI(1e-4) acq")
    pbgi_1e5 = data.get("PBGI(1e-5) acq")
    best_val = float(best_obs_data[seed, k-1])

    votes = 0
    if float(pbgi_1e3[seed, k]) >= best_val: votes += 1
    if float(pbgi_1e4[seed, k]) >= best_val: votes += 1
    if float(pbgi_1e5[seed, k]) >= best_val: votes += 1
    return votes >= 2

# ======== Rule 5: PBGI with stop at earliest signal (OR) ========
def rule_earliest(data, seed, k, best_obs_data):
    pbgi_acq = data.get("PBGI(1e-3) acq")  # Most aggressive
    pbgi_val = float(pbgi_acq[seed, k])
    best_val = float(best_obs_data[seed, k-1])
    return pbgi_val >= best_val

# ======== Rule 6: PBGI with latest signal (AND) ========
def rule_latest(data, seed, k, best_obs_data):
    pbgi_acq = data.get("PBGI(1e-5) acq")  # Most conservative
    pbgi_val = float(pbgi_acq[seed, k])
    best_val = float(best_obs_data[seed, k-1])
    return pbgi_val >= best_val


print("=== Rule 1: Convergence-adaptive threshold ===")
eval_custom_pbgi(rule_convergence_adaptive, "PBGI-adaptive")

print("\n=== Rule 2: Lambda selected by LogEIPC trend ===")
eval_custom_pbgi(rule_lambda_by_logeipc, "PBGI-dynamic")

print("\n=== Rule 3: Oracle per-dataset lambda ===")
# Special handling for oracle
target_acq = "PBGI(1e-4)"
n_datasets = len(datasets)
for dataset in datasets:
    lam_key = optimal_per_dataset[dataset]
    data = metrics_per_acq[dataset][target_acq]
    test_errs = data["final test error"]
    costs = data["estimated cumulative cost"]
    best_obs_data = data["current best observed"]
    pbgi_acq = data[f"PBGI({lam_key}) acq"]
    best_error = 100.0 - best_acc[dataset]
    n_seeds, n_iters = test_errs.shape
    stop_vals = []
    for seed in range(n_seeds):
        stop_idx = n_iters - 1
        for k in range(20, n_iters):
            if float(pbgi_acq[seed, k]) >= float(best_obs_data[seed, k-1]):
                stop_idx = k
                break
        regret = float(test_errs[seed, stop_idx] - best_error) + lam * float(costs[seed, stop_idx])
        stop_vals.append(regret)
    r = float(np.mean(stop_vals))
    other_regrets = [base_regrets[(dataset, target_acq, sr)] for sr in base_srs]
    all_regrets = [r] + other_regrets
    ranks = np.array(all_regrets).argsort().argsort()
    rank = int(ranks[0]) + 1
    print(f"  {dataset}: regret={r:.6f}, rank={rank}/{1+len(base_srs)} (using {lam_key})")

print("\n=== Rule 4: Majority vote ===")
eval_custom_pbgi(rule_majority_vote, "PBGI-majority")

print("\n=== Rule 5: Earliest (1e-3) ===")
eval_custom_pbgi(rule_earliest, "PBGI-earliest")

print("\n=== Rule 6: Latest (1e-5) ===")
eval_custom_pbgi(rule_latest, "PBGI-latest")
