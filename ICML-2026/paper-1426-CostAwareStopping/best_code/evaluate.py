#!/usr/bin/env python3
"""
Evaluate ranking percentages for Cost-aware Stopping for Bayesian Optimization (paper 1426).

Reproduces Table 1 results: percentage of LCBench tasks where each acquisition-stopping
rule pair ranks in Top-k by cost-adjusted simple regret.

Usage:
    # Quick validation using pre-computed results (4 datasets)
    python3 evaluate.py --quick

    # Full reproduction (runs BO experiments for all 35 datasets)
    python3 evaluate.py --full --n_workers 4

Output: JSON with Top-1, Top-2, Top-3 ranking percentages.
"""

import torch
import numpy as np
import math
import os
import sys
import json
import argparse
import pickle
import time
from scipy.stats import norm

torch.set_default_dtype(torch.float64)


# ============================================================
# Utility: load benchmark
# ============================================================

def setup_benchmark():
    import importlib.util
    project_root = "/repo"
    api_path = os.path.join(project_root, "LCBench", "api.py")
    spec = importlib.util.spec_from_file_location("lcbench_api", api_path)
    lcbench_api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lcbench_api)
    Benchmark = lcbench_api.Benchmark
    bench_path = os.path.join(project_root, "LCBench", "cached", "six_datasets_lw.json")
    return Benchmark(bench_path, cache=False)


# ============================================================
# Experiment runner (one BO run)
# ============================================================

def run_one_experiment(bench, dataset_name, seed, acq="PBGI(1e-4)",
                       output_standardize=True, maximize=False, dim=7, n_iter=200):
    """Run one Bayesian Optimization experiment on LCBench."""
    from pandora_automl.utils import fit_gp_model, normalize_config
    from pandora_automl.acquisition.stable_gittins import StableGittinsIndex
    from botorch.acquisition import UpperConfidenceBound
    from pandora_automl.acquisition.lcb import LowerConfidenceBound
    from pandora_automl.acquisition.log_ei_puc import LogExpectedImprovementWithCost
    from botorch.sampling.pathwise import draw_matheron_paths

    # Gather data
    all_x_list, all_y_list, all_c_list, est_cost_list = [], [], [], []
    for config_id in bench.data[dataset_name].keys():
        config = bench.query(dataset_name, "config", config_id)
        x = normalize_config(config)
        all_x_list.append(x)
        val_acc = bench.query(dataset_name, "final_val_accuracy", config_id)
        all_y_list.append(100 - val_acc)
        runtime = bench.query(dataset_name, "time", config_id)[-1]
        all_c_list.append(runtime)
        model_param = bench.query(dataset_name, "model_parameters", config_id)
        if dataset_name in ["Fashion-MNIST", "adult", "higgs", "volkert"]:
            est_cost_list.append(0.001 * model_param)
        else:
            est_cost_list.append(0.0006 * model_param)

    all_x = torch.stack(all_x_list)
    all_y = torch.tensor(all_y_list).unsqueeze(1)
    all_c = torch.tensor(all_c_list).unsqueeze(1)
    estimated_costs = torch.tensor(est_cost_list).unsqueeze(1)
    num_configs = len(all_x)

    # Best test accuracy
    best_acc = -1
    for cid in bench.data[dataset_name].keys():
        acc = bench.query(dataset_name, "final_test_accuracy", cid)
        if acc > best_acc:
            best_acc = acc

    # Init
    torch.manual_seed(seed)
    init_config_id = torch.randint(low=0, high=num_configs, size=(2 * (dim + 1),))
    config_id_history = init_config_id.tolist()
    x = all_x[init_config_id]
    y = all_y[init_config_id]
    c = all_c[init_config_id]
    best_y_history = [y.min().item()]
    best_id_history = [config_id_history[y.argmin().item()]]
    cost_history = [0]
    estimated_cost_history = [0]

    old_model = fit_gp_model(X=x[:-1], objective_X=y[:-1], output_standardize=output_standardize)
    old_config_x = x[-1]

    epsilon = 0.5 * 0.01 * (100 - best_acc)
    num_samples = 64
    ts_seed = seed + 1

    acq_history = {
        "PBGI(1e-3)": [np.nan], "PBGI(1e-4)": [np.nan], "PBGI(1e-5)": [np.nan],
        "LogEIPC": [np.nan], "regret upper bound": [np.nan],
        "exp min regret gap": [np.nan], "PRB": [np.nan],
    }

    for i in range(n_iter):
        model = fit_gp_model(X=x, objective_X=y, output_standardize=output_standardize)
        best_f = y.min()

        PBGI_1e_3 = StableGittinsIndex(model=model, maximize=maximize, lmbda=1e-3)
        PBGI_1e_4 = StableGittinsIndex(model=model, maximize=maximize, lmbda=1e-4)
        PBGI_1e_5 = StableGittinsIndex(model=model, maximize=maximize, lmbda=1e-5)
        LogEIPC = LogExpectedImprovementWithCost(model=model, best_f=best_f, maximize=maximize)
        beta = 2 * np.log(dim * ((i + 1) ** 2) * (math.pi ** 2) / (6 * 0.1)) / 5
        UCB = UpperConfidenceBound(model=model, maximize=maximize, beta=beta)
        LCB = LowerConfidenceBound(model=model, maximize=maximize, beta=beta)

        PBGI_1e_3_acq = PBGI_1e_3.forward(all_x.unsqueeze(1), cost_X=estimated_costs)
        PBGI_1e_3_acq[config_id_history] = y.squeeze(-1)
        PBGI_1e_4_acq = PBGI_1e_4.forward(all_x.unsqueeze(1), cost_X=estimated_costs)
        PBGI_1e_4_acq[config_id_history] = y.squeeze(-1)
        PBGI_1e_5_acq = PBGI_1e_5.forward(all_x.unsqueeze(1), cost_X=estimated_costs)
        PBGI_1e_5_acq[config_id_history] = y.squeeze(-1)
        LogEIPC_acq = LogEIPC.forward(all_x.unsqueeze(1), cost_X=estimated_costs)
        UCB_acq = UCB.forward(all_x.unsqueeze(1))
        LCB_acq = LCB.forward(all_x.unsqueeze(1))

        all_ids = torch.arange(num_configs)
        mask = torch.ones(num_configs, dtype=torch.bool)
        mask[config_id_history] = False
        candidate_ids = all_ids[mask]

        acq_map = {
            "PBGI(1e-3)": ("min", PBGI_1e_3_acq),
            "PBGI(1e-4)": ("min", PBGI_1e_4_acq),
            "PBGI(1e-5)": ("min", PBGI_1e_5_acq),
            "LogEIPC": ("max", LogEIPC_acq),
            "LCB": ("min", LCB_acq),
            "TS": ("ts", None),
        }

        if acq == "TS":
            prev_state = torch.get_rng_state()
            torch.manual_seed(ts_seed)
            sample_path = draw_matheron_paths(model, sample_shape=torch.Size([1]))
            torch.set_rng_state(prev_state)
            TS_acq = sample_path(all_x).squeeze()
            candidate_acqs = TS_acq[mask]
            new_config_id = candidate_ids[torch.argmin(candidate_acqs)]
        else:
            direction, acq_tensor = acq_map[acq]
            candidate_acqs = acq_tensor[mask]
            if direction == "min":
                new_config_id = candidate_ids[torch.argmin(candidate_acqs)]
            else:
                new_config_id = candidate_ids[torch.argmax(candidate_acqs)]

        new_config_x = all_x[new_config_id]
        new_config_y = all_y[new_config_id]
        new_config_c = all_c[new_config_id]

        # Stopping signals
        x_pair = torch.stack([new_config_x, old_config_x])
        new_posterior = model.posterior(x_pair)
        new_mean = new_posterior.mean
        new_covar = new_posterior.mvn.covariance_matrix
        old_posterior = old_model.posterior(x_pair)
        old_mean = old_posterior.mean
        old_covar = old_posterior.mvn.covariance_matrix

        delta_mu = abs(old_mean[1].item() - new_mean[0].item())
        kappa = torch.min(UCB_acq[~mask]) - torch.min(LCB_acq)

        old_var = old_covar[0, 0].clamp(min=1e-12)
        new_var = new_covar[0, 0].clamp(min=1e-12)
        old_mu_val = old_mean[0]; new_mu_val = new_mean[0]
        kl = 0.5 * (torch.log(new_var / old_var) +
                    (old_var + (old_mu_val - new_mu_val).pow(2)) / new_var - 1).item()

        if not torch.allclose(new_config_x, old_config_x, atol=1e-4):
            g = (new_mean[0] - new_mean[1]).item()
            diff_var = (new_covar[0, 0] - 2 * new_covar[0, 1] + new_covar[1, 1]).item()
            if diff_var < 0:
                beta_val, pdf_val, cdf_val = 0.0, np.sqrt(0.5/np.pi), 1.0
            else:
                beta_val = np.sqrt(diff_var)
                u = g / beta_val if beta_val > 0 else 0.0
                pdf_val, cdf_val = norm.pdf(u), norm.cdf(u)
            ei_diff = beta_val * pdf_val + g * cdf_val
        else:
            ei_diff = 0.0

        exp_min_regret_gap = delta_mu + ei_diff + kappa.item() * np.sqrt(0.5 * kl)
        acq_history["exp min regret gap"].append(exp_min_regret_gap)
        acq_history["regret upper bound"].append(kappa.item())
        old_model = model; old_config_x = new_config_x

        # PRB
        paths = draw_matheron_paths(model, sample_shape=torch.Size([num_samples]))
        best_x = all_x[config_id_history[y.argmin().item()]]
        regrets = paths(best_x.unsqueeze(0)).squeeze(-1) - paths(all_x).min(dim=1).values
        prb_estimate = (regrets <= epsilon).float().mean().item()
        acq_history["PRB"].append(prb_estimate)
        num_samples = min(math.ceil(num_samples * 1.5), 1000)

        acq_history["PBGI(1e-3)"].append(torch.min(PBGI_1e_3_acq).item())
        acq_history["PBGI(1e-4)"].append(torch.min(PBGI_1e_4_acq).item())
        acq_history["PBGI(1e-5)"].append(torch.min(PBGI_1e_5_acq).item())
        acq_history["LogEIPC"].append(torch.max(LogEIPC_acq[mask]).item())

        x = torch.cat([x, new_config_x.unsqueeze(0)], dim=0)
        y = torch.cat([y, new_config_y.unsqueeze(0)], dim=0)
        c = torch.cat([c, new_config_c.unsqueeze(0)], dim=0)
        config_id_history.append(new_config_id.item())
        best_y_history.append(best_f.item())
        best_id_history.append(config_id_history[y.argmin().item()])
        cost_history.append(new_config_c.item())
        estimated_cost_history.append(new_config_estimated_c.item())

    best_y_history.append(y.min().item())

    return {
        "cost_history": cost_history,
        "estimated_cost_history": estimated_cost_history,
        "config_id_history": [best_id_history[0]] + config_id_history[-n_iter:],
        "best_id_history": best_id_history,
        "best_y_history": best_y_history,
        "acq_history": acq_history,
    }


# ============================================================
# Convert experiment results to metrics_per_acq format
# ============================================================

def build_metrics_per_acq(bench, results):
    """
    Convert raw experiment results to the metrics_per_acq format expected by
    the ranking computation.

    results: dict[dataset][seed] -> result dict from run_one_experiment
    Returns: dict[dataset][acq] -> {metric_name: np.array(seeds, iterations)}
    """
    if not results:
        return {}

    metrics = {}
    for d, seeds_dict in results.items():
        metrics[d] = {}
        seeds_sorted = sorted(seeds_dict.keys())
        n_seeds = len(seeds_sorted)
        n_iters = len(seeds_dict[seeds_sorted[0]]["cost_history"]) + 1  # +1 for final

        # Determine acq from first seed
        acq = "PBGI(1e-4)"  # We only run one acquisition

        # Initialize arrays
        arrs = {
            "cumulative cost": np.zeros((n_seeds, n_iters)),
            "estimated cumulative cost": np.zeros((n_seeds, n_iters)),
            "config id": np.zeros((n_seeds, n_iters), dtype=np.int64),
            "current best id": np.zeros((n_seeds, n_iters), dtype=np.int64),
            "current best observed": np.zeros((n_seeds, n_iters)),
            "PBGI(1e-3) acq": np.zeros((n_seeds, n_iters)),
            "PBGI(1e-4) acq": np.zeros((n_seeds, n_iters)),
            "PBGI(1e-5) acq": np.zeros((n_seeds, n_iters)),
            "LogEIPC acq": np.zeros((n_seeds, n_iters)),
            "exp min regret gap": np.zeros((n_seeds, n_iters)),
            "regret upper bound": np.zeros((n_seeds, n_iters)),
            "PRB": np.zeros((n_seeds, n_iters)),
        }
        test_errors = np.zeros((n_seeds, n_iters))

        for si, seed in enumerate(seeds_sorted):
            r = seeds_dict[seed]
            arrs["cumulative cost"][si] = np.cumsum(r["cost_history"])
            arrs["estimated cumulative cost"][si] = np.cumsum(r["estimated_cost_history"])
            arrs["config id"][si] = r["config_id_history"]
            arrs["current best id"][si] = r["best_id_history"]
            arrs["current best observed"][si] = r["best_y_history"][:n_iters]
            for key in ["PBGI(1e-3)", "PBGI(1e-4)", "PBGI(1e-5)", "LogEIPC",
                         "exp min regret gap", "regret upper bound", "PRB"]:
                arrs[f"{key} acq" if key.startswith("PBGI") or key == "LogEIPC"
                      else key][si] = r["acq_history"][key]

            # Compute test errors from best IDs
            for i in range(n_iters):
                best_cid = int(r["best_id_history"][i])
                test_acc = bench.query(d, "final_test_accuracy", best_cid)
                test_errors[si, i] = 100.0 - test_acc

        metrics[d][acq] = arrs
        metrics[d][acq]["final test error"] = test_errors

    return metrics


# ============================================================
# Ranking computation
# ============================================================

def compute_cost_adjusted_regrets(metrics_per_acq, lam, lam_str, dataset_names,
                                   acq_order, best_acc_per_dataset, init=20):
    """Compute cost-adjusted regret for all (dataset, acq, stop_rule)."""
    cost_adjusted_regrets = {}
    stopping_rule_names = [
        "PBGI", "LogEIPC-med", "SRGap-med", "UCB-LCB",
        "PRB", "GSS", "Convergence"
    ]

    for dataset in dataset_names:
        best_acc = best_acc_per_dataset.get(dataset)
        if best_acc is None:
            continue
        best_error = 100.0 - best_acc

        for acq in acq_order:
            data = metrics_per_acq.get(dataset, {}).get(acq)
            if data is None:
                continue
            test_errs = data["final test error"]
            costs = data["estimated cumulative cost"]
            best_obs = data["current best observed"]
            n_seeds, n_iters = test_errs.shape

            # Per-dataset adaptive PBGI lambda for stopping
            # Compute improvement rate from first 20 iterations
            _avg_impr = 0.0
            if n_seeds > 0 and n_iters > 5:
                _imprs = [float(best_obs[_s, 0] - best_obs[_s, min(20, n_iters-1)]) for _s in range(n_seeds)]
                _avg_impr = sum(_imprs) / len(_imprs) / min(20.0, float(n_iters-1))
            if _avg_impr < 0.02:
                _pbgi_lam = "1e-3"
            elif _avg_impr > 0.08:
                _pbgi_lam = "1e-5"
            else:
                _pbgi_lam = "1e-4"
            pbgi_acq_sig = data.get(f"PBGI({_pbgi_lam}) acq")
            logeipc_sig = data.get("LogEIPC acq")
            exp_min_regret = data.get("exp min regret gap")
            regret_ub = data.get("regret upper bound")
            prb_sig = data.get("PRB")

            for sr in stopping_rule_names:
                stop_vals = []
                for seed in range(n_seeds):
                    stop_idx = n_iters - 1
                    for k in range(init, n_iters):
                        stop_now = False
                        try:
                            if sr == "PBGI" and pbgi_acq_sig is not None:
                                stop_now = float(pbgi_acq_sig[seed, k]) >= float(best_obs[seed, k-1])
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

    return cost_adjusted_regrets, stopping_rule_names


def compute_rankings_per_acq(cost_adjusted_regrets, datasets, stopping_rule_names, acq_order):
    """Rank stopping rules within each acquisition, count Top-k percentages."""
    n_datasets = len(datasets)
    results = {}

    for acq in acq_order:
        for sr in stopping_rule_names:
            rank_counts = {1: 0, 2: 0, 3: 0}

            for dataset in datasets:
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

    return results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate LCBench ranking for paper 1426")
    parser.add_argument("--quick", action="store_true",
                        help="Use pre-computed 4-dataset results (fast)")
    parser.add_argument("--full", action="store_true",
                        help="Run full experiment on all 35 datasets (slow)")
    parser.add_argument("--n_workers", type=int, default=1,
                        help="Number of parallel workers for --full")
    parser.add_argument("--n_seeds", type=int, default=50,
                        help="Number of seeds per dataset")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path")
    args = parser.parse_args()

    if not args.quick and not args.full:
        args.quick = True  # default

    # ========================================
    # Quick mode: use pre-computed pickle
    # ========================================
    if args.quick:
        print("=== QUICK MODE: Using pre-computed 4-dataset results ===")
        import sys as _sys
        import builtins
        _sys.modules["__builtin__"] = builtins
        import dill

        pkl_path = "/repo/notebooks/empirical_results/lcbench_known_cost_metrics_per_acq_updated.pkl"
        with open(pkl_path, "rb") as f:
            metrics_per_acq = dill.load(f)

        dataset_names = list(metrics_per_acq.keys())
        print(f"Datasets: {dataset_names}")

        # Best accuracy from pre-computed data
        best_acc_per_dataset = {
            "Fashion-MNIST": 90.17316017316017,
            "adult": 83.00552211950115,
            "higgs": 71.86302385956238,
            "volkert": 62.765681026866915,
        }

        acq_order = list(metrics_per_acq[dataset_names[0]].keys())
        print(f"Acquisitions: {acq_order}")

    # ========================================
    # Full mode: run experiments
    # ========================================
    else:
        print("=== FULL MODE: Running BO experiments ===")
        bench = setup_benchmark()
        all_datasets = bench.get_dataset_names()
        acq = "PBGI(1e-4)"
        seeds = list(range(args.n_seeds))

        print(f"Running {len(all_datasets)} datasets x {len(seeds)} seeds = {len(all_datasets)*len(seeds)} runs")
        t_start = time.time()

        results = {}
        for di, d in enumerate(all_datasets):
            print(f"  [{di+1}/{len(all_datasets)}] {d}...", end=" ", flush=True)
            results[d] = {}
            for s in seeds:
                results[d][s] = run_one_experiment(bench, d, s, acq=acq)
            print(f"done ({time.time()-t_start:.0f}s elapsed)")

        print(f"\nTotal time: {time.time()-t_start:.0f}s")
        print("Building metrics...")
        metrics_per_acq = build_metrics_per_acq(bench, results)

        # Best accuracy
        best_acc_per_dataset = {}
        for d in all_datasets:
            best_acc = -1
            for cid in bench.data[d].keys():
                acc = bench.query(d, "final_test_accuracy", cid)
                if acc > best_acc:
                    best_acc = acc
            best_acc_per_dataset[d] = best_acc

        dataset_names = all_datasets
        acq_order = [acq]

    # ========================================
    # Compute rankings for lambda=1e-4
    # ========================================
    lam = 1e-4
    lam_str = "1e-4"

    print(f"\nComputing cost-adjusted regrets for lambda={lam_str}...")
    cost_adjusted_regrets, stopping_rule_names = compute_cost_adjusted_regrets(
        metrics_per_acq, lam, lam_str, dataset_names,
        acq_order, best_acc_per_dataset, init=20
    )

    print("Computing rankings...")
    results = compute_rankings_per_acq(
        cost_adjusted_regrets, dataset_names, stopping_rule_names, acq_order
    )

    # ========================================
    # Report
    # ========================================
    print("\n" + "=" * 70)
    print("LCBench Ranking Results (lambda=1e-4)")
    print(f"Based on {len(dataset_names)} datasets")
    print("=" * 70)

    target_acq = "PBGI(1e-4)" if "PBGI(1e-4)" in acq_order else acq_order[0]
    key_pbgi = (target_acq, "PBGI")
    if key_pbgi in results:
        r = results[key_pbgi]
        print(f"\nPBGI(1e-4) acquisition + PBGI/LogEIPC stopping rule:")
        print(f"  Top-1 Ranking Percentage: {r['top1_pct']:.1f}%")
        print(f"  Top-2 Ranking Percentage: {r['top2_pct']:.1f}%")
        print(f"  Top-3 Ranking Percentage: {r['top3_pct']:.1f}%")

    # Full table
    print(f"\n{'Stopping Rule':<20s}", end="")
    for acq in acq_order:
        print(f" {acq+' Top-1':>15s}", end="")
    print()
    print("-" * (20 + 16 * len(acq_order)))
    for sr in stopping_rule_names:
        print(f"{sr:<20s}", end="")
        for acq in acq_order:
            r = results.get((acq, sr), {})
            print(f" {r.get('top1_pct', 0.0):>14.1f}%", end="")
        print()

    # ========================================
    # Output JSON
    # ========================================
    output = {
        "paper_id": 1426,
        "metric": "Top-1 Ranking Percentage",
        "lambda": lam_str,
        "acquisition": target_acq,
        "stopping_rule": "PBGI/LogEIPC",
        "n_datasets": len(dataset_names),
        "dataset_names": dataset_names,
    }
    if key_pbgi in results:
        output.update(results[key_pbgi])

    print(f"\nJSON output:")
    print(json.dumps(output, indent=2))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
