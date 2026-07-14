#!/usr/bin/env python3
"""Multi-K aggregation evaluation for wa-dCoBET (ALGO-1).
Combines test statistics across K values via Cauchy combination.
Usage: python3 eval_multik.py [--K-list "3,4,5,6"] [--R_eval 500] ...
"""
import time, json, argparse, sys, math
import numpy as np
from scipy.stats import norm as norm_dist
from cobet.wa_dcobet import (
    _precache_full_weights, _generate_once_nd, _ten_folds_indices,
    _Z_stat, _blend_full_weights, compute_full_T, plugin_var_tildeT1,
)
from cobet.cobet import all_nonempty_subsets_indices, block_diag

def parse_args():
    p = argparse.ArgumentParser(description="Multi-K wa-dCoBET evaluation")
    p.add_argument("--K-list", type=str, default="3,4,5,6", help="comma-separated K values")
    p.add_argument("--theta", type=float, default=2.0, help="Clayton copula parameter")
    p.add_argument("--R_eval", type=int, default=500, help="Monte Carlo replications")
    p.add_argument("--seed", type=int, default=123, help="random seed")
    p.add_argument("--n", type=int, default=500, help="sample size")
    p.add_argument("--d_coords", type=int, default=10, help="dimension")
    p.add_argument("--alpha", type=float, default=0.05, help="significance level")
    p.add_argument("--output", type=str, default="eval_multik_results.json", help="output JSON path")
    p.add_argument("--b_values", type=str, default="0.0,0.2",
                   help="comma-separated b values")
    return p.parse_args()

def multik_power(K_list, n, theta, transform_key, b, R_eval, alpha, seed, d_coords):
    """Multi-K aggregated power via Cauchy combination."""
    rng = np.random.RandomState(seed)
    zcrit = norm_dist.ppf(1 - alpha)

    # Precompute weights for each K
    W_all_by_K = {}
    subsets_by_K = {}
    for K_val in K_list:
        subsets = all_nonempty_subsets_indices(K_val)
        subsets_by_K[K_val] = subsets
        W_all_by_K[K_val] = _precache_full_weights(K_val, subsets, d_coords, reuse_J=True)

    rejections_combined = 0
    per_K_rejections = {K: 0 for K in K_list}

    for rep in range(R_eval):
        # Generate data once per replication with fresh seed
        rep_seed = seed * 10000 + rep
        rep_rng = np.random.RandomState(rep_seed)

        Z_values = {}
        for K_val in K_list:
            subsets = subsets_by_K[K_val]
            W_all = W_all_by_K[K_val]
            W_id = W_all["identity"]
            W_J = W_all["J"]

            A, B = _generate_once_nd(n, theta, b, K_val, transform_key, subsets, d_coords, rep_rng)

            # 10-fold weight selection (hard binary voting)
            folds = _ten_folds_indices(n, rep_rng)
            fold_picks = []
            for fidx in folds:
                A_f, B_f = A[:, fidx], B[:, fidx]
                W_A_id, W_B_id, W_C_id, _ = W_id
                W_A_J,  W_B_J,  W_C_J,  _ = W_J
                Z_id = _Z_stat(A_f, B_f, W_A_id, W_B_id, W_C_id, True)
                Z_J  = _Z_stat(A_f, B_f, W_A_J,  W_B_J,  W_C_J,  True)
                fold_picks.append("identity" if Z_id >= Z_J else "J")

            cnt_id = sum(p == "identity" for p in fold_picks)
            cnt_J = 10 - cnt_id
            w_id = cnt_id / 10.0
            w_J = cnt_J / 10.0

            W_A_new, W_B_new, W_C_new, _ = _blend_full_weights(W_id, W_J, w_id, w_J)
            T_new = compute_full_T(A, B, W_A_new, W_B_new, W_C_new)
            vT_new = plugin_var_tildeT1(A, B, W_A_new, W_B_new, unbiased=True)
            Z_new = T_new / np.sqrt(max(vT_new, 1e-16))
            Z_values[K_val] = Z_new

            if Z_new > zcrit:
                per_K_rejections[K_val] += 1

        # Cauchy combination of p-values across K
        m = len(K_list)
        cauchy_sum = 0.0
        for K_val in K_list:
            p_val = 1.0 - norm_dist.cdf(Z_values[K_val])
            p_val = max(min(p_val, 1.0 - 1e-15), 1e-15)
            cauchy_sum += math.tan((0.5 - p_val) * math.pi)

        T_cauchy = cauchy_sum / m
        p_combined = 0.5 - math.atan(T_cauchy) / math.pi

        if p_combined < alpha:
            rejections_combined += 1

    return {
        "power_combined": rejections_combined / R_eval,
        "per_K_power": {str(K): per_K_rejections[K] / R_eval for K in K_list},
        "R_eval": R_eval,
        "alpha": alpha,
        "K_list": K_list,
    }

def main():
    args = parse_args()
    K_list = [int(x) for x in args.K_list.split(",")]
    b_values = [float(x) for x in args.b_values.split(",")]

    all_results = {}
    for b_val in b_values:
        label = "Type_I_Error" if b_val == 0.0 else "Power"
        print("Running Multi-K wa-dCoBET b={} K={}...".format(b_val, K_list), flush=True)
        t0 = time.time()
        r = multik_power(
            K_list=K_list, n=args.n, theta=args.theta,
            transform_key="logquad", b=b_val,
            R_eval=args.R_eval, alpha=args.alpha,
            seed=args.seed, d_coords=args.d_coords,
        )
        elapsed = time.time() - t0
        all_results[label] = r["power_combined"]
        all_results[label + "_runtime_s"] = round(elapsed, 1)
        all_results[label + "_per_K"] = r["per_K_power"]

        print("  {} = {:.4f}  (runtime {:.1f}s)".format(label, r["power_combined"], elapsed), flush=True)
        print("  Per-K: {}".format(r["per_K_power"]), flush=True)

    print("\n=== MULTI-K RESULTS ===", flush=True)
    for k, v in sorted(all_results.items()):
        if not k.endswith("_runtime_s") and not k.endswith("_per_K"):
            print("{}: {:.4f}".format(k, v), flush=True)

    all_results["_config"] = {"K_list": K_list, "n": args.n, "theta": args.theta,
                               "R_eval": args.R_eval, "seed": args.seed}

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print("Saved to {}".format(args.output), flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
