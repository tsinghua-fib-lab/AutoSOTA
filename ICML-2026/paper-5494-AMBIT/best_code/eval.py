#!/usr/bin/env python3
"""Evaluation script for wa-dCoBET (paper 5494) — parameterized version.
Usage: python3 eval.py [--K 4] [--theta 2] [--R_eval 500] [--seed 123] ...
"""
import time, json, argparse, sys
import numpy as np
from cobet import aggregated_weights_power

def parse_args():
    p = argparse.ArgumentParser(description="wa-dCoBET evaluation")
    p.add_argument("--K", type=int, default=4, help="binary expansion depth")
    p.add_argument("--theta", type=float, default=2.0, help="Clayton copula parameter")
    p.add_argument("--R_eval", type=int, default=500, help="Monte Carlo replications")
    p.add_argument("--seed", type=int, default=123, help="random seed")
    p.add_argument("--n", type=int, default=500, help="sample size")
    p.add_argument("--d_coords", type=int, default=10, help="dimension")
    p.add_argument("--alpha", type=float, default=0.05, help="significance level")
    p.add_argument("--output", type=str, default="eval_results.json", help="output JSON path")
    p.add_argument("--b_values", type=str, default="0.0,0.2",
                   help="comma-separated b values (0.0=TypeI, >0=Power)")
    p.add_argument("--label_map", type=str, default="0.0:Type_I_Error,0.2:Power",
                   help="comma-separated b:label pairs")
    p.add_argument("--voting-mode", type=str, default="hard",
                   choices=["hard", "soft"], help="SNR voting mode")
    p.add_argument("--n-folds", type=int, default=10, help="number of SNR voting folds")
    p.add_argument("--alpha-mode", type=str, default="binary",
                   choices=["binary", "continuous"], help="weight selection mode")
    p.add_argument("--alpha-grid", type=str, default=None,
                   help="comma-separated alpha values for continuous mode")
    p.add_argument("--no-plugin", action="store_true", help="disable unbiased plugin variance")
    return p.parse_args()

def main():
    args = parse_args()
    label_map = {}
    for pair in args.label_map.split(","):
        b_str, label = pair.split(":")
        label_map[float(b_str)] = label

    b_values = [float(x) for x in args.b_values.split(",")]

    config = dict(
        n=args.n, theta=args.theta, K=args.K,
        transform_key="logquad", R_eval=args.R_eval,
        alpha=args.alpha, d_coords=args.d_coords,
        unbiased_plugin=not args.no_plugin, reuse_J=True,
        voting_mode=args.voting_mode,
        alpha_mode=args.alpha_mode,
        n_folds=args.n_folds,
        alpha_grid=[float(x) for x in args.alpha_grid.split(",")] if args.alpha_grid else None,
        seed=args.seed,
    )

    results = {}
    for b_val in b_values:
        label = label_map.get(b_val, "b_{}".format(b_val))
        print("Running wa-dCoBET b={} (K={}, theta={}, R={}, seed={})...".format(
            b_val, args.K, args.theta, args.R_eval, args.seed), flush=True)
        t0 = time.time()
        r = aggregated_weights_power(b=b_val, **config)
        elapsed = time.time() - t0
        val = r["power_aggregated"]
        results[label] = val
        key_rt = label + "_runtime_s"
        results[key_rt] = round(elapsed, 1)
        results[label + "_Z_mean"] = r.get("Z_mean_full", None)
        results[label + "_Z_std"] = r.get("Z_std_full", None)
        results[label + "_avg_w_I"] = r.get("avg_w_identity", None)
        results[label + "_avg_w_J"] = r.get("avg_w_J", None)
        print("  {} = {:.4f}  (runtime {:.1f}s)".format(label, val, elapsed), flush=True)
        awi = r.get("avg_w_identity", None)
        awj = r.get("avg_w_J", None)
        if awi is not None:
            print("    avg_w_I={:.3f}, avg_w_J={:.3f}".format(awi, awj), flush=True)

    print("\n=== RESULTS ===", flush=True)
    skip_suffixes = ("_runtime_s", "_Z_mean", "_Z_std", "_avg_w_I", "_avg_w_J")
    for k, v in sorted(results.items()):
        if not any(k.endswith(s) for s in skip_suffixes):
            print("{}: {:.4f}".format(k, v), flush=True)

    results["_config"] = {k: str(v) if isinstance(v, (np.integer, np.floating)) else v
                          for k, v in config.items()}

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print("Saved to {}".format(args.output), flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
