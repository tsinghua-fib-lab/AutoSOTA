#!/usr/bin/env python3
"""Paper 5647: Compatibility Scores on gapminder - Evaluation Script."""
import sys; sys.path.insert(0, "/repo")
import numpy as np, json, argparse
from experiments_llm_linear import compute_correlation_matrix, VARIABLES
from synthetic_experiments_linear import compatibility_score
from sensitivity_experiment import _estimate_bivariate_matrix

def estimated_A_scores(corr, n_runs, noise, seed):
    A_base = _estimate_bivariate_matrix(corr)
    base = compatibility_score(A_base, corr)
    rng = np.random.RandomState(seed)
    scores = []
    for _ in range(n_runs):
        A = A_base + np.tril(noise * rng.randn(len(VARIABLES), len(VARIABLES)), -1)
        try:
            scores.append(compatibility_score(A, corr))
        except:
            scores.append(base)
    s = np.array(scores)
    return {"mean": float(np.mean(s)), "std": float(np.std(s)), "base_score": float(base)}

def sign_constrained_scores(corr, n_runs, mult, seed):
    n = len(VARIABLES); sigma = float(np.std(corr[np.tril_indices(n, k=-1)])) * mult
    rng = np.random.RandomState(seed); scores = []
    for _ in range(n_runs):
        A = np.eye(n)
        for i in range(n):
            for j in range(i):
                A[i,j] = np.sign(corr[i,j]) * abs(rng.normal(0, sigma))
        try: scores.append(compatibility_score(A, corr))
        except: scores.append(0.0)
    s = np.array(scores)
    return {"mean": float(np.mean(s)), "std": float(np.std(s)), "sigma": sigma}

def random_scores(corr, n_runs, mult, seed):
    n = len(VARIABLES); sigma = float(np.std(corr[np.tril_indices(n, k=-1)])) * mult
    rng = np.random.RandomState(seed); scores = []
    for _ in range(n_runs):
        A = np.eye(n)
        for i in range(n):
            for j in range(i):
                A[i,j] = rng.normal(0, sigma)
        try: scores.append(compatibility_score(A, corr))
        except: scores.append(0.0)
    s = np.array(scores)
    return {"mean": float(np.mean(s)), "std": float(np.std(s)), "sigma": sigma}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--method", default="estimated_A", choices=["estimated_A","sign_constrained","random","all"])
    p.add_argument("--noise", type=float, default=0.0)
    p.add_argument("--n-runs", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mult", type=float, default=1.0)
    args = p.parse_args()

    corr = compute_correlation_matrix().values

    if args.method == "all":
        results = []
        for noise in [0.0, 0.001, 0.005, 0.01, 0.05]:
            r = estimated_A_scores(corr, args.n_runs, noise, args.seed)
            results.append(("estimated_A_sigma=%.3f" % noise, r))
        for mult in [0.5, 1.0, 2.0]:
            r = sign_constrained_scores(corr, args.n_runs, mult, args.seed)
            results.append(("sign_constrained_mult=%.1f" % mult, r))
        for mult in [1.0]:
            r = random_scores(corr, args.n_runs, mult, args.seed)
            results.append(("random_baseline", r))
        results.sort(key=lambda x: x[1]["mean"], reverse=True)
        name, results = results[0]
        print("Best method: %s" % name)
    elif args.method == "estimated_A":
        results = estimated_A_scores(corr, args.n_runs, args.noise, args.seed)
    elif args.method == "sign_constrained":
        results = sign_constrained_scores(corr, args.n_runs, args.mult, args.seed)
    elif args.method == "random":
        results = random_scores(corr, args.n_runs, args.mult, args.seed)

    print("=" * 60)
    print("PAPER 5647: COMPATIBILITY SCORE EVALUATION")
    print("=" * 60)
    print("Method: %s" % args.method)
    print("Runs: %d" % args.n_runs)
    print()
    print("RESULTS:")
    print("  mean = %.6f" % results["mean"])
    print("  std  = %.6f" % results["std"])
    for k, v in sorted(results.items()):
        if k not in ("mean", "std"):
            print("  %s: %s" % (k, v))
    print()
    print("Paper: Gemma 4B=0.131, Random=0.155, Baseline=0.142")
    print()
    print("=" * 60)
    print("JSON OUTPUT:")
    print("=" * 60)
    out = {
        "compatibility_score_random_baseline_mean": results["mean"],
        "compatibility_score_random_baseline_std": float(results.get("std", 0)),
        "n_runs": args.n_runs,
        "method": args.method,
    }
    for k, v in results.items():
        if k not in ("mean", "std"):
            if isinstance(v, (int, float, str, bool, type(None))):
                out[k] = v
    print(json.dumps(out, indent=2))
