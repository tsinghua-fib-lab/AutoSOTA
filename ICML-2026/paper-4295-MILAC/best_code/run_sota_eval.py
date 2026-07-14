#!/usr/bin/env python3
"""Parameterized MILCCI evaluation for SOTA optimization."""
import os, sys, time, json
import numpy as np

sys.path.insert(0, "/repo")
import milcci
from milcci import per_trial_r2, global_r2

CACHE_DIR = "/datasets/milcci_wikipedia"
CACHE_FULL = os.path.join(CACHE_DIR, "wiki_full_v2.npz")
SEP = "=" * 60

def main():
    # Parse CLI args: key=value pairs
    params = {
        "num_repeats": 15,
        "lambda_similarity": 100,
        "decor_A": 2,
        "factor_A": 5,
        "n_ensembles_each": "4,4,4",
        "seed": 42,
    }
    for arg in sys.argv[1:]:
        if "=" in arg:
            k, v = arg.split("=", 1)
            if k in params:
                if k == "n_ensembles_each":
                    params[k] = v
                elif k == "seed":
                    params[k] = int(v)
                else:
                    params[k] = float(v) if "." in v else int(v)

    nee = [int(x) for x in params["n_ensembles_each"].split(",")]
    n_ensembles = sum(nee)

    print(SEP, flush=True)
    print(f"MILCCI SOTA Eval: P={n_ensembles}, nee={nee}, "
          f"lr={params["num_repeats"]}, ls={params["lambda_similarity"]}, "
          f"dA={params["decor_A"]}, fA={params["factor_A"]}", flush=True)
    print(SEP, flush=True)

    # Load data
    print("Loading cached data...", flush=True)
    data = dict(np.load(CACHE_FULL, allow_pickle=True))
    Y = data["Y"]
    labels = data["labels"].tolist()
    nt = data["numbers2tuples"].item()
    numbers2tuples = {int(k): tuple(v) for k, v in nt.items()}
    print(f"  Y: {Y.shape}, trials: {len(labels)}", flush=True)

    # Run MILCCI
    t0 = time.time()
    result = milcci.fit(
        Y, labels, numbers2tuples,
        n_ensembles=n_ensembles,
        n_ensembles_each=nee,
        nu=[0.01] * n_ensembles,
        lambda_similarity=params["lambda_similarity"],
        factor_A=params["factor_A"],
        decor_A=params["decor_A"],
        num_repeats=params["num_repeats"],
        cont_axis_list=[],
        split_A=True,
        params_init_A={"ensemble_positive": True},
        verbose=True,
        seed=params["seed"],
    )
    runtime = time.time() - t0

    Phi = result["Phi"]
    A_full = result["A_full"]
    r2_vec = per_trial_r2(Y, A_full, Phi)
    r2_mean = float(np.mean(r2_vec))
    r2_std = float(np.std(r2_vec))
    r2_global = float(global_r2(Y, A_full, Phi))

    print("", flush=True)
    print(SEP, flush=True)
    print("RESULTS", flush=True)
    print(SEP, flush=True)
    print(f"Per-trial R2:  mean={r2_mean:.4f}  std={r2_std:.4f}", flush=True)
    print(f"Global R2:     {r2_global:.4f}", flush=True)
    print(f"Runtime:       {runtime:.2f}s", flush=True)
    print(f"Num iters:     {result.get(num_iters_completed, N/A)}", flush=True)
    if result.get("early_stop_reason"):
        print(f"Early stop:    {result[early_stop_reason]}", flush=True)

    # Save results
    out = {
        "r2_per_trial_mean": r2_mean,
        "r2_per_trial_std": r2_std,
        "r2_global": r2_global,
        "runtime_seconds": runtime,
        "params": params,
        "num_iters_completed": result.get("num_iters_completed"),
        "early_stop_reason": result.get("early_stop_reason"),
    }
    out_path = os.path.join(CACHE_DIR, "sota_eval_result.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}", flush=True)

if __name__ == "__main__":
    main()
