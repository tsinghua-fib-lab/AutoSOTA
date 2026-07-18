"""Train SyNGLER (SyNG-R + SyNG-D) on one real dataset.

Inputs:
  data/real/<dataset>/generator/seed=0.npy   — adjacency matrix
  (the LSM is fit on the fly via syngler.lsm.runners.run_real; or you can
  pass --fitted_pkl <pkl> with pre-fitted (Z, alpha, rho))
"""
import argparse
import pathlib
import pickle
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def load_or_fit(dataset, r, data_root, fitted_pkl):
    if fitted_pkl is not None:
        with open(fitted_pkl, "rb") as f:
            d = pickle.load(f)
        return (np.asarray(d["model_Z"]),
                np.asarray(d["model_alpha"]).flatten(),
                float(d.get("model_sparsity", 0.0)))
    raise SystemExit(
        "No --fitted_pkl given. To fit the LSM from scratch, run "
        "`python -m syngler.lsm.runners.run_real --dataset {} --r {} --out <pkl>` first.".format(
            dataset, r))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    choices=["dblp", "yelp", "youtube", "polblogs"])
    ap.add_argument("--r", type=int, default=6, help="latent dimension; paper default = 6")
    ap.add_argument("--data_root", default="data/real")
    ap.add_argument("--fitted_pkl", default=None,
                    help="pre-fit LSM (Z, alpha, rho) pickle; skip if fitting from scratch")
    ap.add_argument("--output", required=True)
    ap.add_argument("--num_samples", type=int, default=200)
    ap.add_argument("--methods", default="res,diff",
                    help="comma-separated: res, diff (forest), mlp")
    ap.add_argument("--mlp_steps", type=int, default=30_000)
    args = ap.parse_args()

    Z, alpha, rho = load_or_fit(args.dataset, args.r, args.data_root, args.fitted_pkl)
    out = pathlib.Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    methods = set(args.methods.split(","))
    if "res" in methods:
        from syngler.res import generate_graphs
        d = out / "syngr" / "samples"; d.mkdir(parents=True, exist_ok=True)
        for k, A in enumerate(generate_graphs(Z, alpha, n_reps=args.num_samples, rho=rho, seed=0)):
            np.save(d / f"rep{k}.npy", A.astype(np.uint8))
        print(f"SyNG-R ({args.dataset}): {args.num_samples} samples -> {d}")
    if "diff" in methods:
        from syngler.diff.forest import generate_graphs as diff_gen
        d = out / "syngd" / "samples"; d.mkdir(parents=True, exist_ok=True)
        for k, A in enumerate(diff_gen(Z, alpha, n_reps=args.num_samples, rho=rho, seed=0)):
            np.save(d / f"rep{k}.npy", A.astype(np.uint8))
        print(f"SyNG-D ({args.dataset}): {args.num_samples} samples -> {d}")
    if "mlp" in methods:
        from syngler.diff.mlp import generate_graphs as mlp_gen
        d = out / "syngd_mlp" / "samples"; d.mkdir(parents=True, exist_ok=True)
        for k, A in enumerate(mlp_gen(Z, alpha, n_reps=args.num_samples, rho=rho, seed=0,
                                      n_steps=args.mlp_steps, device="cuda")):
            np.save(d / f"rep{k}.npy", A.astype(np.uint8))
        print(f"SyNG-D-MLP ({args.dataset}): {args.num_samples} samples -> {d}")


if __name__ == "__main__":
    main()
