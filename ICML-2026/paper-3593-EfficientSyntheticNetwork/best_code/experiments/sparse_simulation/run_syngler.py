"""Train SyNGLER (SyNG-R + SyNG-D) on sparse-simulation inputs.

For each seed:
  1. Load the DataGenerator pickle (contains the true latents).
  2. Use them directly as the "fitted" LSM (in this sparse-sim setting
     the LSM is fit and recovered well in low dimension; for full-fit see
     ``syngler.lsm.runners.run_sim``).
  3. Run SyNG-R bootstrap and/or SyNG-D diffusion to draw n_reps new
     latents, reconstruct each into a 0/1 adjacency, and save.
"""
import argparse
import pathlib
import pickle
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def parse_seeds(spec):
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(s) for s in spec.split(",")]


def fit_and_sample(seed, r, data_root, out_root, num_samples, methods, mlp_steps):
    from syngler.lsm.source import DataGenerator  # noqa: F401  (needed for pickle)
    from syngler.res import generate_graphs as res_gen
    from syngler.diff.forest import generate_graphs as diff_forest_gen
    from syngler.diff.mlp import generate_graphs as diff_mlp_gen

    pkl_path = pathlib.Path(data_root) / f"r={r}" / f"seed={seed}.pkl"
    with open(pkl_path, "rb") as f:
        dg = pickle.load(f)["data"]
    model_Z = np.asarray(dg.Z)
    model_alpha = np.asarray(dg.alpha).flatten()
    rho = float(getattr(dg, "sparsity", 0.0))

    def dump(name, gen):
        out = pathlib.Path(out_root) / name / f"r={r}" / f"seed={seed}" / "samples"
        out.mkdir(parents=True, exist_ok=True)
        for k, A in enumerate(gen):
            np.save(out / f"rep{k}.npy", A.astype(np.uint8))
        print(f"  {name} r={r} seed={seed}: {num_samples} samples -> {out}")

    if "res" in methods:
        dump("syngr", res_gen(model_Z, model_alpha, n_reps=num_samples, rho=rho, seed=seed))
    if "diff" in methods:
        dump("syngd",
             diff_forest_gen(model_Z, model_alpha, n_reps=num_samples, rho=rho, seed=seed))
    if "mlp" in methods:
        dump("syngd_mlp",
             diff_mlp_gen(model_Z, model_alpha, n_reps=num_samples, rho=rho, seed=seed,
                          n_steps=mlp_steps, device="cuda"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=int, required=True)
    ap.add_argument("--seeds", default="0-19", help="comma-list or A-B range")
    ap.add_argument("--data_root", default="data/sparse_sim")
    ap.add_argument("--out_root", default="runs")
    ap.add_argument("--num_samples", type=int, default=200)
    ap.add_argument("--methods", default="res,diff",
                    help="comma-separated: res (SyNG-R), diff (SyNG-D forest), mlp (SyNG-D MLP)")
    ap.add_argument("--mlp_steps", type=int, default=30_000)
    args = ap.parse_args()

    seeds = parse_seeds(args.seeds)
    methods = set(args.methods.split(","))
    for s in seeds:
        fit_and_sample(s, args.r, args.data_root, args.out_root,
                       args.num_samples, methods, args.mlp_steps)


if __name__ == "__main__":
    main()
