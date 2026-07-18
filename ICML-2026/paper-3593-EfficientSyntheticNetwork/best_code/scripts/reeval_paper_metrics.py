"""Re-evaluate saved samples with the paper's exact metric definitions.

Paper metrics (from syngler.evaluation.metrics):
  - triangle_density        -> "Tri."  : RMSE / MAE / Bias of (gen - ref)
  - global_clustering_coeff -> "Clus." : RMSE / MAE / Bias
  - degree_centrality       -> "Deg."  : W1 / KS / Energy (pooled ref vs gen)
  - eigenvalues (Laplacian) -> "Eig."  : W1 / KS / Energy (pooled ref vs gen)

Reads saved samples without retraining. Writes one JSON per (method, r,
seed) to ``--out_dir/r=<r>/<method>_seed<seed>.json``.
"""
import argparse
import json
import os
import pathlib
import pickle
import sys
import time

import numpy as np
import torch
from scipy.stats import ks_2samp, wasserstein_distance

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from syngler.evaluation.metrics import (  # noqa: E402
    degree_centrality, eigenvalues, energy_distance,
    global_clustering_coefficient, triangle_density,
)


def to_np(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def load_ref_A(data_root, r, seed, n=500):
    """Sample one binary adjacency from the LSM-pickle's P matrix."""
    pkl = pathlib.Path(data_root) / f"r={r}" / f"seed={seed}.pkl"
    with open(pkl, "rb") as f:
        dg = pickle.load(f)["data"]
    rng = np.random.RandomState(seed)
    tril = np.tril(np.ones((n, n), dtype=bool), k=-1)
    A = np.zeros((n, n), dtype=np.float32)
    A[tril] = (rng.rand(*dg.P[tril].shape) < dg.P[tril]).astype(np.float32)
    return A + A.T


def load_gen_batch(samples_root, method, r, seed):
    sdir = pathlib.Path(samples_root) / method / f"r={r}" / f"seed={seed}" / "samples"
    if not sdir.is_dir():
        return None
    files = sorted(p for p in sdir.iterdir() if p.suffix == ".npy")
    if not files:
        return None
    arrs = []
    for f in files:
        A = np.load(f).astype(np.float32)
        if method == "gran" and A.max() <= 1.0 and A.min() >= 0.0:
            # GRAN saves probability matrices; threshold-sample.
            idx = int(f.stem.replace("rep", ""))
            rng = np.random.RandomState(idx)
            A = (rng.rand(*A.shape) < A).astype(np.float32)
            np.fill_diagonal(A, 0)
            A = np.tril(A, -1)
            A = A + A.T
        arrs.append(A)
    return np.stack(arrs)


def dist_triplet(ref, gen):
    ref = np.asarray(ref, float).ravel()
    gen = np.asarray(gen, float).ravel()
    return {
        "w1": float(wasserstein_distance(ref, gen)),
        "ks": float(ks_2samp(ref, gen).statistic),
        "energy": float(energy_distance(ref, gen)),
    }


def eval_one(method, r, seed, data_root, samples_root, out_dir, force):
    out_path = pathlib.Path(out_dir) / f"r={r}" / f"{method}_seed{seed}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if (not force) and out_path.exists():
        return "skip(exists)"

    gen = load_gen_batch(samples_root, method, r, seed)
    if gen is None:
        return "skip(no samples)"
    A_ref = load_ref_A(data_root, r, seed)

    ref_t = torch.from_numpy(A_ref).unsqueeze(0)
    gen_t = torch.from_numpy(gen)

    ref_td = float(to_np(triangle_density(ref_t, device="cpu")).ravel()[0])
    gen_td = to_np(triangle_density(gen_t, device="cpu")).ravel()
    ref_gcc = float(to_np(global_clustering_coefficient(ref_t, device="cpu")).ravel()[0])
    gen_gcc = to_np(global_clustering_coefficient(gen_t, device="cpu")).ravel()

    td_diff = gen_td - ref_td
    gcc_diff = gen_gcc - ref_gcc

    ref_degc = to_np(degree_centrality(ref_t, device="cpu")).ravel()
    gen_degc = to_np(degree_centrality(gen_t, device="cpu")).ravel()
    ref_eig = to_np(eigenvalues(ref_t, device="cpu")).ravel()
    gen_eig = to_np(eigenvalues(gen_t, device="cpu")).ravel()

    deg = dist_triplet(ref_degc, gen_degc)
    eig = dist_triplet(ref_eig, gen_eig)

    res = {
        "r": r, "seed": seed, "method": method.upper(), "n_samples": int(gen.shape[0]),
        "ref_edges": int(A_ref.sum() // 2),
        "gen_edges_mean": float(gen.sum(axis=(1, 2)).mean() / 2),
        "tri_rmse": float(np.sqrt(np.mean(td_diff ** 2))),
        "tri_mae": float(np.mean(np.abs(td_diff))),
        "tri_bias": float(np.mean(td_diff)),
        "gcc_rmse": float(np.sqrt(np.mean(gcc_diff ** 2))),
        "gcc_mae": float(np.mean(np.abs(gcc_diff))),
        "gcc_bias": float(np.mean(gcc_diff)),
        "deg_w1": deg["w1"], "deg_ks": deg["ks"], "deg_energy": deg["energy"],
        "eig_w1": eig["w1"], "eig_ks": eig["ks"], "eig_energy": eig["energy"],
    }
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    return (f"OK tri={res['tri_rmse']:.4g} gcc={res['gcc_rmse']:.4g} "
            f"deg_w1={res['deg_w1']:.4g} eig_w1={res['eig_w1']:.4g}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["gran", "edge", "vgae", "syngr", "syngd", "syngd_mlp"])
    ap.add_argument("--r", type=int, required=True)
    ap.add_argument("--seeds", required=True, help="comma-separated seed list")
    ap.add_argument("--data_root", default="data/sparse_sim",
                    help="dir containing r=<r>/seed=<S>.pkl (sparse-sim reference graphs)")
    ap.add_argument("--samples_root", default="runs",
                    help="dir containing <method>/r=<r>/seed=<S>/samples/rep*.npy")
    ap.add_argument("--out_dir", default="runs/eval_paper")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    t0 = time.time()
    for s in seeds:
        ts = time.time()
        msg = eval_one(args.method, args.r, s, args.data_root, args.samples_root,
                       args.out_dir, args.force)
        print(f"r={args.r} {args.method} seed={s}: {msg} ({time.time()-ts:.0f}s)", flush=True)
    print(f"Total {time.time()-t0:.0f}s for {len(seeds)} seeds", flush=True)


if __name__ == "__main__":
    main()
