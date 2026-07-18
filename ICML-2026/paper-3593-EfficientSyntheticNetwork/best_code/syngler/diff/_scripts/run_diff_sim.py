""" 
Please specify the config csv path for running. For example:
    python sampler.py --grid_csv ../../Latent-Space-Model/config/run_n={n}_r={r}_sparse={sparse_level}.csv
"""
import os
from pathlib import Path

import csv
import pickle
import numpy as np
from tqdm import tqdm
import argparse
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from ForestDiffusion import ForestDiffusionModel as ForestFlowModel

GEN_BASE = "../../datasets/run"
SAVE_BASE = "../../synthetic/simulation/Diff-sample"
N_T = 50
DUP_K = 100
SAMPLES_PER_DATA = 200
XGB_KW = dict(
    max_depth=7,
    n_estimators=100,
    eta=0.3,
    tree_method="hist",
    reg_lambda=0.0,
    reg_alpha=0.0,
    subsample=1.0,
    n_jobs=-1,
)

def load_run_pkl(n, r, sparse_level, seed):
    dir_ = os.path.join(GEN_BASE, f"n={n}_r={r}_sparse={sparse_level}")
    fn = os.path.join(dir_, f"seed={seed}.pkl")
    with open(fn, "rb") as f:
        results = pickle.load(f)
    Z = np.array(results["model_Z"])
    alpha = np.array(results["model_alpha"]).reshape(-1, 1)
    X = np.hstack([Z, alpha])
    return X

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def main(grid_csv):
    jobs = []
    with open(grid_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n = int(row["n"])
            r = int(row["r"])
            seed = int(row["seed"])
            sparse_level = float(row.get("sparse_level", 0.0))
            jobs.append((n, r, seed, sparse_level))

    for (n, r, seed, sparse_level) in tqdm(jobs, desc="Datasets", unit="job"):
        X = load_run_pkl(n, r, sparse_level, seed)
        y_dummy = np.zeros(X.shape[0])

        forest_model = ForestFlowModel(
            X,
            label_y=y_dummy,
            n_t=N_T,
            duplicate_K=DUP_K,
            bin_indexes=[],
            cat_indexes=[],
            int_indexes=[],
            diffusion_type="vp",
            seed=seed,
            **XGB_KW,
        )

        out_dir = os.path.join(
            SAVE_BASE,
            f"n={n}_r={r}_sparse={sparse_level}/seed={seed}"
        )
        ensure_dir(out_dir)

        Xy_fake_all = forest_model.generate(batch_size=X.shape[0] * SAMPLES_PER_DATA)
        for b_id in range(SAMPLES_PER_DATA):
            start = b_id * X.shape[0]
            end = (b_id + 1) * X.shape[0]
            Xy_fake_ = Xy_fake_all[start:end, :]

            x_fake = Xy_fake_[:, :-1]  # drop dummy label
            Z_fake = x_fake[:, :r]
            alpha_fake = x_fake[:, r:r+1]

            np.savez(
                os.path.join(out_dir, f"rep{b_id}.npz"),
                Z=Z_fake,
                alpha=alpha_fake
            )

        tqdm.write(f"[n={n}, r={r}, seed={seed}] saved {SAMPLES_PER_DATA} reps to: {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid_csv", type=str, required=True,
                        help="Path to CSV file containing n,r,seed,sparse_level")
    args = parser.parse_args()
    main(args.grid_csv)
