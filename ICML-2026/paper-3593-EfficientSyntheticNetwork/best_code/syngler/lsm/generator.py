import os
from pathlib import Path
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()
os.chdir(BASE_DIR)
import csv
import json
import pickle
import numpy as np
from absl import flags, app
import sys
from tqdm import tqdm


from LSM_source import (
    sigmoid, symmetrization, UniformCovariateSampler,
    ClippedGaussianCovariateSampler, DataGenerator,
    var_phi_functional_primary, var_beta_functional, bias_est_functional
)
import LSM_source as LSM

FLAGS = flags.FLAGS
flags.DEFINE_string("config", "../config/default.json", "Path to the configuration file")
flags.DEFINE_string("grid_file", "../config/generate_n=500_r=2_sparse=0.0.csv", "CSV with columns n,r[,seed]")

def ClippedGaussianMixture(n, r):
    sample = ClippedGaussianCovariateSampler(
        n, mu=np.zeros((r)), Sigma=np.eye(r), low=-2, up=2
    ) / np.sqrt(r)
    v1 = np.random.uniform(-1, 1, size=r)
    v2 = np.random.uniform(-1, 1, size=r)
    mask = np.random.rand(n) < 0.5
    sample += np.where(mask[:, None], v1, v2)
    sample = sample / np.sqrt(np.linalg.norm(sample @ sample.T, "fro") / n)
    return sample

def run_one(config, n, r, seed,sparse_level):
    print(n,np.log(n),sparse_level)
    p = config["p"]
    alpha_enable = config["alpha_enable"]
    Z_enable = config["Z_enable"]
    print(type(n))
    np.random.seed(seed)
    rho = -np.log(n) * sparse_level

    beta_true = np.zeros(p)
    X = np.zeros((n, n, p))
    X = symmetrization(X)

    data = DataGenerator(
        beta_true, X,
        Z_enable=Z_enable, alpha_enable=alpha_enable,
        act=sigmoid, sparsity=rho
    )
    data.RefreshLatentVar(
        lambda n_: ClippedGaussianMixture(n_, r),
        lambda n_: UniformCovariateSampler(n_, 1, -0.5, 0.5),
        Z_standardize=True
    )

    eta_0 = 0.5
    eta_alpha = eta_0 / (2 * data.n) if alpha_enable else 0
    eta_Z = eta_0 / (2 * np.sum(data.Z ** 2) / data.Z.shape[1]) if Z_enable else 0
    eta_beta = 0
    eta_adj = np.mean([eta_alpha, eta_Z])

    X_adjusted = data.X - LSM.adjustment_functional(data, lr=eta_adj)
    var_phi_oracle = var_phi_functional_primary(data)

    save_dir = f"../../datasets/simulation/generator/n={n}_r={r}_sparse={sparse_level}/"
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(
        save_dir,
        f"seed={seed}.pkl"
    )
    with open(out_path, "wb") as f:
        pickle.dump({
            "data": data,
            "var_phi_oracle": var_phi_oracle,
            "beta_true": beta_true,
        }, f)
    print(f"[Saved] {out_path}")

def main(_):
    with open(FLAGS.config, "r") as f:
        config = json.load(f)
    rows = []
    with open(FLAGS.grid_file, "r", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            try:
                n = int(row["n"])
                r = int(row["r"])
                sparse_level = float(row.get("sparse_level", 0.0))
            except Exception as e:
                raise ValueError(f"Missing/Invalid n/r in csv row {i+2}") from e
            seed = row.get("seed")
            seed = int(seed) if (seed is not None and seed != "") else (config.get("seed", 2025) + i)
            rows.append((n, r, seed))

    for (n, r, seed) in tqdm(rows, desc=f"Generating:", unit="file"):
        run_one(config, n, r, seed, sparse_level)
    print("All data generation complete.")
if __name__ == "__main__":
    app.run(main)
