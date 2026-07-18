import os
import csv
import json
import pickle
import numpy as np
from absl import app, flags
from tqdm import tqdm

from syngler.lsm import source as LSM
from syngler.lsm.source import sigmoid, MatrixBernoilliSampler

FLAGS = flags.FLAGS
flags.DEFINE_string("config", "../config/default.json", "Path to the config file")
flags.DEFINE_string("grid_file", "../config/run_n=500_r=2_sparse=0.0.csv", "CSV with columns: n,r,seed,eta_0[,tau,rep_index]")
flags.DEFINE_string("generator_dir", "../../datasets/simulation/generator/", "Directory where *.pkl are stored")
flags.DEFINE_string("run_dir", "../../datasets/simulation/run/", "Base directory to save run results when looping")
flags.DEFINE_float("tau", 0.0, "Default correlation level if CSV has no tau")
flags.DEFINE_float("sigma_init", 0.1, "Noise std used for init")

def load_generated_data(generator_dir, n, r, p, seed, sparse_level):
    fn = f"n={n}_r={r}_sparse={sparse_level}/seed={seed}.pkl"
    filename = os.path.join(generator_dir, fn)
    with open(filename, "rb") as f:
        loader = pickle.load(f)
    return loader

def run_one(config, row, generator_dir, run_dir, sigma_init=0.1):
    try:
        n = int(row["n"])
        r = int(row["r"])
        seed = int(row["seed"])
        sparse_level = float(row["sparse_level"]) if "sparse_level" in row else config.get("sparse_level", 0.0)
        eta_0 = float(row["eta_0"])

    except Exception as e:
        raise ValueError(f"Error: {row}") from e

    rep_index = seed
    p = config["p"]
    alpha_enable = config["alpha_enable"]
    Z_enable = config["Z_enable"]
    CI_enable = config["CI_enable"]
    Z_standardize = config["Z_standardize"]
    sparsity_estimation = config["sparsity_estimation"]

    if sparsity_estimation:
        rho = -np.log(n) * 0.0
    else:
        rho = -np.log(n) * sparse_level
    print(f"Running with n={n}, r={r}, sparse_level={sparse_level}, seed={seed}")

    loader = load_generated_data(generator_dir, n, r, p, seed, sparse_level)
    data = loader["data"]
    beta_true = loader["beta_true"]
    var_phi_oracle = loader["var_phi_oracle"]

    eta_alpha = eta_0 / (2 * data.n) if alpha_enable else 0
    eta_Z = eta_0 / (2 * np.sum(data.Z ** 2) / data.Z.shape[1]) if Z_enable else 0
    eta_beta = 0
    eta_adj = np.min([eta_alpha, eta_Z]) if (eta_alpha and eta_Z) else max(eta_alpha, eta_Z)

    init_Z = data.Z + sigma_init * np.random.randn(*data.Z.shape) if Z_enable else None

    np.random.seed(rep_index)
    A = data.DataInstance(MatrixBernoilliSampler)

    model = LSM.Model(
        A, data.X,
        alpha=data.alpha.reshape(-1) + sigma_init * np.random.randn(data.n),
        beta=data.beta,
        Z=init_Z,
        alpha_enable=alpha_enable,
        Z_enable=Z_enable,
        Z_standardize=Z_standardize,
        act=sigmoid, sparsity=rho,
        sparsity_estimation=sparsity_estimation,
    )
    model.PGD(eta_alpha=eta_alpha, eta_beta=eta_beta, eta_Z=eta_Z, early_stop=True, eps=1e-6, n_iter=50000)

    results = {
        "model_Z": model.Z,
        "model_alpha": model.alpha,
        "alpha_1_est_error": (model.alpha - data.alpha)[0] if alpha_enable else None,
        "Z_11_est_error": (model.Z - data.Z)[0, 0] if Z_enable else None,
        "beta_1_est_error": (model.beta - data.beta)[0],
        "theta_12_est_error": (model.P - data.P)[0, 1],
        "sparsity_est_error": (model.sparsity - rho) ** 2,
        "Z_overall_est_error": LSM.matched_error(model.Z, data.Z),
        "alpha_overall_est_error": np.linalg.norm(model.alpha - data.alpha) ** 2 / n,
        "converged": model.converged
    }

    if CI_enable:
        X_adjusted = model.X - LSM.adjustment_functional(model, lr=eta_adj)
        var_phi_est = LSM.var_phi_functional_primary(model)

        h = LSM.H_functional(model)
        h_21 = np.hstack([h[1, :], h[0, :]])
        dim = r + 1
        var_12_node = np.zeros((2 * dim, 2 * dim))
        var_12_node[:dim, :dim] = var_phi_est[0]
        var_12_node[dim:, dim:] = var_phi_est[1]
        var_12_link = h_21 @ var_12_node @ h_21.T * (model.P[0, 1] * (1 - model.P[0, 1])) ** 2

        alpha_se = np.sqrt(var_phi_est[0, -1, -1])
        cover_alpha_all = np.abs(model.alpha - data.alpha) < 1.96 * alpha_se

        results |= {
            "cover_z_11": np.abs((model.Z - data.Z))[0, 0] < 1.96 * np.sqrt(var_phi_est[0, 0, 0]),
            "cover_alpha_1": np.abs(model.alpha - data.alpha)[0] < 1.96 * np.sqrt(var_phi_est[0, -1, -1]),
            "cover_alpha_rate": np.mean(cover_alpha_all),
            "cover_theta_12": np.abs((model.P - data.P))[0, 1] < 1.96 * np.sqrt(var_12_link),
            "var_z11_est_error": np.abs(var_phi_est[0, 0, 0] - var_phi_oracle[0, 0, 0]),
            "var_alpha_1_est_error": np.abs(var_phi_est[0, -1, -1] - var_phi_oracle[0, -1, -1]),
        }

    savedir = os.path.join(run_dir, f"n={n}_r={r}_sparse={sparse_level}")
    os.makedirs(savedir, exist_ok=True)

    out_fn = os.path.join(savedir, f"seed={rep_index}.pkl")
    with open(out_fn, "wb") as f:
        pickle.dump(results, f)
    return model

def main(_):
    with open(FLAGS.config, "r") as f:
        config = json.load(f)
    rows = []
    with open(FLAGS.grid_file, "r", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            rows.append(row)

    for row in tqdm(rows, desc="Running PGD over CSV rows", unit="run"):
        model = run_one(
            config=config,
            row=row,
            generator_dir=FLAGS.generator_dir,
            run_dir=FLAGS.run_dir,
            sigma_init=FLAGS.sigma_init
        )

if __name__ == "__main__":
    app.run(main)
