#!/usr/bin/env python3
"""Full reproduction pipeline for paper 3566 (Rank-Learner)."""
import sys, os
from pathlib import Path

# Redirect stderr to suppress tqdm output
sys.stderr = open(os.devnull, "w")

sys.path.insert(0, str(Path("/repo/library")))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data_utils import *
from models import *
from training import *
from eval_utils import *

TRAIN_SIZE = 500
N_SEEDS = 5
DATASET = "synthetic"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFOUNDERS = ["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9"]
INPUT_DIM = len(CONFOUNDERS)
ROOT = Path("/repo")
DATA_DIR = ROOT / "data" / "datasets"
NUISANCE_CHKPT_DIR = ROOT / "experiments" / "nuisances" / "chkpts" / DATASET
POINTWISE_CHKPT_DIR = ROOT / "experiments" / "pointwise" / "chkpts" / DATASET
RANKER_CHKPT_DIR = ROOT / "experiments" / "rankers" / "chkpts" / DATASET

def log(msg):
    sys.__stderr__.write(msg + "\n")
    sys.__stderr__.flush()

log(f"Device: {DEVICE}, Train size: {TRAIN_SIZE}, Seeds: {N_SEEDS}")

# --- Step 1: Generate data ---
log("=== Step 1: Generate synthetic data ===")
data_path = DATA_DIR / f"{DATASET}.csv"
if not data_path.exists():
    def gen_synthetic(n, seed=0):
        np.random.seed(seed)
        p = 10
        X = np.random.multivariate_normal(np.zeros(p), np.eye(p), size=n)
        s = 0.8 * X[:, 0] + 0.6 * X[:, 1] + 0.4 * X[:, 2] + 0.3 * X[:, 0]**2 - 0.2 * X[:, 1] * X[:, 2]
        v = X[:, 7] - 0.5 * X[:, 8]
        u = 0.8 * s + 0.6 * v
        e = 0.2 + 0.6 / (1 + np.exp(-u))
        T = np.random.binomial(1, e)
        M0 = 0.5 * X[:, 3] - 0.4 * X[:, 4] + 0.3 * np.sin(X[:, 5]) + 0.2 * (X[:, 6]**2 - 1)
        tau = s + 0.5 * np.tanh(s)
        M1 = M0 + tau
        eps = np.random.normal(0, 0.6, size=n)
        Y = M0 + T * tau + eps
        df = pd.DataFrame(X, columns=[f"x{i}" for i in range(p)])
        df["T"], df["Y"], df["M0"], df["M1"] = T, Y, M0, M1
        df["cate"], df["e"], df["s"], df["v"] = tau, e, s, v
        return df
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df_full = gen_synthetic(11000, seed=0)
    df_full.to_csv(data_path)
    log(f"Generated {len(df_full)} samples")
else:
    log("Data exists")

df = pd.read_csv(data_path, index_col=0)
log(f"Loaded: {df.shape}")

# --- Step 2: Nuisance models ---
log("=== Step 2: Nuisance models ===")
NUISANCE_CONFIGS = {
    "e":  {"hidden_dim": 128, "lr": 0.0005, "weight_decay": 1e-05, "batch_size": 128},
    "m0": {"hidden_dim": 64,  "lr": 0.001,  "weight_decay": 0.0001, "batch_size": 128},
    "m1": {"hidden_dim": 64,  "lr": 0.001,  "weight_decay": 1e-05, "batch_size": 128},
}

for nuisance in ["e", "m0", "m1"]:
    params = NUISANCE_CONFIGS[nuisance]
    for seed in range(N_SEEDS):
        ckpt_dir = NUISANCE_CHKPT_DIR / f"size_{TRAIN_SIZE}" / f"seed_{seed}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        if nuisance == "e":
            filename = "prop_model.pt"
        else:
            filename = f"mu{0 if nuisance==m0 else 1}_model.pt"
        if (ckpt_dir / filename).exists():
            continue

        set_seed(seed)
        train_df, val_df, _, _, _ = make_splits(df=df, train_size=TRAIN_SIZE, seed=seed)

        if nuisance == "e":
            train_loader, val_loader = make_nuisance_loaders(train_df, val_df, CONFOUNDERS, params["batch_size"])
            model = ClassificationHead(input_dim=INPUT_DIM, hidden_dim=params["hidden_dim"]).to(DEVICE)
            model, info = train_propensity(model, train_loader, val_loader, DEVICE,
                                           lr=params["lr"], weight_decay=params["weight_decay"],
                                           max_epochs=50, patience=5, seed=seed)
        else:
            tv = 0 if nuisance == "m0" else 1
            train = train_df[train_df["T"] == tv]
            val = val_df[val_df["T"] == tv]
            train_loader, val_loader = make_nuisance_loaders(train, val, CONFOUNDERS, params["batch_size"])
            model = RegressionHead(input_dim=INPUT_DIM, hidden_dim=params["hidden_dim"]).to(DEVICE)
            model, info = train_response(model, train_loader, val_loader, DEVICE,
                                         lr=params["lr"], weight_decay=params["weight_decay"],
                                         max_epochs=50, patience=5, seed=seed)
        torch.save(model.state_dict(), ckpt_dir / filename)
        log(f"  [{nuisance} s{seed}] val={info.get(val_loss,0):.6f}")

# --- Step 3: DR-learner (pointwise) ---
log("=== Step 3: DR-learner ===")
PTW_PARAMS = {"hidden_dim": 128, "lr": 0.0005, "weight_decay": 0.0001, "batch_size": 128}

for seed in range(N_SEEDS):
    ckpt_dir = POINTWISE_CHKPT_DIR / f"size_{TRAIN_SIZE}" / f"seed_{seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if (ckpt_dir / "cate_model.pt").exists():
        continue

    set_seed(seed)
    _, _, train_df, val_df, _ = make_splits(df=df, train_size=TRAIN_SIZE, seed=seed)

    ns_seed_dir = NUISANCE_CHKPT_DIR / f"size_{TRAIN_SIZE}" / f"seed_{seed}"
    prop_model = ClassificationHead(input_dim=INPUT_DIM, hidden_dim=128).to(DEVICE)
    prop_model.load_state_dict(torch.load(ns_seed_dir / "prop_model.pt", map_location=DEVICE, weights_only=True))
    m0_model = RegressionHead(input_dim=INPUT_DIM, hidden_dim=64).to(DEVICE)
    m0_model.load_state_dict(torch.load(ns_seed_dir / "mu0_model.pt", map_location=DEVICE, weights_only=True))
    m1_model = RegressionHead(input_dim=INPUT_DIM, hidden_dim=64).to(DEVICE)
    m1_model.load_state_dict(torch.load(ns_seed_dir / "mu1_model.pt", map_location=DEVICE, weights_only=True))

    train_df = compute_dr_scores(train_df, CONFOUNDERS, prop_model, m0_model, m1_model, DEVICE)
    val_df = compute_dr_scores(val_df, CONFOUNDERS, prop_model, m0_model, m1_model, DEVICE)

    train_loader, val_loader = make_cate_loaders(train_df, val_df, CONFOUNDERS, batch_size=PTW_PARAMS["batch_size"])
    cate_model = RegressionHead(input_dim=INPUT_DIM, hidden_dim=PTW_PARAMS["hidden_dim"]).to(DEVICE)
    cate_model, info = train_cate(cate_model, train_loader, val_loader, DEVICE,
                                  lr=PTW_PARAMS["lr"], weight_decay=PTW_PARAMS["weight_decay"],
                                  max_epochs=50, patience=5, seed=seed)
    torch.save(cate_model.state_dict(), ckpt_dir / "cate_model.pt")
    log(f"  [DR seed {seed}] val={info[val_loss]:.6f}")

# --- Step 4: Rankers ---
log("=== Step 4: Rankers ===")
ORTH_PARAMS = {"hidden_dim": 128, "lr": 0.001, "weight_decay": 1e-05, "batch_size": 256, "kappa": 0.5}
PLUGIN_PARAMS = {"hidden_dim": 128, "lr": 0.001, "weight_decay": 1e-05, "batch_size": 256, "kappa": 3.0}

for plug_in, params, method_name in [
    (False, ORTH_PARAMS, "orthogonal"),
    (True,  PLUGIN_PARAMS, "plug_in"),
]:
    for seed in range(N_SEEDS):
        ckpt_dir = RANKER_CHKPT_DIR / f"size_{TRAIN_SIZE}" / f"seed_{seed}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        if (ckpt_dir / f"{method_name}.pt").exists():
            continue

        set_seed(seed)
        _, _, train_df, val_df, _ = make_splits(df=df, train_size=TRAIN_SIZE, seed=seed)

        ns_seed_dir = NUISANCE_CHKPT_DIR / f"size_{TRAIN_SIZE}" / f"seed_{seed}"
        prop_model = ClassificationHead(input_dim=INPUT_DIM, hidden_dim=128).to(DEVICE)
        prop_model.load_state_dict(torch.load(ns_seed_dir / "prop_model.pt", map_location=DEVICE, weights_only=True))
        m0_model = RegressionHead(input_dim=INPUT_DIM, hidden_dim=64).to(DEVICE)
        m0_model.load_state_dict(torch.load(ns_seed_dir / "mu0_model.pt", map_location=DEVICE, weights_only=True))
        m1_model = RegressionHead(input_dim=INPUT_DIM, hidden_dim=64).to(DEVICE)
        m1_model.load_state_dict(torch.load(ns_seed_dir / "mu1_model.pt", map_location=DEVICE, weights_only=True))

        train_df = compute_dr_scores(train_df, CONFOUNDERS, prop_model, m0_model, m1_model, DEVICE)
        val_df = compute_dr_scores(val_df, CONFOUNDERS, prop_model, m0_model, m1_model, DEVICE)

        train_loader, _ = make_ranker_loaders(train_df, val_df, CONFOUNDERS, params["kappa"], params["batch_size"])
        _, val_loader = make_cate_loaders(train_df, val_df, CONFOUNDERS, params["batch_size"])

        ranker = ClassificationHead(INPUT_DIM, hidden_dim=params["hidden_dim"]).to(DEVICE)
        ranker, info = train_ranker(ranker, train_loader, val_loader, DEVICE,
                                    lr=params["lr"], weight_decay=params["weight_decay"],
                                    max_epochs=50, patience=5, seed=seed, plug_in=plug_in, fraction_of_pairs=0.1)
        torch.save(ranker.state_dict(), ckpt_dir / f"{method_name}.pt")
        log(f"  [{method_name} seed {seed}] val_autoc={info[val_autoc]:.6f}")

# --- Step 5: Evaluation ---
log("=== Step 5: Evaluation ===")

# Restore stderr temporarily for eval output
sys.stderr = sys.__stderr__

def load_ckpt(model_cls, path, input_dim, device, hidden_dims=(128, 64)):
    path = Path(path)
    for hidden_dim in hidden_dims:
        try:
            model = model_cls(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
            model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            model.eval()
            return model
        except RuntimeError:
            continue
    raise RuntimeError(f"Could not load: {path}")

all_metrics = []
for seed in range(N_SEEDS):
    set_seed(seed)
    _, _, _, _, test_df = make_splits(df=df, train_size=TRAIN_SIZE, seed=seed)
    test_loader = DataLoader(EvalDataset(test_df, CONFOUNDERS), batch_size=1024, shuffle=False)

    rl_model = load_ckpt(ClassificationHead, RANKER_CHKPT_DIR / f"size_{TRAIN_SIZE}" / f"seed_{seed}" / "orthogonal.pt", INPUT_DIM, DEVICE)
    pi_model = load_ckpt(ClassificationHead, RANKER_CHKPT_DIR / f"size_{TRAIN_SIZE}" / f"seed_{seed}" / "plug_in.pt", INPUT_DIM, DEVICE)
    dr_model = load_ckpt(RegressionHead, POINTWISE_CHKPT_DIR / f"size_{TRAIN_SIZE}" / f"seed_{seed}" / "cate_model.pt", INPUT_DIM, DEVICE)
    m0_model = load_ckpt(RegressionHead, NUISANCE_CHKPT_DIR / f"size_{TRAIN_SIZE}" / f"seed_{seed}" / "mu0_model.pt", INPUT_DIM, DEVICE)
    m1_model = load_ckpt(RegressionHead, NUISANCE_CHKPT_DIR / f"size_{TRAIN_SIZE}" / f"seed_{seed}" / "mu1_model.pt", INPUT_DIM, DEVICE)

    df_eval = get_estimates_all(rl_model, pi_model, dr_model, m0_model, m1_model, test_loader, DEVICE)
    df_metrics = compute_metrics_all(df_eval)
    df_metrics["seed"] = seed
    df_metrics["train_size"] = TRAIN_SIZE
    all_metrics.append(df_metrics)

    for _, row in df_metrics.iterrows():
        print(f"  s{seed} {row[model]:20s} AUTOC={row[autoc]:.4f}  PV={row[policy_value]:.4f}")

df_all = pd.concat(all_metrics, ignore_index=True)
df_summary = df_all.groupby(["model", "train_size"])[["autoc", "policy_value"]].agg(["mean", "std"]).reset_index()

print("\n=== Summary ===")
print(df_summary.to_string())

df_all.to_csv(ROOT / "results_per_seed.csv", index=False)
df_summary.to_csv(ROOT / "results_summary.csv", index=False)
print("\n=== DONE ===")
