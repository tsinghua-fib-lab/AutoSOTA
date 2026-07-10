#!/usr/bin/env python3
"""Optimization pipeline for paper 3566.
Supports configurable hyperparameters, label smoothing, deeper architectures, and LR schedulers.
Saves checkpoints to /repo/experiments_sota/ to avoid overwriting originals.
"""
import sys, os, argparse, json
from pathlib import Path
import time

sys.path.insert(0, str(Path("/repo/library")))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_utils import *
from models import *
from training import *
from eval_utils import *

# ============================================================
# Config
# ============================================================
CONFOUNDERS = ["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9"]
INPUT_DIM = len(CONFOUNDERS)
ROOT = Path("/repo")
DATA_DIR = ROOT / "data" / "datasets"
CHKPT_BASE = ROOT / "experiments_sota"

# ============================================================
# Deep Architectures (CODE-01)
# ============================================================
class DeepClassificationHead(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64], dropout=0.1):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hd in hidden_dims:
            layers.append(nn.Linear(prev_dim, hd))
            layers.append(nn.LayerNorm(hd))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hd
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


class DeepRegressionHead(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64], dropout=0.1):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hd in hidden_dims:
            layers.append(nn.Linear(prev_dim, hd))
            layers.append(nn.LayerNorm(hd))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hd
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


# ============================================================
# Modified training with label smoothing (ALGO-03)
# ============================================================
def train_ranker_smoothed(model, train_loader, val_loader, device, lr=3e-4, weight_decay=1e-5,
                          max_epochs=50, patience=5, seed=0, plug_in=True, fraction_of_pairs=0.10,
                          label_smoothing=0.0):
    set_seed(seed)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss(reduction="mean")

    best_state, best_val, patience_left = None, -float("inf"), patience
    eps = label_smoothing

    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss, n_train = 0.0, 0

        total_batches = len(train_loader)
        max_batches = max(1, int(fraction_of_pairs * total_batches))

        with tqdm(train_loader, desc=f"Epoch {epoch}/{max_epochs}", leave=False) as pbar:
            for b_idx, (x_i, x_j, soft, orth) in enumerate(pbar):
                if b_idx >= max_batches:
                    break

                if plug_in:
                    y = soft.to(device)
                else:
                    y = orth.to(device)

                # Label smoothing
                if eps > 0:
                    y = y * (1 - 2 * eps) + eps

                x_i, x_j = x_i.to(device), x_j.to(device)
                logits_i = model(x_i)
                logits_j = model(x_j)
                logits = logits_i - logits_j

                loss = criterion(logits, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                bs = y.size(0)
                running_loss += loss.item() * bs
                n_train += bs
                pbar.set_postfix(loss=running_loss / n_train)

        val_autoc = approximate_autoc(val_loader, model, device)
        print(f"Epoch {epoch:02d} | val autoc={val_autoc:.6f}")

        if val_autoc - 1e-4 > best_val:
            best_val = val_autoc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left == 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"val_autoc": best_val}


# ============================================================
# Modified training with LR scheduler (CODE-03)
# ============================================================
def train_with_scheduler(model, train_loader, val_loader, device, train_fn, lr=3e-4,
                         weight_decay=1e-5, max_epochs=50, patience=5, seed=0, **kwargs):
    """Generic trainer with OneCycleLR scheduler."""
    set_seed(seed)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if train_fn == "propensity":
        criterion = nn.BCEWithLogitsLoss(reduction="mean")
    elif train_fn == "response" or train_fn == "cate":
        criterion = nn.MSELoss(reduction="mean")
    else:
        raise ValueError(f"Unknown train_fn: {train_fn}")

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=max_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1, div_factor=10, final_div_factor=100
    )

    best_state, best_val, patience_left = None, float("inf"), patience

    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss, n_train = 0.0, 0

        with tqdm(train_loader, desc=f"Epoch {epoch}/{max_epochs}", leave=False) as pbar:
            for x, *rest in pbar:
                x = x.to(device)
                if train_fn == "propensity":
                    t = rest[0].to(device)
                    logits = model(x)
                    loss = criterion(logits, t)
                elif train_fn == "response":
                    y = rest[1].to(device)
                    preds = model(x)
                    loss = criterion(preds, y)
                elif train_fn == "cate":
                    dr = rest[0].to(device)
                    preds = model(x)
                    loss = criterion(preds, dr)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                bs = x.size(0)
                running_loss += loss.item() * bs
                n_train += bs
                pbar.set_postfix(loss=running_loss / n_train)

        model.eval()
        val_sum, n_val = 0.0, 0
        with torch.no_grad():
            for x_val, *rest_val in val_loader:
                x_val = x_val.to(device)
                if train_fn == "propensity":
                    t_val = rest_val[0].to(device)
                    logits_val = model(x_val)
                    loss_val = criterion(logits_val, t_val)
                elif train_fn == "response":
                    y_val = rest_val[1].to(device)
                    preds_val = model(x_val)
                    loss_val = criterion(preds_val, y_val)
                elif train_fn == "cate":
                    dr_val = rest_val[0].to(device)
                    preds_val = model(x_val)
                    loss_val = criterion(preds_val, dr_val)

                bs_val = x_val.size(0)
                val_sum += loss_val.item() * bs_val
                n_val += bs_val
        val_loss = val_sum / max(n_val, 1)
        print(f"Epoch {epoch:02d} | val_loss={val_loss:.6f}")

        if val_loss + 1e-6 < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left == 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"val_loss": best_val}


# ============================================================
# Main optimization pipeline
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Rank-Learner Optimization Pipeline")
    parser.add_argument("--train_size", type=int, default=500)
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--dataset", default="synthetic")
    # Ranker hyperparams
    parser.add_argument("--kappa", type=float, default=0.5)
    parser.add_argument("--fraction_of_pairs", type=float, default=0.1)
    parser.add_argument("--ranker_lr", type=float, default=0.001)
    parser.add_argument("--ranker_wd", type=float, default=1e-5)
    parser.add_argument("--ranker_hidden_dim", type=int, default=128)
    parser.add_argument("--ranker_batch_size", type=int, default=256)
    parser.add_argument("--ranker_epochs", type=int, default=50)
    # Nuisance hyperparams
    parser.add_argument("--prop_hidden_dim", type=int, default=128)
    parser.add_argument("--prop_lr", type=float, default=0.0005)
    parser.add_argument("--m0_hidden_dim", type=int, default=64)
    parser.add_argument("--m0_lr", type=float, default=0.001)
    parser.add_argument("--m1_hidden_dim", type=int, default=64)
    parser.add_argument("--m1_lr", type=float, default=0.001)
    # DR-learner hyperparams
    parser.add_argument("--cate_hidden_dim", type=int, default=128)
    parser.add_argument("--cate_lr", type=float, default=0.0005)
    # Feature flags
    parser.add_argument("--label_smoothing", type=float, default=0.0,
                        help="Label smoothing epsilon (0=disabled)")
    parser.add_argument("--deep_arch", action="store_true",
                        help="Use deeper architectures with LayerNorm and Dropout")
    parser.add_argument("--lr_scheduler", action="store_true",
                        help="Use OneCycleLR scheduler")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate for deep architectures")
    # Output
    parser.add_argument("--tag", default="opt", help="Tag for checkpoint subdirectory")
    args = parser.parse_args()

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {DEVICE}")
    print(f"Config: {vars(args)}")

    # Model classes
    if args.deep_arch:
        def ClsHead(hd): return DeepClassificationHead(INPUT_DIM, hidden_dims=[hd, hd//2], dropout=args.dropout)
        def RegHead(hd): return DeepRegressionHead(INPUT_DIM, hidden_dims=[hd, hd//2], dropout=args.dropout)
        prop_hidden = [args.prop_hidden_dim, args.prop_hidden_dim//2]
        m0_hidden = [args.m0_hidden_dim, args.m0_hidden_dim//2]
        m1_hidden = [args.m1_hidden_dim, args.m1_hidden_dim//2]
        cate_hidden = [args.cate_hidden_dim, args.cate_hidden_dim//2]
        ranker_hidden = [args.ranker_hidden_dim, args.ranker_hidden_dim//2]
    else:
        def ClsHead(hd): return ClassificationHead(INPUT_DIM, hidden_dim=hd)
        def RegHead(hd): return RegressionHead(INPUT_DIM, hidden_dim=hd)
        prop_hidden = [args.prop_hidden_dim]
        m0_hidden = [args.m0_hidden_dim]
        m1_hidden = [args.m1_hidden_dim]
        cate_hidden = [args.cate_hidden_dim]
        ranker_hidden = [args.ranker_hidden_dim]

    TRAIN_SIZE = args.train_size
    N_SEEDS = args.n_seeds
    DATASET = args.dataset

    CHKPT_DIR = CHKPT_BASE / args.tag
    NUISANCE_CHKPT_DIR = CHKPT_DIR / "nuisances" / DATASET
    POINTWISE_CHKPT_DIR = CHKPT_DIR / "pointwise" / DATASET
    RANKER_CHKPT_DIR = CHKPT_DIR / "rankers" / DATASET

    # Load data
    data_path = DATA_DIR / f"{DATASET}.csv"
    df = pd.read_csv(data_path, index_col=0)
    print(f"Loaded data: {df.shape}")

    t_total_start = time.time()

    # ============================================================
    # Step 2: Train nuisance models
    # ============================================================
    print("\n=== Step 2: Train nuisance models ===")
    NUISANCE_CONFIGS = {
        "e":  {"hidden_dim": args.prop_hidden_dim, "lr": args.prop_lr, "weight_decay": 1e-5, "batch_size": 128},
        "m0": {"hidden_dim": args.m0_hidden_dim,  "lr": args.m0_lr,  "weight_decay": 1e-4, "batch_size": 128},
        "m1": {"hidden_dim": args.m1_hidden_dim,  "lr": args.m1_lr,  "weight_decay": 1e-5, "batch_size": 128},
    }

    for nuisance in ["e", "m0", "m1"]:
        params = NUISANCE_CONFIGS[nuisance]
        for seed in range(N_SEEDS):
            ckpt_dir = NUISANCE_CHKPT_DIR / f"size_{TRAIN_SIZE}" / f"seed_{seed}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)

            if nuisance == "e":
                filename = "prop_model.pt"
            else:
                treatment_value = 0 if nuisance == "m0" else 1
                filename = f"mu{treatment_value}_model.pt"

            set_seed(seed)
            train_df, val_df, _, _, _ = make_splits(df=df, train_size=TRAIN_SIZE, seed=seed)

            if nuisance == "e":
                train_loader, val_loader = make_nuisance_loaders(train_df, val_df, CONFOUNDERS, params["batch_size"])
                model = ClsHead(params["hidden_dim"]).to(DEVICE)
                if args.lr_scheduler:
                    model, info = train_with_scheduler(model, train_loader, val_loader, DEVICE,
                                                       "propensity", lr=params["lr"], weight_decay=params["weight_decay"],
                                                       max_epochs=50, patience=5, seed=seed)
                else:
                    model, info = train_propensity(model, train_loader, val_loader, DEVICE,
                                                   lr=params["lr"], weight_decay=params["weight_decay"],
                                                   max_epochs=50, patience=5, seed=seed)
            else:
                treatment_value = 0 if nuisance == "m0" else 1
                train = train_df[train_df["T"] == treatment_value]
                val = val_df[val_df["T"] == treatment_value]
                train_loader, val_loader = make_nuisance_loaders(train, val, CONFOUNDERS, params["batch_size"])
                model = RegHead(params["hidden_dim"]).to(DEVICE)
                if args.lr_scheduler:
                    model, info = train_with_scheduler(model, train_loader, val_loader, DEVICE,
                                                       "response", lr=params["lr"], weight_decay=params["weight_decay"],
                                                       max_epochs=50, patience=5, seed=seed)
                else:
                    model, info = train_response(model, train_loader, val_loader, DEVICE,
                                                 lr=params["lr"], weight_decay=params["weight_decay"],
                                                 max_epochs=50, patience=5, seed=seed)

            torch.save(model.state_dict(), ckpt_dir / filename)
            print(f"  [{nuisance} seed {seed}] done")

    # ============================================================
    # Step 3: Train DR-learner
    # ============================================================
    print("\n=== Step 3: Train DR-learner ===")
    for seed in range(N_SEEDS):
        ckpt_dir = POINTWISE_CHKPT_DIR / f"size_{TRAIN_SIZE}" / f"seed_{seed}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        set_seed(seed)
        _, _, train_df, val_df, _ = make_splits(df=df, train_size=TRAIN_SIZE, seed=seed)

        ns_seed_dir = NUISANCE_CHKPT_DIR / f"size_{TRAIN_SIZE}" / f"seed_{seed}"
        prop_model = ClsHead(args.prop_hidden_dim).to(DEVICE)
        prop_model.load_state_dict(torch.load(ns_seed_dir / "prop_model.pt", map_location=DEVICE, weights_only=True))
        m0_model = RegHead(args.m0_hidden_dim).to(DEVICE)
        m0_model.load_state_dict(torch.load(ns_seed_dir / "mu0_model.pt", map_location=DEVICE, weights_only=True))
        m1_model = RegHead(args.m1_hidden_dim).to(DEVICE)
        m1_model.load_state_dict(torch.load(ns_seed_dir / "mu1_model.pt", map_location=DEVICE, weights_only=True))

        train_df = compute_dr_scores(train_df, CONFOUNDERS, prop_model, m0_model, m1_model, DEVICE)
        val_df = compute_dr_scores(val_df, CONFOUNDERS, prop_model, m0_model, m1_model, DEVICE)

        train_loader, val_loader = make_cate_loaders(train_df, val_df, CONFOUNDERS, batch_size=128)
        cate_model = RegHead(args.cate_hidden_dim).to(DEVICE)
        if args.lr_scheduler:
            cate_model, info = train_with_scheduler(cate_model, train_loader, val_loader, DEVICE,
                                                    "cate", lr=args.cate_lr, weight_decay=1e-4,
                                                    max_epochs=50, patience=5, seed=seed)
        else:
            cate_model, info = train_cate(cate_model, train_loader, val_loader, DEVICE,
                                          lr=args.cate_lr, weight_decay=1e-4,
                                          max_epochs=50, patience=5, seed=seed)

        torch.save(cate_model.state_dict(), ckpt_dir / "cate_model.pt")
        print(f"  [seed {seed}] done")

    # ============================================================
    # Step 4: Train ranker (orthogonal only for optimization)
    # ============================================================
    print("\n=== Step 4: Train Rank-Learner (orthogonal) ===")
    for seed in range(N_SEEDS):
        ckpt_dir = RANKER_CHKPT_DIR / f"size_{TRAIN_SIZE}" / f"seed_{seed}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        set_seed(seed)
        _, _, train_df, val_df, _ = make_splits(df=df, train_size=TRAIN_SIZE, seed=seed)

        ns_seed_dir = NUISANCE_CHKPT_DIR / f"size_{TRAIN_SIZE}" / f"seed_{seed}"
        prop_model = ClsHead(args.prop_hidden_dim).to(DEVICE)
        prop_model.load_state_dict(torch.load(ns_seed_dir / "prop_model.pt", map_location=DEVICE, weights_only=True))
        m0_model = RegHead(args.m0_hidden_dim).to(DEVICE)
        m0_model.load_state_dict(torch.load(ns_seed_dir / "mu0_model.pt", map_location=DEVICE, weights_only=True))
        m1_model = RegHead(args.m1_hidden_dim).to(DEVICE)
        m1_model.load_state_dict(torch.load(ns_seed_dir / "mu1_model.pt", map_location=DEVICE, weights_only=True))

        train_df = compute_dr_scores(train_df, CONFOUNDERS, prop_model, m0_model, m1_model, DEVICE)
        val_df = compute_dr_scores(val_df, CONFOUNDERS, prop_model, m0_model, m1_model, DEVICE)

        train_loader, _ = make_ranker_loaders(train_df, val_df, CONFOUNDERS, args.kappa, args.ranker_batch_size)
        _, val_loader = make_cate_loaders(train_df, val_df, CONFOUNDERS, args.ranker_batch_size)

        ranker = ClsHead(args.ranker_hidden_dim).to(DEVICE)
        ranker, info = train_ranker_smoothed(ranker, train_loader, val_loader, DEVICE,
                                             lr=args.ranker_lr, weight_decay=args.ranker_wd,
                                             max_epochs=args.ranker_epochs, patience=5, seed=seed,
                                             plug_in=False, fraction_of_pairs=args.fraction_of_pairs,
                                             label_smoothing=args.label_smoothing)
        torch.save(ranker.state_dict(), ckpt_dir / "orthogonal.pt")
        print(f"  [seed {seed}] val_autoc={info['val_autoc']:.6f}")

    # ============================================================
    # Step 5: Evaluate
    # ============================================================
    print("\n=== Step 5: Evaluate ===")
    all_metrics = []

    for seed in range(N_SEEDS):
        set_seed(seed)

        # Get test data (same fixed test set as original)
        _, _, _, _, test_df = make_splits(df=df, train_size=TRAIN_SIZE, seed=seed)
        test_loader = DataLoader(EvalDataset(test_df, CONFOUNDERS), batch_size=1024, shuffle=False)

        def load_ckpt(model_cls, path, hd_list, device):
            path = Path(path)
            for hd in hd_list:
                try:
                    model = model_cls(hd).to(device)
                    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
                    model.eval()
                    return model
                except RuntimeError:
                    continue
            raise RuntimeError(f"Cannot load: {path}")

        rl_model = load_ckpt(ClsHead,
            RANKER_CHKPT_DIR / f"size_{TRAIN_SIZE}" / f"seed_{seed}" / "orthogonal.pt",
            ranker_hidden, DEVICE)
        dr_model = load_ckpt(RegHead,
            POINTWISE_CHKPT_DIR / f"size_{TRAIN_SIZE}" / f"seed_{seed}" / "cate_model.pt",
            cate_hidden, DEVICE)
        m0_model = load_ckpt(RegHead,
            NUISANCE_CHKPT_DIR / f"size_{TRAIN_SIZE}" / f"seed_{seed}" / "mu0_model.pt",
            m0_hidden, DEVICE)
        m1_model = load_ckpt(RegHead,
            NUISANCE_CHKPT_DIR / f"size_{TRAIN_SIZE}" / f"seed_{seed}" / "mu1_model.pt",
            m1_hidden, DEVICE)

        df_eval = get_estimates_all(rl_model, rl_model, dr_model, m0_model, m1_model, test_loader, DEVICE)
        df_metrics = compute_metrics_all(df_eval)
        df_metrics["seed"] = seed
        df_metrics["train_size"] = TRAIN_SIZE
        all_metrics.append(df_metrics)

    df_all = pd.concat(all_metrics, ignore_index=True)

    def summarize(df):
        rows = []
        for mn in ["oracle", "DR", "T", "rank_learner", "plug_in"]:
            ss = df[df["model"] == mn]
            if len(ss):
                rows.append({
                    "model": mn,
                    "autoc_mean": ss["autoc"].mean(),
                    "autoc_std": ss["autoc"].std(),
                    "pv_mean": ss["policy_value"].mean(),
                    "pv_std": ss["policy_value"].std(),
                })
        return pd.DataFrame(rows)

    df_summary = summarize(df_all)

    print()
    print("=" * 70)
    print(f"Rank-Learner Optimization Results [{args.tag}]")
    print(f"Dataset: {DATASET}, Train size: {TRAIN_SIZE}, Seeds: {N_SEEDS}")
    print("=" * 70)

    for _, r in df_summary.iterrows():
        print(f'  {r["model"]:<20s} AUTOC={r["autoc_mean"]:.4f}+/-{r["autoc_std"]:.4f}  PV={r["pv_mean"]:.4f}+/-{r["pv_std"]:.4f}')

    rls = df_summary[df_summary["model"] == "rank_learner"].iloc[0]
    drs = df_summary[df_summary["model"] == "DR"].iloc[0]
    print()
    print(f'>>> Rank-Learner AUTOC = {rls["autoc_mean"]:.4f} +/- {rls["autoc_std"]:.4f}')
    print(f'>>> DR-learner    AUTOC = {drs["autoc_mean"]:.4f} +/- {drs["autoc_std"]:.4f}')
    print("=" * 70)

    total_time = time.time() - t_total_start
    print(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f} min)")

    # Output JSON for parsing
    result = {
        "tag": args.tag,
        "AUTOC": float(rls["autoc_mean"]),
        "AUTOC_std": float(rls["autoc_std"]),
        "PV": float(rls["pv_mean"]),
        "PV_std": float(rls["pv_std"]),
        "DR_AUTOC": float(drs["autoc_mean"]),
        "total_time_s": total_time,
    }
    print("\nJSON_RESULT:", json.dumps(result))

if __name__ == "__main__":
    main()
