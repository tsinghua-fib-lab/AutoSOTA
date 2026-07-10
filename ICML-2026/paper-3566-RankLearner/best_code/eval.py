#!/usr/bin/env python3
"""Standalone evaluation for Rank-Learner (paper 3566).
Reproduces AUTOC and Mean Policy Value on the synthetic benchmark.
"""
import sys, os, argparse
from pathlib import Path

# Suppress tqdm output
import tqdm as _tqdm
class _SilentTQDM:
    def __init__(self, iterable=None, *args, **kwargs):
        self.iterable = iterable
        self.total = len(iterable) if hasattr(iterable, "__len__") else None
        self.n = 0
    def __iter__(self):
        for item in self.iterable:
            self.n += 1
            yield item
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def update(self, *a, **kw): pass
    def close(self): pass
    def set_postfix(self, **kw): pass
    def set_description(self, *a, **kw): pass
    @staticmethod
    def write(msg, *args, **kwargs): pass
_tqdm.tqdm = _SilentTQDM

sys.path.insert(0, str(Path("/repo/library")))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from data_utils import *
from models import *
from eval_utils import *

def main():
    parser = argparse.ArgumentParser(description="Rank-Learner Evaluation")
    parser.add_argument("--dataset", default="synthetic")
    parser.add_argument("--train_size", type=int, default=500)
    parser.add_argument("--n_seeds", type=int, default=5)
    args = parser.parse_args()

    TRAIN_SIZE = args.train_size
    N_SEEDS = args.n_seeds
    DATASET = args.dataset
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CONFOUNDERS = ["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9"]
    INPUT_DIM = len(CONFOUNDERS)
    ROOT = Path("/repo")
    DATA_DIR = ROOT / "data" / "datasets"
    NUISANCE_CHKPT_DIR = ROOT / "experiments" / "nuisances" / "chkpts" / DATASET
    POINTWISE_CHKPT_DIR = ROOT / "experiments" / "pointwise" / "chkpts" / DATASET
    RANKER_CHKPT_DIR = ROOT / "experiments" / "rankers" / "chkpts" / DATASET

    # Verify all checkpoints exist
    for seed in range(N_SEEDS):
        base = RANKER_CHKPT_DIR / f"size_{TRAIN_SIZE}" / f"seed_{seed}"
        for fname in ["orthogonal.pt", "plug_in.pt"]:
            p = base / fname
            if not p.exists():
                print(f"ERROR: Missing {p}")
                sys.exit(1)
        base = POINTWISE_CHKPT_DIR / f"size_{TRAIN_SIZE}" / f"seed_{seed}"
        if not (base / "cate_model.pt").exists():
            print(f"ERROR: Missing {base / 'cate_model.pt'}")
            sys.exit(1)
        base = NUISANCE_CHKPT_DIR / f"size_{TRAIN_SIZE}" / f"seed_{seed}"
        for fname in ["prop_model.pt", "mu0_model.pt", "mu1_model.pt"]:
            if not (base / fname).exists():
                print(f"ERROR: Missing {base / fname}")
                sys.exit(1)

    # Load data
    data_path = DATA_DIR / f"{DATASET}.csv"
    if not data_path.exists():
        print(f"ERROR: Data not found at {data_path}")
        sys.exit(1)
    df = pd.read_csv(data_path, index_col=0)

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
        raise RuntimeError(f"Cannot load checkpoint: {path}")

    all_metrics = []
    for seed in range(N_SEEDS):
        set_seed(seed)
        _, _, _, _, test_df = make_splits(df=df, train_size=TRAIN_SIZE, seed=seed)
        test_loader = DataLoader(EvalDataset(test_df, CONFOUNDERS), batch_size=1024, shuffle=False)

        rl_model = load_ckpt(ClassificationHead,
            RANKER_CHKPT_DIR / f"size_{TRAIN_SIZE}" / f"seed_{seed}" / "orthogonal.pt",
            INPUT_DIM, DEVICE)
        pi_model = load_ckpt(ClassificationHead,
            RANKER_CHKPT_DIR / f"size_{TRAIN_SIZE}" / f"seed_{seed}" / "plug_in.pt",
            INPUT_DIM, DEVICE)
        dr_model = load_ckpt(RegressionHead,
            POINTWISE_CHKPT_DIR / f"size_{TRAIN_SIZE}" / f"seed_{seed}" / "cate_model.pt",
            INPUT_DIM, DEVICE)
        m0_model = load_ckpt(RegressionHead,
            NUISANCE_CHKPT_DIR / f"size_{TRAIN_SIZE}" / f"seed_{seed}" / "mu0_model.pt",
            INPUT_DIM, DEVICE)
        m1_model = load_ckpt(RegressionHead,
            NUISANCE_CHKPT_DIR / f"size_{TRAIN_SIZE}" / f"seed_{seed}" / "mu1_model.pt",
            INPUT_DIM, DEVICE)

        df_eval = get_estimates_all(rl_model, pi_model, dr_model, m0_model, m1_model, test_loader, DEVICE)
        df_metrics = compute_metrics_all(df_eval)
        df_metrics["seed"] = seed
        df_metrics["train_size"] = TRAIN_SIZE
        all_metrics.append(df_metrics)

    df_all = pd.concat(all_metrics, ignore_index=True)

    # Compute summary manually to avoid MultiIndex column issues
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
    print(f"Rank-Learner Evaluation Results")
    print(f"Dataset: {DATASET}, Train size: {TRAIN_SIZE}, Seeds: {N_SEEDS}")
    print("=" * 70)

    for _, r in df_summary.iterrows():
        print(f"  {r['model']:<20s} AUTOC={r['autoc_mean']:.4f}+/-{r['autoc_std']:.4f}  PV={r['pv_mean']:.4f}+/-{r['pv_std']:.4f}")

    rls = df_summary[df_summary["model"] == "rank_learner"].iloc[0]
    drs = df_summary[df_summary["model"] == "DR"].iloc[0]
    print()
    print(f">>> Rank-Learner AUTOC = {rls['autoc_mean']:.4f} +/- {rls['autoc_std']:.4f}")
    print(f">>> DR-learner    AUTOC = {drs['autoc_mean']:.4f} +/- {drs['autoc_std']:.4f}")
    print("=" * 70)

if __name__ == "__main__":
    main()
