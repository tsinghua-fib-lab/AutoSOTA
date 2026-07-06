#!/usr/bin/env python3
"""
pTNAS BlockMixed search: for one dataset, use the dataset-specific group
subspace, then evaluate search performance over explored architectures M.

Data loaded ONCE, proxy scores from CSV, training results from CSV.
SH runs for real timing. Final test metric looked up from bench CSV.

Usage:
    python scripts/new_space/search_runs/ptnas_blockmixed_search.py \
        --dataset avito-user-clicks \
        --device cuda:0 \
        --output_csv run_outputs/data/new_space/search_runs/blockmixed/ptnas_block_mixed_search_results.csv
"""
from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import gc
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
import torch
import torch_frame.data
from relbench.base import TaskType
from sklearn.metrics import mean_absolute_error, roc_auc_score
from torch.nn import BCEWithLogitsLoss, L1Loss

from model.base import construct_stype_encoder_dict, default_stype_encoder_cls_kwargs
from search_space.block_mixed import PTNASBlockMixed, BlockSpec
from utils.table_data import TableData

device = torch.device("cpu")

# BlockMixed proxy SRCC direction. The score CSVs keep the historical "v1"
# suffix because those files were produced before the module was renamed.
# Positive means higher score = better model.
# negative means lower score = better model
V1_PROXY_DIRECTION = {
    "event-user-attendance": "negative",   # SRCC = -0.623
    "avito-ad-ctr": "negative",            # SRCC = -0.567
    "hm-user-churn": "positive",           # SRCC = +0.518
    "avito-user-clicks": "negative",       # SRCC = -0.341
    "trial-site-success": "negative",      # SRCC = -0.383
}

# Default paths
SPACE_DIVERSE_JSON = _PROJECT_ROOT / "datasets" / "nas_bench_tabular" / "space_blockmixed" / "architecture" / "random_sampled_arch_blockmixed_metadata.json"
SPACE_DIVERSE_TXT = _PROJECT_ROOT / "datasets" / "nas_bench_tabular" / "space_blockmixed" / "architecture" / "blockmixed.txt"
PROXY_SCORE_DIR = _PROJECT_ROOT / "datasets" / "nas_bench_tabular" / "space_blockmixed" / "proxy_score" / "ptproxy"
BENCH_CSV = _PROJECT_ROOT / "datasets" / "nas_bench_tabular" / "space_blockmixed" / "training" / "block_mixed_diverse_results.csv"

# ---------------------------------------------------------------------------
CSV_FIELDS = [
    "timestamp", "dataset", "space_file", "pool_size", "M", "top_k", "seed",
    "scoring_time_seconds", "sh_time_seconds", "final_train_time_seconds",
    "inference_time_seconds", "total_time_seconds",
    "best_val_metric", "final_test_metric", "best_params", "metric",
]


def write_csv(path, row):
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not exists:
            w.writeheader()
        w.writerow(row)
        fcntl.flock(f, fcntl.LOCK_UN)


def is_done(done_set, dataset, M, seed):
    return (dataset, M, seed) in done_set


def load_done_set(path):
    done = set()
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path) as f:
            for r in csv.DictReader(f):
                done.add((r["dataset"], int(r["M"]), int(r["seed"])))
    return done


# ---------------------------------------------------------------------------
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def deactivate_dropout(net):
    for m in net.modules():
        if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False


# ---------------------------------------------------------------------------
# Load space
# ---------------------------------------------------------------------------
def load_space_diverse(txt_path):
    """Load all 900 models from blockmixed.txt, return list of block_specs strings."""
    specs = []
    with open(txt_path) as f:
        for line in f:
            line = line.strip()
            if line:
                specs.append(line)
    return specs


def get_dataset_subspace(dataset_name, json_path, all_specs, proxy_df):
    """Get dataset-specific subspace: filter by groups from JSON."""
    with open(json_path) as f:
        mapping = json.load(f)

    if dataset_name not in mapping:
        raise ValueError(f"{dataset_name} not in {json_path}")

    # Use the last (largest) group config for this dataset
    config = mapping[dataset_name][-1]
    groups = set(config["groups"])

    # Filter proxy_df by groups
    sub = proxy_df[proxy_df["group"].isin(groups)].copy()
    return sub, groups


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def random_sample_df(df, M, seed=42):
    rng = random.Random(seed)
    indices = list(df.index)
    M = min(M, len(indices))
    sampled = rng.sample(indices, M)
    return df.loc[sampled]


def stratified_sample_df(df, M, seed=42):
    """Stratified by depth, uniform by num_params within each depth."""
    rng = random.Random(seed)
    by_depth = defaultdict(list)
    for idx, row in df.iterrows():
        by_depth[int(row["depth"])].append((idx, int(row["num_params"])))
    total = len(df)
    depths = sorted(by_depth.keys())
    M = min(M, total)
    remaining = M
    alloc = {}
    for d in depths:
        n = max(1, round(M * len(by_depth[d]) / total))
        n = min(n, len(by_depth[d]), remaining)
        alloc[d] = n
        remaining -= n
    for d in sorted(depths, key=lambda d: len(by_depth[d]), reverse=True):
        if remaining <= 0:
            break
        extra = min(remaining, len(by_depth[d]) - alloc[d])
        alloc[d] += extra
        remaining -= extra
    sampled_indices = []
    for d in depths:
        group = sorted(by_depth[d], key=lambda x: x[1])  # sort by num_params
        n = min(alloc[d], len(group))
        if n >= len(group):
            sampled_indices.extend(idx for idx, _ in group)
        else:
            step = len(group) / n
            indices = [min(int(i * step), len(group)-1) for i in range(n)]
            indices = sorted(set(indices))
            fill = 0
            while len(indices) < n:
                if fill not in indices:
                    indices.append(fill)
                fill += 1
            indices = sorted(indices[:n])
            sampled_indices.extend(group[i][0] for i in indices)
    return df.loc[sampled_indices]


def sample_df(df, M, seed=42, method="random"):
    if method == "stratified":
        return stratified_sample_df(df, M, seed)
    return random_sample_df(df, M, seed)


# ---------------------------------------------------------------------------
# Train / Eval / SH
# ---------------------------------------------------------------------------
def train_model(model, train_loader, val_loader, is_regression,
                num_epochs=200, lr=0.001, max_batches_per_epoch=20, early_stop_patience=10):
    print(f"\n  Training: epochs={num_epochs}, lr={lr}")
    loss_fn = L1Loss() if is_regression else BCEWithLogitsLoss()
    if is_regression:
        deactivate_dropout(model)
    higher_is_better = not is_regression
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    model.to(device)
    patience = 0
    best_val = -math.inf if higher_is_better else math.inf
    best_state = None
    t0 = time.time()
    for epoch in range(num_epochs):
        model.train()
        for idx, batch in enumerate(train_loader):
            if idx > max_batches_per_epoch:
                break
            optimizer.zero_grad()
            batch = batch.to(device)
            pred = model(batch)
            pred = pred.view(-1) if pred.size(1) == 1 else pred
            loss = loss_fn(pred, batch.y.float())
            loss.backward()
            optimizer.step()
        val_metric = evaluate_model(model, val_loader, is_regression)[0]
        improved = (higher_is_better and val_metric > best_val) or \
                   (not higher_is_better and val_metric < best_val)
        if improved:
            best_val = val_metric
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience > early_stop_patience:
                break
    train_time = time.time() - t0
    if best_state:
        model.load_state_dict(best_state)
    print(f"  Training completed! Best val: {best_val:.6f}, time: {train_time:.2f}s")
    return model, train_time


@torch.no_grad()
def evaluate_model(model, loader, is_regression):
    model.eval()
    preds, ys = [], []
    t0 = time.time()
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        pred = pred.view(-1) if pred.size(1) == 1 else pred
        preds.append(pred.cpu())
        ys.append(batch.y.float().cpu())
    elapsed = time.time() - t0
    preds = torch.cat(preds).numpy()
    ys = torch.cat(ys).numpy()
    if is_regression:
        return float(mean_absolute_error(ys, preds)), elapsed
    else:
        return float(roc_auc_score(ys, torch.sigmoid(torch.tensor(preds)).numpy())), elapsed


def build_block_mixed_model(block_specs_str, table_data, channels=32):
    """Build a PTNASBlockMixed model from block_specs string."""
    import ast
    block_specs_list = ast.literal_eval(block_specs_str)
    block_specs = [tuple(b) for b in block_specs_list]
    stype_encoder_dict = construct_stype_encoder_dict(default_stype_encoder_cls_kwargs)
    model = PTNASBlockMixed(
        channels=channels, out_channels=1,
        num_layers=len(block_specs),
        col_stats=table_data.col_stats,
        col_names_dict=table_data.col_names_dict,
        stype_encoder_dict=stype_encoder_dict,
        block_specs=block_specs,
    ).to(device)
    return model


SH_DETAIL_FIELDS = [
    "dataset", "M", "round", "candidate_idx", "block_specs",
    "epochs", "train_time_seconds", "eval_time_seconds",
    "val_metric", "kept",
]


def write_sh_detail(path, row):
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        w = csv.DictWriter(f, fieldnames=SH_DETAIL_FIELDS)
        if not exists:
            w.writeheader()
        w.writerow(row)
        fcntl.flock(f, fcntl.LOCK_UN)


def successive_halving(candidates_df, table_data, is_regression,
                       train_loader, val_loader, channels=32,
                       max_epochs=50, min_epochs=1,
                       dataset_name="", M=0, sh_detail_csv="",
                       eta=3):
    """SH with persistent models. Records per-model per-round details."""
    print(f"\n  Successive Halving: {len(candidates_df)} candidates")

    candidates = []
    for _, row in candidates_df.iterrows():
        model = build_block_mixed_model(row["block_specs"], table_data, channels)
        candidates.append((row["block_specs"], row["proxy_score"], model))

    current_epochs = min_epochs
    best_val = 0.0
    scored = []
    round_num = 0

    while len(candidates) > 1 and current_epochs <= max_epochs:
        round_num += 1
        print(f"   Round: {len(candidates)} candidates, {current_epochs} epochs")
        scored = []
        for i, (specs, ps, model) in enumerate(candidates):
            t_train_start = time.time()
            model, train_time = train_model(model, train_loader, val_loader, is_regression,
                                   num_epochs=current_epochs, lr=0.001,
                                   max_batches_per_epoch=20, early_stop_patience=10)
            t_eval_start = time.time()
            val_score = evaluate_model(model, val_loader, is_regression)[0]
            eval_time = time.time() - t_eval_start
            scored.append((specs, ps, model, val_score, train_time, eval_time))

        scored.sort(key=lambda x: x[3], reverse=not is_regression)
        keep = max(1, math.ceil(len(candidates) / eta))

        # Write SH detail
        if sh_detail_csv:
            for i, (specs, ps, model, vs, tt, et) in enumerate(scored):
                write_sh_detail(sh_detail_csv, {
                    "dataset": dataset_name,
                    "M": M,
                    "round": round_num,
                    "candidate_idx": i,
                    "block_specs": specs,
                    "epochs": current_epochs,
                    "train_time_seconds": round(tt, 3),
                    "eval_time_seconds": round(et, 3),
                    "val_metric": round(vs, 6),
                    "kept": 1 if i < keep else 0,
                })

        for _, _, m, _, _, _ in scored[keep:]:
            del m
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
        gc.collect()
        candidates = [(s, p, m) for s, p, m, _, _, _ in scored[:keep]]
        best_val = scored[0][3]
        print(f"     Kept top {len(candidates)}, best val: {best_val:.4f}")
        if current_epochs < max_epochs:
            current_epochs = min(max_epochs, current_epochs * eta)
        else:
            break

    best_specs, _, best_model = candidates[0]
    return best_specs, best_val, best_model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--M_min", type=int, default=10)
    parser.add_argument("--M_max", type=int, default=None)
    parser.add_argument("--M_step", type=int, default=10)
    parser.add_argument("--output_csv", type=str,
                        default=str(_PROJECT_ROOT / "run_outputs" / "data" / "new_space" / "search_runs" / "blockmixed" / "ptnas_block_mixed_search_results.csv"))
    parser.add_argument("--sh_detail_csv", type=str,
                        default=str(_PROJECT_ROOT / "run_outputs" / "data" / "new_space" / "search_runs" / "blockmixed" / "ptnas_block_mixed_search_sh_detail.csv"))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample_method", type=str, default="random", choices=["random", "stratified"])
    parser.add_argument("--sh_min_epochs", type=int, default=1)
    parser.add_argument("--sh_max_epochs", type=int, default=50)
    parser.add_argument("--eta", type=int, default=3)
    args = parser.parse_args()

    global device
    device = torch.device(args.device)
    dataset_name = args.dataset

    # ---- Load all CSVs into memory at start (not counted in timing) ----
    bench_df = pd.read_csv(BENCH_CSV)
    bench_df = bench_df[bench_df["dataset"] == dataset_name]
    bench_lookup = {}
    for _, r in bench_df.iterrows():
        bench_lookup[r["block_specs"]] = {
            "test_metric": float(r["test_metric"]),
            "train_time": float(r["train_time_seconds"]),
            "test_time": float(r["test_time_seconds"]),
        }

    proxy_csv = PROXY_SCORE_DIR / f"score_{dataset_name}_v1.csv"
    print(f"[{os.getpid()}] Loading proxy scores from {proxy_csv}...", flush=True)
    proxy_df = pd.read_csv(proxy_csv)

    sub_df, groups = get_dataset_subspace(dataset_name, SPACE_DIVERSE_JSON, None, proxy_df)
    pool_size = len(sub_df)
    M_max = args.M_max or pool_size
    M_values = list(range(args.M_min, min(M_max, pool_size) + 1, args.M_step))

    done_set = load_done_set(args.output_csv)

    print(f"[{os.getpid()}] dataset={dataset_name}  groups={groups}  pool={pool_size}  seed={args.seed}", flush=True)
    print(f"[{os.getpid()}] M search grid: {M_values[0]} to {M_values[-1]} step {args.M_step} = {len(M_values)} tasks", flush=True)

    # Load data ONCE
    data_dir = _PROJECT_ROOT / "datasets" / "fit-medium-table" / dataset_name
    print(f"[{os.getpid()}] Loading {dataset_name}...", flush=True)
    table_data = TableData.load_from_dir(str(data_dir))
    is_regression = table_data.task_type == TaskType.REGRESSION

    # Create DataLoaders ONCE
    train_loader = torch_frame.data.DataLoader(table_data.train_tf, batch_size=256, shuffle=True,
                                               num_workers=2, persistent_workers=True)
    val_loader = torch_frame.data.DataLoader(table_data.val_tf, batch_size=256, shuffle=False,
                                             num_workers=2, persistent_workers=True)

    # Sweep M
    for M in M_values:
        if is_done(done_set, dataset_name, M, args.seed):
            print(f"[{os.getpid()}] skip M={M} seed={args.seed} (done)", flush=True)
            continue

        set_seed(args.seed)
        top_k = max(1, math.ceil(M / 30))

        # Sample
        sampled = sample_df(sub_df, M, seed=args.seed + M, method=args.sample_method)
        # Sort by proxy score: ascending for negative SRCC (smaller = better)
        proxy_ascending = V1_PROXY_DIRECTION.get(dataset_name, "negative") == "negative"
        sampled_sorted = sampled.sort_values("proxy_score", ascending=proxy_ascending)
        top_candidates = sampled_sorted.head(top_k)

        # Sum proxy times
        per_m_scoring_time = sampled["proxy_time_seconds"].sum()

        print(f"\n[{os.getpid()}] === M={M}  top_k={top_k} ===", flush=True)

        # SH
        sh_start = time.time()
        if top_k == 1:
            best_specs = top_candidates.iloc[0]["block_specs"]
            best_val = 0.0
            final_model = build_block_mixed_model(best_specs, table_data)
            sh_time = 0.0
        else:
            best_specs, best_val, final_model = successive_halving(
                top_candidates, table_data, is_regression,
                train_loader, val_loader,
                max_epochs=args.sh_max_epochs, min_epochs=args.sh_min_epochs,
                dataset_name=dataset_name, M=M,
                sh_detail_csv=args.sh_detail_csv,
                eta=args.eta,
            )
            sh_time = time.time() - sh_start

        # Lookup from bench
        bench_entry = bench_lookup.get(best_specs)
        if bench_entry:
            test_metric = bench_entry["test_metric"]
            train_time = bench_entry["train_time"]
            test_time = bench_entry["test_time"]
        else:
            print(f"[{os.getpid()}] WARN: specs not in bench, skipping", flush=True)
            del final_model
            continue

        total_time = per_m_scoring_time + sh_time + train_time + test_time

        print(f"[{os.getpid()}] M={M}  test={test_metric:.4f}  total={total_time:.1f}s", flush=True)

        del final_model
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
        gc.collect()

        write_csv(args.output_csv, {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": dataset_name,
            "space_file": "random_sampled_arch_blockmixed_metadata.json",
            "pool_size": pool_size,
            "M": M,
            "top_k": top_k,
            "seed": args.seed,
            "scoring_time_seconds": round(per_m_scoring_time, 3),
            "sh_time_seconds": round(sh_time, 2),
            "final_train_time_seconds": round(train_time, 2),
            "inference_time_seconds": round(test_time, 3),
            "total_time_seconds": round(total_time, 2),
            "best_val_metric": round(best_val, 6),
            "final_test_metric": round(test_metric, 6),
            "best_params": best_specs,
            "metric": "mae" if is_regression else "roc_auc",
        })

    print(f"\n[{os.getpid()}] All done for {dataset_name}!", flush=True)


if __name__ == "__main__":
    main()
