#!/usr/bin/env python3
"""
pTNAS ResDNN search: read pre-computed proxy scores and training results,
evaluate search performance over explored architectures M.

Usage:
    python scripts/new_space/search_runs/ptnas_resdnn_search.py \
        --data_dir datasets/fit-medium-table/event-user-attendance \
        --space_file datasets/nas_bench_tabular/space_resdnn/architecture/random_sampled_arch_resdnn_regression.txt \
        --M_min 5 --M_step 5 --seed 42 \
        --device cuda:0 \
        --output_csv run_outputs/data/new_space/search_runs/resdnn/ptnas_pool_search_results.csv
"""
from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import gc
import math
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List

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
from search_space import PTNASResNet
from utils.table_data import TableData

device = torch.device("cpu")

# ---------------------------------------------------------------------------
CSV_FIELDS = [
    "timestamp", "dataset", "space_file", "pool_size", "M", "top_k", "seed",
    "scoring_time_seconds", "sh_time_seconds", "final_train_time_seconds",
    "inference_time_seconds", "total_time_seconds",
    "best_val_metric", "final_test_metric", "best_params", "metric",
]

SH_DETAIL_FIELDS = [
    "dataset", "M", "seed", "round", "candidate_idx", "architecture",
    "epochs", "train_time_seconds", "eval_time_seconds",
    "val_metric", "kept",
]


def write_csv(path, row, fields):
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            w.writeheader()
        w.writerow(row)
        fcntl.flock(f, fcntl.LOCK_UN)


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


def get_num_cols(td):
    return sum(len(v) for v in td.col_names_dict.values())


def deactivate_dropout(net):
    for m in net.modules():
        if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False


# ---------------------------------------------------------------------------
def load_pool(path: str) -> List[List[int]]:
    archs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                archs.append([int(x) for x in line.split("-")])
    return archs


def generate_M_values(pool_size):
    """Non-uniform M schedule: fine at small M, coarse at large M."""
    vals = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 300, 500, 700, 1000]
    return sorted(v for v in vals if v <= pool_size)


def random_sample(pool, M, seed=42):
    rng = random.Random(seed)
    M = min(M, len(pool))
    return rng.sample(pool, M)


def stratified_sample(pool, M, seed=42):
    """Stratified by depth, uniform by width within each depth."""
    from collections import defaultdict
    rng = random.Random(seed)
    by_depth = defaultdict(list)
    for arch in pool:
        by_depth[len(arch)].append(arch)
    total = len(pool)
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
    sampled = []
    for d in depths:
        group = sorted(by_depth[d], key=lambda a: sum(a))  # sort by total width
        n = min(alloc[d], len(group))
        if n >= len(group):
            sampled.extend(group)
        else:
            # evenly spaced indices, deterministic
            step = len(group) / n
            indices = [min(int(i * step), len(group)-1) for i in range(n)]
            indices = sorted(set(indices))
            # fill gaps if dedup reduced count
            fill = 0
            while len(indices) < n:
                if fill not in indices:
                    indices.append(fill)
                fill += 1
            indices = sorted(indices[:n])
            sampled.extend(group[i] for i in indices)
    return sampled


def sample_pool(pool, M, seed=42, method="random"):
    if method == "stratified":
        return stratified_sample(pool, M, seed)
    return random_sample(pool, M, seed)


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


def successive_halving(selected_models, table_data, is_regression,
                       train_loader, val_loader,
                       max_epochs=50, min_epochs=1,
                       dataset_name="", M=0, seed=42, sh_detail_csv="",
                       eta=3):
    """SH with persistent models, records per-model per-round details."""
    print(f"\n  Successive Halving: {len(selected_models)} candidates")
    num_cols = get_num_cols(table_data)
    candidates = []
    for arch, proxy_score in selected_models:
        stype_encoder_dict = construct_stype_encoder_dict(default_stype_encoder_cls_kwargs)
        model = PTNASResNet(
            channels=num_cols, out_channels=1, num_layers=len(arch),
            col_stats=table_data.col_stats, col_names_dict=table_data.col_names_dict,
            stype_encoder_dict=stype_encoder_dict, block_widths=arch,
            normalization="layer_norm", dropout_prob=0.2,
        ).to(device)
        candidates.append((arch, proxy_score, model))

    current_epochs = min_epochs
    best_val = 0.0
    scored = []
    round_num = 0

    while len(candidates) > 1 and current_epochs <= max_epochs:
        round_num += 1
        print(f"   Round: {len(candidates)} candidates, {current_epochs} epochs")
        scored = []
        for i, (arch, ps, model) in enumerate(candidates):
            print(f"     Training {i+1}/{len(candidates)}: {arch}")
            model, train_time = train_model(model, train_loader, val_loader, is_regression,
                                            num_epochs=current_epochs, lr=0.001,
                                            max_batches_per_epoch=20, early_stop_patience=10)
            t_eval = time.time()
            val_score = evaluate_model(model, val_loader, is_regression)[0]
            eval_time = time.time() - t_eval
            scored.append((arch, ps, model, val_score, train_time, eval_time))

        scored.sort(key=lambda x: x[3], reverse=not is_regression)
        keep = max(1, math.ceil(len(candidates) / eta))

        # Write SH detail
        if sh_detail_csv:
            for i, (arch, ps, model, vs, tt, et) in enumerate(scored):
                write_csv(sh_detail_csv, {
                    "dataset": dataset_name,
                    "M": M,
                    "seed": seed,
                    "round": round_num,
                    "candidate_idx": i,
                    "architecture": "-".join(map(str, arch)),
                    "epochs": current_epochs,
                    "train_time_seconds": round(tt, 3),
                    "eval_time_seconds": round(et, 3),
                    "val_metric": round(vs, 6),
                    "kept": 1 if i < keep else 0,
                }, SH_DETAIL_FIELDS)

        for _, _, m, _, _, _ in scored[keep:]:
            del m
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
        gc.collect()
        candidates = [(a, p, m) for a, p, m, _, _, _ in scored[:keep]]
        best_val = scored[0][3]
        print(f"     Kept top {len(candidates)}, best val: {best_val:.4f}")
        if current_epochs < max_epochs:
            current_epochs = min(max_epochs, current_epochs * eta)
        else:
            break

    best_arch, _, best_model = candidates[0]
    return best_arch, best_val, best_model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--space_file", type=str, required=True)
    parser.add_argument("--M_min", type=int, default=5)
    parser.add_argument("--M_max", type=int, default=None)
    parser.add_argument("--M_step", type=int, default=5)
    parser.add_argument("--output_csv", type=str, default=str(_PROJECT_ROOT / "run_outputs" / "data" / "new_space" / "search_runs" / "resdnn" / "ptnas_pool_search_results.csv"))
    parser.add_argument("--sh_detail_csv", type=str,
                        default=str(_PROJECT_ROOT / "run_outputs" / "data" / "new_space" / "search_runs" / "resdnn" / "ptnas_pool_search_sh_detail.csv"))
    parser.add_argument("--bench_csv", type=str,
                        default=str(_PROJECT_ROOT / "datasets" / "nas_bench_tabular" / "space_resdnn" / "training" / "resnet_pool_results.csv"))
    parser.add_argument("--proxy_score_dir", type=str,
                        default=str(_PROJECT_ROOT / "datasets" / "nas_bench_tabular" / "space_resdnn" / "proxy_score" / "ptproxy"))
    parser.add_argument("--proxy_variant", type=str, default="v1")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample_method", type=str, default="random", choices=["random", "stratified"])
    parser.add_argument("--sh_min_epochs", type=int, default=1)
    parser.add_argument("--sh_max_epochs", type=int, default=50)
    parser.add_argument("--eta", type=int, default=3)
    args = parser.parse_args()

    global device
    device = torch.device(args.device)
    dataset_name = Path(args.data_dir).name
    space_name = Path(args.space_file).name

    # ---- Load all CSVs into memory at start (not counted in timing) ----
    bench_df = pd.read_csv(args.bench_csv)
    bench_lookup = {}
    for _, r in bench_df.iterrows():
        bench_lookup[(r["dataset"], r["architecture"])] = {
            "test_metric": float(r["test_metric"]),
            "train_time": float(r["train_time_seconds"]),
            "test_time": float(r["test_time_seconds"]),
        }

    proxy_csv = Path(args.proxy_score_dir) / f"score_{dataset_name}_{args.proxy_variant}.csv"
    print(f"[{os.getpid()}] Loading proxy scores from {proxy_csv}...", flush=True)
    proxy_df = pd.read_csv(proxy_csv)
    all_scores = {}
    all_proxy_times = {}
    for _, r in proxy_df.iterrows():
        key = tuple(int(x) for x in str(r["architecture"]).split("-"))
        all_scores[key] = float(r["proxy_score"]) if pd.notna(r["proxy_score"]) else -1e10
        all_proxy_times[key] = float(r["proxy_time_seconds"]) if pd.notna(r["proxy_time_seconds"]) else 0.0
    print(f"[{os.getpid()}] Loaded {len(all_scores)} proxy scores", flush=True)

    done_set = load_done_set(args.output_csv)

    # Load pool
    pool = load_pool(args.space_file)
    M_values = generate_M_values(len(pool))

    print(f"[{os.getpid()}] dataset={dataset_name}  space={space_name}  pool={len(pool)}  seed={args.seed}", flush=True)
    print(f"[{os.getpid()}] M search grid: {M_values[0]} to {M_values[-1]} step {args.M_step} = {len(M_values)} tasks", flush=True)

    # Load data ONCE
    print(f"[{os.getpid()}] Loading {dataset_name}...", flush=True)
    table_data = TableData.load_from_dir(args.data_dir)
    is_regression = table_data.task_type == TaskType.REGRESSION

    # Create DataLoaders ONCE (outside SH timing)
    train_loader = torch_frame.data.DataLoader(table_data.train_tf, batch_size=256, shuffle=True,
                                               num_workers=2, persistent_workers=True)
    val_loader = torch_frame.data.DataLoader(table_data.val_tf, batch_size=256, shuffle=False,
                                             num_workers=2, persistent_workers=True)

    # Sweep M
    for M in M_values:
        if (dataset_name, M, args.seed) in done_set:
            print(f"[{os.getpid()}] skip M={M} seed={args.seed} (done)", flush=True)
            continue

        set_seed(args.seed)
        top_k = max(1, math.ceil(M / 30))

        # Sample
        sampled = sample_pool(pool, M, seed=args.seed + M, method=args.sample_method)
        scored = [(arch, all_scores.get(tuple(arch), -1e10)) for arch in sampled]
        scored.sort(key=lambda x: x[1], reverse=True)
        top_candidates = scored[:top_k]

        per_m_scoring_time = sum(all_proxy_times.get(tuple(a), 0.0) for a in sampled)

        print(f"\n[{os.getpid()}] === M={M}  top_k={top_k}  seed={args.seed} ===", flush=True)

        # SH
        sh_start = time.time()
        if top_k == 1:
            best_arch = top_candidates[0][0]
            best_val = 0.0
            num_cols = get_num_cols(table_data)
            stype_encoder_dict = construct_stype_encoder_dict(default_stype_encoder_cls_kwargs)
            final_model = PTNASResNet(
                channels=num_cols, out_channels=1, num_layers=len(best_arch),
                col_stats=table_data.col_stats, col_names_dict=table_data.col_names_dict,
                stype_encoder_dict=stype_encoder_dict, block_widths=best_arch,
                normalization="layer_norm", dropout_prob=0.2,
            ).to(device)
            sh_time = 0.0
        else:
            best_arch, best_val, final_model = successive_halving(
                top_candidates, table_data, is_regression,
                train_loader, val_loader,
                max_epochs=args.sh_max_epochs, min_epochs=args.sh_min_epochs,
                dataset_name=dataset_name, M=M, seed=args.seed,
                sh_detail_csv=args.sh_detail_csv,
                eta=args.eta,
            )
            sh_time = time.time() - sh_start

        # Lookup from bench
        arch_str = "-".join(map(str, best_arch))
        bench_entry = bench_lookup.get((dataset_name, arch_str))
        if bench_entry:
            test_metric = bench_entry["test_metric"]
            train_time = bench_entry["train_time"]
            test_time = bench_entry["test_time"]
        else:
            print(f"[{os.getpid()}] WARN: {arch_str} not in bench CSV, skipping", flush=True)
            del final_model
            continue

        total_time = per_m_scoring_time + sh_time + train_time + test_time

        print(f"[{os.getpid()}] M={M}  arch={best_arch}  test={test_metric:.4f}  total={total_time:.1f}s", flush=True)

        del final_model
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
        gc.collect()

        write_csv(args.output_csv, {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": dataset_name,
            "space_file": space_name,
            "pool_size": len(pool),
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
            "best_params": arch_str,
            "metric": "mae" if is_regression else "roc_auc",
        }, CSV_FIELDS)

    print(f"\n[{os.getpid()}] All done for {dataset_name}!", flush=True)


if __name__ == "__main__":
    main()
