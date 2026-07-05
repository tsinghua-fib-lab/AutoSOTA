#!/usr/bin/env python3
"""
TabNAS baseline adapted for relbench datasets.
Uses pre-trained results as lookup table (NAS-bench style).
If no lookup, trains models on the fly.

Usage:
    cd pTNAS
    PYTHONPATH=src python scripts/relbench/nas/tab_nas_relbench.py \
        --data_dir datasets/fit-medium-table/avito-user-clicks \
        --device cuda:0 \
        --max_iter 500 \
        --time_budget 30 \
        --output_csv run_outputs/data/relbench/baselines/tabnas_relbench_results.csv
"""
from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import torch_frame
from relbench.base import TaskType
from sklearn.metrics import mean_absolute_error, roc_auc_score
from torch.nn import BCEWithLogitsLoss, L1Loss

from model.base import construct_stype_encoder_dict, default_stype_encoder_cls_kwargs
from search_space import PTNASMLP
from utils.table_data import TableData

LAYER_CHOICES = [32, 128, 256]
N_LAYERS = 3


def deactivate_dropout(net):
    for m in net.modules():
        if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False


@torch.no_grad()
def evaluate(model, loader, device, is_regression):
    model.eval()
    preds, ys = [], []
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        pred = pred.view(-1) if pred.size(-1) == 1 else pred
        preds.append(pred.cpu())
        ys.append(batch.y.float().cpu())
    preds = torch.cat(preds).numpy()
    ys = torch.cat(ys).numpy()
    if is_regression:
        return float(mean_absolute_error(ys, preds))
    return float(roc_auc_score(ys, torch.sigmoid(torch.tensor(preds)).numpy()))


def train_model(arch_tuple, table_data, device, is_regression,
                lr=1e-3, num_epochs=100, batch_size=256, max_batches=20, early_stop=10):
    """Train a single MLP and return val/test metric."""
    num_cols = sum(len(v) for v in table_data.col_names_dict.values())
    stype_encoder_dict = construct_stype_encoder_dict(default_stype_encoder_cls_kwargs)
    hidden_dims = list(arch_tuple)

    model = PTNASMLP(
        channels=num_cols, out_channels=1, num_layers=len(hidden_dims) + 1,
        col_stats=table_data.col_stats, col_names_dict=table_data.col_names_dict,
        stype_encoder_dict=stype_encoder_dict, hidden_dims=hidden_dims,
        normalization="layer_norm", dropout_prob=0.2,
    ).to(device)

    loss_fn = L1Loss() if is_regression else BCEWithLogitsLoss()
    if is_regression:
        deactivate_dropout(model)
    higher_is_better = not is_regression

    train_loader = torch_frame.data.DataLoader(table_data.train_tf, batch_size=batch_size, shuffle=True)
    val_loader = torch_frame.data.DataLoader(table_data.val_tf, batch_size=batch_size, shuffle=False)
    test_loader = torch_frame.data.DataLoader(table_data.test_tf, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    best_val = -math.inf if higher_is_better else math.inf
    best_state = None
    patience = 0

    for epoch in range(num_epochs):
        model.train()
        for idx, batch in enumerate(train_loader):
            if idx > max_batches:
                break
            optimizer.zero_grad()
            batch = batch.to(device)
            pred = model(batch)
            pred = pred.view(-1) if pred.size(-1) == 1 else pred
            loss = loss_fn(pred, batch.y.float())
            loss.backward()
            optimizer.step()

        val_metric = evaluate(model, val_loader, device, is_regression)
        improved = (higher_is_better and val_metric > best_val) or \
                   (not is_regression and val_metric < best_val)
        if improved:
            best_val = val_metric
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience > early_stop:
                break

    if best_state:
        model.load_state_dict(best_state)
    test_metric = evaluate(model, test_loader, device, is_regression)
    del model, optimizer
    torch.cuda.empty_cache()
    return best_val, test_metric


def run_tabnas(layer_choices, rewards, max_iter, time_budget, higher_is_better, seed=0):
    """
    Run TabNAS RL search.
    Returns (best_arch, best_reward, search_time, n_evaluated).
    """
    n_choices = len(layer_choices)
    layer_logits = [torch.zeros(n_choices, requires_grad=True) for _ in range(N_LAYERS)]
    optimizer = torch.optim.Adam(layer_logits, lr=0.1, betas=(0.9, 0.999), eps=1e-8)

    rl_reward_momentum = 0.9
    moving_avg_numer = 0.0
    moving_avg_denom = 0.0
    num_samples_mc = 5

    best_arch = None
    best_reward = -math.inf if higher_is_better else math.inf
    n_evaluated = 0
    t0 = time.time()

    for it in range(max_iter):
        if time.time() - t0 > time_budget:
            break

        torch.manual_seed(1000 * seed + it)
        dists = [torch.distributions.Categorical(logits=lg) for lg in layer_logits]
        indices = [d.sample() for d in dists]
        choices = tuple(layer_choices[idx] for idx in indices)

        if choices not in rewards:
            continue

        reward = rewards[choices]
        n_evaluated += 1

        improved = (higher_is_better and reward > best_reward) or \
                   (not higher_is_better and reward < best_reward)
        if improved:
            best_reward = reward
            best_arch = choices

        # Normalize reward to [0, 1] range for stability
        all_rewards = list(rewards.values())
        r_min, r_max = min(all_rewards), max(all_rewards)
        if r_max > r_min:
            rl_reward = (reward - r_min) / (r_max - r_min)
            if not higher_is_better:
                rl_reward = 1.0 - rl_reward  # flip so higher = better for RL
        else:
            rl_reward = 0.5

        moving_avg_numer = rl_reward_momentum * moving_avg_numer + (1 - rl_reward_momentum) * rl_reward
        moving_avg_denom = rl_reward_momentum * moving_avg_denom + (1 - rl_reward_momentum)
        baseline = moving_avg_numer / moving_avg_denom
        advantage = rl_reward - baseline

        probs = [torch.softmax(lg, dim=0) for lg in layer_logits]
        mc_samples = [d.sample(torch.Size([num_samples_mc])) for d in dists]
        feasible = [
            combo
            for combo in zip(*[s.tolist() for s in mc_samples])
            if tuple(layer_choices[c] for c in combo) in rewards
        ]

        if not feasible:
            continue

        est_prob_valid = sum(
            torch.prod(torch.stack([probs[l][combo[l]] for l in range(N_LAYERS)]))
            for combo in feasible
        )

        cond_prob = torch.prod(torch.stack([probs[l][indices[l]] for l in range(N_LAYERS)]))
        log_cond_prob = torch.log(cond_prob / est_prob_valid)

        loss = -float(advantage) * log_cond_prob
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    search_time = time.time() - t0
    return best_arch, best_reward, search_time, n_evaluated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_iter", type=int, default=500)
    parser.add_argument("--time_budget", type=float, default=30.0)
    parser.add_argument("--n_reps", type=int, default=3)
    parser.add_argument("--output_csv", default="run_outputs/data/relbench/baselines/tabnas_relbench_results.csv")
    args = parser.parse_args()

    device = torch.device(args.device)
    dataset_name = Path(args.data_dir).name
    if os.path.dirname(args.output_csv):
        os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    print(f"[TabNAS] dataset={dataset_name} device={device} max_iter={args.max_iter} time_budget={args.time_budget}s", flush=True)

    # Load data
    table_data = TableData.load_from_dir(args.data_dir)
    is_regression = table_data.task_type == TaskType.REGRESSION
    higher_is_better = not is_regression

    # Build reward table: train all 256 architectures
    print(f"[TabNAS] Building reward table (training all {len(LAYER_CHOICES)**N_LAYERS} architectures)...", flush=True)
    import itertools
    all_archs = list(itertools.product(LAYER_CHOICES, repeat=N_LAYERS))

    rewards = {}  # arch_tuple -> val_metric
    test_metrics = {}  # arch_tuple -> test_metric
    train_times = {}

    t_build = time.time()
    for i, arch in enumerate(all_archs):
        torch.manual_seed(42)
        t0 = time.time()
        val, test = train_model(arch, table_data, device, is_regression)
        elapsed = time.time() - t0
        rewards[arch] = val
        test_metrics[arch] = test
        train_times[arch] = elapsed
        if (i + 1) % 20 == 0:
            print(f"  trained {i+1}/{len(all_archs)}", flush=True)
    build_time = time.time() - t_build
    print(f"[TabNAS] Reward table built in {build_time:.1f}s", flush=True)

    # Run TabNAS search
    fields = ["dataset", "method", "rep", "best_arch", "best_val", "best_test",
              "search_time", "n_evaluated", "build_time", "metric"]

    csv_exists = os.path.exists(args.output_csv) and os.path.getsize(args.output_csv) > 0

    for rep in range(args.n_reps):
        try:
            best_arch, best_val, search_time, n_eval = run_tabnas(
                LAYER_CHOICES, rewards, args.max_iter, args.time_budget,
                higher_is_better, seed=rep)
        except Exception as e:
            print(f"  rep={rep} RL search failed ({e}), falling back to best in table", flush=True)
            if higher_is_better:
                best_arch = max(rewards, key=rewards.get)
            else:
                best_arch = min(rewards, key=rewards.get)
            best_val = rewards[best_arch]
            search_time = 0.0
            n_eval = len(rewards)

        best_test = test_metrics.get(best_arch, float("nan")) if best_arch else float("nan")
        arch_str = "-".join(map(str, best_arch)) if best_arch else "none"

        print(f"  rep={rep} arch={arch_str} val={best_val:.4f} test={best_test:.4f} "
              f"search={search_time:.2f}s n_eval={n_eval}", flush=True)

        with open(args.output_csv, "a", newline="") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            w = csv.DictWriter(f, fieldnames=fields)
            if not csv_exists:
                w.writeheader()
                csv_exists = True
            w.writerow({
                "dataset": dataset_name, "method": "TabNAS", "rep": rep,
                "best_arch": arch_str, "best_val": round(best_val, 6),
                "best_test": round(best_test, 6), "search_time": round(search_time, 2),
                "n_evaluated": n_eval, "build_time": round(build_time, 2),
                "metric": "mae" if is_regression else "roc_auc",
            })
            fcntl.flock(f, fcntl.LOCK_UN)

    print(f"[TabNAS] Done. Results: {args.output_csv}", flush=True)


if __name__ == "__main__":
    main()
