#!/usr/bin/env python3
"""
pTNAS-Full: Single EA without size grouping

Difference from ptnas.py:
- Step 1: One EA (pop=60, gen=15) over all architectures → top 15 → SH
- No small/medium/large grouping
"""

import argparse
import copy
import csv
import gc
import math
import os
import random
import time
from datetime import datetime
from typing import List, Tuple, Dict

from tqdm import tqdm
import numpy as np
import torch
# Monkey-patch torch.load for PyTorch >= 2.6 compatibility with pre-2.6
# serialized torch_frame data files. Must run BEFORE any torch_frame import.
_original_torch_load = torch.load


def _patched_torch_load(f, *args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(f, *args, **kwargs)


torch.load = _patched_torch_load

from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torch.nn import L1Loss, BCEWithLogitsLoss
from torch.utils.data import Subset
from sklearn.metrics import mean_absolute_error, roc_auc_score
import torch_frame.data

from model.base import construct_stype_encoder_dict, default_stype_encoder_cls_kwargs
from utils import TableData
from search_space import PTNASMLP, PTNASResNet
from proxies.ptproxy import ptproxy_score
from search_algorithm import evolutionary_algorithm
from relbench.base import TaskType

device = torch.device("cpu")

# Profiler-derived dataset-specific timing constants from current pTNAS-Full logs
# (May 2026, eta-growth + checkpoint-reuse implementation).
# These are used only for time-aware planning when --given_time_budget is set.
# The values were estimated from prior logs by:
# - proxy_eval_seconds: selection_time / (population * generations)
# - SH time: a simple linear fit over SH workload
#            sh_time ~= sh_workload_seconds * workload(K, U, eta)
#
# The workload is the total candidate-epoch budget actually consumed by SH:
#   sum_r |C_r| * U_r
# where U_{r+1} = U_r * eta and |C_{r+1}| = ceil(|C_r| / eta).
#
# This keeps the planner simple while allowing different datasets to use
# different time profiles instead of a single global constant.
DEFAULT_PROXY_EVAL_SECONDS = 0.00246
DEFAULT_SH_WORKLOAD_SECONDS = 0.72
PROFILED_FINAL_TRAIN_SECONDS = 9.62

DATASET_TIME_PROFILES = {
    "avito-user-clicks": {
        "proxy_eval_seconds": 0.002418886,
        "sh_workload_seconds": 0.686087926,
    },
    "event-user-attendance": {
        "proxy_eval_seconds": 0.002473236,
        "sh_workload_seconds": 0.403128171,
    },
    "event-user-repeat": {
        "proxy_eval_seconds": 0.002156231,
        "sh_workload_seconds": 0.295340761,
    },
    "hm-item-sales": {
        "proxy_eval_seconds": 0.002374667,
        "sh_workload_seconds": 2.457773336,
    },
    "ratebeer-beer-positive": {
        "proxy_eval_seconds": 0.002565407,
        "sh_workload_seconds": 0.582893054,
    },
    "ratebeer-user-active": {
        "proxy_eval_seconds": 0.003115209,
        "sh_workload_seconds": 0.382646807,
    },
    "trial-site-success": {
        "proxy_eval_seconds": 0.002071836,
        "sh_workload_seconds": 0.587443852,
    },
    "trial-study-outcome": {
        "proxy_eval_seconds": 0.003687986,
        "sh_workload_seconds": 0.360230144,
    },
}


def deactivate_dropout(net):
    for module in net.modules():
        if isinstance(module, (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)):
            module.eval()
            for param in module.parameters():
                param.requires_grad = False


def test(net, loader, early_stop=-1, is_regression=False):
    pred_list, y_list = [], []
    early_stop = early_stop if early_stop > 0 else len(loader.dataset)
    if not is_regression:
        net.eval()
    for idx, batch in tqdm(enumerate(loader), total=len(loader), leave=False, desc="Testing", disable=True):
        with torch.no_grad():
            batch = batch.to(device)
            y = batch.y.float()
            pred = net(batch)
            pred = pred.view(-1) if pred.size(1) == 1 else pred
            pred_list.append(pred.detach().cpu())
            y_list.append(y.detach().cpu())
        if idx > early_stop:
            break
    pred_list = torch.cat(pred_list, dim=0)
    pred_logits = pred_list
    pred_list = torch.sigmoid(pred_list)
    y_list = torch.cat(y_list, dim=0).numpy()
    return pred_logits.numpy(), pred_list.numpy(), y_list


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_num_cols(table_data):
    return sum(len(names) for names in table_data.col_names_dict.values())


def build_model(space_name, arch, table_data, stype_encoder_dict, dropout_prob=0.2, out_channels=1):
    num_cols = get_num_cols(table_data)
    if space_name == 'mlp':
        return PTNASMLP(
            channels=num_cols, out_channels=out_channels,
            num_layers=len(arch) + 1,
            col_stats=table_data.col_stats,
            col_names_dict=table_data.col_names_dict,
            stype_encoder_dict=stype_encoder_dict,
            hidden_dims=arch,
            normalization='layer_norm',
            dropout_prob=dropout_prob,
        ).to(device)
    return PTNASResNet(
        channels=num_cols, out_channels=out_channels,
        num_layers=len(arch),
        col_stats=table_data.col_stats,
        col_names_dict=table_data.col_names_dict,
        stype_encoder_dict=stype_encoder_dict,
        block_widths=arch,
        normalization='layer_norm',
        dropout_prob=dropout_prob,
    ).to(device)


def clone_state_dict_to_cpu(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def compute_sh_workload(top_k: int, unit_epochs: int = 1, eta: int = 3):
    cur_candidates = max(1, int(top_k))
    cur_epochs = max(1, int(unit_epochs))
    workload = 0
    while cur_candidates > 1:
        workload += cur_candidates * cur_epochs
        cur_candidates = math.ceil(cur_candidates / eta)
        cur_epochs *= eta
    return workload


def get_dataset_time_profile(dataset_name: str):
    return DATASET_TIME_PROFILES.get(
        dataset_name,
        {
            "proxy_eval_seconds": DEFAULT_PROXY_EVAL_SECONDS,
            "sh_workload_seconds": DEFAULT_SH_WORKLOAD_SECONDS,
        },
    )


def resolve_time_aware_search_hparams(
    given_time_budget: float,
    dataset_name: str,
    sh_min_epochs: int = 1,
    eta: int = 3,
    mk_ratio: float = 60.0,
):
    """
    Infer (population, generations, top_k) from a search-time budget.

    We keep the planner intentionally simple and close to the paper:
      - population = 4 * generations
      - explored models M ~= population * generations
      - M / K ~= mk_ratio  =>  top_k ~= M / mk_ratio

    The estimated search time is:
      T1 = M * t1
      T2 = workload(K, U, eta) * t2

    where t1 and t2 are dataset-specific constants profiled from prior logs.
    We enumerate generations and choose the largest configuration whose
    estimated search time does not exceed the given budget.

    Final retraining is not included here; it is reported separately.
    """
    if given_time_budget <= 0:
        raise ValueError("given_time_budget must be positive")
    if mk_ratio <= 0:
        raise ValueError("mk_ratio must be positive")

    profile = get_dataset_time_profile(dataset_name)
    best_cfg = None
    for generations in range(1, 61):
        population = 4 * generations
        explored_models = population * generations
        top_k = max(1, round(explored_models / mk_ratio))
        est_selection = profile["proxy_eval_seconds"] * explored_models
        sh_workload = compute_sh_workload(top_k, unit_epochs=sh_min_epochs, eta=eta)
        est_sh = profile["sh_workload_seconds"] * sh_workload
        est_search_total = est_selection + est_sh

        if est_search_total <= given_time_budget:
            best_cfg = {
                "population": population,
                "generations": generations,
                "top_k": top_k,
                "dataset_name": dataset_name,
                "explored_models": explored_models,
                "sh_workload": sh_workload,
                "est_selection_seconds": est_selection,
                "est_sh_seconds": est_sh,
                "est_search_total_seconds": est_search_total,
            }

    if best_cfg is None:
        generations = 1
        population = 4
        explored_models = population * generations
        top_k = max(1, round(explored_models / mk_ratio))
        est_selection = profile["proxy_eval_seconds"] * explored_models
        sh_workload = compute_sh_workload(top_k, unit_epochs=sh_min_epochs, eta=eta)
        est_sh = profile["sh_workload_seconds"] * sh_workload
        best_cfg = {
            "population": population,
            "generations": generations,
            "top_k": top_k,
            "dataset_name": dataset_name,
            "explored_models": explored_models,
            "sh_workload": sh_workload,
            "est_selection_seconds": est_selection,
            "est_sh_seconds": est_sh,
            "est_search_total_seconds": est_selection + est_sh,
        }

    return best_cfg


def prepare_sample_batch_for_proxy(table_data, space_name, sample_size=256):
    print(f"\n🔍 Preparing sample batch for proxy evaluation...")
    sample_size = min(sample_size, len(table_data.train_tf))
    sample_indices = random.sample(range(len(table_data.train_tf)), sample_size)
    sample_subset = Subset(table_data.train_tf, sample_indices)
    sample_loader = torch_frame.data.DataLoader(sample_subset, batch_size=min(4, sample_size), shuffle=False)
    batch = next(iter(sample_loader)).to(device)

    stype_encoder_dict = construct_stype_encoder_dict(default_stype_encoder_cls_kwargs)
    num_cols = get_num_cols(table_data)

    if space_name == 'mlp':
        temp_model = PTNASMLP(
            channels=num_cols, out_channels=1, num_layers=2,
            col_stats=table_data.col_stats, col_names_dict=table_data.col_names_dict,
            stype_encoder_dict=stype_encoder_dict, hidden_dims=[num_cols],
            normalization='layer_norm', dropout_prob=0.2,
        ).to(device)
    else:
        temp_model = PTNASResNet(
            channels=num_cols, out_channels=1, num_layers=2,
            col_stats=table_data.col_stats, col_names_dict=table_data.col_names_dict,
            stype_encoder_dict=stype_encoder_dict, block_widths=[num_cols, num_cols],
            normalization='layer_norm', dropout_prob=0.2,
        ).to(device)

    with torch.no_grad():
        x_encoded, _ = temp_model.encoder(batch)
        x_encoded = torch.mean(x_encoded, dim=1) if space_name == 'mlp' else x_encoded.view(x_encoded.size(0), -1)

    del temp_model
    if str(device).startswith('cuda'):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    print(f"✅ Encoded features: {x_encoded.shape}")
    return x_encoded


def create_evaluation_function(sample_batch_x, table_data, col_stats, col_names_dict,
                                stype_encoder_dict, out_channels, space_name):
    score_cache: Dict[Tuple[int, ...], float] = {}

    def evaluate_func(arch):
        arch_key = tuple(arch)
        if arch_key in score_cache:
            return score_cache[arch_key]
        try:
            model = build_model(
                space_name=space_name,
                arch=arch,
                table_data=table_data,
                stype_encoder_dict=stype_encoder_dict,
                dropout_prob=0.2,
                out_channels=out_channels,
            )
            net_for_proxy = model.mlp if space_name == 'mlp' else model.backbone

            score, _ = ptproxy_score(
                arch=net_for_proxy, batch_data=sample_batch_x, device=str(device),
                use_wo_embedding=False, linearize_target=None,
                epsilon=1e-5, weight_mode="traj_width", use_fp64=False,
            )
            del model
        except Exception as e:
            print(f"  ⚠️  Error on arch {arch}: {e}")
            score = -1e10
        finally:
            if str(device).startswith('cuda'):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        score_cache[arch_key] = float(score)
        return float(score)

    evaluate_func.score_cache = score_cache
    return evaluate_func


def full_ea_selection(
        space_name: str,
        table_data,
        sample_batch_x,
        col_stats, col_names_dict, stype_encoder_dict,
        out_channels: int,
        top_k: int = 15,
        population_size: int = 60,
        generations: int = 15,
) -> List[Tuple[List[int], float, str, float]]:
    """Single evolutionary search over full search space, return top_k candidates."""
    print(f"\n🎯 Full EA Selection (no grouping)")
    print(f"   population={population_size}, generations={generations}, top_k={top_k}")

    model_class = PTNASMLP if space_name == 'mlp' else PTNASResNet

    evaluate_func = create_evaluation_function(
        sample_batch_x=sample_batch_x, table_data=table_data,
        col_stats=col_stats, col_names_dict=col_names_dict,
        stype_encoder_dict=stype_encoder_dict, out_channels=out_channels,
        space_name=space_name,
    )

    ea_results = evolutionary_algorithm(
        model_class=model_class,
        evaluate_func=evaluate_func,
        population_size=population_size,
        generations=generations,
        elite_size=10,
        mutation_rate=0.3,
    )

    ea_results.sort(key=lambda x: x[1], reverse=True)
    top_models = ea_results[:top_k]

    print(f"   ✅ EA complete, top {len(top_models)} candidates:")
    for i, (arch, score) in enumerate(top_models):
        print(f"     {i+1}. {arch} (score: {score:.4f})")

    if str(device).startswith('cuda'):
        torch.cuda.empty_cache()
    gc.collect()

    # Return in same format as ptnas.py: (arch, proxy_score, group, val_score)
    return [(arch, score, 'full', None) for arch, score in top_models]


def successive_halving(selected_models, space_name, table_data, is_regression,
                       max_epochs=50, min_epochs=1, eta=3,
                       train_batch_size=256, dropout_prob=0.2,
                       train_lr=0.001, max_batches_per_epoch=20,
                       early_stop_patience=10):
    print(f"\n🏆 Successive Halving Selection")
    print(f"   Candidates: {len(selected_models)}")

    train_loader = torch_frame.data.DataLoader(table_data.train_tf, batch_size=train_batch_size, shuffle=True)
    val_loader = torch_frame.data.DataLoader(table_data.val_tf, batch_size=train_batch_size, shuffle=False)

    candidates = [
        {
            "arch": arch,
            "proxy_score": proxy_score,
            "group": group,
            "val_score": val_score,
            "state_dict": None,
            "trained_epochs": 0,
        }
        for arch, proxy_score, group, val_score in selected_models
    ]
    current_epochs = min_epochs

    while len(candidates) > 1 and current_epochs <= max_epochs:
        print(f"   Round: {len(candidates)} candidates, +{current_epochs} epochs")
        candidate_scores = []

        for i, candidate in enumerate(candidates):
            arch = candidate["arch"]
            proxy_score = candidate["proxy_score"]
            group = candidate["group"]
            print(f"     Training {i+1}/{len(candidates)}: {arch}")
            stype_encoder_dict = construct_stype_encoder_dict(default_stype_encoder_cls_kwargs)
            model = build_model(
                space_name=space_name,
                arch=arch,
                table_data=table_data,
                stype_encoder_dict=stype_encoder_dict,
                dropout_prob=dropout_prob,
                out_channels=1,
            )
            if candidate["state_dict"] is not None:
                model.load_state_dict(candidate["state_dict"])

            model, _ = train_model(model, train_loader, val_loader, is_regression,
                                   num_epochs=current_epochs, lr=train_lr,
                                   max_batches_per_epoch=max_batches_per_epoch,
                                   early_stop_patience=early_stop_patience)
            val_score = evaluate_model(model, val_loader, is_regression)[0]
            candidate_scores.append({
                "arch": arch,
                "proxy_score": proxy_score,
                "group": group,
                "val_score": val_score,
                "state_dict": clone_state_dict_to_cpu(model),
                "trained_epochs": candidate["trained_epochs"] + current_epochs,
            })

            del model
            if str(device).startswith('cuda'):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()

        candidate_scores.sort(key=lambda x: x["val_score"], reverse=not is_regression)
        keep_count = max(1, math.ceil(len(candidates) / eta))
        candidates = candidate_scores[:keep_count]
        print(
            f"     Kept top {len(candidates)}, best val: {candidates[0]['val_score']:.4f}, "
            f"winner cumulative epochs: {candidates[0]['trained_epochs']}"
        )

        if current_epochs < max_epochs:
            current_epochs = min(max_epochs, current_epochs * eta)
        else:
            break

    winner = candidates[0]
    best_arch = winner["arch"]
    best_val_score = winner["val_score"]
    print(
        f"   🏆 Best: {best_arch}, val={best_val_score:.4f}, "
        f"cumulative epochs={winner['trained_epochs']}"
    )
    return best_arch, best_val_score, winner["state_dict"], winner["trained_epochs"]


def train_model(model, train_loader, val_loader, is_regression,
                num_epochs=200, lr=0.001, max_batches_per_epoch=20, early_stop_patience=10):
    print(f"\n  Training: epochs={num_epochs}, lr={lr}")
    train_start = time.time()

    loss_fn = L1Loss() if is_regression else BCEWithLogitsLoss()
    if is_regression:
        deactivate_dropout(model)
    higher_is_better = not is_regression

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-5)
    warmup_epochs = max(1, int(num_epochs * 0.1))
    cosine_epochs = max(1, num_epochs - warmup_epochs)
    scheduler = SequentialLR(
        optimizer,
        [
            LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs),
            CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=1e-5),
        ],
        milestones=[warmup_epochs],
    )
    model.to(device)
    patience = 0
    best_val_metric = -math.inf if higher_is_better else math.inf
    best_model_state = None

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

        val_logits, _, val_pred_hat = test(model, val_loader, is_regression=is_regression)

        val_metric = (mean_absolute_error(val_logits, val_pred_hat) if is_regression
                      else roc_auc_score(val_pred_hat, val_logits) if len(np.unique(val_pred_hat)) > 1 else 0.5)

        improved = (val_metric > best_val_metric if higher_is_better else val_metric < best_val_metric)
        if improved:
            best_val_metric = val_metric
            best_model_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience > early_stop_patience:
                print(f"  Early stopped at epoch {epoch}")
                break

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: val={val_metric:.4f}")

    if best_model_state:
        model.load_state_dict(best_model_state)

    train_time = time.time() - train_start
    print(f"✅ Training completed! Best val: {best_val_metric:.6f}, time: {train_time:.2f}s")
    return model, train_time


def evaluate_model(model, test_loader, is_regression):
    start_time = time.time()
    test_logits, test_pred_hat, test_y = test(model, test_loader, is_regression=is_regression)
    inference_time = time.time() - start_time

    if is_regression:
        metric = mean_absolute_error(test_y, test_logits)
    else:
        metric = roc_auc_score(test_y, test_pred_hat) if len(np.unique(test_y)) > 1 else 0.5
    return metric, inference_time


def main():
    parser = argparse.ArgumentParser(description='pTNAS-Full')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--space_name', type=str, required=True, choices=['mlp', 'resnet'])
    parser.add_argument('--output_csv', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--population', type=int, default=60)
    parser.add_argument('--generations', type=int, default=15)
    parser.add_argument('--top_k', type=int, default=15)
    parser.add_argument('--given_time_budget', type=float, default=None)
    parser.add_argument('--mk_ratio', type=float, default=60.0)
    parser.add_argument('--sh_min_epochs', type=int, default=1)
    parser.add_argument('--sh_max_epochs', type=int, default=50)
    parser.add_argument('--eta', type=int, default=3)
    parser.add_argument('--final_epochs', type=int, default=200)
    parser.add_argument('--final_lr', type=float, default=0.001)
    parser.add_argument('--final_batch_size', type=int, default=256)
    parser.add_argument('--final_dropout', type=float, default=0.2)
    parser.add_argument('--final_max_batches', type=int, default=20)
    parser.add_argument('--final_early_stop', type=int, default=10)
    args = parser.parse_args()

    global device
    device = torch.device(args.device)
    set_seed(args.seed)

    time_aware_cfg = None
    dataset_name = os.path.basename(args.data_dir)

    if args.given_time_budget is not None:
        time_aware_cfg = resolve_time_aware_search_hparams(
            args.given_time_budget,
            dataset_name=dataset_name,
            sh_min_epochs=args.sh_min_epochs,
            eta=args.eta,
            mk_ratio=args.mk_ratio,
        )
        args.population = time_aware_cfg["population"]
        args.generations = time_aware_cfg["generations"]
        args.top_k = time_aware_cfg["top_k"]

    print("=" * 80)
    print(f"🧬 pTNAS-Full: Single EA (pop={args.population}, gen={args.generations}, top_k={args.top_k})")
    print("=" * 80)
    print(f"Data: {args.data_dir}")
    print(f"Space: {args.space_name}, Device: {args.device}")
    if time_aware_cfg is not None:
        profile = get_dataset_time_profile(dataset_name)
        print(
            "Time-aware search config: "
            f"budget={args.given_time_budget:.2f}s, "
            f"mk_ratio={args.mk_ratio:.0f}, "
            f"est_selection={time_aware_cfg['est_selection_seconds']:.2f}s, "
            f"est_sh={time_aware_cfg['est_sh_seconds']:.2f}s, "
            f"est_search_total={time_aware_cfg['est_search_total_seconds']:.2f}s"
        )
        print(
            "Dataset profile: "
            f"proxy_eval={profile['proxy_eval_seconds']:.6f}s, "
            f"sh_workload_unit={profile['sh_workload_seconds']:.4f}s, "
            f"workload={time_aware_cfg['sh_workload']}"
        )
        print(
            "Final-train reference: "
            f"final_train_ref={PROFILED_FINAL_TRAIN_SECONDS:.2f}s"
        )
    print(
        "Final train config: "
        f"epochs={args.final_epochs}, lr={args.final_lr}, batch_size={args.final_batch_size}, "
        f"dropout={args.final_dropout}, max_batches={args.final_max_batches}, early_stop={args.final_early_stop}"
    )

    table_data = TableData.load_from_dir(args.data_dir)
    is_regression = table_data.task_type == TaskType.REGRESSION
    print(f"Task type: {table_data.task_type}")

    x_encoded = prepare_sample_batch_for_proxy(table_data, args.space_name, sample_size=256)
    stype_encoder_dict = construct_stype_encoder_dict(default_stype_encoder_cls_kwargs)

    # Step 1: Full evolutionary selection
    print(f"\n🎯 Step 1: Full EA selection")
    selection_start = time.time()
    selected_models = full_ea_selection(
        space_name=args.space_name,
        table_data=table_data,
        sample_batch_x=x_encoded,
        col_stats=table_data.col_stats,
        col_names_dict=table_data.col_names_dict,
        stype_encoder_dict=stype_encoder_dict,
        out_channels=1,
        top_k=args.top_k,
        population_size=args.population,
        generations=args.generations,
    )
    selection_time = time.time() - selection_start
    print(f"✅ Selection complete: {selection_time:.2f}s")

    # Step 2: Successive Halving
    print(f"\n🏆 Step 2: Successive Halving")
    sh_start_time = time.time()
    best_arch, best_val_score, best_state_dict, sh_trained_epochs = successive_halving(
        selected_models=selected_models,
        space_name=args.space_name,
        table_data=table_data,
        is_regression=is_regression,
        max_epochs=args.sh_max_epochs,
        min_epochs=args.sh_min_epochs,
        eta=args.eta,
        train_batch_size=args.final_batch_size,
        dropout_prob=args.final_dropout,
        train_lr=args.final_lr,
        max_batches_per_epoch=args.final_max_batches,
        early_stop_patience=args.final_early_stop,
    )
    sh_time = time.time() - sh_start_time
    print(f"✅ Best architecture: {best_arch}, val={best_val_score:.4f}")
    print(f"   Successive halving time: {sh_time:.2f}s")

    # Step 3: Final training
    print(f"\n🚀 Step 3: Training final model")
    num_cols = get_num_cols(table_data)
    stype_encoder_dict = construct_stype_encoder_dict(default_stype_encoder_cls_kwargs)

    if args.space_name == 'mlp':
        final_model = build_model(
            space_name='mlp',
            arch=best_arch,
            table_data=table_data,
            stype_encoder_dict=stype_encoder_dict,
            dropout_prob=args.final_dropout,
            out_channels=1,
        )
    else:
        final_model = build_model(
            space_name='resnet',
            arch=best_arch,
            table_data=table_data,
            stype_encoder_dict=stype_encoder_dict,
            dropout_prob=args.final_dropout,
            out_channels=1,
        )

    if is_regression:
        deactivate_dropout(final_model)

    if best_state_dict is not None:
        final_model.load_state_dict(best_state_dict)

    train_loader = torch_frame.data.DataLoader(table_data.train_tf, batch_size=args.final_batch_size, shuffle=True)
    val_loader = torch_frame.data.DataLoader(table_data.val_tf, batch_size=args.final_batch_size, shuffle=False)
    test_loader = torch_frame.data.DataLoader(table_data.test_tf, batch_size=256, shuffle=False)

    extra_final_epochs = max(0, args.final_epochs - sh_trained_epochs)
    print(f"   Warm-starting from SH checkpoint trained for {sh_trained_epochs} epochs")
    print(f"   Additional final-training epochs: {extra_final_epochs}")
    if extra_final_epochs > 0:
        final_model, train_time = train_model(
            final_model, train_loader, val_loader, is_regression,
            num_epochs=extra_final_epochs,
            lr=args.final_lr,
            max_batches_per_epoch=args.final_max_batches,
            early_stop_patience=args.final_early_stop,
        )
    else:
        train_time = 0.0
    print(f"✅ Final training: {train_time:.2f}s")

    # Step 4: Test
    print(f"\n🧪 Step 4: Testing")
    test_metric, inference_time = evaluate_model(final_model, test_loader, is_regression)
    print(f"✅ Test metric: {test_metric:.4f}, inference: {inference_time:.2f}s")

    total_time = selection_time + sh_time + train_time + inference_time

    # Save CSV
    csv_exists = os.path.exists(args.output_csv)
    result_row = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'dataset': os.path.basename(args.data_dir),
        'architecture': args.space_name,
        'selection_time_seconds': selection_time,
        'sh_time_seconds': sh_time,
        'final_train_time_seconds': train_time,
        'inference_time_seconds': inference_time,
        'total_time_seconds': total_time,
        'best_val_metric': best_val_score,
        'final_test_metric': test_metric,
        'best_params': str(best_arch),
        'metric': 'mae' if is_regression else 'roc_auc',
        'ea_population': args.population,
        'ea_generations': args.generations,
        'top_k': args.top_k,
        'winner_sh_epochs': sh_trained_epochs,
        'final_extra_epochs': extra_final_epochs,
    }
    with open(args.output_csv, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=result_row.keys())
        if not csv_exists:
            writer.writeheader()
        writer.writerow(result_row)

    print(f"\n{'=' * 80}")
    print(f"🎉 pTNAS-Full Complete!")
    print(f"{'=' * 80}")
    print(f"   Best architecture: {best_arch}")
    print(f"   Test metric: {test_metric:.4f}")
    print(f"   Selection time: {selection_time:.2f}s")
    print(f"   SH time: {sh_time:.2f}s")
    print(f"   Training time: {train_time:.2f}s")
    print(f"   Winner SH epochs: {sh_trained_epochs}")
    print(f"   Extra final epochs: {extra_final_epochs}")
    print(f"   Total time: {total_time:.2f}s")
    print(f"📁 Results: {args.output_csv}")


if __name__ == "__main__":
    main()
