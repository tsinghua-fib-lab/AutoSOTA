#!/usr/bin/env python3
"""
Score models with different weight initialization methods.
Supports: kaiming (default), xavier, lecun.

Usage:
    # ResNet
    python scripts/new_space/proxy_scores/score_init_strategies.py \
        --dataset event-user-attendance \
        --space resnet \
        --init_method xavier \
        --device cuda:0 \
        --output_csv datasets/nas_bench_tabular/space_resdnn/proxy_score/init/score_event-user-attendance_resnet_xavier.csv

    # BlockMixed
    python scripts/new_space/proxy_scores/score_init_strategies.py \
        --dataset avito-user-clicks \
        --space blockmixed \
        --init_method lecun \
        --device cuda:0 \
        --output_csv datasets/nas_bench_tabular/space_blockmixed/proxy_score/init/score_avito-user-clicks_blockmixed_lecun.csv
"""
from __future__ import annotations

import argparse
import ast
import csv
import fcntl
import json
import os
import random
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch_frame
from torch.utils.data import Subset

from model.base import construct_stype_encoder_dict, default_stype_encoder_cls_kwargs
from proxies.ptproxy import ptproxy_score
from proxies.ptproxy_blockmixed import ptproxy_blockmixed_score
from search_space import PTNASResNet
from search_space.block_mixed import PTNASBlockMixed
from utils.table_data import TableData


# ---------------------------------------------------------------------------
# Initialization methods
# ---------------------------------------------------------------------------
def apply_init(model, method: str):
    """Apply weight initialization to all linear layers."""
    for m in model.modules():
        if isinstance(m, nn.Linear):
            if method == "xavier":
                nn.init.xavier_uniform_(m.weight)
            elif method == "xavier_normal":
                nn.init.xavier_normal_(m.weight)
            elif method == "lecun":
                # LeCun: fan_in mode, normal distribution
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="linear")
            elif method == "kaiming":
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            elif method == "orthogonal":
                nn.init.orthogonal_(m.weight)
            # bias unchanged (zeros by default)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def load_cached_batch(dataset_name, space):
    """Load the cached encoded batch used by the new-space scoring scripts."""
    if space == "resnet":
        cache_path = _PROJECT_ROOT / "datasets" / "nas_bench_tabular" / "space_resdnn" / "proxy_score" / "ptproxy" / "batch_cache" / f"{dataset_name}_resnet_b32.pt"
    else:
        cache_path = _PROJECT_ROOT / "datasets" / "nas_bench_tabular" / "space_blockmixed" / "proxy_score" / "ptproxy" / "batch_cache" / f"{dataset_name}_c32_b32.pt"
    payload = torch.load(cache_path, map_location="cpu")
    return payload["batch_x"]


def score_resnet(arch_str, table_data, x_encoded, device, num_cols, init_method):
    block_widths = [int(x) for x in arch_str.split("-")]
    stype_encoder_dict = construct_stype_encoder_dict(default_stype_encoder_cls_kwargs)
    model = PTNASResNet(
        channels=num_cols, out_channels=1, num_layers=len(block_widths),
        col_stats=table_data.col_stats, col_names_dict=table_data.col_names_dict,
        stype_encoder_dict=stype_encoder_dict, block_widths=block_widths,
        normalization="layer_norm", dropout_prob=0.2,
    ).to(device)

    if init_method != "kaiming":
        apply_init(model, init_method)

    try:
        score, elapsed = ptproxy_blockmixed_score(
            arch=model, batch_data=x_encoded, device=str(device),
            respect_input=True,
        )
        return float(score), float(elapsed), ""
    except Exception as e:
        return float("nan"), 0.0, str(e)
    finally:
        del model
        torch.cuda.empty_cache()


def score_blockmixed(block_specs_str, table_data, x_encoded, device, num_cols, init_method, channels=32):
    block_specs_list = ast.literal_eval(block_specs_str)
    block_specs = [tuple(b) for b in block_specs_list]
    stype_encoder_dict = construct_stype_encoder_dict(default_stype_encoder_cls_kwargs)
    model = PTNASBlockMixed(
        channels=channels, out_channels=1, num_layers=len(block_specs),
        col_stats=table_data.col_stats, col_names_dict=table_data.col_names_dict,
        stype_encoder_dict=stype_encoder_dict, block_specs=block_specs,
    ).to(device)

    if init_method != "kaiming":
        apply_init(model, init_method)

    try:
        score, elapsed = ptproxy_blockmixed_score(
            arch=model, batch_data=x_encoded, device=str(device),
            respect_input=True,
        )
        return float(score), float(elapsed), ""
    except Exception as e:
        return float("nan"), 0.0, str(e)
    finally:
        del model
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--space", type=str, required=True, choices=["resnet", "blockmixed"])
    parser.add_argument("--init_method", type=str, required=True,
                        choices=["kaiming", "xavier", "xavier_normal", "lecun", "orthogonal"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--max_models", type=int, default=None)
    args = parser.parse_args()

    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    device = torch.device(args.device)

    print(f"[{os.getpid()}] dataset={args.dataset} space={args.space} init={args.init_method}", flush=True)

    # Load data
    data_dir = _PROJECT_ROOT / "datasets" / "fit-medium-table" / args.dataset
    table_data = TableData.load_from_dir(str(data_dir))
    num_cols = sum(len(v) for v in table_data.col_names_dict.values())

    # Load the cached encoded batch produced by the pTProxy scoring scripts.
    x_encoded = load_cached_batch(args.dataset, args.space)

    # Load architectures
    if args.space == "resnet":
        # Determine space file
        from relbench.base import TaskType
        is_regr = table_data.task_type == TaskType.REGRESSION
        space_file = "random_sampled_arch_resdnn_regression.txt" if is_regr else "random_sampled_arch_resdnn_classification.txt"
        with open(_PROJECT_ROOT / "datasets" / "nas_bench_tabular" / "space_resdnn" / "architecture" / space_file) as f:
            archs = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(archs)} ResNet archs from {space_file}", flush=True)
    else:
        # BlockMixed: load the finalized dataset-specific group selection.
        with open(_PROJECT_ROOT / "datasets" / "nas_bench_tabular" / "space_blockmixed" / "architecture" / "random_sampled_arch_blockmixed_metadata.json") as f:
            space_json = json.load(f)
        proxy_df = pd.read_csv(_PROJECT_ROOT / "datasets" / "nas_bench_tabular" / "space_blockmixed" / "proxy_score" / "ptproxy" / f"score_{args.dataset}_v1.csv")
        config = space_json[args.dataset][-1]
        groups = set(config["groups"])
        sub = proxy_df[proxy_df["group"].isin(groups)]
        archs = sub["block_specs"].tolist()
        print(f"Loaded {len(archs)} BlockMixed archs", flush=True)

    if args.max_models:
        archs = archs[:args.max_models]

    # Score
    fields = ["dataset", "space", "init_method", "architecture", "proxy_score", "proxy_time_seconds", "error"]
    csv_exists = os.path.exists(args.output_csv) and os.path.getsize(args.output_csv) > 0

    for i, arch in enumerate(archs):
        if args.space == "resnet":
            score, elapsed, err = score_resnet(arch, table_data, x_encoded, device, num_cols, args.init_method)
        else:
            score, elapsed, err = score_blockmixed(arch, table_data, x_encoded, device, num_cols, args.init_method)

        with open(args.output_csv, "a", newline="") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            w = csv.DictWriter(f, fieldnames=fields)
            if not csv_exists:
                w.writeheader()
                csv_exists = True
            w.writerow({
                "dataset": args.dataset, "space": args.space, "init_method": args.init_method,
                "architecture": arch, "proxy_score": score, "proxy_time_seconds": elapsed, "error": err,
            })
            fcntl.flock(f, fcntl.LOCK_UN)

        if (i + 1) % 100 == 0:
            print(f"  scored {i+1}/{len(archs)}", flush=True)

    print(f"[{os.getpid()}] Done. {len(archs)} models scored.", flush=True)


if __name__ == "__main__":
    main()
