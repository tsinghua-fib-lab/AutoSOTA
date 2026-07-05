"""
Batch trainer for PTNASResNet search space.
Loads dataset ONCE, trains a list of architectures sequentially.

Usage (called by run_train_resdnn.sh):
    python scripts/new_space/training/train_resdnn_batch.py \
        --data_dir datasets/fit-medium-table/avito-user-clicks \
        --architectures "64-128,32-64-256,..." \
        --device cuda:0 \
        --output_csv datasets/nas_bench_tabular/space_resdnn/training/resnet_pool_results.csv
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

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import torch
import torch_frame
from relbench.base import TaskType
from sklearn.metrics import mean_absolute_error, roc_auc_score
from torch.nn import BCEWithLogitsLoss, L1Loss

from model.base import construct_stype_encoder_dict, default_stype_encoder_cls_kwargs
from search_space import PTNASMLP, PTNASResNet
from utils.resource import get_text_embedder_cfg
from utils.table_data import TableData

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--architectures", type=str, required=True,
                    help="Comma-separated arch strings, e.g. '64-128,32-64-256'")
parser.add_argument("--space_name", type=str, default="resnet", choices=["resnet", "mlp"])
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--output_csv", type=str,
                    default=str(_PROJECT_ROOT / "datasets" / "nas_bench_tabular" / "space_resdnn" / "training" / "resnet_pool_results.csv"))
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--num_epochs", type=int, default=200)
parser.add_argument("--early_stop_threshold", type=int, default=10)
parser.add_argument("--max_round_epoch", type=int, default=20)
args = parser.parse_args()

device = torch.device(args.device if torch.cuda.is_available() else "cpu")
dataset_name = Path(args.data_dir).name
arch_list = [a for a in args.architectures.split(",") if a]

print(f"[batch] device={device}  dataset={dataset_name}  n_archs={len(arch_list)}", flush=True)

# ---------------------------------------------------------------------------
# Load dataset ONCE
# ---------------------------------------------------------------------------
print(f"[data] loading {dataset_name} ...", flush=True)
t0 = time.time()
table_data = TableData.load_from_dir(args.data_dir)
if not table_data.is_materialize:
    table_data.materilize(col_to_text_embedder_cfg=get_text_embedder_cfg(device="cpu"))
print(f"[data] loaded in {time.time()-t0:.1f}s", flush=True)

stype_encoder_dict = construct_stype_encoder_dict(default_stype_encoder_cls_kwargs)
num_cols = sum(len(v) for v in table_data.col_names_dict.values())

is_regression = table_data.task_type == TaskType.REGRESSION
if is_regression:
    loss_fn = L1Loss()
    eval_fn = mean_absolute_error
    higher_is_better = False
else:
    loss_fn = BCEWithLogitsLoss()
    eval_fn = roc_auc_score
    higher_is_better = True

data_loaders = {
    split: torch_frame.data.DataLoader(
        getattr(table_data, f"{split}_tf"),
        batch_size=args.batch_size,
        shuffle=(split == "train"),
        pin_memory=device.type == "cuda",
        num_workers=2,
        persistent_workers=True,
    )
    for split in ["train", "val", "test"]
}

# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------
CSV_FIELDS = [
    "dataset", "space_name", "architecture", "num_params",
    "best_val_metric", "best_val_epoch", "test_metric",
    "train_time_seconds", "test_time_seconds", "metric", "device",
]

def _is_done(arch_str: str) -> bool:
    if not os.path.exists(args.output_csv) or os.path.getsize(args.output_csv) == 0:
        return False
    with open(args.output_csv, "r") as f:
        for row in csv.DictReader(f):
            if row["dataset"] == dataset_name and row["architecture"] == arch_str:
                return True
    return False

def _write_csv(row: dict) -> None:
    exists = os.path.exists(args.output_csv) and os.path.getsize(args.output_csv) > 0
    with open(args.output_csv, "a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        fcntl.flock(f, fcntl.LOCK_UN)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def deactivate_dropout(net):
    for m in net.modules():
        if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False

@torch.no_grad()
def evaluate(net, loader):
    net.eval()
    preds, ys = [], []
    t0 = time.time()
    for batch in loader:
        batch = batch.to(device)
        pred = net(batch)
        pred = pred.view(-1) if pred.size(-1) == 1 else pred
        preds.append(pred.cpu())
        ys.append(batch.y.float().cpu())
    elapsed = time.time() - t0
    preds = torch.cat(preds)
    ys = torch.cat(ys).numpy()
    if not torch.isfinite(preds).all():
        return float("nan"), elapsed
    if is_regression:
        return float(eval_fn(ys, preds.numpy())), elapsed
    else:
        return float(eval_fn(ys, torch.sigmoid(preds).numpy())), elapsed

# ---------------------------------------------------------------------------
# Train one architecture
# ---------------------------------------------------------------------------
def train_one(arch_str: str) -> None:
    if _is_done(arch_str):
        print(f"[skip] {arch_str} already done", flush=True)
        return

    block_widths = [int(x) for x in arch_str.split("-")]

    if args.space_name == "resnet":
        net = PTNASResNet(
            channels=num_cols,
            out_channels=1,
            num_layers=len(block_widths),
            col_stats=table_data.col_stats,
            col_names_dict=table_data.col_names_dict,
            stype_encoder_dict=stype_encoder_dict,
            block_widths=block_widths,
            normalization="layer_norm",
            dropout_prob=0.2,
        )
    else:
        net = PTNASMLP(
            channels=num_cols,
            out_channels=1,
            num_layers=len(block_widths) + 1,
            col_stats=table_data.col_stats,
            col_names_dict=table_data.col_names_dict,
            stype_encoder_dict=stype_encoder_dict,
            hidden_dims=block_widths,
            normalization="layer_norm",
            dropout_prob=0.2,
        )

    num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    if is_regression:
        deactivate_dropout(net)
    net.to(device)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr)

    best_val = -math.inf if higher_is_better else math.inf
    best_epoch = -1
    best_state = None
    patience = 0
    train_start = time.time()

    for epoch in range(args.num_epochs):
        net.train()
        for idx, batch in enumerate(data_loaders["train"]):
            if idx > args.max_round_epoch:
                break
            optimizer.zero_grad()
            batch = batch.to(device)
            pred = net(batch)
            pred = pred.view(-1) if pred.size(-1) == 1 else pred
            loss = loss_fn(pred, batch.y.float())
            if not torch.isfinite(loss):
                print(f"[nan] {arch_str} epoch={epoch}", flush=True)
                break
            loss.backward()
            optimizer.step()

        val_metric, _ = evaluate(net, data_loaders["val"])
        if not math.isfinite(val_metric):
            break

        improved = (higher_is_better and val_metric > best_val) or \
                   (not higher_is_better and val_metric < best_val)
        if improved:
            best_val = val_metric
            best_epoch = epoch
            best_state = copy.deepcopy(net.state_dict())
            patience = 0
        else:
            patience += 1
            if patience > args.early_stop_threshold:
                break

    train_time = time.time() - train_start

    if best_state is None:
        test_metric, test_time = float("nan"), 0.0
        best_val = float("nan")
    else:
        net.load_state_dict(best_state)
        test_metric, test_time = evaluate(net, data_loaders["test"])

    print(f"[done] {arch_str}  params={num_params:,}  "
          f"val={best_val:.4f}@ep{best_epoch}  test={test_metric:.4f}  "
          f"train={train_time:.1f}s", flush=True)

    _write_csv({
        "dataset": dataset_name,
        "space_name": args.space_name,
        "architecture": arch_str,
        "num_params": num_params,
        "best_val_metric": round(best_val, 6),
        "best_val_epoch": best_epoch,
        "test_metric": round(test_metric, 6),
        "train_time_seconds": round(train_time, 2),
        "test_time_seconds": round(test_time, 3),
        "metric": eval_fn.__name__,
        "device": str(device),
    })

    del net, optimizer, best_state
    torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
for i, arch_str in enumerate(arch_list):
    print(f"\n[batch] {i+1}/{len(arch_list)}  arch={arch_str}", flush=True)
    try:
        train_one(arch_str)
    except Exception as e:
        print(f"[error] {arch_str}: {e}", flush=True)

print(f"\n[batch] done. dataset={dataset_name}  device={device}", flush=True)
