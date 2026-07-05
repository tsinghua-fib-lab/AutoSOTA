"""
Train a batch of PTNASBlockMixed models on ONE dataset in a single process.
Dataset is loaded ONCE; models are trained sequentially on the same GPU.

Usage (called by run_train_blockmixed.sh):
    python scripts/new_space/training/train_blockmixed_batch.py \
        --data_dir datasets/fit-medium-table/trial-study-outcome \
        --model_indices 0,8,16,24,...   \
        --device cuda:0
"""

from __future__ import annotations

import argparse
import copy
import csv
import datetime
import fcntl
import json
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
from search_space.block_mixed import PTNASBlockMixed
from utils.resource import get_text_embedder_cfg
from utils.table_data import TableData

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--space_file", type=str,
                    default=str(_PROJECT_ROOT / "datasets" / "nas_bench_tabular" / "space_blockmixed" / "architecture" / "blockmixed.txt"))
parser.add_argument("--model_indices", type=str, required=True,
                    help="Comma-separated model indices to run, e.g. 0,8,16,...")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--channels", type=int, default=32)
parser.add_argument("--out_channels", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--num_epochs", type=int, default=200)
parser.add_argument("--early_stop_threshold", type=int, default=10)
parser.add_argument("--max_round_epoch", type=int, default=20)
parser.add_argument("--log_dir", type=str,
                    default=str(_PROJECT_ROOT / "datasets" / "nas_bench_tabular" / "space_blockmixed" / "training" / "logs"))
parser.add_argument("--csv_file", type=str, default=None,
                    help="Path to output CSV. Defaults to datasets/nas_bench_tabular/space_blockmixed/training/block_mixed_diverse_results.csv")
parser.add_argument("--verbose", action="store_true", default=False)

args = parser.parse_args()

os.makedirs(args.log_dir, exist_ok=True)
dataset_name = Path(args.data_dir).name
csv_filename = args.csv_file if args.csv_file else str(
    _PROJECT_ROOT / "datasets" / "nas_bench_tabular" / "space_blockmixed" / "training" / "block_mixed_diverse_results.csv"
)
model_indices = [int(x) for x in args.model_indices.split(",")]

device = torch.device(args.device if torch.cuda.is_available() else "cpu")
print(f"[batch] device={device}  dataset={dataset_name}  n_models={len(model_indices)}", flush=True)

# ---------------------------------------------------------------------------
# Load space file
# ---------------------------------------------------------------------------
with open(args.space_file, "r") as f:
    space_lines = f.readlines()

# ---------------------------------------------------------------------------
# Load dataset ONCE
# ---------------------------------------------------------------------------
print(f"[data] loading {dataset_name} ...", flush=True)
t0 = time.time()
table_data = TableData.load_from_dir(args.data_dir)
if not table_data.is_materialize:
    text_cfg = get_text_embedder_cfg(device="cpu")
    table_data.materilize(col_to_text_embedder_cfg=text_cfg)
print(f"[data] loaded in {time.time()-t0:.1f}s", flush=True)

stype_encoder_dict = construct_stype_encoder_dict(default_stype_encoder_cls_kwargs)

# Task setup
if table_data.task_type == TaskType.REGRESSION:
    loss_fn = L1Loss()
    eval_fn = mean_absolute_error
    higher_is_better = False
    is_regression = True
elif table_data.task_type == TaskType.BINARY_CLASSIFICATION:
    loss_fn = BCEWithLogitsLoss()
    eval_fn = roc_auc_score
    higher_is_better = True
    is_regression = False
else:
    raise ValueError(f"Unsupported task type: {table_data.task_type}")

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
    "timestamp", "dataset", "model_index", "model_rank", "ref_capacity",
    "depth", "block_specs", "num_params", "channels",
    "best_val_metric", "best_val_epoch", "test_metric",
    "train_time_seconds", "val_time_seconds", "test_time_seconds",
    "metric", "device",
]

def _is_done(model_idx: int) -> bool:
    if not os.path.exists(csv_filename) or os.path.getsize(csv_filename) == 0:
        return False
    with open(csv_filename, "r") as f:
        for row in csv.DictReader(f):
            if row["dataset"] == dataset_name and int(row["model_index"]) == model_idx:
                return True
    return False

def _write_csv(row: dict) -> None:
    csv_exists = os.path.exists(csv_filename) and os.path.getsize(csv_filename) > 0
    with open(csv_filename, "a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not csv_exists:
            writer.writeheader()
        writer.writerow(row)
        fcntl.flock(f, fcntl.LOCK_UN)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def deactivate_dropout(module: torch.nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False


@torch.no_grad()
def evaluate(net, loader):
    net.eval()
    pred_list, y_list = [], []
    t0 = time.time()
    for batch in loader:
        batch = batch.to(device)
        pred = net(batch)
        pred = pred.view(-1) if pred.size(-1) == 1 else pred
        pred_list.append(pred.cpu())
        y_list.append(batch.y.float().cpu())
    elapsed = time.time() - t0
    preds = torch.cat(pred_list, dim=0)
    ys = torch.cat(y_list, dim=0).numpy()
    if not torch.isfinite(preds).all():
        return float("nan"), elapsed
    if is_regression:
        score = eval_fn(ys, preds.numpy())
    else:
        score = eval_fn(ys, torch.sigmoid(preds).numpy())
    score = float(score)
    if not math.isfinite(score):
        return float("nan"), elapsed
    return score, elapsed


# ---------------------------------------------------------------------------
# Train one model
# ---------------------------------------------------------------------------
def train_one(model_idx: int) -> None:
    if _is_done(model_idx):
        print(f"[skip] index={model_idx} already in CSV", flush=True)
        return

    if model_idx >= len(space_lines):
        print(f"[skip] index={model_idx} out of range", flush=True)
        return

    record = json.loads(space_lines[model_idx].strip())
    model_rank = record.get("rank", model_idx)
    ref_capacity = record.get("ref_capacity", -1)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    print(
        f"[start] index={model_idx}  rank={model_rank}  depth={record['depth']}  "
        f"capacity={ref_capacity}",
        flush=True,
    )

    net = PTNASBlockMixed.from_space_record(
        record,
        channels=args.channels,
        out_channels=args.out_channels,
        col_stats=table_data.col_stats,
        col_names_dict=table_data.col_names_dict,
        stype_encoder_dict=stype_encoder_dict,
    )
    block_specs_str = json.dumps([list(spec) for spec in net.block_specs])
    num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

    if is_regression:
        deactivate_dropout(net)
    net.to(device)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr
    )

    best_val_metric = -math.inf if higher_is_better else math.inf
    best_val_epoch = -1
    best_model_state = None
    patience = 0
    train_start = time.time()

    for epoch in range(args.num_epochs):
        net.train()
        loss_accum = 0.0
        count_accum = 0
        non_finite = False

        for idx, batch in enumerate(data_loaders["train"]):
            if idx > args.max_round_epoch:
                break
            optimizer.zero_grad()
            batch = batch.to(device)
            pred = net(batch)
            pred = pred.view(-1) if pred.size(-1) == 1 else pred
            loss = loss_fn(pred, batch.y.float())
            if not torch.isfinite(loss):
                non_finite = True
                print(f"[nan] index={model_idx} non-finite loss epoch={epoch}", flush=True)
                break
            loss.backward()
            optimizer.step()
            loss_accum += loss.item()
            count_accum += 1

        train_loss = loss_accum / max(count_accum, 1)
        if non_finite or not math.isfinite(train_loss):
            break

        val_metric, _ = evaluate(net, data_loaders["val"])
        if not math.isfinite(val_metric):
            print(f"[nan] index={model_idx} non-finite val epoch={epoch}", flush=True)
            break

        improved = (
            (higher_is_better and val_metric > best_val_metric) or
            (not higher_is_better and val_metric < best_val_metric)
        )
        if improved:
            best_val_metric = val_metric
            best_val_epoch = epoch
            best_model_state = copy.deepcopy(net.state_dict())
            patience = 0
            if args.verbose:
                test_tmp, _ = evaluate(net, data_loaders["test"])
                print(f"  ep{epoch:4d} val={val_metric:.4f} test={test_tmp:.4f} [*]", flush=True)
        else:
            patience += 1
            if patience > args.early_stop_threshold:
                print(f"[early_stop] index={model_idx} epoch={epoch}", flush=True)
                break

    train_time = time.time() - train_start

    if best_model_state is None:
        final_test_metric = float("nan")
        test_time = val_time = 0.0
        best_val_metric = float("nan")
        best_val_epoch = -1
        print(f"[failed] index={model_idx} no valid checkpoint", flush=True)
    else:
        net.load_state_dict(best_model_state)
        final_test_metric, test_time = evaluate(net, data_loaders["test"])
        _, val_time = evaluate(net, data_loaders["val"])

    print(
        f"[done] index={model_idx}  rank={model_rank}  params={num_params:,}  "
        f"val={best_val_metric:.4f}@ep{best_val_epoch}  test={final_test_metric:.4f}  "
        f"train={train_time:.1f}s",
        flush=True,
    )

    _write_csv({
        "timestamp": timestamp,
        "dataset": dataset_name,
        "model_index": model_idx,
        "model_rank": model_rank,
        "ref_capacity": ref_capacity,
        "depth": record["depth"],
        "block_specs": block_specs_str,
        "num_params": num_params,
        "channels": args.channels,
        "best_val_metric": round(best_val_metric, 6),
        "best_val_epoch": best_val_epoch,
        "test_metric": round(final_test_metric, 6),
        "train_time_seconds": round(train_time, 2),
        "val_time_seconds": round(val_time, 3),
        "test_time_seconds": round(test_time, 3),
        "metric": eval_fn.__name__,
        "device": str(device),
    })

    # free GPU memory before next model
    del net, optimizer, best_model_state
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
for i, model_idx in enumerate(model_indices):
    print(f"\n[batch] {i+1}/{len(model_indices)}  index={model_idx}", flush=True)
    try:
        train_one(model_idx)
    except Exception as e:
        print(f"[error] index={model_idx}: {e}", flush=True)

print(f"\n[batch] done. dataset={dataset_name}  device={device}", flush=True)
