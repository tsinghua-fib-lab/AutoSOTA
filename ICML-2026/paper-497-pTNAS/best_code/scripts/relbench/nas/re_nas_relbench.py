#!/usr/bin/env python3
"""
EA-NAS baseline: Random search within a time budget.
Randomly sample architectures, train each, keep the best.

Usage:
    PYTHONPATH=src python scripts/relbench/nas/re_nas_relbench.py \
        --data_dir datasets/fit-medium-table/avito-user-clicks \
        --device cuda:0 --time_budget 7
"""
from __future__ import annotations
import argparse, copy, csv, fcntl, math, os, random, sys, time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import torch, torch_frame
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--time_budget", type=float, default=7.0)
    parser.add_argument("--n_reps", type=int, default=3)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--max_batches", type=int, default=20)
    parser.add_argument("--output_csv", default="run_outputs/data/relbench/baselines/renas_relbench_results.csv")
    args = parser.parse_args()

    device = torch.device(args.device)
    dataset_name = Path(args.data_dir).name
    if os.path.dirname(args.output_csv):
        os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    table_data = TableData.load_from_dir(args.data_dir)
    is_regression = table_data.task_type == TaskType.REGRESSION
    higher_is_better = not is_regression
    num_cols = sum(len(v) for v in table_data.col_names_dict.values())
    loss_fn = L1Loss() if is_regression else BCEWithLogitsLoss()

    train_loader = torch_frame.data.DataLoader(table_data.train_tf, batch_size=256, shuffle=True,
                                               num_workers=2, persistent_workers=True)
    val_loader = torch_frame.data.DataLoader(table_data.val_tf, batch_size=256, shuffle=False)
    test_loader = torch_frame.data.DataLoader(table_data.test_tf, batch_size=256, shuffle=False)

    fields = ["dataset", "method", "rep", "best_arch", "best_val", "best_test",
              "search_time", "n_models_trained", "metric"]
    csv_exists = os.path.exists(args.output_csv) and os.path.getsize(args.output_csv) > 0

    print(f"[EA-NAS] {dataset_name} budget={args.time_budget}s device={device}", flush=True)

    for rep in range(args.n_reps):
        rng = random.Random(rep + 42)
        best_val = -math.inf if higher_is_better else math.inf
        best_test = float("nan")
        best_arch = None
        n_trained = 0
        t_start = time.time()

        while time.time() - t_start < args.time_budget:
            # Random architecture
            hidden_dims = [rng.choice(LAYER_CHOICES) for _ in range(N_LAYERS)]
            arch_str = "-".join(map(str, hidden_dims))

            torch.manual_seed(42)
            stype_encoder_dict = construct_stype_encoder_dict(default_stype_encoder_cls_kwargs)
            model = PTNASMLP(
                channels=num_cols, out_channels=1, num_layers=N_LAYERS + 1,
                col_stats=table_data.col_stats, col_names_dict=table_data.col_names_dict,
                stype_encoder_dict=stype_encoder_dict, hidden_dims=hidden_dims,
                normalization="layer_norm", dropout_prob=0.2,
            ).to(device)

            if is_regression:
                deactivate_dropout(model)

            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
            best_model_val = -math.inf if higher_is_better else math.inf
            best_state = None
            patience = 0

            for epoch in range(args.max_epochs):
                if time.time() - t_start >= args.time_budget:
                    break
                model.train()
                for idx, batch in enumerate(train_loader):
                    if idx > args.max_batches:
                        break
                    optimizer.zero_grad()
                    batch = batch.to(device)
                    pred = model(batch)
                    pred = pred.view(-1) if pred.size(-1) == 1 else pred
                    loss = loss_fn(pred, batch.y.float())
                    loss.backward()
                    optimizer.step()

                val_m = evaluate(model, val_loader, device, is_regression)
                improved = (higher_is_better and val_m > best_model_val) or \
                           (not higher_is_better and val_m < best_model_val)
                if improved:
                    best_model_val = val_m
                    best_state = copy.deepcopy(model.state_dict())
                    patience = 0
                else:
                    patience += 1
                    if patience > 5:
                        break

            # Evaluate best checkpoint
            if best_state:
                model.load_state_dict(best_state)
            test_m = evaluate(model, test_loader, device, is_regression)
            n_trained += 1

            # Update global best
            is_better = (higher_is_better and best_model_val > best_val) or \
                        (not higher_is_better and best_model_val < best_val)
            if is_better:
                best_val = best_model_val
                best_test = test_m
                best_arch = arch_str

            del model, optimizer, best_state
            torch.cuda.empty_cache()

        search_time = time.time() - t_start
        print(f"  rep={rep} arch={best_arch} val={best_val:.4f} test={best_test:.4f} "
              f"n_trained={n_trained} time={search_time:.1f}s", flush=True)

        with open(args.output_csv, "a", newline="") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            w = csv.DictWriter(f, fieldnames=fields)
            if not csv_exists:
                w.writeheader()
                csv_exists = True
            w.writerow({
                "dataset": dataset_name, "method": "EA-NAS", "rep": rep,
                "best_arch": best_arch, "best_val": round(best_val, 6),
                "best_test": round(best_test, 6), "search_time": round(search_time, 2),
                "n_models_trained": n_trained,
                "metric": "mae" if is_regression else "roc_auc",
            })
            fcntl.flock(f, fcntl.LOCK_UN)

    print(f"[EA-NAS] Done. Results: {args.output_csv}", flush=True)


if __name__ == "__main__":
    main()
