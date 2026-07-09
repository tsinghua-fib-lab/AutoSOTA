import os
import sys
import random
import argparse
import numpy as np
import torch
import torch.nn as nn

# ------------------------------------------------------------------
# Path setup
# ------------------------------------------------------------------
PROJECT_ROOT = "/opt/data/private/icml2026_2/mol_qm"
sys.path.append(PROJECT_ROOT)

from dataset.dataset import QM7bDataset, QM7bDataModule
from model.model import GINRegressorModes

# ------------------------------------------------------------------
# Utils
# ------------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_one_epoch(model, loader, optimizer=None, device="cuda"):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_mae = 0.0
    total_se = 0.0   # squared error sum (for RMSE)
    total_samples = 0

    for batch in loader:
        batch = batch.to(device)

        # ===== forward =====
        if model.mode == "state_emb":
            out = model(batch, fidelity=batch.fidelity)
        else:
            out = model(batch)
        loss = out["loss"]

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # ===== statistics =====
        num_graphs = batch.num_graphs
        total_samples += num_graphs

        total_loss += loss.item() * num_graphs

        # 用预测均值 μ 来算 MAE / RMSE
        pred = out["mu"].view(-1)
        target = batch.y.view(-1)

        mae = torch.mean(torch.abs(pred - target))
        total_mae += mae.item() * num_graphs

        se = torch.mean((pred - target) ** 2)
        total_se += se.item() * num_graphs

    avg_loss = total_loss / total_samples
    avg_mae = total_mae / total_samples
    rmse = (total_se / total_samples) ** 0.5

    return {
        "loss": avg_loss,
        "mae": avg_mae,
        "rmse": rmse,
    }


def freeze_gnn_backbone(model):
    for conv in model.convs:
        for p in conv.parameters():
            p.requires_grad = False
    for bn in model.batch_norms:
        for p in bn.parameters():
            p.requires_grad = False

# ------------------------------------------------------------------
# Training pipelines
# ------------------------------------------------------------------
def train_multifidelity(dm, args, device, patience=50):
    print(f"\n===== Multi-fidelity Training | mode={args.mode} =====")

    model_mode = "default" if args.mode == "1_fi" else "state_emb"

    model = GINRegressorModes(
        node_feat_dim=16,
        edge_feat_dim=100,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        mode=model_mode,
        alpha_rank=args.alpha_rank,
        alpha_tau=args.alpha_tau,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_rmse = float("inf")
    best_model_state = None
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_one_epoch(model, dm.train_loader, optimizer, device)
        val_metrics = run_one_epoch(model, dm.val_loader, None, device)

        print(f"[Epoch {epoch:03d}] "
            f"Train | loss: {train_metrics['loss']:.6f} MAE: {train_metrics['mae']:.6f} RMSE: {train_metrics['rmse']:.6f} || "
            f"Val   | loss: {val_metrics['loss']:.6f} MAE: {val_metrics['mae']:.6f} RMSE: {val_metrics['rmse']:.6f}", flush=True)

        # Early stopping
        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    # 恢复最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # -------------------- Test --------------------
    test_metrics = run_one_epoch(model, dm.test_loader, None, device)
    print(
        f"[Test] "
        f"loss: {test_metrics['loss']:.6f} | "
        f"MAE: {test_metrics['mae']:.6f} | "
        f"RMSE: {test_metrics['rmse']:.6f}"
    )

    return model


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True,
                        choices=["1_fi", "2_fi", "3_fi", "mp2_transfer", "hf_transfer"])
    parser.add_argument("--ccsd_num", type=int, default=1500)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--pretrain_epochs", type=int, default=200)
    parser.add_argument("--finetune_epochs", type=int, default=200)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--alpha_rank", type=float, default=1.0)
    parser.add_argument("--alpha_tau", type=float, default=1e-10)
    args = parser.parse_args()

    set_seed(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"

    # --------------------------------------------------
    # Dataset & DataModule
    # --------------------------------------------------
    dataset = QM7bDataset(
        root=os.path.join(PROJECT_ROOT, "dataset/qm7b_pyg"),
        raw_json_path=os.path.join(PROJECT_ROOT, "dataset/qm7b.json")
    )

    dm = QM7bDataModule(
        dataset=dataset,
        mode=args.mode,
        ccsd_num=args.ccsd_num,
        batch_size=args.batch_size,
        seed=args.seed
    )

    train_multifidelity(dm, args, device)


if __name__ == "__main__":
    main()
