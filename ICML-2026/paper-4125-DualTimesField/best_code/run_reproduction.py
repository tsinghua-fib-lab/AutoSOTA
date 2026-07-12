import sys
from pathlib import Path
sys.path.insert(0, "/repo")

import argparse
import random
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dualfield import DualTimesField
from reconstruction.datasets import MultiDatasetLoader

SEP50 = "=" * 50
SEP60 = "=" * 60

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ETTh1")
    parser.add_argument("--seq_length", type=int, default=336)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_seeds", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_dir = Path("/repo/data")
    save_dir = Path("/repo/outputs/reproduction")
    save_dir.mkdir(parents=True, exist_ok=True)

    loader = MultiDatasetLoader(data_dir)
    train_dataset, val_dataset, test_dataset = loader.get_dataset(
        args.dataset,
        seq_length=args.seq_length,
        stride=max(1, args.seq_length // 4),
        normalize=True,
    )
    num_variables = loader.get_num_variables(args.dataset)
    print(f"Dataset: {args.dataset}, Variables: {num_variables}")
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    mse_list = []
    mae_list = []

    for seed in range(args.num_seeds):
        print(f"\n{SEP50}")
        print(f"Seed {seed}")
        print(SEP50)
        set_seed(seed)

        actual_batch_size = min(args.batch_size, len(train_dataset))
        train_loader = DataLoader(train_dataset, batch_size=actual_batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_dataset, batch_size=min(actual_batch_size, len(val_dataset)), shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=min(actual_batch_size, len(test_dataset)), shuffle=False)

        model = DualTimesField(
            num_variables=num_variables,
            seq_length=args.seq_length,
            num_frequencies=16,
            hidden_dim=64,
            num_layers=3,
            freq_cutoff=8.0,
            num_atoms=16,
            sigma_base=0.05,
            sparsity_lambda=0.001,
            smoothness_lambda=0.001,
        ).to(device)

        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Parameters: {num_params}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        init_batch = next(iter(train_loader))
        x_init, t_init = init_batch
        model.initialize_atoms(x_init.to(device), t_init.to(device))

        best_val_loss = float("inf")
        best_state = None

        pbar = tqdm(range(args.epochs), desc=f"  seed={seed}")
        for epoch in pbar:
            model.train()
            model.set_epoch(epoch)
            train_loss = 0.0
            num_batches = 0
            for x, t in train_loader:
                x, t = x.to(device), t.to(device)
                optimizer.zero_grad()
                losses = model.compute_loss(x, t)
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += losses["reconstruction"].item()
                num_batches += 1
            scheduler.step()

            model.eval()
            val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for x, t in val_loader:
                    x, t = x.to(device), t.to(device)
                    output, _, _ = model(x, t)
                    val_loss += F.mse_loss(output, x).item()
                    val_batches += 1

            val_loss /= max(val_batches, 1)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            pbar.set_postfix({
                "train": f"{train_loss/max(num_batches,1):.6f}",
                "val": f"{val_loss:.6f}",
                "best": f"{best_val_loss:.6f}",
            })

        if best_state is not None:
            model.load_state_dict(best_state)

        model.eval()
        test_mse = 0.0
        test_mae = 0.0
        test_batches = 0
        with torch.no_grad():
            for x, t in test_loader:
                x, t = x.to(device), t.to(device)
                output, _, _ = model(x, t)
                test_mse += F.mse_loss(output, x).item()
                test_mae += F.l1_loss(output, x).item()
                test_batches += 1

        test_mse /= max(test_batches, 1)
        test_mae /= max(test_batches, 1)
        active_atoms = model.dgf.get_active_atoms()

        mse_list.append(test_mse)
        mae_list.append(test_mae)

        print(f"Seed {seed}: MSE={test_mse:.6f}, MAE={test_mae:.6f}, Active atoms={active_atoms}")

        ckpt_dir = save_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt_dir / f"{args.dataset}_seed{seed}.pt")

    mse_mean = float(np.mean(mse_list))
    mse_std = float(np.std(mse_list))
    mae_mean = float(np.mean(mae_list))
    mae_std = float(np.std(mae_list))

    print(f"\n{SEP60}")
    print("REPRODUCTION RESULTS")
    print(SEP60)
    print(f"Dataset: {args.dataset}")
    print(f"MSE (mean +/- std): {mse_mean:.6f} +/- {mse_std:.6f}")
    print(f"MAE (mean +/- std): {mae_mean:.6f} +/- {mae_std:.6f}")
    print(f"Per-seed MSE: {[float(v) for v in mse_list]}")
    print(f"Per-seed MAE: {[float(v) for v in mae_list]}")
    print(f"Paper MSE: 0.0084 +/- 0.0008")
    print(f"Paper MAE: 0.0655 +/- 0.0041")

    results = {
        "dataset": args.dataset,
        "config": {
            "seq_length": args.seq_length,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": 1e-4,
            "num_variables": num_variables,
            "num_seeds": args.num_seeds,
        },
        "mse": {"mean": mse_mean, "std": mse_std, "values": [float(v) for v in mse_list]},
        "mae": {"mean": mae_mean, "std": mae_std, "values": [float(v) for v in mae_list]},
        "paper_mse": "0.0084 +/- 0.0008",
        "paper_mae": "0.0655 +/- 0.0041",
    }
    with open(save_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {save_dir / results.json}")

if __name__ == "__main__":
    main()
