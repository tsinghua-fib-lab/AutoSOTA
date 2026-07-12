"""DualTimesField ETTh1 reproduction evaluation script.
Trains model with paper settings and outputs metrics in parseable JSON format.
"""
import sys, os, json, random, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, "/repo")
from src.dualfield import DualTimesField
from reconstruction.datasets import MultiDatasetLoader

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ETTh1")
    parser.add_argument("--seq_length", type=int, default=336)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to checkpoint to evaluate (skips training)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    data_dir = Path("/repo/data")
    loader = MultiDatasetLoader(data_dir)
    train_dataset, val_dataset, test_dataset = loader.get_dataset(
        args.dataset, seq_length=args.seq_length,
        stride=max(1, args.seq_length // 4), normalize=True,
    )
    num_variables = loader.get_num_variables(args.dataset)
    batch_size = min(args.batch_size, len(train_dataset))

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = DualTimesField(
        num_variables=num_variables, seq_length=args.seq_length,
        num_frequencies=16, hidden_dim=64, num_layers=3, freq_cutoff=8.0,
        num_atoms=16, sigma_base=0.05, sparsity_lambda=0.001, smoothness_lambda=0.001,
    ).to(device)

    if args.ckpt and os.path.exists(args.ckpt):
        model.load_state_dict(torch.load(args.ckpt, map_location=device))
        print(f"Loaded checkpoint: {args.ckpt}")
    else:
        # Train
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_dataset, batch_size=min(batch_size, len(val_dataset)), shuffle=False)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        init_batch = next(iter(train_loader))
        x_init, t_init = init_batch
        model.initialize_atoms(x_init.to(device), t_init.to(device))

        best_val_loss = float("inf")
        best_state = None

        for epoch in tqdm(range(args.epochs), desc="Training"):
            model.train()
            model.set_epoch(epoch)
            for x, t in train_loader:
                x, t = x.to(device), t.to(device)
                optimizer.zero_grad()
                losses = model.compute_loss(x, t)
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            scheduler.step()

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x, t in val_loader:
                    x, t = x.to(device), t.to(device)
                    output, _, _ = model(x, t)
                    val_loss += F.mse_loss(output, x).item()
            val_loss /= max(len(val_loader), 1)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state)

    # Evaluate
    model.eval()
    test_mse, test_mae = 0.0, 0.0
    with torch.no_grad():
        for x, t in test_loader:
            x, t = x.to(device), t.to(device)
            output, _, _ = model(x, t)
            test_mse += F.mse_loss(output, x).item()
            test_mae += F.l1_loss(output, x).item()
    test_mse /= max(len(test_loader), 1)
    test_mae /= max(len(test_loader), 1)
    active_atoms = model.dgf.get_active_atoms()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    result = {
        "dataset": args.dataset,
        "test_mse": round(test_mse, 6),
        "test_mae": round(test_mae, 6),
        "active_atoms": active_atoms,
        "num_parameters": n_params,
        "seq_length": args.seq_length,
        "epochs": args.epochs,
    }
    print("\n" + "=" * 50)
    print("REPRODUCTION RESULT")
    print("=" * 50)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
