import argparse
import os
import random
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .compare import get_model, train_model, evaluate_model
from .datasets import MultiDatasetLoader


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


DEFAULT_DATASETS = [
    "ETTh1",
    "ETTh2",
    "ETTm1",
    "ETTm2",
    "electricity",
    "exchange_rate",
    "weather",
    "illness",
    "traffic",
]

DEFAULT_MODELS = [
    "DualTimesField",
    "SIREN",
    "WIRE",
    "N-BEATS",
    "TimesNet",
    "PatchTST",
    "iTransformer",
    "PCA",
    "Fourier",
]


def update_summary_and_pivots(summary_csv: str, mse_pivot_csv: str, mae_pivot_csv: str, row: Dict) -> None:
    os.makedirs(os.path.dirname(summary_csv), exist_ok=True)

    row_df = pd.DataFrame([row])
    if os.path.exists(summary_csv):
        df = pd.read_csv(summary_csv)
        key_mask = (df["dataset"] == row["dataset"]) & (df["model"] == row["model"])
        df = df[~key_mask]
        df = pd.concat([df, row_df], ignore_index=True)
    else:
        df = row_df

    df = df.sort_values(["model", "dataset"]).reset_index(drop=True)
    df.to_csv(summary_csv, index=False)

    mse_pivot = df.pivot(index="model", columns="dataset", values="mse_mean_std")
    mae_pivot = df.pivot(index="model", columns="dataset", values="mae_mean_std")

    mse_pivot.to_csv(mse_pivot_csv)
    mae_pivot.to_csv(mae_pivot_csv)


def run_single_seed(
    loader: MultiDatasetLoader,
    dataset_name: str,
    model_name: str,
    seed: int,
    seq_length: int,
    batch_size: int,
    epochs: int,
    lr: float,
    hidden_dim: int,
    device: str,
    data_stride: int,
) -> Dict[str, float]:
    set_seed(seed)

    train_dataset, val_dataset, test_dataset = loader.get_dataset(
        dataset_name,
        seq_length=seq_length,
        stride=data_stride,
        normalize=True,
    )

    sample_x, _ = train_dataset[0]
    actual_seq_length = sample_x.shape[0]
    num_variables = sample_x.shape[1]

    actual_batch_size = min(batch_size, len(train_dataset))
    train_loader = DataLoader(train_dataset, batch_size=actual_batch_size, shuffle=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=max(1, min(actual_batch_size, len(val_dataset))),
        shuffle=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=max(1, min(actual_batch_size, len(test_dataset))),
        shuffle=False,
    )

    model = get_model(model_name, num_variables, actual_seq_length, hidden_dim)
    model = train_model(model, train_loader, val_loader, epochs, lr, device, model_name)
    metrics = evaluate_model(model, test_loader, device)

    return {
        "mse": float(metrics["mse"]),
        "mae": float(metrics["mae"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run reconstruction comparison with multiple seeds and incremental CSV updates."
    )
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--save_dir", type=str, default="outputs/reconstruction/multi_seed")
    parser.add_argument("--seq_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_seeds", type=int, default=5)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    args = parser.parse_args()

    if args.seeds is not None and len(args.seeds) > 0:
        seeds: List[int] = args.seeds
    else:
        seeds = list(range(args.num_seeds))

    os.makedirs(args.save_dir, exist_ok=True)
    summary_csv = os.path.join(args.save_dir, "comparison_multi_seed_summary.csv")
    mse_pivot_csv = os.path.join(args.save_dir, "comparison_multi_seed_pivot_mse_mean_std.csv")
    mae_pivot_csv = os.path.join(args.save_dir, "comparison_multi_seed_pivot_mae_mean_std.csv")

    print("\nRunning multi-seed reconstruction comparison")
    print(f"  Datasets: {args.datasets}")
    print(f"  Models: {args.models}")
    print(f"  Seeds: {seeds}")
    print(f"  Save dir: {args.save_dir}")

    loader = MultiDatasetLoader(args.data_dir)
    stride = max(1, args.seq_length // 4)

    for dataset_name in args.datasets:
        for model_name in args.models:
            print(f"\n[Start] dataset={dataset_name}, model={model_name}")
            mse_runs: List[float] = []
            mae_runs: List[float] = []

            for seed in seeds:
                try:
                    metrics = run_single_seed(
                        loader=loader,
                        dataset_name=dataset_name,
                        model_name=model_name,
                        seed=seed,
                        seq_length=args.seq_length,
                        batch_size=args.batch_size,
                        epochs=args.epochs,
                        lr=args.lr,
                        hidden_dim=args.hidden_dim,
                        device=args.device,
                        data_stride=stride,
                    )
                    mse_runs.append(metrics["mse"])
                    mae_runs.append(metrics["mae"])
                    print(f"  seed={seed}: mse={metrics['mse']:.6f}, mae={metrics['mae']:.6f}")
                except Exception as exc:
                    print(f"  seed={seed}: failed with error: {exc}")

            if len(mse_runs) == 0:
                print("  No successful runs for this dataset-model pair, skipping CSV update.")
                continue

            mean_mse = float(np.mean(mse_runs))
            std_mse = float(np.std(mse_runs))
            mean_mae = float(np.mean(mae_runs))
            std_mae = float(np.std(mae_runs))

            row = {
                "dataset": dataset_name,
                "model": model_name,
                "num_runs": len(mse_runs),
                "seeds": " ".join(str(s) for s in seeds),
                "mean_mse": mean_mse,
                "std_mse": std_mse,
                "mean_mae": mean_mae,
                "std_mae": std_mae,
                "mse_mean_std": f"{mean_mse:.6f} ± {std_mse:.6f}",
                "mae_mean_std": f"{mean_mae:.6f} ± {std_mae:.6f}",
            }

            update_summary_and_pivots(summary_csv, mse_pivot_csv, mae_pivot_csv, row)
            print(
                f"[Done] dataset={dataset_name}, model={model_name}: "
                f"MSE {row['mse_mean_std']}, MAE {row['mae_mean_std']}"
            )
            print("  CSV updated incrementally.")

    print("\nAll experiments completed.")
    print(f"Summary CSV: {summary_csv}")
    print(f"MSE Pivot CSV: {mse_pivot_csv}")
    print(f"MAE Pivot CSV: {mae_pivot_csv}")


if __name__ == "__main__":
    main()
