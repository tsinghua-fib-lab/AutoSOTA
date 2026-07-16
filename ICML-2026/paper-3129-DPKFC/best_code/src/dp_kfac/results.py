from typing import Any, Dict, List, Optional
from pathlib import Path

import pandas as pd


def save_results_csv(
    results: List[Dict[str, Any]],
    output_path: str | Path,
    columns: Optional[List[str]] = None,
) -> None:
    """Save experiment results as CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)
    if columns is not None:
        df = df[[c for c in columns if c in df.columns]]
    df.to_csv(output_path, index=False)


def load_results_csv(path: str | Path) -> pd.DataFrame:
    """Load results CSV with column name normalization."""
    df = pd.read_csv(path)
    rename_map = {
        "Acc": "Accuracy",
        "acc": "Accuracy",
        "accuracy": "Accuracy",
        "Test Loss": "Loss",
        "test_loss": "Loss",
        "loss": "Loss",
        "method": "Method",
        "epsilon": "Epsilon",
        "seed": "Seed",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    return df


def aggregate_results(
    df: pd.DataFrame,
    group_by: Optional[List[str]] = None,
    metric: str = "Accuracy",
) -> pd.DataFrame:
    """Group by method+epsilon, compute mean +/- std."""
    if group_by is None:
        group_by = ["Method", "Epsilon"]

    agg = df.groupby(group_by)[metric].agg(["mean", "std", "count"]).reset_index()
    agg = agg.rename(columns={"mean": f"{metric}_mean", "std": f"{metric}_std", "count": "n_seeds"})
    return agg


def print_summary_table(df: pd.DataFrame, metric: str = "Accuracy") -> None:
    """Print a formatted summary table to console."""
    agg = aggregate_results(df, metric=metric)
    print(f"\n{'Method':<35} {'Epsilon':>8} {f'{metric} Mean':>12} {'Std':>8}")
    print("-" * 70)
    for _, row in agg.iterrows():
        std_val = row[f"{metric}_std"]
        std_str = f"{std_val:.4f}" if pd.notna(std_val) else "N/A"
        print(f"{row['Method']:<35} {row['Epsilon']:>8.1f} {row[f'{metric}_mean']:>12.4f} {std_str:>8}")
