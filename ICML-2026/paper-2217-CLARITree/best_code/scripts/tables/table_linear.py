from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


FINAL_CSV = Path("results/final.csv")
OPENML_FINAL_CSV = Path("results/final_openml.csv")
OUT_ROOT = Path("results/tables")
OPENML_OUT_ROOT = Path("results/tables_openml")
TIME_LIMIT = 590.0
R2_ROUND = 2

METHODS = ["claritree", "streed", "streed_s", "guide", "greedy", "pilot", "m5"]
LABELS = {
    "claritree": "CLARITree",
    "streed": "STreeD",
    "streed_s": "STreeD-S",
    "guide": "GUIDE",
    "greedy": "Greedy",
    "pilot": "PILOT",
    "m5": "M5",
}
DATASET_ORDER = [
    "airfoil",
    "auction",
    "auto_mpg",
    "energe_c",
    "energe_h",
    "insurance",
    "optical_net",
    "real_estate",
    "servo",
    "synch",
    "yacht",
    "california_housing",
    "seoul_bike",
    "temperature_max",
    "temperature_min",
    "walmart",
]

KEY_COLS = [
    "dataset",
    "method",
    "depth",
    "lambda",
    "kappa",
    "cv_folds",
    "n_thresholds",
    "threshold_label",
    "thresholds_strategy",
    "min_leaf_node_size",
    "cost_complexity",
    "pilot_grid_index",
    "ridge_penalty",
    "lasso_penalty",
]
METRICS = [
    "val_r2",
    "train_r2",
    "test_r2",
    "train_time_s",
]


def default_final_csv(openml: bool, threshold_label: str | None = None) -> Path:
    if threshold_label is None:
        return OPENML_FINAL_CSV if openml else FINAL_CSV
    base = "final_openml" if openml else "final"
    return Path("results") / f"{base}_{threshold_label}.csv"


def threshold_group_column(rows: pd.DataFrame) -> str:
    if "threshold_label" in rows.columns and rows["threshold_label"].notna().any():
        return "threshold_label"
    return "n_thresholds"


def threshold_display(value: object) -> str:
    if pd.isna(value):
        return "unknown"
    text = str(value)
    try:
        numeric = float(text)
    except (TypeError, ValueError):
        return text
    if np.isfinite(numeric) and numeric.is_integer():
        return str(int(numeric))
    return text


def threshold_path_token(value: object) -> str:
    label = threshold_display(value)
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in label)


def display_dataset(name: str) -> str:
    name = Path(name).name
    special = {
        "energe_c": "Energy (Cooling)",
        "energe_h": "Energy (Heating)",
        "temperature_min": "Temperature (Min)",
        "temperature_max": "Temperature (Max)",
    }
    return special.get(name, name.replace("_", " ").title())


def load_best_rows(path: Path, threshold_label: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if threshold_label is not None:
        if "threshold_label" not in df.columns:
            df["threshold_label"] = threshold_label
        else:
            df["threshold_label"] = df["threshold_label"].fillna(threshold_label)
    keys = [c for c in KEY_COLS if c in df.columns]
    metrics = [c for c in METRICS if c in df.columns]

    best = df[df["outer"].eq("best_by_mean_val_r2")][keys + metrics].copy()
    std = df[df["outer"].eq("std")][keys + metrics].copy()
    best = best.rename(columns={c: f"{c}_mean" for c in metrics})
    std = std.rename(columns={c: f"{c}_std" for c in metrics})
    return best.merge(std, on=keys, how="left")


def fmt_number(value: float, *, scientific: bool = False) -> str:
    if not np.isfinite(value):
        return ""
    if scientific or abs(value) >= 100:
        return f"{value:.1e}".replace("e+0", "e").replace("e+", "e").replace("e0", "e")
    return f"{value:.2f}"


def fmt_r2(value: float) -> str:
    rounded = round(float(value), R2_ROUND)
    return fmt_number(rounded, scientific=not (0 <= rounded <= 1))


def fmt_std(value: float, unstable_mean: bool) -> str:
    if not np.isfinite(value):
        return ""
    rounded = round(float(value), R2_ROUND)
    return fmt_number(rounded, scientific=unstable_mean or rounded >= 100)


def mse_ratio(method_r2: float, claritree_r2: float) -> float | None:
    if not (np.isfinite(method_r2) and np.isfinite(claritree_r2)):
        return None
    method_r2 = round(float(method_r2), R2_ROUND)
    claritree_r2 = round(float(claritree_r2), R2_ROUND)
    denom = 1 - claritree_r2
    if abs(denom) < 1e-12:
        return None
    return (1 - method_r2) / denom


def fmt_ratio(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return ""
    return fmt_number(round(float(value), R2_ROUND), scientific=abs(value) >= 100)


def rank_styles(rows_by_method: dict[str, pd.Series], metric: str) -> dict[str, str]:
    ranking = []
    for method, row in rows_by_method.items():
        mean = row[f"{metric}_mean"]
        std = row[f"{metric}_std"]
        if np.isfinite(mean):
            std_rank = round(float(std), R2_ROUND) if np.isfinite(std) else np.inf
            ranking.append((method, round(float(mean), R2_ROUND), std_rank))
    if not ranking:
        return {}

    keys = sorted({(mean, std) for _, mean, std in ranking}, key=lambda x: (-x[0], x[1]))
    first = {method for method, mean, std in ranking if (mean, std) == keys[0]}
    styles = {method: "bold" for method in first}
    if len(first) == 1 and len(keys) > 1:
        styles.update(
            {
                method: "underline"
                for method, mean, std in ranking
                if (mean, std) == keys[1]
            }
        )
    return styles


def apply_style(cell: str, style: str | None) -> str:
    if not cell:
        return cell
    if style == "bold":
        return f"**{cell}**"
    if style == "underline":
        return f"<u>{cell}</u>"
    return cell


def make_cell(
    row: pd.Series | None,
    metric: str,
    claritree_row: pd.Series | None,
    style: str | None,
) -> str:
    if row is None:
        return ""

    mean = row[f"{metric}_mean"]
    std = row[f"{metric}_std"]
    if not np.isfinite(mean):
        return ""

    rounded = round(float(mean), R2_ROUND)
    unstable = not (0 <= rounded <= 1)
    base = f"{fmt_r2(mean)} +/- {fmt_std(std, unstable)}"

    ratio = None
    if claritree_row is not None:
        ratio = mse_ratio(mean, claritree_row[f"{metric}_mean"])
    ratio_text = fmt_ratio(ratio)
    if ratio_text:
        base = f"{base} ({ratio_text})"
    if row["train_time_s_mean"] > TIME_LIMIT:
        base = f"{base}*"
    return apply_style(base, style)


def make_table(rows: pd.DataFrame, metric: str, title: str) -> str:
    datasets = [d for d in DATASET_ORDER if d in set(rows["dataset"])]
    datasets += sorted(set(rows["dataset"]) - set(datasets))

    lines = [
        f"# {title}",
        "",
        "Cell format: R^2 mean +/- std (MSE ratio vs CLARITree). `*` marks timeout.",
        "",
        "| Dataset | " + " | ".join(LABELS[m] for m in METHODS) + " |",
        "| --- | " + " | ".join("---" for _ in METHODS) + " |",
    ]

    for dataset in datasets:
        dataset_rows = rows[rows["dataset"].eq(dataset)]
        by_method = {
            method: dataset_rows[dataset_rows["method"].eq(method)].iloc[0]
            for method in METHODS
            if not dataset_rows[dataset_rows["method"].eq(method)].empty
        }
        styles = rank_styles(by_method, metric)
        claritree = by_method.get("claritree")
        cells = [
            make_cell(by_method.get(method), metric, claritree, styles.get(method))
            for method in METHODS
        ]
        lines.append(f"| {display_dataset(dataset)} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_tables(rows: pd.DataFrame, out_root: Path, openml: bool = False) -> None:
    suffix = "_openml" if openml else ""
    threshold_col = threshold_group_column(rows)
    for (depth, threshold), group in rows.groupby(["depth", threshold_col], dropna=False):
        threshold_text = threshold_display(threshold)
        threshold_token = threshold_path_token(threshold)
        out_dir = out_root / f"linear_regression_tree_depth{int(depth)}_threshold_{threshold_token}{suffix}"
        out_dir.mkdir(parents=True, exist_ok=True)
        title_suffix = " OpenML" if openml else ""
        train_title = f"Train R^2{title_suffix}, depth {int(depth)}, threshold {threshold_text}"
        test_title = f"Test R^2{title_suffix}, depth {int(depth)}, threshold {threshold_text}"
        train_path = out_dir / f"train{suffix}.md"
        test_path = out_dir / f"test{suffix}.md"
        train_path.write_text(make_table(group, "train_r2", train_title), encoding="utf-8")
        test_path.write_text(make_table(group, "test_r2", test_title), encoding="utf-8")
        print(f"Wrote {train_path}")
        print(f"Wrote {test_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", type=Path, default=None)
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--openml", action="store_true", help="Read results/final_openml.csv and write OpenML tables.")
    parser.add_argument("--5", dest="threshold_5", action="store_true", help="Read/write threshold=5 tables.")
    parser.add_argument("--full", action="store_true", help="Read/write threshold=full tables.")
    args = parser.parse_args()

    if args.full and args.threshold_5:
        parser.error("--full and --5 cannot be used together")
    threshold_label = "full" if args.full else "5" if args.threshold_5 else None
    final_csv = args.final or default_final_csv(args.openml, threshold_label)
    out_root = args.out_root or (OPENML_OUT_ROOT if args.openml else OUT_ROOT)
    write_tables(load_best_rows(final_csv, threshold_label), out_root, args.openml)


if __name__ == "__main__":
    main()
