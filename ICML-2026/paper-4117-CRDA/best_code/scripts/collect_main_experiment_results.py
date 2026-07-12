"""
collect_main_experiment_results.py

Aggregate per-seed results for the main MLP / XGBoost experiments.

Walks:
    experiments/<dataset>/<baseline>_<timestamp>/interim_results/<tag>_interim_results.csv

and, for every (dataset, baseline, sample_size), computes the mean and the
standard error of the mean (SEM = sample_std / sqrt(n)) for each metric directly
from the per-seed values.

It deliberately does NOT read the pre-aggregated `results.csv` `std` column,
whose meaning is ambiguous across experiment versions (raw std in early runs,
SEM in later runs). Computing from the per-seed data makes the result
unambiguous and independent of when the experiment was run.

Outputs (written next to this script, in scripts/):
    all_results.csv
    all_results.json

Columns:
    dataset, baseline, sample_size,
    mse, mse_se, aug_mse, aug_mse_se,
    delta_mse, delta_mse_se, p_wilcoxon, p_wilcoxon_se,
    should_proceed, n_seeds
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

METRICS = ["mse", "aug_mse", "delta_mse", "p_wilcoxon"]


def _proceed_fraction(series: pd.Series) -> float:
    """Fraction of seeds that proceeded, robust to bool or 'True'/'False' strings."""
    truthy = series.dropna().map(lambda v: str(v).strip().lower() in ("true", "1", "1.0"))
    return truthy.mean() if len(truthy) else float("nan")


def collect_experiment_rows(exp_root: Path) -> list[dict]:
    """One row per (dataset, baseline, sample_size), aggregated from per-seed interim files."""
    rows: list[dict] = []

    for dataset_dir in sorted(exp_root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        dataset_name = dataset_dir.name

        for run_dir in sorted(dataset_dir.iterdir()):
            if not run_dir.is_dir():
                continue

            interim_dir = run_dir / "interim_results"
            if not interim_dir.is_dir():
                continue

            baseline = run_dir.name.split("_")[0]  # e.g. "mlp", "xgboost"

            for csv_path in sorted(interim_dir.glob("*_interim_results.csv")):
                try:
                    df = pd.read_csv(csv_path)
                except Exception as exc:
                    print(f"Could not read {csv_path}: {exc}")
                    continue

                tag = csv_path.name.replace("_interim_results.csv", "")
                sample_size = int(tag.split("_")[-1]) if "sample_" in tag else None

                row: dict[str, object] = {
                    "dataset": dataset_name,
                    "baseline": baseline,
                    "sample_size": sample_size,
                }

                n_seeds = 0
                for metric in METRICS:
                    if metric not in df.columns:
                        row[metric] = None
                        row[f"{metric}_se"] = None
                        continue
                    vals = df[metric].dropna()
                    n = len(vals)
                    n_seeds = max(n_seeds, n)
                    row[metric] = vals.mean() if n > 0 else None
                    # Standard error of the mean: sample std (ddof=1) / sqrt(n).
                    row[f"{metric}_se"] = vals.std() / np.sqrt(n) if n > 1 else np.nan

                if "should_proceed" in df.columns:
                    frac = _proceed_fraction(df["should_proceed"])
                    row["should_proceed"] = bool(frac > 0.5) if frac == frac else None
                else:
                    row["should_proceed"] = None

                row["n_seeds"] = n_seeds
                rows.append(row)

    return rows


def main() -> None:
    scripts_dir = Path(__file__).resolve().parent
    exp_root = scripts_dir.parent / "experiments"
    if not exp_root.exists():
        raise SystemExit(f"Path not found: {exp_root}")

    rows = collect_experiment_rows(exp_root)
    if not rows:
        raise SystemExit(f"No interim_results found under {exp_root}")

    col_order = (
        ["dataset", "baseline", "sample_size"]
        + [c for m in METRICS for c in (m, f"{m}_se")]
        + ["should_proceed", "n_seeds"]
    )

    df = (
        pd.DataFrame(rows)
        .sort_values(["dataset", "baseline", "sample_size"])
        .loc[:, col_order]
    )

    csv_out = scripts_dir / "all_results.csv"
    json_out = scripts_dir / "all_results.json"

    df.to_csv(csv_out, index=False)
    json_out.write_text(json.dumps(json.loads(df.to_json(orient="records")), indent=2))

    print(df.to_string(index=False))
    print(f"\nSaved {csv_out}")
    print(f"Saved {json_out}")


if __name__ == "__main__":
    main()
