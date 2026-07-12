from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

METRICS = ["dataset", "seed", "mse", "aug_mse", "delta_mse", "p_wilcoxon", "should_proceed", "features_perturbed"]

def collect_experiment_rows(exp_root: Path) -> list[dict]:
    """
    Walk `exp_root` and pull rows from every CSV file under interim_results dir.
    """
    rows: list[dict] = []

    # level-1: dataset directory
    for dataset_dir in exp_root.iterdir():
        if not dataset_dir.is_dir() or dataset_dir.name == "15-seeds":
            continue
        dataset_name = dataset_dir.name

        # level-2: run directory (name encodes baseline: {baseline}_{timestamp})
        for run_dir in dataset_dir.iterdir():
            if not run_dir.is_dir():
                continue

            # look for "interim_results" dir
            interim_results_dir = run_dir / "interim_results"
            if not interim_results_dir.exists() or not interim_results_dir.is_dir():
                continue

            # grab all *_interim_results.csv files
            for csv_path in interim_results_dir.glob("*_interim_results.csv"):
                try:
                    df = pd.read_csv(csv_path)
                except Exception as exc:
                    print(f"⚠️  Could not read {csv_path}: {exc}")
                    continue

                for _, row in df.iterrows():
                    rec = {k: row[k] for k in METRICS if k in row}
                    # Optionally add: baseline/run info, sample_size, etc. if needed
                    # sample_size: try extracting from dataset column (e.g. "foobar_sample_300" => 300)
                    if "dataset" in rec:
                        tag = str(rec["dataset"])
                        if "sample_" in tag:
                            try:
                                rec["sample_size"] = int(tag.split("_")[-1])
                            except Exception:
                                rec["sample_size"] = None
                        else:
                            rec["sample_size"] = None
                    else:
                        rec["sample_size"] = None

                    # Include top-level info (dataset, run, etc) if not already
                    rec["dataset_name"] = dataset_name
                    rows.append(rec)

    return rows

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename `dataset_name` -> `dataset`, drop the raw sample-tag column, and
    return the per-seed rows with a fixed column order.
    """

    df = df.drop(columns=["dataset"])
    df = df.rename(columns={"dataset_name": "dataset"})

    df = df[["dataset", "sample_size", "seed", "mse", "aug_mse", "delta_mse", "p_wilcoxon", "should_proceed"]]

    return df

def main() -> None:
    # Root of the CatBoost experiment tree, resolved relative to this script
    # so it works regardless of the current working directory.
    exp_root = Path(__file__).resolve().parent.parent / "experiments_catboost"
    if not exp_root.exists():
        raise SystemExit(f"Path not found: {exp_root}")


    rows = collect_experiment_rows(exp_root)

    # save to exp_root as all_results_catboost.csv
    csv_out = exp_root / "all_results_catboost.csv"

    df = pd.DataFrame(rows)
    df = clean_df(df)
    df.to_csv(csv_out, index=False)
    print(f"Saved {csv_out}")

    # group by dataset and sample_size (averaged across all seeds)
    grouped = df.groupby(["dataset", "sample_size"])
    summary_rows = []

    for (dataset, sample_size), group in grouped:
        deltas = group["delta_mse"].dropna()
        n = len(deltas)

        mean_delta_mse = deltas.mean()
        # Standard error of the mean: sample std (ddof=1) / sqrt(n).
        std_error_delta_mse = deltas.std() / np.sqrt(n) if n > 1 else np.nan

        summary_rows.append({
            "dataset": dataset,
            "sample_size": sample_size,
            "mean_delta_mse": mean_delta_mse,
            "std_err_delta_mse": std_error_delta_mse,
            "n": n
        })

    summary_df = pd.DataFrame(summary_rows)

    summary_df.to_csv(exp_root / "summary_results_catboost.csv", index=False)

    datasets = summary_df["dataset"].unique()
    sample_sizes = [300, 500, 700]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, sample_size in enumerate(sample_sizes):
        ax = axes[idx]
        df_size = summary_df[summary_df["sample_size"] == sample_size]

        x = np.arange(len(datasets))
        width = 0.6

        means = []
        errs = []

        for d in datasets:
            row = df_size[df_size["dataset"] == d]
            if not row.empty:
                means.append(row["mean_delta_mse"].values[0])
                err_val = row["std_err_delta_mse"].values[0]
                errs.append(err_val if pd.notna(err_val) else 0)
            else:
                means.append(np.nan)
                errs.append(0)

        ax.bar(x, means, width, yerr=errs, capsize=5, color='tab:blue')

        ax.set_xlabel("Dataset")
        ax.set_ylabel("Mean Delta MSE")
        ax.set_title(f"Sample Size: {sample_size}")
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=45, ha='right')
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)

    plt.suptitle("Mean Delta MSE by Dataset (averaged across all seeds)", fontsize=14)
    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    main()