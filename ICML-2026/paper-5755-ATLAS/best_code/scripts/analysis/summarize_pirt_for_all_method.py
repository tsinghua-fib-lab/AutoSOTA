import pandas as pd
import numpy as np

def calculate_mae_sd(filename):
    df = pd.read_csv(filename)

    abs_error = df["abs_error"].values

    mae = np.mean(abs_error)
    sd = np.std(abs_error)
    se = sd / np.sqrt(len(abs_error))
    n_sub_item=np.mean( df["n_subset_items"].values)
    n_all_item=np.mean(df["n_all_items"].values)

    return {
        "filename": filename,
        "mae": mae,
        "sd": sd,
        "se": se,
        "n": len(abs_error),
        "n_subset_items":n_sub_item,
        "n_all_items": n_all_item

    }

def process_all_files(BM):
    files = [
        # f"baseline_compare/pirt_random_100_vs_actual_{BM}.csv",
        # f"baseline_compare/pirt_tiny{BM}_vs_actual.csv",
        # f"baseline_compare/pirt_metabench_vs_actual_{BM}_primary.csv",
        # f"baseline_compare/pirt_metabench_vs_actual_{BM}_secondary.csv",
        f"{BM}/pirt_vs_actual_se_0.1.csv",
        f"{BM}/pirt_vs_actual_se_0.2.csv",
        f"{BM}/pirt_vs_actual_se_0.3.csv",
    ]

    rows = []
    for f in files:
        row = calculate_mae_sd(f)
        row["benchmark"] = BM  # add BM column
        rows.append(row)

    return pd.DataFrame(rows)

def main():
    BM_list = ["winogrande", "truthfulqa", "hellaswag", "gsm8k", "arc"]

    all_results = []

    for bm in BM_list:
        df = process_all_files(bm)
        all_results.append(df)

    # concatenate all results
    out_df = pd.concat(all_results, ignore_index=True)

    out_path = "summary_pirt_mae_sd_se.csv"
    out_df.to_csv(out_path, index=False)

    print(f"Saved summary: {out_path}")

main()
