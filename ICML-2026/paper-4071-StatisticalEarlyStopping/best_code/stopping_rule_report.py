import os
import json
import argparse
import pandas as pd
from config import MODELS_LIST

DATASETS = ["gpqa", "hle"] # ["mmlu", "umwp", "mc", "mip"] # 


def main():
    parser = argparse.ArgumentParser(
        description="Flatten stopping-rule JSONL results into a table (CSV)."
    )
    parser.add_argument(
        "--results_path",
        type=str,
        default="results",
        help="Directory containing model_name/dataset_stopping_rule_results.jsonl",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="stopping_rule_results_summary.csv",
        help="Output CSV filename",
    )
    args = parser.parse_args()

    rows = []

    print("Gathering stopping rule results...")

    for model_name in MODELS_LIST:
        for dataset in DATASETS:
            path = f"{args.results_path}/{model_name}/{dataset}_stopping_rule_results.jsonl"

            if not os.path.exists(path):
                print(f"Missing: {path}")
                continue

            print(f"Loading {path} ...")

            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line_num, line in enumerate(f, start=1):
                        line = line.strip()
                        if not line:
                            continue

                        try:
                            res = json.loads(line)
                        except json.JSONDecodeError as e:
                            print(f"JSON decode error in {path}:{line_num} → {e}")
                            continue

                        # Remove large array
                        res.pop("tokens_saved_list_ill_posed", None)

                        # Add identifying info
                        res["model_name"] = model_name
                        res["dataset"] = dataset

                        rows.append(res)

            except Exception as e:
                print(f"Error reading {path}: {e}")

    if not rows:
        print("No data found. Exiting.")
        return

    print(f"\nCollected {len(rows)} rows")
    df = pd.DataFrame(rows)

    # Optional: reorder columns
    ordered_cols = [
        "model_name",
        "dataset",
        "stopping_rule",
        "alpha",
        "lazy_interval",
        "ablation",
        "early_stopping_rate_well_posed",
        "early_stopping_rate_ill_posed",
        "avg_tokens_saved_ill_posed",
        "avg_percentage_saved_ill_posed",
        "soft_upperbound",
    ]
    df = df[[c for c in ordered_cols if c in df.columns] +
            [c for c in df.columns if c not in ordered_cols]]

    output_csv = args.output_csv
    if DATASETS == ["mmlu", "umwp", "mc", "mip"]:
        # make directory "math"
        os.makedirs("math", exist_ok=True)
        output_csv = os.path.join("math", output_csv)
    elif DATASETS == ["gpqa", "hle"]:
        os.makedirs("cross", exist_ok=True)
        output_csv = os.path.join("cross", output_csv)
    
    df.to_csv(output_csv, index=False)
    print(f"\nSaved CSV to: {output_csv}")
    print("\n--- Complete ---")


if __name__ == "__main__":
    main()

