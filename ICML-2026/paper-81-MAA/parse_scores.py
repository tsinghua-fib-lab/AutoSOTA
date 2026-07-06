#!/usr/bin/env python3
"""Parse evaluation results from VLMEvalKit output CSV files."""
import csv, os, sys, glob, json

def parse_csv(csv_path):
    """Parse VLMEvalKit accuracy CSV. Extract Overall row."""
    if not os.path.exists(csv_path):
        return None
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("split") == "Overall":
                return float(row["Overall"]) * 100
    return None

def main():
    output_dir = "/repo/outputs/maa_llava"
    dis_csvs = sorted(glob.glob(os.path.join(output_dir, "T*", "maa_llava_R-Bench-Dis_acc.csv")))
    ref_csvs = sorted(glob.glob(os.path.join(output_dir, "T*", "maa_llava_R-Bench-Ref_acc.csv")))

    dis_acc = parse_csv(dis_csvs[-1]) if dis_csvs else None
    ref_acc = parse_csv(ref_csvs[-1]) if ref_csvs else None

    result = {
        "R-Bench-Dis_Accuracy": dis_acc,
        "R-Bench-Ref_Accuracy": ref_acc,
    }
    print(json.dumps(result, indent=2))
    return result

if __name__ == "__main__":
    main()
