#!/usr/bin/env python3
import argparse, re
import pandas as pd
import matplotlib.pyplot as plt

def parse_trnas_out(path):
    rows = []
    true_score = None
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('-') or s.startswith('Sequence') or s.startswith('Name'):
                continue
            # first token = sequence name
            name = s.split()[0]
            # last numeric token on the line = score (Note, if any, is non-numeric)
            nums = re.findall(r'[-+]?\d+(?:\.\d+)?', s)
            if not nums:
                continue
            score = float(nums[-1])

            if name == "true":
                true_score = score
                continue

            m = re.match(r'^trna_(cons|nocons)_(\d+)$', name)
            if not m:
                continue
            cond, prefix = m.group(1), int(m.group(2))
            rows.append({"prefix": prefix, "cond": cond, "score": score})
    df = pd.DataFrame(rows)
    return df, true_score

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trnas_out", default="/home/mica/gamba/trnas.out")
    ap.add_argument("--out-png", default="/home/mica/gamba/data_processing/data/generation_datainfernal_by_prefix.png")
    ap.add_argument("--out-csv", default="/home/mica/gamba/data_processing/data/generation_datainfernal_by_prefix.csv")
    args = ap.parse_args()

    df, true_score = parse_trnas_out(args.trnas_out)
    if df.empty:
        raise SystemExit("No cons/nocons rows parsed. Check file format or names.")

    # One score per prefix per condition. If duplicates, keep max.
    pivot = (
        df.groupby(["prefix", "cond"])["score"].max()
          .unstack("cond")
          .sort_index()
    )
    pivot.to_csv(args.out_csv, index=True)

    ax = pivot.plot(marker="o")
    if true_score is not None:
        ax.axhline(true_score, linestyle="--", label="true")
    ax.set_xlabel("Prefix length")
    ax.set_ylabel("Infernal score")
    ax.set_title("tRNAscan-SE Infernal score by prefix")
    ax.legend()
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=200)
    print(f"Wrote {args.out_png} and {args.out_csv}")

if __name__ == "__main__":
    main()
