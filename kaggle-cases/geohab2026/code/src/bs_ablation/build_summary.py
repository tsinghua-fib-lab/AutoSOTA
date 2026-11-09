"""Aggregate the four experiment CSVs into a single summary table.

Outputs:
  outputs/bs_ablation/summary_table.csv
  outputs/bs_ablation/summary_table.md  (markdown for the Discussion post)
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.bs_ablation.common import OUT_DIR

EXP_CSVS = [
    ("Experiment A — GLCM quantization",  OUT_DIR / "exp_a_glcm_quantization.csv",
     "quantization_level"),
    ("Experiment B — BS Gaussian smoothing", OUT_DIR / "exp_b_smoothing.csv",
     "sigma_pixels"),
    ("Experiment C — Destriped vs original BS", OUT_DIR / "exp_c_destriped.csv",
     "bs_source"),
    ("Experiment D — Texture order",       OUT_DIR / "exp_d_texture_order.csv",
     "texture_type"),
]


def _df_to_markdown(df: pd.DataFrame, float_fmt: str = "{:.4f}") -> str:
    """Hand-rolled markdown table (avoids the `tabulate` dependency)."""
    cols = list(df.columns)
    def _fmt(v):
        if isinstance(v, float):
            return float_fmt.format(v)
        return str(v)
    header = "| " + " | ".join(cols) + " |"
    sep    = "|" + "|".join(["---"] * len(cols)) + "|"
    body   = "\n".join("| " + " | ".join(_fmt(r[c]) for c in cols) + " |"
                       for _, r in df.iterrows())
    return "\n".join([header, sep, body])

DROP_FOR_SUMMARY = {
    "fold0_F1", "fold1_F1", "fold2_F1", "fold3_F1", "fold4_F1",
    "fold_mean_F1", "fold_SE", "fold_SD",
}


def main() -> None:
    summary_rows: list[dict] = []
    md_blocks: list[str] = [
        "# Backscatter Preprocessing Ablation — Summary",
        f"_Generated: {datetime.now():%Y-%m-%d %H:%M}_",
        "",
        "**Baseline:** 87 features = v07 (86) + acr_sapa_w21 (1).  "
        "LightGBM with ALG class_weight=2.0, n_estimators=1000 (no early stopping).  "
        "5-fold 150m BlockKFold (seed=42, from `kaggle_dataset/fold_indices.csv`).  "
        "OOF metrics are pre-Bayes raw probability argmax.",
        "",
    ]

    for label, path, variant_col in EXP_CSVS:
        if not path.exists():
            md_blocks += [f"## {label}", f"_(no results — `{path.name}` missing)_", ""]
            print(f"  SKIP — {path.name} not found")
            continue
        df = pd.read_csv(path)
        md_blocks += [f"## {label}", ""]
        cols = [variant_col, "OOF_weighted_F1", "OOF_kappa", "wasserstein_drift_mean"]
        if "n_features_total" in df.columns:
            cols.insert(1, "n_features_total")
        if "n_features" in df.columns:
            cols.insert(1, "n_features")
        cols = [c for c in cols if c in df.columns]
        md_blocks.append(_df_to_markdown(df[cols]))
        md_blocks.append("")
        for _, r in df.iterrows():
            row = {
                "experiment": label,
                "variant":    r[variant_col],
                "OOF_weighted_F1": r.get("OOF_weighted_F1"),
                "OOF_kappa":       r.get("OOF_kappa"),
                "wasserstein_drift_mean": r.get("wasserstein_drift_mean"),
                "n_features":     r.get("n_features_total", r.get("n_features")),
            }
            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    csv_path = OUT_DIR / "summary_table.csv"
    md_path  = OUT_DIR / "summary_table.md"
    summary_df.to_csv(csv_path, index=False)
    md_path.write_text("\n".join(md_blocks), encoding="utf-8")
    print(f"Saved {csv_path}")
    print(f"Saved {md_path}")
    if not summary_df.empty:
        print("\n--- Compact summary ---")
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
