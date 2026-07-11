#!/usr/bin/env python3
"""Reproduction script for APORIA Label Propagation (Table 2)."""
import os, sys, time, json
os.environ["OPENBLAS_NUM_THREADS"] = "4"

import numpy as np
import pandas as pd

import aporia as ap

# ---------------------------------------------------------------
# Config
# ---------------------------------------------------------------
CONFIG_PATH = "config/socrates.toml"
cfg = ap.load_config(CONFIG_PATH)
best_reg_lambda = cfg.experiment.best_lambda  # 1.2

print("=" * 60)
print("APORIA Label Propagation Reproduction")
print(f"Config: {CONFIG_PATH}")
print(f"Dataset: {cfg.dataset.name}")
print(f"Lambda: {best_reg_lambda}")
print(f"Models: {cfg.model_names}")
print("=" * 60)

# ---------------------------------------------------------------
# Load data
# ---------------------------------------------------------------
print("\n[1/3] Loading dataset...")
df = ap.load_dataframe(cfg)
print(f"  Loaded {len(df)} rows, {df['model_id'].nunique()} models, {df['prompt_id'].nunique()} prompts")

# ---------------------------------------------------------------
# Structural Analysis (prerequisite cache)
# ---------------------------------------------------------------
print("\n[2/3] Running structural analysis (prerequisite)...")
t0 = time.time()
results_df, geometry_store, null_store = ap.run_structural_analysis(
    df, cfg,
    use_cache=True,
    overwrite_cache=False,
)
print(f"  Done in {time.time() - t0:.1f}s")

# ---------------------------------------------------------------
# Label Propagation Study (main experiment - Table 2)
# ---------------------------------------------------------------
print("\n[3/3] Running full label propagation study...")
print("  Settings: FisherProjection, lambda=1.2, n_splits=20, test_fraction=1/3")
print("  This reproduces Table 2 of the paper (~20 min runtime)...")

t0 = time.time()
results_lp_max = ap.run_full_label_propagation_study(
    df, cfg,
    projector_class=ap.FisherProjection,
    projector_kwargs={
        "lambda_reg":          best_reg_lambda,
        "normalise":           True,
        "normalise_by_trace":  True,
    },
    train_fractions=None,
    n_iter=20,
    test_fraction=1/3,
    n_splits=20,
    ref_lambda_reg=None,
    use_cache=True,
    cache_dir=f"{cfg.cache.root}/LP-fisher",
    overwrite_cache=False,
    logskip=True,
)
elapsed = time.time() - t0
print(f"  Done in {elapsed:.1f}s ({elapsed/60:.1f} min)")

# ---------------------------------------------------------------
# Results: Aggregate and report
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("RESULTS: Table 2 Reproduction")
print("=" * 60)

# Aggregate by model
agg = (
    results_lp_max
    .groupby("model_id")
    .agg(
        acc_mean=("accuracy", "mean"),
        acc_std=("accuracy", "std"),
        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),
    )
    .reset_index()
)
agg["model_name"] = agg["model_id"].map(cfg.model_names)

# Print per-model results
print(f"\n{Model:<20s} {Accuracy:>12s} {F1:>12s}")
print("-" * 44)
for _, row in agg.iterrows():
    print(f"{row[model_name]:<20s} {row[acc_mean]:>5.1f}% ({row[acc_std]:.1f})  {row[f1_mean]:>5.1f}% ({row[f1_std]:.1f})")

# Overall average
avg_acc = agg["acc_mean"].mean()
avg_f1 = agg["f1_mean"].mean()
avg_acc_std = agg["acc_std"].mean()
avg_f1_std = agg["f1_std"].mean()
print("-" * 44)
print(f"{Average:<20s} {avg_acc:>5.1f}% ({avg_acc_std:.1f})  {avg_f1:>5.1f}% ({avg_f1_std:.1f})")

# ---------------------------------------------------------------
# Focus on mistral-7B (model_id=0)
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("TARGET: mistral-7B (model_id=0)")
print("=" * 60)
mistral_row = agg[agg["model_id"] == 0]
if len(mistral_row) > 0:
    m = mistral_row.iloc[0]
    print(f"  Accuracy: {m[acc_mean]:.1f}% (std={m[acc_std]:.1f})")
    print(f"  F1:       {m[f1_mean]:.1f}% (std={m[f1_std]:.1f})")
    print(f"\n  Paper reports:")
    print(f"    Accuracy: 86.8% (6.6) -> Reproduced: {m[acc_mean]:.1f}% ({m[acc_std]:.1f})")
    print(f"    F1:       92.1% (4.4) -> Reproduced: {m[f1_mean]:.1f}% ({m[f1_std]:.1f})")

# Save results
out_path = "/repo/reproduction_results.json"
output = {
    "paper_id": 3062,
    "dataset": cfg.dataset.name,
    "config": {
        "lambda": best_reg_lambda,
        "n_splits": 20,
        "test_fraction": 1/3,
        "projector": "FisherProjection",
        "encoder": "all-MiniLM-L6-v2",
    },
    "per_model": [
        {
            "model_id": int(row["model_id"]),
            "model_name": row["model_name"],
            "accuracy_mean": float(row["acc_mean"]),
            "accuracy_std": float(row["acc_std"]),
            "f1_mean": float(row["f1_mean"]),
            "f1_std": float(row["f1_std"]),
        }
        for _, row in agg.iterrows()
    ],
    "average": {
        "accuracy_mean": float(avg_acc),
        "accuracy_std": float(avg_acc_std),
        "f1_mean": float(avg_f1),
        "f1_std": float(avg_f1_std),
    },
    "runtime_seconds": elapsed,
}
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nResults saved to {out_path}")
print("\nDONE.")
