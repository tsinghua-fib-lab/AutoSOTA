#!/usr/bin/env python3
"""Fast single-model eval for APORIA optimization."""
import os, json, sys
os.environ["OPENBLAS_NUM_THREADS"] = "4"
import numpy as np
import pandas as pd
import aporia as ap

# Get model_id from arg or default to 0 (Mistral-7B)
model_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
overwrite = "--overwrite" in sys.argv

cfg = ap.load_config("config/socrates.toml")
best_lambda = cfg.experiment.best_lambda

print(f"Loading dataset (model {cfg.model_names[model_id]})...")
df = ap.load_dataframe(cfg)

print("Running structural analysis...")
ap.run_structural_analysis(df, cfg, use_cache=True, overwrite_cache=False)

print("Running label propagation study...")
results = ap.run_full_label_propagation_study(
    df, cfg,
    projector_class=ap.FisherProjection,
    projector_kwargs={
        "lambda_reg": best_lambda,
        "normalise": True,
        "normalise_by_trace": True,
    },
    train_fractions=None,
    n_iter=20,
    test_fraction=1/3,
    n_splits=20,
    use_cache=True,
    cache_dir=f"{cfg.cache.root}/LP-fisher-single",
    overwrite_cache=overwrite,
    logskip=True,
    model_ids=[model_id],
)

results = results[results["train_fraction"] == 1.0]

agg = results.groupby("model_id").agg(
    acc_mean=("accuracy", lambda x: x.mean() * 100),
    acc_std=("accuracy", lambda x: x.std() * 100),
    f1_mean=("f1", lambda x: x.mean() * 100),
    f1_std=("f1", lambda x: x.std() * 100),
).reset_index()
agg["model_name"] = agg["model_id"].map(cfg.model_names)

print()
print("%-20s %16s %16s" % ("Model", "Accuracy", "F1"))
print("-" * 52)
for _, r in agg.iterrows():
    a_m, a_s = r["acc_mean"], r["acc_std"]
    f_m, f_s = r["f1_mean"], r["f1_std"]
    print("%-20s %6.1f%% (%4.1f)  %6.1f%% (%4.1f)" % (r["model_name"], a_m, a_s, f_m, f_s))

# Write JSON
out = {}
for _, r in agg.iterrows():
    out[r["model_name"]] = {
        "accuracy_mean": round(float(r["acc_mean"]), 2),
        "accuracy_std": round(float(r["acc_std"]), 2),
        "f1_mean": round(float(r["f1_mean"]), 2),
        "f1_std": round(float(r["f1_std"]), 2),
    }
with open("reproduction_results.json", "w") as f:
    json.dump(out, f, indent=2)

print("\nSingle-model results saved.")
