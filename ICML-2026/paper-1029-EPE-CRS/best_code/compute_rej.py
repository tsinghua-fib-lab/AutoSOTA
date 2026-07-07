import sys
sys.path.insert(0, "/repo/src")
import pandas as pd
import numpy as np
import os
from utils.metrics import rej_frac
from data.load import load_data
from evaluation.run_experiment import get_job_list

# Load selected best regions
selected = pd.read_pickle("/repo/results/aids_karnof/processed/selected_best.pkl")

# Focus on ddgroup
ddgroup_best = selected[selected["method"] == "ddgroup"]
print("Computing rejection fractions for %d ddgroup regions..." % len(ddgroup_best))

REJ_THRESHOLDS = [0.01, 0.05, 0.10]

results = []
for _, row in ddgroup_best.iterrows():
    seed = row["seed"]
    R = row["R"]
    beta = row["beta"]

    print("Seed %d: loading data..." % seed)
    X_adjust, X_subgp, Y, X_adjust_test, X_subgp_test, Y_test, _, _ = load_data(
        "aids", [2], [0], seed
    )

    print("  Computing train rejection fractions...")
    train_rej = rej_frac(X_adjust, X_subgp, Y, R, beta, REJ_THRESHOLDS, n_jobs=-1)

    print("  Computing test rejection fractions...")
    test_rej = rej_frac(X_adjust_test, X_subgp_test, Y_test, R, beta, REJ_THRESHOLDS, n_jobs=-1)

    results.append({
        "seed": seed,
        "train_rej_01": train_rej[0],
        "train_rej_05": train_rej[1],
        "train_rej_10": train_rej[2],
        "test_rej_01": test_rej[0],
        "test_rej_05": test_rej[1],
        "test_rej_10": test_rej[2],
    })
    print("  Done. test_rej_10=%.4f" % test_rej[2])

rej_df = pd.DataFrame(results)
print("\n=== Rejection Fraction Summary (DDGroup) ===")
for col in ["test_rej_01", "test_rej_05", "test_rej_10"]:
    vals = rej_df[col].dropna()
    print("  %s: %.4f (%.4f)" % (col, vals.mean(), vals.sem()))

os.makedirs("/repo/results/aids_karnof/processed", exist_ok=True)
rej_df.to_pickle("/repo/results/aids_karnof/processed/rejection_fractions.pkl")
print("\nSaved to results/aids_karnof/processed/rejection_fractions.pkl")
