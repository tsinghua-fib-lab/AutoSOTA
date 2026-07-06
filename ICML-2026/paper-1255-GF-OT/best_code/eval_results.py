import pandas as pd
import numpy as np
import torch
import sys

# Load results
results = pd.read_pickle("results/exp_gaussian/results_penalized.pkl")
print("Loaded results:", len(results), "rows")
print("Columns:", results.columns.tolist())

# Find result with cost_diff closest to the expected ~0.07 at moderate penalty
# The rubric says: penalty at log-space midpoint of [1, 1000] penalty grid
# logspace(0, 3, 80) -> midpoint index = 40
mid_idx = len(results) // 2
mid_row = results.iloc[mid_idx]
print(f"\nMidpoint (index {mid_idx}):")
print(f"  Penalty: {mid_row['penalty']}")
print(f"  Cost diff: {mid_row['cost_diff']}")
print(f"  Fairness loss: {mid_row['fairness_loss_value']}")

# Also print summary stats
print(f"\nAll results summary:")
print(f"  Penalty range: [{results['penalty'].min():.2f}, {results['penalty'].max():.2f}]")
print(f"  Cost diff range: [{results['cost_diff'].min():.6f}, {results['cost_diff'].max():.6f}]")
print(f"  Fairness loss range: [{results['fairness_loss_value'].min():.6f}, {results['fairness_loss_value'].max():.6f}]")

# Print first and last few rows
print("\nFirst 5 rows:")
print(results[["penalty", "cost_diff", "fairness_loss_value"]].head())
print("\nLast 5 rows:")
print(results[["penalty", "cost_diff", "fairness_loss_value"]].tail())

# Print rows near the expected values
print("\nRows with cost_diff near 0.07:")
near_expected = results.iloc[(results["cost_diff"] - 0.07).abs().argsort()[:5]]
print(near_expected[["penalty", "cost_diff", "fairness_loss_value"]])
