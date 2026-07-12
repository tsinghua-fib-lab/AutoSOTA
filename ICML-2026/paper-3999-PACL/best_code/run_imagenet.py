import pandas as pd
import numpy as np
import sys
import os
from pac_utils import pac_labeling, zero_one_loss

# Parameters per rubric: ImageNet, ResNet-152, epsilon=0.05, alpha=0.05,
# loss=0-1, uncertainty_source=max_softmax, n_repeats=1000
dataset = "imagenet"
alpha = 0.05
epsilon = 0.05
num_trials = 1000
asymptotic = False

# Load data
data_path = "datasets/imagenet.csv"
if not os.path.exists(data_path):
    # Try alternate path
    data_path = os.path.join("/repo", "datasets/imagenet.csv")
data = pd.read_csv(data_path)
print(f"Loaded {len(data)} rows from {data_path}")
print(f"Columns: {list(data.columns)}")

Y = data["Y"].to_numpy()
Yhat = data["Yhat"].to_numpy()
confidence = data["confidence"].to_numpy()
n = len(Y)

# K = n // 10 per notebook for ImageNet
K = n // 10
print(f"n={n}, K={K}, asymptotic={asymptotic}, alpha={alpha}, epsilon={epsilon}")
print(f"Running {num_trials} trials...")

pi = np.ones(n)
errs = np.zeros(num_trials)
percent_saved = np.zeros(num_trials)

for i in range(num_trials):
    if i % 100 == 0:
        print(f"  Trial {i}/{num_trials}...")
    uncertainty = 1 - confidence + 1e-5 * np.random.normal(size=n)  # break ties
    Y_tilde, labeled_inds, _ = pac_labeling(
        Y, Yhat, zero_one_loss, epsilon, alpha, uncertainty, pi, K, asymptotic=asymptotic
    )
    errs[i] = zero_one_loss(Y, Y_tilde) * 100  # as percentage
    percent_saved[i] = np.mean(labeled_inds == 0.0) * 100

# Report results matching rubric format
error_quantile = np.quantile(errs, 1 - alpha)
budget_save_mean = np.mean(percent_saved)
budget_save_std = np.std(percent_saved)

print()
print("=" * 60)
print("REPRODUCTION RESULTS")
print("=" * 60)
print(f"Dataset: ImageNet (ResNet-152, max_softmax uncertainty)")
print(f"Parameters: epsilon={epsilon}, alpha={alpha}, loss=0-1, trials={num_trials}")
print(f"K (calibration draws): {K}, asymptotic: {asymptotic}")
print()
print(f"Error (1-alpha quantile over {num_trials} trials): {error_quantile:.4f}%")
print(f"  Paper reports: 4.73%")
print(f"  CI bounds: [2.0, 5.003]")
print()
print(f"Budget save (mean +/- std): {budget_save_mean:.2f} +/- {budget_save_std:.2f} %")
print(f"  Paper reports: 59.64 +/- 1.49 %")
print(f"  CI bounds: [58.15, 61.13]")
print()
# Check if within CI bounds
bs_lower, bs_upper = 58.15, 61.13
err_lower, err_upper = 2.0, 5.003
budget_ok = bs_lower <= budget_save_mean <= bs_upper
error_ok = err_lower <= error_quantile <= err_upper
print(f"Budget save within CI [{bs_lower}, {bs_upper}]: {budget_save_ok}")
print(f"Error within CI [{err_lower}, {err_upper}]: {error_ok}")
print("=" * 60)

# Save raw results for manifest
np.savez("/repo/reproduction_results.npz",
         errs=errs, percent_saved=percent_saved,
         budget_save_mean=budget_save_mean, budget_save_std=budget_save_std,
         error_quantile=error_quantile)
print("Raw results saved to /repo/reproduction_results.npz")
