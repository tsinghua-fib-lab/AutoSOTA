#!/usr/bin/env python3
"""Calibrate: understand the MPE scale by running oracle predictions."""
import sys, os
sys.path.insert(0, '/repo/simulation_distribution')

import numpy as np
import torch
from DeSI import DeSI_distribution
from generate_dist import generate_simulation_data_torch_true
from scipy.stats import norm

# Run with true theta to get oracle-level MPE
n = 200
qf_size = 100
p = 4
link = "quadratic"
seed = 0

torch.manual_seed(seed)
np.random.seed(seed)

X, Y, theta, mu, sigma = generate_simulation_data_torch_true(
    n=n, qf_size=qf_size, p=p, link=link, seed=seed
)

# Split: 40% train, 10% val, rest test
idx = np.arange(n)
np.random.shuffle(idx)
n_train = int(0.4 * n)
n_val = int(0.1 * n)
n_test = n - n_train - n_val
idx_train = idx[:n_train]
idx_test = idx[n_train+n_val:]

X_train, X_test = X[idx_train], X[idx_test]
mu_test, sigma_test = mu[idx_test], sigma[idx_test]

# Standardize
X_mean = X_train.mean(dim=0, keepdim=True)
X_std = X_train.std(dim=0, keepdim=True) + 1e-8
X_train_s = (X_train - X_mean) / X_std
X_test_s = (X_test - X_mean) / X_std

# Oracle: use true theta
Z_train = X_train_s @ theta
Z_test = X_test_s @ theta

y_train = [Y[i] for i in idx_train]
result = DeSI_distribution(y=y_train, x=Z_train, xOut=Z_test, h=0.5)
qf_pred = result.get("qf")

# True quantile functions
qfSupp = np.linspace(0, 1, qf_size + 2)[1:-1]
qf_true = np.zeros((n_test, qf_size))
for i in range(n_test):
    si = max(float(sigma_test[i]), 1e-8)
    qf_true[i, :] = norm.ppf(qfSupp, loc=float(mu_test[i]), scale=si)
qf_true_t = torch.tensor(qf_true, dtype=torch.float32)

# Metrics
l2_norms = torch.norm(qf_pred - qf_true_t, dim=1)
avg_l2 = l2_norms.mean().item()
w2_dists = l2_norms / np.sqrt(qf_size)
avg_w2 = w2_dists.mean().item()

print(f"Oracle (true theta, h=0.5):")
print(f"  avg L2 norm:   {avg_l2:.6f}")
print(f"  avg W2 (norm/sqrt({qf_size})): {avg_w2:.6f}")
print(f"  Paper MPE for n=200 quad: 0.2031")
print(f"  Ratio W2/paper: {avg_w2/0.2031:.3f}")

# Try different bandwidths
for h in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
    result_h = DeSI_distribution(y=y_train, x=Z_train, xOut=Z_test, h=h)
    qf_pred_h = result_h.get("qf")
    l2_h = torch.norm(qf_pred_h - qf_true_t, dim=1)
    w2_h = l2_h.mean().item() / np.sqrt(qf_size)
    print(f"  h={h:.1f}: W2={w2_h:.6f}")
