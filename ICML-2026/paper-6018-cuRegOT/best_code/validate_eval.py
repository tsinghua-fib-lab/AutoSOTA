"""Validate eval command works from any directory."""
import sys
import os

# Remove cwd from path to avoid local curegot conflict
cwd = os.getcwd()
sys.path = [p for p in sys.path if cwd not in p]

import curegot
print(f"curegot loaded from: {curegot.__file__}")

import numpy as np
import time

rng = np.random.RandomState(0)
n, m = 200, 150
x = rng.randn(n).astype(np.float64)
y = rng.randn(m).astype(np.float64)
M = np.abs(x[:, np.newaxis] - y[np.newaxis, :])
M = M / np.max(M)
a = np.ones(n) / n
b = np.ones(m) / m

result = curegot.numpy.sinkhorn_splr(
    M, a, b, 0.001, tol=1e-8, max_iter=5000, verbose=0,
    sparsity_pattern_cycle=10, density=0.05
)
row_err = np.max(np.abs(np.sum(result["plan"], axis=1) - a))
col_err = np.max(np.abs(np.sum(result["plan"], axis=0) - b))
marg_err = max(row_err, col_err)
print(f"Quick test: niter={result['niter']}, marg_err={marg_err:.6e}")
print("Eval command validation PASSED")
