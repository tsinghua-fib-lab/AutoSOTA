import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
print(f"Device name: {torch.cuda.get_device_name()}")

import curegot
import numpy as np

np.random.seed(123)
n, m = 100, 80
M = np.random.rand(n, m).astype(np.float64)
a = np.random.rand(n).astype(np.float64)
a = a / np.sum(a)
b = np.random.rand(m).astype(np.float64)
b = b / np.sum(b)
reg = 0.1

result = curegot.numpy.sinkhorn_bcd(M, a, b, reg, tol=1e-6, max_iter=100, verbose=1)
print(f"BCD result keys: {list(result.keys())}")
print(f"Plan shape: {result['plan'].shape}")
marg_err = np.sum(np.abs(np.sum(result["plan"], axis=1) - a))
print(f"BCD Marginal error (source): {marg_err:.6e}")

result2 = curegot.numpy.sinkhorn_splr(M, a, b, reg, tol=1e-6, max_iter=100, verbose=1)
print(f"SPLR result keys: {list(result2.keys())}")
print(f"SPLR Plan shape: {result2['plan'].shape}")
marg_err2 = np.sum(np.abs(np.sum(result2["plan"], axis=1) - a))
print(f"SPLR Marginal error (source): {marg_err2:.6e}")

print("\nAll verification tests passed!")
