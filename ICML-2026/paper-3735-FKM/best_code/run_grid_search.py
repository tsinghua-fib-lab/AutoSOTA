import warnings
warnings.filterwarnings("ignore")

import torch
import cupy as cp
import cufinufft
import numpy as np
import time
import sys
import json
from datetime import datetime

device = torch.device("cuda:0")
torch.set_default_dtype(torch.float64)

SEP = "=" * 60

# ===== Helper functions from notebook =====
def conjugate_gradient(A_function, b, x0, tol=1e-10, display=True):
    m = b.shape[0]
    x = x0
    r = b - A_function(x)
    p = r.clone()
    rs_old = torch.linalg.vector_norm(r)**2
    for iteration in range(3*m-2):
        Ap = A_function(p)
        alpha = rs_old / torch.vdot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = torch.linalg.vector_norm(r)**2
        if torch.sqrt(rs_new) < tol:
            if display:
                print("  CG converged in {} iterations.".format(iteration))
            break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    return x

def NUFFT_Y(x_gpu, y_gpu, m, threshold):
    n = x_gpu.shape[0]
    f_gpu = cufinufft.nufft1d1(x_gpu, y_gpu, (2*m+1,), eps=threshold)/n
    return torch.utils.dlpack.from_dlpack(f_gpu)

def kernel_vect(x1_gpu, x2_gpu, m, threshold):
    n = x1_gpu.shape[0]
    y_gpu = cp.ones(n).astype(cp.complex128)
    f_gpu = cufinufft.nufft2d1(x1_gpu, x2_gpu, y_gpu, (2*m+1,2*m+1), eps=threshold)/n
    return torch.utils.dlpack.from_dlpack(f_gpu)

def A_function(x, lambda_n, mat):
    return mat @ x + lambda_n * x

def NUFFT_inv(x_gpu, f_gpu, threshold):
    y_estimated = cufinufft.nufft1d2(x_gpu, f_gpu, eps=threshold)
    return torch.utils.dlpack.from_dlpack(y_estimated)

# ===== Experiment parameters =====
cp.random.seed(seed=1)
torch.manual_seed(1)

d = 5
s = 2
lambda_N_list = [10**(-j/10) for j in range(300)]  # 300 lambda candidates

results = {}

for i in range(1, 17):
    N = int(10**(i/2))
    
    print("\n" + SEP)
    print("i={}, N={:,}".format(i, N))
    print(SEP)
    sys.stdout.flush()

    # Training data
    x_gpu = cp.random.uniform(size=(d, N), dtype=cp.float64)
    f_gpu = cp.zeros(N, dtype=cp.float64)
    for dim in range(d):
        f_gpu += cp.exp(x_gpu[dim]/(dim+1)) - 1
    y_gpu = f_gpu.astype(cp.complex128) + cp.random.normal(size=N).astype(cp.complex128)

    threshold = N**(-2*s/(2*s+1))/10
    m = 1 + int(N**(1/(2*s+1))/d)
    M = 2*m + 1
    
    print("  m={}, M={}, d*M={}, threshold={:.2e}".format(m, M, d*M, threshold))
    sys.stdout.flush()

    # Validation data
    N_val = min(N, 10**4)
    x_val = cp.random.uniform(size=(d, N_val), dtype=cp.float64)
    f_val = cp.zeros(N_val, dtype=cp.float64)
    for dim in range(d):
        f_val += cp.exp(x_val[dim]/(dim+1)) - 1
    y_val = f_val.astype(cp.complex128) + cp.random.normal(size=N_val).astype(cp.complex128)
    err_val = torch.inf

    # Fitting
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()

    # Compute cov_y via NUFFT
    cov_y = torch.empty(d * M, dtype=torch.complex128, device=device)
    for dim in range(d):
        cov_y[dim*M:(dim+1)*M] = NUFFT_Y(x_gpu[dim], y_gpu, m, threshold)

    # Compute cov_x blocks via 2D NUFFT
    cov_x = torch.empty((d * M, d * M), dtype=torch.complex128, device=device)
    for i1 in range(d):
        for i2 in range(i1, d):
            cov_x[i1*M:(i1+1)*M, i2*M:(i2+1)*M] = kernel_vect(x_gpu[i1], -x_gpu[i2], m, threshold)
            if i1 != i2:
                cov_x[i2*M:(i2+1)*M, i1*M:(i1+1)*M] = cov_x[i1*M:(i1+1)*M, i2*M:(i2+1)*M].conj().T

    print("  cov_x and cov_y computed.")
    sys.stdout.flush()

    # Grid search over lambda
    for li, lambda_n in enumerate(lambda_N_list):
        A_funct_ = lambda x, ln=lambda_n: A_function(x, ln, cov_x)
        
        if li == 0:
            hat_theta = conjugate_gradient(A_funct_, cov_y, cov_y, threshold, display=False)
        else:
            hat_theta = conjugate_gradient(A_funct_, cov_y, hat_theta, threshold, display=False)

        estimator_val = cp.zeros(N_val, dtype=cp.float64)
        for dim in range(d):
            estimator_val += torch.real(NUFFT_inv(x_val[dim], hat_theta[(2*m+1)*dim:(2*m+1)*(dim+1)], threshold))

        error = torch.tensor((estimator_val-y_val).get(), dtype=torch.float64)
        mse = torch.mean(torch.square(torch.abs(error)))
        if mse < err_val:
            err_val = mse
            hat_theta_best = hat_theta

    end_event.record()
    torch.cuda.synchronize()
    elapsed = start_event.elapsed_time(end_event) / 1e3
    
    print("  Total time: {:.3f}s".format(elapsed))
    sys.stdout.flush()

    # Test error
    N_test = 10**4
    x_test = cp.random.uniform(size=(d, N_test), dtype=cp.float64)
    f_test = cp.zeros(N_test, dtype=cp.float64)
    for dim in range(d):
        f_test += cp.exp(x_test[dim]/(dim+1)) - 1
    y_test = f_test.astype(cp.complex128)

    estimator = cp.zeros(N_test, dtype=cp.float64)
    for dim in range(d):
        estimator += torch.real(NUFFT_inv(x_test[dim], hat_theta_best[(2*m+1)*dim:(2*m+1)*(dim+1)], threshold))

    error = torch.tensor((estimator-y_test).get(), dtype=torch.float64)
    mse = torch.mean(torch.square(torch.abs(error)))
    
    print("  Test MSE: {:.6e}".format(mse.item()))
    sys.stdout.flush()

    results[str(N)] = {
        "N": int(N),
        "m": int(m),
        "M": int(M),
        "running_time_s": float(elapsed),
        "test_mse": float(mse.item()),
        "best_val_mse": float(err_val.item()) if isinstance(err_val, torch.Tensor) else float(err_val),
        "n_lambda": len(lambda_N_list),
        "d": d,
        "s": s,
    }

# ===== Save results =====
output = {
    "paper_id": 3735,
    "experiment": "additive_kernel_grid_search",
    "parameters": {
        "d": d,
        "s": s,
        "n_lambda": len(lambda_N_list),
        "lambda_range": "[{}, {}]".format(lambda_N_list[-1], lambda_N_list[0]),
        "hardware": torch.cuda.get_device_name(0),
    },
    "results": results,
    "timestamp": datetime.now().isoformat(),
}

with open("/repo/grid_search_results.json", "w") as f:
    json.dump(output, f, indent=2)

# Summary
print("\n" + SEP)
print("GRID SEARCH RESULTS SUMMARY")
print(SEP)
for key in sorted(results.keys(), key=lambda k: results[k]["N"]):
    r = results[key]
    print("N={:>12,}: time={:8.3f}s, test_mse={:.6e}".format(r["N"], r["running_time_s"], r["test_mse"]))

# Save as numpy for plotting compatibility
N_list = np.array([r["N"] for r in results.values()])
time_list = np.array([r["running_time_s"] for r in results.values()])
err_list = np.array([r["test_mse"] for r in results.values()])

with open("/repo/results_weakl_gpu_gs.npy", "wb") as f:
    np.save(f, N_list)
    np.save(f, err_list)
    np.save(f, time_list)

print("\nResults saved to /repo/grid_search_results.json and /repo/results_weakl_gpu_gs.npy")
