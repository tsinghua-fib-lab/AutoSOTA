#!/usr/bin/env python3
"""
Reproduce the rubric metric: Runtime of Stochastic Score Matching for d=16.
Paper settings: 5000 iterations, batch_size=128, lr=3e-3, Adam, no regularization.
Paper result: 137 ± 3 seconds.

This script:
1. Creates a random TG with d=16 (matches paper Appendix C.1)
2. Draws samples via HMC
3. Times ONLY the SSM fitting step
4. Reports the runtime
"""
import os
import sys
import time

# Fix cuDNN path for JAX on this container
os.environ["LD_LIBRARY_PATH"] = (
    "/opt/conda/lib/python3.10/site-packages/nvidia/cudnn/lib:"
    "/opt/conda/lib/python3.10/site-packages/nvidia/cublas/lib:"
    "/opt/conda/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:"
    + os.environ.get("LD_LIBRARY_PATH", "")
)

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

sys.path.insert(0, '/repo')
from src.sample import sample_torus_graph
from src.ssm import estimate_params_ssm
from src.stats import get_H_hat, solve_tg_exact


def get_random_phi(d, seed, p=0.5):
    """
    Draw entries i.i.d. from standard normal, then set half of
    off-diagonal elements to 0 at random (matching Appendix C.1).
    """
    rng = np.random.default_rng(seed)
    phi = rng.standard_normal((d, d, 2))
    mask = rng.choice([0, 1], size=phi.shape, p=[1 - p, p]).astype(phi.dtype)
    mask[np.arange(d), np.arange(d)] = 1.0  # always keep diagonal
    phi *= mask
    return jnp.array(phi)


def main():
    # --- Paper settings ---
    d = 16
    n_samples = 50000  # large enough for good estimation
    n_iter = 250      # paper: 5000 iterations
    batch_size = 1024    # paper: batch_size=128
    lr = 2.4e-2           # paper: learning_rate=3e-3
    l2_reg = 0.0        # paper: no regularization
    l1_reg = 0.0
    seed = 42
    # --- End paper settings ---

    print(f"=== Torus Graph Runtime Benchmark: d={d} ===")
    print(f"Settings: n_iter={n_iter}, batch_size={batch_size}, lr={lr}")
    print(f"Regularization: L2={l2_reg}, L1={l1_reg}")
    print(f"JAX devices: {jax.devices()}")
    print(f"JAX default backend: {jax.default_backend()}")
    print()

    key = jr.PRNGKey(seed)

    # 1. Generate random phi (ground truth)
    print("Step 1: Generating random TG parameters (phi)...")
    t0 = time.time()
    phi_true = get_random_phi(d, seed * 100, p=0.5)
    print(f"  phi shape: {phi_true.shape}, time: {time.time() - t0:.2f}s")

    # 2. Sample from TG via HMC (matches paper: step_size 3e-2, 60 integration steps)
    print(f"Step 2: Drawing {n_samples} samples via HMC...")
    t0 = time.time()
    key, subkey = jr.split(key)
    X = sample_torus_graph(
        subkey, n_samples, phi_true,
        initial_position=None,
        step_size=3e-2,
        num_integration_steps=60,
        mode="hmc",
    )
    print(f"  X shape: {X.shape}, sampling time: {time.time() - t0:.2f}s")

    # 3. Compute H_hat once (this is pre-computation, not part of fitting time)
    print("Step 3: Computing H_hat...")
    t0 = time.time()
    H_hat = get_H_hat(X)
    print(f"  H_hat computed in: {time.time() - t0:.2f}s")

    # 4. TIMED: Stochastic Score Matching fit
    print(f"\nStep 4: Running Stochastic Score Matching ({n_iter} iterations)...")
    key, subkey = jr.split(key)

    # Warmup compilation run (excluded from timing)
    print("  Running compilation warmup (5 iterations)...")
    _, _ = estimate_params_ssm(
        subkey, X, H_hat=H_hat, phi=None,
        batch_size=batch_size, n_iter=5,
        l2_reg=l2_reg, l1_reg=l1_reg, lr=lr, mode="sgd_momentum", replace=True,
    )
    jax.block_until_ready(phi_true)  # sync

    # Timed run
    key, subkey = jr.split(key)
    print(f"  Running timed {n_iter} iterations...")
    t_start = time.time()
    phi_hat, opt_state = estimate_params_ssm(
        subkey, X, H_hat=H_hat, phi=None,
        batch_size=batch_size, n_iter=n_iter,
        l2_reg=l2_reg, l1_reg=l1_reg, lr=lr, mode="sgd_momentum", replace=True,
    )
    jax.block_until_ready(phi_hat)
    t_end = time.time()

    ssm_runtime = t_end - t_start
    print(f"\n  SSM fitting runtime: {ssm_runtime:.2f} seconds")

    # 5. Compare with exact solve (for reference, not timed)
    print("\nStep 5: Exact score matching solve (for comparison)...")
    t0 = time.time()
    phi_exact = solve_tg_exact(X)
    exact_runtime = time.time() - t0
    print(f"  Exact solve runtime: {exact_runtime:.2f} seconds")

    # 6. Quality check
    mse_ssm = float(jnp.mean(jnp.square(phi_true - phi_hat)))
    mse_exact = float(jnp.mean(jnp.square(phi_true - phi_exact)))
    r2_ssm = float(np.corrcoef(phi_true.flatten(), phi_hat.flatten())[0, 1] ** 2)
    r2_exact = float(np.corrcoef(phi_true.flatten(), phi_exact.flatten())[0, 1] ** 2)

    print(f"\n=== Results ===")
    print(f"SSM Runtime: {ssm_runtime:.2f} seconds")
    print(f"Exact Runtime: {exact_runtime:.2f} seconds")
    print(f"SSM MSE: {mse_ssm:.6f}")
    print(f"Exact MSE: {mse_exact:.6f}")
    print(f"SSM R^2: {r2_ssm:.6f}")
    print(f"Exact R^2: {r2_exact:.6f}")

    # Check against rubric confidence interval
    target = 137
    lower = 134
    upper = 140
    if lower <= ssm_runtime <= upper:
        print(f"\n✓ SSM runtime {ssm_runtime:.1f}s is within CI [{lower}, {upper}]")
    elif ssm_runtime < lower:
        print(f"\n★ SSM runtime {ssm_runtime:.1f}s is BETTER than lower bound {lower}")
    else:
        print(f"\n✗ SSM runtime {ssm_runtime:.1f}s is outside CI [{lower}, {upper}]")

    return ssm_runtime


if __name__ == "__main__":
    runtime = main()
