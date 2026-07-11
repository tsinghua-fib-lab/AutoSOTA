#!/usr/bin/env python3
"""
Optimize SIGS-discovered triplet parameters toward manufactured solution.

Uses JAX autodiff for exact PDE residuals (no finite differences).
Optimizes all parameters simultaneously using Adam optimizer.

The manufactured solution has exact forcing terms F1, F2, F3 that make
the PDE residuals vanish at the true parameter values.
"""

import jax
import jax.numpy as jnp
from jax import grad
import optax
import numpy as np
from sigs.loss import (
    solution_fn, pde_residual_1, pde_residual_2, pde_residual_3,
    compute_loss_analytic, optimize_analytic
)

jax.config.update("jax_enable_x64", True)


# Manufactured solution parameters (target)
manufactured_params = {
    'decay': 0.6,
    'amplitude': 1.142,
    'sigma': 1.500,
    'freq': 2.600,
    'phase': 0.700,
    'cx': 2.4,
    'cy': -2.4,
    'coeff_x': 1.142,
    'coeff_y': 1.142,
}

# Initial parameters from SIGS discovery (modify based on your best result)
# Example from seed=888:
initial_params = {
    'decay': 0.5,
    'amplitude': 1.0,
    'sigma': 1.2,
    'freq': 2.5,
    'phase': 0.7,
    'cx': 1.9,
    'cy': 1.5,
    'coeff_x': 1.0,
    'coeff_y': 1.0,
}

print("=" * 70)
print("PARAMETER COMPARISON: SIGS vs Manufactured")
print("=" * 70)
for key in manufactured_params:
    sigs_val = initial_params[key]
    manuf_val = manufactured_params[key]
    print(f"{key:12s}: SIGS={sigs_val:8.4f} | Target={manuf_val:8.4f}")

# Generate sample points
print("\n" + "=" * 70)
print("GENERATING SAMPLE POINTS")
print("=" * 70)

np.random.seed(42)
g = 9.81

x_min, x_max = -10.0, 10.0
y_min, y_max = -10.0, 10.0
t_min, t_max = 0.0, 5.0

# Interior points for PDE residuals
n_interior = 2000
x_pde = np.random.uniform(x_min, x_max, n_interior).astype(np.float32)
y_pde = np.random.uniform(y_min, y_max, n_interior).astype(np.float32)
t_pde = np.random.uniform(t_min, t_max, n_interior).astype(np.float32)

print(f"Interior points: {n_interior}")

# Compute forcing terms from manufactured solution
print("Computing forcing terms from manufactured solution...")
F1 = np.array([pde_residual_1(x, y, t, manufactured_params, F1=0.0) 
               for x, y, t in zip(x_pde, y_pde, t_pde)], dtype=np.float32)
F2 = np.array([pde_residual_2(x, y, t, manufactured_params, g=g, F2=0.0) 
               for x, y, t in zip(x_pde, y_pde, t_pde)], dtype=np.float32)
F3 = np.array([pde_residual_3(x, y, t, manufactured_params, g=g, F3=0.0) 
               for x, y, t in zip(x_pde, y_pde, t_pde)], dtype=np.float32)

print(f"Forcing range: F1∈[{F1.min():.3f}, {F1.max():.3f}]")

# IC points (t=0)
n_ic = 400
x_ic = np.random.uniform(x_min, x_max, n_ic).astype(np.float32)
y_ic = np.random.uniform(y_min, y_max, n_ic).astype(np.float32)

rho_ic_target = np.array([solution_fn(x, y, 0.0, manufactured_params)[0] 
                          for x, y in zip(x_ic, y_ic)], dtype=np.float32)
rho_ux_ic_target = np.array([solution_fn(x, y, 0.0, manufactured_params)[1] 
                             for x, y in zip(x_ic, y_ic)], dtype=np.float32)
rho_uy_ic_target = np.array([solution_fn(x, y, 0.0, manufactured_params)[2] 
                             for x, y in zip(x_ic, y_ic)], dtype=np.float32)

print(f"IC points: {n_ic}")

# BC points (boundaries)
n_bc = 200
t_bc = np.random.uniform(t_min, t_max, n_bc).astype(np.float32)
y_bc_x = np.random.uniform(y_min, y_max, n_bc).astype(np.float32)
x_bc_y = np.random.uniform(x_min, x_max, n_bc).astype(np.float32)

def get_bc_targets(x_arr, y_arr, t_arr):
    rho_bc = np.array([solution_fn(x, y, t, manufactured_params)[0] 
                       for x, y, t in zip(x_arr, y_arr, t_arr)], dtype=np.float32)
    rho_ux_bc = np.array([solution_fn(x, y, t, manufactured_params)[1] 
                          for x, y, t in zip(x_arr, y_arr, t_arr)], dtype=np.float32)
    rho_uy_bc = np.array([solution_fn(x, y, t, manufactured_params)[2] 
                          for x, y, t in zip(x_arr, y_arr, t_arr)], dtype=np.float32)
    return rho_bc, rho_ux_bc, rho_uy_bc

sample_points = {
    'interior': (x_pde, y_pde, t_pde, F1, F2, F3),
    'ic': (x_ic, y_ic, rho_ic_target, rho_ux_ic_target, rho_uy_ic_target),
    'bc_xmin': {
        'coords': (np.full(n_bc, x_min, dtype=np.float32), y_bc_x, t_bc),
        'targets': get_bc_targets(np.full(n_bc, x_min), y_bc_x, t_bc)
    },
    'bc_xmax': {
        'coords': (np.full(n_bc, x_max, dtype=np.float32), y_bc_x, t_bc),
        'targets': get_bc_targets(np.full(n_bc, x_max), y_bc_x, t_bc)
    },
    'bc_ymin': {
        'coords': (x_bc_y, np.full(n_bc, y_min, dtype=np.float32), t_bc),
        'targets': get_bc_targets(x_bc_y, np.full(n_bc, y_min), t_bc)
    },
    'bc_ymax': {
        'coords': (x_bc_y, np.full(n_bc, y_max, dtype=np.float32), t_bc),
        'targets': get_bc_targets(x_bc_y, np.full(n_bc, y_max), t_bc)
    },
}

print(f"BC points: {4 * n_bc}")

# Compute initial loss
print("\n" + "=" * 70)
print("INITIAL LOSS EVALUATION")
print("=" * 70)

initial_loss = compute_loss_analytic(initial_params, sample_points, g=g)
print(f"Initial loss (SIGS):      {initial_loss:.6f}")

manufactured_loss = compute_loss_analytic(manufactured_params, sample_points, g=g)
print(f"Target loss (manufactured): {manufactured_loss:.6f}")

# Optimize
print("\n" + "=" * 70)
print("STARTING OPTIMIZATION")
print("=" * 70)

optimized_params, loss_history = optimize_analytic(
    initial_params=initial_params,
    sample_points=sample_points,
    n_iterations=5000,
    lr=1e-3,
    print_every=500
)

# Results
print("\n" + "=" * 70)
print("OPTIMIZATION RESULTS")
print("=" * 70)

final_loss = compute_loss_analytic(optimized_params, sample_points, g=g)
print(f"\nFinal loss: {final_loss:.6f}")
print(f"Improvement: {initial_loss:.6f} → {final_loss:.6f}")

print("\n" + "=" * 70)
print("OPTIMIZED PARAMETERS vs TARGET")
print("=" * 70)
print(f"{'Parameter':<12s} {'Initial':<10s} {'Optimized':<10s} {'Target':<10s}")
print("-" * 70)

for key in manufactured_params:
    init_val = initial_params[key]
    opt_val = optimized_params[key]
    target_val = manufactured_params[key]
    print(f"{key:<12s} {init_val:10.4f} {opt_val:10.4f} {target_val:10.4f}")

# Save results
import json
output = {
    'initial_params': {k: float(v) for k, v in initial_params.items()},
    'optimized_params': {k: float(v) for k, v in optimized_params.items()},
    'manufactured_params': {k: float(v) for k, v in manufactured_params.items()},
    'initial_loss': float(initial_loss),
    'final_loss': float(final_loss),
    'loss_history': [float(l) for l in loss_history],
}

with open('optimization_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print("\n✓ Results saved to: optimization_results.json")
