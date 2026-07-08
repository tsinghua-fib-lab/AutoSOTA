#!/usr/bin/env python3
"""Test trust-constr as an alternative to augmented Lagrangian for torus geodesic."""
import time, numpy as np
from scipy.optimize import minimize, NonlinearConstraint

R, r = 0.8, 0.2
def phi_single(z):
    """Constraint for a single point: torus implicit function."""
    rho = np.sqrt(z[0]**2 + z[1]**2)
    return (rho - R)**2 + z[2]**2 - r**2

def torus_point(u, v):
    x = (R + r * np.cos(v)) * np.cos(u)
    y = (R + r * np.cos(v)) * np.sin(u)
    z = r * np.sin(v)
    return np.array([x, y, z])

def compute_path_energy(geodesic, K):
    return K * float(np.sum((geodesic[1:] - geodesic[:-1]) ** 2))

def slerp_init(uA, vA, uB, vB, resolution):
    t = np.linspace(0, 1, resolution + 2)[1:-1]
    us = uA + t * (uB - uA)
    vs = vA + t * (vB - vB)
    x = (R + r * np.cos(vs)) * np.cos(us)
    y = (R + r * np.cos(vs)) * np.sin(us)
    z = r * np.sin(vs)
    return np.column_stack([x, y, z])

uA, vA = 0.0, 0.0
uB, vB = 2.7658, 1.1597
resolution = 48
n = 3  # 3D ambient space
K = resolution + 1

xA = torus_point(uA, vA)
xB = torus_point(uB, vB)
x0_slerp = slerp_init(uA, vA, uB, vB, resolution)

# Energy function: E(x) = K * sum(|z_k - z_{k-1}|^2)
# x is flat vector of intermediate nodes: shape (resolution * n,)
def energy_flat(x_flat):
    coords = x_flat.reshape(resolution, n)
    full = np.vstack([xA, coords, xB])
    diffs = full[1:] - full[:-1]
    return K * np.sum(diffs**2)

def energy_grad(x_flat):
    coords = x_flat.reshape(resolution, n)
    full = np.vstack([xA, coords, xB])
    grad = np.zeros_like(coords)
    for i in range(resolution):
        zi = full[i+1]
        zi_prev = full[i]
        zi_next = full[i+2]
        grad[i] = K * (2*(zi - zi_prev) + 2*(zi - zi_next))
    return grad.ravel()

# Constraint: phi(z_k) = 0 for each intermediate node
def constraints_flat(x_flat):
    coords = x_flat.reshape(resolution, n)
    result = np.zeros(resolution)
    for i in range(resolution):
        result[i] = phi_single(coords[i])
    return result

x0 = x0_slerp.ravel()

t0 = time.perf_counter()
# Try trust-constr
try:
    nlc = NonlinearConstraint(constraints_flat, 0, 0, jac=2-point)
    result = minimize(energy_flat, x0, method=trust-constr,
                      jac=energy_grad, constraints=[nlc],
                      options={gtol: 1e-6, xtol: 1e-8, maxiter: 1000, disp: True})
    elapsed = time.perf_counter() - t0
    geodesic = np.vstack([xA, result.x.reshape(resolution, n), xB])
    pe = compute_path_energy(geodesic, K)
    maxc = np.max(np.abs([phi_single(geodesic[i]) for i in range(len(geodesic))]))
    print("Trust-constr result:")
    print("  Energy: {:.6f}".format(pe))
    print("  Time:   {:.4f}s".format(elapsed))
    print("  Max constraint: {:.2e}".format(maxc))
    print("  Success:", result.success)
    print("  Message:", result.message)
except Exception as e:
    elapsed = time.perf_counter() - t0
    print("Trust-constr FAILED after {:.2f}s: {}".format(elapsed, e))
