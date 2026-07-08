#!/usr/bin/env python3
"""Fine-tune endpoint search for path energy ~4.7889."""
import time
import numpy as np
from latentgeodesics import GeodesicSolverNumpy

R, r = 4.0/5.0, 0.2

def phi(coord):
    rho = np.sqrt(coord[:, 0]**2 + coord[:, 1]**2)
    return ((rho - R)**2 + coord[:, 2]**2 - r**2).reshape(1, -1)

def dphi(coord):
    rho = np.sqrt(coord[:, 0]**2 + coord[:, 1]**2)
    safe_rho = np.where(rho == 0, np.finfo(coord.dtype).eps, rho)
    grad = np.column_stack((
        2 * coord[:, 0] * (rho - R) / safe_rho,
        2 * coord[:, 1] * (rho - R) / safe_rho,
        2 * coord[:, 2],
    ))
    return grad.reshape(1, -1, 3)

def torus_point(u, v):
    x = (R + r * np.cos(v)) * np.cos(u)
    y = (R + r * np.cos(v)) * np.sin(u)
    z = r * np.sin(v)
    return np.array([x, y, z], dtype=np.float64)

def compute_pe(geodesic, K_segments):
    energy = 0.0
    for k in range(1, len(geodesic)):
        energy += np.sum((geodesic[k] - geodesic[k-1]) ** 2)
    return K_segments * energy

# Based on previous scan: (0,0)->(2.8,1.2) gave 4.8555
# Need slightly lower energy -> slightly closer endpoints
# Try a grid around the best
candidates = [
    # Pure major circle geodesics (same v): L ≈ delta_u * (R+r*cos(v))
    # For v=0: radius=1.0, sqrt(4.7889) ≈ 2.1883
    (0.0, 0.0, 2.1883, 0.0),   # pure major circle estimate
    (0.0, 0.0, 2.185, 0.0),
    (0.0, 0.0, 2.19, 0.0),
    (0.0, 0.0, 2.18, 0.0),
    # Mixed geodesics
    (0.0, 0.0, 2.75, 1.15),
    (0.0, 0.0, 2.78, 1.15),
    (0.0, 0.0, 2.76, 1.18),
    (0.0, 0.0, 2.77, 1.17),
    (0.0, 0.0, 2.76, 1.10),
    (0.0, 0.0, 2.75, 1.12),
    (0.0, 0.0, 2.74, 1.14),
    # Try v=0 on endpoint A and B for simpler geodesic
    (0.0, 0.0, 2.5, 0.0),
    # Symmetric endpoints
    (0.0, 0.0, 2.2, 0.5),
    (0.0, 0.0, 2.2, 0.3),
    (0.0, 0.0, 2.15, 0.4),
    # Figure 15 looks like it goes about halfway through major+minor
    (0.0, 0.0, 2.7, 1.2),
    (0.0, 0.0, 2.72, 1.21),
    (0.0, 0.0, 2.73, 1.22),
]

print("Fine-tuning endpoints for path_energy = 4.7889...")
print()
best = None
for ua, va, ub, vb in candidates:
    try:
        xA = torus_point(ua, va)
        xB = torus_point(ub, vb)
        solver = GeodesicSolverNumpy(3, phi, dphi)
        t0 = time.perf_counter()
        geodesic = solver.AugLagrangeMinimize(resolution=48, xA=xA, xB=xB, mu=100, alpha=100, disp=False)
        elapsed = time.perf_counter() - t0
        pe = compute_pe(geodesic, 49)
        maxc = float(np.max(np.abs(phi(geodesic))))
        diff = abs(pe - 4.7889)
        mark = "***" if diff < 0.001 else ("**" if diff < 0.01 else ("*" if diff < 0.05 else ""))
        print(f"  ({ua:.4f},{va:.4f})->({ub:.4f},{vb:.4f})  pe={pe:.6f}  t={elapsed:.4f}s  |d|={diff:.6f}  {mark}")
        if best is None or diff < best[0]:
            best = (diff, pe, elapsed, maxc, ua, va, ub, vb)
    except Exception as e:
        print(f"  ({ua},{va})->({ub},{vb})  FAILED: {e}")

print()
print(f"Best: ({best[4]},{best[5]})->({best[6]},{best[7]}) pe={best[1]:.6f} t={best[2]:.4f}s")
