#!/usr/bin/env python3
"""Fine-tune endpoint search v4: narrow grid around (0,0)->(2.77,1.17)"""
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

# Narrow grid around (2.77, 1.17)
# pe=4.793678 at (2.77, 1.17), need slightly lower pe (4.7889)
# So decrease ub or vb slightly
candidates = []
for ub in [2.760, 2.762, 2.764, 2.766, 2.768, 2.770, 2.772]:
    for vb in [1.155, 1.160, 1.162, 1.165, 1.168, 1.170, 1.172]:
        candidates.append((ub, vb))

print(f"Testing {len(candidates)} endpoint pairs around (0,0)->(2.76-2.77, 1.155-1.172)...")
print()

best = None
for ub, vb in candidates:
    try:
        xA = torus_point(0.0, 0.0)
        xB = torus_point(ub, vb)
        solver = GeodesicSolverNumpy(3, phi, dphi)
        t0 = time.perf_counter()
        geodesic = solver.AugLagrangeMinimize(resolution=48, xA=xA, xB=xB, mu=100, alpha=100, disp=False)
        elapsed = time.perf_counter() - t0
        pe = compute_pe(geodesic, 49)
        diff = abs(pe - 4.7889)
        mark = "***" if diff < 0.0005 else ("**" if diff < 0.001 else ("*" if diff < 0.003 else ""))
        print(f"  ub={ub:.3f} vb={vb:.3f}  pe={pe:.6f}  |d|={diff:.6f}  {mark}")
        if best is None or diff < best[0]:
            best = (diff, pe, elapsed, ub, vb)
    except Exception as e:
        print(f"  ub={ub} vb={vb}  FAILED: {e}")

print()
if best:
    print(f"Best: (0,0)->({best[3]:.4f},{best[4]:.4f}) pe={best[1]:.6f} |diff|={best[0]:.6f} t={best[2]:.4f}s")
    print(f"Paper: pe=4.7889")
