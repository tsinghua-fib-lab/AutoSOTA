#!/usr/bin/env python3
"""
Reproduction script v2: Torus geodesic experiment from Appendix C.1.
Try endpoint pairs to find the one producing ~4.7889 path energy.

Paper: "As a toy model, we compute geodesics on the torus
M = {z = (z^1, z^2, z^3) in R^3 | 0 = zeta(z) := (sqrt((z^1)^2+(z^2)^2) - R)^2 + (z^3)^2 - r^2}"
R=4/5, r=0.2 (from notebook)

Paper reports: K=49, path_energy=4.7889, time=1.89s
"""

import time
import numpy as np
from latentgeodesics import GeodesicSolverNumpy

R = 4.0 / 5.0
r = 0.2


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
    """u = azimuthal, v = polar angle on torus"""
    x = (R + r * np.cos(v)) * np.cos(u)
    y = (R + r * np.cos(v)) * np.sin(u)
    z = r * np.sin(v)
    return np.array([x, y, z], dtype=np.float64)


def compute_path_energy(geodesic, K_segments):
    energy = 0.0
    for k in range(1, len(geodesic)):
        energy += np.sum((geodesic[k] - geodesic[k-1]) ** 2)
    return K_segments * energy


def run_one(uA, vA, uB, vB, resolution=48):
    xA = torus_point(uA, vA)
    xB = torus_point(uB, vB)
    solver = GeodesicSolverNumpy(3, phi, dphi)
    t0 = time.perf_counter()
    geodesic = solver.AugLagrangeMinimize(
        resolution=resolution, xA=xA, xB=xB,
        mu=100, alpha=100, disp=False,
    )
    elapsed = time.perf_counter() - t0
    K_segments = resolution + 1
    pe = compute_path_energy(geodesic, K_segments)
    max_cons = float(np.max(np.abs(phi(geodesic))))
    return pe, elapsed, max_cons, geodesic


# Try various endpoint pairs
# Paper path energy is 4.7889
# The endpoints should be on opposite sides of the torus for a longer geodesic

# Candidates - trying to get path energy around the paper's 4.7889
endpoint_pairs = [
    # Candidates from the notebook figures / common torus test points
    ("notebook_default", 1.0, 0.0, 0.0, 1.0),
    # Opposite points on the major circle - geodesic wraps around outer circle
    ("major_opposite", 0.0, 0.0, np.pi, 0.0),
    # Quadrant points
    ("quadrant", 0.0, 0.0, np.pi/2, np.pi/2),
    # Anitpodal-like
    ("diag_opp", 0.0, 0.0, 2.0, 2.0),
    ("diag_1", 0.0, 0.0, 1.5, 1.5),
    ("diag_2", 1.0, 0.0, 0.0, 2.0),
    ("diag_3", 0.5, 0.5, 2.5, 2.5),
    # Try longer arcs
    ("long_arc", 0.0, 0.0, 2.5, 0.0),
    ("long_arc2", 0.0, 0.0, 3.0, 1.0),
    ("long_arc3", 0.5, 0.0, 2.5, 1.5),
    ("long_arc4", 0.0, 0.0, 2.8, 1.2),
    ("long_arc5", 0.2, 0.3, 2.8, 1.5),
    ("long_arc6", 0.3, 0.2, 3.0, 1.2),
    # The ones from Figure 15 look like they might wrap about 1/3 of the way
    ("mid_arc1", 0.0, 0.0, 2.0, 0.0),
    ("mid_arc2", 0.0, 0.0, 1.8, 0.8),
    ("mid_arc3", 0.2, 0.0, 2.0, 1.0),
]

print("Scanning endpoint pairs...")
print(f"Target: path_energy = 4.7889")
print()

best = None
for label, ua, va, ub, vb in endpoint_pairs:
    try:
        pe, elapsed, max_cons, _ = run_one(ua, va, ub, vb)
        diff = abs(pe - 4.7889)
        print(f"  {label:20s}  pe={pe:.6f}  time={elapsed:.4f}s  constraint={max_cons:.2e}  |diff|={diff:.4f}")
        if best is None or diff < best[0]:
            best = (diff, label, pe, elapsed, max_cons, (ua, va, ub, vb))
    except Exception as e:
        print(f"  {label:20s}  FAILED: {e}")

print()
if best:
    print(f"Best match: {best[1]} (uA,vA)=({best[5][0]},{best[5][1]}) (uB,vB)=({best[5][2]},{best[5][3]})")
    print(f"  Path Energy: {best[2]:.6f} (paper: 4.7889)")
    print(f"  Time: {best[3]:.4f}s (paper: 1.89s)")
    print(f"  |diff|: {best[0]:.6f}")
