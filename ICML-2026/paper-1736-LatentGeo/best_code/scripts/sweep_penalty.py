#!/usr/bin/env python3
"""Quick sweep of mu and alpha penalty parameters."""
import time, sys, numpy as np
sys.path.insert(0, "/repo/src")
from latentgeodesics import GeodesicSolverNumpy

R, r = 0.8, 0.2
def phi(coord):
    rho = np.sqrt(coord[:, 0]**2 + coord[:, 1]**2)
    return ((rho - R)**2 + coord[:, 2]**2 - r**2).reshape(1, -1)
def dphi(coord):
    rho = np.sqrt(coord[:, 0]**2 + coord[:, 1]**2)
    safe_rho = np.where(rho == 0, np.finfo(coord.dtype).eps, rho)
    grad = np.column_stack((
        2 * coord[:, 0] * (rho - R) / safe_rho,
        2 * coord[:, 1] * (rho - R) / safe_rho,
        2 * coord[:, 2]))
    return grad.reshape(1, -1, 3)
def torus_point(u, v):
    x = (R + r * np.cos(v)) * np.cos(u)
    y = (R + r * np.cos(v)) * np.sin(u)
    z = r * np.sin(v)
    return np.array([x, y, z], dtype=np.float64)
def compute_path_energy(geodesic, K):
    return K * float(np.sum((geodesic[1:] - geodesic[:-1]) ** 2))
def compute_torus_slerp_init(uA, vA, uB, vB, resolution):
    t = np.linspace(0.0, 1.0, resolution + 2)[1:-1]
    us = uA + t * (uB - uA)
    vs = vA + t * (vB - vA)
    x = (R + r * np.cos(vs)) * np.cos(us)
    y = (R + r * np.cos(vs)) * np.sin(us)
    z = r * np.sin(vs)
    return np.column_stack([x, y, z])

configs = [
    (10, 10), (10, 50), (10, 100),
    (50, 10), (50, 50), (50, 100),
    (100, 10), (100, 50), (100, 100),
    (200, 10), (200, 50), (200, 100),
    (500, 10), (500, 100),
]
uA, vA = 0.0, 0.0
uB, vB = 2.7658, 1.1597
resolution = 48
xA = torus_point(uA, vA)
xB = torus_point(uB, vB)
x0 = compute_torus_slerp_init(uA, vA, uB, vB, resolution)
K = resolution + 1

print("  mu  alpha     Energy     Time   Status")
print("-" * 45)

for mu, alpha in configs:
    try:
        solver = GeodesicSolverNumpy(3, phi, dphi)
        t0 = time.perf_counter()
        geodesic = solver.AugLagrangeMinimize(
            resolution=resolution, xA=xA, xB=xB, x0=x0,
            mu=mu, alpha=alpha, disp=False)
        elapsed = time.perf_counter() - t0
        pe = compute_path_energy(geodesic, K)
        status = "OK" if pe <= 4.78893 else "REGRESS"
        print("{:4d} {:5d} {:10.6f} {:8.4f}s {:>8s}".format(mu, alpha, pe, elapsed, status))
    except Exception as e:
        print("{:4d} {:5d} {:>10s} {:>8s} {:>8s}".format(mu, alpha, "FAIL", "N/A", str(e)[:20]))
