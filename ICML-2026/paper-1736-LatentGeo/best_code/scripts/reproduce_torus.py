#!/usr/bin/env python3
"""
Reproduction script for the Torus geodesic experiment from Appendix C.1 of
"Geodesic Calculus on Implicitly Defined Latent Manifolds" (ICML 2026).

Settings:
  - Torus: zeta(z) = (sqrt(z1^2+z2^2) - R)^2 + z3^2 - r^2  with R=4/5, r=0.2
  - K = 49  (50 nodes)
  - Metric: Euclidean (WE)
  - Endpoints: torus_point(u=1, v=0) and torus_point(u=0, v=1)

Expected (paper):
  - Path Energy: 4.7889
  - Computation Time: 1.89 seconds
"""

import time
import numpy as np
from latentgeodesics import GeodesicSolverNumpy

# --- Torus definition (matches paper Appendix C.1 and notebook) ---
R = 4.0 / 5.0   # outer radius
r = 0.2         # inner radius


def phi(coord):
    """Implicit function zeta(z) for the torus. Returns shape (1, m)."""
    rho = np.sqrt(coord[:, 0] ** 2 + coord[:, 1] ** 2)
    return ((rho - R) ** 2 + coord[:, 2] ** 2 - r**2).reshape(1, -1)


def dphi(coord):
    """Gradient of phi. Returns shape (1, m, 3)."""
    rho = np.sqrt(coord[:, 0] ** 2 + coord[:, 1] ** 2)
    safe_rho = np.where(rho == 0, np.finfo(coord.dtype).eps, rho)
    grad = np.column_stack(
        (
            2 * coord[:, 0] * (rho - R) / safe_rho,
            2 * coord[:, 1] * (rho - R) / safe_rho,
            2 * coord[:, 2],
        )
    )
    return grad.reshape(1, -1, 3)


def main():
    # --- Endpoints (from the notebook) ---
    a = np.array([0, 1])   # v angles
    b = np.array([1, 0])   # u angles

    x = (R + r * np.cos(a)) * np.cos(b)
    y = (R + r * np.cos(a)) * np.sin(b)
    z = r * np.sin(a)
    xA, xB = np.column_stack((x, y, z))

    # K = 49 means 49 segments between 50 nodes (resolution = 48 interior points)
    K = 48  # number of interior (free) points, i.e. 49 segments

    print(f"Reproducing Torus geodesic (Appendix C.1)")
    print(f"  Endpoint A: {xA}")
    print(f"  Endpoint B: {xB}")
    print(f"  K = {K} (interior points) -> {K+1} segments, {K+2} total nodes")
    print(f"  Metric: Euclidean (WE)")
    print()

    # Verify endpoints lie on torus
    cons_A = phi(xA.reshape(1, -1))
    cons_B = phi(xB.reshape(1, -1))
    print(f"  Constraint at xA: {float(np.abs(cons_A)):.3e}")
    print(f"  Constraint at xB: {float(np.abs(cons_B)):.3e}")
    print()

    # --- Initialize solver ---
    n = 3  # ambient dimension
    solver = GeodesicSolverNumpy(n, phi, dphi)

    # --- Time the geodesic computation ---
    t0 = time.perf_counter()
    geodesic = solver.AugLagrangeMinimize(
        resolution=K,
        xA=xA,
        xB=xB,
        mu=100,
        alpha=100,      # higher alpha is safe for ground truth constraints
        disp=False,
    )
    t1 = time.perf_counter()
    elapsed = t1 - t0

    # --- Compute path energy ---
    # E^K = K_segments * sum_{k=1}^{K_segments} |z_k - z_{k-1}|^2
    K_segments = K + 1  # = 49
    energy = 0.0
    for k in range(1, len(geodesic)):
        energy += np.sum((geodesic[k] - geodesic[k-1]) ** 2)
    path_energy = K_segments * energy

    # --- Constraint check ---
    max_constraint = float(np.max(np.abs(phi(geodesic))))

    # --- Results ---
    print("=" * 50)
    print("REPRODUCTION RESULTS")
    print("=" * 50)
    print(f"  Path Energy:       {path_energy:.6f}")
    print(f"  Computation Time:  {elapsed:.4f} seconds")
    print(f"  Max Constraint:    {max_constraint:.3e}")
    print(f"  Geodesic shape:    {geodesic.shape}")
    print()

    print("PAPER VALUES (Appendix C.1)")
    print(f"  Path Energy:       4.7889")
    print(f"  Computation Time:  1.89 seconds")
    print()

    # --- Compare ---
    pe_diff = abs(path_energy - 4.7889)
    time_diff = abs(elapsed - 1.89)
    print(f"  |Path Energy - paper|:  {pe_diff:.6f}")
    print(f"  |Time - paper|:         {time_diff:.4f}s")
    print()

    # Check against rubric bounds
    print("RUBRIC BOUNDS:")
    print(f"  Path Energy CI:    [4.78887, 4.7892]")
    print(f"  Time CI:           [1.831, 2.48]")

    pe_in_ci = 4.78887 <= path_energy <= 4.7892
    time_in_ci = 1.831 <= elapsed <= 2.48
    print(f"  Path Energy in CI: {pe_in_ci}")
    print(f"  Time in CI:        {time_in_ci}")

    if pe_in_ci and time_in_ci:
        print("\n*** REPRODUCTION SUCCEEDED — both metrics in CI bounds ***")
    elif pe_in_ci:
        print("\n*** Path Energy reproduced within CI bounds ***")
    else:
        print("\n*** Path Energy outside CI bounds ***")


if __name__ == "__main__":
    main()
