"""
Compute the ground-truth Laplace solution on a 3D mesh-domain for use as the
Poisson 3D evaluation target.

PDE:    Laplacian(u) = 0   in   [0,1]^3 \\ mesh
BC:     u = 1   on the mesh surface
        u = 0   on the outer cube faces

The mesh is normalized to [0.1, 0.9]^3 (same convention as
examples/poisson3d_bunny.py). Voxels inside the mesh are left as NaN, BC
voxels carry the prescribed value, exterior voxels are solved with a 6-point
finite-difference Laplacian and scipy sparse CG.

Output:  <output_dir>/<mesh_name>_gt_volume_<resolution>.npy
         shape (R, R, R), float32, NaN inside the mesh.

Usage:
    python scripts/compute_poisson_gt.py \\
        --mesh data/meshes/bunny.ply \\
        --resolution 256 \\
        --output_dir data/meshes
"""

import argparse
import os
import sys
import time

import numpy as np
import trimesh
from scipy import sparse
from scipy.sparse.linalg import cg


def load_and_normalize_mesh(mesh_path):
    mesh = trimesh.load(mesh_path, force="mesh")
    v = mesh.vertices.copy()
    v -= (v.min(axis=0) + v.max(axis=0)) / 2
    v /= (v.max(axis=0) - v.min(axis=0)).max()  # [-0.5, 0.5]
    v = v + 0.5                                  # [0, 1]
    v = v * 0.8 + 0.1                            # [0.1, 0.9]
    return v, mesh.faces.astype(np.int64)


def compute_sdf_grid(vertices, faces, resolution):
    try:
        from pysdf import SDF
    except ImportError:
        sys.exit("ERROR: pysdf is required. Install with: pip install pysdf")
    sdf_func = SDF(vertices.astype(np.float32), faces.astype(np.int32))
    lin = np.linspace(0, 1, resolution)
    gx, gy, gz = np.meshgrid(lin, lin, lin, indexing="ij")
    q = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1).astype(np.float32)
    # pysdf: positive inside; flip sign so positive = outside.
    return (-sdf_func(q)).reshape(resolution, resolution, resolution).astype(np.float32)


def solve_laplace_domain_bc(vertices, faces, resolution, mesh_bc=1.0, domain_bc=0.0,
                            cg_tol=1e-10, cg_maxiter=10000):
    n = resolution
    h = 1.0 / (n - 1)
    surface_band = h * 1.5

    print(f"[gt-solver] resolution={n}, h={h:.6f}, mesh_bc={mesh_bc}, domain_bc={domain_bc}")

    print("[gt-solver] computing SDF on grid...")
    t0 = time.time()
    sdf = compute_sdf_grid(vertices, faces, n).ravel()
    print(f"  SDF range [{sdf.min():.4f}, {sdf.max():.4f}]  ({time.time()-t0:.1f}s)")

    is_interior   = sdf < -surface_band
    is_surface    = np.abs(sdf) <= surface_band
    is_exterior   = sdf > surface_band
    is_surface_bc = is_surface & ~is_interior

    I, J, K = np.meshgrid(range(n), range(n), range(n), indexing="ij")
    is_domain_bc = (I.ravel() == 0) | (I.ravel() == n-1) | \
                   (J.ravel() == 0) | (J.ravel() == n-1) | \
                   (K.ravel() == 0) | (K.ravel() == n-1)

    is_solve = is_exterior & ~is_domain_bc
    is_bc    = is_surface_bc | is_domain_bc

    bc_values = np.zeros(n**3)
    bc_values[is_surface_bc] = mesh_bc
    bc_values[is_domain_bc]  = domain_bc

    solve_idx = np.where(is_solve)[0]
    g2s = np.full(n**3, -1, dtype=np.int32)
    g2s[solve_idx] = np.arange(len(solve_idx))
    N = len(solve_idx)
    print(f"[gt-solver] unknowns={N:,}  BC_mesh={is_surface_bc.sum():,}  "
          f"BC_domain={is_domain_bc.sum():,}  interior={is_interior.sum():,}")

    print("[gt-solver] assembling FD system...")
    t0 = time.time()
    rows, cols, data = [], [], []
    b = np.zeros(N)
    offsets = [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]
    for local, gidx in enumerate(solve_idx):
        i, j, k = gidx // (n*n), (gidx // n) % n, gidx % n
        diag = 6.0
        rhs = 0.0
        for di, dj, dk in offsets:
            ni, nj, nk = i+di, j+dj, k+dk
            if 0 <= ni < n and 0 <= nj < n and 0 <= nk < n:
                ngidx = ni*n*n + nj*n + nk
                ns = g2s[ngidx]
                if ns >= 0:
                    rows.append(local); cols.append(ns); data.append(-1.0)
                elif is_bc[ngidx]:
                    rhs += bc_values[ngidx]
                else:  # neighbor is inside the mesh -> Neumann (drop the term)
                    diag -= 1.0
            else:
                diag -= 1.0
        rows.append(local); cols.append(local); data.append(diag)
        b[local] = rhs
    A = sparse.csr_matrix((data, (rows, cols)), shape=(N, N))
    print(f"  matrix {A.shape}, nnz {A.nnz:,}  ({time.time()-t0:.1f}s)")

    print("[gt-solver] solving with CG...")
    t0 = time.time()
    it = [0]
    def cb(xk):
        it[0] += 1
        if it[0] % 500 == 0:
            print(f"    iter {it[0]:5d}  residual {np.linalg.norm(A @ xk - b):.3e}")
    x0 = np.full(N, (mesh_bc + domain_bc) / 2)
    u_solve, info = cg(A, b, x0=x0, rtol=cg_tol, maxiter=cg_maxiter, callback=cb)
    print(f"  done: info={info}, {it[0]} iters, "
          f"residual={np.linalg.norm(A @ u_solve - b):.3e}  ({time.time()-t0:.1f}s)")

    u_full = np.full(n**3, np.nan)
    u_full[solve_idx] = u_solve
    u_full[np.where(is_bc)[0]] = bc_values[is_bc]
    return u_full.reshape(n, n, n).astype(np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mesh", required=True, help="Mesh file (e.g., data/meshes/bunny.ply)")
    p.add_argument("--resolution", type=int, default=256, help="Grid resolution (default: 256)")
    p.add_argument("--output_dir", default="data/meshes",
                   help="Where to write <mesh_name>_gt_volume_<R>.npy")
    p.add_argument("--mesh_bc", type=float, default=1.0)
    p.add_argument("--domain_bc", type=float, default=0.0)
    args = p.parse_args()

    if not os.path.exists(args.mesh):
        sys.exit(f"ERROR: mesh not found at {args.mesh}")
    os.makedirs(args.output_dir, exist_ok=True)

    v, f = load_and_normalize_mesh(args.mesh)
    print(f"[gt-solver] mesh: {len(v)} verts, {len(f)} faces, normalized to "
          f"[{v.min():.3f}, {v.max():.3f}]^3")

    u = solve_laplace_domain_bc(v, f, args.resolution, args.mesh_bc, args.domain_bc)

    name = os.path.splitext(os.path.basename(args.mesh))[0]
    out = os.path.join(args.output_dir, f"{name}_gt_volume_{args.resolution}.npy")
    np.save(out, u)
    print(f"\nSaved -> {out}")
    print(f"  shape   {u.shape}")
    print(f"  finite  {np.isfinite(u).sum():,} / {u.size:,}")
    print(f"  range   [{np.nanmin(u):.4f}, {np.nanmax(u):.4f}]")


if __name__ == "__main__":
    main()
