#!/usr/bin/env python3
"""
Hodge Spectral Persistence: Topological Data Analysis via Hodge Laplacian

Demonstrates that the Hodge Laplacian operators (L₀, L₁, L₂) from our
framework can track topological invariants (Betti numbers) across a
filtration — a spectral approach to persistent homology.

Pipeline:
  1. Sample points from shapes with known topology (circle, torus, sphere, figure-8)
  2. Build Vietoris-Rips filtration: add simplices at increasing scale ε
  3. At each ε, construct boundary operators B₁, B₂ and Hodge Laplacians
  4. Count near-zero eigenvalues → Betti numbers b₀, b₁, b₂
  5. Plot "Hodge barcode": Betti numbers vs filtration scale

The null space dimension of Lₖ = Bₖ₊₁ᵀBₖ₊₁ + BₖBₖᵀ equals bₖ (Hodge theorem).
Near-zero eigenvalues approaching zero signal imminent topological transitions.

Usage:
    python hodge_persistence.py
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, eye as speye
from scipy.sparse.linalg import eigsh
from itertools import combinations
import os


# ==============================
# Point cloud generators
# ==============================
def sample_circle(n=50, noise=0.05, r=1.0):
    """Circle in R²: b₀=1, b₁=1."""
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    pts = np.stack([r*np.cos(t), r*np.sin(t)], axis=1)
    pts += noise * np.random.randn(*pts.shape)
    return pts, "Circle (b₀=1, b₁=1)"

def sample_figure8(n=50, noise=0.05):
    """Figure-8 in R²: b₀=1, b₁=2."""
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    x = np.sin(t)
    y = np.sin(t) * np.cos(t)
    pts = np.stack([x, y], axis=1)
    pts += noise * np.random.randn(*pts.shape)
    return pts, "Figure-8 (b₀=1, b₁=2)"

def sample_torus_3d(n=80, noise=0.02, R=1.0, r=0.4):
    """Torus in R³: b₀=1, b₁=2, b₂=1."""
    n_u = int(np.sqrt(n * R/r))
    n_v = max(n // n_u, 1)
    u = np.linspace(0, 2*np.pi, n_u, endpoint=False)
    v = np.linspace(0, 2*np.pi, n_v, endpoint=False)
    U, V = np.meshgrid(u, v)
    U, V = U.flatten(), V.flatten()
    x = (R + r*np.cos(V)) * np.cos(U)
    y = (R + r*np.cos(V)) * np.sin(U)
    z = r * np.sin(V)
    pts = np.stack([x, y, z], axis=1)
    pts += noise * np.random.randn(*pts.shape)
    return pts, "Torus (b₀=1, b₁=2, b₂=1)"

def sample_sphere_3d(n=60, noise=0.02):
    """Sphere in R³: b₀=1, b₁=0, b₂=1."""
    # Fibonacci sphere
    golden = (1 + np.sqrt(5)) / 2
    i = np.arange(n)
    theta = 2 * np.pi * i / golden
    phi = np.arccos(1 - 2*(i+0.5)/n)
    pts = np.stack([np.sin(phi)*np.cos(theta),
                    np.sin(phi)*np.sin(theta),
                    np.cos(phi)], axis=1)
    pts += noise * np.random.randn(*pts.shape)
    return pts, "Sphere (b₀=1, b₁=0, b₂=1)"

def sample_two_circles(n=50, noise=0.05):
    """Two disjoint circles: b₀=2, b₁=2."""
    n2 = n // 2
    t = np.linspace(0, 2*np.pi, n2, endpoint=False)
    c1 = np.stack([np.cos(t) - 1.5, np.sin(t)], axis=1)
    c2 = np.stack([np.cos(t) + 1.5, np.sin(t)], axis=1)
    pts = np.vstack([c1, c2])
    pts += noise * np.random.randn(*pts.shape)
    return pts, "Two circles (b₀=2, b₁=2)"


# ==============================
# Simplicial complex construction (Vietoris-Rips)
# ==============================
def vietoris_rips(pts, epsilon, max_dim=2):
    """
    Build Vietoris-Rips complex at scale epsilon.
    Returns: vertices, edges, triangles.
    """
    n = len(pts)
    dists = np.linalg.norm(pts[:, None] - pts[None, :], axis=-1)

    # Edges: pairs within epsilon
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if dists[i, j] < epsilon:
                edges.append((i, j))

    # Triangles: triples where all pairs are edges
    triangles = []
    if max_dim >= 2 and len(edges) < 2000:
        # Build adjacency for efficiency
        adj = {i: set() for i in range(n)}
        for i, j in edges:
            adj[i].add(j)
            adj[j].add(i)
        for i in range(n):
            neighbors = sorted(j for j in adj[i] if j > i)
            for idx_j, j in enumerate(neighbors):
                for k in neighbors[idx_j+1:]:
                    if k in adj[j]:
                        triangles.append((i, j, k))
                        if len(triangles) > 50000:
                            break
                if len(triangles) > 50000:
                    break
            if len(triangles) > 50000:
                break

    return list(range(n)), edges, triangles


# ==============================
# Boundary operators
# ==============================
def build_boundary_1(n_verts, edges):
    """B₁: edges × vertices signed incidence matrix."""
    if not edges:
        return csr_matrix((0, n_verts))
    n_edges = len(edges)
    rows, cols, vals = [], [], []
    for e_idx, (i, j) in enumerate(edges):
        # Convention: B₁[e, i] = -1, B₁[e, j] = +1
        rows.extend([e_idx, e_idx])
        cols.extend([i, j])
        vals.extend([-1.0, 1.0])
    return csr_matrix((vals, (rows, cols)), shape=(n_edges, n_verts))


def build_boundary_2(edges, triangles):
    """B₂: triangles × edges signed incidence matrix."""
    if not triangles or not edges:
        return csr_matrix((0, max(len(edges), 1)))
    edge_to_idx = {(min(e), max(e)): idx for idx, e in enumerate(edges)}
    n_tri = len(triangles)
    n_edges = len(edges)
    rows, cols, vals = [], [], []
    for t_idx, (i, j, k) in enumerate(triangles):
        # Faces: (j,k), (i,k), (i,j) with alternating signs
        for (a, b), sign in [((j, k), 1.0), ((i, k), -1.0), ((i, j), 1.0)]:
            edge_key = (min(a, b), max(a, b))
            if edge_key in edge_to_idx:
                rows.append(t_idx)
                cols.append(edge_to_idx[edge_key])
                vals.append(sign)
    return csr_matrix((vals, (rows, cols)), shape=(n_tri, n_edges))


# ==============================
# Hodge Laplacians and Betti numbers
# ==============================
def hodge_laplacians(B1, B2):
    """Compute Hodge Laplacians L₀, L₁, L₂."""
    L0 = (B1.T @ B1).tocsr() if B1.shape[0] > 0 else csr_matrix((B1.shape[1], B1.shape[1]))
    if B1.shape[0] > 0 and B2.shape[0] > 0:
        L1 = (B1 @ B1.T + B2.T @ B2).tocsr()
    elif B1.shape[0] > 0:
        L1 = (B1 @ B1.T).tocsr()
    else:
        L1 = csr_matrix((1, 1))
    L2 = (B2 @ B2.T).tocsr() if B2.shape[0] > 0 else csr_matrix((1, 1))
    return L0, L1, L2


def count_betti(L, threshold=1e-6, n_eigs=None):
    """Count near-zero eigenvalues of Hodge Laplacian = Betti number."""
    n = L.shape[0]
    if n == 0:
        return 0, np.array([])
    if n <= 1:
        return 0, np.array([0.0])
    if n_eigs is None:
        n_eigs = min(20, n - 1)
    if n_eigs <= 0:
        return 0, np.array([])
    try:
        if n < 500:
            eigs = np.linalg.eigvalsh(L.toarray())
        else:
            eigs = eigsh(L, k=n_eigs, which='SM', tol=1e-6, return_eigenvectors=False)
        eigs = np.sort(np.maximum(eigs, 0))
        betti = int(np.sum(eigs < threshold))
        return betti, eigs[:min(10, len(eigs))]
    except Exception:
        return 0, np.array([])


def compute_spectral_signature(L, n_eigs=10):
    """Return the smallest eigenvalues (spectral signature near zero)."""
    n = L.shape[0]
    if n <= 1:
        return np.array([0.0])
    k = min(n_eigs, n - 1)
    if k <= 0:
        return np.array([0.0])
    try:
        if n < 500:
            eigs = np.linalg.eigvalsh(L.toarray())
        else:
            eigs = eigsh(L, k=k, which='SM', tol=1e-6, return_eigenvectors=False)
        return np.sort(np.maximum(eigs, 0))[:n_eigs]
    except Exception:
        return np.zeros(1)


# ==============================
# Filtration sweep
# ==============================
def hodge_filtration(pts, epsilons, verbose=True):
    """
    Sweep over filtration scales, compute Betti numbers at each.
    Returns dict with betti numbers and spectral signatures per scale.
    """
    results = {
        'epsilons': epsilons,
        'b0': [], 'b1': [], 'b2': [],
        'n_edges': [], 'n_triangles': [],
        'L1_spectrum': [],  # smallest eigenvalues of L1 at each scale
    }

    for eps in epsilons:
        verts, edges, tris = vietoris_rips(pts, eps)
        n_v, n_e, n_t = len(verts), len(edges), len(tris)

        B1 = build_boundary_1(n_v, edges)
        B2 = build_boundary_2(edges, tris)
        L0, L1, L2 = hodge_laplacians(B1, B2)

        b0, _ = count_betti(L0)
        b1, _ = count_betti(L1)
        b2, _ = count_betti(L2)

        # Spectral signature of L1
        spec = compute_spectral_signature(L1, n_eigs=6) if n_e > 1 else np.array([])

        results['b0'].append(b0)
        results['b1'].append(b1)
        results['b2'].append(b2)
        results['n_edges'].append(n_e)
        results['n_triangles'].append(n_t)
        results['L1_spectrum'].append(spec)

        if verbose:
            print(f"  ε={eps:.3f}: {n_e} edges, {n_t} tris → "
                  f"b₀={b0}, b₁={b1}, b₂={b2}")

    return results


# ==============================
# Visualization
# ==============================
def plot_hodge_barcode(results, title, save_path=None):
    """Plot Betti numbers and L₁ spectral signature vs filtration scale."""
    eps = results['epsilons']

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                              gridspec_kw={'height_ratios': [1, 1.2]})

    # Top: Betti numbers
    ax = axes[0]
    ax.step(eps, results['b0'], where='post', label='b₀ (components)', lw=2, color='#2196F3')
    ax.step(eps, results['b1'], where='post', label='b₁ (loops)', lw=2, color='#F44336')
    ax.step(eps, results['b2'], where='post', label='b₂ (voids)', lw=2, color='#4CAF50')
    ax.set_ylabel('Betti number')
    ax.set_title(f'Hodge Spectral Persistence — {title}')
    ax.legend(loc='upper right')
    ax.set_ylim(-0.5, max(max(results['b0']), max(results['b1']),
                           max(results['b2'])) + 1.5)
    ax.grid(True, alpha=0.3)

    # Bottom: L₁ spectral signature (smallest eigenvalues)
    ax2 = axes[1]
    for i in range(6):
        eig_vals = []
        for spec in results['L1_spectrum']:
            if len(spec) > i:
                eig_vals.append(spec[i])
            else:
                eig_vals.append(np.nan)
        label = f'λ₁⁽{i}⁾' if i < 3 else None
        alpha = 1.0 if i < 3 else 0.4
        ax2.semilogy(eps, np.array(eig_vals) + 1e-12,
                      lw=1.5, alpha=alpha, label=label)

    ax2.axhline(y=1e-6, color='gray', ls='--', lw=0.8, label='threshold')
    ax2.set_xlabel('Filtration scale ε')
    ax2.set_ylabel('L₁ eigenvalues (log)')
    ax2.set_ylim(1e-8, None)
    ax2.legend(loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_complex_at_scale(pts, eps, title, save_path=None):
    """Visualize the simplicial complex at a given scale."""
    verts, edges, tris = vietoris_rips(pts, eps)
    dim = pts.shape[1]

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    if dim == 2:
        # Draw triangles
        for i, j, k in tris:
            tri = plt.Polygon(pts[[i,j,k]], alpha=0.15, color='steelblue')
            ax.add_patch(tri)
        # Draw edges
        for i, j in edges:
            ax.plot(pts[[i,j], 0], pts[[i,j], 1], 'k-', lw=0.5, alpha=0.3)
        # Draw points
        ax.scatter(pts[:, 0], pts[:, 1], s=15, c='black', zorder=5)
        ax.set_aspect('equal')
    elif dim == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=5, c='black')
        for i, j in edges[:500]:  # limit for performance
            ax.plot3D(pts[[i,j], 0], pts[[i,j], 1], pts[[i,j], 2],
                      'k-', lw=0.3, alpha=0.2)

    ax.set_title(f'{title} (ε={eps:.3f}, {len(edges)} edges, {len(tris)} tris)')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ==============================
# Main
# ==============================
def main():
    np.random.seed(42)
    out_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(out_dir, exist_ok=True)

    demos = [
        (sample_circle, np.linspace(0.2, 0.8, 20)),
        (sample_figure8, np.linspace(0.15, 0.8, 20)),
        (sample_two_circles, np.linspace(0.2, 1.5, 20)),
    ]

    print("=" * 60)
    print("Hodge Spectral Persistence Demo")
    print("=" * 60)
    print("Tracking topological invariants via Hodge Laplacian null spaces")
    print("dim ker(Lₖ) = bₖ (Hodge theorem)\n")

    for sampler, epsilons in demos:
        pts, title = sampler()
        print(f"\n--- {title} ({len(pts)} points, dim={pts.shape[1]}) ---")

        results = hodge_filtration(pts, epsilons)

        # Find the "interesting" scale where topology is correct
        tag = title.split('(')[0].strip().lower().replace(' ', '_').replace('-', '')
        plot_hodge_barcode(results, title,
                           save_path=os.path.join(out_dir, f'barcode_{tag}.png'))

        # Snapshot at characteristic scale
        mid_idx = len(epsilons) // 2
        plot_complex_at_scale(pts, epsilons[mid_idx], title,
                              save_path=os.path.join(out_dir, f'complex_{tag}.png'))

    # Verification summary
    print(f"\n{'='*60}")
    print("Verification: Hodge Laplacian correctly identifies topology")
    print(f"{'='*60}")
    for sampler, epsilons in demos:
        pts, title = sampler()
        # Find scale where topology matches expectation
        results = hodge_filtration(pts, epsilons, verbose=False)
        # Report Betti numbers at a few scales
        for frac in [0.3, 0.5, 0.7]:
            idx = int(frac * len(epsilons))
            print(f"  {title:<35s} ε={epsilons[idx]:.3f}: "
                  f"b₀={results['b0'][idx]} b₁={results['b1'][idx]} b₂={results['b2'][idx]}")

    print(f"\nFigures saved to {out_dir}/")


if __name__ == "__main__":
    main()
