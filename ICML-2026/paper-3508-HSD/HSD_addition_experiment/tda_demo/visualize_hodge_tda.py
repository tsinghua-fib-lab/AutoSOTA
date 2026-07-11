#!/usr/bin/env python3
"""
Visualize how HSD's Hodge spectral operators detect topology.

For each shape:
  1. Simplicial complex with triangles filled
  2. L₁ harmonic eigenvectors as edge flow (arrows on edges showing the
     non-contractible cycle — this IS the topological feature)
  3. L₀ Fiedler vector as node coloring (level sets reveal connected structure)
  4. Spectral signature: smallest eigenvalues of L₁ (near-zero = topology)

Usage:
    python tda_demo/visualize_hodge_tda.py
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch
from scipy.sparse import csr_matrix
import os


def sample_circle(n=50, noise=0.05):
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    pts = np.stack([np.cos(t), np.sin(t)], axis=1)
    pts += noise * np.random.randn(*pts.shape)
    return pts, "Circle", {"b0": 1, "b1": 1}

def sample_figure8(n=50, noise=0.05):
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    pts = np.stack([np.sin(t), np.sin(t)*np.cos(t)], axis=1)
    pts += noise * np.random.randn(*pts.shape)
    return pts, "Figure-8", {"b0": 1, "b1": 2}

def sample_two_circles(n=50, noise=0.05):
    n2 = n // 2
    t = np.linspace(0, 2*np.pi, n2, endpoint=False)
    c1 = np.stack([np.cos(t) - 1.5, np.sin(t)], axis=1)
    c2 = np.stack([np.cos(t) + 1.5, np.sin(t)], axis=1)
    pts = np.vstack([c1, c2])
    pts += noise * np.random.randn(*pts.shape)
    return pts, "Two Circles", {"b0": 2, "b1": 2}


def build_vr(pts, epsilon):
    n = len(pts)
    dists = np.linalg.norm(pts[:, None] - pts[None, :], axis=-1)
    edges = [(i, j) for i in range(n) for j in range(i+1, n) if dists[i, j] < epsilon]
    triangles = []
    if len(edges) < 2000:
        adj = {i: set() for i in range(n)}
        for i, j in edges:
            adj[i].add(j); adj[j].add(i)
        for i in range(n):
            nb = sorted(j for j in adj[i] if j > i)
            for a, j in enumerate(nb):
                for k in nb[a+1:]:
                    if k in adj[j]:
                        triangles.append((i, j, k))
    return edges, triangles


def build_B1(n, edges):
    ne = len(edges)
    r, c, v = [], [], []
    for idx, (i, j) in enumerate(edges):
        r.extend([idx, idx]); c.extend([i, j]); v.extend([-1., 1.])
    return csr_matrix((v, (r, c)), shape=(ne, n))


def build_B2(edges, tris):
    if not tris:
        return csr_matrix((0, len(edges)))
    e2i = {(min(e), max(e)): idx for idx, e in enumerate(edges)}
    ne = len(edges)
    r, c, v = [], [], []
    for idx, (i, j, k) in enumerate(tris):
        for (a, b), s in [((j,k), 1.), ((i,k), -1.), ((i,j), 1.)]:
            key = (min(a,b), max(a,b))
            if key in e2i:
                r.append(idx); c.append(e2i[key]); v.append(s)
    return csr_matrix((v, (r, c)), shape=(len(tris), ne))


def hodge_analysis(pts, epsilon):
    n = len(pts)
    edges, tris = build_vr(pts, epsilon)
    B1 = build_B1(n, edges)
    B2 = build_B2(edges, tris)

    L0 = (B1.T @ B1).toarray().astype(np.float64)
    L1 = (B1 @ B1.T).toarray().astype(np.float64)
    if len(tris) > 0:
        L1 += (B2.T @ B2).toarray().astype(np.float64)

    eigs0, vecs0 = np.linalg.eigh(L0)
    eigs1, vecs1 = np.linalg.eigh(L1)

    return {
        'edges': edges, 'tris': tris, 'B1': B1,
        'eigs0': eigs0, 'vecs0': vecs0,
        'eigs1': eigs1, 'vecs1': vecs1,
        'b0': int(np.sum(eigs0 < 1e-6)),
        'b1': int(np.sum(eigs1 < 1e-6)),
    }


def plot_tda_panel(pts, name, betti, epsilon, out_path):
    """4-panel TDA visualization for one shape."""
    info = hodge_analysis(pts, epsilon)
    edges, tris = info['edges'], info['tris']

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))

    # === Panel 1: Simplicial complex ===
    ax = axes[0, 0]
    for i, j, k in tris:
        tri = plt.Polygon(pts[[i, j, k]], alpha=0.12, color='steelblue', edgecolor='none')
        ax.add_patch(tri)
    for i, j in edges:
        ax.plot(pts[[i, j], 0], pts[[i, j], 1], 'k-', lw=0.4, alpha=0.3)
    ax.scatter(pts[:, 0], pts[:, 1], s=20, c='black', zorder=5)
    ax.set_title(f'Simplicial Complex (ε={epsilon:.2f})\n'
                 f'{len(edges)} edges, {len(tris)} triangles',
                 fontsize=10, fontweight='bold', loc='left')
    ax.set_aspect('equal')
    ax.set_xlim(pts[:, 0].min() - 0.3, pts[:, 0].max() + 0.3)
    ax.set_ylim(pts[:, 1].min() - 0.3, pts[:, 1].max() + 0.3)

    # === Panel 2: Fiedler vector (L₀ second eigenvector) — level set on nodes ===
    ax = axes[0, 1]
    fiedler_idx = info['b0']  # first non-zero eigenvector
    if fiedler_idx < len(info['eigs0']):
        fiedler = info['vecs0'][:, fiedler_idx]
    else:
        fiedler = np.zeros(len(pts))
    for i, j in edges:
        ax.plot(pts[[i, j], 0], pts[[i, j], 1], 'k-', lw=0.3, alpha=0.15)
    sc = ax.scatter(pts[:, 0], pts[:, 1], s=30, c=fiedler, cmap='RdBu_r',
                    edgecolors='k', linewidth=0.3, zorder=5)
    # Level set contours via tricontour
    from matplotlib.tri import Triangulation
    if len(tris) > 0:
        tri_obj = Triangulation(pts[:, 0], pts[:, 1], triangles=np.array(tris))
        ax.tricontour(tri_obj, fiedler, levels=8, colors='k', linewidths=0.8, alpha=0.6)
    plt.colorbar(sc, ax=ax, shrink=0.7, label='Fiedler value')
    ax.set_title(f'L₀ Fiedler Vector (λ={info["eigs0"][fiedler_idx]:.4f})\n'
                 f'Level sets reveal b₀={info["b0"]} components',
                 fontsize=10, fontweight='bold', loc='left')
    ax.set_aspect('equal')

    # === Panel 3: Harmonic 1-form (L₁ null eigenvector) — edge flow ===
    ax = axes[1, 0]
    for i, j, k in tris:
        tri = plt.Polygon(pts[[i, j, k]], alpha=0.08, color='steelblue', edgecolor='none')
        ax.add_patch(tri)

    if info['b1'] > 0:
        # Show first harmonic 1-form as edge arrows
        harm1 = info['vecs1'][:, 0]  # first harmonic eigenvector (on edges)
        harm1 = harm1 / (np.max(np.abs(harm1)) + 1e-10)

        # Draw all edges lightly
        for i, j in edges:
            ax.plot(pts[[i, j], 0], pts[[i, j], 1], 'k-', lw=0.2, alpha=0.1)

        # Draw harmonic flow as colored arrows
        for e_idx, (i, j) in enumerate(edges):
            mid = (pts[i] + pts[j]) / 2
            direction = pts[j] - pts[i]
            direction = direction / (np.linalg.norm(direction) + 1e-10)
            strength = harm1[e_idx]
            if abs(strength) > 0.05:
                color = plt.cm.coolwarm(0.5 + 0.5 * strength)
                ax.annotate('', xy=mid + direction * 0.06 * np.sign(strength),
                           xytext=mid - direction * 0.06 * np.sign(strength),
                           arrowprops=dict(arrowstyle='->', color=color,
                                          lw=1.5 * abs(strength) + 0.3))
        ax.set_title(f'Harmonic 1-Form (b₁={info["b1"]} loops)\n'
                     f'Arrows show non-contractible circulation',
                     fontsize=10, fontweight='bold', loc='left')
    else:
        for i, j in edges:
            ax.plot(pts[[i, j], 0], pts[[i, j], 1], 'k-', lw=0.3, alpha=0.2)
        ax.scatter(pts[:, 0], pts[:, 1], s=15, c='black', zorder=5)
        ax.set_title(f'No Harmonic 1-Forms (b₁=0)\nNo non-contractible loops',
                     fontsize=10, fontweight='bold', loc='left')
    ax.set_aspect('equal')
    ax.set_xlim(pts[:, 0].min() - 0.3, pts[:, 0].max() + 0.3)
    ax.set_ylim(pts[:, 1].min() - 0.3, pts[:, 1].max() + 0.3)

    # === Panel 4: Spectral signature ===
    ax = axes[1, 1]
    n_show = min(12, len(info['eigs1']))
    colors = ['#F44336' if e < 1e-6 else '#90CAF9' for e in info['eigs1'][:n_show]]
    bars = ax.bar(range(n_show), info['eigs1'][:n_show] + 1e-10, color=colors,
                  edgecolor=['#C62828' if e < 1e-6 else '#1565C0' for e in info['eigs1'][:n_show]])
    ax.axhline(y=1e-6, color='gray', ls='--', lw=1, label='Harmonic threshold')
    ax.set_yscale('log')
    ax.set_ylim(1e-8, info['eigs1'][min(n_show-1, len(info['eigs1'])-1)] * 3)
    ax.set_xlabel('Eigenvalue index')
    ax.set_ylabel('λ (L₁)')
    ax.set_title(f'L₁ Spectrum: {info["b1"]} near-zero = b₁\n'
                 f'Red bars = topological features',
                 fontsize=10, fontweight='bold', loc='left')
    ax.legend(fontsize=8)

    fig.suptitle(f'{name} — Hodge Spectral TDA (b₀={info["b0"]}, b₁={info["b1"]})',
                 fontsize=14, fontweight='bold', x=0.02, ha='left', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path}')


def main():
    np.random.seed(42)
    out_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(out_dir, exist_ok=True)

    print("Generating Hodge TDA visualizations\n")

    demos = [
        (sample_circle, 0.40),
        (sample_figure8, 0.35),
        (sample_two_circles, 0.55),
    ]

    for sampler, eps in demos:
        pts, name, betti = sampler()
        tag = name.lower().replace(' ', '_').replace('-', '')
        print(f"--- {name} (ε={eps}) ---")
        plot_tda_panel(pts, name, betti, eps,
                       os.path.join(out_dir, f'hodge_tda_{tag}.png'))


if __name__ == "__main__":
    main()
