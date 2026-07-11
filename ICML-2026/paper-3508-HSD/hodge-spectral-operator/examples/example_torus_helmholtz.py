"""
Example: Helmholtz Vortex Flow on Torus (genus-1)
==================================================
Task: 0-form → 1-form (vortex source → tangential velocity field)

Physics:
  Input:  multi-scale source f (scalar 0-form on torus)
  Solve:  (Δ + κ²)ψ = f (Helmholtz equation, κ=0.5 preserves mid-high frequency)
  Output: v = n × ∇ψ (tangential velocity, 1-form proxy)

Features:
  - Genus-1 topology: two independent homology cycles (toroidal + poloidal)
  - Vortex pairs with opposite-sign Gaussians
  - Toroidal modulation: sin(nθ) creates azimuthal structure
  - Helmholtz (not Poisson): κ>0 preserves richer frequency content
"""
from hodge_spectral import HodgeOperator
import numpy as np
import pickle
import os
import time
from sklearn.model_selection import train_test_split


def generate_torus_helmholtz(output_dir, n_samples=1000, kappa=0.5, seed=42):
    """Generate Helmholtz vortex flow on torus. Fast: ~1ms/sample."""
    import pyvista as pv
    from scipy.sparse import csr_matrix, eye as speye
    from scipy.sparse.linalg import factorized

    os.makedirs(output_dir, exist_ok=True)

    mesh = pv.ParametricTorus(1.0, 0.4, u_res=60, v_res=30)
    mesh = pv.PolyData(mesh).triangulate().clean()
    pts = mesh.points.copy()
    fcs = mesh.faces.reshape((-1, 4))[:, 1:].astype(np.int64)
    pts -= pts.mean(0)
    pts /= (np.max(np.linalg.norm(pts, axis=1)) + 1e-9)
    n = len(pts)

    normals = np.zeros((n, 3))
    for f in fcs:
        i, j, k = f
        nm = np.cross(pts[j] - pts[i], pts[k] - pts[i])
        normals[i] += nm; normals[j] += nm; normals[k] += nm
    normals /= (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-9)

    rows, cols, vals = [], [], []
    for f in fcs:
        i, j, k = f
        vi, vj, vk = pts[i], pts[j], pts[k]
        eij, ejk, eki = vj - vi, vk - vj, vi - vk
        area = np.linalg.norm(np.cross(eij, -eki)) / 2
        if area < 1e-12: continue
        def cot(a, b):
            c = np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12), -0.999, 0.999)
            return c / (np.sqrt(1 - c ** 2) + 1e-12)
        for a, b, cv in [(j, k, cot(eij, -eki)), (k, i, cot(ejk, -eij)), (i, j, cot(eki, -ejk))]:
            w = cv / 2
            rows.extend([a, b, a, b]); cols.extend([b, a, a, b]); vals.extend([w, w, -w, -w])
    L = csr_matrix((vals, (rows, cols)), shape=(n, n))
    solve = factorized((L + kappa ** 2 * speye(n)).tocsc())

    fi, fj, fk = fcs[:, 0], fcs[:, 1], fcs[:, 2]
    e1, e2 = pts[fj] - pts[fi], pts[fk] - pts[fi]
    fn = np.cross(e1, e2)
    fa2 = np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12
    fn /= fa2
    cr_kj = np.cross(fn, pts[fk] - pts[fj])
    cr_ik = np.cross(fn, pts[fi] - pts[fk])
    cr_ji = np.cross(fn, pts[fj] - pts[fi])
    nc = np.zeros(n)
    np.add.at(nc, fi, 1); np.add.at(nc, fj, 1); np.add.at(nc, fk, 1)
    nc[nc == 0] = 1

    edges = set()
    for f in fcs:
        for i in range(3): edges.add(tuple(sorted([f[i], f[(i + 1) % 3]])))
    edge_index = np.array(list(edges), dtype=np.int64)

    rng = np.random.RandomState(seed)
    theta = np.arctan2(pts[:, 1], pts[:, 0])
    X_data = np.zeros((n_samples, n), dtype=np.float32)
    Y_data = np.zeros((n_samples, n, 3), dtype=np.float32)

    t0 = time.time()
    for s in range(n_samples):
        f = np.zeros(n)
        for _ in range(rng.randint(3, 8)):
            c = pts[rng.randint(0, n)]; sig = rng.uniform(0.05, 0.25); st = rng.uniform(-2, 2)
            f += st * np.exp(-np.linalg.norm(pts - c, axis=1) ** 2 / (2 * sig ** 2))
        for _ in range(rng.randint(3, 8)):
            c1 = pts[rng.randint(0, n)]; c2 = c1 + rng.randn(3) * 0.04
            sig = rng.uniform(0.02, 0.06); st = rng.uniform(1, 4)
            f += st * np.exp(-np.linalg.norm(pts - c1, axis=1) ** 2 / (2 * sig ** 2))
            f -= st * np.exp(-np.linalg.norm(pts - c2, axis=1) ** 2 / (2 * sig ** 2))
        f += rng.uniform(0.3, 1.5) * np.sin(rng.randint(1, 4) * theta + rng.uniform(0, 2 * np.pi))
        f -= f.mean()
        X_data[s] = f
        psi = solve(f)
        pi, pj, pk = psi[fi], psi[fj], psi[fk]
        gt = (pi[:, None] * cr_kj + pj[:, None] * cr_ik + pk[:, None] * cr_ji) / fa2
        g = np.zeros((n, 3))
        for d in range(3):
            np.add.at(g[:, d], fi, gt[:, d])
            np.add.at(g[:, d], fj, gt[:, d])
            np.add.at(g[:, d], fk, gt[:, d])
        g /= nc[:, None]
        v = np.cross(normals, g)
        v -= np.sum(v * normals, axis=1, keepdims=True) * normals
        Y_data[s] = v.astype(np.float32)
    print(f"Generated {n_samples} samples in {time.time() - t0:.1f}s")

    dataset = {
        'points': pts, 'faces': fcs, 'edge_index': edge_index,
        'X_data': X_data, 'Y_data': Y_data,
        'n_nodes': n, 'n_samples': n_samples,
        'task': 'helmholtz_vortex_flow', 'topology': 'torus_genus1',
        'physics': f'Helmholtz(kappa={kappa}): (L+k^2)psi=f, v=n x grad(psi)',
    }
    path = os.path.join(output_dir, 'torus_helmholtz_dataset.pkl')
    with open(path, 'wb') as f:
        pickle.dump(dataset, f)
    return dataset


def run(data_dir='./example_data/torus_helmholtz', n_samples=1000, epochs=80, k=64):
    # --- Generate or load data ---
    pkl_path = os.path.join(data_dir, 'torus_helmholtz_dataset.pkl')
    if not os.path.exists(pkl_path):
        print("Generating torus Helmholtz dataset...")
        generate_torus_helmholtz(data_dir, n_samples=n_samples)
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    points, faces = data['points'], data['faces']
    X, Y = data['X_data'], data['Y_data']
    print(f"Dataset: {data['n_samples']} samples, {data['n_nodes']} nodes")
    print(f"Task: 0-form → 1-form ({data['physics']})")
    print(f"Topology: {data['topology']}")

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    results = {}
    for mode in ['mesh', 'pointcloud', 'graph']:
        print(f"\n{'='*50}")
        print(f"  Input mode: {mode}")
        print(f"{'='*50}")

        if mode == 'mesh':
            model = HodgeOperator.from_mesh(points, faces, task="0to1", k=k)
        elif mode == 'pointcloud':
            noisy = points + np.random.RandomState(123).randn(*points.shape) * 0.005
            model = HodgeOperator.from_pointcloud(noisy, task="0to1", k=k)
        elif mode == 'graph':
            model = HodgeOperator.from_graph(
                data['edge_index'], len(points), points, task="0to1", k=k)

        model.fit(X_train, Y_train, epochs=epochs, verbose=True)
        metrics = model.evaluate(X_test, Y_test)
        results[mode] = metrics

        print(f"\n  Relative L2:             {metrics['relative_l2']:.4f}")
        print(f"  Riemannian IP Fidelity:  {metrics['riemannian_ip_fidelity']:.4f}")

    # --- Summary ---
    print(f"\n{'='*50}")
    print(f"  Cross-Input Summary (Torus, genus-1)")
    print(f"{'='*50}")
    rels = [r['relative_l2'] for r in results.values()]
    for mode, r in results.items():
        print(f"  {mode:<12s}  Rel_L2 = {r['relative_l2']:.4f}")
    print(f"  Spread: {max(rels)-min(rels):.4f}")

    return results


if __name__ == "__main__":
    run()
