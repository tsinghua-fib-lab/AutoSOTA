"""
Example: External Aerodynamics on Ellipsoid (genus-0)
=====================================================
Task: 0-form → 1-form (vorticity source → tangential velocity field)

Physics:
  Input:  vorticity ω (scalar 0-form on surface)
  Solve:  Δψ = ω (Poisson stream function)
  Output: v = n × ∇ψ + global coupling (tangential velocity, 1-form proxy)

Features:
  - Vortex pairs with separation
  - Global coupling: vorticity moments → background flow
  - Rich topological features: stagnation points, separation lines
"""
from hodge_spectral import HodgeOperator
from hodge_spectral.data.generate_ellipsoid_aero import generate_ellipsoid_aero
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split


def run(data_dir='./example_data/ellipsoid_aero', n_samples=1000, epochs=80, k=64):
    # --- Generate or load data ---
    pkl_path = os.path.join(data_dir, 'ellipsoid_aero_dataset.pkl')
    if not os.path.exists(pkl_path):
        print("Generating ellipsoid aerodynamics dataset...")
        generate_ellipsoid_aero(data_dir, n_samples=n_samples)
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    points, faces = data['points'], data['faces']
    X, Y = data['X_data'], data['Y_data']  # X: (B, N) scalar, Y: (B, N, 3) vector
    print(f"Dataset: {data['n_samples']} samples, {data['n_nodes']} nodes")
    print(f"Task: 0-form → 1-form ({data['physics']})")
    print(f"Topology: {data['topology']}")

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # --- Three input modes ---
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
        print(f"  MSE:                     {metrics['mse']:.6f}")

    # --- Summary ---
    print(f"\n{'='*50}")
    print(f"  Cross-Input Summary")
    print(f"{'='*50}")
    rels = [r['relative_l2'] for r in results.values()]
    for mode, r in results.items():
        print(f"  {mode:<12s}  Rel_L2 = {r['relative_l2']:.4f}")
    print(f"  Spread: {max(rels)-min(rels):.4f}")

    return results


if __name__ == "__main__":
    run()
