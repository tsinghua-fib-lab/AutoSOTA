"""
Quick start example: learn Darcy flow on an ellipsoid.
"""
import numpy as np
from hodge_spectral import HodgeOperator
from hodge_spectral.data.generate_ellipsoid_aero import generate_ellipsoid_aero

# Generate data (or load your own)
print("Generating data...")
generate_ellipsoid_aero('./example_data', n_samples=1000)

import pickle
with open('./example_data/ellipsoid_aero_dataset.pkl', 'rb') as f:
    data = pickle.load(f)

points, faces = data['points'], data['faces']
X, Y = data['X_data'], data['Y_data']

# Split
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# ===== From mesh =====
print("\n--- Mesh input ---")
model = HodgeOperator.from_mesh(points, faces, task="0to1", k=64)
model.fit(X_train, Y_train, epochs=50)
metrics = model.evaluate(X_test, Y_test)
print(f"Mesh:  Rel_L2 = {metrics['relative_l2']:.4f}")

# ===== From point cloud (same data, auto-triangulates) =====
print("\n--- Point cloud input ---")
noisy_points = points + np.random.randn(*points.shape) * 0.005
model_pc = HodgeOperator.from_pointcloud(noisy_points, task="0to1", k=64)
model_pc.fit(X_train, Y_train, epochs=50)
metrics_pc = model_pc.evaluate(X_test, Y_test)
print(f"PC:    Rel_L2 = {metrics_pc['relative_l2']:.4f}")

# ===== From graph =====
print("\n--- Graph input ---")
model_g = HodgeOperator.from_graph(data['edge_index'], len(points), points, task="0to1", k=64)
model_g.fit(X_train, Y_train, epochs=50)
metrics_g = model_g.evaluate(X_test, Y_test)
print(f"Graph: Rel_L2 = {metrics_g['relative_l2']:.4f}")

print(f"\nSpread: {max(metrics['relative_l2'], metrics_pc['relative_l2'], metrics_g['relative_l2']) - min(metrics['relative_l2'], metrics_pc['relative_l2'], metrics_g['relative_l2']):.4f}")
