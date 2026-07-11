"""
Headless data generation - avoids PyVista Plotter (which needs X11),
but uses PyVista mesh operations (which don't need X11).
"""
import os
import time
import pickle
import warnings
import numpy as np
import torch
import pyvista as pv

pv.OFF_SCREEN = True
warnings.filterwarnings("ignore")

# Import the original GPU solver and config
from generate_data import Config as OrigConfig, GPUPoissonStreamFunctionGenerator

cfg = OrigConfig()
cfg.DATA_DIR_OUT = "./flux_field_data"


def load_mesh_headless(file_idx=0, target_nodes=3000):
    """Load mesh using PyVista operations (no Plotter/rendering needed)."""
    rx, ry, rz = cfg.ELLIPSOID_AXES
    print(f"[Mesh] Creating ellipsoid with PyVista...")
    print(f"       Target: {target_nodes} nodes, axes: ({rx}, {ry}, {rz})")

    # Check for VTK mesh files
    mesh = None
    if os.path.exists(cfg.DATA_DIR):
        files = sorted([f for f in os.listdir(cfg.DATA_DIR) if f.endswith('.vtk')])
        if len(files) > file_idx:
            fpath = os.path.join(cfg.DATA_DIR, files[file_idx])
            print(f"       Loading: {files[file_idx]}")
            try:
                mesh = pv.read(fpath)
            except Exception as e:
                print(f"       Failed: {e}")
                mesh = None

    if mesh is None:
        # Create uniform ellipsoid
        icosahedron = pv.Icosahedron()
        level = 0
        while (10 * 4**level + 2) < target_nodes:
            level += 1
        mesh = icosahedron.subdivide(level, subfilter='linear') if level > 0 else icosahedron
        pts = mesh.points.copy()
        radii = np.linalg.norm(pts, axis=1, keepdims=True)
        pts_sphere = pts / (radii + 1e-9)
        mesh.points = pts_sphere * np.array([rx, ry, rz])

    if mesh.n_points > target_nodes * 1.1:
        mesh = mesh.decimate(1.0 - target_nodes / mesh.n_points)

    mesh = mesh.triangulate().clean()
    print(f"       After triangulate+clean: {mesh.n_points} nodes, {mesh.n_cells} faces")

    if not mesh.is_all_triangles:
        mesh = mesh.triangulate()
    mesh = mesh.clean()

    if mesh.n_points > target_nodes:
        mesh = mesh.decimate(1.0 - target_nodes / mesh.n_points).clean().triangulate()

    mesh = mesh.compute_normals(
        cell_normals=True, point_normals=True,
        auto_orient_normals=True, consistent_normals=True
    )

    norm_points = mesh.points.copy()
    centroid = np.mean(norm_points, axis=0)
    norm_points = norm_points - centroid
    scale = np.max(np.linalg.norm(norm_points, axis=1))
    norm_points = norm_points / (scale + 1e-9)
    print(f"       Normalized to unit sphere, scale={scale:.4f}")

    raw_faces = mesh.faces
    faces = raw_faces.reshape((-1, 4))[:, 1:] if raw_faces.size % 4 == 0 else None
    if faces is None:
        mesh = mesh.triangulate().clean()
        faces = mesh.faces.reshape((-1, 4))[:, 1:]

    if 'Normals' in mesh.point_data:
        vertex_normals = mesh.point_data['Normals'].copy()
    else:
        vertex_normals = np.zeros_like(norm_points)
        for face in faces:
            v0, v1, v2 = norm_points[face]
            n = np.cross(v1 - v0, v2 - v0)
            vertex_normals[face] += n

    norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True) + 1e-9
    vertex_normals = vertex_normals / norms

    print(f"       Final: {len(norm_points)} nodes, {len(faces)} faces")

    return (
        norm_points.astype(np.float64),
        faces.astype(np.int32),
        vertex_normals.astype(np.float64),
        mesh
    )


def main():
    print("\n" + "="*70)
    print("HEADLESS GPU-ACCELERATED POISSON STREAM FUNCTION DATASET GENERATOR")
    print("(PyVista mesh + original GPU solver, no rendering)")
    print("="*70)

    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        print(f"[GPU] {torch.cuda.get_device_name()}")
        print(f"      Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Create mesh with PyVista operations (no X11 needed)
    pts, faces, normals, mesh_pv = load_mesh_headless(cfg.FILE_IDX, cfg.TARGET_NODES)

    os.makedirs(cfg.DATA_DIR_OUT, exist_ok=True)
    pickle_path = os.path.join(cfg.DATA_DIR_OUT, cfg.PICKLE_FILE)

    # Use original GPU solver
    generator = GPUPoissonStreamFunctionGenerator(pts, faces, normals)

    t_start = time.time()
    X_data, Y_data = generator.generate_dataset(cfg.N_SAMPLES, cfg.BATCH_SIZE)
    t_total = time.time() - t_start

    print(f"\n[Time] Total: {t_total:.2f}s ({cfg.N_SAMPLES/t_total:.1f} samples/s)")

    # Skip preview visualization (would need pv.Plotter which requires X11)
    print("\n[Preview] Skipping visualization (headless mode)")

    data = {
        'X_data': X_data,
        'Y_data': Y_data,
        'points': pts.astype(np.float32),
        'faces': faces,
        'normals': normals.astype(np.float32),
        'config': {
            'n_samples': len(X_data),
            'n_nodes': len(pts),
            'physics': 'poisson_stream_function_with_background_coupling',
            'field_type': 'divergence_free_velocity_with_global_flow',
            'method': 'laplace_beltrami_poisson_plus_polarization_moment',
            'interface': {
                'input': '0-form (vorticity scalar)',
                'output': '1-form (velocity vector, n x grad_psi + global_flow)'
            },
            'source_params': {
                'n_sources': cfg.N_SOURCES_RANGE,
                'sigma': cfg.SIGMA_RANGE,
                'strength': cfg.STRENGTH_RANGE,
                'separation_prob': cfg.VORTEX_SEPARATION_PROB
            },
            'coupling_params': {
                'global_scale': cfg.GLOBAL_COUPLING_SCALE,
                'global_sharpness': cfg.GLOBAL_COUPLING_SHARPNESS
            }
        }
    }

    print(f"\n[Save] Writing to {pickle_path}...")
    with open(pickle_path, 'wb') as f:
        pickle.dump(data, f)

    file_size = os.path.getsize(pickle_path) / (1024 * 1024)
    print(f"[Done] Size: {file_size:.2f} MB")
    print(f"       X: {X_data.shape} (vorticity)")
    print(f"       Y: {Y_data.shape} (velocity with global flow)")

    print("\n[Verify] Reloading...")
    with open(pickle_path, 'rb') as f:
        loaded = pickle.load(f)
    print(f"         Loaded {loaded['config']['n_samples']} samples")
    print(f"         Physics: {loaded['config']['physics']}")
    print("\n[DONE] Data generation complete!")


if __name__ == "__main__":
    main()
