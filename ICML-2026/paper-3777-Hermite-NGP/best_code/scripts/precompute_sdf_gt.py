"""
Precompute SDF and Gradient Ground Truth for meshes.

For each mesh in the mesh directory, computes:
- 256^3 SDF values
- 256^3 x 3 SDF gradient values (computed numerically from SDF)

Meshes are normalized to [0.1, 0.9]^3 domain.

Usage:
    python precompute_sdf_gt.py
    python precompute_sdf_gt.py --resolution 128  # Lower resolution for testing
    python precompute_sdf_gt.py --mesh bunny.ply  # Single mesh only

Output:
    For each mesh, creates {meshname}_sdf_gt.pt containing:
    - sdf: [R, R, R] SDF values
    - grad_x, grad_y, grad_z: [R, R, R] gradient components
    - grid_x, grid_y, grid_z: [R] coordinate arrays
    - mesh_vertices, mesh_faces: normalized mesh data
    - metadata: normalization info
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import torch
import numpy as np
import time

try:
    import kaolin
    from kaolin.ops.mesh import check_sign, index_vertices_by_faces
    from kaolin.metrics.trianglemesh import point_to_mesh_distance
    KAOLIN_AVAILABLE = True
except ImportError:
    KAOLIN_AVAILABLE = False
    print("Warning: Kaolin not available. Install with: pip install kaolin")

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    print("Warning: trimesh not available. Install with: pip install trimesh")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def normalize_mesh(vertices, domain_min=0.1, domain_max=0.9):
    """
    Normalize mesh vertices to [domain_min, domain_max]^3.

    Args:
        vertices: numpy array [V, 3]
        domain_min: minimum coordinate
        domain_max: maximum coordinate

    Returns:
        normalized_vertices: numpy array [V, 3]
        normalization_info: dict with center, scale for inverse transform
    """
    verts = np.array(vertices, dtype=np.float32)

    # Compute bounding box
    verts_min = verts.min(axis=0)
    verts_max = verts.max(axis=0)
    center = (verts_min + verts_max) / 2
    extent = (verts_max - verts_min).max()

    # Normalize to [0, 1] then scale to [domain_min, domain_max]
    # with 0.9 factor to keep mesh away from boundary
    scale = (domain_max - domain_min) / extent * 0.9
    verts_normalized = (verts - center) * scale + 0.5

    normalization_info = {
        'center': center,
        'extent': extent,
        'scale': scale,
        'domain_min': domain_min,
        'domain_max': domain_max,
    }

    return verts_normalized, normalization_info


def compute_sdf_kaolin(points, vertices, faces, face_vertices):
    """
    Compute SDF using Kaolin (CUDA accelerated).

    Args:
        points: [N, 3] query points
        vertices: [1, V, 3] mesh vertices (batched)
        faces: [F, 3] face indices
        face_vertices: [1, F, 3, 3] face vertices

    Returns:
        sdf: [N] signed distance values
    """
    pts = points.unsqueeze(0)  # [1, N, 3]

    # Get inside/outside sign
    signs = check_sign(vertices, faces, pts)  # [1, N], True = inside

    # Get unsigned squared distance
    dist_sq, _, _ = point_to_mesh_distance(pts, face_vertices)  # [1, N]

    # Take square root
    dist = torch.sqrt(dist_sq + 1e-10)

    # Signed distance: negative inside, positive outside
    sdf = torch.where(signs.squeeze(0), -dist.squeeze(0), dist.squeeze(0))

    return sdf


def compute_gradient_fd(sdf_volume, grid_spacing):
    """
    Compute SDF gradient using central finite differences.

    Args:
        sdf_volume: [R, R, R] SDF values
        grid_spacing: spacing between grid points (dx = dy = dz)

    Returns:
        grad_x, grad_y, grad_z: [R, R, R] gradient components
    """
    R = sdf_volume.shape[0]

    # Initialize gradient arrays
    grad_x = torch.zeros_like(sdf_volume)
    grad_y = torch.zeros_like(sdf_volume)
    grad_z = torch.zeros_like(sdf_volume)

    # Central differences for interior points
    # d(sdf)/dx = (sdf[i+1] - sdf[i-1]) / (2 * dx)
    grad_x[1:-1, :, :] = (sdf_volume[2:, :, :] - sdf_volume[:-2, :, :]) / (2 * grid_spacing)
    grad_y[:, 1:-1, :] = (sdf_volume[:, 2:, :] - sdf_volume[:, :-2, :]) / (2 * grid_spacing)
    grad_z[:, :, 1:-1] = (sdf_volume[:, :, 2:] - sdf_volume[:, :, :-2]) / (2 * grid_spacing)

    # Forward/backward differences for boundary points
    # x boundaries
    grad_x[0, :, :] = (sdf_volume[1, :, :] - sdf_volume[0, :, :]) / grid_spacing
    grad_x[-1, :, :] = (sdf_volume[-1, :, :] - sdf_volume[-2, :, :]) / grid_spacing

    # y boundaries
    grad_y[:, 0, :] = (sdf_volume[:, 1, :] - sdf_volume[:, 0, :]) / grid_spacing
    grad_y[:, -1, :] = (sdf_volume[:, -1, :] - sdf_volume[:, -2, :]) / grid_spacing

    # z boundaries
    grad_z[:, :, 0] = (sdf_volume[:, :, 1] - sdf_volume[:, :, 0]) / grid_spacing
    grad_z[:, :, -1] = (sdf_volume[:, :, -1] - sdf_volume[:, :, -2]) / grid_spacing

    return grad_x, grad_y, grad_z


def precompute_sdf_for_mesh(mesh_path, output_dir, resolution=256, domain_min=0.1, domain_max=0.9):
    """
    Precompute SDF and gradient ground truth for a single mesh.

    Args:
        mesh_path: path to mesh file
        output_dir: directory to save output
        resolution: grid resolution (default 256)
        domain_min, domain_max: domain bounds

    Returns:
        output_path: path to saved file
    """
    mesh_name = os.path.splitext(os.path.basename(mesh_path))[0]
    print(f"\n{'='*60}")
    print(f"Processing: {mesh_name}")
    print(f"{'='*60}")

    # Load mesh
    print(f"  Loading mesh from {mesh_path}...")
    mesh = trimesh.load(mesh_path, force='mesh')
    print(f"    Vertices: {len(mesh.vertices)}")
    print(f"    Faces: {len(mesh.faces)}")

    # Normalize mesh
    print(f"  Normalizing to [{domain_min}, {domain_max}]^3...")
    verts_normalized, norm_info = normalize_mesh(mesh.vertices, domain_min, domain_max)

    # Convert to torch tensors
    vertices = torch.tensor(verts_normalized, dtype=torch.float32, device=device).unsqueeze(0)  # [1, V, 3]
    faces = torch.tensor(mesh.faces, dtype=torch.long, device=device)  # [F, 3]
    face_vertices = index_vertices_by_faces(vertices, faces)  # [1, F, 3, 3]

    # Create 3D grid
    print(f"  Creating {resolution}^3 grid...")
    grid_1d = torch.linspace(domain_min, domain_max, resolution, device=device)
    grid_spacing = (domain_max - domain_min) / (resolution - 1)

    # Initialize SDF volume
    sdf_volume = torch.zeros(resolution, resolution, resolution, device=device)

    # Compute SDF slice by slice to avoid OOM
    print(f"  Computing SDF (slice by slice)...")
    t0 = time.time()

    batch_size = resolution * resolution  # One z-slice at a time

    for iz in range(resolution):
        if iz % 32 == 0:
            print(f"    z-slice {iz}/{resolution}...")

        # Create grid for this z-slice
        X, Y = torch.meshgrid(grid_1d, grid_1d, indexing='ij')
        pts = torch.zeros(batch_size, 3, device=device)
        pts[:, 0] = X.flatten()
        pts[:, 1] = Y.flatten()
        pts[:, 2] = grid_1d[iz]

        # Compute SDF
        sdf_slice = compute_sdf_kaolin(pts, vertices, faces, face_vertices)
        sdf_volume[:, :, iz] = sdf_slice.reshape(resolution, resolution)

    sdf_time = time.time() - t0
    print(f"    SDF computation: {sdf_time:.1f}s")
    print(f"    SDF range: [{sdf_volume.min().item():.4f}, {sdf_volume.max().item():.4f}]")

    # Compute gradient using finite differences
    print(f"  Computing gradient (finite differences)...")
    t0 = time.time()
    grad_x, grad_y, grad_z = compute_gradient_fd(sdf_volume, grid_spacing)
    grad_time = time.time() - t0
    print(f"    Gradient computation: {grad_time:.1f}s")

    # Compute gradient magnitude for verification
    grad_norm = torch.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
    print(f"    |grad| range: [{grad_norm.min().item():.4f}, {grad_norm.max().item():.4f}]")
    print(f"    |grad| mean: {grad_norm.mean().item():.4f} (should be ~1 for SDF)")

    # Save results
    output_path = os.path.join(output_dir, f'{mesh_name}_sdf_gt.pt')

    save_dict = {
        # SDF and gradient volumes
        'sdf': sdf_volume.cpu(),
        'grad_x': grad_x.cpu(),
        'grad_y': grad_y.cpu(),
        'grad_z': grad_z.cpu(),
        # Grid coordinates
        'grid_x': grid_1d.cpu(),
        'grid_y': grid_1d.cpu(),
        'grid_z': grid_1d.cpu(),
        'resolution': resolution,
        'grid_spacing': grid_spacing,
        # Mesh data (normalized)
        'mesh_vertices': torch.tensor(verts_normalized, dtype=torch.float32),
        'mesh_faces': torch.tensor(mesh.faces, dtype=torch.long),
        # Normalization info
        'norm_center': torch.tensor(norm_info['center'], dtype=torch.float32),
        'norm_scale': norm_info['scale'],
        'domain_min': domain_min,
        'domain_max': domain_max,
        # Original mesh path
        'original_mesh_path': mesh_path,
    }

    torch.save(save_dict, output_path)
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Saved: {output_path} ({file_size:.1f} MB)")

    return output_path


def main():
    parser = argparse.ArgumentParser(description='Precompute SDF and gradient ground truth for meshes')
    parser.add_argument('--mesh-dir', type=str, default=None, help='Directory containing mesh files')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory')
    parser.add_argument('--resolution', type=int, default=256, help='Grid resolution (default 256)')
    parser.add_argument('--mesh', type=str, default=None, help='Process single mesh only')
    parser.add_argument('--domain-min', type=float, default=0.1, help='Domain minimum')
    parser.add_argument('--domain-max', type=float, default=0.9, help='Domain maximum')
    args = parser.parse_args()

    if not KAOLIN_AVAILABLE:
        print("ERROR: Kaolin is required. Install with: pip install kaolin")
        return

    if not TRIMESH_AVAILABLE:
        print("ERROR: trimesh is required. Install with: pip install trimesh")
        return

    # Default paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mesh_dir = args.mesh_dir or os.path.join(script_dir, 'mesh')
    output_dir = args.output_dir or mesh_dir

    print("=" * 70)
    print("SDF Ground Truth Precomputation")
    print("=" * 70)
    print(f"Mesh directory: {mesh_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Resolution: {args.resolution}^3")
    print(f"Domain: [{args.domain_min}, {args.domain_max}]^3")
    print(f"Total grid points: {args.resolution**3:,}")
    print(f"Total gradient values: {3 * args.resolution**3:,}")
    print(f"Device: {device}")

    os.makedirs(output_dir, exist_ok=True)

    # Find mesh files
    if args.mesh:
        # Single mesh
        mesh_path = os.path.join(mesh_dir, args.mesh) if not os.path.isabs(args.mesh) else args.mesh
        if not os.path.exists(mesh_path):
            print(f"ERROR: Mesh not found: {mesh_path}")
            return
        mesh_files = [os.path.basename(mesh_path)]
        mesh_dir = os.path.dirname(mesh_path)
    else:
        # All meshes in directory
        mesh_files = [f for f in os.listdir(mesh_dir)
                     if f.endswith(('.obj', '.ply', '.stl'))]

    print(f"\nFound {len(mesh_files)} mesh files:")
    for f in mesh_files:
        print(f"  - {f}")

    # Process each mesh
    results = []
    total_start = time.time()

    for mesh_file in mesh_files:
        mesh_path = os.path.join(mesh_dir, mesh_file)

        try:
            output_path = precompute_sdf_for_mesh(
                mesh_path, output_dir,
                resolution=args.resolution,
                domain_min=args.domain_min,
                domain_max=args.domain_max
            )
            results.append((mesh_file, output_path, 'SUCCESS'))
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append((mesh_file, None, f'FAILED: {e}'))

    total_time = time.time() - total_start

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total time: {total_time:.1f}s")
    print(f"Processed: {sum(1 for r in results if r[2] == 'SUCCESS')} / {len(results)} meshes")

    for mesh_file, output_path, status in results:
        if status == 'SUCCESS':
            file_size = os.path.getsize(output_path) / (1024 * 1024)
            print(f"  [OK] {mesh_file} -> {os.path.basename(output_path)} ({file_size:.1f} MB)")
        else:
            print(f"  [FAIL] {mesh_file}: {status}")


if __name__ == '__main__':
    main()
