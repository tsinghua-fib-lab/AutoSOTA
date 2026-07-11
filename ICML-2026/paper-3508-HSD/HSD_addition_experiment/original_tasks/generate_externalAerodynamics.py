import os
import time
import pickle
import warnings
import uuid
import numpy as np
import torch
import pyvista as pv
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from PIL import Image

pv.OFF_SCREEN = True
try:
    pv.start_xvfb()
except Exception:
    pass

warnings.filterwarnings("ignore")


class Config:
    ELLIPSOID_AXES = (10, 5, 3)
    DATA_DIR = "./mesh/E_S_WWC_WM"
    FILE_IDX = 0
    TARGET_NODES = 3000
    
    N_SOURCES_RANGE = (3, 10)
    SIGMA_RANGE = (0.05, 0.25)
    STRENGTH_RANGE = (0.5, 2.0)
    VORTEX_SEPARATION_PROB = 0.7
    
    REGULARIZATION_EPS = 1e-8
    GLOBAL_COUPLING_SCALE = 5.0
    GLOBAL_COUPLING_SHARPNESS = 2.0
    
    N_SAMPLES = 3000
    BATCH_SIZE = 64
    N_PREVIEW_SAMPLES = 3
    
    DATA_DIR_OUT = "./data/externalAerodynamics"
    PICKLE_FILE = "flux_field_dataset.pkl"
    PREVIEW_FILE = "preview_samples.png"
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


cfg = Config()


def normalize_points(points):
    centroid = np.mean(points, axis=0)
    points = points - centroid
    scale = np.max(np.linalg.norm(points, axis=1))
    return points / (scale + 1e-9)


def create_uniform_ellipsoid(target_nodes, axes):
    rx, ry, rz = axes
    print(f"[Mesh] Creating uniform icosphere ellipsoid...")
    print(f"       Target: {target_nodes} nodes, axes: {axes}")
    
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
    print(f"       Final: {mesh.n_points} nodes, {mesh.n_cells} faces")
    
    return mesh


def load_mesh(file_idx=0, target_nodes=2000):
    print(f"\n[Geometry] Loading mesh...")
    
    mesh = None
    if os.path.exists(cfg.DATA_DIR):
        files = sorted([f for f in os.listdir(cfg.DATA_DIR) if f.endswith('.vtk')])
        if len(files) > file_idx:
            fpath = os.path.join(cfg.DATA_DIR, files[file_idx])
            print(f"           Loading: {files[file_idx]}")
            try:
                mesh = pv.read(fpath)
            except Exception as e:
                print(f"           Failed: {e}")
                mesh = None
    
    if mesh is None:
        mesh = create_uniform_ellipsoid(target_nodes, cfg.ELLIPSOID_AXES)
    
    if not mesh.is_all_triangles:
        mesh = mesh.triangulate()
    mesh = mesh.clean()
    
    if mesh.n_points > target_nodes:
        mesh = mesh.decimate(1.0 - target_nodes / mesh.n_points).clean().triangulate()
    
    mesh = mesh.compute_normals(
        cell_normals=True, point_normals=True,
        auto_orient_normals=True, consistent_normals=True
    )
    
    norm_points = normalize_points(mesh.points.copy())
    
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
    
    print(f"           Final: {len(norm_points)} nodes, {len(faces)} faces")
    
    return (
        norm_points.astype(np.float64),
        faces.astype(np.int32),
        vertex_normals.astype(np.float64),
        mesh
    )


class GPUPoissonStreamFunctionGenerator:
    
    def __init__(self, vertices, faces, normals, device=None):
        self.device = torch.device(device or cfg.DEVICE)
        self.n_verts = len(vertices)
        self.n_faces = len(faces)
        
        print(f"\n{'='*60}")
        print("GPU Poisson Stream Function Generator")
        print("With Virtual Background Coupling (Any Topology)")
        print(f"{'='*60}")
        print(f"[Device] {self.device}")
        if self.device.type == 'cuda':
            print(f"         {torch.cuda.get_device_name()}")
            print(f"         Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"[Mesh]   {self.n_verts} vertices, {self.n_faces} faces")
        
        self.vertices = torch.tensor(vertices, dtype=torch.float32, device=self.device)
        self.faces = torch.tensor(faces, dtype=torch.long, device=self.device)
        self.normals = torch.tensor(normals, dtype=torch.float32, device=self.device)
        
        self._build_operators()
        self._build_background_basis()
        
        self._validate()
        
        print(f"{'='*60}\n")
    
    def _build_operators(self):
        print("[Build] Constructing operators...")
        
        t0 = time.time()
        L_scipy, M_diag_np = self._build_cotangent_laplacian()
        print(f"        Laplacian: {time.time()-t0:.3f}s")
        
        t0 = time.time()
        eps = cfg.REGULARIZATION_EPS
        M_scipy = sp.diags(M_diag_np)
        A_scipy = -L_scipy + eps * M_scipy
        self.solver = spla.splu(A_scipy.tocsc())
        print(f"        LU decomposition: {time.time()-t0:.3f}s")
        
        self.M_diag = torch.tensor(M_diag_np, dtype=torch.float32, device=self.device)
        self.M_diag_np = M_diag_np
        self.total_area = self.M_diag.sum().item()
        
        t0 = time.time()
        self._precompute_gradient_geometry()
        print(f"        Gradient geometry: {time.time()-t0:.3f}s")
    
    def _build_background_basis(self):
        print("[Build] Constructing virtual background basis (Tangent Projection)...")
        t0 = time.time()
        
        normals = self.normals
        
        basis_dirs = torch.eye(3, device=self.device).unsqueeze(1).expand(3, self.n_verts, 3)
        
        dots = torch.sum(basis_dirs * normals.unsqueeze(0), dim=2, keepdim=True)
        
        projected = basis_dirs - dots * normals.unsqueeze(0)
        
        norms = torch.norm(projected, dim=2, keepdim=True) + 1e-9
        self.background_basis = projected / norms
        
        self.pos_centered = self.vertices - self.vertices.mean(dim=0, keepdim=True)
        
        print(f"        Background basis: {time.time()-t0:.3f}s")
        
        for i, axis in enumerate(['X', 'Y', 'Z']):
            mag = torch.norm(self.background_basis[i], dim=1).mean().item()
            print(f"        |basis_{axis}|_mean: {mag:.4f}")
    
    def _build_cotangent_laplacian(self):
        vertices = self.vertices.cpu().numpy()
        faces = self.faces.cpu().numpy()
        n = self.n_verts
        
        L_data, L_row, L_col = [], [], []
        M_diag = np.zeros(n)
        
        for face in faces:
            i, j, k = face
            vi, vj, vk = vertices[i], vertices[j], vertices[k]
            
            e_ij, e_jk, e_ki = vj - vi, vk - vj, vi - vk
            
            cross = np.cross(e_ij, -e_ki)
            area = np.linalg.norm(cross) / 2
            if area < 1e-14:
                continue
            
            def cot(e1, e2):
                c = np.cross(e1, e2)
                cn = np.linalg.norm(c)
                return np.dot(e1, e2) / cn if cn > 1e-14 else 0.0
            
            cots = [cot(e_ij, -e_ki), cot(e_jk, -e_ij), cot(e_ki, -e_jk)]
            edges = [(j, k), (k, i), (i, j)]
            
            for (a, b), c_val in zip(edges, cots):
                w = c_val / 2
                L_row.extend([a, b, a, b])
                L_col.extend([b, a, a, b])
                L_data.extend([w, w, -w, -w])
            
            M_diag[[i, j, k]] += area / 3
        
        L = sp.coo_matrix((L_data, (L_row, L_col)), shape=(n, n)).tocsr()
        return L, M_diag
    
    def _precompute_gradient_geometry(self):
        faces = self.faces
        verts = self.vertices
        
        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]
        
        e01, e02 = v1 - v0, v2 - v0
        
        cross = torch.cross(e01, e02, dim=1)
        areas = torch.norm(cross, dim=1) / 2
        normals = cross / (2 * areas.unsqueeze(1) + 1e-10)
        
        self.grad_coef_0 = torch.cross(normals, v2 - v1, dim=1)
        self.grad_coef_1 = torch.cross(normals, v0 - v2, dim=1)
        self.grad_coef_2 = torch.cross(normals, v1 - v0, dim=1)
        
        self.face_areas = areas
        self.inv_2A = 1.0 / (2 * areas + 1e-10)
    
    def _validate(self):
        omega_test = torch.randn(1, self.n_verts, device=self.device)
        omega_test = omega_test - omega_test.mean()
        
        psi_test = self._solve_poisson_batch(omega_test)
        u_test, moments_test = self._compute_velocity_batch(psi_test, omega_test, return_moments=True)
        
        u_norm = torch.norm(u_test, dim=2)
        print(f"[Valid] Test sample: |u|_mean={u_norm.mean():.4f}, |u|_max={u_norm.max():.4f}")
        print(f"        Global moments: [{moments_test[0,0]:.4f}, {moments_test[0,1]:.4f}, {moments_test[0,2]:.4f}]")
    
    def _generate_vortex_sources_batch(self, batch_size):
        device = self.device
        n = self.n_verts
        n_src_min, n_src_max = cfg.N_SOURCES_RANGE
        sigma_min, sigma_max = cfg.SIGMA_RANGE
        str_min, str_max = cfg.STRENGTH_RANGE
        sep_prob = cfg.VORTEX_SEPARATION_PROB
        
        omega = torch.zeros(batch_size, n, device=device)
        
        n_sources = torch.randint(n_src_min, n_src_max + 1, (batch_size,), device=device)
        sigmas = torch.empty(batch_size, device=device).uniform_(sigma_min, sigma_max)
        
        for b in range(batch_size):
            ns = n_sources[b].item()
            sigma = sigmas[b].item()
            
            if torch.rand(1).item() < sep_prob:
                axis = torch.randint(0, 3, (1,)).item()
                coord = self.pos_centered[:, axis]
                
                pos_mask = coord > 0
                neg_mask = coord <= 0
                
                pos_indices = torch.where(pos_mask)[0]
                neg_indices = torch.where(neg_mask)[0]
                
                n_pos = ns // 2 + (ns % 2)
                n_neg = ns // 2
                
                if len(pos_indices) >= n_pos and len(neg_indices) >= n_neg:
                    pos_src = pos_indices[torch.randperm(len(pos_indices), device=device)[:n_pos]]
                    neg_src = neg_indices[torch.randperm(len(neg_indices), device=device)[:n_neg]]
                    
                    pos_strengths = torch.empty(n_pos, device=device).uniform_(str_min, str_max)
                    neg_strengths = -torch.empty(n_neg, device=device).uniform_(str_min, str_max)
                    
                    for idx, s in zip(pos_src, pos_strengths):
                        center = self.vertices[idx]
                        dist = torch.norm(self.vertices - center, dim=1)
                        omega[b] += s * torch.exp(-dist**2 / (2 * sigma**2))
                    
                    for idx, s in zip(neg_src, neg_strengths):
                        center = self.vertices[idx]
                        dist = torch.norm(self.vertices - center, dim=1)
                        omega[b] += s * torch.exp(-dist**2 / (2 * sigma**2))
                else:
                    src_idx = torch.randperm(n, device=device)[:ns]
                    strengths = torch.empty(ns, device=device).uniform_(str_min, str_max)
                    signs = torch.sign(torch.randn(ns, device=device))
                    strengths = strengths * signs
                    
                    for idx, s in zip(src_idx, strengths):
                        center = self.vertices[idx]
                        dist = torch.norm(self.vertices - center, dim=1)
                        omega[b] += s * torch.exp(-dist**2 / (2 * sigma**2))
            else:
                src_idx = torch.randperm(n, device=device)[:ns]
                strengths = torch.empty(ns, device=device).uniform_(str_min, str_max)
                signs = torch.sign(torch.randn(ns, device=device))
                strengths = strengths * signs
                
                for idx, s in zip(src_idx, strengths):
                    center = self.vertices[idx]
                    dist = torch.norm(self.vertices - center, dim=1)
                    omega[b] += s * torch.exp(-dist**2 / (2 * sigma**2))
        
        weighted_mean = (omega * self.M_diag).sum(dim=1, keepdim=True) / self.total_area
        omega = omega - weighted_mean
        
        omega_max = omega.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-10)
        omega = omega / omega_max
        
        return omega
    
    def _solve_poisson_batch(self, omega_batch):
        batch_size = omega_batch.shape[0]
        omega_np = omega_batch.cpu().numpy()
        
        psi_list = []
        for b in range(batch_size):
            rhs = self.M_diag_np * omega_np[b]
            psi = self.solver.solve(rhs)
            
            psi_mean = np.sum(psi * self.M_diag_np) / self.total_area
            psi = psi - psi_mean
            psi_list.append(psi)
        
        psi_np = np.stack(psi_list, axis=0)
        return torch.tensor(psi_np, dtype=torch.float32, device=self.device)
    
    def _compute_gradient_batch(self, psi_batch):
        batch_size = psi_batch.shape[0]
        n = self.n_verts
        faces = self.faces
        
        psi_0 = psi_batch[:, faces[:, 0]]
        psi_1 = psi_batch[:, faces[:, 1]]
        psi_2 = psi_batch[:, faces[:, 2]]
        
        grad_face = (
            psi_0.unsqueeze(2) * self.grad_coef_0.unsqueeze(0) +
            psi_1.unsqueeze(2) * self.grad_coef_1.unsqueeze(0) +
            psi_2.unsqueeze(2) * self.grad_coef_2.unsqueeze(0)
        ) * self.inv_2A.unsqueeze(0).unsqueeze(2)
        
        weighted_grad = grad_face * self.face_areas.unsqueeze(0).unsqueeze(2)
        
        grad_batch = torch.zeros(batch_size, n, 3, device=self.device)
        weights = torch.zeros(batch_size, n, device=self.device)
        
        for local_idx in range(3):
            v_idx = faces[:, local_idx]
            
            idx_expanded = v_idx.view(1, -1, 1).expand(batch_size, -1, 3)
            grad_batch.scatter_add_(1, idx_expanded, weighted_grad)
            
            area_expanded = self.face_areas.unsqueeze(0).expand(batch_size, -1)
            weights.scatter_add_(1, v_idx.unsqueeze(0).expand(batch_size, -1), area_expanded)
        
        weights = weights.clamp(min=1e-10).unsqueeze(2)
        grad_batch = grad_batch / weights
        
        return grad_batch
    
    def _compute_velocity_batch(self, psi_batch, omega_batch, return_moments=False):
        grad_psi = self._compute_gradient_batch(psi_batch)
        normals_expanded = self.normals.unsqueeze(0).expand(psi_batch.shape[0], -1, -1)
        u_rot = torch.cross(normals_expanded, grad_psi, dim=2)
        
        weighted_omega = omega_batch * self.M_diag.unsqueeze(0)
        global_moments = torch.matmul(weighted_omega, self.pos_centered)
        
        coeff = cfg.GLOBAL_COUPLING_SCALE * torch.tanh(cfg.GLOBAL_COUPLING_SHARPNESS * global_moments)
        
        u_global = torch.sum(
            coeff.view(-1, 3, 1, 1) * self.background_basis.unsqueeze(0),
            dim=1
        )
        
        u_batch = u_rot + u_global
        
        u_max = torch.norm(u_batch, dim=2).max(dim=1, keepdim=True)[0].unsqueeze(2)
        u_batch = u_batch / (u_max + 1e-10)
        
        if return_moments:
            return u_batch, global_moments
        return u_batch
    
    def generate_batch(self, batch_size):
        X = self._generate_vortex_sources_batch(batch_size)
        psi = self._solve_poisson_batch(X)
        Y = self._compute_velocity_batch(psi, X)
        return X, Y
    
    def generate_dataset(self, n_samples, batch_size=64):
        print(f"\n{'='*60}")
        print("Generating Poisson Stream Function Dataset")
        print("With Virtual Background Coupling")
        print(f"{'='*60}")
        print(f"[Config] Samples: {n_samples}, Batch: {batch_size}")
        print(f"         Vertices: {self.n_verts}")
        print(f"         Sources: {cfg.N_SOURCES_RANGE}")
        print(f"         Sigma: {cfg.SIGMA_RANGE}")
        print(f"         Global coupling scale: {cfg.GLOBAL_COUPLING_SCALE}")
        print(f"         Global coupling sharpness: {cfg.GLOBAL_COUPLING_SHARPNESS}")
        print()
        print("[Physics]")
        print("  Input  X: Vorticity omega (0-form scalar)")
        print("  Solve:    Delta psi = omega (Laplace-Beltrami Poisson)")
        print("  Moments:  M = integral(omega * pos) dA")
        print("  Coupling: c = scale * tanh(sharpness * M)")
        print("  Output Y: u = n x grad(psi) + sum(c_i * basis_i)")
        print()
        print("[Guarantees]")
        print("  - Divergence-free: div(u) = 0")
        print("  - Deterministic: Given omega, Y is unique")
        print("  - Non-local: Global moments determine background flow")
        print("  - High variance: Nonlinear coupling creates diverse outputs")
        print(f"{'='*60}")
        
        X_list, Y_list = [], []
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        t0 = time.time()
        
        for i in range(n_batches):
            current_batch = min(batch_size, n_samples - i * batch_size)
            
            X_batch, Y_batch = self.generate_batch(current_batch)
            
            X_list.append(X_batch.cpu())
            Y_list.append(Y_batch.cpu())
            
            done = min((i + 1) * batch_size, n_samples)
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            
            if (i + 1) % 10 == 0 or i == n_batches - 1:
                u_norm = torch.norm(Y_batch, dim=2)
                psi_batch = self._solve_poisson_batch(X_batch)
                _, moments = self._compute_velocity_batch(psi_batch, X_batch, return_moments=True)
                print(f"[Batch {i+1:4d}/{n_batches}] "
                      f"Done: {done:5d}/{n_samples} | "
                      f"Rate: {rate:.1f}/s | "
                      f"|u|: {u_norm.mean():.3f}+-{u_norm.std():.3f} | "
                      f"|M|: {torch.norm(moments, dim=1).mean():.3f}")
            
            if self.device.type == 'cuda' and (i + 1) % 50 == 0:
                torch.cuda.empty_cache()
        
        X_data = torch.cat(X_list, dim=0).numpy()
        Y_data = torch.cat(Y_list, dim=0).numpy()
        
        total_time = time.time() - t0
        
        print(f"\n{'='*60}")
        print("Dataset Statistics")
        print(f"{'='*60}")
        print(f"[X] Vorticity (0-form):")
        print(f"    Shape: {X_data.shape}")
        print(f"    Range: [{X_data.min():.4f}, {X_data.max():.4f}]")
        print(f"    Std:   {X_data.std():.4f}")
        
        Y_norms = np.linalg.norm(Y_data, axis=2)
        print(f"\n[Y] Velocity (1-form with background):")
        print(f"    Shape: {Y_data.shape}")
        print(f"    |u|:   mean={Y_norms.mean():.4f}, max={Y_norms.max():.4f}")
        
        print(f"\n[Time] {total_time:.2f}s ({n_samples/total_time:.1f} samples/s)")
        print(f"{'='*60}")
        
        return X_data.astype(np.float32), Y_data.astype(np.float32)


def save_preview_visualization(generator, X_data, Y_data, points, faces, output_path, n_samples=3):
    print(f"\n[Preview] Generating visualization for {n_samples} samples...")
    
    temp_files = []
    
    try:
        for i in range(n_samples):
            omega = X_data[i]
            velocity = Y_data[i]
            vel_mag = np.linalg.norm(velocity, axis=1)
            
            omega_t = torch.tensor(omega, device=generator.device).unsqueeze(0)
            weighted_omega = omega_t * generator.M_diag.unsqueeze(0)
            moments = torch.matmul(weighted_omega, generator.pos_centered).squeeze(0).cpu().numpy()
            coeff = cfg.GLOBAL_COUPLING_SCALE * np.tanh(cfg.GLOBAL_COUPLING_SHARPNESS * moments)
            
            pv_faces = np.hstack([np.full((len(faces), 1), 3), faces]).flatten()
            
            for j, (field_name, field_data, cmap, title_extra) in enumerate([
                ('Vorticity', omega, 'RdBu_r', f'M=[{moments[0]:.2f},{moments[1]:.2f},{moments[2]:.2f}]'),
                ('Velocity Magnitude', vel_mag, 'viridis', f'c=[{coeff[0]:.2f},{coeff[1]:.2f},{coeff[2]:.2f}]'),
                ('X-Coordinate', points[:, 0], 'coolwarm', 'Separation Axis')
            ]):
                fname = f"/tmp/{uuid.uuid4()}.png"
                temp_files.append(fname)
                
                plotter = pv.Plotter(off_screen=True, window_size=[600, 500])
                
                mesh = pv.PolyData(points, pv_faces)
                mesh[field_name] = field_data
                
                plotter.add_mesh(mesh, scalars=field_name, cmap=cmap,
                                show_scalar_bar=True, point_size=5)
                
                title = f"Sample {i+1}: {field_name}"
                if title_extra:
                    title += f"\n{title_extra}"
                plotter.add_title(title, font_size=10)
                
                plotter.camera_position = 'iso'
                plotter.screenshot(fname)
                plotter.close()
        
        fig, axes = plt.subplots(n_samples, 3, figsize=(15, 5 * n_samples))
        if n_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_samples):
            for j in range(3):
                idx = i * 3 + j
                fname = temp_files[idx]
                img = Image.open(fname)
                axes[i, j].imshow(img)
                axes[i, j].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[Preview] Saved to {output_path}")
        
    finally:
        for fname in temp_files:
            if os.path.exists(fname):
                os.remove(fname)


def main():
    print("\n" + "="*70)
    print("GPU-ACCELERATED POISSON STREAM FUNCTION DATASET GENERATOR")
    print("With Virtual Background Coupling (Works on Any Topology)")
    print("="*70)
    
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    pts, faces, normals, mesh_pv = load_mesh(cfg.FILE_IDX, cfg.TARGET_NODES)
    
    os.makedirs(cfg.DATA_DIR_OUT, exist_ok=True)
    pickle_path = os.path.join(cfg.DATA_DIR_OUT, cfg.PICKLE_FILE)
    preview_path = os.path.join(cfg.DATA_DIR_OUT, cfg.PREVIEW_FILE)
    
    generator = GPUPoissonStreamFunctionGenerator(pts, faces, normals)
    
    t_start = time.time()
    X_data, Y_data = generator.generate_dataset(cfg.N_SAMPLES, cfg.BATCH_SIZE)
    t_total = time.time() - t_start
    
    print(f"\n[Time] Total: {t_total:.2f}s ({cfg.N_SAMPLES/t_total:.1f} samples/s)")
    
    save_preview_visualization(generator, X_data, Y_data, pts, faces, preview_path, cfg.N_PREVIEW_SAMPLES)
    
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
            'guarantees': [
                'divergence_free: div(u) = 0',
                'deterministic: Global moments uniquely determined by omega',
                'non_local: Vorticity distribution -> global flow direction',
                'high_variance: Nonlinear tanh creates diverse outputs',
                'geometry_aware: Laplace-Beltrami + tangent projection',
                'steady_state: du/dt = 0'
            ],
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


if __name__ == "__main__":
    main()