import os
import time
import pickle
import warnings
import numpy as np
import torch
import torch.sparse as tsparse
import scipy.sparse as sp
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

warnings.filterwarnings("ignore")

class Config:
    DOMAIN_RADIUS = 2.0
    SPHERE_RADIUS = 0.4
    SPHERE_CENTER = (0.0, 0.0, 0.0)
    TARGET_NODES = 3019
    
    N_BLOBS_RANGE = (4, 10)
    BLOB_SIGMA_RANGE = (0.5, 1.2)
    BLOB_STRENGTH_RANGE = (1.0, 2.0)
    
    N_FOURIER_MODES = 4
    FOURIER_FREQ_RANGE = (0.3, 1.5)
    
    COULOMB_STRENGTH = 0.5
    HARMONIC_STRENGTH = 0.8
    DIPOLE_STRENGTH = 0.2
    
    UNIFORM_STRENGTH_RANGE = (0.5, 1.5)
    LINEAR_STRENGTH_RANGE = (0.2, 0.6)
    
    CG_MAX_ITER = 500
    CG_TOL = 1e-6
    REGULARIZATION_EPS = 1e-8
    
    N_SAMPLES = 3000
    BATCH_SIZE = 64
    N_PREVIEW_SAMPLES = 3
    
    DATA_DIR_OUT = "./flux_field_data"
    PICKLE_FILE = "flux_field_dataset.pkl"
    PREVIEW_FILE = "preview_samples.png"
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

cfg = Config()

class TetrahedralMeshGPU:
    
    def __init__(self, n_nodes_target, domain_radius, sphere_radius, sphere_center, 
                 device=None, seed=None):
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        self.device = torch.device(device or cfg.DEVICE)
        self.n_nodes_target = n_nodes_target
        self.domain_radius = domain_radius
        self.sphere_radius = sphere_radius
        self.sphere_center_np = np.array(sphere_center)
        self.sphere_center = torch.tensor(sphere_center, dtype=torch.float32, device=self.device)
        
        self._generate_points()
        self._build_tetrahedra()
        self._identify_boundaries()
        self._build_gpu_fem_operators()
    
    def _generate_points(self):
        print(f"[Mesh] Generating points in spherical shell domain...")
        print(f"       Target: {self.n_nodes_target} nodes")
        
        points = []
        center = self.sphere_center_np
        
        n_interior = int(self.n_nodes_target * 0.84)
        while len(points) < n_interior:
            batch_size = (n_interior - len(points)) * 5
            candidates = (np.random.rand(batch_size, 3) - 0.5) * 2 * self.domain_radius
            
            r = np.linalg.norm(candidates - center, axis=1)
            mask = (r > self.sphere_radius * 1.15) & (r < self.domain_radius * 0.92)
            valid = candidates[mask]
            points.extend(valid.tolist())
        
        points = points[:n_interior]
        
        n_inner = int(self.n_nodes_target * 0.08)
        for _ in range(n_inner):
            theta = np.random.uniform(0, np.pi)
            phi = np.random.uniform(0, 2 * np.pi)
            r = self.sphere_radius * np.random.uniform(1.16, 1.28)
            x = r * np.sin(theta) * np.cos(phi) + center[0]
            y = r * np.sin(theta) * np.sin(phi) + center[1]
            z = r * np.cos(theta) + center[2]
            points.append([x, y, z])
        
        n_outer = int(self.n_nodes_target * 0.08)
        for _ in range(n_outer):
            theta = np.random.uniform(0, np.pi)
            phi = np.random.uniform(0, 2 * np.pi)
            r = self.domain_radius * np.random.uniform(0.85, 0.91)
            x = r * np.sin(theta) * np.cos(phi) + center[0]
            y = r * np.sin(theta) * np.sin(phi) + center[1]
            z = r * np.cos(theta) + center[2]
            points.append([x, y, z])
        
        self.nodes_np = np.array(points[:self.n_nodes_target], dtype=np.float64)
        self.nodes = torch.tensor(self.nodes_np, dtype=torch.float32, device=self.device)
        self.n_nodes = len(self.nodes_np)
        
        print(f"       Generated: {self.n_nodes} nodes")
    
    def _build_tetrahedra(self):
        print("[Mesh] Building Delaunay tetrahedralization...")
        
        delaunay = Delaunay(self.nodes_np)
        elements = delaunay.simplices.copy()
        
        valid_elements = []
        center = self.sphere_center_np
        
        for elem in elements:
            coords = self.nodes_np[elem]
            centroid = coords.mean(axis=0)
            r_centroid = np.linalg.norm(centroid - center)
            r_vertices = np.linalg.norm(coords - center, axis=1)
            
            if r_centroid > self.sphere_radius * 1.08 and r_vertices.min() > self.sphere_radius * 1.02:
                valid_elements.append(elem)
        
        self.elements_np = np.array(valid_elements, dtype=np.int64)
        self.elements = torch.tensor(self.elements_np, dtype=torch.long, device=self.device)
        self.n_elements = len(self.elements_np)
        
        print(f"       Elements: {self.n_elements} tetrahedra")
    
    def _identify_boundaries(self):
        r = torch.norm(self.nodes - self.sphere_center, dim=1)
        
        self.inner_boundary = torch.where(r < self.sphere_radius * 1.35)[0]
        self.outer_boundary = torch.where(r > self.domain_radius * 0.80)[0]
        self.interior = torch.where(
            (r >= self.sphere_radius * 1.35) & (r <= self.domain_radius * 0.80)
        )[0]
        
        self.is_outer_boundary = torch.zeros(self.n_nodes, dtype=torch.bool, device=self.device)
        self.is_outer_boundary[self.outer_boundary] = True
        
        self.is_inner_boundary = torch.zeros(self.n_nodes, dtype=torch.bool, device=self.device)
        self.is_inner_boundary[self.inner_boundary] = True
        
        print(f"       Inner boundary: {len(self.inner_boundary)} nodes")
        print(f"       Outer boundary: {len(self.outer_boundary)} nodes")
        print(f"       Interior: {len(self.interior)} nodes")
    
    def _build_gpu_fem_operators(self):
        print("[FEM] Building GPU finite element operators...")
        t0 = time.time()
        
        n = self.n_nodes
        device = self.device
        
        coords = self.nodes[self.elements]
        
        ones = torch.ones(self.n_elements, 4, 1, device=device)
        mat = torch.cat([ones, coords], dim=2)
        
        inv_mat = torch.linalg.inv(mat)
        
        self.grad_N = inv_mat[:, 1:, :].transpose(1, 2)
        
        det = torch.linalg.det(mat)
        self.volumes = torch.abs(det) / 6.0
        
        valid_mask = self.volumes > 1e-14
        if not valid_mask.all():
            n_invalid = (~valid_mask).sum().item()
            print(f"       Warning: removing {n_invalid} degenerate elements")
            self.elements = self.elements[valid_mask]
            self.grad_N = self.grad_N[valid_mask]
            self.volumes = self.volumes[valid_mask]
            self.n_elements = len(self.elements)
        
        print(f"       Valid elements: {self.n_elements}")
        
        self._build_sparse_matrices()
        
        self._precompute_gradient_indices()
        
        self._compute_boundary_normals()
        
        print(f"       FEM setup time: {time.time()-t0:.3f}s")
    
    def _build_sparse_matrices(self):
        n = self.n_nodes
        device = self.device
        
        elem_flat = self.elements.reshape(-1)
        
        row_idx = []
        col_idx = []
        stiff_vals = []
        mass_vals = []
        
        for local_i in range(4):
            for local_j in range(4):
                rows = self.elements[:, local_i]
                cols = self.elements[:, local_j]
                
                grad_i = self.grad_N[:, local_i, :]
                grad_j = self.grad_N[:, local_j, :]
                k_local = self.volumes * (grad_i * grad_j).sum(dim=1)
                
                m_local = self.volumes / 20.0 * (2.0 if local_i == local_j else 1.0)
                
                row_idx.append(rows)
                col_idx.append(cols)
                stiff_vals.append(k_local)
                mass_vals.append(m_local)
        
        row_idx = torch.cat(row_idx)
        col_idx = torch.cat(col_idx)
        stiff_vals = torch.cat(stiff_vals)
        mass_vals = torch.cat(mass_vals)
        
        indices = torch.stack([row_idx, col_idx])
        
        self.stiffness = torch.sparse_coo_tensor(
            indices, stiff_vals, (n, n), device=device
        ).coalesce()
        
        self.mass = torch.sparse_coo_tensor(
            indices, mass_vals, (n, n), device=device
        ).coalesce()
        
        eps = cfg.REGULARIZATION_EPS
        
        diag_idx = torch.arange(n, device=device)
        diag_indices = torch.stack([diag_idx, diag_idx])
        diag_vals = torch.full((n,), eps, device=device)
        
        A_reg = torch.sparse_coo_tensor(diag_indices, diag_vals, (n, n), device=device)
        self.A_system = (self.stiffness + A_reg).coalesce()
        
        A_dense_diag = torch.zeros(n, device=device)
        A_indices = self.A_system.indices()
        A_values = self.A_system.values()
        diag_mask = A_indices[0] == A_indices[1]
        A_dense_diag.scatter_add_(0, A_indices[0, diag_mask], A_values[diag_mask])
        
        A_dense_diag[self.outer_boundary] = 1.0
        self.A_diag_inv = 1.0 / A_dense_diag.clamp(min=1e-10)
        
        print(f"       Stiffness nnz: {self.stiffness._nnz()}")
    
    def _precompute_gradient_indices(self):
        n_entries = self.n_elements * 4
        
        self.grad_node_idx = self.elements.reshape(-1)
        self.grad_elem_idx = torch.arange(self.n_elements, device=self.device).repeat_interleave(4)
        self.grad_local_idx = torch.arange(4, device=self.device).repeat(self.n_elements)
        
        node_weights = torch.zeros(self.n_nodes, device=self.device)
        node_weights.scatter_add_(0, self.grad_node_idx, 
                                   self.volumes.repeat_interleave(4))
        self.node_weight_inv = 1.0 / node_weights.clamp(min=1e-15)
    
    def _compute_boundary_normals(self):
        device = self.device
        center = self.sphere_center
        
        r_vec_inner = self.nodes[self.inner_boundary] - center
        r_inner = torch.norm(r_vec_inner, dim=1, keepdim=True).clamp(min=1e-8)
        self.inner_normals = -r_vec_inner / r_inner
        
        r_vec_outer = self.nodes[self.outer_boundary] - center
        r_outer = torch.norm(r_vec_outer, dim=1, keepdim=True).clamp(min=1e-8)
        self.outer_normals = r_vec_outer / r_outer
        
        self.inner_boundary_area = 4 * np.pi * (self.sphere_radius * 1.25) ** 2
        self.outer_boundary_area = 4 * np.pi * (cfg.DOMAIN_RADIUS * 0.85) ** 2
    
    def sparse_matvec(self, A, x):
        if x.dim() == 1:
            return torch.sparse.mm(A, x.unsqueeze(1)).squeeze(1)
        else:
            return torch.sparse.mm(A, x.t()).t()
    
    def solve_poisson_batch_gpu(self, rho_batch):
        batch_size = rho_batch.shape[0]
        n = self.n_nodes
        device = self.device
        
        b = self.sparse_matvec(self.mass, rho_batch)
        
        b[:, self.outer_boundary] = 0.0
        
        x = torch.zeros(batch_size, n, device=device)
        
        r = b - self._apply_A_batch(x)
        r[:, self.outer_boundary] = 0.0
        
        z = r * self.A_diag_inv
        p = z.clone()
        
        rz_old = (r * z).sum(dim=1)
        
        for iteration in range(cfg.CG_MAX_ITER):
            Ap = self._apply_A_batch(p)
            Ap[:, self.outer_boundary] = p[:, self.outer_boundary]
            
            pAp = (p * Ap).sum(dim=1)
            alpha = rz_old / pAp.clamp(min=1e-15)
            
            x = x + alpha.unsqueeze(1) * p
            r = r - alpha.unsqueeze(1) * Ap
            r[:, self.outer_boundary] = 0.0
            
            r_norm = torch.norm(r, dim=1)
            if r_norm.max() < cfg.CG_TOL:
                break
            
            z = r * self.A_diag_inv
            rz_new = (r * z).sum(dim=1)
            
            beta = rz_new / rz_old.clamp(min=1e-15)
            p = z + beta.unsqueeze(1) * p
            
            rz_old = rz_new
        
        x = x - x.mean(dim=1, keepdim=True)
        
        return x
    
    def _apply_A_batch(self, x):
        return self.sparse_matvec(self.A_system, x)
    
    def compute_gradient_batch_gpu(self, phi_batch):
        batch_size = phi_batch.shape[0]
        n = self.n_nodes
        device = self.device
        
        phi_elem = phi_batch[:, self.elements]
        
        grad_elem = torch.einsum('bei,eij->bej', phi_elem, self.grad_N)
        
        grad_elem_weighted = grad_elem * self.volumes.unsqueeze(0).unsqueeze(2)
        
        grad_batch = torch.zeros(batch_size, n, 3, device=device)
        
        grad_expanded = grad_elem_weighted.unsqueeze(2).expand(-1, -1, 4, -1)
        grad_expanded = grad_expanded.reshape(batch_size, -1, 3)
        
        node_idx = self.grad_node_idx.unsqueeze(0).unsqueeze(2).expand(batch_size, -1, 3)
        
        grad_batch.scatter_add_(1, node_idx, grad_expanded)
        
        grad_batch = grad_batch * self.node_weight_inv.unsqueeze(0).unsqueeze(2)
        
        return grad_batch
    
    def compute_inner_flux_batch(self, field_batch):
        inner_field = field_batch[:, self.inner_boundary, :]
        flux_density = (inner_field * self.inner_normals.unsqueeze(0)).sum(dim=2)
        avg_flux_density = flux_density.mean(dim=1)
        total_flux = avg_flux_density * self.inner_boundary_area
        return total_flux

class GPUMonopoleHarmonicFieldGenerator:
    
    def __init__(self, mesh, device=None):
        self.device = torch.device(device or cfg.DEVICE)
        self.mesh = mesh
        self.n_verts = mesh.n_nodes
        
        print(f"\n{'='*60}")
        print("GPU Field Generator (Topological Harmonic 2-Form Version)")
        print(f"{'='*60}")
        
        self.vertices = mesh.nodes
        self.sphere_radius = mesh.sphere_radius
        self.sphere_center = mesh.sphere_center
        
        self.normalized_coords = (self.vertices - self.sphere_center) / cfg.DOMAIN_RADIUS
        
        self._build_harmonic_basis()
        self._validate()
    
    def _build_harmonic_basis(self):
        print("[Build] Constructing topological harmonic 2-form basis...")
        t0 = time.time()
        
        vertices = self.vertices
        center = self.sphere_center
        R_in = self.sphere_radius
        R_out = cfg.DOMAIN_RADIUS
        
        r_vec = vertices - center
        r = torch.norm(r_vec, dim=1, keepdim=True)
        r_safe = r.clamp(min=R_in * 0.5)
        r_hat = r_vec / r_safe
        
        self.harmonic_monopole = r_hat / (r_safe ** 2)
        
        r_mid = (R_in * 1.3 + R_out * 0.85) / 2
        mid_mask = (r.squeeze() > R_in * 1.2) & (r.squeeze() < R_out * 0.9)
        if mid_mask.sum() > 10:
            reference_flux = torch.norm(self.harmonic_monopole[mid_mask], dim=1).mean()
            self.harmonic_monopole = self.harmonic_monopole / reference_flux.clamp(min=1e-8)
        
        surface_dist = r.squeeze() - R_in
        inner_cutoff = torch.sigmoid((surface_dist - R_in * 0.1) * 15)
        outer_cutoff = torch.sigmoid((R_out * 0.9 - r.squeeze()) * 10)
        self.harmonic_monopole = self.harmonic_monopole * (inner_cutoff * outer_cutoff).unsqueeze(1)
        
        self._verify_harmonic_properties()
        
        print(f"       Harmonic basis construction: {time.time()-t0:.3f}s")
    
    def _verify_harmonic_properties(self):
        H = self.harmonic_monopole
        r = torch.norm(self.vertices - self.sphere_center, dim=1)
        
        shell_mask = (r > self.sphere_radius * 1.4) & (r < cfg.DOMAIN_RADIUS * 0.7)
        
        if shell_mask.sum() > 100:
            H_shell = H[shell_mask]
            H_mag = torch.norm(H_shell, dim=1)
            r_shell = r[shell_mask]
            
            expected_ratio = (r_shell.max() / r_shell.min()) ** 2
            actual_ratio = H_mag.max() / H_mag.min().clamp(min=1e-8)
            
            print(f"       Harmonic field 1/r² verification: expected ratio ~{expected_ratio:.2f}, actual ~{actual_ratio:.2f}")
    
    def _validate(self):
        X_test = self._generate_source_field_batch(2)
        Y_test = self._compute_output_batch(X_test)
        Y_norm = torch.norm(Y_test, dim=2)
        
        flux_test = self.mesh.compute_inner_flux_batch(Y_test)
        print(f"[Valid] |Y|_mean={Y_norm.mean():.4f}, |Y|_max={Y_norm.max():.4f}")
        print(f"[Valid] Inner boundary flux: {flux_test.mean():.4f}")

    def _generate_source_field_batch(self, batch_size):
        device = self.device
        n = self.n_verts
        coords = self.normalized_coords
        
        X = torch.zeros(batch_size, n, device=device)
        
        r = torch.norm(self.vertices - self.sphere_center, dim=1)
        R_in = self.sphere_radius
        R_out = cfg.DOMAIN_RADIUS
        
        inner_weight = torch.exp(-((r - R_in * 1.5) / (R_in * 0.5)) ** 2)
        outer_weight = torch.exp(-((r - R_out * 0.7) / (R_out * 0.3)) ** 2)
        
        for b in range(batch_size):
            mode_weights = torch.softmax(torch.randn(5, device=device), dim=0)
            field = torch.zeros(n, device=device)
            
            fourier_field = torch.zeros(n, device=device)
            for _ in range(cfg.N_FOURIER_MODES):
                freq = torch.empty(3, device=device).uniform_(*cfg.FOURIER_FREQ_RANGE)
                phase = torch.empty(3, device=device).uniform_(0, 2 * np.pi)
                wave = torch.sin(
                    freq[0] * coords[:, 0] * np.pi + 
                    freq[1] * coords[:, 1] * np.pi + 
                    freq[2] * coords[:, 2] * np.pi + 
                    phase.sum()
                )
                fourier_field += wave
            field += mode_weights[0] * fourier_field
            
            blob_field = torch.zeros(n, device=device)
            n_blobs = torch.randint(*cfg.N_BLOBS_RANGE, (1,)).item()
            
            rand_r = torch.empty(n_blobs, device=device).uniform_(R_in * 1.3, R_out * 0.85)
            rand_theta = torch.empty(n_blobs, device=device).uniform_(0, np.pi)
            rand_phi = torch.empty(n_blobs, device=device).uniform_(0, 2*np.pi)
            
            bx = rand_r * torch.sin(rand_theta) * torch.cos(rand_phi)
            by = rand_r * torch.sin(rand_theta) * torch.sin(rand_phi)
            bz = rand_r * torch.cos(rand_theta)
            blob_centers = torch.stack([bx, by, bz], dim=1) + self.sphere_center.unsqueeze(0)
            
            sigmas = torch.empty(n_blobs, device=device).uniform_(*cfg.BLOB_SIGMA_RANGE)
            strengths = torch.empty(n_blobs, device=device).uniform_(*cfg.BLOB_STRENGTH_RANGE)
            signs = torch.randint(0, 2, (n_blobs,), device=device).float() * 2 - 1
            
            diff = self.vertices.unsqueeze(1) - blob_centers.unsqueeze(0)
            dist_sq = (diff ** 2).sum(dim=2)
            gaussians = torch.exp(-dist_sq / (2 * sigmas.unsqueeze(0) ** 2))
            blob_field = (gaussians * (signs * strengths).unsqueeze(0)).sum(dim=1)
            
            field += mode_weights[1] * blob_field
            
            monopole_charge = torch.empty(1, device=device).uniform_(-1.5, 1.5)
            monopole_field = monopole_charge * inner_weight
            field += mode_weights[2] * monopole_field
            
            dipole_axis = torch.randn(3, device=device)
            dipole_axis = dipole_axis / torch.norm(dipole_axis)
            dipole_strength = torch.empty(1, device=device).uniform_(-1.0, 1.0)
            r_vec = self.vertices - self.sphere_center
            cos_theta = (r_vec @ dipole_axis) / r.clamp(min=1e-6)
            dipole_pattern = cos_theta * inner_weight
            field += mode_weights[3] * dipole_strength * dipole_pattern
            
            quadrupole_axis1 = torch.randn(3, device=device)
            quadrupole_axis1 = quadrupole_axis1 / torch.norm(quadrupole_axis1)
            quadrupole_axis2 = torch.randn(3, device=device)
            quadrupole_axis2 = quadrupole_axis2 - (quadrupole_axis2 @ quadrupole_axis1) * quadrupole_axis1
            quadrupole_axis2 = quadrupole_axis2 / torch.norm(quadrupole_axis2).clamp(min=1e-6)
            
            cos_theta1 = (r_vec @ quadrupole_axis1) / r.clamp(min=1e-6)
            cos_theta2 = (r_vec @ quadrupole_axis2) / r.clamp(min=1e-6)
            quadrupole_pattern = (3 * cos_theta1 ** 2 - 1) + cos_theta1 * cos_theta2
            quadrupole_strength = torch.empty(1, device=device).uniform_(-0.5, 0.5)
            field += mode_weights[4] * quadrupole_strength * quadrupole_pattern * (inner_weight + 0.3 * outer_weight)
            
            X[b] = field
        
        X_max = X.abs().amax(dim=1, keepdim=True).clamp(min=1e-10)
        X = X / X_max
        X = torch.tanh(X * 1.5)
        
        return X

    def _compute_topological_charge(self, X_batch):
        batch_size = X_batch.shape[0]
        elements = self.mesh.elements
        volumes = self.mesh.volumes
        
        X_elem = X_batch[:, elements].mean(dim=2)
        total_charge = (X_elem * volumes).sum(dim=1)
        
        return total_charge
    
    def _compute_multipole_moments(self, X_batch):
        batch_size = X_batch.shape[0]
        device = self.device
        
        r_vec = self.vertices - self.sphere_center
        r = torch.norm(r_vec, dim=1, keepdim=True).clamp(min=1e-6)
        r_hat = r_vec / r
        
        elements = self.mesh.elements
        volumes = self.mesh.volumes
        
        X_elem = X_batch[:, elements].mean(dim=2)
        
        monopole = (X_elem * volumes).sum(dim=1)
        
        r_elem = r_vec[elements].mean(dim=2)
        dipole_x = (X_elem * volumes * r_elem[:, 0]).sum(dim=1)
        dipole_y = (X_elem * volumes * r_elem[:, 1]).sum(dim=1)
        dipole_z = (X_elem * volumes * r_elem[:, 2]).sum(dim=1)
        dipole = torch.stack([dipole_x, dipole_y, dipole_z], dim=1)
        
        return {
            'monopole': monopole,
            'dipole': dipole
        }

    def _compute_harmonic_field_batch(self, X_batch):
        batch_size = X_batch.shape[0]
        n = self.n_verts
        device = self.device
        
        moments = self._compute_multipole_moments(X_batch)
        
        Q = moments['monopole']
        harmonic_field = Q.view(-1, 1, 1) * self.harmonic_monopole.unsqueeze(0)
        
        p = moments['dipole']
        r_vec = self.vertices - self.sphere_center
        r = torch.norm(r_vec, dim=1, keepdim=True).clamp(min=self.sphere_radius * 0.8)
        r_hat = r_vec / r
        
        p_dot_r = torch.einsum('bi,ni->bn', p, r_hat)
        dipole_term1 = 3 * p_dot_r.unsqueeze(2) * r_hat.unsqueeze(0)
        dipole_term2 = p.unsqueeze(1).expand(-1, n, -1)
        dipole_field = (dipole_term1 - dipole_term2) / (r ** 3).unsqueeze(0)
        
        surface_dist = r.squeeze() - self.sphere_radius
        decay = torch.sigmoid((surface_dist - 0.05) * 20)
        dipole_field = dipole_field * decay.unsqueeze(0).unsqueeze(2)
        
        dipole_scale = 0.3
        total_harmonic = harmonic_field + dipole_scale * dipole_field
        
        return total_harmonic

    def _compute_output_batch(self, X_batch):
        batch_size = X_batch.shape[0]
        n = self.n_verts
        device = self.device
        
        phi = self.mesh.solve_poisson_batch_gpu(X_batch)
        exact_field = self.mesh.compute_gradient_batch_gpu(phi)
        
        harmonic_field = self._compute_harmonic_field_batch(X_batch)
        
        Y = (cfg.COULOMB_STRENGTH * exact_field +
             cfg.HARMONIC_STRENGTH * harmonic_field)
        
        r = torch.norm(self.vertices - self.sphere_center, dim=1)
        inner_dist = r - self.sphere_radius
        outer_dist = cfg.DOMAIN_RADIUS - r
        
        inner_factor = torch.sigmoid((inner_dist - 0.03) * 25)
        outer_factor = torch.sigmoid((outer_dist - 0.1) * 15)
        boundary_factor = inner_factor * outer_factor
        Y = Y * boundary_factor.view(1, -1, 1)
        
        Y_mag = torch.norm(Y, dim=2, keepdim=True)
        mean_mag = Y_mag.mean(dim=1, keepdim=True).clamp(min=1e-8)
        clamp_threshold = mean_mag * 5.0
        scale_factor = torch.tanh(Y_mag / clamp_threshold) * clamp_threshold / (Y_mag + 1e-8)
        Y = Y * scale_factor
        
        Y_max = torch.norm(Y, dim=2).max(dim=1, keepdim=True)[0].unsqueeze(2).clamp(min=1e-10)
        Y = Y / Y_max
        
        return Y

    def generate_batch(self, batch_size):
        X = self._generate_source_field_batch(batch_size)
        Y = self._compute_output_batch(X)
        return X, Y

    def generate_dataset(self, n_samples, batch_size=64):
        print(f"\n{'='*60}")
        print("Generating Dataset (Topological Harmonic 2-Form)")
        print(f"{'='*60}")
        print(f"[Config] Samples: {n_samples}, Batch: {batch_size}")
        print(f"         Vertices: {self.n_verts}, Elements: {self.mesh.n_elements}")
        print(f"         CG solver: max_iter={cfg.CG_MAX_ITER}, tol={cfg.CG_TOL}")
        print()
        print("[Physics Model]")
        print("  Domain: Spherical shell with inner obstacle")
        print("  H^2(M) ≅ Z: One non-trivial harmonic 2-form")
        print("  Harmonic basis: (1/r²)r̂ - monopole field")
        print()
        print("[Source Field Modes]")
        print("  1. Fourier waves")
        print("  2. Gaussian blobs")
        print("  3. Monopole charge near inner boundary")
        print("  4. Dipole pattern")
        print("  5. Quadrupole pattern")
        print()
        print("[Output Field Components]")
        print("  1. Exact field (gradient of Poisson solution)")
        print("  2. Harmonic 2-form (topological, from multipole moments)")
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
            
            if (i + 1) % 5 == 0 or i == n_batches - 1:
                Y_norm = torch.norm(Y_batch, dim=2)
                X_range = X_batch.max() - X_batch.min()
                
                flux = self.mesh.compute_inner_flux_batch(Y_batch)
                
                print(f"[Batch {i+1:4d}/{n_batches}] "
                      f"Done: {done:5d}/{n_samples} | "
                      f"Rate: {rate:.1f}/s | "
                      f"X_range: {X_range:.2f} | "
                      f"|Y|: {Y_norm.mean():.3f} | "
                      f"Flux: {flux.mean():.3f}")
            
            if self.device.type == 'cuda' and (i + 1) % 20 == 0:
                torch.cuda.empty_cache()
        
        X_data = torch.cat(X_list, dim=0).numpy()
        Y_data = torch.cat(Y_list, dim=0).numpy()
        
        total_time = time.time() - t0
        
        print(f"\n{'='*60}")
        print("Dataset Statistics")
        print(f"{'='*60}")
        print(f"[X] Shape: {X_data.shape}, Range: [{X_data.min():.3f}, {X_data.max():.3f}]")
        
        Y_norms = np.linalg.norm(Y_data, axis=2)
        print(f"[Y] Shape: {Y_data.shape}, |Y|: {Y_norms.mean():.3f}±{Y_norms.std():.3f}")
        print(f"[Time] {total_time:.2f}s ({n_samples/total_time:.1f} samples/s)")
        print(f"{'='*60}")
        
        return X_data.astype(np.float32), Y_data.astype(np.float32)

def save_preview_visualization(mesh, X_data, Y_data, output_path, n_samples=3):
    print(f"\n[Preview] Generating visualization...")
    
    from scipy.interpolate import NearestNDInterpolator
    
    points = mesh.nodes_np
    sphere_radius = mesh.sphere_radius
    sphere_center = mesh.sphere_center_np
    
    fig = plt.figure(figsize=(20, 7 * n_samples))
    
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    sx = sphere_center[0] + sphere_radius * np.outer(np.cos(u), np.sin(v))
    sy = sphere_center[1] + sphere_radius * np.outer(np.sin(u), np.sin(v))
    sz = sphere_center[2] + sphere_radius * np.outer(np.ones(np.size(u)), np.cos(v))
    
    r_from_center = np.linalg.norm(points - sphere_center, axis=1)
    
    outer_limit = cfg.DOMAIN_RADIUS * 0.85
    inner_limit = sphere_radius * 1.2
    valid_mask = (r_from_center < outer_limit) & (r_from_center > inner_limit)
    valid_indices = np.where(valid_mask)[0]
    
    r_valid = r_from_center[valid_mask]
    dist_sort_idx = valid_indices[np.argsort(r_valid)]
    
    points_plot = points[dist_sort_idx]
    
    for i in range(n_samples):
        X = X_data[i]
        Y = Y_data[i]
        
        r_vec = points - sphere_center
        r_norm = np.linalg.norm(r_vec, axis=1, keepdims=True)
        r_hat = r_vec / (r_norm + 1e-6)
        Y_radial = np.sum(Y * r_hat, axis=1)
        
        Y_radial_max = np.abs(Y_radial).max()
        if Y_radial_max > 1e-10:
            Y_radial_normalized = Y_radial / Y_radial_max
        else:
            Y_radial_normalized = Y_radial
        
        X_plot = X[dist_sort_idx]
        Y_radial_plot = Y_radial_normalized[dist_sort_idx]
        
        ax1 = fig.add_subplot(n_samples, 3, i * 3 + 1, projection='3d')
        
        sc1 = ax1.scatter(points_plot[:, 0], 
                          points_plot[:, 1], 
                          points_plot[:, 2],
                          c=X_plot, cmap='RdBu_r', s=15, alpha=0.9,
                          vmin=-1, vmax=1, edgecolors='none')
        ax1.plot_surface(sx, sy, sz, color='gray', alpha=0.3)
        ax1.set_title(f'Sample {i+1}: Input X (ρ_m)\nRange: [{X.min():.2f}, {X.max():.2f}]', 
                      fontweight='bold')
        plt.colorbar(sc1, ax=ax1, shrink=0.6, label='Charge density')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        
        ax2 = fig.add_subplot(n_samples, 3, i * 3 + 2, projection='3d')
        
        sc2 = ax2.scatter(points_plot[:, 0], 
                          points_plot[:, 1], 
                          points_plot[:, 2],
                          c=Y_radial_plot, cmap='RdBu_r', s=15, alpha=0.9,
                          vmin=-1, vmax=1, edgecolors='none')
        
        stride = max(1, len(points) // 200)
        idx = np.arange(0, len(points), stride)
        ax2.quiver(points[idx, 0], points[idx, 1], points[idx, 2],
                   Y[idx, 0], Y[idx, 1], Y[idx, 2],
                   length=0.12, normalize=True, alpha=0.6, color='black')
        ax2.plot_surface(sx, sy, sz, color='purple', alpha=0.5)
        ax2.set_title(f'Sample {i+1}: Output Y (B field)\nRadial range: [{Y_radial.min():.2f}, {Y_radial.max():.2f}]', 
                      fontweight='bold')
        plt.colorbar(sc2, ax=ax2, shrink=0.6, label='Radial component (normalized)')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        
        ax3 = fig.add_subplot(n_samples, 3, i * 3 + 3, projection='3d')
        
        interp = [NearestNDInterpolator(points, Y[:, j]) for j in range(3)]
        
        bounds_min, bounds_max = points.min(axis=0), points.max(axis=0)
        
        pos_mask = X > 0.5
        neg_mask = X < -0.5
        
        seeds = []
        if pos_mask.sum() > 0:
            n_pos = min(25, pos_mask.sum())
            idx_pos = np.random.choice(np.where(pos_mask)[0], n_pos, replace=False)
            seeds.extend([(points[j], 'red') for j in idx_pos])
        if neg_mask.sum() > 0:
            n_neg = min(25, neg_mask.sum())
            idx_neg = np.random.choice(np.where(neg_mask)[0], n_neg, replace=False)
            seeds.extend([(points[j], 'blue') for j in idx_neg])
        
        for seed, color in seeds[:50]:
            traj = [seed.copy()]
            pos = seed.copy()
            for _ in range(100):
                if np.any(pos < bounds_min * 0.95) or np.any(pos > bounds_max * 0.95):
                    break
                if np.linalg.norm(pos - sphere_center) < sphere_radius * 1.05:
                    break
                vel = np.array([interp[j](pos)[0] for j in range(3)])
                if np.linalg.norm(vel) < 1e-6:
                    break
                pos = pos + 0.004 * vel / (np.linalg.norm(vel) + 1e-8)
                traj.append(pos.copy())
            if len(traj) > 5:
                traj = np.array(traj)
                ax3.plot(traj[:, 0], traj[:, 1], traj[:, 2], 
                         color=color, alpha=0.6, lw=0.8)
        
        ax3.plot_surface(sx, sy, sz, color='purple', alpha=0.5)
        ax3.set_title(f'Sample {i+1}: Field Lines', fontweight='bold')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Preview] Saved to {output_path}")

def save_slice_visualization(mesh, X_data, Y_data, output_path, n_samples=3):
    print(f"[Slice Preview] Generating slice visualization...")
    
    from scipy.interpolate import NearestNDInterpolator
    
    points = mesh.nodes_np
    sphere_center = mesh.sphere_center_np
    sphere_radius = mesh.sphere_radius
    
    fig, axes = plt.subplots(n_samples, 4, figsize=(20, 5 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    grid_size = 100
    x_range = np.linspace(-cfg.DOMAIN_RADIUS * 0.95, cfg.DOMAIN_RADIUS * 0.95, grid_size)
    y_range = np.linspace(-cfg.DOMAIN_RADIUS * 0.95, cfg.DOMAIN_RADIUS * 0.95, grid_size)
    X_grid, Y_grid = np.meshgrid(x_range, y_range)
    
    for i in range(n_samples):
        X = X_data[i]
        Y = Y_data[i]
        Y_mag = np.linalg.norm(Y, axis=1)
        
        interp_X = NearestNDInterpolator(points, X)
        interp_Y_mag = NearestNDInterpolator(points, Y_mag)
        
        Z_slice = np.zeros_like(X_grid)
        query_points = np.column_stack([X_grid.ravel(), Y_grid.ravel(), Z_slice.ravel()])
        
        X_slice = interp_X(query_points).reshape(grid_size, grid_size)
        Y_slice = interp_Y_mag(query_points).reshape(grid_size, grid_size)
        
        r_grid = np.sqrt(X_grid**2 + Y_grid**2)
        mask = r_grid < sphere_radius * 1.1
        X_slice[mask] = np.nan
        Y_slice[mask] = np.nan
        
        ax1 = axes[i, 0]
        im1 = ax1.pcolormesh(X_grid, Y_grid, X_slice, cmap='RdBu_r', 
                              vmin=-1, vmax=1, shading='auto')
        circle = plt.Circle((0, 0), sphere_radius, fill=True, color='gray', alpha=0.8)
        ax1.add_patch(circle)
        ax1.set_aspect('equal')
        ax1.set_title(f'Sample {i+1}: X (z=0 slice)', fontweight='bold')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        plt.colorbar(im1, ax=ax1, shrink=0.8)
        
        ax2 = axes[i, 1]
        im2 = ax2.pcolormesh(X_grid, Y_grid, Y_slice, cmap='viridis', shading='auto')
        circle = plt.Circle((0, 0), sphere_radius, fill=True, color='gray', alpha=0.8)
        ax2.add_patch(circle)
        ax2.set_aspect('equal')
        ax2.set_title(f'Sample {i+1}: |Y| (z=0 slice)', fontweight='bold')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        plt.colorbar(im2, ax=ax2, shrink=0.8)
        
        Z_grid_xz, X_grid_xz = np.meshgrid(x_range, y_range)
        Y_slice_xz = np.zeros_like(X_grid_xz)
        query_points_xz = np.column_stack([X_grid_xz.ravel(), Y_slice_xz.ravel(), Z_grid_xz.ravel()])
        
        X_slice_xz = interp_X(query_points_xz).reshape(grid_size, grid_size)
        r_grid_xz = np.sqrt(X_grid_xz**2 + Z_grid_xz**2)
        mask_xz = r_grid_xz < sphere_radius * 1.1
        X_slice_xz[mask_xz] = np.nan
        
        ax3 = axes[i, 2]
        im3 = ax3.pcolormesh(X_grid_xz, Z_grid_xz, X_slice_xz, cmap='RdBu_r',
                              vmin=-1, vmax=1, shading='auto')
        circle = plt.Circle((0, 0), sphere_radius, fill=True, color='gray', alpha=0.8)
        ax3.add_patch(circle)
        ax3.set_aspect('equal')
        ax3.set_title(f'Sample {i+1}: X (y=0 slice)', fontweight='bold')
        ax3.set_xlabel('x')
        ax3.set_ylabel('z')
        plt.colorbar(im3, ax=ax3, shrink=0.8)
        
        ax4 = axes[i, 3]
        ax4.hist(X.ravel(), bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax4.axvline(x=0, color='red', linestyle='--', lw=2)
        ax4.set_xlabel('X value')
        ax4.set_ylabel('Count')
        ax4.set_title(f'Sample {i+1}: X histogram', fontweight='bold')
        ax4.set_xlim(-1.2, 1.2)
    
    plt.tight_layout()
    slice_path = output_path.replace('.png', '_slices.png')
    plt.savefig(slice_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Slice Preview] Saved to {slice_path}")

def main():
    print("\n" + "="*70)
    print("TOPOLOGICAL HARMONIC 2-FORM FIELD GENERATOR")
    print("Spherical Shell Domain with H^2 ≅ Z Cohomology")
    print("="*70)
    
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    print("\n[Step 1] Building GPU tetrahedral mesh...")
    mesh = TetrahedralMeshGPU(
        n_nodes_target=cfg.TARGET_NODES,
        domain_radius=cfg.DOMAIN_RADIUS,
        sphere_radius=cfg.SPHERE_RADIUS,
        sphere_center=cfg.SPHERE_CENTER,
        device=cfg.DEVICE,
        seed=42
    )
    
    os.makedirs(cfg.DATA_DIR_OUT, exist_ok=True)
    pickle_path = os.path.join(cfg.DATA_DIR_OUT, cfg.PICKLE_FILE)
    preview_path = os.path.join(cfg.DATA_DIR_OUT, cfg.PREVIEW_FILE)
    
    print("\n[Step 2] Building field generator...")
    generator = GPUMonopoleHarmonicFieldGenerator(mesh)
    
    print("\n[Step 3] Generating dataset...")
    t_start = time.time()
    X_data, Y_data = generator.generate_dataset(cfg.N_SAMPLES, cfg.BATCH_SIZE)
    t_total = time.time() - t_start
    
    print(f"\n[Total Time] {t_total:.2f}s ({cfg.N_SAMPLES/t_total:.1f} samples/s)")
    
    save_preview_visualization(mesh, X_data, Y_data, preview_path, cfg.N_PREVIEW_SAMPLES)
    
    save_slice_visualization(mesh, X_data, Y_data, preview_path, cfg.N_PREVIEW_SAMPLES)
    
    data = {
        'X_data': X_data,
        'Y_data': Y_data,
        'nodes': mesh.nodes_np.astype(np.float32),
        'elements': mesh.elements_np,
        'inner_boundary': mesh.inner_boundary.cpu().numpy(),
        'outer_boundary': mesh.outer_boundary.cpu().numpy(),
        'config': {
            'n_samples': len(X_data),
            'n_nodes': mesh.n_nodes,
            'n_elements': mesh.n_elements,
            'physics': 'topological_harmonic_2form',
            'mesh_type': 'tetrahedral_gpu',
            'solver': 'batched_cg_jacobi',
            'topology': 'H^2(M) ≅ Z, monopole harmonic',
            'source_modes': ['fourier', 'gaussian_blob', 'monopole', 'dipole', 'quadrupole'],
            'output_components': ['exact_field', 'harmonic_2form'],
            'geometry': {
                'domain_radius': cfg.DOMAIN_RADIUS,
                'sphere_radius': cfg.SPHERE_RADIUS,
                'sphere_center': cfg.SPHERE_CENTER
            }
        }
    }
    
    print(f"\n[Save] Writing to {pickle_path}...")
    with open(pickle_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"[Done] {os.path.getsize(pickle_path)/1e6:.2f} MB")
    print(f"       X: {X_data.shape}, Y: {Y_data.shape}")

if __name__ == "__main__":
    main()