import os
import time
import pickle
import warnings
import numpy as np
import pyvista as pv
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu
from numba import njit, prange

pv.OFF_SCREEN = True
try:
    pv.start_xvfb()
except Exception:
    pass

warnings.filterwarnings("ignore")
np.random.seed(42)

class Config:
    TORUS_MAJOR_RADIUS = 1.0
    TORUS_MINOR_RADIUS = 0.4
    TORUS_U_RES = 80
    TORUS_V_RES = 40
    TARGET_NODES = 2500
    
    DIFFUSIVITY = 0.01
    ADVECTION_SPEED = 1.0
    DT = 0.02
    N_TIMESTEPS = 15
    
    HARMONIC_BIAS_MIN = -1.0
    HARMONIC_BIAS_MAX = 1.0
    
    N_SAMPLES = 3000
    N_BLOBS_MIN = 5
    N_BLOBS_MAX = 11
    BLOB_SIGMA = 0.15
    
    OUTPUT_DIR = "./data/toroidalTransport"
    PICKLE_FILE = "advdiff_torus_dataset.pkl"

cfg = Config()

def generate_torus_mesh(major_radius=1.0, minor_radius=0.4, u_res=80, v_res=40, target_nodes=2500):
    print(f"\n[Geometry] Generating Torus Mesh...")
    mesh = pv.ParametricTorus(
        ringradius=major_radius,
        crosssectionradius=minor_radius,
        u_res=u_res, v_res=v_res
    )
    mesh = pv.PolyData(mesh)
    
    if not mesh.is_all_triangles:
        mesh = mesh.triangulate()
    
    if mesh.n_points > target_nodes:
        ratio = 1.0 - target_nodes / mesh.n_points
        mesh = mesh.decimate(ratio).clean().triangulate()
    
    mesh = mesh.compute_normals(cell_normals=True, point_normals=True, 
                                 auto_orient_normals=True, consistent_normals=True)
    
    points = mesh.points.copy()
    faces = mesh.faces.reshape((-1, 4))[:, 1:]
    normals = mesh.point_data.get('Normals', np.zeros_like(points)).copy()
    
    print(f"          Final: {len(points)} nodes, {len(faces)} faces")
    return points.astype(np.float32), faces.astype(np.int32), normals.astype(np.float32), mesh

def compute_torus_tangent_basis(points, normals, major_radius=1.0):
    n_points = len(points)
    theta = np.arctan2(points[:, 1], points[:, 0])
    t_u = np.stack([-np.sin(theta), np.cos(theta), np.zeros(n_points)], axis=1)
    t_v = np.cross(normals, t_u)
    t_v = t_v / (np.linalg.norm(t_v, axis=1, keepdims=True) + 1e-9)
    return t_u.astype(np.float32), t_v.astype(np.float32), theta.astype(np.float32)

@njit(parallel=True, cache=True)
def compute_advection_flux_numba(u, points, faces, velocity, normals, M_lumped_inv):
    n_nodes = len(points)
    n_faces = len(faces)
    flux = np.zeros(n_nodes, dtype=np.float64)
    
    for f_idx in prange(n_faces):
        i, j, k = faces[f_idx, 0], faces[f_idx, 1], faces[f_idx, 2]
        
        v_face_x = (velocity[i, 0] + velocity[j, 0] + velocity[k, 0]) / 3.0
        v_face_y = (velocity[i, 1] + velocity[j, 1] + velocity[k, 1]) / 3.0
        v_face_z = (velocity[i, 2] + velocity[j, 2] + velocity[k, 2]) / 3.0
        
        for edge_idx in range(3):
            if edge_idx == 0:   a, b, c = i, j, k
            elif edge_idx == 1: a, b, c = j, k, i
            else:               a, b, c = k, i, j
            
            edge_x = points[b, 0] - points[a, 0]
            edge_y = points[b, 1] - points[a, 1]
            edge_z = points[b, 2] - points[a, 2]
            edge_len = np.sqrt(edge_x**2 + edge_y**2 + edge_z**2)
            if edge_len < 1e-12: continue
            
            fn_x, fn_y, fn_z = normals[a, 0], normals[a, 1], normals[a, 2]
            en_x = fn_y * edge_z - fn_z * edge_y
            en_y = fn_z * edge_x - fn_x * edge_z
            en_z = fn_x * edge_y - fn_y * edge_x
            en_len = np.sqrt(en_x**2 + en_y**2 + en_z**2) + 1e-12
            en_x, en_y, en_z = en_x/en_len, en_y/en_len, en_z/en_len
            
            v_normal = v_face_x * en_x + v_face_y * en_y + v_face_z * en_z
            u_upwind = (u[a] + u[b]) / 2.0 if v_normal > 0 else u[c]
            
            flux_contrib = v_normal * u_upwind * edge_len / 3.0
            flux[a] -= flux_contrib
            flux[b] -= flux_contrib
    
    for i in prange(n_nodes):
        flux[i] *= M_lumped_inv[i]
    
    return flux.astype(np.float32)

@njit(cache=True)
def generate_gaussian_blob(points, center_idx, sigma, amplitude):
    n_nodes = len(points)
    blob = np.zeros(n_nodes, dtype=np.float32)
    cx, cy, cz = points[center_idx, 0], points[center_idx, 1], points[center_idx, 2]
    inv_2sigma2 = 1.0 / (2.0 * sigma * sigma)
    
    for i in range(n_nodes):
        dx = points[i, 0] - cx
        dy = points[i, 1] - cy
        dz = points[i, 2] - cz
        blob[i] = amplitude * np.exp(-(dx*dx + dy*dy + dz*dz) * inv_2sigma2)
    return blob

def build_homology_cycles(points, major_radius, minor_radius):
    theta = np.arctan2(points[:, 1], points[:, 0])
    dist_from_axis = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
    
    theta_tol = np.pi / 20
    poloidal_mask = np.abs(theta) < theta_tol
    poloidal_idx = np.where(poloidal_mask)[0]
    if len(poloidal_idx) > 0:
        phi_p = np.arctan2(points[poloidal_idx, 2], 
                          dist_from_axis[poloidal_idx] - major_radius)
        poloidal_cycle = poloidal_idx[np.argsort(phi_p)]
    else:
        poloidal_cycle = np.array([], dtype=np.int32)
    
    z_tol = minor_radius * 0.3
    outer_thresh = major_radius + minor_radius * 0.7
    toroidal_mask = (np.abs(points[:, 2]) < z_tol) & (dist_from_axis > outer_thresh)
    toroidal_idx = np.where(toroidal_mask)[0]
    if len(toroidal_idx) > 0:
        theta_t = np.arctan2(points[toroidal_idx, 1], points[toroidal_idx, 0])
        toroidal_cycle = toroidal_idx[np.argsort(theta_t)]
    else:
        toroidal_cycle = np.array([], dtype=np.int32)
    
    return poloidal_cycle, toroidal_cycle

def compute_circulation(cycle, velocity, points):
    if len(cycle) < 2:
        return 0.0
    circ = 0.0
    for i in range(len(cycle)):
        a, b = cycle[i], cycle[(i+1) % len(cycle)]
        dl = points[b] - points[a]
        v_avg = 0.5 * (velocity[a] + velocity[b])
        circ += np.dot(v_avg, dl)
    return float(circ)

def compute_cycle_integral(cycle, u, M_lumped):
    if len(cycle) < 2:
        return 0.0
    weights = M_lumped[cycle]
    return float(np.dot(weights, u[cycle]) / (np.sum(weights) + 1e-12))

@njit(cache=True)
def compute_conservation_metrics_numba(trajectory, M_lumped):
    n_steps = trajectory.shape[0]
    n_nodes = trajectory.shape[1]
    total_area = np.sum(M_lumped)
    
    mass = np.zeros(n_steps, dtype=np.float64)
    energy = np.zeros(n_steps, dtype=np.float64)
    
    for t in range(n_steps):
        for i in range(n_nodes):
            mass[t] += M_lumped[i] * trajectory[t, i]
            energy[t] += M_lumped[i] * trajectory[t, i] * trajectory[t, i]
    
    mean = mass / total_area
    variance = energy / total_area - mean * mean
    
    return mass, mean, energy, variance

def compute_enstrophy(u, S_matrix, M_lumped):
    grad_energy = u @ S_matrix @ u
    return float(grad_energy)

def compute_entropy(u, M_lumped, eps=1e-10):
    u_shifted = u - np.min(u) + eps
    u_normalized = u_shifted / (np.sum(M_lumped * u_shifted) + eps)
    entropy = -np.sum(M_lumped * u_normalized * np.log(u_normalized + eps))
    return float(entropy)

def compute_higher_moments(u, M_lumped):
    total_area = np.sum(M_lumped)
    mean = np.dot(M_lumped, u) / total_area
    u_centered = u - mean
    var = np.dot(M_lumped, u_centered**2) / total_area
    std = np.sqrt(var + 1e-12)
    u_std = u_centered / (std + 1e-12)
    skewness = np.dot(M_lumped, u_std**3) / total_area
    kurtosis = np.dot(M_lumped, u_std**4) / total_area
    return float(skewness), float(kurtosis)

class AdvectionDiffusionSolver:
    def __init__(self, points, faces, normals, diffusivity=0.01, dt=0.02):
        self.points = np.ascontiguousarray(points.astype(np.float64))
        self.faces = np.ascontiguousarray(faces.astype(np.int32))
        self.normals = np.ascontiguousarray(normals.astype(np.float64))
        self.n_nodes = len(points)
        self.nu = diffusivity
        self.dt = dt
        
        print(f"\n[Solver] Initializing...")
        self._build_fem_operators()
        self._compute_background_velocity()
        self._setup_implicit_solver()
        
        print("         Compiling Numba kernels...")
        _ = compute_advection_flux_numba(
            np.zeros(self.n_nodes, dtype=np.float64),
            self.points, self.faces, self.velocity_f64,
            self.normals, self.M_lumped_inv_f64
        )
        print(f"[Solver] Ready.")
    
    def _build_fem_operators(self):
        print("         Building FEM operators...")
        n = self.n_nodes
        M_data, M_rows, M_cols = [], [], []
        S_data, S_rows, S_cols = [], [], []
        
        for face in self.faces:
            i, j, k = face
            vi, vj, vk = self.points[i], self.points[j], self.points[k]
            
            e1, e2, e3 = vj - vi, vk - vi, vk - vj
            area = 0.5 * np.linalg.norm(np.cross(e1, e2))
            if area < 1e-12: continue
            
            mass_contrib = area / 3.0
            for idx in [i, j, k]:
                M_rows.append(idx)
                M_cols.append(idx)
                M_data.append(mass_contrib)
            
            cross_norm = 2 * area
            cot_k = np.dot(e1, e2) / (cross_norm + 1e-12)
            cot_i = np.dot(-e1, e3) / (cross_norm + 1e-12)
            cot_j = np.dot(-e2, -e3) / (cross_norm + 1e-12)
            
            for a, b, w in [(i, j, cot_k), (j, k, cot_i), (k, i, cot_j)]:
                S_rows.extend([a, b, a, b])
                S_cols.extend([b, a, a, b])
                S_data.extend([-w, -w, w, w])
        
        self.M = csr_matrix((M_data, (M_rows, M_cols)), shape=(n, n))
        self.S = csr_matrix((S_data, (S_rows, S_cols)), shape=(n, n))
        
        self.M_lumped = np.array(self.M.sum(axis=1)).flatten()
        self.M_lumped[self.M_lumped < 1e-12] = 1e-12
        self.M_lumped_inv = 1.0 / self.M_lumped
        self.M_lumped_inv_f64 = np.ascontiguousarray(self.M_lumped_inv.astype(np.float64))

    def _compute_background_velocity(self):
        t_u, t_v, theta = compute_torus_tangent_basis(
            self.points.astype(np.float32), self.normals.astype(np.float32), cfg.TORUS_MAJOR_RADIUS
        )
        omega = cfg.ADVECTION_SPEED
        v_rot = np.stack([-omega * self.points[:, 1], omega * self.points[:, 0], 
                          np.zeros(self.n_nodes)], axis=1)
        normal_comp = np.sum(v_rot * self.normals, axis=1, keepdims=True)
        v_tangent = v_rot - normal_comp * self.normals
        self.velocity = v_tangent.astype(np.float32)
        self.velocity_f64 = np.ascontiguousarray(v_tangent.astype(np.float64))

    def _setup_implicit_solver(self):
        self.implicit_mat = self.M + self.nu * self.dt * self.S
        self.implicit_solver = splu(self.implicit_mat.tocsc())

    def step(self, u):
        u_f64 = np.ascontiguousarray(u.astype(np.float64))
        adv_flux = compute_advection_flux_numba(
            u_f64, self.points, self.faces, self.velocity_f64, self.normals, self.M_lumped_inv_f64
        )
        u_star = u - self.dt * adv_flux
        
        rhs = self.M @ u_star
        return self.implicit_solver.solve(rhs).astype(np.float32)

    def generate_initial_condition(self, rng, n_blobs_min=2, n_blobs_max=5, sigma=0.15, harmonic_bias=0.0):
        n_blobs = rng.integers(n_blobs_min, n_blobs_max + 1)
        u = np.zeros(self.n_nodes, dtype=np.float32)
        pts = self.points.astype(np.float32)
        for _ in range(n_blobs):
            center_idx = rng.integers(0, self.n_nodes)
            amplitude = rng.uniform(0.5, 2.0)
            u += generate_gaussian_blob(pts, center_idx, sigma, amplitude)
            
        u += harmonic_bias
        
        return u

    def generate_trajectory(self, u0, n_steps):
        trajectory = [u0.copy()]
        u = u0.copy()
        for _ in range(n_steps):
            u = self.step(u)
            trajectory.append(u.copy())
        return np.array(trajectory, dtype=np.float32)

class PostProcessor:
    def __init__(self, solver, points, major_radius, minor_radius):
        self.solver = solver
        self.points = points.astype(np.float64)
        self.M_lumped = solver.M_lumped
        self.S = solver.S
        self.total_area = np.sum(self.M_lumped)
        
        self.poloidal_cycle, self.toroidal_cycle = build_homology_cycles(
            points, major_radius, minor_radius
        )
        
        self.Gamma_poloidal = compute_circulation(
            self.poloidal_cycle, solver.velocity, self.points
        )
        self.Gamma_toroidal = compute_circulation(
            self.toroidal_cycle, solver.velocity, self.points
        )
    
    def process_trajectory(self, trajectory):
        mass, mean, energy, variance = compute_conservation_metrics_numba(
            trajectory, self.M_lumped
        )
        
        n_steps = len(trajectory)
        enstrophy = np.zeros(n_steps, dtype=np.float32)
        entropy = np.zeros(n_steps, dtype=np.float32)
        skewness = np.zeros(n_steps, dtype=np.float32)
        kurtosis = np.zeros(n_steps, dtype=np.float32)
        poloidal_mode = np.zeros(n_steps, dtype=np.float32)
        toroidal_mode = np.zeros(n_steps, dtype=np.float32)
        
        for t in range(n_steps):
            u = trajectory[t]
            enstrophy[t] = compute_enstrophy(u, self.S, self.M_lumped)
            entropy[t] = compute_entropy(u, self.M_lumped)
            skewness[t], kurtosis[t] = compute_higher_moments(u, self.M_lumped)
            poloidal_mode[t] = compute_cycle_integral(self.poloidal_cycle, u, self.M_lumped)
            toroidal_mode[t] = compute_cycle_integral(self.toroidal_cycle, u, self.M_lumped)
        
        mass_init = mass[0]
        mass_error = np.abs(mass - mass_init) / (np.abs(mass_init) + 1e-12)
        
        return {
            'mass': mass.astype(np.float32),
            'mean': mean.astype(np.float32),
            'energy': energy.astype(np.float32),
            'variance': variance.astype(np.float32),
            'enstrophy': enstrophy,
            'entropy': entropy,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'poloidal_mode': poloidal_mode,
            'toroidal_mode': toroidal_mode,
            'mass_error': mass_error.astype(np.float32),
            'mass_conservation_ratio': float(mass[-1] / (mass[0] + 1e-12))
        }
    
    def compute_batch_statistics(self, all_diagnostics):
        n_samples = len(all_diagnostics)
        
        final_mass_errors = np.array([d['mass_error'][-1] for d in all_diagnostics])
        mass_conservation_ratios = np.array([d['mass_conservation_ratio'] for d in all_diagnostics])
        
        energy_decay = np.array([
            (d['energy'][0] - d['energy'][-1]) / (d['energy'][0] + 1e-12) 
            for d in all_diagnostics
        ])
        
        variance_decay = np.array([
            (d['variance'][0] - d['variance'][-1]) / (d['variance'][0] + 1e-12) 
            for d in all_diagnostics
        ])
        
        return {
            'mean_mass_error': float(np.mean(final_mass_errors)),
            'max_mass_error': float(np.max(final_mass_errors)),
            'std_mass_error': float(np.std(final_mass_errors)),
            'mean_mass_conservation_ratio': float(np.mean(mass_conservation_ratios)),
            'mean_energy_decay': float(np.mean(energy_decay)),
            'std_energy_decay': float(np.std(energy_decay)),
            'mean_variance_decay': float(np.mean(variance_decay)),
            'Gamma_poloidal': self.Gamma_poloidal,
            'Gamma_toroidal': self.Gamma_toroidal,
            'n_poloidal_nodes': len(self.poloidal_cycle),
            'n_toroidal_nodes': len(self.toroidal_cycle)
        }

def main():
    print("\n" + "="*70)
    print("TORUS ADVECTION-DIFFUSION DATASET (ONE-TO-ONE MAPPING)")
    print("="*70)
    
    pts, faces, normals, mesh_pv = generate_torus_mesh(
        cfg.TORUS_MAJOR_RADIUS, cfg.TORUS_MINOR_RADIUS, 
        cfg.TORUS_U_RES, cfg.TORUS_V_RES, cfg.TARGET_NODES
    )
    
    pickle_path = os.path.join(cfg.OUTPUT_DIR, cfg.PICKLE_FILE)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    solver = AdvectionDiffusionSolver(pts, faces, normals, cfg.DIFFUSIVITY, cfg.DT)
    
    postprocessor = PostProcessor(
        solver, pts, cfg.TORUS_MAJOR_RADIUS, cfg.TORUS_MINOR_RADIUS
    )
    
    print(f"\n[Topology] Velocity field circulations:")
    print(f"          Gamma_poloidal = {postprocessor.Gamma_poloidal:.6f}")
    print(f"          Gamma_toroidal = {postprocessor.Gamma_toroidal:.6f}")
    
    print(f"\n[Data Generation] Starting generation of {cfg.N_SAMPLES} trajectories...")
    rng = np.random.default_rng(42)
    
    trajectories = []
    harmonic_coeffs = [] 
    all_diagnostics = []
    
    t0 = time.time()
    for i in range(cfg.N_SAMPLES):
        c_harm = rng.uniform(cfg.HARMONIC_BIAS_MIN, cfg.HARMONIC_BIAS_MAX)
        
        u0 = solver.generate_initial_condition(
            rng, 
            cfg.N_BLOBS_MIN, 
            cfg.N_BLOBS_MAX, 
            cfg.BLOB_SIGMA, 
            harmonic_bias=c_harm
        )
        
        traj = solver.generate_trajectory(u0, cfg.N_TIMESTEPS)
        
        diagnostics = postprocessor.process_trajectory(traj)
        
        trajectories.append(traj)
        harmonic_coeffs.append(c_harm)
        all_diagnostics.append(diagnostics)
        
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            speed = (i + 1) / elapsed
            print(f"       [{i+1}/{cfg.N_SAMPLES}] Speed: {speed:.1f} traj/s | Elapsed: {elapsed:.1f}s")
            
    total_time = time.time() - t0
    print(f"\n[Data Generation] Completed in {total_time:.2f}s")
    
    print(f"\n[Post-Processing] Computing conservation and topological statistics...")
    batch_stats = postprocessor.compute_batch_statistics(all_diagnostics)
    
    print(f"\n[Conservation Summary]")
    print(f"          Mean Mass Error: {batch_stats['mean_mass_error']:.2e}")
    print(f"          Max Mass Error:  {batch_stats['max_mass_error']:.2e}")
    print(f"          Mean Mass Ratio: {batch_stats['mean_mass_conservation_ratio']:.6f}")
    print(f"          Mean Energy Decay: {batch_stats['mean_energy_decay']:.4f}")
    print(f"          Mean Variance Decay: {batch_stats['mean_variance_decay']:.4f}")
    
    print(f"\n[Topological Invariants]")
    print(f"          Poloidal Circulation: {batch_stats['Gamma_poloidal']:.6f}")
    print(f"          Toroidal Circulation: {batch_stats['Gamma_toroidal']:.6f}")
    print(f"          Poloidal Cycle Nodes: {batch_stats['n_poloidal_nodes']}")
    print(f"          Toroidal Cycle Nodes: {batch_stats['n_toroidal_nodes']}")
    
    data = {
        'trajectories': trajectories, 
        'points': pts, 
        'faces': faces, 
        'normals': normals,
        'harmonic_coeffs': np.array(harmonic_coeffs, dtype=np.float32),
        'diagnostics': all_diagnostics,
        'conservation_stats': batch_stats,
        'topological_invariants': {
            'Gamma_poloidal': postprocessor.Gamma_poloidal,
            'Gamma_toroidal': postprocessor.Gamma_toroidal,
            'poloidal_cycle': postprocessor.poloidal_cycle,
            'toroidal_cycle': postprocessor.toroidal_cycle
        }
    }
    
    print(f"\n[Saving] Writing to {pickle_path}...")
    with open(pickle_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"[Finished] Dataset saved.")
    print(f"           Traj Shape: {trajectories[0].shape}")
    print(f"           Note: Harmonic component is embedded in u0 as background level.")

if __name__ == "__main__":
    main()