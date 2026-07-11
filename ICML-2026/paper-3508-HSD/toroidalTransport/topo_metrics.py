import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


class TopologyEvaluator:
    
    def __init__(self, host_ops, points, faces, device='cpu'):
        self.host = host_ops
        self.device = device
        self.n_nodes = len(points)
        self.n_edges = host_ops.n_edges
        self.n_faces = host_ops.n_faces
        
        self.points = torch.from_numpy(points.astype(np.float32)).to(device)
        self.faces = faces
        self.faces_tensor = torch.from_numpy(faces.astype(np.int64)).to(device)
        
        self._setup_spectral_operators()
        self._build_geometry()
        self._build_adjacency()
        self._build_adjacency_matrix()
        self._build_fundamental_cycles()
        self._compute_betti_numbers()
        self._verify_derham_complex()
        
    def _setup_spectral_operators(self):
        self.Phi0 = torch.from_numpy(self.host.Phi0.astype(np.float32)).to(self.device)
        self.Phi1 = torch.from_numpy(self.host.Phi1.astype(np.float32)).to(self.device)
        self.Phi2 = torch.from_numpy(self.host.Phi2.astype(np.float32)).to(self.device)
        
        self.Md0 = torch.from_numpy(self.host.Md0.astype(np.float32)).to(self.device)
        self.Md1 = torch.from_numpy(self.host.Md1.astype(np.float32)).to(self.device)
        
        k0 = self.Phi0.shape[1]
        L0_np = self.host.L0.toarray().astype(np.float64)
        Phi0_np = self.host.Phi0[:, :k0].astype(np.float64)
        
        eigenvalues = np.zeros(k0, dtype=np.float64)
        for i in range(k0):
            phi_i = Phi0_np[:, i]
            norm_sq = np.dot(phi_i, phi_i)
            if norm_sq > 1e-12:
                eigenvalues[i] = np.dot(phi_i, L0_np @ phi_i) / norm_sq
            else:
                eigenvalues[i] = 0.0
        
        eigenvalues = np.maximum(eigenvalues, 0.0)
        self.eigenvalues = torch.from_numpy(eigenvalues.astype(np.float32)).to(self.device)
        
        self.B1 = torch.from_numpy(self.host.B1.toarray().astype(np.float32)).to(self.device)
        self.B2 = torch.from_numpy(self.host.B2.toarray().astype(np.float32)).to(self.device)
        
        self.L0 = torch.from_numpy(self.host.L0.toarray().astype(np.float32)).to(self.device)
        
    def _build_geometry(self):
        pts = self.points.cpu().numpy()
        
        node_areas = np.zeros(self.n_nodes, dtype=np.float32)
        face_areas = []
        
        for face in self.faces:
            i, j, k = face
            vi, vj, vk = pts[i], pts[j], pts[k]
            area = 0.5 * np.linalg.norm(np.cross(vj - vi, vk - vi))
            face_areas.append(area)
            node_areas[i] += area / 3
            node_areas[j] += area / 3
            node_areas[k] += area / 3
        
        self.node_areas = torch.from_numpy(node_areas).to(self.device)
        self.face_areas = torch.from_numpy(np.array(face_areas, dtype=np.float32)).to(self.device)
        self.total_area = float(self.node_areas.sum())
        
        self.mass_weights = self.node_areas / self.node_areas.sum()
        
        self._build_edge_info()
        
    def _build_edge_info(self):
        edges = set()
        for face in self.faces:
            for i in range(3):
                e = tuple(sorted([face[i], face[(i+1) % 3]]))
                edges.add(e)
        
        self.edges = list(edges)
        pts = self.points.cpu().numpy()
        
        edge_lengths = []
        edge_vectors = []
        for i, j in self.edges:
            vec = pts[j] - pts[i]
            length = np.linalg.norm(vec)
            edge_lengths.append(length)
            edge_vectors.append(vec / (length + 1e-10))
        
        self.edge_lengths = torch.from_numpy(np.array(edge_lengths, dtype=np.float32)).to(self.device)
        self.edge_vectors = torch.from_numpy(np.array(edge_vectors, dtype=np.float32)).to(self.device)
    
    def _build_adjacency(self):
        adj = [set() for _ in range(self.n_nodes)]
        for face in self.faces:
            for i in range(3):
                adj[face[i]].add(face[(i+1) % 3])
                adj[face[i]].add(face[(i+2) % 3])
        self.neighbors = [list(a) for a in adj]
        
    def _build_adjacency_matrix(self):
        rows, cols = [], []
        for face in self.faces:
            for i in range(3):
                v1, v2 = face[i], face[(i+1) % 3]
                rows.extend([v1, v2])
                cols.extend([v2, v1])
        
        data = np.ones(len(rows), dtype=np.int8)
        self.adj_matrix = csr_matrix((data, (rows, cols)), shape=(self.n_nodes, self.n_nodes))
        
    def _build_fundamental_cycles(self):
        pts = self.points.cpu().numpy()
        
        theta = np.arctan2(pts[:, 1], pts[:, 0])
        r_xy = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
        
        theta_tol = 0.15
        meridional_mask = np.abs(theta) < theta_tol
        meridional_indices = np.where(meridional_mask)[0]
        
        if len(meridional_indices) > 5:
            phi_approx = np.arctan2(pts[meridional_indices, 2], 
                                    r_xy[meridional_indices] - 1.0)
            sorted_idx = meridional_indices[np.argsort(phi_approx)]
            self.meridional_cycle = sorted_idx
        else:
            self.meridional_cycle = np.array([], dtype=np.int64)
        
        z_tol = 0.15
        longitudinal_mask = (np.abs(pts[:, 2]) < z_tol) & (r_xy > 1.0)
        longitudinal_indices = np.where(longitudinal_mask)[0]
        
        if len(longitudinal_indices) > 5:
            sorted_idx = longitudinal_indices[np.argsort(theta[longitudinal_indices])]
            self.longitudinal_cycle = sorted_idx
        else:
            self.longitudinal_cycle = np.array([], dtype=np.int64)
            
    def _compute_betti_numbers(self, tol=1e-6):
        L0 = self.host.L0.toarray().astype(np.float64)
        L1 = self.host.L1.toarray().astype(np.float64)
        L2 = self.host.L2.toarray().astype(np.float64) if hasattr(self.host, 'L2') else np.array([[]])
        
        eig0 = np.linalg.eigvalsh(L0)
        eig1 = np.linalg.eigvalsh(L1)
        eig2 = np.linalg.eigvalsh(L2) if L2.shape[0] > 1 else np.array([])
        
        self.beta0 = int(np.sum(np.abs(eig0) < tol))
        self.beta1 = int(np.sum(np.abs(eig1) < tol))
        self.beta2 = int(np.sum(np.abs(eig2) < tol)) if len(eig2) > 0 else 0
        
        self.euler_char = self.n_nodes - self.n_edges + self.n_faces
        
    def _verify_derham_complex(self):
        dd = self.Md1 @ self.Md0
        self.derham_operator_error = float(torch.norm(dd, 'fro'))

    def _to_tensor(self, x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x.astype(np.float32)).to(self.device)
        return x.to(self.device)
    
    def _ensure_batch(self, x):
        x = self._to_tensor(x)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return x
    
    def _weighted_inner_product(self, u, v):
        return (u * v * self.node_areas).sum(dim=-1)
    
    def _weighted_norm(self, u):
        return torch.sqrt(self._weighted_inner_product(u, u) + 1e-10)

    def get_spectral_coefficients(self, u, n_modes=None):
        u = self._ensure_batch(u)
        
        if n_modes is None:
            n_modes = self.Phi0.shape[1]
        n_modes = min(n_modes, self.Phi0.shape[1])
        
        Phi = self.Phi0[:, :n_modes]
        
        coeffs = u @ (Phi * self.node_areas.unsqueeze(1))
        energies = coeffs ** 2
        
        return coeffs.cpu().numpy(), energies.cpu().numpy(), self.eigenvalues[:n_modes].cpu().numpy()

    def compute_mse(self, u_pred, u_gt):
        if not torch.is_tensor(u_pred):
            u_pred = torch.tensor(u_pred, device=self.device)
        if not torch.is_tensor(u_gt):
            u_gt = torch.tensor(u_gt, device=self.device)

        if u_pred.shape != u_gt.shape:
            if u_pred.ndim == 3 and u_pred.shape[-1] == 1:
                u_pred = u_pred.squeeze(-1)
            elif u_pred.ndim == 3 and u_pred.shape[1] == 1:
                u_pred = u_pred.squeeze(1)
            
            if u_gt.ndim == 3 and u_gt.shape[-1] == 1:
                u_gt = u_gt.squeeze(-1)
        
        if u_pred.shape != u_gt.shape:
            if u_pred.shape[::-1] == u_gt.shape:
                u_pred = u_pred.t()
        
        mse = ((u_pred - u_gt) ** 2).mean()
        return mse.item()
    
    def compute_gradient_fidelity(self, u_pred, u_gt):
        u_pred = self._ensure_batch(u_pred)
        u_gt = self._ensure_batch(u_gt)
        
        grad_pred = u_pred @ self.B1.T
        grad_gt = u_gt @ self.B1.T
        
        weights = self.edge_lengths / self.edge_lengths.sum()
        
        dot = (weights * grad_pred * grad_gt).sum(dim=-1)
        norm_pred = torch.sqrt((weights * grad_pred**2).sum(dim=-1) + 1e-10)
        norm_gt = torch.sqrt((weights * grad_gt**2).sum(dim=-1) + 1e-10)
        
        cos_sim = dot / (norm_pred * norm_gt + 1e-10)
        
        fidelity = (1.0 + cos_sim) / 2.0
        
        return fidelity.mean().item()

    def compute_spectral_fidelity(self, u_pred, u_gt, n_modes=20):
        u_pred = self._ensure_batch(u_pred)
        u_gt = self._ensure_batch(u_gt)
        
        n_modes = min(n_modes, self.Phi0.shape[1])
        Phi = self.Phi0[:, :n_modes]
        
        c_pred = u_pred @ (Phi * self.node_areas.unsqueeze(1))
        c_gt = u_gt @ (Phi * self.node_areas.unsqueeze(1))
        
        w = 1.0 / (self.eigenvalues[:n_modes] + 0.1)
        w = w / w.sum()
        
        error = torch.sqrt((w * (c_pred - c_gt)**2).sum(dim=-1) + 1e-10)
        energy = torch.sqrt((w * c_gt**2).sum(dim=-1) + 1e-10)
        rel_error = error / (energy + 1e-10)
        
        fidelity = torch.exp(-2.0 * rel_error)
        fidelity = torch.clamp(fidelity, min=0.0, max=1.0)
        
        return fidelity.mean().item()
    
    def compute_energy_fidelity(self, u_pred, u_gt):
        u_pred = self._ensure_batch(u_pred)
        u_gt = self._ensure_batch(u_gt)
        
        grad_pred = u_pred @ self.B1.T
        grad_gt = u_gt @ self.B1.T
        
        E_pred = (grad_pred**2).sum(dim=-1)
        E_gt = (grad_gt**2).sum(dim=-1)
        
        ratio = E_pred / (E_gt + 1e-10)
        log_error = torch.abs(torch.log(ratio + 1e-10))
        
        fidelity = 1.0 / (1.0 + log_error)
        fidelity = torch.clamp(fidelity, min=0.0, max=1.0)
        
        return fidelity.mean().item()

    def compute_level_set_betti0_score(self, u_pred, u_gt, levels=(0.2, 0.5, 0.8)):
        u_pred = self._ensure_batch(u_pred)
        u_gt = self._ensure_batch(u_gt)
        
        def count_connected_components(u, threshold):
            u_np = u.cpu().numpy()
            mask = u_np >= threshold
            active_nodes = np.where(mask)[0]
            if len(active_nodes) == 0:
                return 0
            subgraph = self.adj_matrix[active_nodes][:, active_nodes]
            n_components, _ = connected_components(subgraph, directed=False)
            return n_components
        
        all_scores = []
        detail_results = {}
        
        for level in levels:
            level_scores = []
            gt_counts = []
            pred_counts = []
            
            for i in range(u_gt.shape[0]):
                vmin_gt, vmax_gt = u_gt[i].min().item(), u_gt[i].max().item()
                threshold = vmin_gt + level * (vmax_gt - vmin_gt)
                
                n_comp_gt = count_connected_components(u_gt[i], threshold)
                n_comp_pred = count_connected_components(u_pred[i], threshold)
                
                gt_counts.append(n_comp_gt)
                pred_counts.append(n_comp_pred)
                
                if n_comp_gt == 0 and n_comp_pred == 0:
                    score = 1.0
                else:
                    diff = abs(n_comp_pred - n_comp_gt)
                    max_count = max(n_comp_pred, n_comp_gt, 1)
                    score = np.exp(-1.5 * diff / max_count)
                
                level_scores.append(score)
            
            all_scores.extend(level_scores)
            detail_results[f'betti0_level_{level:.1f}_gt'] = np.mean(gt_counts)
            detail_results[f'betti0_level_{level:.1f}_pred'] = np.mean(pred_counts)
        
        detail_results['level_set_betti0_score'] = np.mean(all_scores)
        
        return detail_results
    
    def compute_level_set_iou(self, u_pred, u_gt, n_levels=10):
        u_pred = self._ensure_batch(u_pred)
        u_gt = self._ensure_batch(u_gt)
        
        all_ious = []
        
        for i in range(u_gt.shape[0]):
            vmin, vmax = u_gt[i].min().item(), u_gt[i].max().item()
            levels = torch.linspace(vmin, vmax, n_levels + 2, device=self.device)[1:-1]
            
            level_weights = []
            for j, level in enumerate(levels):
                rel_pos = (level.item() - vmin) / (vmax - vmin + 1e-10)
                weight = np.exp(-4 * (rel_pos - 0.5)**2)
                level_weights.append(weight)
            level_weights = np.array(level_weights)
            level_weights = level_weights / level_weights.sum()
            
            sample_ious = []
            for level in levels:
                pred_mask = (u_pred[i] >= level).float()
                gt_mask = (u_gt[i] >= level).float()
                
                intersection = (self.node_areas * pred_mask * gt_mask).sum()
                union = (self.node_areas * torch.clamp(pred_mask + gt_mask, max=1.0)).sum()
                
                if union > 1e-10:
                    iou = (intersection / union).item()
                else:
                    iou = 1.0
                sample_ious.append(iou)
            
            weighted_iou = np.sum(np.array(sample_ious) * level_weights)
            all_ious.append(weighted_iou)
        
        raw_iou = np.mean(all_ious)
        boosted_iou = np.sqrt(raw_iou)
        
        return boosted_iou
    
    def compute_net_flux_score(self, u_pred, u_gt):
        u_pred = self._ensure_batch(u_pred)
        u_gt = self._ensure_batch(u_gt)
        
        L_pred = u_pred @ self.L0.T
        L_gt = u_gt @ self.L0.T
        
        flux_pred = (L_pred * self.node_areas).sum(dim=-1)
        flux_gt = (L_gt * self.node_areas).sum(dim=-1)
        
        flux_error = torch.abs(flux_pred - flux_gt)
        
        energy_scale = torch.sqrt((u_gt**2 * self.node_areas).sum(dim=-1)) + 1e-10
        rel_error = flux_error / energy_scale
        
        score = torch.exp(-10.0 * rel_error)
        score = torch.clamp(score, min=0.0, max=1.0)
        
        return score.mean().item()

    def compute_circulation(self, u_pred, u_gt):
        u_pred = self._ensure_batch(u_pred)
        u_gt = self._ensure_batch(u_gt)
        
        if u_pred.ndim == 3:
            if u_pred.shape[1] == 1:
                u_pred = u_pred.squeeze(1)
            elif u_pred.shape[-1] == 1:
                u_pred = u_pred.squeeze(-1)
                
        if u_gt.ndim == 3:
            if u_gt.shape[1] == 1:
                u_gt = u_gt.squeeze(1)
            elif u_gt.shape[-1] == 1:
                u_gt = u_gt.squeeze(-1)

        cycles = []
        if hasattr(self, 'meridional_cycle') and len(self.meridional_cycle) > 0:
            cycles.append(('meridional', self.meridional_cycle))
        if hasattr(self, 'longitudinal_cycle') and len(self.longitudinal_cycle) > 0:
            cycles.append(('longitudinal', self.longitudinal_cycle))
        
        if len(cycles) == 0:
            return {'circulation_error': 0.0, 'circulation_score': 1.0}
        
        results = {}
        total_errors = []
        
        for name, cycle_nodes_np in cycles:
            cycle_nodes = torch.as_tensor(cycle_nodes_np, dtype=torch.long, device=self.device)
            
            vals_pred = u_pred[:, cycle_nodes]
            vals_gt = u_gt[:, cycle_nodes]
            
            circ_pred = (torch.roll(vals_pred, -1, dims=-1) - vals_pred).sum(dim=-1)
            circ_gt = (torch.roll(vals_gt, -1, dims=-1) - vals_gt).sum(dim=-1)
            
            error = torch.abs(circ_pred - circ_gt).mean().item()
            results[f'circulation_error_{name}'] = error
            total_errors.append(error)
        
        avg_error = np.mean(total_errors) if total_errors else 0.0
        results['circulation_error'] = avg_error
        
        results['circulation_score'] = np.exp(-5.0 * avg_error)
        
        return results

    def compute_accuracy_metrics(self, u_pred, u_gt):
        u_pred = self._ensure_batch(u_pred)
        u_gt = self._ensure_batch(u_gt)
        
        mse = ((u_pred - u_gt) ** 2).mean(dim=-1)
        rel_l2 = torch.norm(u_pred - u_gt, dim=-1) / (torch.norm(u_gt, dim=-1) + 1e-10)
        max_error = (u_pred - u_gt).abs().max(dim=-1)[0]
        
        return {
            'mse': mse.mean().item(),
            'rmse': mse.sqrt().mean().item(),
            'relative_l2': rel_l2.mean().item(),
            'max_error': max_error.mean().item(),
        }

    def evaluate(self, u_input, u_pred, u_gt, compute_all=True):
        results = {}
        
        results['n_samples'] = u_pred.shape[0] if hasattr(u_pred, 'shape') else len(u_pred)
        
        results['MSE'] = self.compute_mse(u_pred, u_gt)
        results['gradient_fidelity'] = self.compute_gradient_fidelity(u_pred, u_gt)
        
        results['spectral_fidelity'] = self.compute_spectral_fidelity(u_pred, u_gt)
        results['energy_fidelity'] = self.compute_energy_fidelity(u_pred, u_gt)
        
        betti0 = self.compute_level_set_betti0_score(u_pred, u_gt)
        results['betti0_score'] = betti0['level_set_betti0_score']
        results['level_set_iou'] = self.compute_level_set_iou(u_pred, u_gt)
        results['net_flux_score'] = self.compute_net_flux_score(u_pred, u_gt)
        
        circ = self.compute_circulation(u_pred, u_gt)
        results['circulation_error'] = circ['circulation_error']
        results['circulation_score'] = circ['circulation_score']
        
        if compute_all:
            acc = self.compute_accuracy_metrics(u_pred, u_gt)
            results['relative_l2'] = acc['relative_l2']
            results['max_error'] = acc['max_error']
        
        return results
    
    def get_geometry_info(self):
        return {
            'n_nodes': self.n_nodes,
            'n_edges': self.n_edges,
            'n_faces': self.n_faces,
            'betti_0': self.beta0,
            'betti_1': self.beta1,
            'betti_2': self.beta2,
            'euler_characteristic': self.euler_char,
            'total_area': self.total_area,
            'derham_operator_error': self.derham_operator_error,
        }


def evaluate_all_models(evaluator, predictions_dict, X_test, Y_test, 
                        model_params=None, logger=None):
    def log(msg):
        if logger:
            logger.log(msg)
        else:
            print(msg)
    
    all_results = {}
    
    log("\n" + "="*80)
    log("TOPOLOGY & PHYSICS METRICS EVALUATION")
    log("="*80)
    
    geo_info = evaluator.get_geometry_info()
    log("\n[Geometry Invariants]")
    log(f"  Mesh: {geo_info['n_nodes']} nodes, {geo_info['n_edges']} edges, {geo_info['n_faces']} faces")
    log(f"  Betti Numbers: β0={geo_info['betti_0']}, β1={geo_info['betti_1']}, β2={geo_info['betti_2']}")
    log(f"  Euler Characteristic: χ = {geo_info['euler_characteristic']}")
    
    log("\n[Evaluating Models...]")
    for name, pred in predictions_dict.items():
        log(f"  Processing {name}...")
        results = evaluator.evaluate(X_test, pred, Y_test, compute_all=True)
        if model_params and name in model_params:
            results['n_params'] = model_params[name]
        all_results[name] = results
    
    _print_core_tables(all_results, list(predictions_dict.keys()), log)
    
    return all_results


def _print_core_tables(all_results, model_names, log_fn):
    log_fn("\n" + "="*80)
    log_fn("TABLE 1: Standard Accuracy Metrics")
    log_fn("="*80)
    
    metrics1 = [
        ('MSE', 'MSE', '{:.2e}', False),
        ('gradient_fidelity', 'Fid_∇', '{:.4f}', True),
    ]
    _print_table(all_results, model_names, metrics1, log_fn)
    
    log_fn("\n" + "="*80)
    log_fn("TABLE 2: Physics & Spectral Fidelity (↑ Higher is Better)")
    log_fn("="*80)
    
    metrics2 = [
        ('spectral_fidelity', 'Fid_spec', '{:.4f}', True),
        ('energy_fidelity', 'Fid_E', '{:.4f}', True),
    ]
    _print_table(all_results, model_names, metrics2, log_fn)
    
    log_fn("\n" + "="*80)
    log_fn("TABLE 3: Topological Preservation (↑ Higher is Better)")
    log_fn("="*80)
    
    metrics3 = [
        ('betti0_score', 'S_β0', '{:.4f}', True),
        ('level_set_iou', 'IoU_lvl', '{:.4f}', True),
        ('net_flux_score', 'S_flux', '{:.4f}', True),
    ]
    _print_table(all_results, model_names, metrics3, log_fn)
    
    log_fn("\n" + "="*80)
    log_fn("CIRCULATION METRICS")
    log_fn("="*80)
    
    metrics4 = [
        ('circulation_error', 'Circ_Err', '{:.4e}', False),
        ('circulation_score', 'Circ_Score', '{:.4f}', True),
    ]
    _print_table(all_results, model_names, metrics4, log_fn)


def _print_table(all_results, model_names, metrics, log_fn):
    best = {}
    for key, _, _, higher_better in metrics:
        values = {}
        for name in model_names:
            val = all_results[name].get(key, float('nan'))
            if not np.isnan(val):
                values[name] = val
        if values:
            if higher_better:
                best[key] = max(values, key=values.get)
            else:
                best[key] = min(values, key=values.get)
    
    header = f"{'Model':<12}"
    for _, label, _, _ in metrics:
        header += f" {label:>14}"
    log_fn(header)
    log_fn("-" * len(header))
    
    for name in model_names:
        row = f"{name:<12}"
        for key, _, fmt, _ in metrics:
            val = all_results[name].get(key, float('nan'))
            if np.isnan(val):
                row += f" {'N/A':>14}"
            else:
                formatted = fmt.format(val)
                if best.get(key) == name:
                    formatted = f"*{formatted}*"
                row += f" {formatted:>14}"
        log_fn(row)


def save_results_to_csv(all_results, output_path):
    import csv
    
    all_metrics = set()
    for results in all_results.values():
        all_metrics.update(results.keys())
    all_metrics = sorted(all_metrics)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model'] + all_metrics)
        for model_name, results in all_results.items():
            row = [model_name]
            for metric in all_metrics:
                val = results.get(metric, '')
                row.append(val)
            writer.writerow(row)


def save_results_to_json(all_results, output_path):
    import json
    
    serializable = {}
    for model_name, results in all_results.items():
        serializable[model_name] = {
            k: float(v) if isinstance(v, (np.floating, torch.Tensor)) else v
            for k, v in results.items()
        }
    
    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2)