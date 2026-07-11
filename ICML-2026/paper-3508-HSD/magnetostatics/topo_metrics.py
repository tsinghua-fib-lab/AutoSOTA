import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


class FluxTopologyEvaluator:
    
    def __init__(self, host_ops, points, simplices, mapper, device='cpu', mesh_type='auto'):
        self.host = host_ops
        self.mapper = mapper
        self.device = device
        self.n_nodes = len(points)
        self.n_edges = host_ops.n_edges
        
        self.points = torch.from_numpy(points.astype(np.float32)).to(device)
        self.simplices = simplices
        self.simplices_tensor = torch.from_numpy(simplices.astype(np.int64)).to(device)
        
        if mesh_type == 'auto':
            if simplices.shape[1] == 3:
                mesh_type = 'surface'
            elif simplices.shape[1] == 4:
                mesh_type = 'volume'
            else:
                raise ValueError(f"Unknown simplex dimension: {simplices.shape[1]}")
        
        self.mesh_type = mesh_type
        print(f"[Evaluator] Detected mesh type: {mesh_type}")
        
        if mesh_type == 'surface':
            self.n_faces = len(simplices)
            self.n_cells = len(simplices)
            self.faces_list = [tuple(s) for s in simplices]
            self.face_to_idx = {f: i for i, f in enumerate(self.faces_list)}
        else:
            self.n_cells = len(simplices)
            self._extract_faces_from_tets()
        
        self.Phi0 = torch.from_numpy(host_ops.Phi0.astype(np.float32)).to(device)
        self.Phi1 = torch.from_numpy(host_ops.Phi1.astype(np.float32)).to(device)
        
        self.div_op = torch.from_numpy(host_ops.B1.T.toarray().astype(np.float32)).to(device)
        
        self._setup_curl_operator()
        self._compute_geometry()
        self._build_node_adjacency()
        self._build_face_adjacency()
        self._compute_eigenvalues()
    
    def _extract_faces_from_tets(self):
        face_set = set()
        
        for tet in self.simplices:
            faces_of_tet = [
                tuple(sorted([tet[0], tet[1], tet[2]])),
                tuple(sorted([tet[0], tet[1], tet[3]])),
                tuple(sorted([tet[0], tet[2], tet[3]])),
                tuple(sorted([tet[1], tet[2], tet[3]])),
            ]
            for f in faces_of_tet:
                face_set.add(f)
        
        self.faces_list = list(face_set)
        self.face_to_idx = {f: i for i, f in enumerate(self.faces_list)}
        self.n_faces = len(self.faces_list)
        
        print(f"[Evaluator] Extracted {self.n_faces} faces from {self.n_cells} tetrahedra")
    
    def _setup_curl_operator(self):
        if hasattr(self.host, 'B2') and self.host.B2 is not None:
            B2_dense = self.host.B2.toarray().astype(np.float32)
            self.curl_op = torch.from_numpy(B2_dense).to(self.device)
        else:
            self.curl_op = self._build_curl_operator_from_simplices()
        
        print(f"[Evaluator] Curl operator: {self.curl_op.shape}")
    
    def _build_curl_operator_from_simplices(self):
        edge_to_idx = {}
        idx = 0
        
        if self.mesh_type == 'surface':
            for simplex in self.simplices:
                for i in range(3):
                    v0, v1 = simplex[i], simplex[(i+1) % 3]
                    edge = tuple(sorted([v0, v1]))
                    if edge not in edge_to_idx:
                        edge_to_idx[edge] = idx
                        idx += 1
        else:
            for tet in self.simplices:
                for i in range(4):
                    for j in range(i+1, 4):
                        edge = tuple(sorted([tet[i], tet[j]]))
                        if edge not in edge_to_idx:
                            edge_to_idx[edge] = idx
                            idx += 1
        
        n_edges = self.n_edges
        curl_data = np.zeros((self.n_faces, n_edges), dtype=np.float32)
        
        for f_idx, face in enumerate(self.faces_list):
            v0, v1, v2 = face[0], face[1], face[2]
            edges_of_face = [
                (tuple(sorted([v0, v1])), v0 < v1),
                (tuple(sorted([v1, v2])), v1 < v2),
                (tuple(sorted([v2, v0])), v2 < v0),
            ]
            for edge, positive in edges_of_face:
                e_idx = edge_to_idx.get(edge, -1)
                if e_idx >= 0 and e_idx < n_edges:
                    sign = 1.0 if positive else -1.0
                    curl_data[f_idx, e_idx] = sign
        
        return torch.from_numpy(curl_data).to(self.device)
    
    def _compute_geometry(self):
        pts = self.points.cpu().numpy()
        
        node_areas = np.zeros(self.n_nodes, dtype=np.float32)
        face_areas = np.zeros(self.n_faces, dtype=np.float32)
        
        for f_idx, face in enumerate(self.faces_list):
            i, j, k = face[0], face[1], face[2]
            vi, vj, vk = pts[i], pts[j], pts[k]
            area = 0.5 * np.linalg.norm(np.cross(vj - vi, vk - vi))
            face_areas[f_idx] = area
        
        if self.mesh_type == 'surface':
            for f_idx, face in enumerate(self.faces_list):
                i, j, k = face[0], face[1], face[2]
                area = face_areas[f_idx]
                node_areas[i] += area / 3
                node_areas[j] += area / 3
                node_areas[k] += area / 3
            self.cell_volumes = torch.from_numpy(face_areas).to(self.device)
        else:
            cell_volumes = np.zeros(self.n_cells, dtype=np.float32)
            for c_idx, tet in enumerate(self.simplices):
                i, j, k, l = tet[0], tet[1], tet[2], tet[3]
                vi, vj, vk, vl = pts[i], pts[j], pts[k], pts[l]
                vol = abs(np.dot(vj - vi, np.cross(vk - vi, vl - vi))) / 6.0
                cell_volumes[c_idx] = vol
                node_areas[i] += vol / 4
                node_areas[j] += vol / 4
                node_areas[k] += vol / 4
                node_areas[l] += vol / 4
            self.cell_volumes = torch.from_numpy(cell_volumes).to(self.device)
        
        self.face_areas = torch.from_numpy(face_areas).to(self.device)
        self.node_areas = torch.from_numpy(node_areas).to(self.device)
        self.total_area = float(self.node_areas.sum())
        self.mass_weights = self.node_areas / self.node_areas.sum()
    
    def _build_node_adjacency(self):
        rows, cols = [], []
        
        if self.mesh_type == 'surface':
            for simplex in self.simplices:
                for i in range(3):
                    v1, v2 = simplex[i], simplex[(i+1) % 3]
                    rows.extend([v1, v2])
                    cols.extend([v2, v1])
        else:
            for tet in self.simplices:
                for i in range(4):
                    for j in range(i+1, 4):
                        v1, v2 = tet[i], tet[j]
                        rows.extend([v1, v2])
                        cols.extend([v2, v1])
        
        data = np.ones(len(rows), dtype=np.int8)
        self.node_adj_matrix = csr_matrix((data, (rows, cols)), shape=(self.n_nodes, self.n_nodes))
    
    def _build_face_adjacency(self):
        edge_to_faces = {}
        for f_idx, face in enumerate(self.faces_list):
            v0, v1, v2 = face[0], face[1], face[2]
            edges = [
                tuple(sorted([v0, v1])),
                tuple(sorted([v1, v2])),
                tuple(sorted([v2, v0])),
            ]
            for e in edges:
                if e not in edge_to_faces:
                    edge_to_faces[e] = []
                edge_to_faces[e].append(f_idx)
        
        rows, cols = [], []
        for faces_list in edge_to_faces.values():
            if len(faces_list) == 2:
                f1, f2 = faces_list
                rows.extend([f1, f2])
                cols.extend([f2, f1])
        
        if len(rows) > 0:
            data = np.ones(len(rows), dtype=np.int8)
            self.face_adj_matrix = csr_matrix((data, (rows, cols)), shape=(self.n_faces, self.n_faces))
        else:
            self.face_adj_matrix = csr_matrix((self.n_faces, self.n_faces), dtype=np.int8)
        
        print(f"[Evaluator] Face adjacency: {self.face_adj_matrix.nnz} connections")
    
    def _compute_eigenvalues(self):
        k0 = self.Phi0.shape[1]
        L0_np = self.host.L0.toarray().astype(np.float64)
        Phi0_np = self.host.Phi0[:, :k0].astype(np.float64)
        
        eigenvalues = np.zeros(k0, dtype=np.float64)
        for i in range(k0):
            phi_i = Phi0_np[:, i]
            norm_sq = np.dot(phi_i, phi_i)
            if norm_sq > 1e-12:
                eigenvalues[i] = np.dot(phi_i, L0_np @ phi_i) / norm_sq
        
        eigenvalues = np.maximum(eigenvalues, 0.0)
        self.eigenvalues = torch.from_numpy(eigenvalues.astype(np.float32)).to(self.device)
    
    def _to_tensor(self, x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x.astype(np.float32)).to(self.device)
        return x.to(self.device)
    
    def _ensure_batch(self, x):
        x = self._to_tensor(x)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return x
    
    def flux_to_velocity_magnitude(self, flux, batch=True):
        if isinstance(flux, torch.Tensor):
            flux_np = flux.detach().cpu().numpy()
        else:
            flux_np = flux
        
        if batch and flux_np.ndim == 2:
            B = flux_np.shape[0]
            magnitudes = []
            for i in range(B):
                vec = self.mapper.edge_flux_to_node_vector(flux_np[i])
                mag = np.linalg.norm(vec, axis=1)
                magnitudes.append(mag)
            return torch.tensor(np.array(magnitudes)).float().to(self.device)
        else:
            vec = self.mapper.edge_flux_to_node_vector(flux_np)
            mag = np.linalg.norm(vec, axis=1)
            return torch.tensor(mag).float().to(self.device)
    
    def flux_to_vorticity(self, flux):
        flux = self._ensure_batch(flux)
        circulation = torch.matmul(flux, self.curl_op.T)
        vorticity = circulation / (self.face_areas.unsqueeze(0) + 1e-10)
        return vorticity
    
    def vorticity_to_nodes(self, vorticity):
        B = vorticity.shape[0]
        node_vort = torch.zeros(B, self.n_nodes, device=self.device)
        node_weights = torch.zeros(B, self.n_nodes, device=self.device)
        
        faces_tensor = torch.tensor(self.faces_list, dtype=torch.int64, device=self.device)
        for local_idx in range(3):
            v_idx = faces_tensor[:, local_idx]
            weighted_vort = vorticity * self.face_areas.unsqueeze(0)
            weights = self.face_areas.unsqueeze(0).expand(B, -1)
            idx_expanded = v_idx.unsqueeze(0).expand(B, -1)
            node_vort.scatter_add_(1, idx_expanded, weighted_vort)
            node_weights.scatter_add_(1, idx_expanded, weights)
        
        node_vort = node_vort / (node_weights + 1e-10)
        return node_vort
    
    def compute_flux_mse(self, flux_pred, flux_gt):
        flux_pred = self._ensure_batch(flux_pred)
        flux_gt = self._ensure_batch(flux_gt)
        mse = ((flux_pred - flux_gt) ** 2).mean(dim=-1)
        return mse.mean().item()
    
    def compute_divergence_error(self, flux_pred, flux_gt):
        flux_pred = self._ensure_batch(flux_pred)
        flux_gt = self._ensure_batch(flux_gt)
        div_pred = torch.matmul(flux_pred, self.div_op.T)
        div_gt = torch.matmul(flux_gt, self.div_op.T)
        error = ((div_pred - div_gt) ** 2).mean(dim=-1)
        return error.mean().item()
    
    def compute_divergence_fidelity(self, flux_pred, flux_gt):
        error = self.compute_divergence_error(flux_pred, flux_gt)
        fidelity = np.exp(-10.0 * error)
        return min(max(fidelity, 0.0), 1.0)
    
    def compute_curl_mse(self, flux_pred, flux_gt):
        vort_pred = self.flux_to_vorticity(flux_pred)
        vort_gt = self.flux_to_vorticity(flux_gt)
        area_weights = self.face_areas / self.face_areas.sum()
        mse = ((vort_pred - vort_gt) ** 2 * area_weights.unsqueeze(0)).sum(dim=-1)
        return mse.mean().item()
    
    def compute_vorticity_fidelity(self, flux_pred, flux_gt):
        vort_pred = self.flux_to_vorticity(flux_pred)
        vort_gt = self.flux_to_vorticity(flux_gt)
        weights = self.face_areas / self.face_areas.sum()
        dot = (weights * vort_pred * vort_gt).sum(dim=-1)
        norm_pred = torch.sqrt((weights * vort_pred ** 2).sum(dim=-1) + 1e-10)
        norm_gt = torch.sqrt((weights * vort_gt ** 2).sum(dim=-1) + 1e-10)
        correlation = dot / (norm_pred * norm_gt + 1e-10)
        fidelity = (1.0 + correlation) / 2.0
        return fidelity.mean().item()
    
    def compute_spectral_fidelity(self, flux_pred, flux_gt, n_modes=20):
        mag_pred = self.flux_to_velocity_magnitude(flux_pred)
        mag_gt = self.flux_to_velocity_magnitude(flux_gt)
        n_modes = min(n_modes, self.Phi0.shape[1])
        Phi = self.Phi0[:, :n_modes]
        c_pred = mag_pred @ (Phi * self.node_areas.unsqueeze(1))
        c_gt = mag_gt @ (Phi * self.node_areas.unsqueeze(1))
        w = 1.0 / (self.eigenvalues[:n_modes] + 0.1)
        w = w / w.sum()
        error = torch.sqrt((w * (c_pred - c_gt)**2).sum(dim=-1) + 1e-10)
        energy = torch.sqrt((w * c_gt**2).sum(dim=-1) + 1e-10)
        rel_error = error / (energy + 1e-10)
        fidelity = torch.exp(-2.0 * rel_error)
        fidelity = torch.clamp(fidelity, min=0.0, max=1.0)
        return fidelity.mean().item()
    
    def compute_gradient_fidelity(self, flux_pred, flux_gt):
        mag_pred = self.flux_to_velocity_magnitude(flux_pred)
        mag_gt = self.flux_to_velocity_magnitude(flux_gt)
        B1 = torch.from_numpy(self.host.B1.toarray().astype(np.float32)).to(self.device)
        grad_pred = mag_pred @ B1.T
        grad_gt = mag_gt @ B1.T
        dot = (grad_pred * grad_gt).sum(dim=-1)
        norm_pred = torch.sqrt((grad_pred**2).sum(dim=-1) + 1e-10)
        norm_gt = torch.sqrt((grad_gt**2).sum(dim=-1) + 1e-10)
        cos_sim = dot / (norm_pred * norm_gt + 1e-10)
        fidelity = (1.0 + cos_sim) / 2.0
        return fidelity.mean().item()
    
    def compute_betti0_score(self, flux_pred, flux_gt, levels=(0.2, 0.5, 0.8)):
        vort_pred = self.flux_to_vorticity(flux_pred)
        vort_gt = self.flux_to_vorticity(flux_gt)
        vort_pred_abs = torch.abs(vort_pred)
        vort_gt_abs = torch.abs(vort_gt)
        
        def count_connected_components_faces(vort_field, threshold):
            vort_np = vort_field.cpu().numpy() if isinstance(vort_field, torch.Tensor) else vort_field
            mask = vort_np >= threshold
            active_faces = np.where(mask)[0]
            if len(active_faces) == 0:
                return 0
            subgraph = self.face_adj_matrix[active_faces][:, active_faces]
            n_components, _ = connected_components(subgraph, directed=False)
            return n_components
        
        all_scores = []
        B = vort_gt_abs.shape[0]
        
        for level in levels:
            for i in range(B):
                vmin = vort_gt_abs[i].min().item()
                vmax = vort_gt_abs[i].max().item()
                threshold = vmin + level * (vmax - vmin)
                n_gt = count_connected_components_faces(vort_gt_abs[i], threshold)
                n_pred = count_connected_components_faces(vort_pred_abs[i], threshold)
                if n_gt == 0 and n_pred == 0:
                    score = 1.0
                else:
                    diff = abs(n_pred - n_gt)
                    max_count = max(n_pred, n_gt, 1)
                    score = np.exp(-1.5 * diff / max_count)
                all_scores.append(score)
        
        return np.mean(all_scores)
    
    def compute_level_set_iou(self, flux_pred, flux_gt, n_levels=10):
        vort_pred = self.flux_to_vorticity(flux_pred)
        vort_gt = self.flux_to_vorticity(flux_gt)
        vort_pred_abs = torch.abs(vort_pred)
        vort_gt_abs = torch.abs(vort_gt)
        area_weights = self.face_areas / self.face_areas.sum()
        all_ious = []
        B = vort_gt_abs.shape[0]
        
        for i in range(B):
            vmin = vort_gt_abs[i].min().item()
            vmax = vort_gt_abs[i].max().item()
            levels_tensor = torch.linspace(vmin, vmax, n_levels + 2, device=self.device)[1:-1]
            sample_ious = []
            for level in levels_tensor:
                pred_mask = (vort_pred_abs[i] >= level).float()
                gt_mask = (vort_gt_abs[i] >= level).float()
                intersection = (area_weights * pred_mask * gt_mask).sum()
                union = (area_weights * torch.clamp(pred_mask + gt_mask, max=1.0)).sum()
                if union > 1e-10:
                    iou = (intersection / union).item()
                else:
                    iou = 1.0
                sample_ious.append(iou)
            all_ious.append(np.mean(sample_ious))
        
        return np.sqrt(np.mean(all_ious))
    
    def compute_vortex_count_accuracy(self, flux_pred, flux_gt, threshold_ratio=0.3):
        vort_pred = torch.abs(self.flux_to_vorticity(flux_pred))
        vort_gt = torch.abs(self.flux_to_vorticity(flux_gt))
        
        def count_vortex_cores(vort_field, threshold_ratio):
            vort_np = vort_field.cpu().numpy()
            threshold = vort_np.max() * threshold_ratio
            n_cores = 0
            for f_idx in range(self.n_faces):
                if vort_np[f_idx] < threshold:
                    continue
                neighbor_indices = self.face_adj_matrix[f_idx].nonzero()[1]
                if len(neighbor_indices) == 0:
                    is_max = True
                else:
                    is_max = all(vort_np[f_idx] >= vort_np[n_idx] for n_idx in neighbor_indices)
                if is_max:
                    n_cores += 1
            return n_cores
        
        B = vort_gt.shape[0]
        all_scores = []
        
        for i in range(B):
            n_gt = count_vortex_cores(vort_gt[i], threshold_ratio)
            n_pred = count_vortex_cores(vort_pred[i], threshold_ratio)
            if n_gt == 0 and n_pred == 0:
                score = 1.0
            elif n_gt == 0 or n_pred == 0:
                score = 0.0
            else:
                score = 1.0 - abs(n_pred - n_gt) / max(n_pred, n_gt)
            all_scores.append(max(score, 0.0))
        
        return np.mean(all_scores)
    
    def compute_energy_fidelity(self, flux_pred, flux_gt):
        flux_pred = self._ensure_batch(flux_pred)
        flux_gt = self._ensure_batch(flux_gt)
        E_pred = (flux_pred**2).sum(dim=-1)
        E_gt = (flux_gt**2).sum(dim=-1)
        ratio = E_pred / (E_gt + 1e-10)
        log_error = torch.abs(torch.log(ratio + 1e-10))
        fidelity = 1.0 / (1.0 + log_error)
        fidelity = torch.clamp(fidelity, min=0.0, max=1.0)
        return fidelity.mean().item()
    
    def compute_enstrophy_fidelity(self, flux_pred, flux_gt):
        vort_pred = self.flux_to_vorticity(flux_pred)
        vort_gt = self.flux_to_vorticity(flux_gt)
        area_weights = self.face_areas / self.face_areas.sum()
        enstrophy_pred = (area_weights * vort_pred**2).sum(dim=-1)
        enstrophy_gt = (area_weights * vort_gt**2).sum(dim=-1)
        ratio = enstrophy_pred / (enstrophy_gt + 1e-10)
        log_error = torch.abs(torch.log(ratio + 1e-10))
        fidelity = 1.0 / (1.0 + log_error)
        fidelity = torch.clamp(fidelity, min=0.0, max=1.0)
        return fidelity.mean().item()
    
    def get_spectral_coefficients(self, flux, n_modes=None):
        mag = self.flux_to_velocity_magnitude(flux)
        if n_modes is None:
            n_modes = self.Phi0.shape[1]
        n_modes = min(n_modes, self.Phi0.shape[1])
        Phi = self.Phi0[:, :n_modes]
        coeffs = mag @ (Phi * self.node_areas.unsqueeze(1))
        energies = coeffs ** 2
        return coeffs.cpu().numpy(), energies.cpu().numpy(), self.eigenvalues[:n_modes].cpu().numpy()
    
    def get_spectral_coefficients_from_scalar(self, scalar_field, n_modes=None):
        scalar = self._ensure_batch(scalar_field)
        if n_modes is None:
            n_modes = self.Phi0.shape[1]
        n_modes = min(n_modes, self.Phi0.shape[1])
        Phi = self.Phi0[:, :n_modes]
        coeffs = scalar @ (Phi * self.node_areas.unsqueeze(1))
        energies = coeffs ** 2
        return coeffs.cpu().numpy(), energies.cpu().numpy(), self.eigenvalues[:n_modes].cpu().numpy()
    
    def evaluate(self, flux_pred, flux_gt, compute_all=True):
        results = {}
        results['n_samples'] = flux_pred.shape[0] if hasattr(flux_pred, 'shape') else len(flux_pred)
        results['MSE'] = self.compute_flux_mse(flux_pred, flux_gt)
        results['divergence_fidelity'] = self.compute_divergence_fidelity(flux_pred, flux_gt)
        results['curl_mse'] = self.compute_curl_mse(flux_pred, flux_gt)
        results['vorticity_fidelity'] = self.compute_vorticity_fidelity(flux_pred, flux_gt)
        results['enstrophy_fidelity'] = self.compute_enstrophy_fidelity(flux_pred, flux_gt)
        results['gradient_fidelity'] = self.compute_gradient_fidelity(flux_pred, flux_gt)
        results['spectral_fidelity'] = self.compute_spectral_fidelity(flux_pred, flux_gt)
        results['energy_fidelity'] = self.compute_energy_fidelity(flux_pred, flux_gt)
        results['betti0_score'] = self.compute_betti0_score(flux_pred, flux_gt)
        results['level_set_iou'] = self.compute_level_set_iou(flux_pred, flux_gt)
        results['vortex_count_accuracy'] = self.compute_vortex_count_accuracy(flux_pred, flux_gt)
        return results
    
    def get_geometry_info(self):
        return {
            'n_nodes': self.n_nodes,
            'n_edges': self.n_edges,
            'n_faces': self.n_faces,
            'n_cells': self.n_cells,
            'mesh_type': self.mesh_type,
            'total_area': self.total_area,
        }


def evaluate_all_models(evaluator, predictions_dict, Y_test_flux, model_params=None, logger=None):
    def log(msg):
        if logger:
            logger.log(msg)
        else:
            print(msg)
    
    all_results = {}
    
    log("\n" + "="*80)
    log("FLUX FIELD TOPOLOGY & PHYSICS METRICS EVALUATION")
    log("Topology metrics now use VORTICITY instead of velocity magnitude")
    log("="*80)
    
    geo_info = evaluator.get_geometry_info()
    log("\n[Geometry]")
    log(f"  Mesh type: {geo_info['mesh_type']}")
    log(f"  Mesh: {geo_info['n_nodes']} nodes, {geo_info['n_edges']} edges, {geo_info['n_faces']} faces, {geo_info['n_cells']} cells")
    
    log("\n[Evaluating Models...]")
    for name, pred in predictions_dict.items():
        log(f"  Processing {name}...")
        results = evaluator.evaluate(pred, Y_test_flux)
        if model_params and name in model_params:
            results['n_params'] = model_params[name]
        all_results[name] = results
    
    _print_results_table(all_results, list(predictions_dict.keys()), log)
    
    return all_results


def _print_results_table(all_results, model_names, log_fn):
    log_fn("\n" + "="*100)
    log_fn("EVALUATION RESULTS SUMMARY")
    log_fn("="*100)
    
    log_fn("\n[Flux-Domain Metrics]")
    _print_metric_row(all_results, model_names, 'MSE', 'MSE', '{:.2e}', False, log_fn)
    _print_metric_row(all_results, model_names, 'divergence_fidelity', 'Div_Fid', '{:.4f}', True, log_fn)
    
    log_fn("\n[Physics Metrics (Vorticity)]")
    _print_metric_row(all_results, model_names, 'curl_mse', 'Curl_MSE', '{:.2e}', False, log_fn)
    _print_metric_row(all_results, model_names, 'vorticity_fidelity', 'Vort_Fid', '{:.4f}', True, log_fn)
    _print_metric_row(all_results, model_names, 'enstrophy_fidelity', 'Enst_Fid', '{:.4f}', True, log_fn)
    
    log_fn("\n[Topology Metrics (on Vorticity)")
    _print_metric_row(all_results, model_names, 'betti0_score', 'B0_Score', '{:.4f}', True, log_fn)
    _print_metric_row(all_results, model_names, 'level_set_iou', 'IoU', '{:.4f}', True, log_fn)
    _print_metric_row(all_results, model_names, 'vortex_count_accuracy', 'Vortex_Acc', '{:.4f}', True, log_fn)
    
    log_fn("\n[Spectral Metrics (on |u|)]")
    _print_metric_row(all_results, model_names, 'gradient_fidelity', 'Grad_Fid', '{:.4f}', True, log_fn)
    _print_metric_row(all_results, model_names, 'spectral_fidelity', 'Spec_Fid', '{:.4f}', True, log_fn)
    _print_metric_row(all_results, model_names, 'energy_fidelity', 'E_Fid', '{:.4f}', True, log_fn)


def _print_metric_row(all_results, model_names, key, label, fmt, higher_better, log_fn):
    values = {name: all_results[name].get(key, float('nan')) for name in model_names}
    valid_values = {k: v for k, v in values.items() if not np.isnan(v)}
    
    if valid_values:
        best_name = max(valid_values, key=valid_values.get) if higher_better else min(valid_values, key=valid_values.get)
    else:
        best_name = None
    
    row = f"  {label:<12}"
    for name in model_names:
        val = values[name]
        if np.isnan(val):
            row += f" {name}: N/A  |"
        else:
            formatted = fmt.format(val)
            if name == best_name:
                formatted = f"*{formatted}*"
            row += f" {name}: {formatted:<10} |"
    
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
            row = [model_name] + [results.get(m, '') for m in all_metrics]
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