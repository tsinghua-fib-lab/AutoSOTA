"""
Dataset classes for Flux Field Prediction
Supports both triangular surface meshes and tetrahedral volume meshes
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import toponetx as tnx


class VectorFluxMapper:
    """
    Maps between node vectors and edge flux.
    Supports both triangular surface meshes and tetrahedral volume meshes.
    
    Used for:
    - Converting GT vectors to GT flux during dataset creation
    - Converting baseline model predictions (vectors) to flux for evaluation
    """
    
    def __init__(self, points, simplices, mesh_type='auto'):
        """
        Args:
            points: (N, 3) node coordinates
            simplices: Faces (N_faces, 3) or tetrahedra (N_tets, 4)
            mesh_type: 'auto', 'surface', or 'volume'
        """
        self.points = points
        self.simplices = simplices
        
        if mesh_type == 'auto':
            if simplices.shape[1] == 3:
                mesh_type = 'surface'
            elif simplices.shape[1] == 4:
                mesh_type = 'volume'
            else:
                raise ValueError(f"Unknown simplex dimension: {simplices.shape[1]}")
        
        self.mesh_type = mesh_type
        
        self.sc = tnx.SimplicialComplex(simplices)
        
        self.B1 = self.sc.incidence_matrix(rank=1, signed=True)
        if self.B1.shape[1] != len(points):
            self.B1 = self.B1.T
        
        self.edge_vectors = self.B1 @ self.points
        self.edge_lengths = np.linalg.norm(self.edge_vectors, axis=1, keepdims=True) + 1e-9
        self.edge_dirs = self.edge_vectors / self.edge_lengths
        
        self.node_degree = np.abs(self.B1).T @ np.ones((self.B1.shape[0], 1))
        self.node_degree[self.node_degree < 1] = 1.0
        
        self.n_edges = self.B1.shape[0]
        self.n_nodes = len(points)
    
    def node_vector_to_edge_flux(self, node_vectors):
        """Convert node vectors (N, 3) to edge flux (E,)."""
        avg_vel = (np.abs(self.B1) @ node_vectors) / 2.0
        flux = np.sum(avg_vel * self.edge_vectors, axis=1)
        return flux
    
    def edge_flux_to_node_vector(self, edge_flux):
        """Convert edge flux (E,) back to node vectors (N, 3)."""
        if edge_flux.ndim == 1:
            edge_flux = edge_flux[:, None]
        edge_contribs = edge_flux * self.edge_dirs
        node_accum = np.abs(self.B1).T @ edge_contribs
        return node_accum / self.node_degree


class FluxFieldDataset(Dataset):
    """
    Dataset for flux field (1-form) prediction.
    
    Uses DEC lifting to derive spectral coefficients:
    - c0: scalar input projected onto Φ0
    - c1: gradient of input projected onto Φ1
    - c2: curl (should be ~0 due to d²=0)
    
    Target: GT flux projected onto Φ1
    
    Supports both surface and volume meshes.
    """
    
    def __init__(self, host_ops, mapper, X_node_data, Y_node_data, k_list,
                 x_scale=None, y_scale=None, logger=None):
        super().__init__()
        self.k0, self.k1, self.k2 = k_list
        self.logger = logger
        self.mapper = mapper
        
        if x_scale is None:
            self.x_scale = np.max(np.abs(X_node_data)) + 1e-9
        else:
            self.x_scale = x_scale
            
        if y_scale is None:
            self.y_scale = np.max(np.abs(Y_node_data)) + 1e-9
        else:
            self.y_scale = y_scale
        
        self._log(f"\n[Dataset] Global Scaling Factors:")
        self._log(f"         X_scale (Input): {self.x_scale:.6f}")
        self._log(f"         Y_scale (Output): {self.y_scale:.6f}")
        
        X_norm = X_node_data / self.x_scale
        Y_norm = Y_node_data / self.y_scale
        
        self.X_norm = X_norm.astype(np.float32)
        self.Y_norm = Y_norm.astype(np.float32)
        
        self._log(f"         Normalized X: min={np.min(X_norm):.4f}, max={np.max(X_norm):.4f}")
        self._log(f"         Normalized Y: min={np.min(Y_norm):.4f}, max={np.max(Y_norm):.4f}")
        
        self._log("[Dataset] Lifting Input to Full de Rham Complex...")
        fX, gX, hX = host_ops.lift_signal(X_norm)
        self.c0_in = (fX @ host_ops.Phi0[:, :self.k0]).astype(np.float32)
        self.c1_in = (gX @ host_ops.Phi1[:, :self.k1]).astype(np.float32)
        self.c2_in = (hX @ host_ops.Phi2[:, :self.k2]).astype(np.float32)
        
        self._log("[Dataset] Computing GT Edge Flux...")
        flux_list = []
        for i in range(len(Y_norm)):
            flux = mapper.node_vector_to_edge_flux(Y_norm[i])
            flux_list.append(flux)
        self.gt_flux = np.array(flux_list).astype(np.float32)
        
        self._log("[Dataset] Projecting GT Flux to Spectral Coefficients...")
        self.c1_tgt = (self.gt_flux @ host_ops.Phi1[:, :self.k1]).astype(np.float32)
        
        self._log(f"\n[Dataset] Spectral Coefficient Statistics:")
        self._log(f"         c0_in: mean={np.mean(np.abs(self.c0_in)):.6f}, max={np.max(np.abs(self.c0_in)):.6f}")
        self._log(f"         c1_in: mean={np.mean(np.abs(self.c1_in)):.6f}, max={np.max(np.abs(self.c1_in)):.6f}")
        self._log(f"         c2_in: mean={np.mean(np.abs(self.c2_in)):.6f}, max={np.max(np.abs(self.c2_in)):.6f}")
        self._log(f"         c1_tgt: mean={np.mean(np.abs(self.c1_tgt)):.6f}, max={np.max(np.abs(self.c1_tgt)):.6f}")
        self._log(f"         gt_flux: mean={np.mean(np.abs(self.gt_flux)):.6f}, max={np.max(np.abs(self.gt_flux)):.6f}")
        
        self._log(f"\n[Dataset] Shapes: c0_in:{self.c0_in.shape}, c1_in:{self.c1_in.shape}, c2_in:{self.c2_in.shape}")
        self._log(f"         Target: c1_tgt:{self.c1_tgt.shape}, gt_flux:{self.gt_flux.shape}")

    def _log(self, message):
        if self.logger:
            self.logger.log(message)
        else:
            print(message)

    def __len__(self):
        return self.c0_in.shape[0]

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.c0_in[idx]),
            torch.from_numpy(self.c1_in[idx]),
            torch.from_numpy(self.c2_in[idx]),
            torch.from_numpy(self.c1_tgt[idx]),
            torch.from_numpy(self.gt_flux[idx]),
            torch.from_numpy(self.X_norm[idx]),
            torch.from_numpy(self.Y_norm[idx])
        )


class DataManager:
    """
    Data manager for baseline models (GNO, FNO, MGN, DeepONet, GeoFNO).
    Supports both surface and volume meshes.
    """
    
    def __init__(self, n_nodes, pts, device, simplices=None, grid_res=16, mesh_type='auto'):
        """
        Args:
            n_nodes: Number of nodes
            pts: (N, 3) node coordinates
            device: torch device
            simplices: Faces (N, 3) or tetrahedra (N, 4)
            grid_res: Resolution for FNO grid
            mesh_type: 'auto', 'surface', or 'volume'
        """
        self.n_nodes = n_nodes
        self.pts = torch.from_numpy(pts).float().to(device)
        self.device = device
        self.grid_res = grid_res

        self.pts_min = self.pts.min(dim=0)[0]
        self.pts_max = self.pts.max(dim=0)[0]
        self.pts_norm = (self.pts - self.pts_min) / (self.pts_max - self.pts_min + 1e-6)
        
        self.grid_sample_coords = (self.pts_norm * 2.0 - 1.0).view(1, 1, 1, n_nodes, 3)

        if simplices is not None:
            if mesh_type == 'auto':
                if simplices.shape[1] == 3:
                    mesh_type = 'surface'
                elif simplices.shape[1] == 4:
                    mesh_type = 'volume'
            self.mesh_type = mesh_type
            self._build_edge_index(simplices)
        else:
            self.mesh_type = None

    def _build_edge_index(self, simplices):
        """Build edge index from simplices for MGN."""
        edges = set()
        
        if self.mesh_type == 'surface':
            for simplex in simplices:
                for i in range(3):
                    e = tuple(sorted([simplex[i], simplex[(i+1) % 3]]))
                    edges.add(e)
        elif self.mesh_type == 'volume':
            for simplex in simplices:
                for i in range(4):
                    for j in range(i+1, 4):
                        e = tuple(sorted([simplex[i], simplex[j]]))
                        edges.add(e)
        
        edges = list(edges)
        src = [e[0] for e in edges]
        dst = [e[1] for e in edges]
        
        edge_index = np.array([src + dst, dst + src], dtype=np.int64)
        self.edge_index = torch.from_numpy(edge_index).to(self.device)
        
        pts_np = self.pts.cpu().numpy()
        edge_attr = []
        for i, j in zip(src + dst, dst + src):
            rel_pos = pts_np[j] - pts_np[i]
            edge_attr.append(rel_pos)
        self.edge_attr = torch.from_numpy(np.array(edge_attr, dtype=np.float32)).to(self.device)
        
        self.n_edges = len(edges)

    def prepare_gno_batch(self, x_batch, y_batch=None):
        """Prepare batch for GNO. y_batch is (B, N, 3) vectors."""
        B, N = x_batch.shape
        x_in = x_batch.unsqueeze(1)
        pos_expand = self.pts.T.unsqueeze(0).expand(B, -1, -1)
        x_cat = torch.cat([x_in, pos_expand], dim=1)

        if y_batch is not None:
            y_out = y_batch.permute(0, 2, 1)
            return x_cat, y_out
        return x_cat, None

    def prepare_fno_batch(self, x_batch, y_batch=None):
        """Prepare batch for FNO. y_batch is (B, N, 3) vectors."""
        B = x_batch.shape[0]
        Res = self.grid_res

        grid_x = torch.zeros(B, 2, Res, Res, Res, device=self.device)
        grid_y = torch.zeros(B, 3, Res, Res, Res, device=self.device)

        indices = (self.pts_norm * (Res - 1)).long().clamp(0, Res - 1)
        idx_x, idx_y, idx_z = indices[:, 0], indices[:, 1], indices[:, 2]

        for b in range(B):
            grid_x[b, 0, idx_x, idx_y, idx_z] = x_batch[b]
            grid_x[b, 1, idx_x, idx_y, idx_z] = 1.0

            if y_batch is not None:
                grid_y[b, 0, idx_x, idx_y, idx_z] = y_batch[b, :, 0]
                grid_y[b, 1, idx_x, idx_y, idx_z] = y_batch[b, :, 1]
                grid_y[b, 2, idx_x, idx_y, idx_z] = y_batch[b, :, 2]

        return grid_x, grid_y

    def prepare_geofno_batch(self, x_batch):
        B, N = x_batch.shape
        x_coords = self.pts.unsqueeze(0).expand(B, -1, -1)
        features = x_batch.unsqueeze(-1)
        return x_coords, features

    def decode_fno_output(self, grid_out):
        """Decode FNO grid output back to mesh nodes."""
        B = grid_out.shape[0]
        sample_coords = self.grid_sample_coords.expand(B, -1, -1, -1, -1)
        
        sampled = F.grid_sample(
            grid_out, 
            sample_coords, 
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )
        
        out_mesh = sampled.view(B, grid_out.shape[1], -1).permute(0, 2, 1)
        return out_mesh


def load_flux_field_data(pickle_path, logger=None):
    """
    Load flux field dataset from pickle file.
    Returns nodes, simplices, X_data, Y_data, and mesh_type.
    """
    import pickle
    
    def _log(msg):
        if logger:
            logger.log(msg)
        else:
            print(msg)
    
    _log(f"[Data] Loading from {pickle_path}...")
    
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    X_data = data['X_data']
    Y_data = data['Y_data']
    nodes = data['nodes']
    
    if 'elements' in data:
        simplices = data['elements']
        mesh_type = 'volume'
        _log(f"[Data] Detected tetrahedral volume mesh")
    elif 'faces' in data:
        simplices = data['faces']
        mesh_type = 'surface'
        _log(f"[Data] Detected triangular surface mesh")
    else:
        raise ValueError("No 'elements' or 'faces' found in data")
    
    _log(f"[Data] X_data: {X_data.shape}, Y_data: {Y_data.shape}")
    _log(f"[Data] Nodes: {nodes.shape}, Simplices: {simplices.shape}")
    _log(f"[Data] Mesh type: {mesh_type}")
    
    config = data.get('config', {})
    _log(f"[Data] Config: {config}")
    
    return {
        'nodes': nodes,
        'simplices': simplices,
        'X_data': X_data,
        'Y_data': Y_data,
        'mesh_type': mesh_type,
        'config': config,
        'inner_boundary': data.get('inner_boundary', None),
        'outer_boundary': data.get('outer_boundary', None)
    }


def create_flux_dataset(data_dict, host_ops, k_list, train_ratio=0.8, 
                        x_scale=None, y_scale=None, logger=None):
    """
    Create train and test FluxFieldDataset from loaded data.
    
    Args:
        data_dict: Output from load_flux_field_data
        host_ops: DEC operator host
        k_list: [k0, k1, k2] spectral truncation
        train_ratio: Ratio for train/test split
        x_scale: Optional input scaling factor
        y_scale: Optional output scaling factor
        logger: Optional logger
    
    Returns:
        train_dataset, test_dataset, mapper, data_manager
    """
    nodes = data_dict['nodes']
    simplices = data_dict['simplices']
    X_data = data_dict['X_data']
    Y_data = data_dict['Y_data']
    mesh_type = data_dict['mesh_type']
    
    n_samples = len(X_data)
    n_train = int(n_samples * train_ratio)
    
    indices = np.random.permutation(n_samples)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    
    X_train, X_test = X_data[train_idx], X_data[test_idx]
    Y_train, Y_test = Y_data[train_idx], Y_data[test_idx]
    
    mapper = VectorFluxMapper(nodes, simplices, mesh_type=mesh_type)
    
    if x_scale is None:
        x_scale = np.max(np.abs(X_train)) + 1e-9
    if y_scale is None:
        y_scale = np.max(np.abs(Y_train)) + 1e-9
    
    train_dataset = FluxFieldDataset(
        host_ops, mapper, X_train, Y_train, k_list,
        x_scale=x_scale, y_scale=y_scale, logger=logger
    )
    
    test_dataset = FluxFieldDataset(
        host_ops, mapper, X_test, Y_test, k_list,
        x_scale=x_scale, y_scale=y_scale, logger=logger
    )
    
    return train_dataset, test_dataset, mapper