"""
Dataset classes for Flux Field Prediction
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import toponetx as tnx


class VectorFluxMapper:
    """
    Maps between node vectors and edge flux.
    Used for:
    - Converting GT vectors to GT flux during dataset creation
    - Converting baseline model predictions (vectors) to flux for evaluation
    """
    
    def __init__(self, points, faces):
        self.points = points
        self.faces = faces
        self.sc = tnx.SimplicialComplex(faces)
        
        self.B1 = self.sc.incidence_matrix(rank=1, signed=True)
        if self.B1.shape[1] != len(points):
            self.B1 = self.B1.T
        
        self.edge_vectors = self.B1 @ self.points
        self.edge_lengths = np.linalg.norm(self.edge_vectors, axis=1, keepdims=True) + 1e-9
        self.edge_dirs = self.edge_vectors / self.edge_lengths
        
        self.node_degree = np.abs(self.B1).T @ np.ones((self.B1.shape[0], 1))
        self.node_degree[self.node_degree < 1] = 1.0
        
        self.n_edges = self.B1.shape[0]
    
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


# ==========================================
# Flux Dataset for Spectral Models
# ==========================================
class FluxFieldDataset(Dataset):
    """
    Dataset for flux field (1-form) prediction.
    
    Uses DEC lifting to derive spectral coefficients:
    - c0: scalar input projected onto Φ0
    - c1: gradient of input projected onto Φ1
    - c2: curl (should be ~0 due to d²=0)
    
    Target: GT flux projected onto Φ1
    """
    
    def __init__(self, host_ops, mapper, X_node_data, Y_node_data, k_list,
                 x_scale=None, y_scale=None, logger=None):
        super().__init__()
        self.k0, self.k1, self.k2 = k_list
        self.logger = logger
        self.mapper = mapper
        
        # Compute scaling factors
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
        
        # Normalize data
        X_norm = X_node_data / self.x_scale
        Y_norm = Y_node_data / self.y_scale
        
        self.X_norm = X_norm.astype(np.float32)
        self.Y_norm = Y_norm.astype(np.float32)  # (B, N, 3) vectors
        
        self._log(f"         Normalized X: min={np.min(X_norm):.4f}, max={np.max(X_norm):.4f}")
        self._log(f"         Normalized Y: min={np.min(Y_norm):.4f}, max={np.max(Y_norm):.4f}")
        
        # Lift INPUT to spectral space using DEC
        self._log("[Dataset] Lifting Input to Full de Rham Complex...")
        fX, gX, hX = host_ops.lift_signal(X_norm)
        self.c0_in = (fX @ host_ops.Phi0[:, :self.k0]).astype(np.float32)
        self.c1_in = (gX @ host_ops.Phi1[:, :self.k1]).astype(np.float32)
        self.c2_in = (hX @ host_ops.Phi2[:, :self.k2]).astype(np.float32)
        
        # Compute GT Edge Flux from normalized vectors
        self._log("[Dataset] Computing GT Edge Flux...")
        flux_list = []
        for i in range(len(Y_norm)):
            flux = mapper.node_vector_to_edge_flux(Y_norm[i])
            flux_list.append(flux)
        self.gt_flux = np.array(flux_list).astype(np.float32)
        
        # Project GT flux to spectral coefficients
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
            torch.from_numpy(self.Y_norm[idx])  # GT vector for baselines
        )


# ==========================================
# Data Manager for Baseline Models
# ==========================================
class DataManager:
    """Data manager for baseline models (GNO, FNO, MGN, DeepONet, GeoFNO)."""
    
    def __init__(self, n_nodes, pts, device, faces=None, grid_res=16):
        self.n_nodes = n_nodes
        self.pts = torch.from_numpy(pts).float().to(device)
        self.device = device
        self.grid_res = grid_res

        self.pts_min = self.pts.min(dim=0)[0]
        self.pts_max = self.pts.max(dim=0)[0]
        self.pts_norm = (self.pts - self.pts_min) / (self.pts_max - self.pts_min + 1e-6)
        
        self.grid_sample_coords = (self.pts_norm * 2.0 - 1.0).view(1, 1, 1, n_nodes, 3)

        # Build edge information for MGN
        if faces is not None:
            self._build_edge_index(faces)

    def _build_edge_index(self, faces):
        """Build edge index from faces for MGN."""
        edges = set()
        for face in faces:
            for i in range(3):
                e = tuple(sorted([face[i], face[(i+1) % 3]]))
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

    def prepare_gno_batch(self, x_batch, y_batch=None):
        """Prepare batch for GNO. y_batch is (B, N, 3) vectors."""
        B, N = x_batch.shape
        x_in = x_batch.unsqueeze(1)
        pos_expand = self.pts.T.unsqueeze(0).expand(B, -1, -1)
        x_cat = torch.cat([x_in, pos_expand], dim=1)

        if y_batch is not None:
            y_out = y_batch.permute(0, 2, 1)  # (B, 3, N)
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