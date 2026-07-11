
"""
High-Order Spectral Operators (HOST) - Optimized & Robust
"""
import numpy as np
from scipy.sparse import spdiags, identity
from scipy.sparse.linalg import eigsh
import toponetx as tnx
import torch 

class HighOrderSpectralOperators:
    """
    Full de Rham Complex with Spectral Operators.
    Optimized for stability and performance on irregular meshes.
    """
    
    def __init__(self, points, faces, k_list=(64, 64, 64), normalize_laplacian=True, logger=None):
        self.points = points
        self.k0, self.k1, self.k2 = k_list
        self.n_nodes = len(points)
        self.logger = logger
        self.normalize = normalize_laplacian
        
        self._log(f"[HOST] Building simplicial complex ({len(points)} nodes)...")
        self.sc = tnx.SimplicialComplex(faces)
        if len(self.sc.nodes) < self.n_nodes:
             for i in range(self.n_nodes):
                if i not in self.sc.nodes:
                    self.sc.add_node(i)

        self._log("[HOST] Extracting Boundary Matrices (B1, B2)...")

        _B1 = self.sc.incidence_matrix(rank=1, signed=True)
        if _B1.shape[1] != self.n_nodes:
            self.B1 = _B1.T
        else:
            self.B1 = _B1

        n_edges = self.B1.shape[0]
        _B2 = self.sc.incidence_matrix(rank=2, signed=True)
        if _B2.shape[1] != n_edges:
            self.B2 = _B2.T
        else:
            self.B2 = _B2

        self.n_edges = self.B1.shape[0]
        self.n_faces = self.B2.shape[0]

        self._log("[HOST] Computing Hodge Laplacians (L0, L1, L2)...")
        # L0 = B1.T @ B1 (Graph Laplacian)
        # L1 = B1 @ B1.T + B2.T @ B2
        # L2 = B2 @ B2.T
        self.L0 = self.sc.hodge_laplacian_matrix(rank=0, signed=True)
        self.L1 = self.sc.hodge_laplacian_matrix(rank=1, signed=True)
        self.L2 = self.sc.hodge_laplacian_matrix(rank=2, signed=True)

        if self.normalize:
            self._log("[HOST] Applying Symmetric Normalization to Laplacians...")
            self.L0 = self._normalize_laplacian(self.L0)
            self.L1 = self._normalize_laplacian(self.L1)
            self.L2 = self._normalize_laplacian(self.L2)

        self._log("[HOST] Computing Eigenbases (Φ0, Φ1, Φ2)...")
        self.Phi0 = self._compute_basis_fast(self.L0, self.k0)
        self.Phi1 = self._compute_basis_fast(self.L1, self.k1)
        self.Phi2 = self._compute_basis_fast(self.L2, self.k2)

        self._log("[HOST] Building spectral de Rham operators...")
        self._build_spectral_operators()
        
        self._log(f"[HOST] Complete: Nodes={self.n_nodes}, Edges={self.n_edges}, Faces={self.n_faces}")

    def _log(self, message):
        if self.logger:
            self.logger.log(message)
        else:
            print(message)
    
    def _normalize_laplacian(self, L):
        """
        Compute D^-1/2 * L * D^-1/2 efficiently.
        """
        deg = L.diagonal()

        d_inv_sqrt = np.power(deg, -0.5, where=deg > 1e-10)
        d_inv_sqrt[deg <= 1e-10] = 0.0
        
        D_mat = spdiags(d_inv_sqrt, 0, L.shape[0], L.shape[0])
        
        return D_mat @ L @ D_mat

    def _compute_basis_fast(self, L, k):
        n = L.shape[0]
        if n == 0 or k <= 0:
            return np.zeros((n, 0))
        k = min(k, n - 1) if n > 1 else 1
        
        L_float = L.astype(float)


        if n < 2000:
            vals, vecs = np.linalg.eigh(L_float.toarray())
            return vecs[:, :k]
        
    
        try:

            vals, vecs = eigsh(L_float, k=k, sigma=-1e-5, which='LM', tol=1e-4, maxiter=5000)
            return vecs
        except Exception as e:
            self._log(f"[Warn] Shift-invert eigsh failed ({e}), falling back to standard SA...")
            try:
                vals, vecs = eigsh(L_float, k=k, which='SA', tol=1e-3)
                return vecs
            except Exception:
                self._log("[Warn] Sparse solver failed completely, using dense fallback.")
                vals, vecs = np.linalg.eigh(L_float.toarray())
                return vecs[:, :k]

    def _build_spectral_operators(self):
        """
        Build spectral representation of de Rham operators:
        M_d = Phi_out.T @ B @ Phi_in
        """
        k0 = self.Phi0.shape[1]
        k1 = self.Phi1.shape[1]
        k2 = self.Phi2.shape[1]

        # 1. Gradient operator: d0 (0-forms -> 1-forms)
        if k0 == 0 or k1 == 0:
            self.Md0 = np.zeros((k1, k0), dtype=np.float32)
            self.Mdelta1 = np.zeros((k0, k1), dtype=np.float32)
        else:

            B1 = self.B1.astype(float) 
            B1_Phi0 = B1.dot(self.Phi0[:, :k0]) 
            self.Md0 = (self.Phi1[:, :k1].T @ B1_Phi0).astype(np.float32)
            self.Mdelta1 = self.Md0.T.astype(np.float32)

        # 2. Curl operator: d1 (1-forms -> 2-forms)
        if k1 == 0 or k2 == 0:
            self.Md1 = np.zeros((k2, k1), dtype=np.float32)
            self.Mdelta2 = np.zeros((k1, k2), dtype=np.float32)
        else:
            B2 = self.B2.astype(float)
            # (F, E) @ (E, k1) -> (F, k1)
            B2_Phi1 = B2.dot(self.Phi1[:, :k1])
            self.Md1 = (self.Phi2[:, :k2].T @ B2_Phi1).astype(np.float32)
            self.Mdelta2 = self.Md1.T.astype(np.float32)
        
        self._log(f"       [Ops] Md0(Grad): {self.Md0.shape}, Mdelta1(Div): {self.Mdelta1.shape}")
        self._log(f"       [Ops] Md1(Curl): {self.Md1.shape}, Mdelta2(Rot): {self.Mdelta2.shape}")

    def lift_signal(self, f_vertex):
        """
        Lift scalar signal from nodes to edges.
        
        Args:
            f_vertex: (N,) or (1, N) Scalar field on nodes
            
        Returns:
            f_vertex: Original signal
            g_edge: Gradient on edges (B1 @ f)
            h_face: ZERO (by definition d^2=0). Returned as zero array for compatibility.
        """
        f_vertex = np.asarray(f_vertex, dtype=float)
        if f_vertex.ndim == 1:
            f_vertex = f_vertex.reshape(1, -1)

        # Gradient: d0(f)
        g_edge = (self.B1 @ f_vertex.T).T
        

        h_face = np.zeros((f_vertex.shape[0], self.n_faces))
        
        return f_vertex, g_edge, h_face