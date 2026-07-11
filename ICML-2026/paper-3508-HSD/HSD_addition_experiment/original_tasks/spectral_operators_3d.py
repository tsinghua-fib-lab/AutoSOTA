
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
        # 优化：直接使用 scipy 稀疏矩阵构建，避免 tnx 在大图上的开销 (这里保留 tnx 接口但优化后续处理)
        self.sc = tnx.SimplicialComplex(faces)
        # 确保孤立点也被包含
        if len(self.sc.nodes) < self.n_nodes:
             for i in range(self.n_nodes):
                if i not in self.sc.nodes:
                    self.sc.add_node(i)

        self._log("[HOST] Extracting Boundary Matrices (B1, B2)...")
        # 1. 获取 B1 (Edges -> Nodes)
        # 注意：标准 DEC 定义 B1 是 (Nodes, Edges) 还是 (Edges, Nodes) 取决于上下文
        # 这里统一为：算子矩阵维度 (Target_Dim, Source_Dim)
        # 即 B1: 0-form -> 1-form (Gradient), 形状应为 (Edges, Nodes)
        _B1 = self.sc.incidence_matrix(rank=1, signed=True)
        if _B1.shape[1] != self.n_nodes:
            self.B1 = _B1.T
        else:
            self.B1 = _B1
            
        # 2. 获取 B2 (Faces -> Edges)
        # 即 B2: 1-form -> 2-form (Curl), 形状应为 (Faces, Edges)
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

        # 优化：对称归一化 (Symmetric Normalization)
        # L_norm = D^-0.5 * L * D^-0.5
        # 这能防止特征向量在网格密集处过拟合，提高模型泛化能力
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
        # 计算对角线度数 (Degree)
        deg = L.diagonal()
        # 避免除以 0
        d_inv_sqrt = np.power(deg, -0.5, where=deg > 1e-10)
        d_inv_sqrt[deg <= 1e-10] = 0.0
        
        # 构建对角矩阵
        D_mat = spdiags(d_inv_sqrt, 0, L.shape[0], L.shape[0])
        
        # 稀疏矩阵乘法: D * L * D
        return D_mat @ L @ D_mat

    def _compute_basis_fast(self, L, k):
        n = L.shape[0]
        if n == 0 or k <= 0:
            return np.zeros((n, 0))
        # 修正：防止 k 超过矩阵秩
        k = min(k, n - 1) if n > 1 else 1
        
        L_float = L.astype(float)

        # 优化：小矩阵直接用稠密解法，更快且稳定
        if n < 2000:
            vals, vecs = np.linalg.eigh(L_float.toarray())
            return vecs[:, :k]
        
        # 大矩阵使用 Shift-Invert 模式求解最小特征值
        # sigma = -0.01 将最小特征值映射为最大绝对值，利于 Lanczos 迭代快速收敛
        try:
            # 增加 maxiter 确保收敛
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
            # 使用稀疏矩阵乘法减少内存占用: (E, N) @ (N, k0) -> (E, k0) (Dense)
            # 只有最后一步才变成稠密矩阵
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
        
        # [优化] Curl of Gradient is ALWAYS 0.
        # 不需要计算 h_face = B2 @ g_edge，直接返回全0矩阵
        # 这避免了浮点数误差 (e.g. 1e-16)，这在物理上是很重要的
        h_face = np.zeros((f_vertex.shape[0], self.n_faces))
        
        return f_vertex, g_edge, h_face