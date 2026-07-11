"""
High-Order Spectral Operators (HOST) - Unified Multi-Input Version

Supports:
  - Standard simplicial complex from mesh faces (original)
  - Heat-kernel-weighted Laplacian from point cloud adapter
  - Any input that provides (points, faces) via adapters
"""
import numpy as np
import gc
from scipy.sparse import spdiags, identity, csr_matrix
from scipy.sparse.linalg import eigsh
import toponetx as tnx
import torch


class HighOrderSpectralOperators:
    """
    Full de Rham Complex with Spectral Operators.
    Supports optional heat kernel Laplacian override for point cloud inputs.
    """

    def __init__(self, points, faces, k_list=(64, 64, 64), normalize_laplacian=True,
                 logger=None, heat_kernel_laplacian=None):
        """
        Args:
            points: (N, D) node coordinates
            faces: (M, 3) triangle faces
            k_list: (k0, k1, k2) number of eigenfunctions
            normalize_laplacian: apply D^{-1/2} L D^{-1/2}
            logger: optional logger
            heat_kernel_laplacian: optional (N,N) sparse matrix from PointCloudAdapter.
                If provided, replaces L0 with this smoother Laplacian for
                better eigenfunctions on noisy point clouds.
        """
        self.points = points
        self.k0, self.k1, self.k2 = k_list
        self.n_nodes = len(points)
        self.logger = logger
        self.normalize = normalize_laplacian
        self.heat_kernel_laplacian = heat_kernel_laplacian

        self._log(f"[HOST] Building simplicial complex ({len(points)} nodes, {len(faces)} faces)...")

        sc = tnx.SimplicialComplex(faces)

        if len(sc.nodes) < self.n_nodes:
            for i in range(self.n_nodes):
                if i not in sc.nodes:
                    sc.add_node(i)

        self._log("[HOST] Extracting Boundary Matrices (B1, B2)...")

        _B1 = sc.incidence_matrix(rank=1, signed=True)
        if _B1.shape[1] != self.n_nodes:
            self.B1 = _B1.T
        else:
            self.B1 = _B1

        n_edges = self.B1.shape[0]
        _B2 = sc.incidence_matrix(rank=2, signed=True)
        if _B2.shape[1] != n_edges:
            self.B2 = _B2.T
        else:
            self.B2 = _B2

        self.n_edges = self.B1.shape[0]
        self.n_faces = self.B2.shape[0]

        self._log("[HOST] Computing Hodge Laplacians (L0, L1, L2)...")

        # Compute L1, L2 from boundary operators (always combinatorial)
        self.L1 = (self.B1 @ self.B1.T + self.B2.T @ self.B2).tocsr()
        self.L2 = (self.B2 @ self.B2.T).tocsr()

        # L0: use heat kernel Laplacian if provided, else combinatorial
        if heat_kernel_laplacian is not None:
            self._log("[HOST] Using heat-kernel-weighted L0 from point cloud adapter")
            self.L0 = heat_kernel_laplacian
        else:
            self.L0 = (self.B1.T @ self.B1).tocsr()

        del sc
        gc.collect()

        if self.normalize:
            self._log("[HOST] Applying Symmetric Normalization to Laplacians...")
            self.L0 = self._normalize_laplacian(self.L0)
            self.L1 = self._normalize_laplacian(self.L1)
            self.L2 = self._normalize_laplacian(self.L2)

        self._log("[HOST] Computing Eigenbases (Phi0, Phi1, Phi2)...")
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
        deg = np.array(L.diagonal()).flatten()
        d_inv_sqrt = np.power(deg, -0.5, where=deg > 1e-10)
        d_inv_sqrt[deg <= 1e-10] = 0.0
        D_mat = spdiags(d_inv_sqrt, 0, L.shape[0], L.shape[0])
        return D_mat @ L @ D_mat

    def _compute_basis_fast(self, L, k):
        n = L.shape[0]
        if n == 0 or k <= 0:
            return np.zeros((n, 0))

        k = min(k, n - 1) if n > 1 else 1
        L_float = L.astype(np.float64)

        if n < 2000:
            vals, vecs = np.linalg.eigh(L_float.toarray())
            return vecs[:, :k]

        try:
            vals, vecs = eigsh(L_float, k=k, sigma=-1e-4, which='LM', tol=1e-4, maxiter=5000)
            return vecs
        except Exception as e:
            self._log(f"[Warn] Shift-invert eigsh failed ({e}), falling back to standard SA...")
            try:
                vals, vecs = eigsh(L_float, k=k, which='SA', tol=1e-3, maxiter=5000)
                return vecs
            except Exception:
                self._log("[Warn] Sparse solver failed completely, using dense fallback.")
                vals, vecs = np.linalg.eigh(L_float.toarray())
                return vecs[:, :k]

    def _fix_sign(self):
        """Fix Z2 sign ambiguity only (safe, no eigenspace rotation).

        For each eigenvector, enforce that the component with the largest
        absolute value is POSITIVE. This is a deterministic convention
        that is consistent across different eigensolvers.

        Note: degenerate eigenspace rotation is NOT fixed here because
        it can change the effective basis direction and hurt accuracy.
        For cross-mesh generalization, use gauge-invariant features instead.
        """
        for Phi_name in ['Phi0', 'Phi1', 'Phi2']:
            Phi = getattr(self, Phi_name)
            if Phi.shape[1] == 0:
                continue
            for i in range(Phi.shape[1]):
                idx = np.argmax(np.abs(Phi[:, i]))
                if Phi[idx, i] < 0:
                    Phi[:, i] *= -1
            setattr(self, Phi_name, Phi)

    def _build_spectral_operators(self):
        k0 = self.Phi0.shape[1]
        k1 = self.Phi1.shape[1]
        k2 = self.Phi2.shape[1]

        if k0 == 0 or k1 == 0:
            self.Md0 = np.zeros((k1, k0), dtype=np.float32)
            self.Mdelta1 = np.zeros((k0, k1), dtype=np.float32)
        else:
            B1 = self.B1.astype(float)
            B1_Phi0 = B1.dot(self.Phi0[:, :k0])
            self.Md0 = (self.Phi1[:, :k1].T @ B1_Phi0).astype(np.float32)
            self.Mdelta1 = self.Md0.T.astype(np.float32)

        if k1 == 0 or k2 == 0:
            self.Md1 = np.zeros((k2, k1), dtype=np.float32)
            self.Mdelta2 = np.zeros((k1, k2), dtype=np.float32)
        else:
            B2 = self.B2.astype(float)
            B2_Phi1 = B2.dot(self.Phi1[:, :k1])
            self.Md1 = (self.Phi2[:, :k2].T @ B2_Phi1).astype(np.float32)
            self.Mdelta2 = self.Md1.T.astype(np.float32)

        self._log(f"       [Ops] Md0(Grad): {self.Md0.shape}, Mdelta1(Div): {self.Mdelta1.shape}")
        self._log(f"       [Ops] Md1(Curl): {self.Md1.shape}, Mdelta2(Rot): {self.Mdelta2.shape}")

    def compute_enriched_input(self, f_vertex, n_heat_scales=6):
        """Multi-scale heat kernel diffusion of input signal.
        Returns enriched c0 with n_heat_scales additional filtered versions."""
        f_vertex = np.asarray(f_vertex, dtype=float)
        if f_vertex.ndim == 1:
            f_vertex = f_vertex.reshape(1, -1)

        # Standard projection
        c0_raw = (f_vertex @ self.Phi0[:, :self.k0]).astype(np.float32)

        # Eigenvalues of L0
        L0_Phi0 = self.L0.dot(self.Phi0[:, :self.k0])
        lambdas = np.sum(self.Phi0[:, :self.k0] * L0_Phi0, axis=0)
        lambdas = np.maximum(lambdas, 0)

        # Multi-scale heat diffusion in spectral domain
        t_values = np.logspace(-1, 2, n_heat_scales)
        c0_parts = [c0_raw]
        for t in t_values:
            heat_filter = np.exp(-t * lambdas).astype(np.float32)
            c0_parts.append(c0_raw * heat_filter[np.newaxis, :])

        c0_enriched = np.concatenate(c0_parts, axis=-1)  # (B, k0*(1+n_scales))

        # Gradient (1-form)
        g_edge = (self.B1 @ f_vertex.T).T
        c1 = (g_edge @ self.Phi1[:, :self.k1]).astype(np.float32)

        # Nonlinear gradient: d(f^2) gives independent 1-form info
        f_sq = f_vertex ** 2
        g_sq = (self.B1 @ f_sq.T).T
        c1_nl = (g_sq @ self.Phi1[:, :self.k1]).astype(np.float32)
        c1_enriched = np.concatenate([c1, c1_nl], axis=-1)  # (B, 2*k1)

        # --- Higher-order 2-form: genuinely nonzero curl features ---
        # d^2=0 means B2 @ B1 @ f = 0 for any f.
        # But edge-averaged node values are NOT exact 1-forms,
        # so their curl via B2 IS nonzero. This gives independent 2-form info.
        c2_parts = []
        abs_B1 = np.abs(self.B1.toarray()) if hasattr(self.B1, 'toarray') else np.abs(self.B1)
        f_avg_edges = (abs_B1 @ f_vertex.T).T / 2.0  # (B, E)
        curl_f_avg = (self.B2 @ f_avg_edges.T).T  # (B, F) -- NONZERO
        c2_avg = (curl_f_avg @ self.Phi2[:, :self.k2]).astype(np.float32)
        c2_parts.append(c2_avg)

        # Pointwise product of coordinate gradients with signal gradient
        # grad(x_i) * grad(f) -> curl gives curvature-coupled 2-form
        if hasattr(self, 'points') and self.points.shape[1] >= 3:
            grad_f = g_edge  # already computed above: (B, E)
            for dim in range(min(3, self.points.shape[1])):
                grad_coord = np.array((self.B1 @ self.points[:, dim]).flatten())
                mixed_1form = grad_f * grad_coord[np.newaxis, :]
                curl_mixed = (self.B2 @ mixed_1form.T).T
                c2_mixed = (curl_mixed @ self.Phi2[:, :self.k2]).astype(np.float32)
                c2_parts.append(c2_mixed)

        c2_enriched = np.concatenate(c2_parts, axis=-1)  # (B, k2 * (1 + n_coords))

        return c0_enriched, c1_enriched, c2_enriched

    def lift_signal(self, f_vertex):
        f_vertex = np.asarray(f_vertex, dtype=float)
        if f_vertex.ndim == 1:
            f_vertex = f_vertex.reshape(1, -1)
        g_edge = (self.B1 @ f_vertex.T).T
        h_face = np.zeros((f_vertex.shape[0], self.n_faces))
        return f_vertex, g_edge, h_face
