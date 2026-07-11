"""
Input Adapters: Unified interface to convert any geometric input
(mesh, point cloud, graph) into a simplicial complex with boundary operators.

The key insight: HSD only needs (points, faces, B1, B2) to build
the de Rham complex. Any input can be adapted to produce these.

Supported adapters:
    MeshAdapter      - triangulated mesh (points + faces) -> direct
    PointCloudAdapter - raw points -> KNN heat kernel -> Delaunay/Alpha -> simplicial complex
    GraphAdapter     - graph (nodes + adjacency) -> flag/clique complex -> simplicial complex
"""
import numpy as np
from scipy.spatial import Delaunay
from scipy.sparse import csr_matrix, coo_matrix
import warnings


class BaseAdapter:
    """Base class for all input adapters."""

    def __init__(self, logger=None):
        self.logger = logger
        self.points = None
        self.faces = None
        self.n_nodes = 0
        self.n_faces = 0

    def _log(self, msg):
        if self.logger:
            self.logger.log(msg)
        else:
            print(msg)

    def adapt(self, **kwargs):
        """Convert input to (points, faces). Subclasses must implement."""
        raise NotImplementedError

    def get_simplicial_data(self):
        """Return (points, faces) tuple."""
        return self.points, self.faces


class MeshAdapter(BaseAdapter):
    """
    Direct mesh adapter. Input is already (points, faces).
    This is the trivial case - no conversion needed.
    """

    def adapt(self, points, faces, **kwargs):
        self._log(f"[MeshAdapter] Direct mesh input: {len(points)} nodes, {len(faces)} faces")
        self.points = np.asarray(points, dtype=np.float64)
        self.faces = np.asarray(faces, dtype=np.int64)
        self.n_nodes = len(self.points)
        self.n_faces = len(self.faces)
        return self.points, self.faces


class PointCloudAdapter(BaseAdapter):
    """
    Point cloud -> Simplicial complex via:
    1. KNN graph construction
    2. Heat kernel weighting (optional, for better Laplacian)
    3. Delaunay or Alpha complex triangulation
    4. Filter degenerate / too-large triangles

    This mirrors the approach used in diffusion models on point clouds
    (e.g., constructing the graph Laplacian via heat kernel KNN).
    """

    def __init__(self, k=15, heat_t=1.0, use_alpha=True,
                 max_edge_ratio=3.0, logger=None):
        super().__init__(logger)
        self.k = k
        self.heat_t = heat_t
        self.use_alpha = use_alpha
        self.max_edge_ratio = max_edge_ratio
        # Store heat kernel weights for potential downstream use
        self.heat_weights = None
        self.knn_edges = None

    def _build_knn(self, points):
        """Build KNN graph. Returns (src, dst, distances)."""
        from sklearn.neighbors import NearestNeighbors
        n = len(points)
        k = min(self.k, n - 1)
        nn_model = NearestNeighbors(n_neighbors=k + 1, algorithm='auto')
        nn_model.fit(points)
        distances, indices = nn_model.kneighbors(points)
        # Skip self-loop (index 0)
        src = np.repeat(np.arange(n), k)
        dst = indices[:, 1:].flatten()
        dists = distances[:, 1:].flatten()
        return src, dst, dists

    def _heat_kernel_weights(self, dists):
        """Compute heat kernel weights: w = exp(-d^2 / (4t))."""
        return np.exp(-dists ** 2 / (4 * self.heat_t + 1e-9))

    def _delaunay_triangulate(self, points):
        """Delaunay triangulation, works for 2D and 3D."""
        dim = points.shape[1]

        if dim == 3:
            # 3D Delaunay produces tetrahedra; extract surface triangles
            try:
                tri = Delaunay(points)
                simplices = tri.simplices  # (M, 4) tetrahedra
                # Extract all triangular faces
                face_set = set()
                for tet in simplices:
                    for combo in [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]:
                        face = tuple(sorted([tet[combo[0]], tet[combo[1]], tet[combo[2]]]))
                        face_set.add(face)
                faces = np.array(list(face_set), dtype=np.int64)
            except Exception as e:
                self._log(f"[PointCloudAdapter] Delaunay failed ({e}), falling back to KNN triangulation")
                faces = self._knn_triangulate(points)
        elif dim == 2:
            tri = Delaunay(points)
            faces = tri.simplices.astype(np.int64)
        else:
            raise ValueError(f"Unsupported dimension: {dim}")

        return faces

    def _alpha_complex_triangulate(self, points):
        """
        Alpha complex: subset of Delaunay filtered by circumradius.
        Produces a cleaner triangulation for point clouds.
        """
        try:
            from scipy.spatial import Delaunay
            tri = Delaunay(points)

            dim = points.shape[1]
            if dim == 3:
                simplices = tri.simplices
            else:
                simplices = tri.simplices

            # Compute median edge length for alpha threshold
            all_edges = []
            for s in simplices:
                for i in range(len(s)):
                    for j in range(i + 1, len(s)):
                        all_edges.append(np.linalg.norm(points[s[i]] - points[s[j]]))
            median_edge = np.median(all_edges)
            alpha_threshold = median_edge * self.max_edge_ratio

            # Filter simplices: keep only those with all edges < threshold
            good_faces = []
            if dim == 3:
                # Extract triangular faces from tetrahedra
                for tet in simplices:
                    for combo in [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]:
                        face = [tet[combo[0]], tet[combo[1]], tet[combo[2]]]
                        edges_ok = True
                        for a in range(3):
                            for b in range(a + 1, 3):
                                if np.linalg.norm(points[face[a]] - points[face[b]]) > alpha_threshold:
                                    edges_ok = False
                                    break
                            if not edges_ok:
                                break
                        if edges_ok:
                            good_faces.append(tuple(sorted(face)))
                good_faces = list(set(good_faces))
            else:
                for tri_s in simplices:
                    edges_ok = True
                    for a in range(3):
                        for b in range(a + 1, 3):
                            if np.linalg.norm(points[tri_s[a]] - points[tri_s[b]]) > alpha_threshold:
                                edges_ok = False
                                break
                        if not edges_ok:
                            break
                    if edges_ok:
                        good_faces.append(tuple(sorted(tri_s)))

            if len(good_faces) == 0:
                self._log("[PointCloudAdapter] Alpha complex produced 0 faces, using full Delaunay")
                return self._delaunay_triangulate(points)

            faces = np.array(good_faces, dtype=np.int64)
            return faces

        except Exception as e:
            self._log(f"[PointCloudAdapter] Alpha complex failed ({e}), falling back to Delaunay")
            return self._delaunay_triangulate(points)

    def _knn_triangulate(self, points):
        """
        Fallback: construct triangles from KNN graph.
        For each node, try to form triangles with pairs of its neighbors.
        """
        src, dst, _ = self._build_knn(points)
        n = len(points)

        # Build adjacency
        neighbors = [set() for _ in range(n)]
        for s, d in zip(src, dst):
            neighbors[s].add(d)
            neighbors[d].add(s)

        face_set = set()
        for i in range(n):
            nbrs = list(neighbors[i])
            for a_idx in range(len(nbrs)):
                for b_idx in range(a_idx + 1, len(nbrs)):
                    j, k = nbrs[a_idx], nbrs[b_idx]
                    if k in neighbors[j]:
                        face = tuple(sorted([i, j, k]))
                        face_set.add(face)

        if len(face_set) == 0:
            warnings.warn("KNN triangulation produced 0 triangles")
            return np.zeros((0, 3), dtype=np.int64)

        return np.array(list(face_set), dtype=np.int64)

    def adapt(self, points, **kwargs):
        """
        Convert point cloud to simplicial complex.

        Args:
            points: (N, D) point coordinates
        Returns:
            points, faces
        """
        self.points = np.asarray(points, dtype=np.float64)
        n = len(self.points)
        self._log(f"[PointCloudAdapter] Input: {n} points in R^{self.points.shape[1]}")

        # 1. Build KNN graph with heat kernel weights
        self._log(f"[PointCloudAdapter] Building KNN graph (k={self.k})...")
        src, dst, dists = self._build_knn(self.points)
        self.heat_weights = self._heat_kernel_weights(dists)
        self.knn_edges = np.stack([src, dst], axis=1)
        self._log(f"[PointCloudAdapter] KNN edges: {len(src)}, heat kernel t={self.heat_t}")

        # 2. Triangulate
        if self.use_alpha:
            self._log("[PointCloudAdapter] Running Alpha complex triangulation...")
            self.faces = self._alpha_complex_triangulate(self.points)
        else:
            self._log("[PointCloudAdapter] Running Delaunay triangulation...")
            self.faces = self._delaunay_triangulate(self.points)

        self.n_nodes = len(self.points)
        self.n_faces = len(self.faces)
        self._log(f"[PointCloudAdapter] Result: {self.n_nodes} nodes, {self.n_faces} faces")

        return self.points, self.faces

    def get_heat_kernel_laplacian(self):
        """
        Return a heat-kernel-weighted graph Laplacian.
        This can be used as an alternative to the combinatorial Laplacian,
        providing smoother eigenfunctions on noisy point clouds.

        Returns:
            L_heat: (N, N) sparse CSR matrix
        """
        if self.knn_edges is None or self.heat_weights is None:
            raise RuntimeError("Must call adapt() first")

        n = self.n_nodes
        src = self.knn_edges[:, 0]
        dst = self.knn_edges[:, 1]
        w = self.heat_weights

        # Symmetrize
        src_full = np.concatenate([src, dst])
        dst_full = np.concatenate([dst, src])
        w_full = np.concatenate([w, w])

        W = csr_matrix((w_full, (src_full, dst_full)), shape=(n, n))
        # Ensure exact symmetry
        W = (W + W.T) / 2.0

        D = np.array(W.sum(axis=1)).flatten()
        D_mat = csr_matrix((D, (np.arange(n), np.arange(n))), shape=(n, n))
        L_heat = D_mat - W

        return L_heat


class GraphAdapter(BaseAdapter):
    """
    Graph -> Simplicial complex via:
    1. Use node positions (or learn implicit positions)
    2. Build flag/clique complex from graph
    3. Extract triangular faces (2-cliques)

    Supports:
    - Graph with coordinates: (nodes, edges, positions)
    - Graph without coordinates: implicit encoding via spectral embedding
    """

    def __init__(self, use_spectral_pos=False, embed_dim=3, logger=None):
        super().__init__(logger)
        self.use_spectral_pos = use_spectral_pos
        self.embed_dim = embed_dim

    def _spectral_embedding(self, adjacency, n_nodes, dim=3):
        """
        Compute spectral positions from graph Laplacian.
        Uses first `dim` non-trivial eigenvectors as coordinates.
        """
        from scipy.sparse.linalg import eigsh
        from scipy.sparse import diags

        D = np.array(adjacency.sum(axis=1)).flatten()
        D_inv_sqrt = np.power(D, -0.5, where=D > 1e-10)
        D_inv_sqrt[D <= 1e-10] = 0.0
        D_mat = diags(D_inv_sqrt)
        L_norm = diags(np.ones(n_nodes)) - D_mat @ adjacency @ D_mat

        k = min(dim + 1, n_nodes - 1)
        try:
            vals, vecs = eigsh(L_norm.astype(np.float64), k=k, which='SM', tol=1e-4)
            # Skip first eigenvector (constant)
            positions = vecs[:, 1:dim + 1]
        except Exception:
            vals, vecs = np.linalg.eigh(L_norm.toarray())
            positions = vecs[:, 1:dim + 1]

        # Normalize to unit ball
        scale = np.max(np.linalg.norm(positions, axis=1)) + 1e-9
        positions = positions / scale

        return positions

    def _find_triangles(self, edge_index, n_nodes):
        """
        Find all triangles (3-cliques) in the graph.
        """
        neighbors = [set() for _ in range(n_nodes)]
        for i, j in edge_index:
            neighbors[i].add(j)
            neighbors[j].add(i)

        triangles = set()
        for i in range(n_nodes):
            for j in neighbors[i]:
                if j > i:
                    common = neighbors[i] & neighbors[j]
                    for k in common:
                        if k > j:
                            triangles.add((i, j, k))

        return np.array(list(triangles), dtype=np.int64) if triangles else np.zeros((0, 3), dtype=np.int64)

    def adapt(self, edge_index, n_nodes, positions=None, **kwargs):
        """
        Convert graph to simplicial complex.

        Args:
            edge_index: (E, 2) array of edges
            n_nodes: number of nodes
            positions: (N, D) optional node coordinates
        """
        edge_index = np.asarray(edge_index, dtype=np.int64)
        self._log(f"[GraphAdapter] Input: {n_nodes} nodes, {len(edge_index)} edges")

        # Get or compute positions
        if positions is not None:
            self.points = np.asarray(positions, dtype=np.float64)
            self._log(f"[GraphAdapter] Using provided positions in R^{self.points.shape[1]}")
        elif self.use_spectral_pos:
            self._log(f"[GraphAdapter] Computing spectral embedding (dim={self.embed_dim})...")
            # Build adjacency
            rows = np.concatenate([edge_index[:, 0], edge_index[:, 1]])
            cols = np.concatenate([edge_index[:, 1], edge_index[:, 0]])
            data = np.ones(len(rows))
            adj = csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
            self.points = self._spectral_embedding(adj, n_nodes, self.embed_dim)
        else:
            raise ValueError("GraphAdapter requires positions or use_spectral_pos=True")

        # Find triangles (2-cliques) to form faces
        self._log("[GraphAdapter] Finding triangles (3-cliques)...")
        self.faces = self._find_triangles(edge_index, n_nodes)

        self.n_nodes = n_nodes
        self.n_faces = len(self.faces)
        self._log(f"[GraphAdapter] Result: {self.n_nodes} nodes, {self.n_faces} triangular faces")

        if self.n_faces == 0:
            self._log("[GraphAdapter] WARNING: No triangles found. The graph may be too sparse.")
            self._log("[GraphAdapter] Falling back to Delaunay triangulation on positions...")
            pc_adapter = PointCloudAdapter(logger=self.logger)
            self.points, self.faces = pc_adapter.adapt(self.points)
            self.n_faces = len(self.faces)

        return self.points, self.faces


def create_adapter(mode, logger=None, **kwargs):
    """Factory function to create the appropriate adapter."""
    if mode == "mesh":
        return MeshAdapter(logger=logger)
    elif mode == "pointcloud":
        return PointCloudAdapter(
            k=kwargs.get('k', 15),
            heat_t=kwargs.get('heat_t', 1.0),
            use_alpha=kwargs.get('use_alpha', True),
            max_edge_ratio=kwargs.get('max_edge_ratio', 3.0),
            logger=logger
        )
    elif mode == "graph":
        return GraphAdapter(
            use_spectral_pos=kwargs.get('use_spectral_pos', False),
            embed_dim=kwargs.get('embed_dim', 3),
            logger=logger
        )
    else:
        raise ValueError(f"Unknown adapter mode: {mode}")
