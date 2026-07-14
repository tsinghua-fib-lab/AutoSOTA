"""Geometry module."""

import numpy as np
import ot
import scipy.sparse as sp
import torch
import torch_geometric.utils as PyGu
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.manifold import TSNE
from sklearn.manifold import Isomap
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from torch_geometric.nn import knn_graph
from torch_geometric.nn import radius_graph
from torch_scatter import scatter_add

try:
    from .ptu_dijkstra import connections, tangent_frames  # type: ignore[attr-defined]  # isort:skip
except ModuleNotFoundError as err:  # pragma: no cover - fallback path
    from ._ptu_loader import build_and_import_ptu_dijkstra

    _ptu_mod = build_and_import_ptu_dijkstra(err)
    connections = _ptu_mod.connections
    tangent_frames = _ptu_mod.tangent_frames

from data_processing.cknn import cknneighbors_graph  # isort:skip
from . import utils  # isort:skip


def furthest_point_sampling(x, N=None, spacing=0.0, start_idx=0):
    """A greedy O(N^2) algorithm to do furthest points sampling

    Args:
        x (nxdim matrix): input data
        N (int): number of sampled points
        stop_crit: when reaching this fraction of the total manifold diameter, we stop sampling
        start_idx: index of starting node

    Returns:
        perm: node indices of the N sampled points
        lambdas: list of distances of furthest points
    """
    if spacing == 0.0:
        return torch.arange(len(x)), None

    D = utils.np2torch(pairwise_distances(x))
    n = D.shape[0] if N is None else N
    diam = D.max()

    perm = torch.zeros(n, dtype=torch.int64)
    perm[0] = start_idx
    lambdas = torch.zeros(n)
    ds = D[start_idx, :].flatten()
    for i in range(1, n):
        idx = torch.argmax(ds)
        perm[i] = idx
        lambdas[i] = ds[idx]
        ds = torch.minimum(ds, D[idx, :])

        if N is None:
            if lambdas[i] / diam < spacing:
                perm = perm[:i]
                lambdas = lambdas[:i]
                break

    assert len(perm) == len(np.unique(perm)), "Returned duplicated points"

    return perm, lambdas


def cluster(x, cluster_typ="kmeans", n_clusters=15, seed=0):
    """Cluster data.

    Args:
        x (nxdim matrix) data
        cluster_typ: Clustering method.
        n_clusters: Number of clusters.
        seed: seed

    Returns:
        clusters: sklearn cluster object
    """
    clusters = {}
    if cluster_typ == "kmeans":
        kmeans = KMeans(n_clusters=n_clusters, random_state=seed).fit(x)
        clusters["n_clusters"] = n_clusters
        clusters["labels"] = kmeans.labels_
        clusters["centroids"] = kmeans.cluster_centers_
    elif cluster_typ == "meanshift":
        meanshift = MeanShift(bandwidth=n_clusters).fit(x)
        clusters["n_clusters"] = len(set(meanshift.labels_))
        clusters["labels"] = meanshift.labels_
        clusters["centroids"] = meanshift.cluster_centers_
    else:
        raise NotImplementedError

    return clusters


def embed(x, embed_typ="umap", dim_emb=2, manifold=None, verbose=True, seed=0, **kwargs):
    """Embed data to 2D space.

    Args:
        x (nxdim matrix): data
        embed_typ: embedding method. The default is 'tsne'.

    Returns:
        emb (nx2 matrix): embedded data
    """
    if x.shape[1] <= 2:
        print(
            f"\n No {embed_typ} embedding performed. Embedding seems to be \
              already in 2D."
        )
        return x, None

    if embed_typ == "tsne":
        x = StandardScaler().fit_transform(x)
        if manifold is not None:
            raise Exception("t-SNE cannot fit on existing manifold")

        emb = TSNE(init="random", learning_rate="auto", random_state=seed).fit_transform(x)

    elif embed_typ == "umap":
        import umap  # isort:skip
        x = StandardScaler().fit_transform(x)
        if manifold is None:
            manifold = umap.UMAP(n_components=dim_emb, random_state=seed, **kwargs).fit(x)

        emb = manifold.transform(x)

    elif embed_typ == "MDS":
        if manifold is not None:
            raise Exception("MDS cannot fit on existing manifold")

        emb = MDS(
            n_components=dim_emb, n_init=20, dissimilarity="precomputed", random_state=seed
        ).fit_transform(x)

    elif embed_typ == "PCA":
        if manifold is None:
            manifold = PCA(n_components=dim_emb).fit(x)

        emb = manifold.transform(x)

    elif embed_typ == "Isomap":
        radius = pairwise_distances(x)
        radius = 0.1 * (radius.max() - radius.min())
        if manifold is None:
            manifold = Isomap(n_components=dim_emb, n_neighbors=None, radius=radius).fit(x)

        emb = manifold.transform(x)

    else:
        raise NotImplementedError

    if verbose:
        print(f"Performed {embed_typ} embedding on embedded results.")

    return emb, manifold


def relabel_by_proximity(clusters):
    """Update clusters labels such that nearby clusters in the embedding get similar labels.

    Args:
        clusters: sklearn object containing 'centroids', 'n_clusters', 'labels' as attributes

    Returns:
        clusters: sklearn object with updated labels
    """
    pd = pairwise_distances(clusters["centroids"], metric="euclidean")
    pd += np.max(pd) * np.eye(clusters["n_clusters"])

    mapping = {}
    id_old = 0
    for i in range(clusters["n_clusters"]):
        id_new = np.argmin(pd[id_old, :])
        while id_new in mapping:
            pd[id_old, id_new] += np.max(pd)
            id_new = np.argmin(pd[id_old, :])
        mapping[id_new] = i
        id_old = id_new

    labels = clusters["labels"]
    clusters["labels"] = np.array([mapping[label] for label in labels])
    clusters["centroids"] = clusters["centroids"][list(mapping.keys())]

    return clusters


def compute_distribution_distances(clusters=None, data=None, slices=None):
    """Compute the distance between clustered distributions across datasets.

    Args:
        clusters: sklearn object containing 'centroids', 'slices', 'labels' as attributes

    Returns:
        dist: distance matrix
        gamma: optimal transport matrix
        centroid_distances: distances between cluster centroids
    """
    s = slices
    pdists, cdists = None, None
    if clusters is not None:
        # compute discrete measures supported on cluster centroids
        labels = clusters["labels"]
        labels = [labels[s[i] : s[i + 1]] + 1 for i in range(len(s) - 1)]
        nc, nl = clusters["n_clusters"], len(labels)
        bins_dataset = []
        for l_ in labels:  # loop over datasets
            bins = [(l_ == i + 1).sum() for i in range(nc)]  # loop over clusters
            bins = np.array(bins)
            bins_dataset.append(bins / bins.sum())

        cdists = pairwise_distances(clusters["centroids"])
        gamma = np.zeros([nl, nl, nc, nc])

    elif data is not None:
        # compute empirical measures from datapoints
        nl = len(s) - 1

        bins_dataset = []
        for i in range(nl):
            mu = np.ones(s[i + 1] - s[i])
            mu /= len(mu)
            bins_dataset.append(mu)

        pdists = pairwise_distances(data.emb)
    else:
        raise Exception("No input provided.")

    # compute distance between measures
    dist = np.zeros([nl, nl])
    for i in range(nl):
        for j in range(i + 1, nl):
            mu, nu = bins_dataset[i], bins_dataset[j]

            if data is not None and pdists is not None:
                cdists = pdists[s[i] : s[i + 1], s[j] : s[j + 1]]

            dist[i, j] = ot.emd2(mu, nu, cdists)
            dist[j, i] = dist[i, j]

            if clusters is not None:
                gamma[i, j, ...] = ot.emd(mu, nu, cdists)
                gamma[j, i, ...] = gamma[i, j, ...]
            else:
                gamma = None

    return dist, gamma


def neighbour_vectors(pos, edge_index):
    """Local out-going edge vectors around each node.

    Args:
        pos (nxdim matrix): node positions
        edge_index (2xE matrix): edge indices

    Returns:
        nvec (Exdim matrix): neighbourhood vectors.

    """
    ei, ej = edge_index[0], edge_index[1]
    nvec = pos[ej] - pos[ei]

    return nvec


def project_gauge_to_neighbours(nvec, gauges, edge_index):
    """Project the gauge vectors to local edge vectors.

    Args:
        nvec (Exdim matrix): neighbourhood vectors
        local_gauge (dimxnxdim torch tensor): if None, global gauge is generated

    Returns:
        list of (nxn) torch tensors of projected components
    """
    n, _, d = gauges.shape
    ei = edge_index[0]
    proj = torch.einsum("bi,bic->bc", nvec, gauges[ei])

    proj = [sp.coo_matrix((proj[:, i], (edge_index)), [n, n]).tocsr() for i in range(d)]

    return proj


def gradient_op(pos, edge_index, gauges):
    """Directional derivative kernel from Beaini et al. 2021.

    Args:
        pos (nxdim Matrix) node positions
        edge_index (2x|E| matrix) edge indices
        gauge (list): orthonormal unit vectors

    Returns:
        list of (nxn) Anisotropic kernels
    """
    nvec = neighbour_vectors(pos, edge_index)
    F = project_gauge_to_neighbours(nvec, gauges, edge_index)

    K = []
    for _F in F:
        norm = np.repeat(np.add.reduceat(np.abs(_F.data), _F.indptr[:-1]), np.diff(_F.indptr))
        # Avoid divide-by-zero when a node has zero summed magnitude; keep original values in that case
        norm_safe = norm.copy()
        norm_safe[norm_safe == 0] = 1.0
        _F.data /= norm_safe
        _F -= sp.diags(np.array(_F.sum(1)).flatten())
        _F = _F.tocoo()
        K.append(torch.sparse_coo_tensor(np.vstack([_F.row, _F.col]), _F.data.data))

    return K


def normalize_sparse_matrix(sp_tensor):
    """Normalize sparse matrix."""
    row_sum = sp_tensor.sum(axis=1)
    row_sum[row_sum == 0] = 1  # to avoid divide by zero
    sp_tensor = sp_tensor.multiply(1.0 / row_sum)

    return sp_tensor


def global_to_local_frame(x, gauges, length_correction=False, reverse=False):
    """Transform signal into local coordinates."""

    if reverse:
        proj = torch.einsum("bji,bi->bj", gauges, x)
    else:
        proj = torch.einsum("bij,bi->bj", gauges, x)

    if length_correction:
        norm_x = x.norm(p=2, dim=1, keepdim=True)
        norm_proj = proj.norm(p=2, dim=1, keepdim=True)
        proj = proj / norm_proj * norm_x

    return proj


def project_to_gauges(x, gauges, dim=2):
    """Project to gauges."""
    coeffs = torch.einsum("bij,bi->bj", gauges, x)
    return torch.einsum("bj,bij->bi", coeffs[:, :dim], gauges[:, :, :dim])


def manifold_dimension(Sigma, frac_explained=0.9):
    """Estimate manifold dimension based on singular vectors"""

    if frac_explained == 1.0:
        return Sigma.shape[1]

    Sigma **= 2
    Sigma /= Sigma.sum(1, keepdim=True)
    Sigma = Sigma.cumsum(dim=1)
    var_exp = Sigma.mean(0) - Sigma.std(0)
    dim_man = torch.where(var_exp >= frac_explained)[0][0] + 1

    print("\nFraction of variance explained: ", var_exp.tolist())

    return int(dim_man)


def fit_graph(x, graph_type="cknn", par=1, delta=1.0, metric="euclidean"):
    """Fit graph to node positions"""

    if graph_type == "cknn":
        edge_index = cknneighbors_graph(x, n_neighbors=par, delta=delta, metric=metric).tocoo()
        edge_index = np.vstack([edge_index.row, edge_index.col])
        edge_index = utils.np2torch(edge_index, dtype="double")

    elif graph_type == "knn":
        edge_index = knn_graph(x, k=par)
        edge_index = PyGu.add_self_loops(edge_index)[0]

    elif graph_type == "radius":
        edge_index = radius_graph(x, r=par)
        edge_index = PyGu.add_self_loops(edge_index)[0]

    else:
        raise NotImplementedError

    assert is_connected(edge_index), "Graph is not connected! Try increasing k."

    edge_index = PyGu.to_undirected(edge_index)
    pdist = torch.nn.PairwiseDistance(p=2)
    edge_weight = pdist(x[edge_index[0]], x[edge_index[1]])
    edge_weight = 1 / edge_weight

    return edge_index, edge_weight


def is_connected(edge_index):
    """Check if it is connected."""
    adj = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.shape[1]))
    deg = torch.sparse.sum(adj, 0).values()

    return (deg > 1).all()


def compute_laplacian(data, normalization="rw"):
    """Compute Laplacian."""
    edge_index, edge_attr = PyGu.get_laplacian(
        data.edge_index,
        edge_weight=data.edge_weight,
        normalization=normalization,
        num_nodes=data.num_nodes,
    )

    # return PyGu.to_dense_adj(edge_index, edge_attr=edge_attr).squeeze()
    return torch.sparse_coo_tensor(edge_index, edge_attr).coalesce()


def compute_connection_laplacian(data, R, normalization="rw"):
    r"""Connection Laplacian

    Args:
        data: Pytorch geometric data object.
        R (nxnxdxd): Connection matrices between all pairs of nodes. Default is None,
            in case of a global coordinate system.
        normalization: None, 'rw'
                 1. None: No normalization
                 :math:`\mathbf{L} = \mathbf{D} - \mathbf{A}`

                 2. "rw"`: Random-walk normalization
                 :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1} \mathbf{A}`

    Returns:
        ndxnd normalised connection Laplacian matrix.
    """
    n = data.x.shape[0]
    d = R.size()[0] // n

    # unnormalised (combinatorial) laplacian, to be normalised later
    L = compute_laplacian(data, normalization=None)  # .to_sparse()

    # rearrange into block form (kron(L, ones(d,d)))
    edge_index = utils.expand_edge_index(L.indices(), dim=d)
    L = torch.sparse_coo_tensor(edge_index, L.values().repeat_interleave(d * d))

    # unnormalised connection laplacian
    # Lc(i,j) = L(i,j)*R(i,j) if (i,j)=\in E else 0
    Lc = L * R

    # normalize
    edge_index, edge_weight = PyGu.remove_self_loops(data.edge_index, data.edge_weight)
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)

    # degree matrix
    deg = scatter_add(edge_weight, edge_index[0], dim=0, dim_size=n)

    if normalization == "rw":
        deg_inv = 1.0 / deg
        deg_inv.masked_fill_(deg_inv == float("inf"), 0)
        deg_inv = deg_inv.repeat_interleave(d, dim=0)
        Lc = torch.diag(deg_inv).to_sparse() @ Lc

    return Lc.coalesce()


def compute_gauges(data, dim_man=None, n_geodesic_nb=10, processes=1):
    """Orthonormal gauges for the tangent space at each node.

    Args:
        data: Pytorch geometric data object.
        n_geodesic_nb: number of geodesic neighbours. The default is 10.
        processes: number of CPUs to use

    Returns:
        gauges (nxdimxdim matrix): Matrix containing dim unit vectors for each node.
        Sigma: Singular valued
    """
    X = data.pos.numpy().astype(np.float64)
    A = PyGu.to_scipy_sparse_matrix(data.edge_index).tocsr()

    # make chunks for data processing
    sl = data._slice_dict["x"]  # pylint: disable=protected-access
    n = len(sl) - 1
    X = [X[sl[i] : sl[i + 1]] for i in range(n)]
    A = [A[sl[i] : sl[i + 1], :][:, sl[i] : sl[i + 1]] for i in range(n)]

    if dim_man is None:
        dim_man = X[0].shape[1]

    inputs = [X, A, dim_man, n_geodesic_nb]
    out = utils.parallel_proc(
        _compute_gauges,
        range(n),
        inputs,
        processes=processes,
        desc="\n---- Computing tangent spaces...",
    )

    gauges, Sigma = zip(*out)
    gauges, Sigma = np.vstack(gauges), np.vstack(Sigma)

    return utils.np2torch(gauges), utils.np2torch(Sigma)


def _compute_gauges(inputs, i):
    """Helper function to compute_gauges()"""
    X_chunks, A_chunks, dim_man, n_geodesic_nb = inputs
    gauges, Sigma = tangent_frames(X_chunks[i], A_chunks[i], dim_man, n_geodesic_nb)

    return gauges, Sigma


def compute_connections(data, gauges, processes=1):
    """Find smallest rotations R between gauges pairs. It is assumed that the first
    row of edge_index is what we want to align to, i.e.,
    gauges(i) = gauges(j)@R[i,j].T

    R[i,j] is optimal rotation that minimises ||X - RY||_F computed by SVD:
    X, Y = gauges[i].T, gauges[j].T
    U, _, Vt = scipy.linalg.svd(X.T@Y)
    R[i,j] = U@Vt

    Args:
        data: Pytorch geometric data object
        gauges (nxdxd matrix): Orthogonal unit vectors for each node
        processes: number of CPUs to use

    Returns:
        (n*dim,n*dim) matrix of rotation matrices
    """
    gauges = np.array(gauges, dtype=np.float64)
    A = PyGu.to_scipy_sparse_matrix(data.edge_index).tocsr()

    # make chunks for data processing
    sl = data._slice_dict["x"]  # pylint: disable=protected-access
    dim_man = gauges.shape[-1]

    n = len(sl) - 1
    gauges = [gauges[sl[i] : sl[i + 1]] for i in range(n)]
    A = [A[sl[i] : sl[i + 1], :][:, sl[i] : sl[i + 1]] for i in range(n)]

    inputs = [gauges, A, dim_man]
    out = utils.parallel_proc(
        _compute_connections,
        range(n),
        inputs,
        processes=processes,
        desc="\n---- Computing connections...",
    )

    return utils.to_block_diag(out)


def _compute_connections(inputs, i):
    """helper function to compute_connections()"""
    gauges_chunks, A_chunks, dim_man = inputs

    R = connections(gauges_chunks[i], A_chunks[i], dim_man)

    edge_index = np.vstack([A_chunks[i].tocoo().row, A_chunks[i].tocoo().col])
    edge_index = torch.tensor(edge_index)
    edge_index = utils.expand_edge_index(edge_index, dim=R.shape[-1])
    return torch.sparse_coo_tensor(edge_index, R.flatten(), dtype=torch.float32).coalesce()


def compute_eigendecomposition(A, k=None, eps=1e-8):
    """Eigendecomposition of a square matrix A.

    Args:
        A: square matrix A
        k: number of eigenvectors
        eps: small error term

    Returns:
        evals (k): eigenvalues of the Laplacian
        evecs (V,k): matrix of eigenvectors of the Laplacian
    """
    if A is None:
        return None

    if k is None:
        A = A.to_dense().double()
    else:
        indices, values, size = A.indices(), A.values(), A.size()
        A = sp.coo_array((values, (indices[0], indices[1])), shape=size)

    failcount = 0
    while True:
        try:
            if k is None:
                evals, evecs = torch.linalg.eigh(A)  # pylint: disable=not-callable
            else:
                evals, evecs = sp.linalg.eigsh(A, k=k, which="SM")
                evals, evecs = torch.tensor(evals), torch.tensor(evecs)

            evals = torch.clamp(evals, min=0.0)
            evecs *= np.sqrt(len(evecs))

            break
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(e)
            if failcount > 3:
                raise ValueError("failed to compute eigendecomp") from e
            failcount += 1
            print("--- decomp failed; adding eps ===> count: " + str(failcount))
            if k is None:
                A += torch.eye(A.shape[0]) * (eps * 10 ** (failcount - 1))
            else:
                A += sp.eye(A.shape[0]) * (eps * 10 ** (failcount - 1))

    return evals.float(), evecs.float()
