import numpy as np
import networkx as nx
import torch
from matplotlib.tri import Triangulation
from scipy.spatial.distance import pdist, squareform

from torch_geometric.data import Data
import torch_geometric.utils as tgut

def nx2tg(G, node_attr='node_attr'):
    """Convert NetworkX graph to PyTorch Geometric Data object."""
    # Safely handle graphs with 0 edges
    if len(G.edges) == 0:
        edgelist = torch.empty((2, 0), dtype=torch.long)
    else:
        edgelist = torch.LongTensor([list(e) for e in G.edges]).t()

    edgelist = tgut.to_undirected(edgelist)
    D = Data(x=torch.ones(len(G)), edge_index=edgelist)

    if node_attr is not None:
        X = torch.Tensor([G.nodes[i][node_attr] for i in G.nodes])
        if len(X.shape) == 1:
            X = X.view(-1, 1)
        D.x = X
    return D

def surface_uniform(n, fz=lambda x, y: 0):
    """Generate uniform nodes on a surface."""
    pos = np.zeros((n, 3))
    pos[:, 0], pos[:, 1] = np.random.rand(n), np.random.rand(n)
    pos[:, 2] = fz(pos[:, 0], pos[:, 1])
    tri = Triangulation(pos[:, 0], pos[:, 1])
    return pos, tri.triangles

def generate_edges_gaussian(X, sigma, alpha):
    """
    Pure NumPy/SciPy implementation for edge generation.
    Utilizes pdist to avoid slow nested loops.
    """
    n = X.shape[0]
    
    # Compute squared Euclidean distance matrix for all node pairs
    sq_dists = pdist(X, metric='sqeuclidean')
    sq_dists_mat = squareform(sq_dists)
    
    # Compute connection probabilities
    probs_mat = alpha * np.exp(-sq_dists_mat / (2 * sigma**2))
    
    # Extract lower triangle indices to avoid self-loops and duplicate edges
    i, j = np.tril_indices(n, k=-1)
    
    # Generate random mask for edge sampling
    edge_mask = np.random.rand(len(i)) < probs_mat[i, j]
    
    # Return list of sampled edges
    return list(zip(i[edge_mask], j[edge_mask]))

def random_graph_similarity(X, f=None, alpha=1, bandwidth=1):
    """Construct a random graph based on node similarity."""
    n = X.shape[0]
    G = nx.empty_graph(n)

    for i in range(n):
        G.nodes[i]['latent'] = X[i, :]
        if f is not None:
            G.nodes[i]['node_attr'] = f(X[i, :])

    edgelist = generate_edges_gaussian(X, bandwidth, alpha)
    G.add_edges_from(edgelist)

    if f is None:
        for i in range(n):
            G.nodes[i]['node_attr'] = G.degree[i]

    return G