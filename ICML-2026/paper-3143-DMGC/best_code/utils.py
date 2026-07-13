import math
import random
import sys

import numpy as np
import scipy.sparse as sp
import torch
import _pickle as pkl
import networkx as nx
from scipy.linalg import fractional_matrix_power, inv
from sklearn.preprocessing import MinMaxScaler
import scipy.io as sio
import random
import pandas as pd
import itertools
from sklearn.neighbors import NearestNeighbors
import h5py
from sklearn.neighbors import kneighbors_graph
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def add_gaussian_noise(X, sigma):
    """Add Gaussian noise to the input."""
    noise = torch.normal(mean=0.0, std=sigma, size=X.size(), device=X.device, dtype=X.dtype)
    return X + noise


def normalize_spadj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

def normalize_weight(weights, p):

    ws = np.array(weights, dtype=np.float64)
    mu = ws.mean()
    f = 1 / (1 + np.exp(-p * (ws - mu)))

    return f / f.max()

def normalize_spfeatures(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)

    return mx


def normalize_features(x):
    rowsum = np.array(x.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_inv = r_inv.reshape((x.shape[0], -1))
    x = x * r_inv
    return x


def normalize_adj(x):
    rowsum = np.array(x.sum(1))
    r_inv = np.power(rowsum, -1.).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_inv = r_inv.reshape((x.shape[0], -1))
    x = x * r_inv
    return x


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_planetoid(dataset, path):
    path = path + dataset
    print('data loading.....')
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("{}/ind.{}.{}".format(path, dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("{}/ind.{}.test.index".format(path, dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features_unorm = sp.vstack((allx, tx)).tolil()
    features_unorm[test_idx_reorder, :] = features_unorm[test_idx_range, :]
    features = normalize_spfeatures(features_unorm)
    adj_noeye = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj_temp = adj_noeye

    # norm
    adj = adj_noeye + adj_noeye.T.multiply(adj_noeye.T > adj_noeye) - adj_noeye.multiply(adj_noeye.T > adj_noeye)
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize_spadj(adj)

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    adj = torch.FloatTensor(np.array(adj.todense()))
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.argmax(labels, -1))
    adj_labels = torch.FloatTensor(np.array(adj_noeye.todense()) != 0)
    feature_labels = torch.FloatTensor(np.array(features_unorm.todense()))
    print('complete data loading')

    return labels, adj, features, adj_labels, feature_labels


def load_multi(dataset, root):
    # load the data: x, tx, allx, graph
    if dataset == 'acm':
        path = root + 'ACM3025.mat'
    elif dataset == 'dblp':
        path = root + 'DBLP4057.mat'
    elif dataset == 'imdb':
        path = root + 'imdb5k.mat'
    data = sio.loadmat(path)

    if dataset == "acm":
        truelabels, truefeatures = data['label'], data['feature'].astype(float)
        N = truefeatures.shape[0]  # nodes num
        rownetworks = np.array([(data['PAP']).tolist(), (data['PLP']).tolist()])
    elif dataset == "dblp":
        truelabels, truefeatures = data['label'], data['features'].astype(float)
        N = truefeatures.shape[0]
        rownetworks = np.array([(data['net_APA']).tolist(), (data['net_APCPA']).tolist(), (data['net_APTPA']).tolist()])
        rownetworks[2] += np.eye(rownetworks[2].shape[0])
    elif dataset == 'imdb':
        truelabels, truefeatures = data['label'], data['feature'].astype(float)
        N = truefeatures.shape[0]
        rownetworks = np.array([(data['MAM']).tolist(), (data['MDM']).tolist(), (data['MYM']).tolist()])

    numView = rownetworks.shape[0]
    adjs_labels = []
    adjs = []
    feature_labels = torch.FloatTensor(np.array(truefeatures)).contiguous()
    features = torch.FloatTensor(normalize_features(truefeatures)).contiguous()
    for i in range(numView):
        adjs_labels.append(torch.FloatTensor(np.array(rownetworks[i])))
        adjs.append(torch.FloatTensor(normalize_adj(np.array(rownetworks[i]))))

    labels = torch.LongTensor(np.argmax(truelabels, -1)).contiguous()

    return labels, adjs, features, adjs_labels, feature_labels, numView


def eliminate_self_loops1(A):
    """Remove self-loops from the adjacency matrix."""
    A = A.tolil()
    A.setdiag(0)
    A = A.tocsr()
    A.eliminate_zeros()
    return A


def eliminate_self_loops(G):
    G.adj_matrix = eliminate_self_loops1(G.adj_matrix)
    return G


def create_subgraph(sparse_graph, _sentinel=None, nodes_to_remove=None, nodes_to_keep=None):
    """Create a graph with the specified subset of nodes.

    Exactly one of (nodes_to_remove, nodes_to_keep) should be provided, while the other stays None.
    Note that to avoid confusion, it is required to pass node indices as named arguments to this function.

    Parameters
    ----------
    sparse_graph : SparseGraph
        Input graph.
    _sentinel : None
        Internal, to prevent passing positional arguments. Do not use.
    nodes_to_remove : array-like of int
        Indices of nodes that have to removed.
    nodes_to_keep : array-like of int
        Indices of nodes that have to be kept.

    Returns
    -------
    sparse_graph : SparseGraph
        Graph with specified nodes removed.

    """
    # Check that arguments are passed correctly
    if _sentinel is not None:
        raise ValueError("Only call `create_subgraph` with named arguments',"
                         " (nodes_to_remove=...) or (nodes_to_keep=...)")
    if nodes_to_remove is None and nodes_to_keep is None:
        raise ValueError("Either nodes_to_remove or nodes_to_keep must be provided.")
    elif nodes_to_remove is not None and nodes_to_keep is not None:
        raise ValueError("Only one of nodes_to_remove or nodes_to_keep must be provided.")
    elif nodes_to_remove is not None:
        nodes_to_keep = [i for i in range(sparse_graph.num_nodes()) if i not in nodes_to_remove]
    elif nodes_to_keep is not None:
        nodes_to_keep = sorted(nodes_to_keep)
    else:
        raise RuntimeError("This should never happen.")

    sparse_graph.adj_matrix = sparse_graph.adj_matrix[nodes_to_keep][:, nodes_to_keep]
    if sparse_graph.attr_matrix is not None:
        sparse_graph.attr_matrix = sparse_graph.attr_matrix[nodes_to_keep]
    if sparse_graph.labels is not None:
        sparse_graph.labels = sparse_graph.labels[nodes_to_keep]
    if sparse_graph.node_names is not None:
        sparse_graph.node_names = sparse_graph.node_names[nodes_to_keep]
    return sparse_graph


def largest_connected_components(sparse_graph, n_components=1):
    """Select the largest connected components in the graph.

    Parameters
    ----------
    sparse_graph : SparseGraph
        Input graph.
    n_components : int, default 1
        Number of largest connected components to keep.

    Returns
    -------
    sparse_graph : SparseGraph
        Subgraph of the input graph where only the nodes in largest n_components are kept.

    """
    _, component_indices = sp.csgraph.connected_components(sparse_graph.adj_matrix)
    component_sizes = np.bincount(component_indices)
    components_to_keep = np.argsort(component_sizes)[::-1][:n_components]  # reverse order to sort descending
    nodes_to_keep = [
        idx for (idx, component) in enumerate(component_indices) if component in components_to_keep
    ]
    return create_subgraph(sparse_graph, nodes_to_keep=nodes_to_keep)


class SparseGraph:
    """Attributed labeled graph stored in sparse matrix form.

    """

    def __init__(self, adj_matrix, attr_matrix=None, labels=None,
                 node_names=None, attr_names=None, class_names=None, metadata=None):
        """Create an attributed graph.

        Parameters
        ----------
        adj_matrix : sp.csr_matrix, shape [num_nodes, num_nodes]
            Adjacency matrix in CSR format.
        attr_matrix : sp.csr_matrix or np.ndarray, shape [num_nodes, num_attr], optional
            Attribute matrix in CSR or numpy format.
        labels : np.ndarray, shape [num_nodes], optional
            Array, where each entry represents respective node's label(s).
        node_names : np.ndarray, shape [num_nodes], optional
            Names of nodes (as strings).
        attr_names : np.ndarray, shape [num_attr]
            Names of the attributes (as strings).
        class_names : np.ndarray, shape [num_classes], optional
            Names of the class labels (as strings).
        metadata : object
            Additional metadata such as text.

        """
        # Make sure that the dimensions of matrices / arrays all agree
        if sp.isspmatrix(adj_matrix):
            adj_matrix = adj_matrix.tocsr().astype(np.float32)
        else:
            raise ValueError("Adjacency matrix must be in sparse format (got {0} instead)"
                             .format(type(adj_matrix)))

        if adj_matrix.shape[0] != adj_matrix.shape[1]:
            raise ValueError("Dimensions of the adjacency matrix don't agree")

        if attr_matrix is not None:
            if sp.isspmatrix(attr_matrix):
                attr_matrix = attr_matrix.tocsr().astype(np.float32)
            elif isinstance(attr_matrix, np.ndarray):
                attr_matrix = attr_matrix.astype(np.float32)
            else:
                raise ValueError("Attribute matrix must be a sp.spmatrix or a np.ndarray (got {0} instead)"
                                 .format(type(attr_matrix)))

            if attr_matrix.shape[0] != adj_matrix.shape[0]:
                raise ValueError("Dimensions of the adjacency and attribute matrices don't agree")

        if labels is not None:
            if labels.shape[0] != adj_matrix.shape[0]:
                raise ValueError("Dimensions of the adjacency matrix and the label vector don't agree")

        if node_names is not None:
            if len(node_names) != adj_matrix.shape[0]:
                raise ValueError("Dimensions of the adjacency matrix and the node names don't agree")

        if attr_names is not None:
            if len(attr_names) != attr_matrix.shape[1]:
                raise ValueError("Dimensions of the attribute matrix and the attribute names don't agree")

        self.adj_matrix = adj_matrix
        self.attr_matrix = attr_matrix
        self.labels = labels
        self.node_names = node_names
        self.attr_names = attr_names
        self.class_names = class_names
        self.metadata = metadata

    def num_nodes(self):
        """Get the number of nodes in the graph."""
        return self.adj_matrix.shape[0]

    def num_edges(self):
        """Get the number of edges in the graph.

        For undirected graphs, (i, j) and (j, i) are counted as single edge.
        """
        if self.is_directed():
            return int(self.adj_matrix.nnz)
        else:
            return int(self.adj_matrix.nnz / 2)

    def get_neighbors(self, idx):
        """Get the indices of neighbors of a given node.

        Parameters
        ----------
        idx : int
            Index of the node whose neighbors are of interest.

        """
        return self.adj_matrix[idx].indices

    def is_directed(self):
        """Check if the graph is directed (adjacency matrix is not symmetric)."""
        return (self.adj_matrix != self.adj_matrix.T).sum() != 0

    def to_undirected(self):
        """Convert to an undirected graph (make adjacency matrix symmetric)."""
        if self.is_weighted():
            raise ValueError("Convert to unweighted graph first.")
        else:
            self.adj_matrix = self.adj_matrix + self.adj_matrix.T
            self.adj_matrix[self.adj_matrix != 0] = 1
        return self

    def is_weighted(self):
        """Check if the graph is weighted (edge weights other than 1)."""
        return np.any(np.unique(self.adj_matrix[self.adj_matrix != 0].A1) != 1)

    def to_unweighted(self):
        """Convert to an unweighted graph (set all edge weights to 1)."""
        self.adj_matrix.data = np.ones_like(self.adj_matrix.data)
        return self

    # Quality of life (shortcuts)
    def standardize(self):
        """Select the LCC of the unweighted/undirected/no-self-loop graph.

        All changes are done inplace.

        """
        G = self.to_unweighted().to_undirected()
        G = eliminate_self_loops(G)
        G = largest_connected_components(G, 1)
        return G

    def unpack(self):
        """Return the (A, X, z) triplet."""
        return self.adj_matrix, self.attr_matrix, self.labels


def load_npz_to_sparse_graph(file_name):
    """Load a SparseGraph from a Numpy binary file.

    Parameters
    ----------
    file_name : str
        Name of the file to load.

    Returns
    -------
    sparse_graph : SparseGraph
        Graph in sparse matrix format.

    """
    with np.load(file_name) as loader:
        loader = dict(loader)
        adj_matrix = sp.csr_matrix((loader['adj_data'], loader['adj_indices'], loader['adj_indptr']),
                                   shape=loader['adj_shape'])

        if 'attr_data' in loader:
            # Attributes are stored as a sparse CSR matrix
            attr_matrix = sp.csr_matrix((loader['attr_data'], loader['attr_indices'], loader['attr_indptr']),
                                        shape=loader['attr_shape'])
        elif 'attr_matrix' in loader:
            # Attributes are stored as a (dense) np.ndarray
            attr_matrix = loader['attr_matrix']
        else:
            attr_matrix = None

        if 'labels_data' in loader:
            # Labels are stored as a CSR matrix
            labels = sp.csr_matrix((loader['labels_data'], loader['labels_indices'], loader['labels_indptr']),
                                   shape=loader['labels_shape'])
        elif 'labels' in loader:
            # Labels are stored as a numpy array
            labels = loader['labels']
        else:
            labels = None

        node_names = loader.get('node_names')
        attr_names = loader.get('attr_names')
        class_names = loader.get('class_names')
        metadata = loader.get('metadata')

    return SparseGraph(adj_matrix, attr_matrix, labels, node_names, attr_names, class_names, metadata)


def load_data(dataset, path):
    """Load data."""
    if dataset in ['cora', 'citeseer', 'pubmed']:
        num_graph = 2
        labels, adjs, feature, adjs_labels, feature_label = load_planetoid(dataset, path)
        feature1_label = torch.matmul(feature_label, feature_label.t())
        feature1 = normalize_features(feature1_label)
        features = [feature, feature1]
        feature_labels = [feature_label, feature1_label]
        shared_feature = torch.cat(features, dim=-1)
        shared_feature_label = torch.cat(feature_labels, dim=1)
        adjs = [adjs, adjs.clone()]
        adjs_labels = [adjs_labels, adjs_labels.clone()]
    elif dataset in ['acm', 'dblp', 'imdb']:
        labels, adjs, feature, adjs_labels, feature_label, num_graph = load_multi(dataset, path)
        features = []
        feature_labels = []
        for _ in range(num_graph):
            features.append(feature)
            feature_labels.append(feature_label)
        shared_feature = feature
        shared_feature_label = feature_label

    elif dataset in ['chameleon', 'texas', 'squirrel', 'wisconsin', 'cornell']:
        labels, adjs, features, adjs_labels, feature_labels, shared_feature, shared_feature_label, num_graph = load_hete_data(dataset, 1)
    elif dataset == 'acm00':
        labels, adjs_labels, shared_feature, shared_feature_label, num_graph = load_synthetic_data(r=0.00, path=path)
    elif dataset == 'acm01':
        labels, adjs_labels, shared_feature, shared_feature_label, num_graph = load_synthetic_data(r=0.10, path=path)
    elif dataset == 'acm02':
        labels, adjs_labels, shared_feature, shared_feature_label, num_graph = load_synthetic_data(r=0.20, path=path)
    elif dataset == 'acm03':
        labels, adjs_labels, shared_feature, shared_feature_label, num_graph = load_synthetic_data(r=0.30, path=path)
    elif dataset == 'acm04':
        labels, adjs_labels, shared_feature, shared_feature_label, num_graph = load_synthetic_data(r=0.40, path=path)
    elif dataset == 'acm05':
        labels, adjs_labels, shared_feature, shared_feature_label, num_graph = load_synthetic_data(r=0.50, path=path)
    else:
        assert 'Dataset is not exist.'
    return labels, adjs_labels, shared_feature, shared_feature_label, num_graph


def load_graph(dataset, k):
    if k:
        path = 'graph/{}{}_graph.txt'.format(dataset, k)
    else:
        path = 'graph/{}_graph.txt'.format(dataset)

    data = np.loadtxt('data/{}.txt'.format(dataset))
    n, _ = data.shape

    idx = np.array([i for i in range(n)], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(path, dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(n, n), dtype=np.float32)

    # build symmetric adjacency matrix
    adj_noeye = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj_noeye + sp.eye(adj_noeye.shape[0])
    adj = normalize(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    adj_label = sparse_mx_to_torch_sparse_tensor(adj_noeye)

    return adj, adj_label


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def _to_tensor_same_device(x, device=None):
    """Utility: convert numpy/tensor to torch tensor on target device."""
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if device is not None and x.device != device:
        x = x.to(device)
    return x


def cal_homo_ratio(adj, label, self_loop=True):
    """
    同质性计算，支持GPU：输入可为numpy或torch，内部统一到torch并放在adj所在device。
    返回Python浮点值，避免在计算图中残留。
    """
    if isinstance(adj, np.ndarray):
        device = None
    else:
        device = adj.device
    adj_t = _to_tensor_same_device(adj, device=device).float()
    label_t = _to_tensor_same_device(label, device=adj_t.device).long()

    class_num = int(label_t.max().item() + 1)
    y_onehot = torch.nn.functional.one_hot(label_t, class_num).float()
    adj_y = torch.matmul(y_onehot, y_onehot.T)

    adj_used = adj_t
    if self_loop:
        adj_used = adj_t - torch.eye(adj_t.shape[0], device=adj_t.device, dtype=adj_t.dtype)

    homo = torch.sum(adj_y * adj_used)
    denom = torch.sum(adj_used)
    if denom.item() == 0:
        return 0.0
    homo_ratio = homo / denom
    return float(homo_ratio.item())



def load_hete_data(dataset, self_loop):
    path = './data/{}/'.format(dataset)

    f = np.loadtxt(path + '{}.feature'.format(dataset), dtype=float)
    l = np.loadtxt(path + '{}.label'.format(dataset), dtype=int)
    features = sp.csr_matrix(f, dtype=np.float32)
    features = torch.FloatTensor(np.array(features.todense()))

    label = torch.LongTensor(np.array(l))

    struct_edges = np.genfromtxt(path + '{}.edge'.format(dataset), dtype=np.int32)
    sedges = np.array(list(struct_edges), dtype=np.int32).reshape(struct_edges.shape)
    sadj = sp.coo_matrix((np.ones(sedges.shape[0]), (sedges[:, 0], sedges[:, 1])),
                         shape=(features.shape[0], features.shape[0]), dtype=np.float32)
    sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)
    sadj = sadj + self_loop * sp.eye(sadj.shape[0])
    adjs_labels = [torch.FloatTensor(sadj.todense())]
    adjs = [torch.FloatTensor(normalize_adj(a.cpu().numpy())) for a in adjs_labels]

    feature_labels = [features]
    features = [normalize_features(x) for x in feature_labels]
    shared_feature = features[0]
    shared_feature_label = feature_labels[0]

    adjs_labels.append(torch.FloatTensor(sadj.todense()))
    adjs.append(adjs[0])
    num_graph = 2
    return label, adjs, features, adjs_labels, feature_labels, shared_feature, shared_feature_label, num_graph

def load_synthetic_data(r=0.00, path='./data/'):
    dataset = 'acm_hete_r{:.2f}.npz'.format(r)
    acm_dataset = np.load(path + dataset)
    shared_feature = torch.FloatTensor(acm_dataset['feature'])
    shared_feature_label = torch.FloatTensor(acm_dataset['feature_label'])
    adj_hete_v0 = torch.FloatTensor(acm_dataset['adj_hete_v0'])
    adj_hete_v1 = torch.FloatTensor(acm_dataset['adj_hete_v1'])
    labels = torch.LongTensor(acm_dataset['labels'])
    adjs_labels = []
    adjs_labels.append(adj_hete_v0)
    adjs_labels.append(adj_hete_v1)
    graph_num = len(adjs_labels)
    return labels, adjs_labels, shared_feature, shared_feature_label, graph_num



def load_data_airports(dataset_str, data_path):
    """Read the data and preprocess the task information."""
    dataset_G = data_path + "Airports/" + "{}-airports.edgelist".format(dataset_str)
    dataset_L = data_path + "Airports/" + "labels-{}-airports.txt".format(dataset_str)
    label_raw, nodes = [], []
    with open(dataset_L, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline()
            if not lines:
                break
            node, label = lines.split()
            if label == 'label': continue
            label_raw.append(int(label))
            nodes.append(int(node))
    label_raw = np.array(label_raw)
    labels = torch.from_numpy(label_raw)
    # print(label_raw)
    G = nx.read_edgelist(open(dataset_G, 'rb'), nodetype=int)
    adj = nx.adjacency_matrix(G, nodelist=nodes)
    adj_torch = torch.tensor(adj.toarray()).to(torch.float32)
    adjs_labels = [adj_torch]

    # task information
    degreeNode = np.sum(adj, axis=1)
    degreeNode = degreeNode.astype(np.int32)
    features = np.zeros((degreeNode.size, degreeNode.max() + 1))
    features[np.arange(degreeNode.size), degreeNode] = 1
    features = sp.csr_matrix(features)
    features = torch.tensor(features.toarray())
    features = features.to(torch.float32)

    num_graph = 2
    adjs_labels.append(torch.tensor(adj.toarray()).to(torch.float32))

    return labels, adjs_labels, features, num_graph



def load_synthetic_data(r=0.00, path='./data/'):
    dataset = 'acm_hete_r{:.2f}.npz'.format(r)
    acm_dataset = np.load(path + dataset)
    shared_feature = torch.FloatTensor(acm_dataset['feature'])
    shared_feature_label = torch.FloatTensor(acm_dataset['feature_label'])
    adj_hete_v0 = torch.FloatTensor(acm_dataset['adj_hete_v0'])
    adj_hete_v1 = torch.FloatTensor(acm_dataset['adj_hete_v1'])
    labels = torch.LongTensor(acm_dataset['labels'])
    adjs_labels = []
    adjs_labels.append(adj_hete_v0)
    adjs_labels.append(adj_hete_v1)
    graph_num = len(adjs_labels)
    return labels, adjs_labels, shared_feature, shared_feature_label, graph_num





