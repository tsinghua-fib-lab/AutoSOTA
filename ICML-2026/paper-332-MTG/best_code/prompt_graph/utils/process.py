import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
from ..data import load4node
from torch_geometric.utils import to_scipy_sparse_matrix


def parse_skipgram(fname):
    with open(fname) as f:
        toks = list(f.read().split())
    nb_nodes = int(toks[0])
    nb_features = int(toks[1])
    ret = np.empty((nb_nodes, nb_features))
    it = 2
    for i in range(nb_nodes):
        cur_nd = int(toks[it]) - 1
        it += 1
        for j in range(nb_features):
            cur_ft = float(toks[it])
            ret[cur_nd][j] = cur_ft
            it += 1
    return ret


# Process a (subset of) a TU dataset into standard form
def process_tu(data, class_num, node_class):
    nb_nodes = data.num_nodes
    nb_graphs = data.num_graphs
    # print("len",nb_graphs)
    ft_size = data.num_features

    node_class_num = range(node_class)

    # print("data",data)
    labels = np.zeros((nb_graphs, class_num))
    for g in range(nb_graphs):
        if g == 0:
            # sizes = data[g].x.shape[0]
            features = data[g].x[:, node_class_num]
            rawlabels = data[g].y[0]
            # masks[g, :sizes[g]] = 1.0
            e_ind = data[g].edge_index
            # print("e_ind",e_ind)
            coo = sp.coo_matrix((np.ones(e_ind.shape[1]), (e_ind[0, :], e_ind[1, :])),
                                shape=(features.shape[0], features.shape[0]))
            # print("coo",coo)
            adjacency = coo.todense()
        else:
            tmpfeature = data[g].x[:, node_class_num]
            features = np.row_stack((features, tmpfeature))
            tmplabel = data[g].y[0]
            rawlabels = np.row_stack((rawlabels, tmplabel))
            e_ind = data[g].edge_index
            coo = sp.coo_matrix((np.ones(e_ind.shape[1]), (e_ind[0, :], e_ind[1, :])),
                                shape=(tmpfeature.shape[0], tmpfeature.shape[0]))
            # print("coo",coo)
            tmpadj = coo.todense()
            zero = np.zeros((adjacency.shape[0], tmpfeature.shape[0]))
            tmpadj1 = np.column_stack((adjacency, zero))
            tmpadj2 = np.column_stack((zero.T, tmpadj))
            adjacency = np.row_stack((tmpadj1, tmpadj2))

    for x in range(nb_graphs):
        if nb_graphs == 1:
            labels[0][rawlabels.item()] = 1
            break
        labels[x][rawlabels[x][0]] = 1

    adj = sp.csr_matrix(adjacency)

    return features, adj


def micro_f1(logits, labels):
    # Compute predictions
    preds = torch.round(nn.Sigmoid()(logits))

    # Cast to avoid trouble
    preds = preds.long()
    labels = labels.long()

    # Count true positives, true negatives, false positives, false negatives
    tp = torch.nonzero(preds * labels).shape[0] * 1.0
    tn = torch.nonzero((preds - 1) * (labels - 1)).shape[0] * 1.0
    fp = torch.nonzero(preds * (labels - 1)).shape[0] * 1.0
    fn = torch.nonzero((preds - 1) * labels).shape[0] * 1.0

    # Compute micro-f1 score
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = (2 * prec * rec) / (prec + rec)
    return f1


"""
 Prepare adjacency matrix by expanding up to a given neighbourhood.
 This will insert loops on every node.
 Finally, the matrix is converted to bias vectors.
 Expected shape: [graph, nodes, nodes]
"""


def adj_to_bias(adj, sizes, nhood=1):
    nb_graphs = adj.shape[0]
    mt = np.empty(adj.shape)
    for g in range(nb_graphs):
        mt[g] = np.eye(adj.shape[1])
        for _ in range(nhood):
            mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))
        for i in range(sizes[g]):
            for j in range(sizes[g]):
                if mt[g][i][j] > 0.0:
                    mt[g][i][j] = 1.0
    return -1e9 * (1.0 - mt)


###############################################
# This section of code adapted from tkipf/gcn #
###############################################

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


# def load_pyg_data(dataset_str): # {'pubmed', 'citeseer', 'cora'}
#     """Load data."""
#     if dataset_str == 'Cora':
#         dataset_str1 = 'cora'
#     if dataset_str == 'CiteSeer':
#         dataset_str1 = 'citeseer'
#     if dataset_str == 'PubMed':
#         dataset_str1 = 'pubmed'
#     current_path = os.path.dirname(__file__)
#     names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
#     objects = []
#     for i in range(len(names)):
#         with open("./data/Planetoid/"+ dataset_str +"/raw/ind.{}.{}".format(dataset_str1, names[i]), 'rb') as f:
#             objects.append(pkl.load(f, encoding='latin1'))


#     x, y, tx, ty, allx, ally, graph = tuple(objects)
#     test_idx_reorder = parse_index_file("./data/Planetoid/"+ dataset_str +"/raw/ind.{}.test.index".format(dataset_str1))
#     test_idx_range = np.sort(test_idx_reorder)

#     if dataset_str1 == 'citeseer':
#         # Fix citeseer dataset (there are some isolated nodes in the graph)
#         # Find isolated nodes, add them as zero-vecs into the right position
#         test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
#         tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
#         tx_extended[test_idx_range-min(test_idx_range), :] = tx
#         tx = tx_extended
#         ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
#         ty_extended[test_idx_range-min(test_idx_range), :] = ty
#         ty = ty_extended

#     features = sp.vstack((allx, tx)).tolil()
#     features[test_idx_reorder, :] = features[test_idx_range, :]
#     adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

#     labels = np.vstack((ally, ty))
#     labels[test_idx_reorder, :] = labels[test_idx_range, :]

#     return adj, features, labels


def load_data(dataset):
    data, _, _ = load4node(dataset)
    adj = to_scipy_sparse_matrix(data.edge_index).tocsr()

    # Convert features to dense format and then to scipy sparse matrix in lil format
    features = sp.lil_matrix(data.x.numpy())

    # Convert labels to one-hot encoding
    labels = np.zeros((data.num_nodes, data.y.max().item() + 1))
    labels[np.arange(data.num_nodes), data.y.numpy()] = 1

    return adj, features, labels


def sparse_to_tuple(sparse_mx, insert_batch=False):
    """Convert sparse matrix to tuple representation."""
    """Set insert_batch=True if you want to insert a batch dimension."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        if insert_batch:
            coords = np.vstack((np.zeros(mx.row.shape[0]), mx.row, mx.col)).transpose()
            values = mx.data
            shape = (1,) + mx.shape
        else:
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def standardize_data(f, train_mask):
    """Standardize feature matrix and convert to tuple representation"""
    # standardize data
    f = f.todense()
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = f[:, np.squeeze(np.array(sigma > 0))]
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = (f - mu) / sigma
    return f


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
