import random
import torch
from torch_geometric.data import Data
import scipy.io as sio
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score, average_precision_score
import torch.nn.functional as F
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def test_eval(labels, probs):
    score = {}
    with torch.no_grad():
        if torch.is_tensor(labels):
            labels = labels.cpu().numpy()
        if torch.is_tensor(probs):
            probs = probs.cpu().numpy()

        if np.isnan(probs).any():
            probs = np.nan_to_num(probs)

        try:
            score['AUROC'] = roc_auc_score(labels, probs)
            score['AUPRC'] = average_precision_score(labels, probs)
        except ValueError:
            score['AUROC'] = 0.0
            score['AUPRC'] = 0.0

    return score

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def preprocess_features(features):
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()

def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def cosine_similarity_matrix(x):
    sim = torch.mm(x, x.t())
    sim.fill_diagonal_(0)
    return sim

class Dataset:
    def __init__(self, dims, name='cora', prefix='./dataset/'):
        self.shot_mask = None
        self.shot_idx = None
        self.graph = None
        self.x_list = None
        self.name = name

        data = sio.loadmat(f"{prefix + name}.mat")
        adj = data['Network']
        feat = data['Attributes']
        adj_sp = sp.csr_matrix(adj)
        row, col = adj_sp.nonzero()
        edge_index = torch.tensor([row, col], dtype=torch.long)
        if name in ['Amazon', 'YelpChi', 'tolokers', 'tfinance']:
            feat = sp.lil_matrix(feat)
            feat = preprocess_features(feat)
        else:
            feat = sp.lil_matrix(feat).toarray()
        feat = torch.FloatTensor(feat)

        adj = data['Network'] if 'Network' in data else data['A']
        if name in ['YelpChi', 'Facebook']:
            adj_norm = normalize_adj(adj)
        else:
            adj_norm = normalize_adj(adj + sp.eye(adj.shape[0]))
        adj_norm = sparse_mx_to_torch_sparse_tensor(adj_norm)
        label = data['Label'] if ('Label' in data) else data['gnd']

        self.label = label
        self.adj_norm = adj_norm
        self.feat = feat
        ano_labels = torch.tensor(np.squeeze(np.array(self.label)), dtype=torch.float)

        data = Data(x=torch.tensor(self.feat, dtype=torch.float),
                    x_list=self.x_list,
                    adj=self.adj_norm,
                    ano_labels=ano_labels,
                    shot_idx=self.shot_idx,
                    shot_mask=self.shot_mask
                    )
        self.graph = data

    def few_shot(self, shot=10):
        y = self.graph.ano_labels
        num_nodes = y.shape[0]
        normal_idx = torch.where(y == 0)[0].tolist()
        random.shuffle(normal_idx)
        shot_idx = torch.tensor(normal_idx[:shot])
        shot_mask = torch.zeros(num_nodes, dtype=torch.bool)
        self.graph.shot_idx = shot_idx
        shot_mask[shot_idx] = True
        self.graph.shot_mask = shot_mask

    def propagated(self, k):
        x = torch.FloatTensor(self.feat).cuda()
        h_list = [x]
        for _ in range(k):
            h_list.append(torch.spmm(self.adj_norm.cuda(), h_list[-1]))
        self.graph.conv_list = h_list

    def sim_conv(self, k, sim_metric='dot'):
        x = torch.FloatTensor(self.feat).cuda()
        n = x.size(0)
        h_list = [x]
        device = x.device

        is_large_graph = n > 40000

        if is_large_graph:
            def calc_edge_sim(src_idx, dst_idx, x_feat, metric):
                x_src = x_feat[src_idx]
                x_dst = x_feat[dst_idx]

                if metric == 'cosine':
                    x_src = F.normalize(x_src, p=2, dim=1)
                    x_dst = F.normalize(x_dst, p=2, dim=1)
                    sim_val = (x_src * x_dst).sum(dim=1)
                    sim_val = (sim_val + 1) / 2
                elif metric == 'dot':
                    sim_val = (x_src * x_dst).sum(dim=1)
                    v_min = sim_val.min()
                    v_max = sim_val.max()
                    sim_val = (sim_val - v_min) / (v_max - v_min + 1e-8)
                else:
                    raise ValueError(f"Unknown sim_metric: {metric}")
                return sim_val

            eye_indices = torch.arange(n, device=x.device).unsqueeze(0).repeat(2, 1)
            eye_values = torch.ones(n, device=x.device)
            eye_sparse = torch.sparse_coo_tensor(eye_indices, eye_values, (n, n)).coalesce()

            A = self.adj_norm.clone().coalesce()
            indices, values = A.indices().to(device), A.values().to(device)

            sim_values = calc_edge_sim(indices[0], indices[1], x, sim_metric)

            weighted_values = values * sim_values
            A_weighted = torch.sparse_coo_tensor(indices, weighted_values, (n, n)).coalesce()

            A_weighted = (A_weighted + eye_sparse).coalesce()
            h_1 = torch.sparse.mm(A_weighted, x)
            h_list.append(h_1)

            if k >= 2:
                A2_sparse = torch.sparse.mm(A_weighted, A_weighted).coalesce()
                indices2 = A2_sparse.indices()

                values2 = torch.ones(indices2.shape[1], device=x.device)

                deg = torch.zeros(n, device=x.device)
                deg.index_add_(0, indices2[0], values2)
                deg_inv_sqrt = (deg + 1e-8).pow(-0.5)
                norm_values2 = deg_inv_sqrt[indices2[0]] * values2 * deg_inv_sqrt[indices2[1]]

                sim_values2 = calc_edge_sim(indices2[0], indices2[1], x, sim_metric)
                weighted_values2 = norm_values2.to(device) * sim_values2.to(device)

                A2_weighted = torch.sparse_coo_tensor(indices2, weighted_values2, (n, n)).coalesce()
                A2_weighted = (A2_weighted + eye_sparse).coalesce()

                h_2 = torch.sparse.mm(A2_weighted, x)
                h_list.append(h_2)

        else:
            if sim_metric == 'cosine':
                x_norm = F.normalize(x, p=2, dim=1)
                sim = torch.mm(x_norm, x_norm.t())
                sim = (sim + 1) / 2
            elif sim_metric == 'dot':
                sim = torch.mm(x, x.t())
                sim = (sim - sim.min()) / (sim.max() - sim.min() + 1e-8)
            else:
                raise ValueError(f"Unknown sim_metric: {sim_metric}")

            sim.fill_diagonal_(0)
            max_sim, _ = sim.max(dim=1, keepdim=True)
            sim = sim.cuda()

            eye_indices = torch.arange(n, device=x.device).unsqueeze(0).repeat(2, 1)
            eye_values = torch.ones(n, device=x.device)
            eye_sparse = torch.sparse_coo_tensor(eye_indices, eye_values, (n, n)).coalesce()

            A = self.adj_norm.clone().coalesce()
            indices, values = A.indices().cuda(), A.values().cuda()
            sim_values = sim[indices[0], indices[1]].cuda()
            weighted_values = values * sim_values
            A_weighted = torch.sparse_coo_tensor(indices, weighted_values, (n, n)).coalesce()
            A_weighted = (A_weighted + eye_sparse).coalesce()
            h_1 = torch.sparse.mm(A_weighted, x)
            h_list.append(h_1)

            A2_sparse = A_weighted.clone().detach()

            if k >= 2:
                A2_sparse = torch.sparse.mm(A2_sparse, A_weighted)
                indices2 = A2_sparse._indices()
                values2 = torch.ones_like(A2_sparse._values())

                deg = torch.zeros(n, device=x.device)
                deg.index_add_(0, indices2[0], values2)
                deg_inv_sqrt = (deg + 1e-8).pow(-0.5)
                norm_values2 = deg_inv_sqrt[indices2[0]] * values2 * deg_inv_sqrt[indices2[1]]

                sim_values2 = sim[indices2[0], indices2[1]]
                weighted_values2 = norm_values2 * sim_values2

                A2_weighted = torch.sparse_coo_tensor(indices2, weighted_values2, (n, n)).coalesce()
                A2_weighted = (A2_weighted + eye_sparse).coalesce()

                h_2 = torch.sparse.mm(A2_weighted, x)
                h_list.append(h_2)

        self.graph.sim_conv = h_list