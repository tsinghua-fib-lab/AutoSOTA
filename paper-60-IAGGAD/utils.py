import random
import torch
import os
from torch_geometric.data import Data
import json
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score, average_precision_score
import dgl
import os
import copy
from pathlib import Path


def test_eval(labels, probs):
    score = {}
    with torch.no_grad():
        if torch.is_tensor(labels):
            labels = labels.cpu().numpy()
        if torch.is_tensor(probs):
            probs = probs.cpu().numpy()
        score['AUROC'] = roc_auc_score(labels, probs)
        score['AUPRC'] = average_precision_score(labels, probs)
    return score


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def feat_alignment(X, edges, dims):
    edge_src, edge_dst = edges
    num_edges = len(edge_src)

    if X.shape[1] < dims:
        transformer = GaussianRandomProjection(n_components=256, random_state=0)
        X = transformer.fit_transform(X.cpu().numpy())
    # Proj
    pca = PCA(n_components=dims, random_state=0)
    X_transformed = pca.fit_transform(X)
    X_transformed = torch.FloatTensor(X_transformed)
    # normalize
    X_min, _ = torch.min(X_transformed, dim=0)
    X_max, _ = torch.max(X_transformed, dim=0)
    X_s = (X_transformed - X_min) / (X_max - X_min)

    smooth_coefficients = torch.zeros(X_transformed.shape[1])
    for k in range(X_transformed.shape[1]):
        # X_{i k}-X_{j k}, (v_i, v_j) \in Edge set
        differences = X_s[edge_src, k] - X_s[edge_dst, k]
        smooth_coefficients[k] = torch.sum(differences ** 2) / num_edges
    # sort
    _, sorted_indices = torch.sort(smooth_coefficients)
    X_reordered = X_transformed[:, sorted_indices]
    return X_reordered


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def calc_sim_optimized(adj_matrix, attr_matrix):
    """
    计算节点之间的余弦相似度，使用 GPU 加速。

    Args:
        adj_matrix (torch.Tensor): 邻接矩阵 (dense tensor)
        attr_matrix (torch.Tensor): 特征矩阵 (dense tensor)

    Returns:
        torch.Tensor: 相似度矩阵 (dense tensor)
    """
    device = attr_matrix.device  # 确保计算在正确的设备上
    row_indices, col_indices = adj_matrix.nonzero(as_tuple=True)  # 获取邻接矩阵中的非零索引
    src_features = attr_matrix[row_indices]  # 源节点的特征
    dst_features = attr_matrix[col_indices]  # 目标节点的特征

    # 批量计算余弦相似度
    similarities = torch.cosine_similarity(src_features, dst_features, dim=1)

    # 初始化相似度矩阵，并填充计算结果
    sim_array = torch.zeros_like(adj_matrix).to(device)
    sim_array[row_indices, col_indices] = similarities

    return sim_array


def calc_sim(adj_matrix, attr_matrix):
    row = adj_matrix.shape[0]
    col = adj_matrix.shape[1]
    dis_array = torch.zeros((row, col))
    for i in range(row):
        # print(i)
        node_index = torch.argwhere(adj_matrix[i, :] > 0)[:, 0]
        for j in node_index:
            # dis = get_cos_similar(attr_matrix[i].tolist(), attr_matrix[j].tolist())
            dis = get_cos_similar(attr_matrix[i], attr_matrix[j])
            dis_array[i][j] = dis

    return dis_array


def get_cos_similar(v1: list, v2: list):
    num = float(torch.dot(v1, v2))  # 向量点乘
    denom = torch.linalg.norm(v1) * torch.linalg.norm(v2)  # 求模长的乘积
    return 0.5 + 0.5 * (num / denom) if denom != 0 else 0


# compute the distance between each node
def calc_distance(adj, seq):
    dis_array = torch.zeros((adj.shape[0], adj.shape[1]))
    row = adj.shape[0]

    for i in range(row):
        # print(i)
        node_index = torch.argwhere(adj[i, :] > 0)
        for j in node_index:
            dis = torch.sqrt(torch.sum((seq[i] - seq[j]) * (seq[i] - seq[j])))
            dis_array[i][j] = dis

    return dis_array


def calc_distance_optimized(adj, seq):
    """
    计算节点之间的欧式距离，使用 GPU 加速。
    adj: 邻接矩阵 (dense tensor)
    seq: 特征矩阵 (tensor)
    """
    device = seq.device  # 确保计算在正确的设备上
    # 转换为稀疏张量
    # adj_sparse = adj.to_sparse()
    # 获取非零元素的索引
    # row_indices, col_indices = adj_sparse.nonzero(as_tuple=True)
    row_indices, col_indices = adj.nonzero(as_tuple=True)  # 获取所有边的索引
    src_features = seq[row_indices]  # 源节点的特征
    dst_features = seq[col_indices]  # 目标节点的特征

    # 计算欧式距离
    distances = torch.sqrt(torch.sum((src_features - dst_features) ** 2, dim=1))

    # 将距离填入距离矩阵
    dis_array = torch.zeros_like(adj).to(device)  # 初始化一个距离矩阵
    dis_array[row_indices, col_indices] = distances  # 填入距离

    return dis_array


def graph_nsgt_3(dis_array, adj, sim_adj, args):
    dis_array = dis_array.to("cpu")
    sim_adj = sim_adj.to("cpu")
    adj = adj.to("cpu")
    row = dis_array.shape[0]
    dis_array_u = dis_array * adj
    sim_adj = sim_adj * adj
    mean_dis = dis_array_u[dis_array_u != 0].mean()
    cut_edge_num = 0
    # sim_threshold = sim_adj[sim_adj != 0 ].mean()
    sim_threshold = sim_adj[(sim_adj != 1) & (sim_adj != 0)].mean()

    for i in range(row):
        node_index = torch.argwhere(adj[i, :] > 0).flatten()
        if node_index.shape[0] != 0:
            max_dis = dis_array[i, node_index].max()
            min_dis = mean_dis

            top_k = node_index.shape[0] // 2
            _, min_sim_indices = torch.topk(sim_adj[i, node_index], k=top_k, largest=False)

            if max_dis > min_dis:
                random_value = (max_dis - min_dis) * np.random.random_sample() + min_dis
                cutting_edge = torch.argwhere((dis_array[i, node_index[min_sim_indices]] > random_value) & (
                        sim_adj[i, node_index[min_sim_indices]] < sim_threshold))

                if cutting_edge.shape[0] != 0:
                    cut_edge_num += cutting_edge.shape[0]
                    adj[i, node_index[min_sim_indices[cutting_edge[:, 0]]]] = 0

    adj = adj + adj.T
    adj[adj > 1] = 1
    return adj, cut_edge_num


def graph_nsgt_2(dis_array, adj, sim_adj, args):
    dis_array = dis_array.cuda()
    sim_adj = sim_adj.cuda()
    row = dis_array.shape[0]
    dis_array_u = dis_array * adj
    sim_adj = sim_adj * adj
    mean_dis = dis_array_u[dis_array_u != 0].mean()
    cut_edge_num = 0
    # sim_threshold = sim_adj[sim_adj != 0 ].mean()
    sim_threshold = sim_adj[(sim_adj != 1) & (sim_adj != 0)].mean()

    for i in range(row):
        node_index = torch.argwhere(adj[i, :] > 0).flatten()
        if node_index.shape[0] != 0:
            max_dis = dis_array[i, node_index].max()
            min_dis = mean_dis

            top_k = node_index.shape[0] // 2
            _, min_sim_indices = torch.topk(sim_adj[i, node_index], k=top_k, largest=False)

            if max_dis > min_dis:
                random_value = (max_dis - min_dis) * np.random.random_sample() + min_dis
                cutting_edge = torch.argwhere((dis_array[i, node_index[min_sim_indices]] > random_value) & (
                        sim_adj[i, node_index[min_sim_indices]] < sim_threshold))

                if cutting_edge.shape[0] != 0:
                    cut_edge_num += cutting_edge.shape[0]
                    adj[i, node_index[min_sim_indices[cutting_edge[:, 0]]]] = 0

    adj = adj + adj.T
    adj[adj > 1] = 1
    return adj, cut_edge_num


def normalize_adj_tensor(raw_adj):
    # adj = raw_adj[0]

    adj = raw_adj

    row_sum = torch.sum(adj, 0)
    r_inv = torch.pow(row_sum, -0.5).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    adj = torch.mm(adj, torch.diag_embed(r_inv))
    adj = torch.mm(torch.diag_embed(r_inv), adj)
    # adj = adj.unsqueeze(0)
    return adj


class Dataset:
    def __init__(self, args, dims, type, name='cora', prefix='./dataset/', device="cuda:0"):
        # initiation
        self.shot_mask = None
        self.shot_idx = None
        self.graph = None
        self.x_list = None
        self.name = name

        preprocess_filename = f'{prefix}{name}_{dims}.npz'

        if os.path.exists(preprocess_filename):
            with np.load(preprocess_filename, allow_pickle=True) as f:
                data = f['data'].item()
                feat = f['feat']
        else:
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
            feat = feat_alignment(feat, edge_index, dims)
            np.savez(preprocess_filename, data=data, feat=feat)

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

        adj_coo = adj_norm.coalesce()
        row, col = adj_coo.indices()
        self_loop_mask = (row == col)
        new_row = row[~self_loop_mask]
        new_col = col[~self_loop_mask]
        # new_values = adj_coo.values()[~self_loop_mask]
        # 2. 设置边权为1
        new_values = torch.ones_like(new_row, dtype=torch.float32, device=adj_norm.device)
        # 构建一个新的邻接矩阵（去掉自环）
        adj_matrix_no_self_loop = torch.sparse_coo_tensor(
            indices=torch.stack([new_row, new_col]),
            values=new_values,
            size=adj_norm.size()
        )

        dgl_graph = dgl.from_scipy(adj)
        dgl_graph = dgl.remove_self_loop(dgl_graph)
        dgl_graph = dgl.add_self_loop(dgl_graph)
        x = torch.tensor(self.feat, dtype=torch.float).to(device)
        ano_labels = torch.tensor(np.squeeze(np.array(self.label)), dtype=torch.float)

        l = 0.0
        if type == "test":
            l = args.dataset_config[name]["lambda"]

        # l = args.lamda

        data = Data(x=torch.tensor(self.feat, dtype=torch.float),
                    x_list=self.x_list,
                    adj=self.adj_norm,
                    adj_without_loop=adj_matrix_no_self_loop,
                    dgl_graph=dgl_graph,
                    dgl_cut_graph=dgl_graph,
                    l=l,
                    ano_labels=ano_labels,
                    shot_idx=self.shot_idx,
                    shot_mask=self.shot_mask
                    )

        self.graph = data

    def random_sample(self, shot=10):
        y = self.graph.ano_labels
        num_nodes = y.shape[0]
        normal_idx = torch.where(y == 0)[0].tolist()
        random.shuffle(normal_idx)
        shot_idx = torch.tensor(normal_idx[:shot])
        shot_mask = torch.zeros(num_nodes, dtype=torch.bool)
        self.graph.shot_idx = shot_idx
        shot_mask[shot_idx] = True
        self.graph.shot_mask = shot_mask

    def test_random_sample(self, shot=10):
        y = self.graph.ano_labels
        num_nodes = y.shape[0]
        all_idx = list(range(num_nodes))
        random.shuffle(all_idx)
        shot_idx = torch.tensor(all_idx[:shot])
        shot_mask = torch.zeros(num_nodes, dtype=torch.bool)
        shot_mask[shot_idx] = True
        self.graph.shot_idx = shot_idx
        self.graph.shot_mask = shot_mask

    def propagated(self, k):
        x = torch.FloatTensor(self.feat).cuda()
        h_list = [x]
        for _ in range(k):
            h_list.append(torch.spmm(self.adj_norm.cuda(), h_list[-1]))
        self.graph.x_list = h_list


def read_data_config(json_dir):
    # Construct the filename based on the dataset name and shot
    filename = f"{json_dir}/dataset_config.json"

    # Check if the file exists
    if os.path.exists(filename):
        # Read the JSON file and return the dictionary
        with open(filename, 'r') as file:
            try:
                data = json.load(file)
                return data
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON file {filename}: {e}")
                return None
    else:
        print(f"JSON file {filename} not found.")
        return None


def read_json(model, shot, json_dir):
    # Construct the filename based on the dataset name and shot
    filename = f"{json_dir}/{model}_{shot}.json"

    # Check if the file exists
    if os.path.exists(filename):
        # Read the JSON file and return the dictionary
        with open(filename, 'r') as file:
            try:
                data = json.load(file)
                return data
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON file {filename}: {e}")
                return None
    else:
        print(f"JSON file {filename} not found.")
        return None
