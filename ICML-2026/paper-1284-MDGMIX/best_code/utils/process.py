import os

import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import sys
import torch
import torch.nn as nn
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold


def find_2hop_neighbors(adj, node):
    neighbors = []
    for i in range(len(adj[node])):
        if len(neighbors) >= 10:
            break
        if adj[node][i] != 0 and node != i:
            neighbors.append(i)
    neighbors_2hop = []
    for i in neighbors:
        cnt = 0
        for j in range(len(adj[i])):
            if cnt >= 4:
                break
            if adj[i][j] != 0 and j != i:
                neighbors_2hop.append(j)
                cnt += 1
    return neighbors, neighbors_2hop
def plotlabels(feature, Trure_labels, name):
    S_lowDWeights = visual(feature)
    colors = ['#e38c7a', '#656667', '#99a4bc', 'cyan', 'blue', 'lime', 'r', 'violet', 'm', 'peru', 'olivedrab','hotpink']
    True_labels = Trure_labels.reshape((-1, 1))
    S_data = np.hstack((S_lowDWeights, True_labels))
    S_data = pd.DataFrame({'x': S_data[:, 0], 'y': S_data[:, 1], 'label': S_data[:, 2]})
    print(S_data)
    print(S_data.shape)
    for index in range(4): 
        X = S_data.loc[S_data['label'] == index]['x']
        Y = S_data.loc[S_data['label'] == index]['y']
        plt.scatter(X, Y, cmap='brg', s=20, marker='.', c=colors[index], edgecolors=colors[index])
        plt.xticks([]) 
        plt.yticks([]) 
    plt.title(name, fontsize=32, fontweight='normal', pad=20)
    
    plt.savefig('plt_graph/exceptcomputers/{}.png'.format(name),dpi=500)
    plt.show()
    plt.clf()

def visual(feat):
    ts = manifold.TSNE(n_components=2, init='pca', random_state=0)
    x_ts = ts.fit_transform(feat)
    print(x_ts.shape) 
    x_min, x_max = x_ts.min(0), x_ts.max(0)
    x_final = (x_ts - x_min) / (x_max - x_min)
    return x_final

def combine_dataset(*args):
    for step,adj in enumerate(args):
        if step == 0:
            adj1 = adj.todense()
        else:
            adj2 = adj.todense()
            zeroadj = np.zeros((adj1.shape[0], adj2.shape[0]))
            tmpadj1 = np.column_stack((adj1, zeroadj))
            tmpadj2 = np.column_stack((zeroadj.T, adj2))
            adj1 = np.row_stack((tmpadj1, tmpadj2))
            
    adj = sp.csr_matrix(adj1)
    
    return adj
def combine_dataset_list_sp(args):

    adj1 = None
    
    for step, adj in enumerate(args):
        if step == 0:
            adj1 = adj 
        else:
            num_rows1, num_cols1 = adj1.shape
            num_rows2, num_cols2 = adj.shape
            zeroadj1 = sp.csr_matrix((num_rows1, num_cols2))  
            zeroadj2 = sp.csr_matrix((num_rows2, num_cols1)) 
            
            top = sp.hstack([adj1, zeroadj1])
            bottom = sp.hstack([zeroadj2, adj])
            adj1 = sp.vstack([top, bottom])
    
    return adj1.tocsr()
def combine_label(*labels):

    label_tem=0
    for step, label in enumerate(labels):
        _,num=np.unique(label, return_counts=True)
        label_tem+=num
        if step == 0:
            combined_label = label.reshape(-1, 1) 
        else:
            if label.shape[0] != combined_label.shape[0]:
                raise ValueError("所有标签的行数必须相同")
            combined_label = np.column_stack((combined_label, label.reshape(-1, 1)))

    return combined_label

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


def process_tu(data, class_num):


    ft_size = data.num_features

    num = range(class_num)

    labelnum=range(class_num,ft_size)
    features = data.x[:, num]
    rawlabels = data.x[:, labelnum]

    e_ind = data.edge_index
    coo = sp.coo_matrix((np.ones(e_ind.shape[1]), (e_ind[0, :], e_ind[1, :])),
                        shape=(features.shape[0], features.shape[0]))
    adjacency = coo
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



def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def load_data(dataset_str): 
    current_path = os.path.dirname(__file__)
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    return adj, features, labels, idx_train, idx_val, idx_test

def sparse_to_tuple(sparse_mx, insert_batch=False):
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

    f = f.todense()
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = f[:, np.squeeze(np.array(sigma > 0))]
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = (f - mu) / sigma
    return f

def preprocess_features(features):
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)

def normalize_adj(adj):
    
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)




import csv
from datetime import datetime

def write_results_to_csv(boundary_ratio,similarity_threshold ,shot_num, csv_name, pretrain_dataset_str, downstream_dataset, 
                        acc_mean, acc_std, microf_mean, microf_std, 
                        macrof_mean, macrof_std, model_name="", timestamp=True):

    # 格式化数值，保留3位小数
    def format_value(mean, std):
        return f"{mean:.2f}±{std:.2f}"
    def format_float(value, decimals=1):
        return f"{value:.{decimals}f}"
    
    with open(f'{csv_name}', mode='a', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file, dialect="excel")
        
        file.seek(0, 2)  # 移动到文件末尾
        if file.tell() == 0:
            headers = [
                "Boundary", "Similarity","ShotNum", "Timestamp", "Model", "Pretrain Dataset", "Downstream Dataset",
                "Micro F1", "Macro F1"
            ]
            writer.writerow(headers)
        
        # 准备数据行
        row_data = [
            format_float(boundary_ratio),
            format_float(similarity_threshold),
            shot_num,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S") if timestamp else "",
            model_name,
            pretrain_dataset_str,
            downstream_dataset,
            #format_value(acc_mean, acc_std),
            format_value(microf_mean, microf_std),
            format_value(macrof_mean, macrof_std)
        ]
        
        writer.writerow(row_data)

import subprocess




