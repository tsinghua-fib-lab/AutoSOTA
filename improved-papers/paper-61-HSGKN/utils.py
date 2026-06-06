import numpy as np
import torch
import time
from tqdm import tqdm
import torch.nn.functional as F
from collections import defaultdict
import igraph as ig
import warnings
warnings.filterwarnings("ignore")

def normalize_attr(x):
    a = torch.nn.functional.normalize(x, p=2, dim=1)
    return a

def count_single_graph_count(paths):
    dict = {}
    for path in paths:
        if dict.get(len(path), None) is None:
            dict[len(path)] = 1
        else:
            dict[len(path)] += 1
    return dict

def path_resp_matrix_synthic(G, node_attr):
    src, dst = G.edges

    g_igraph = ig.Graph(edges=list(zip(src, dst)), directed=False, n=G.number_of_nodes())

    paths = []
    for i in range(G.number_of_nodes()):
        path = g_igraph.get_shortest_paths(i, output='vpath', algorithm='dijkstra')
        paths.extend(path)

    idx = defaultdict(list)
    for path in paths:
        idx[len(path)].append(path)
    idx = {key: np.array(val) for key, val in idx.items()}
    if idx.get(0, None) is not None:
        del idx[0]
    dict_count = {}
    for key, value in idx.items():
        n_count = [np.bincount(value[:, i], minlength=G.number_of_nodes()) for i in range(value.shape[1])]
        dict_count[len(n_count)] = torch.stack([torch.tensor(arr) for arr in n_count], dim=1)

    for i in range(min(dict_count.keys()), max(dict_count.keys()) + 1):
        if dict_count.get(i, None) is None:
            dict_count[i] = torch.zeros((G.num_nodes(), i), dtype=torch.float32)

    dict_count = dict(sorted(dict_count.items()))  
    count_matrix = torch.cat(list(dict_count.values()), dim=1)
    len_distribution = {k: int(v[:, 0].sum().item()) for k, v in dict_count.items()}

    return node_attr.t() @ count_matrix.to(torch.float32), len_distribution

def path_resp_matrix_ig(G, node_attr):
    src, dst = G.edges()

    g_igraph = ig.Graph(edges=list(zip(src.tolist(), dst.tolist())), directed=False, n=G.number_of_nodes())

    paths = []
    for i in range(G.number_of_nodes()):
        path = g_igraph.get_shortest_paths(i, output='vpath', algorithm='dijkstra')
        paths.extend(path)

    idx = defaultdict(list)
    for path in paths:
        idx[len(path)].append(path)
    idx = {key: np.array(val) for key, val in idx.items()}
    if idx.get(0, None) is not None:
        del idx[0]
    dict_count = {}
    for key, value in idx.items():
        n_count = [np.bincount(value[:, i], minlength=G.number_of_nodes()) for i in range(value.shape[1])]
        dict_count[len(n_count)] = torch.stack([torch.tensor(arr) for arr in n_count], dim=1)

    for i in range(min(dict_count.keys()), max(dict_count.keys()) + 1):
        if dict_count.get(i, None) is None:
            dict_count[i] = torch.zeros((G.num_nodes(), i), dtype=torch.float32)

    dict_count = dict(sorted(dict_count.items()))
    count_matrix = torch.cat(list(dict_count.values()), dim=1)
    len_distribution = {k: int(v[:, 0].sum().item()) for k, v in dict_count.items()}

    return node_attr.t() @ count_matrix.to(torch.float32), len_distribution

def process_dataset_matrix_ig(dataset, cutoff=None):
    print('Cutoff is {}'.format(cutoff))
    path_lens = {}
    graphs_paths_attrs = []
    len_distributions = []
    max_path_dim = 0
    start = time.time()
    print(dataset)
    for g, label in tqdm(dataset):
        path_attr_dict, len_distribution = path_resp_matrix_ig(g,g.ndata['node_attr'].to(torch.float32))
        path_attr_dict = torch.flatten(path_attr_dict.t())
        if path_attr_dict.shape[0] > max_path_dim:
            max_path_dim = path_attr_dict.shape[0]
        len_distributions.append(len_distribution)
        graphs_paths_attrs.append(path_attr_dict)

    graphs_attr = torch.zeros((len(graphs_paths_attrs), max_path_dim), dtype=torch.float32)
    for i, graph_attr in enumerate(graphs_paths_attrs):
        graphs_attr[i, :graph_attr.size(0)] = graph_attr

    print('Total time of preprocess: {:.2f}s'.format(time.time() - start))
    len_count = merge_len_distribution(len_distributions)
    return graphs_attr, len_count

def merge_len_distribution(len_distributions):
    merged_dict = {}
    for d in len_distributions:
        for key, value in d.items():
            merged_dict[key] = merged_dict.get(key, 0) + value

    return merged_dict

def set_node_degree_as_feature(dataset, norm_degree=True):
    for graph, l in dataset:
        degrees = graph.in_degrees().float()
        if norm_degree:
            degrees = torch.nn.functional.normalize(degrees.unsqueeze(1), p=2, dim=0)
        graph.ndata['node_attr'] = degrees
    return dataset


def set_attr_as_label(dataset):
    all_labels = torch.cat([g.ndata['node_labels'] for g, label in dataset])
    unique_labels = torch.unique(all_labels)
    num_classes = unique_labels.shape[0]
    label_map = {label.item(): idx for idx, label in enumerate(unique_labels)}

    for g, label in dataset:
        node_labels = g.ndata['node_labels']

        mapped_labels = torch.tensor([label_map[label.item()] for label in node_labels])
        one_hot_labels = F.one_hot(mapped_labels, num_classes=num_classes).float()
        g.ndata['node_attr'] = one_hot_labels
    return dataset


def set_label_as_degree(dataset):
    for g, label in dataset:
        degrees = g.in_degrees().float()
        g.ndata['node_labels'] = degrees.unsqueeze(1)


def cat_attr_label(dataset):
    all_labels = torch.cat([g.ndata['node_labels'] for g, label in dataset])
    unique_labels = torch.unique(all_labels)
    num_classes = unique_labels.shape[0]
    label_map = {label.item(): idx for idx, label in enumerate(unique_labels)}

    for g, label in dataset:
        node_labels = g.ndata['node_labels']
        mapped_labels = torch.tensor([label_map[label.item()] for label in node_labels])

        if 'node_attr' in g.ndata:
            node_feats = g.ndata['node_attr']
        else:
            node_feats = torch.zeros(g.num_nodes(), 0)

        one_hot_labels = F.one_hot(mapped_labels, num_classes=num_classes).float()
        new_node_feats = torch.cat([node_feats, one_hot_labels], dim=1)

        g.ndata['node_attr'] = new_node_feats
    return dataset
