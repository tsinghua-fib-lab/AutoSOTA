import numpy as np
import torch
import torch.nn.functional as F
import scipy.sparse as sp

def rank_score(score):
    rank_scores = torch.argsort(score).argsort()
    max_rank = torch.max(rank_scores)
    if max_rank == 0:
        return rank_scores.float()
    rank_scores = rank_scores.float() / max_rank * 100
    return rank_scores

def _to_scipy_binary_adj(adj_norm):
    if torch.is_tensor(adj_norm):
        if adj_norm.is_sparse:
            adj_norm = adj_norm.coalesce()
            indices = adj_norm.indices().cpu().numpy()
            values = adj_norm.values().cpu().numpy()
            adj_scipy = sp.csr_matrix((values, (indices[0], indices[1])), shape=adj_norm.shape)
        else:
            adj_scipy = sp.csr_matrix(adj_norm.cpu().numpy())
    elif isinstance(adj_norm, np.ndarray):
        adj_scipy = sp.csr_matrix(adj_norm)
    else:
        adj_scipy = adj_norm
    adj_scipy.data = np.ones_like(adj_scipy.data)
    adj_scipy.setdiag(0)
    adj_scipy.eliminate_zeros()
    return adj_scipy

def conv_residual(graph):
    h_list = graph.conv_list
    residual_list = []
    h_embed_list = []
    first_element = h_list[0]
    for h_i in h_list[1:]:
        dif = h_i - first_element
        residual_list.append(dif)
        h_embed_list.append(h_i)
    residual_embed = residual_list[-1]
    h_embed = h_embed_list[-1]
    return residual_embed, h_embed

def get_global_sim(embed, adj, mode='dis'):
    global_center = embed.mean(dim=0, keepdim=True)
    if mode == 'dis':
        dist_to_center = torch.norm(embed - global_center, p=2, dim=1)
        global_sim = dist_to_center
    elif mode == 'cos':
        norm_embed = F.normalize(embed, p=2, dim=1)
        norm_center = F.normalize(global_center, p=2, dim=1)
        global_sim = torch.mm(norm_embed, norm_center.t()).squeeze()
    return global_sim

def get_neibour_sim(embed, adj, step=1, mode='dis'):
    is_large_graph = embed.size(0) > 40000

    if is_large_graph:
        if not adj.is_sparse:
            adj = adj.to_sparse()

        degree = torch.sparse.sum(adj, dim=1).to_dense() + 1e-8

        if mode == 'cos':
            norm_embed = F.normalize(embed, p=2, dim=1)
            agg_embed = torch.sparse.mm(adj, norm_embed)
            numerator = (norm_embed * agg_embed).sum(dim=1)
            neibour_sim = numerator / degree

        elif mode == 'dis':
            indices = adj.coalesce().indices()
            rows, cols = indices[0], indices[1]
            edge_dists = torch.norm(embed[rows] - embed[cols], p=2, dim=1)
            numerator = torch.zeros(embed.size(0), device=embed.device)
            numerator.index_add_(0, rows, edge_dists)
            neibour_sim = numerator / degree

        neibour_sim = (neibour_sim - neibour_sim.min()) / (neibour_sim.max() - neibour_sim.min() + 1e-8)
        return neibour_sim
    else:
        if step > 1:
            adj1 = adj.clone()
            for i in range(step - 1):
                adj = torch.sparse.mm(adj, adj1)
        if adj.is_sparse:
            adj = adj.to_dense()
        adj = (adj > 0).float()
        adj.fill_diagonal_(0)
        if mode == 'cos':
            norm_embed = F.normalize(embed, p=2, dim=1)
            sim_matrix = torch.mm(norm_embed, norm_embed.t())
            neibour_sim = (sim_matrix * adj).sum(dim=1) / (adj.sum(dim=1) + 1e-8)
            neibour_sim = (neibour_sim - neibour_sim.min()) / (neibour_sim.max() - neibour_sim.min() + 1e-8)
        elif mode == 'dis':
            dist_matrix = torch.cdist(embed, embed, p=2)
            neibour_sim = (dist_matrix * adj).sum(dim=1) / (adj.sum(dim=1) + 1e-8)
            neibour_sim = (neibour_sim - neibour_sim.min()) / (neibour_sim.max() - neibour_sim.min() + 1e-8)
    return neibour_sim

def get_degree_centrality(adj_norm):
    device = adj_norm.device if torch.is_tensor(adj_norm) else torch.device('cpu')
    adj_scipy = _to_scipy_binary_adj(adj_norm)
    degree_np = np.array(adj_scipy.sum(axis=1)).flatten()
    return torch.tensor(degree_np, dtype=torch.float32, device=device)

def get_clustering_coefficient(adj_norm):
    device = adj_norm.device if torch.is_tensor(adj_norm) else torch.device('cpu')
    adj = _to_scipy_binary_adj(adj_norm)
    adj_sq = adj.dot(adj)
    triangles_matrix = adj.multiply(adj_sq)
    triangles = np.array(triangles_matrix.sum(axis=1)).flatten() / 2
    degree = np.array(adj.sum(axis=1)).flatten()
    possible_edges = degree * (degree - 1) / 2
    with np.errstate(divide='ignore', invalid='ignore'):
        cc = triangles / possible_edges
        cc[np.isnan(cc)] = 0
        cc[np.isinf(cc)] = 0
    return torch.tensor(cc, dtype=torch.float32, device=device)

def construct_features(args, dataset, mode='all'):
    graph = dataset.graph.to(args.device)
    labels = graph.ano_labels

    residual_embed, h_embed = conv_residual(graph)
    query_scores2 = get_global_sim(h_embed, graph.adj, mode='cos')
    query_scores_neibour = get_neibour_sim(graph.sim_conv[-2], graph.adj, step=1, mode='cos')
    query_scores_neibour1 = get_neibour_sim(graph.sim_conv[-2], graph.adj, step=1, mode='dis')
    degree = get_degree_centrality(graph.adj)
    clustering_co = get_clustering_coefficient(graph.adj)

    query_scores2 = rank_score(query_scores2)
    query_scores_neibour = rank_score(query_scores_neibour)
    query_scores_neibour1 = rank_score(query_scores_neibour1)
    degree = rank_score(degree)
    clustering_co = rank_score(clustering_co)

    P_features = torch.cat([
        query_scores2.unsqueeze(1),
        query_scores_neibour.unsqueeze(1),
        query_scores_neibour1.unsqueeze(1),
        degree.unsqueeze(1),
        clustering_co.unsqueeze(1),
    ], dim=1)

    return P_features, labels