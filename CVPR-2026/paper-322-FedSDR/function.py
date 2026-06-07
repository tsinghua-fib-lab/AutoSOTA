import torch
from fgl.model.gcn import *
from fgl.data.processing import *
import os
import random
import torch
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.utils import degree, to_dense_adj, remove_isolated_nodes


def load_client_list(client_data_dir, client_num=200):
    client_data_list = []
    for num in range(client_num):
        filename = f"data_{num}.pt"
        file_path = os.path.join(client_data_dir, filename)
        if os.path.exists(file_path):
            client_data = torch.load(file_path)
            client_data_list.append(client_data)
    return client_data_list


def create_corrupted_client(corruption_ratio, args, client_data_dir, corrupted_client_dir):

    os.makedirs(corrupted_client_dir, exist_ok=True)
    client_data_list = load_client_list(client_data_dir=client_data_dir)

    num_clients = len(client_data_list)
    num_corrupted_clients = int(num_clients * corruption_ratio)
    print(num_corrupted_clients)
    corrupted_client_indices = random.sample(
        range(num_clients), num_corrupted_clients)
    print("corrupted_client_indices: ", corrupted_client_indices)

    for client_id in corrupted_client_indices:
        splitted_data = client_data_list[client_id]
        processed_data = random_topology_noise(
            splitted_data,
            noise_prob=args.noise_extent
        )
        client_data_list[client_id] = processed_data

    for client_id, client_data in enumerate(client_data_list):
        torch.save(client_data, os.path.join(
            corrupted_client_dir, f"data_{client_id}.pt"))

    return client_data_list


def cosine_similarity_gpu(x):
    x_norm = x / x.norm(dim=1)[:, None]
    return torch.mm(x_norm, x_norm.transpose(0, 1))


def compute_client_similarity_matrix(x):
    device = x.device
    if device.type == 'cpu':
        return torch.tensor(cosine_similarity(x.numpy()), dtype=torch.float32, device=device)
    elif device.type == 'cuda':
        return cosine_similarity_gpu(x)
    else:
        raise ValueError(f"Unsupported device: {device}. Use 'cpu' or 'cuda'.")


def A_D(data, device):
    edge_index = data.edge_index

    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    edge_index, _, mask = remove_isolated_nodes(
        edge_index, num_nodes=data.num_nodes)
    num_nodes = mask.sum().item()

    deg = degree(edge_index[0], num_nodes=num_nodes, dtype=torch.float)
    assert not (deg == 0).any(), "isolate nodes remaining!"

    D = torch.diag(deg).to(device)

    A = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0].to(device)

    return A, D


def compute_S_noi(data, device):
    A, D = A_D(data, device)
    L = D - A

    D_T_L_D = torch.matmul(D.T, torch.matmul(L, D))
    D_T_D = torch.matmul(D.T, D)

    numerator = torch.sum(D_T_L_D)
    denominator = torch.sum(D_T_D)
    s = numerator / denominator

    return s.item()


def S_client_weights_s(S_noi_list, N_list):
    weighted_noise_sum = sum(N_i * S_noi_i for N_i,
                             S_noi_i in zip(N_list, S_noi_list))
    total_nodes = sum(N_list)
    global_mean_noise = weighted_noise_sum / total_nodes

    delta_list = [abs(S_noi_k - global_mean_noise) for S_noi_k in S_noi_list]

    min_delta = min(delta_list)
    max_delta = max(delta_list)
    if min_delta == max_delta:
        gamma_list = [1.0] * len(delta_list)
    else:
        gamma_list = [(delta_k - min_delta) / (max_delta - min_delta)
                      for delta_k in delta_list]

    exp_weights = [np.exp(-gamma_k) for gamma_k in gamma_list]
    weight_sum = sum(exp_weights)
    weights = [w / weight_sum for w in exp_weights]

    return weights


def modify_edges(similarity_matrix, edge_index, num_nodes, device, alpha):
    if device is None:
        device = similarity_matrix.device

    connected_edges = edge_index.t()
    edge_similarities = similarity_matrix[connected_edges[:,
                                                          0], connected_edges[:, 1]]

    threshold = torch.quantile(edge_similarities, alpha)
    mask = edge_similarities >= threshold
    edge_index_kept = edge_index[:, mask]
    num_edges_removed = edge_index.size(1) - edge_index_kept.size(1)

    if num_edges_removed == 0:
        return edge_index_kept

    existing_edges = torch.sparse_coo_tensor(
        edge_index,
        torch.ones(edge_index.size(1), device=device),
        size=(num_nodes, num_nodes),
        device=device
    ).to_dense().bool()

    triu_mask = torch.triu(torch.ones(
        num_nodes, num_nodes, device=device), diagonal=1).bool()

    candidate_mask = triu_mask & (~existing_edges) & (~existing_edges.t())

    candidate_pairs = torch.nonzero(candidate_mask)

    if num_edges_removed > 0 and len(candidate_pairs) > 0:
        candidate_similarities = similarity_matrix[candidate_pairs[:,
                                                                   0], candidate_pairs[:, 1]]
        num_edges_to_add = min(num_edges_removed, len(candidate_pairs))
        topk_values, topk_indices = torch.topk(
            candidate_similarities, k=num_edges_to_add)
        selected_pairs = candidate_pairs[topk_indices]

        edge_index = torch.cat([edge_index_kept, selected_pairs.t()], dim=1)
    else:
        edge_index = edge_index_kept

    return edge_index
