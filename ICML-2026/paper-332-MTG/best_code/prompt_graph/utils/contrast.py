import torch
from copy import deepcopy
import numpy as np
from torch_geometric.data import Data


def contrastive_loss(x1, x2, temperature=0.1):
    # TODO: hyper-parameter checking, currently fix temperature = 0.1
    batch_size, _ = x1.size()
    x1_abs, x2_abs = x1.norm(dim=1), x2.norm(dim=1)
    sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
    sim_matrix = torch.exp(sim_matrix / temperature)
    pos_sim = sim_matrix[range(batch_size), range(batch_size)]
    loss = pos_sim / ((sim_matrix.sum(dim=1) - pos_sim) + 1e-4)
    loss = - torch.log(loss).mean() + 10
    return loss


def generate_random_model_output(data, model):
    vice_model = deepcopy(model)

    for (vice_name, vice_model_param), (name, param) in zip(vice_model.named_parameters(), model.named_parameters()):
        if vice_name.split('.')[0] == 'projection_head':
            vice_model_param.data = param.data
        else:
            vice_model_param.data = param.data + 0.1 * torch.normal(0, torch.ones_like(
                param.data) * param.data.std())
    z2 = vice_model.forward(data.x, data.edge_index, data.batch)

    return z2


def generate_corrupted_graph(graph_data, aug='dropE', aug_ratio=0.1):
    if aug == 'dropN':
        return generate_corrupted_graph_via_drop_node(graph_data, aug_ratio)
    elif aug == 'dropE':
        return generate_corrupted_graph_via_drop_edge(graph_data, aug_ratio)
    elif aug == 'shuffleX':
        return genereate_corrupted_graph_via_shuffle_X(graph_data)
    else:
        raise KeyError("[Pretrain] Encounter unsupported corruption method")


def generate_corrupted_graph_via_drop_node(data, aug_ratio=0.1):
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    drop_num = int(node_num * aug_ratio)

    idx_perm = np.random.permutation(node_num)

    idx_drop = idx_perm[:drop_num]
    idx_nondrop = idx_perm[drop_num:]
    idx_nondrop.sort()
    idx_dict = {idx_nondrop[n]: n for n in list(range(idx_nondrop.shape[0]))}

    edge_index = data.edge_index.numpy()

    edge_index = [[idx_dict[edge_index[0, n]], idx_dict[edge_index[1, n]]] for n in range(edge_num) if
                  (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)]

    new_edge_index = torch.tensor(edge_index).transpose_(0, 1)
    new_x = data.x[idx_nondrop]
    return Data(x=new_x, edge_index=new_edge_index)


def generate_corrupted_graph_via_drop_edge(data, aug_ratio):
    _, edge_num = data.edge_index.size()
    permute_num = int(edge_num * aug_ratio)

    idx_delete = np.random.choice(edge_num, (edge_num - permute_num), replace=False)
    new_edge_index = data.edge_index[:, idx_delete]

    return Data(x=data.x, edge_index=new_edge_index)


def genereate_corrupted_graph_via_shuffle_X(data):
    r"""
        Perturb one graph by row-wise shuffling X (node features) without changing the A (adjacency matrix).
    """
    node_num = data.x.shape[0]
    idx = np.random.permutation(node_num)
    new_x = data.x[idx, :]
    return Data(x=new_x, edge_index=data.edge_index)
