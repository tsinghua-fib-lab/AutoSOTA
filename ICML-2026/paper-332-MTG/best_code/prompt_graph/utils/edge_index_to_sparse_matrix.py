import torch


def edge_index_to_sparse_matrix(edge_index: torch.LongTensor, num_node: int):
    node_idx = torch.LongTensor([i for i in range(num_node)])
    self_loop = torch.stack([node_idx, node_idx], dim=0)
    edge_index = torch.cat([edge_index, self_loop], dim=1)
    sp_adj = torch.sparse.FloatTensor(edge_index, torch.ones(edge_index.size(1)), torch.Size((num_node, num_node)))

    return sp_adj