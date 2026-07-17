from torch_geometric.utils import to_undirected, subgraph
from torch_geometric.data import Data
import torch

# 1. Obtain the node indices for each graph in the G_m sequence
# Maps: (sorted nodes, list of increment sizes) -> List of accumulated node indices
def build_accumulated_subgraphs(sorted_nodes, group_add_sizes):
    acc_nodesIdx = []
    start = 0
    for size in group_add_sizes:
        start += size
        acc_nodesIdx.append(sorted_nodes[:start])
        
    return acc_nodesIdx

# Compute the edge density of a subgraph
# Inputs: (Full graph data, Node indices)
def compute_density(data, nodes_Idx):
    # Ensure nodes_Idx and data are on the same device
    if nodes_Idx.device != data.edge_index.device:
        nodes_Idx = nodes_Idx.to(data.edge_index.device)
    
    edge_index_sub, _ = subgraph(
        subset=nodes_Idx, edge_index=data.edge_index, relabel_nodes=False
    )
    edge_index_sub = to_undirected(edge_index_sub)
    m_edges = edge_index_sub.size(1) // 2
    n = nodes_Idx.numel()
    return 2 * m_edges / (n * (n - 1)) if n > 1 else 0.0


# 2. Select the accumulated subgraph closest to the target density
# Inputs: (Full graph data, Accumulated node indices, Target density function, Params, Graph size)
def find_closest_subgraph(data, acc_nodesIdx, n, target_density_func, param=None):
    target_density = target_density_func(n, param)
    best_nodes, best_diff = None, float("inf")
    for nodes_Idx in acc_nodesIdx:
        dens = compute_density(data, nodes_Idx)
        diff = abs(dens - target_density)
        if diff < best_diff:
            best_diff = diff
            best_nodes = nodes_Idx
    return best_nodes

# 3. Sample a complete G_m
# Input: Size of the accumulated graph
def sample_acc_subgraph_with_features(data, sum_nodes):
    device = data.x.device
    
    # 1. Sort nodes by degree
    deg = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
    for edge in data.edge_index.t():
        deg[edge[0]] += 1
    sorted_nodes = torch.argsort(deg, descending=True)
    
    # 2. Sample nodes, edges, and features
    sampled_nodes = sorted_nodes[: sum_nodes]
    edge_index_sub, _ = subgraph(
        subset=sampled_nodes,
        edge_index=data.edge_index,
        relabel_nodes=True
    )
    edge_index_sub = to_undirected(edge_index_sub)
    x_sub = data.x[sampled_nodes]
    y_sub = data.y[sampled_nodes]
    
    # 3. Return PyG Data object
    sub_data = Data(x=x_sub, y=y_sub, edge_index=edge_index_sub)
    return sub_data


def sample_random_subgraph_with_features_new(data, acc_nodesIdx, n, target_density_func, param=None):
    # 1. Select the accumulated subgraph based on the target density function
    Gm_nodes = find_closest_subgraph(data, acc_nodesIdx, n, target_density_func, param)

    # Ensure n does not exceed the number of available nodes
    n = min(n, Gm_nodes.numel())

    # 2. Randomly sample n nodes
    available_nodes = Gm_nodes
    perm = torch.randperm(available_nodes.numel(), device=data.x.device)[:n]
    sampled_nodes = available_nodes[perm]

    # 3. Construct the subgraph (relabel nodes while preserving original properties)
    edge_index_sub, _ = subgraph(
        subset=sampled_nodes,
        edge_index=data.edge_index,
        relabel_nodes=True
    )
    edge_index_sub = to_undirected(edge_index_sub)
    x_sub = data.x[sampled_nodes]
    y_sub = data.y[sampled_nodes]

    # Return the PyG Data object and the original node IDs
    return Data(x=x_sub, y=y_sub, edge_index=edge_index_sub), sampled_nodes