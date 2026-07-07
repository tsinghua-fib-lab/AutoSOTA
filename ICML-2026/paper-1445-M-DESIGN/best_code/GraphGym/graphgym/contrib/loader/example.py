
import torch
from deepsnap.dataset import GraphDataset
from torch_geometric.datasets import QM7b

from graphgym.register import register_loader
from graphgym.contrib.loader.datasets import add_node_attr_nc


def load_dataset_example(format, name, dataset_dir):
    dataset_dir = '{}/{}'.format(dataset_dir, name)
    if format == 'PyG':
        if name == 'QM7b':
            dataset_raw = QM7b(dataset_dir)
            graphs = GraphDataset.pyg_to_graphs(dataset_raw)
            return graphs
        
def load_AmazonComputers_node(format, name, dataset_dir):
    if name != 'AmazonComputers_node':
        return None
    specific_task = 'AmazonComputers_node'
    dataset_raw = torch.load(f'../../knowledge/graphgymbench/design_v1_grid_round1/graphs/{specific_task}.pth')

    attr_list = []
    for dataset in dataset_raw:
        attr_list.append(add_node_attr_nc(dataset))

    graphs = GraphDataset.pyg_to_graphs(dataset_raw)
    for (graph, attr_dict) in zip(graphs, attr_list):
        for key, value in attr_dict.items():
            graph[key] = value
    
    return graphs

def load_AmazonPhoto_node(format, name, dataset_dir):
    if name != 'AmazonPhoto_node':
        return None
    specific_task = 'AmazonPhoto_node'
    dataset_raw = torch.load(f'../../knowledge/graphgymbench/design_v1_grid_round1/graphs/{specific_task}.pth')

    attr_list = []
    for dataset in dataset_raw:
        attr_list.append(add_node_attr_nc(dataset))

    graphs = GraphDataset.pyg_to_graphs(dataset_raw)
    for (graph, attr_dict) in zip(graphs, attr_list):
        for key, value in attr_dict.items():
            graph[key] = value
    
    return graphs

def load_cora_node(format, name, dataset_dir):
    if name != 'cora_node':
        return None
    specific_task = 'cora_node'
    dataset_raw = torch.load(f'../../knowledge/{specific_task}/{specific_task}.pth')

    attr_list = []
    for dataset in dataset_raw:
        attr_list.append(add_node_attr_nc(dataset))

    graphs = GraphDataset.pyg_to_graphs(dataset_raw)
    for (graph, attr_dict) in zip(graphs, attr_list):
        for key, value in attr_dict.items():
            graph[key] = value
    
    return graphs

def load_cora_node_pyg(format, name, dataset_dir):
    if name != 'cora_node':
        return None
    specific_task = 'cora_node'
    dataset_raw = torch.load(f'../../knowledge/{specific_task}/{specific_task}.pth')

    attr_list = []
    for dataset in dataset_raw:
        attr_list.append(add_node_attr_nc(dataset))

    graphs = GraphDataset.pyg_to_graphs(dataset_raw)
    for (graph, attr_dict) in zip(graphs, attr_list):
        for key, value in attr_dict.items():
            graph[key] = value
    
    return graphs

register_loader('example', load_dataset_example)
register_loader('AmazonComputers_node', load_AmazonComputers_node)
register_loader('AmazonPhoto_node', load_AmazonPhoto_node)
register_loader('cora_node', load_cora_node_pyg)
