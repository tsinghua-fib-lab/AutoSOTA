import torch
import matplotlib.pyplot as plt
from torch_geometric.datasets import WebKB, Planetoid, Amazon, Coauthor, WikipediaNetwork, Reddit, \
    Flickr, PPI, Yelp, Twitch, Actor, KarateClub, FacebookPagePage, LastFMAsia,LINKXDataset
 #BitcoinOTC 

from torch_geometric.utils import degree
from ogb.nodeproppred import PygNodePropPredDataset
#from ogb.lsc import MAG240MDataset
from torch_geometric.utils import to_networkx, degree
import networkx as nx
import numpy as np
import torch_geometric.transforms as T

WebKB_datasets = ['Texas', 'Cornell', 'Wisconsin']
Planetoid_datasets = ['Cora', 'Citeseer', 'Pubmed']
Amazon_datasets = ['Photo', 'Computers']
Coauthor_datasets = ['CS', 'Physics']
WikipediaNetwork_datasets = ['chameleon', 'squirrel']
Reddit_datasets = ['Reddit']
OGB_datasets = ['ogbn-arxiv', 'ogbn-products', 'ogbn-proteins', 'ogbn-papers100M', 'ogbn-mag']

Flickr_datasets = ['Flickr']
PPI_datasets = ['PPI']
Yelp_datasets = ['Yelp']
Twitch_datasets = ['DE', 'EN', 'ES', 'FR', 'PT', 'RU']
Actor_datasets = ['Actor']
KarateClub_datasets = ['KarateClub']
FacebookPagePage_datasets = ['FacebookPagePage']
LastFMAsia_datasets = ['LastFMAsia']


def load_dataset(name, path='/datasets'):
    if name in Planetoid_datasets:
        dataset = Planetoid(root=path, name=name)
    elif name in Amazon_datasets:
        dataset = Amazon(root=path, name=name)
    elif name in Coauthor_datasets:
        dataset = Coauthor(root=path, name=name)
    elif name in WebKB_datasets:
        dataset = WebKB(root=path, name=name)
    elif name in WikipediaNetwork_datasets:
        dataset = WikipediaNetwork(root=path, name=name)
    elif name in Reddit_datasets:
        dataset = Reddit(root=f'{path}/Reddit')
    elif name in OGB_datasets:
        dataset = PygNodePropPredDataset(root=path, name=name, transform=T.ToUndirected())
    elif name in Flickr_datasets:
        dataset = Flickr(root=f'{path}/Flickr')
    elif name in PPI_datasets:
        dataset = PPI(root=f'{path}/PPI')
    elif name in Yelp_datasets:
        dataset = Yelp(f'{path}/Yelp')
    elif name in Twitch_datasets:
        dataset = Twitch(root=path,name=name)
    elif name in Actor_datasets:
        dataset = Actor(root=f'{path}/Actor')
    elif name in KarateClub_datasets:
        dataset = KarateClub()
    elif name in FacebookPagePage_datasets:
        dataset = FacebookPagePage(root=f'{path}/Facebook')
    elif name in LastFMAsia_datasets:
        dataset = LastFMAsia(root=f'{path}/LastFMAsia')

    elif name in ['penn94']:
        dataset=LINKXDataset(root=path,name=name)

    else:
        raise ValueError(f"Unknown dataset name: {name}")
    #print(f'{name}: {dataset[0].num_nodes}')
    return dataset

def analyze_dataset(name, graph = False):
    data_ = load_dataset(name)
    #print(len(data_))
    data = data_[0]
    deg = degree(data.edge_index[0], data.num_nodes)
    average_degree = deg.mean().item()
    print(f'\n{name}: avg_degree:{average_degree:.4f}, num_nodes:{data.num_nodes}, num_edges:{data.num_edges}, num_classes:{data_.num_classes}, num_features:{data_.num_features}')
    G = to_networkx(data, to_undirected=True)

    print('Clustering Coefficient:')
    global_clustering_coefficient = nx.transitivity(G) 
    print(f'Global Clustering Coefficient: {global_clustering_coefficient:.4f}')

    average_clustering_coefficient = nx.average_clustering(G)
    print(f'Average Clustering Coefficient: {average_clustering_coefficient:.4f}')

    shortest_path_lengths = dict(nx.all_pairs_shortest_path_length(G))

    path_lengths = []
    for node, lengths in shortest_path_lengths.items():
        path_lengths.extend(lengths.values())
    
    network_diameter = max(path_lengths)
    print(f'Network Diameter:')

    average_shortest_path_length = np.mean(path_lengths)
    print(f'Average Shortest Path Length: {average_shortest_path_length:.4f}')

    percentile_90_shortest_path_length = np.percentile(path_lengths, 90)
    print(f'90th Percentile Shortest Path Length: {percentile_90_shortest_path_length:.4f}')
    if graph:
        plt.figure()
        plt.hist(deg.cpu().numpy(), bins=range(int(deg.min()), int(deg.max()) + 1), edgecolor='gray')
        plt.title(f"Degree Distribution of {name}")
        plt.xlabel("Degree")
        plt.ylabel("Frequency")
        plt.show()
    

def analyze_dataset_multi(name, graph = False):
    data_ = load_dataset(name)
    #print(len(data_))
    for i, data in enumerate(data_):
        deg = degree(data.edge_index[0], data.num_nodes)
        average_degree = deg.mean().item()
        print(f'{name}_{i+1}: avg_degree:{average_degree:.4f}, num_nodes:{data.num_nodes}, num_edges:{data.num_edges}, num_classes:{data_.num_classes}, num_features:{data_.num_features}')
        if graph:
            plt.figure()
            plt.hist(deg.cpu().numpy(), bins=range(int(deg.min()), int(deg.max()) + 1), edgecolor='gray')
            plt.title(f"Degree Distribution of {name}")
            plt.xlabel("Degree")
            plt.ylabel("Frequency")
            plt.show()



