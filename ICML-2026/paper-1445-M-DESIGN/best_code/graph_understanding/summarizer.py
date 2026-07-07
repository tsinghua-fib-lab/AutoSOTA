# Copyright (c) 2024-Current Jialiang Wang
# License: Apache-2.0 license
# This file is responsible for translating subgraphs to a graph description language.

import os
import json
import torch
import random
import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from community import community_louvain
from torch_geometric.transforms import ToUndirected
from torch_geometric.utils import (
    degree,
    to_networkx,
    get_laplacian,
    to_undirected
)


class GraphSummarizer:
    def __init__(self, dataset, dataset_name=None, seed=42):
        """
        Initializes the GraphSummarizer module.

        Parameters:
        - dataset: PyG dataset object.
        - dataset_name: Name of the dataset.
        - seed: Random seed for reproducibility.

        Attributes:
        - properties: A dictionary containing the properties of the dataset.
        """
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.seed = seed
        self.properties = {}
    
    def summarize(self, dataset_path):
        """
        Summarizes the properties of the dataset.

        Parameters:
        - dataset_path: Path to the folder containing the dataset.

        Returns:
        - properties: A dictionary containing the properties of the dataset.
        """
        properties_file = os.path.join(dataset_path, f'{self.dataset_name}/{self.dataset_name}_properties.json')
        if os.path.exists(properties_file):
            self.load_properties(properties_file)
            return self.properties

        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Check if dataset is graph classification (multiple graphs)
        if isinstance(self.dataset, list) or hasattr(self.dataset, '__getitem__'):
            graph_list = [to_networkx(data, to_undirected=True) for data in self.dataset]
            #pyg_list = [ToUndirected()(data) for data in self.dataset]
            pyg_list = []
            for data in self.dataset:
                if data.edge_index.numel() > 0:
                    edge_index, edge_attr = to_undirected(
                        data.edge_index, data.edge_attr if data.edge_attr is not None else torch.ones((data.edge_index.size(1), 1)),
                        num_nodes=data.num_nodes, reduce='mean'  # You can also use 'sum' or 'max'
                    )
                    data.edge_index = edge_index
                    data.edge_attr = edge_attr
                pyg_list.append(data)
            self.summarize_multiple_graphs(pyg_list, graph_list)
        else:
            pyg_data = ToUndirected()(self.dataset[0])
            nx_graph = to_networkx(self.dataset[0], to_undirected=True)
            self.summarize_single_graph(pyg_data, nx_graph)

        self.properties = {key: self.convert_numpy_types(value) for key, value in self.properties.items()}
        self.save_properties(properties_file)
        return self.properties
    
    def summarize_single_graph(self, pyg_data, nx_graph, graph_classifications=False):
        """
        Computes graph statistics for a single graph dataset.
        """
        nodes = list(nx_graph.nodes())
        sample_size = min(1000, len(nodes))
        sampled_nodes = np.random.choice(nodes, size=sample_size, replace=False)

        self.summarize_degree_distribution(pyg_data)
        self.summarize_graph_density(pyg_data)
        self.summarize_diameter(nx_graph, sampled_nodes)
        if not graph_classifications:
            self.summarize_homophily(pyg_data)
        self.summarize_spectral_properties(pyg_data)
        self.summarize_community_structure(nx_graph)
        self.summarize_node_feature(pyg_data)
        self.summarize_hubs(pyg_data)
        self.summarize_assortativity(nx_graph)
        self.summarize_random_walk_properties(nx_graph)
        self.summarize_edge_weight_distribution(pyg_data)
        self.summarize_structural_diversity(nx_graph)
        self.summarize_node_centrality(nx_graph, sampled_nodes)

    def summarize_multiple_graphs(self, pyg_list, graph_list):
        """
        Computes graph statistics for multiple graph datasets.
        """
        num_graphs = len(pyg_list)
        for pyg_data, nx_graph in zip(pyg_list, graph_list):
            self.summarize_single_graph(pyg_data, nx_graph, True)
        
        for key in self.properties.keys():
            self.properties[key] /= num_graphs

    def to_natural_language(self, is_bench=False, graph_classifications=False):
        """
        Converts the computed properties to a natural language description.

        Returns:
        - description: A natural language description of the dataset properties.
        """
        # Generate Natural Language Description
        if is_bench:
            dataset_name = 'prior ' + self.dataset_name + ' graph dataset (benchmark)'
        else:
            dataset_name = 'target graph dataset (unseen)'
        
        if graph_classifications:
            description = (
                f"The {dataset_name} has {self.properties['num_nodes']} nodes and {self.properties['num_edges']} edges, "
                f"resulting in a density of {self.properties['density']:.4f}. The average degree is "
                f"{self.properties['degree_mean']:.2f} (±{self.properties['degree_std']:.2f}), "
                f"ranging from {self.properties['degree_min']:.0f} to {self.properties['degree_max']:.0f}. "
                f"There are {self.properties['num_hubs']} hubs in the graph, defined as nodes with a degree "
                f"higher than two standard deviations above the mean. "
                f"This graph's approximate diameter is {self.properties['diameter']}, and the approximate average shortest path length is "
                f"{self.properties['avg_shortest_path_length']:.2f}. "
                f"Spectral analysis reveals a spectral gap of {self.properties['spectral_gap']:.8f}. This graph "
                f"has {self.properties['num_communities']} communities with a modularity of "
                f"{self.properties['modularity']:.4f}. The average clustering coefficient is "
                f"{self.properties['avg_clustering_coefficient']:.4f}, and the graph contains "
                f"{self.properties['total_triangles']:.0f} triangles. "
                f"Node features have a dimensionality of {self.properties['feature_dimensionality']} with an "
                f"average variance of {self.properties['feature_variance']:.4f}. The assortativity coefficient is "
                f"{self.properties['assortativity']:.4f}, indicating "
                f"{'positive' if self.properties['assortativity'] > 0 else 'negative' if self.properties['assortativity'] < 0 else 'no'} degree correlation. "
                f"The eigenvector centrality has a mean of {self.properties['eigenvector_centrality_mean']:.4f} (±{self.properties['eigenvector_centrality_std']:.4f}). "
                f"Centrality measures show that the mean betweenness centrality is "
                f"{self.properties['betweenness_centrality_mean']:.4f} (±{self.properties['betweenness_centrality_std']:.4f}), closeness centrality is "
                f"{self.properties['closeness_centrality_mean']:.4f} (±{self.properties['closeness_centrality_std']:.4f}), and PageRank is "
                f"{self.properties['pagerank_mean']:.4f} (±{self.properties['pagerank_std']:.4f}). "
            )
        else:
            description = (
                f"The {dataset_name} has {self.properties['num_nodes']} nodes and {self.properties['num_edges']} edges, "
                f"resulting in a density of {self.properties['density']:.4f}. The average degree is "
                f"{self.properties['degree_mean']:.2f} (±{self.properties['degree_std']:.2f}), "
                f"ranging from {self.properties['degree_min']:.0f} to {self.properties['degree_max']:.0f}. "
                f"There are {self.properties['num_hubs']} hubs in the graph, defined as nodes with a degree "
                f"higher than two standard deviations above the mean. "
                f"This graph's approximate diameter is {self.properties['diameter']}, and the approximate average shortest path length is "
                f"{self.properties['avg_shortest_path_length']:.2f}. The label homophily ratio is "
                f"{self.properties['homophily_ratio']:.4f}, indicating that {self.properties['homophily_ratio']*100:.2f}% "
                f"of connected nodes share the same label. "
                f"Spectral analysis reveals a spectral gap of {self.properties['spectral_gap']:.8f}. This graph "
                f"has {self.properties['num_communities']} communities with a modularity of "
                f"{self.properties['modularity']:.4f}. The average clustering coefficient is "
                f"{self.properties['avg_clustering_coefficient']:.4f}, and the graph contains "
                f"{self.properties['total_triangles']:.0f} triangles. "
                f"Node features have a dimensionality of {self.properties['feature_dimensionality']} with an "
                f"average variance of {self.properties['feature_variance']:.4f}. The assortativity coefficient is "
                f"{self.properties['assortativity']:.4f}, indicating "
                f"{'positive' if self.properties['assortativity'] > 0 else 'negative' if self.properties['assortativity'] < 0 else 'no'} degree correlation. "
                f"The eigenvector centrality has a mean of {self.properties['eigenvector_centrality_mean']:.4f} (±{self.properties['eigenvector_centrality_std']:.4f}). "
                f"Centrality measures show that the mean betweenness centrality is "
                f"{self.properties['betweenness_centrality_mean']:.4f} (±{self.properties['betweenness_centrality_std']:.4f}), closeness centrality is "
                f"{self.properties['closeness_centrality_mean']:.4f} (±{self.properties['closeness_centrality_std']:.4f}), and PageRank is "
                f"{self.properties['pagerank_mean']:.4f} (±{self.properties['pagerank_std']:.4f}). "
            )

        return description

    def save_properties(self, save_path):
        """
        Saves the computed properties to a JSON file.

        Parameters:
        - save_path: Path to the file where the properties will be saved.
        """
        with open(save_path, 'w') as file:
            json.dump(self.properties, file, indent=4)
    
    def load_properties(self, load_path):
        """
        Loads the computed properties from a JSON file.

        Parameters:
        - load_path: Path to the file where the properties are saved.
        """
        with open(load_path, 'r') as file:
            self.properties = json.load(file)

    def summarize_degree_distribution(self, pyg_data):
        deg = degree(pyg_data.edge_index[0], num_nodes=pyg_data.num_nodes)
        if 'degree_mean' in self.properties:
            self.properties['degree_mean'] += deg.mean().item()
            self.properties['degree_std'] += deg.std().item()
            self.properties['degree_max'] += deg.max().item()
            self.properties['degree_min'] += deg.min().item()
        else:
            self.properties['degree_mean'] = deg.mean().item()
            self.properties['degree_std'] = deg.std().item()
            self.properties['degree_max'] = deg.max().item()
            self.properties['degree_min'] = deg.min().item()
    
    def summarize_graph_density(self, pyg_data):
        num_nodes = pyg_data.num_nodes
        num_edges = pyg_data.edge_index.size(1)
        possible_edges = num_nodes * (num_nodes - 1) / 2
        density = num_edges / possible_edges
        if 'num_nodes' in self.properties:
            self.properties['num_nodes'] += num_nodes
            self.properties['num_edges'] += num_edges
            self.properties['density'] += density
        else:
            self.properties['num_nodes'] = num_nodes
            self.properties['num_edges'] = num_edges
            self.properties['density'] = density
    
    def summarize_diameter(self, nx_graph, sampled_nodes):
        # Approximate diameter using double sweep algorithm
        nodes = list(nx_graph.nodes())
        u = np.random.choice(nodes)
        lengths = nx.single_source_shortest_path_length(nx_graph, u)
        v = max(lengths, key=lengths.get)
        lengths = nx.single_source_shortest_path_length(nx_graph, v)
        w = max(lengths, key=lengths.get)
        approx_diameter = lengths[w]

        # Approximate average shortest path length by sampling
        path_lengths = []
        for node in sampled_nodes:
            lengths = nx.single_source_shortest_path_length(nx_graph, node)
            path_lengths.extend([l for target, l in lengths.items() if target in sampled_nodes])
        avg_shortest_path_length = sum(path_lengths) / len(path_lengths)
        if 'avg_shortest_path_length' in self.properties:
            self.properties['diameter'] += approx_diameter
            self.properties['avg_shortest_path_length'] += avg_shortest_path_length
        else:
            self.properties['diameter'] = approx_diameter
            self.properties['avg_shortest_path_length'] = avg_shortest_path_length

    def summarize_homophily(self, pyg_data):
        def homophily_ratio(edge_index, labels):
            src, dst = edge_index
            same_label = (labels[src] == labels[dst]).sum().item()
            total_edges = edge_index.size(1)
            return same_label / total_edges
        labels = pyg_data.y
        h_ratio = homophily_ratio(pyg_data.edge_index, labels)
        if 'homophily_ratio' in self.properties:
            self.properties['homophily_ratio'] += h_ratio
        else:
            self.properties['homophily_ratio'] = h_ratio

    def summarize_spectral_properties(self, pyg_data):
        edge_index, edge_weight = get_laplacian(
            pyg_data.edge_index, normalization='sym', num_nodes=pyg_data.num_nodes
        )
        L = sp.coo_matrix(
            (edge_weight.numpy(), (edge_index[0].numpy(), edge_index[1].numpy())),
            shape=(pyg_data.num_nodes, pyg_data.num_nodes),
        )
        # Compute the smallest non-zero eigenvalues
        k = max(1, min(6, pyg_data.num_nodes - 2))  # Ensure k is less than n - 1
        eigenvalues, _ = eigsh(L, k=k, which='SM')
        # Spectral Gap
        spectral_gap = eigenvalues[1] - eigenvalues[0] if len(eigenvalues) > 1 else 0
        if 'spectral_gap' in self.properties:
            self.properties['spectral_gap'] += spectral_gap
        else:
            self.properties['spectral_gap'] = spectral_gap
    
    def summarize_community_structure(self, nx_graph):
        partition = community_louvain.best_partition(nx_graph, random_state=self.seed)
        num_communities = len(set(partition.values()))
        modularity = community_louvain.modularity(partition, nx_graph)
        clustering_coeffs = nx.clustering(nx_graph)
        avg_clustering_coefficient = sum(clustering_coeffs.values()) / len(clustering_coeffs)
        if 'num_communities' in self.properties:
            self.properties['num_communities'] += num_communities
            self.properties['modularity'] += modularity
            self.properties['avg_clustering_coefficient'] += avg_clustering_coefficient
        else:
            self.properties['num_communities'] = num_communities
            self.properties['modularity'] = modularity
            self.properties['avg_clustering_coefficient'] = avg_clustering_coefficient

    def summarize_node_feature(self, pyg_data):
        feature_dimensionality = pyg_data.num_node_features
        features = pyg_data.x.numpy()
        feature_variance = np.var(features, axis=0).mean()
        if 'feature_dimensionality' in self.properties:
            self.properties['feature_dimensionality'] += feature_dimensionality
            self.properties['feature_variance'] += feature_variance
        else:
            self.properties['feature_dimensionality'] = feature_dimensionality
            self.properties['feature_variance'] = feature_variance

    def summarize_hubs(self, pyg_data):
        deg = degree(pyg_data.edge_index[0], num_nodes=pyg_data.num_nodes)
        degree_threshold = deg.mean().item() + 2 * deg.std().item()
        hubs = deg > degree_threshold
        num_hubs = hubs.sum().item()
        if 'num_hubs' in self.properties:
            self.properties['num_hubs'] += num_hubs
        else:
            self.properties['num_hubs'] = num_hubs

    def summarize_assortativity(self, nx_graph):
        if nx_graph.number_of_nodes() < 3:
            assort_coeff = 0.0
        else:
            assort_coeff = nx.degree_assortativity_coefficient(nx_graph)
        if 'assortativity' in self.properties:
            self.properties['assortativity'] += assort_coeff
        else:
            self.properties['assortativity'] = assort_coeff
    
    def summarize_random_walk_properties(self, nx_graph):
        if nx_graph.number_of_nodes() < 3:
            eigen_centrality_values = np.array([0, 0])
        else:
            eigen_centrality = nx.eigenvector_centrality_numpy(nx_graph, max_iter=1000)
            eigen_centrality_values = np.array(list(eigen_centrality.values()))
        if 'eigenvector_centrality_mean' in self.properties:
            self.properties['eigenvector_centrality_mean'] += eigen_centrality_values.mean()
            self.properties['eigenvector_centrality_std'] += eigen_centrality_values.std()
        else:
            self.properties['eigenvector_centrality_mean'] = eigen_centrality_values.mean()
            self.properties['eigenvector_centrality_std'] = eigen_centrality_values.std()
    
    def summarize_edge_weight_distribution(self, pyg_data):
        edge_weights = torch.ones(pyg_data.edge_index.size(1))
        if 'edge_weight_mean' in self.properties:
            self.properties['edge_weight_mean'] += edge_weights.mean().item()
            self.properties['edge_weight_std'] += edge_weights.std().item()
        else:
            self.properties['edge_weight_mean'] = edge_weights.mean().item()
            self.properties['edge_weight_std'] = edge_weights.std().item()
    
    def summarize_structural_diversity(self, nx_graph):
        num_triangles = sum(nx.triangles(nx_graph).values()) // 3
        if 'total_triangles' in self.properties:
            self.properties['total_triangles'] += num_triangles
        else:
            self.properties['total_triangles'] = num_triangles
    
    def summarize_node_centrality(self, nx_graph, sampled_nodes):
        # Approximate Betweenness Centrality
        num_nodes = nx_graph.number_of_nodes()
        k = min(100, num_nodes)
        betweenness = nx.betweenness_centrality(nx_graph, k=k, seed=self.seed, normalized=True, endpoints=False)
        betweenness_values = np.array(list(betweenness.values()))

        # Closeness Centrality (Approximate)
        closeness = {}
        for node in sampled_nodes:
            lengths = nx.single_source_shortest_path_length(nx_graph, node)
            total_length = sum(lengths.values())
            if total_length > 0.0 and len(lengths) > 1:
                closeness[node] = (len(lengths) - 1) / total_length
            else:
                closeness[node] = 0.0
        closeness_values = np.array(list(closeness.values()))

        # PageRank (Use fewer iterations)
        pagerank = nx.pagerank(nx_graph, max_iter=100, tol=1e-04)
        pagerank_values = np.array(list(pagerank.values()))

        if 'closeness_centrality_mean' in self.properties:
            self.properties['closeness_centrality_mean'] += closeness_values.mean()
            self.properties['closeness_centrality_std'] += closeness_values.std()
            self.properties['betweenness_centrality_mean'] += betweenness_values.mean()
            self.properties['betweenness_centrality_std'] += betweenness_values.std()
            self.properties['pagerank_mean'] += pagerank_values.mean()
            self.properties['pagerank_std'] += pagerank_values.std()
        else:
            self.properties['closeness_centrality_mean'] = closeness_values.mean()
            self.properties['closeness_centrality_std'] = closeness_values.std()
            self.properties['betweenness_centrality_mean'] = betweenness_values.mean()
            self.properties['betweenness_centrality_std'] = betweenness_values.std()
            self.properties['pagerank_mean'] = pagerank_values.mean()
            self.properties['pagerank_std'] = pagerank_values.std()

    @staticmethod
    def convert_numpy_types(obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        else:
            return obj

