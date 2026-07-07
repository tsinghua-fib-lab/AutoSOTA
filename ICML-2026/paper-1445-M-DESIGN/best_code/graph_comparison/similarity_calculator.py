# Copyright (c) 2024-Current Jialiang Wang
# License: Apache-2.0 license

import networkx as nx
import numpy as np
import os
import json
from scipy.spatial.distance import cosine

class GraphSimilarityCalculator:
    def __init__(self, root_dir='data/'):
        """
        Initialize the GraphSimilarityCalculator.
        :param root_dir: Root directory where subgraphs are stored.
        """
        self.root_dir = root_dir

    def calculate_similarity(self, dataset_name_1, dataset_name_2, measures, num_samples=5):
        """
        Calculate similarity scores between two datasets using specified measures.
        :param dataset_name_1: Name of the first dataset.
        :param dataset_name_2: Name of the second dataset.
        :param measures: List of similarity measures to be used.
        :param num_samples: Number of subgraphs to sample.
        :return: A dictionary with similarity scores.
        """
        total_scores = {measure: 0 for measure in measures}

        for i in range(1, num_samples + 1):
            # Load subgraphs from files
            subgraph_1 = self._load_subgraph(dataset_name_1, i)
            subgraph_2 = self._load_subgraph(dataset_name_2, i)

            # Convert subgraphs to NetworkX graphs
            graph_1 = self._convert_to_networkx(subgraph_1)
            graph_2 = self._convert_to_networkx(subgraph_2)

            # Calculate similarity based on the chosen measures
            for measure in measures:
                if measure == 'jaccard':
                    score = self._jaccard_similarity(graph_1, graph_2)
                    total_scores['jaccard'] += score
                if measure == 'graph_edit_distance':
                    score = self._graph_edit_distance(graph_1, graph_2)
                    total_scores['graph_edit_distance'] += score
                if measure == 'cosine':
                    score = self._cosine_similarity(graph_1, graph_2)
                    total_scores['cosine'] += score
                if measure == 'spectral_distance':
                    score = self._spectral_distance(graph_1, graph_2)
                    total_scores['spectral_distance'] += score

        # Average the scores
        averaged_scores = {measure: total / num_samples for measure, total in total_scores.items()}

        return averaged_scores

    def _load_subgraph(self, dataset_name, subgraph_number):
        """
        Load a subgraph from a file.
        :param dataset_name: Name of the dataset.
        :param subgraph_number: Number of the subgraph.
        :return: Subgraph data.
        """
        file_path = os.path.join(self.root_dir, f"{dataset_name}_subgraph_{subgraph_number}.json")
        with open(file_path, 'r') as file:
            subgraph = json.load(file)
        return subgraph

    def _convert_to_networkx(self, subgraph):
        """
        Convert subgraph data to a NetworkX graph.
        :param subgraph: Subgraph data.
        :return: A NetworkX graph.
        """
        g = nx.Graph()
        g.add_edges_from(subgraph['edge_index'])
        return g

    def _jaccard_similarity(self, graph_1, graph_2):
        """
        Calculate Jaccard similarity between two graphs.
        :param graph_1: First graph.
        :param graph_2: Second graph.
        :return: Jaccard similarity score.
        """
        intersection = len(set(graph_1.edges()).intersection(set(graph_2.edges())))
        union = len(set(graph_1.edges()).union(set(graph_2.edges())))
        return intersection / union if union != 0 else 0

    def _graph_edit_distance(self, graph_1, graph_2):
        """
        Calculate the graph edit distance between two graphs.
        This can be computationally expensive for large graphs.
        :param graph_1: First graph.
        :param graph_2: Second graph.
        :return: Graph edit distance score.
        """
        return nx.graph_edit_distance(graph_1, graph_2)

    def _cosine_similarity(self, graph_1, graph_2):
        """
        Calculate cosine similarity between vector representations of two graphs.
        :param graph_1: First graph.
        :param graph_2: Second graph.
        :return: Cosine similarity score.
        """
        # Example using degree sequences as graph vectors
        vec_1 = [graph_1.degree(n) for n in sorted(graph_1.nodes())]
        vec_2 = [graph_2.degree(n) for n in sorted(graph_2.nodes())]

        # Padding shorter vector with zeros
        length_diff = len(vec_1) - len(vec_2)
        if length_diff > 0:
            vec_2.extend([0] * length_diff)
        elif length_diff < 0:
            vec_1.extend([0] * -length_diff)

        return 1 - cosine(vec_1, vec_2)

    def _spectral_distance(self, graph_1, graph_2):
        """
        Calculate spectral distance based on the eigenvalues of the graph Laplacian.
        :param graph_1: First graph.
        :param graph_2: Second graph.
        :return: Spectral distance score.
        """
        laplacian_1 = nx.normalized_laplacian_matrix(graph_1).todense()
        laplacian_2 = nx.normalized_laplacian_matrix(graph_2).todense()

        eigenvalues_1 = np.sort(np.linalg.eigvals(laplacian_1))
        eigenvalues_2 = np.sort(np.linalg.eigvals(laplacian_2))

        # Padding shorter eigenvalue array with zeros
        length_diff = len(eigenvalues_1) - len(eigenvalues_2)
        if length_diff > 0:
            eigenvalues_2 = np.pad(eigenvalues_2, (0, length_diff), 'constant')
        elif length_diff < 0:
            eigenvalues_1 = np.pad(eigenvalues_1, (0, -length_diff), 'constant')

        return np.linalg.norm(eigenvalues_1 - eigenvalues_2)

