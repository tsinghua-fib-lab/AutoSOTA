# Copyright (c) 2024-Current Jialiang Wang
# License: Apache-2.0 license

import json
import os


class GraphDatasetUnderstanding:
    def __init__(self, dataset_name, task, user_description=None, root_dir='GraphGym/run/datasets/',
                 no_statistics=False, use_semantic=False, is_bench=False, seed=42):
        self.root_dir = root_dir
        self.no_statistics = no_statistics
        self.is_bench = is_bench
        self.dataset_name = dataset_name
        self.seed = seed

        if task == 'node_classification':
            task = 'NC'
        elif task == 'link_prediction':
            task = 'LP'
        elif task == 'graph_classification':
            task = 'GC'
        else:
            raise ValueError(f"Unknown task: {task}")
        self.if_GC = task == 'GC'

        if is_bench:
            predefined_descriptions_path = 'graph_understanding/predefined_descriptions (bench).json'
        else:
            predefined_descriptions_path = 'graph_understanding/predefined_descriptions (unseen).json'
        with open(predefined_descriptions_path, 'r') as file:
            predefined_semantic_descriptions = json.load(file)

        if is_bench:
            if use_semantic:
                self.semantic_description = user_description or predefined_semantic_descriptions.get(
                    f"{task}-{self.dataset_name}", None
                )
                if self.semantic_description is None:
                    raise ValueError(
                        f"Semantic description for benchmark dataset {self.dataset_name} is not provided."
                    )
            else:
                self.semantic_description = None
        else:
            default_msg = (
                "However, the user does not provide a semantic description for the target "
                "graph dataset (unseen), please understand it based entirely on the following "
                "graph properties. "
            )
            self.semantic_description = user_description or predefined_semantic_descriptions.get(
                f"{task}-{self.dataset_name}", default_msg
            )

    def process(self):
        if self.no_statistics:
            description = ""
        else:
            properties_file = os.path.join(
                self.root_dir,
                f"{self.dataset_name}/{self.dataset_name}_properties.json",
            )
            if os.path.exists(properties_file):
                with open(properties_file, "r", encoding="utf-8") as file:
                    properties = json.load(file)
                description = properties_to_natural_language(
                    properties,
                    self.dataset_name,
                    self.is_bench,
                    self.if_GC,
                )
            else:
                from .loader import load_pyg
                from .summarizer import GraphSummarizer

                dataset = load_pyg(self.dataset_name, self.root_dir)
                summarizer = GraphSummarizer(dataset, self.dataset_name, self.seed)
                summarizer.summarize(self.root_dir)
                description = summarizer.to_natural_language(self.is_bench, self.if_GC)

        if self.semantic_description:
            description += self.semantic_description

        if description == "":
            raise ValueError("No description is generated for the dataset.")
        return description


def properties_to_natural_language(properties, dataset_name, is_bench=False, graph_classifications=False):
    if is_bench:
        dataset_label = 'prior ' + dataset_name + ' graph dataset (benchmark)'
    else:
        dataset_label = 'target graph dataset (unseen)'

    homophily_sentence = ""
    if not graph_classifications:
        homophily_sentence = (
            f"The label homophily ratio is {properties['homophily_ratio']:.4f}, indicating "
            f"that {properties['homophily_ratio'] * 100:.2f}% of connected nodes share the "
            "same label. "
        )

    return (
        f"The {dataset_label} has {properties['num_nodes']} nodes and "
        f"{properties['num_edges']} edges, resulting in a density of "
        f"{properties['density']:.4f}. The average degree is "
        f"{properties['degree_mean']:.2f} (±{properties['degree_std']:.2f}), ranging from "
        f"{properties['degree_min']:.0f} to {properties['degree_max']:.0f}. There are "
        f"{properties['num_hubs']} hubs in the graph, defined as nodes with a degree higher "
        f"than two standard deviations above the mean. This graph's approximate diameter is "
        f"{properties['diameter']}, and the approximate average shortest path length is "
        f"{properties['avg_shortest_path_length']:.2f}. {homophily_sentence}"
        f"Spectral analysis reveals a spectral gap of {properties['spectral_gap']:.8f}. "
        f"This graph has {properties['num_communities']} communities with a modularity of "
        f"{properties['modularity']:.4f}. The average clustering coefficient is "
        f"{properties['avg_clustering_coefficient']:.4f}, and the graph contains "
        f"{properties['total_triangles']:.0f} triangles. Node features have a dimensionality "
        f"of {properties['feature_dimensionality']} with an average variance of "
        f"{properties['feature_variance']:.4f}. The assortativity coefficient is "
        f"{properties['assortativity']:.4f}. The eigenvector centrality has a mean of "
        f"{properties['eigenvector_centrality_mean']:.4f} "
        f"(±{properties['eigenvector_centrality_std']:.4f}). Centrality measures show that "
        f"the mean betweenness centrality is {properties['betweenness_centrality_mean']:.4f} "
        f"(±{properties['betweenness_centrality_std']:.4f}), closeness centrality is "
        f"{properties['closeness_centrality_mean']:.4f} "
        f"(±{properties['closeness_centrality_std']:.4f}), and PageRank is "
        f"{properties['pagerank_mean']:.4f} (±{properties['pagerank_std']:.4f}). "
    )
