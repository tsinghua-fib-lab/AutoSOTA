from typing import Optional

import os 
import copy, random, math
import numpy as np 
import pandas as pd 
import networkx as nx
import igraph as ig
import pyciphod.causal_discovery.federated.regret_based.iperi.utils as utils

from pyciphod.causal_discovery.federated.regret_based.ges.utils import dag_to_cpdag
from pyciphod.causal_discovery.federated.regret_based.iperi.utils import cpdag_to_ucpdag


CLIENT_SAMPLE_SIZE = [500, 1000, 2000]

NON_LINEAR_FUNCTIONS = [
        lambda m, x: np.dot(m, np.tanh(x)),
        lambda m, x: np.dot(m, np.sin(x)),
        lambda m, x: np.dot(m, x**2 * np.sign(x)),   # signed square
        lambda m, x: np.dot(m, np.abs(x)**0.5 * np.sign(x)),
    ]


class Dataset:
    
    def __init__(
            self,
            graph_type: str = 'erdos_renyi',
            n_samples_client: int = 1000, 
            n_clients: int = 5, 
            n_variables: Optional[int] = None,
            horizontal_split: bool = False,
            linear: bool = True,
            noise_distribution: str = 'uniform',
            seed: int = 1846
        ):
            self.graph_type = graph_type  
            self.n_samples_client = n_samples_client    
            self.n_clients = n_clients
            self.horizontal_split = horizontal_split 
            self.n_variables = n_variables 
            self.linear = linear
            self.noise_distribution = noise_distribution
            self.seed = seed

            # generate graph
            self.graph = self.get_graph(self.graph_type, self.n_variables)
            self.func = lambda m, x: np.dot(m, x) 

    def get_graph(self, graph_type: str = 'er', n_variables: Optional[int] = None):
        if graph_type == 'erdos_renyi':
            graph = nx.DiGraph(self.simulate_dag(n_variables, n_variables, 'ER'))
            assert nx.is_directed_acyclic_graph(graph)
            graph = self._add_shiedled_colliders(graph)
            return graph
        else:
            raise ValueError(f"Unknown graph type: {graph_type}")
        
    ########################################################################################
        
    """
    Code sourced from:
    https://github.com/lokali/FedCDH/blob/main/causallearn/utils/data_utils.py
    """
    
    def simulate_dag(self, d, s0, graph_type):
        """Simulate random DAG with some expected number of edges.

        Args:
            d (int): num of nodes
            s0 (int): expected num of edges
            graph_type (str): ER, SF, BP

        Returns:
            B (np.ndarray): [d, d] binary adj matrix of DAG
        """
        def _random_permutation(M):
            # np.random.permutation permutes first axis only
            P = np.random.permutation(np.eye(M.shape[0]))
            return P.T @ M @ P

        def _random_acyclic_orientation(B_und):
            return np.tril(_random_permutation(B_und), k=-1)

        def _graph_to_adjmat(G):
            return np.array(G.get_adjacency().data)

        if graph_type == 'ER':
            # Erdos-Renyi
            G_und = ig.Graph.Erdos_Renyi(n=d, m=d)#m=d, p=0.3 p=0.5
            B_und = _graph_to_adjmat(G_und)
            B = _random_acyclic_orientation(B_und)
        elif graph_type == 'SF':
            # Scale-free, Barabasi-Albert
            G = ig.Graph.Barabasi(n=d, m=d, directed=True)
            B = _graph_to_adjmat(G)
        elif graph_type == 'BP':
            # Bipartite, Sec 4.1 of (Gu, Fu, Zhou, 2018)
            top = int(0.2 * d)
            G = ig.Graph.Random_Bipartite(top, d - top, m=s0, directed=True, neimode=ig.OUT)
            B = _graph_to_adjmat(G)
        else:
            raise ValueError('unknown graph type')
        B_perm = _random_permutation(B)
        return B_perm
    
    def _add_shiedled_colliders(self, G: nx.DiGraph):
        v_structures = nx.dag.compute_v_structures(G)
        
        for collider in v_structures:
            G.add_edge(collider[0], collider[2]) 
            if not nx.is_directed_acyclic_graph(G):
                G.remove_edge(collider[0], collider[2])
            G.add_edge(collider[2], collider[0])
            if not nx.is_directed_acyclic_graph(G):
                G.remove_edge(collider[2], collider[0])
        return G
    
    ########################################################################################
        
    def generate(self, data_type='observational', save=True):
        splits = [self.n_samples_client] * self.n_clients

        if self.horizontal_split:
            for client in range(self.n_clients):
                splits[client] = random.choice(CLIENT_SAMPLE_SIZE)      
        
        if data_type == 'obs':
            datasets, interventions = self._gen_obs(splits=splits)
        elif data_type == 'struct':
            datasets, interventions = self._gen_struct(splits=splits)
        elif data_type == 'param':
            datasets, interventions = self._gen_param(splits=splits)
        elif data_type == 'param+struct':
            datasets, interventions = self._gen_param(splits=splits, mode='struct')
        else:
            raise ValueError(f"Unknown generation mode: {data_type}")
        
        if save: 
            # save each part into a csv file
            self.folder_name = f'data/{data_type}_{self.n_samples_client}_{self.n_clients}'
            self.folder_name += f'_{len(self.graph.nodes)}v_{len(self.graph.edges)}e'
            if self.horizontal_split:
                self.folder_name += '_uneven' 

            # if folder does not exist, create it
            if not os.path.exists(self.folder_name):
                os.makedirs(self.folder_name)
            
            for i, ds in enumerate(datasets):
                ds = pd.DataFrame(ds)
                ds.to_csv(f'{self.folder_name}/client_{i}.csv', index=False)

        cpdag = dag_to_cpdag(nx.to_numpy_array(self.graph, nodelist=list(self.graph.nodes)))
        if len(interventions) == 0:
            ucpdag = cpdag
        if len(interventions) > 0:
            interventions = [list(self.graph.nodes).index(i) for i in interventions]
            ucpdag = cpdag_to_ucpdag(
                nx.to_numpy_array(self.graph, nodelist=list(self.graph.nodes)),
                interventions
            )

        graphs = [nx.to_numpy_array(self.graph, nodelist=list(self.graph.nodes))]
        for intervention in interventions:
        
            graphs.append(
                utils.intervened_graph(
                    nx.to_numpy_array(self.graph, nodelist=list(self.graph.nodes)),
                    [intervention]
                )
            )
        
        return datasets, cpdag, ucpdag, graphs
    
    def _discretize(self,x, n_bins=3):
        """Quantile-based discretization"""
        bins = np.quantile(x, np.linspace(0, 1, n_bins + 1))
        bins = np.unique(bins)
        return np.digitize(x, bins[1:-1])  # returns 0, 1, ..., n_bins-1
    
    def _gen_obs(self, splits):

        n_samples = sum(splits)

        prime_causes = []
        for node in self.graph.nodes:
            if self.graph.in_degree(node) == 0:
                prime_causes.append(node)

        noise = {}
        for node in self.graph.nodes:
            if self.noise_distribution == 'normal':
                # random choice 1 or -1 to avoid noise close to zero
                i = random.choice([1, -1])
                noise[node] = i * np.abs(np.random.normal(loc=0.55, scale=0.15, size=(n_samples)))
            elif self.noise_distribution == 'uniform':
                # random choice 1 or -1
                i = random.choice([1, -1])
                if i > 0:
                    low = 0.4
                    high = 0.7
                else:
                    low = -0.7
                    high = -0.4
                noise[node] = np.random.uniform(low=low, high=high, size=(n_samples))
            else:
                raise ValueError(f"Unknown noise type: {self.noise_distribution}")

        dataset = copy.copy(noise)

        for node in self.graph.nodes:
            if self.graph.in_degree(node) > 0:
                if not self.linear:
                    self.func = random.choice(NON_LINEAR_FUNCTIONS)
                in_edges = self.graph.in_edges(node)
                m = np.array(
                    [random.choice([random.uniform(0.1, 1), random.uniform(-0.1, -1)]) 
                     for _ in range(len(in_edges))]
                )
                x = np.array([dataset[edge[0]].ravel() for edge in in_edges])
                epsilon = noise[node].ravel()
                dataset[node] = self.func(m, x) + epsilon

        dataset = pd.DataFrame(dataset, columns=self.graph.nodes)
        datasets = []
        start = 0
        for split in splits:
            datasets.append(dataset.iloc[start:start+split, :].reset_index(drop=True))
            start += split

        if not self.linear:
            # discretize the dataset for non-linear case
            datasets = [ds.apply(lambda col: self._discretize(col)) for ds in datasets]
        
        return datasets, []

    def _gen_struct(self, splits):
        datasets = []
        interventions = []
        dataset, _ = self._gen_obs(splits=[splits[0]])
        datasets.append(dataset[0])
        # find shielded colliders 
        # colliders = all triples (u,v,w) where u->v<-w (compatible with networkx >=3.0)
        colliders = []
        for v in self.graph.nodes():
            in_edges_v = list(self.graph.in_edges(v))
            for i in range(len(in_edges_v)):
                for j in range(i + 1, len(in_edges_v)):
                    colliders.append((in_edges_v[i][0], v, in_edges_v[j][0]))
        shielded_colliders = [c for c in colliders 
                              if (self.graph.has_edge(c[0], c[2]) or
                                  self.graph.has_edge(c[2], c[0]))]
        
        for split in splits[1:]:
            observational_graph = copy.deepcopy(self.graph)
            # randomly choose a node to intervene on
            if len(shielded_colliders) > 0:
                collider = shielded_colliders.pop()
                if self.graph.has_edge(collider[0], collider[2]):
                    intervention_node = collider[2]
                else:  
                    intervention_node = collider[0]
            else:
                intervention_node = random.choice(list(observational_graph.nodes))

            interventions.append(intervention_node)
            # create a copy of the graph and remove all incoming edges to the intervention node
            intervened_graph = copy.deepcopy(observational_graph)
            intervened_graph.remove_edges_from(list(intervened_graph.in_edges(intervention_node)))
            self.graph = intervened_graph
            # generate data from the intervened graph
            dataset, _ = self._gen_obs(splits=[split])
            self.graph = observational_graph
            datasets.append(dataset[0])
        
        return datasets, interventions
        
    def _gen_param(self, splits, mode='obs'):

        if mode == 'obs':
            f = self._gen_obs
        elif mode == 'struct':
            f = self._gen_struct
        else:
            raise ValueError(f"Unknown generation mode: {mode}")
        
        datasets = []
        interventions = []
        for split in splits:
            dataset, ints = f(splits=[split])
            interventions.extend(ints)
            datasets.append(dataset[0])

        return datasets, interventions
