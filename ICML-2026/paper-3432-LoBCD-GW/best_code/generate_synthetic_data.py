import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
import time
import ot
import scipy.sparse
import torch
from scipy import linalg
from scipy import sparse
import scipy.io
from collections import defaultdict
import pickle
from scipy.sparse import csr_matrix, lil_matrix
import warnings
warnings.filterwarnings("ignore")
seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

clique_size = 200
p_in = 0.2
p_out = 0.02
num_trials = 5
num_nodes = [500, 1000, 1500, 2000, 2500]
noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
graph_pairs = {}
for num_node in num_nodes:
    for noise_level in noise_levels:
        print((num_node, noise_level))
        graph_pairs[(num_node, noise_level)] = []
        for ii in range(num_trials):
            G_src = nx.powerlaw_cluster_graph(n=num_node, m=int(clique_size * p_in),
                                      p=p_out * clique_size / num_node)
            G_dst1 = add_noisy_edges(G_src, noise_level)
            G_dst = add_noisy_nodes(G_dst1, noise_level)
            graph_pairs[(num_node, noise_level)].append((G_src, G_dst))
            G_src = nx.gaussian_random_partition_graph(n=num_node, s=clique_size, v=5, 
                                            p_in=p_in, p_out=p_out)
            G_dst1 = add_noisy_edges(G_src, noise_level)
            G_dst = add_noisy_nodes(G_dst1, noise_level)
            graph_pairs[(num_node, noise_level)].append((G_src, G_dst))

with open('data/Random/graph1.pk', 'wb') as f:
    pickle.dump(graph_pairs, f)
