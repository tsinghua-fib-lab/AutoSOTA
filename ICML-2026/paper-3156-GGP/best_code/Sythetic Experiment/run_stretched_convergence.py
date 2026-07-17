import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from matplotlib.tri import Triangulation

from torch_geometric.data import Data
import torch_geometric.utils as tgut
from torch_geometric.nn import MessagePassing

# ==========================================
# 1. Model Definitions
# ==========================================
class SpecificAggrLayer(MessagePassing):
    def __init__(self):
        super(SpecificAggrLayer, self).__init__(aggr='add', flow='source_to_target')

    def forward(self, x, edge_index):
        num_2edges = edge_index.size(1)
        
        # Safely handle extremely sparse graphs to prevent division by zero
        if num_2edges == 0:
            return x
            
        ax = self.propagate(edge_index, x=x)
        norm_factor = 1.0 / (num_2edges ** 0.5)
        out = ax * norm_factor + x
        return out

    def message(self, x_j):
        return x_j

class StretchedGCN(nn.Module):
    def __init__(self, size_layers, variant='inv'):
        super(StretchedGCN, self).__init__()
        self.aggr_layer = SpecificAggrLayer()
        self.variant = variant
        
        self.fcs = nn.ModuleList()
        for i in range(len(size_layers) - 1):
            self.fcs.append(nn.Linear(size_layers[i], size_layers[i+1]))

    def forward(self, x, edge_index):
        for i in range(len(self.fcs) - 1):
            x = self.aggr_layer(x, edge_index)
            x = self.fcs[i](x)
            x = F.relu(x)
            
        x = self.aggr_layer(x, edge_index)
        x = self.fcs[-1](x)
        
        if self.variant == 'inv':
            return x.mean(axis=0)
        return x

# ==========================================
# 2. Graph Generation Utilities
# ==========================================
def nx2tg(G, node_attr='node_attr'):
    if len(G.edges) == 0:
        edgelist = torch.empty((2, 0), dtype=torch.long)
    else:
        edgelist = torch.LongTensor([list(e) for e in G.edges]).t()
        
    edgelist = tgut.to_undirected(edgelist)
    D = Data(x=torch.ones(len(G)), edge_index=edgelist)

    if node_attr is not None:
        X = torch.Tensor([G.nodes[i][node_attr] for i in G.nodes])
        if len(X.shape) == 1:
            X = X.view(-1, 1)
        D.x = X
    return D

def surface_uniform(n, fz=lambda x, y: 0):
    pos = np.zeros((n, 3))
    pos[:, 0], pos[:, 1] = np.random.rand(n), np.random.rand(n)
    pos[:, 2] = fz(pos[:, 0], pos[:, 1])
    tri = Triangulation(pos[:, 0], pos[:, 1])
    return pos, tri.triangles

def generate_edges_gaussian(X, sigma, alpha):
    n = X.shape[0]
    sq_dists = pdist(X, metric='sqeuclidean')
    sq_dists_mat = squareform(sq_dists)
    probs_mat = alpha * np.exp(-sq_dists_mat / (2 * sigma**2))
    
    i, j = np.tril_indices(n, k=-1)
    edge_mask = np.random.rand(len(i)) < probs_mat[i, j]
    return list(zip(i[edge_mask], j[edge_mask]))

def random_graph_similarity(X, f=None, alpha=1, bandwidth=1):
    n = X.shape[0]
    G = nx.empty_graph(n)
    for i in range(n):
        G.nodes[i]['latent'] = X[i, :]
        if f is not None:
            G.nodes[i]['node_attr'] = f(X[i, :])

    edgelist = generate_edges_gaussian(X, bandwidth, alpha)
    G.add_edges_from(edgelist)

    if f is None:
        for i in range(n):
            G.nodes[i]['node_attr'] = G.degree[i]
    return G

# ==========================================
# 3. Main Experiment Logic
# ==========================================
if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)
    
    ns = np.arange(100, 1001, 100).astype(int) 
    big_n = 3000  
    nexp = 10     
    size_layers = [1, 10, 10, 10, 10, 10, 1]  
    
    # 4 Sparsity scaling schemes
    alpha_labels = ['n^(-1/4)', 'n^(-1/2)', 'log(n)/n', '1/n']
    num_alphas = len(alpha_labels)
    
    model = StretchedGCN(size_layers, variant='inv')
    model.eval() 

    fz = lambda x, y: np.cos(10 * x)
    fsig = lambda x: 1

    # ----------------------------------------------------
    # Step A: Compute continuous limit for each sparsity scheme
    # ----------------------------------------------------
    output_limit = np.zeros((nexp, num_alphas))
    
    for e in range(nexp):
        print(f'=== Computing limit values for Exp {e+1}/{nexp} (big_n={big_n}) ===')
        big_n_alphas = [
            1 / (big_n**(1/4)), 
            1 / (big_n**(1/2)), 
            np.log(big_n) / big_n, 
            1 / big_n
        ]
        
        for aind, alpha_limit in enumerate(big_n_alphas):
            X, _ = surface_uniform(big_n, fz=fz)
            G = random_graph_similarity(X, f=fsig, alpha=alpha_limit, bandwidth=0.15)
            D = nx2tg(G, node_attr='node_attr')
            
            with torch.no_grad():
                output_limit[e, aind] = model(D.x, D.edge_index).item()

    cont_values_per_alpha = output_limit.mean(axis=0)

    # ----------------------------------------------------
    # Step B: Evaluate outputs and density under varying sparsity
    # ----------------------------------------------------
    output = np.zeros((nexp, len(ns), num_alphas))
    densities = np.zeros((nexp, len(ns), num_alphas))
    
    for e in range(nexp):
        for nind, n in enumerate(ns):
            alphas = [
                1 / (n**(1/4)), 
                1 / (n**(1/2)), 
                np.log(n) / n, 
                1 / n
            ]
            
            for aind, alpha in enumerate(alphas):
                print(f'Eval Alpha {aind+1}/{num_alphas}, Node {nind+1}/{len(ns)}, Exp {e+1}/{nexp}')
                X, _ = surface_uniform(n, fz=fz)
                G = random_graph_similarity(X, f=fsig, alpha=alpha, bandwidth=0.15)
                
                max_edges = n * (n - 1) / 2
                actual_edges = G.number_of_edges()
                density = actual_edges / max_edges if max_edges > 0 else 0
                densities[e, nind, aind] = density
                
                D = nx2tg(G, node_attr='node_attr')
                with torch.no_grad():
                    output[e, nind, aind] = model(D.x, D.edge_index).item()

    # ----------------------------------------------------
    # Step C: Compute statistics
    # ----------------------------------------------------
    output_err = np.abs(output - cont_values_per_alpha[np.newaxis, np.newaxis, :])
    
    err_mean = output_err.mean(axis=0)  
    err_std = output_err.std(axis=0)
    den_mean = densities.mean(axis=0)
    den_std = densities.std(axis=0)

    # ----------------------------------------------------
    # Step D: Export combined CSV
    # ----------------------------------------------------
    df_err_mean = pd.DataFrame(err_mean, index=ns, columns=alpha_labels)
    df_err_std = pd.DataFrame(err_std, index=ns, columns=alpha_labels)
    df_den_mean = pd.DataFrame(den_mean, index=ns, columns=alpha_labels)
    df_den_std = pd.DataFrame(den_std, index=ns, columns=alpha_labels)
    
    df_combined = pd.concat(
        [df_err_mean, df_err_std, df_den_mean, df_den_std], 
        axis=1, 
        keys=['Error_Mean', 'Error_Std', 'Density_Mean', 'Density_Std']
    )
    df_combined.index.name = 'Num_Nodes'

    save_path = 'stretched_convergence_results_combined.csv'
    df_combined.to_csv(save_path)
    print(f"\nExperiment complete. Data saved to {save_path}.")