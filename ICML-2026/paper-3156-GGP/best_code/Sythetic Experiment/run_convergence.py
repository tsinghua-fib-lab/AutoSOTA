import numpy as np
import pandas as pd
import torch
import torch.nn.functional as Fc

from models import MyGCN
from graph_utils import surface_uniform, random_graph_similarity, nx2tg

if __name__ == '__main__':
    # ==========================================
    # Initialization
    # ==========================================
    np.random.seed(0)
    torch.manual_seed(0)
    
    # Graph sizes: {100, 200, ..., 1000}
    ns = np.arange(100, 1001, 100).astype(int) 
    big_n = 3000  
    
    # Number of experimental trials
    nexp = 10     
    size_layers = [1, 10, 10, 10, 10, 10, 1]
    
    # Instantiate the standard c-GCN network
    GN = MyGCN(size_layers, variant='inv', nonlin=Fc.relu, normalization='laplacian')
    GN.eval() 

    fz = lambda x, y: np.cos(10 * x)
    fsig = lambda x: 1

    # ==========================================
    # Step 1: Compute the continuous limit value
    # ==========================================
    output_limit = np.zeros(nexp)
    for e in range(nexp):
        print(f'Computing continuous limit value {e+1}/{nexp}')
        X, _ = surface_uniform(big_n, fz=fz)
        # c-GCN relies on the dense graph (alpha=1) as the continuous graphon limit
        G = random_graph_similarity(X, f=fsig, alpha=1, bandwidth=0.15)
        D = nx2tg(G, node_attr='node_attr')
        
        with torch.no_grad():
            output_limit[e] = GN(D.x, D.edge_index).item()

    cont_value = output_limit.mean()

    # ==========================================
    # Step 2: Evaluate outputs and density under varying sparsity
    # ==========================================
    # 4 Sparsity scaling schemes
    alpha_labels = ['n^(-1/4)', 'n^(-1/2)', 'log(n)/n', '1/n']
    num_alphas = len(alpha_labels)
    
    # Data storage arrays (Trials, Graph Sizes, Sparsity Schemes)
    output = np.zeros((nexp, len(ns), num_alphas))
    densities = np.zeros((nexp, len(ns), num_alphas))
    
    for e in range(nexp):
        for nind, n in enumerate(ns):
            # Dynamic sparsity coefficients for current n
            alphas = [
                1 / (n**(1/4)), 
                1 / (n**(1/2)), 
                np.log(n) / n, 
                1 / n
            ]
            
            for aind, alpha in enumerate(alphas):
                print(f'Eval Sparsity {aind+1}/{num_alphas}, Node {nind+1}/{len(ns)}, Exp {e+1}/{nexp}')
                X, _ = surface_uniform(n, fz=fz)
                G = random_graph_similarity(X, f=fsig, alpha=alpha, bandwidth=0.15)
                
                # Record empirical edge density
                max_edges = n * (n - 1) / 2
                actual_edges = G.number_of_edges()
                density = actual_edges / max_edges if max_edges > 0 else 0
                densities[e, nind, aind] = density
                
                D = nx2tg(G, node_attr='node_attr')
                with torch.no_grad():
                    output[e, nind, aind] = GN(D.x, D.edge_index).item()

    # ==========================================
    # Step 3: Compute statistics
    # ==========================================
    # Absolute error compared to the continuous limit
    output_err = np.abs(output - cont_value)
    
    err_mean = output_err.mean(axis=0)  
    err_std = output_err.std(axis=0)
    den_mean = densities.mean(axis=0)
    den_std = densities.std(axis=0)

    # ==========================================
    # Step 4: Export combined CSV
    # ==========================================
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

    save_path = 'convergence_results_combined.csv'
    df_combined.to_csv(save_path)
    print(f"\nExperiment complete. Data saved to {save_path}.")