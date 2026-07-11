import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

from LoBCD_GW import DynamicVI_GW, DynamicVI_GW_Simple

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/scGEM')
    parser.add_argument('--graph_mode', type=str, default='dense', choices=['dense', 'knn'])
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--pca', type=int, default=14)
    parser.add_argument('--rescale', action='store_true', default=True)
    parser.add_argument('--rho', type=float, default=0.0005)
    parser.add_argument('--min_rho', type=float, default=1e-5)
    parser.add_argument('--sinkhorn_iters', type=int, default=1)
    parser.add_argument('--max_iter', type=int, default=2200)
    parser.add_argument('--device', type=str, default='cuda')
    return parser.parse_args()

def load_data(base_path):
    if not os.path.exists(os.path.join(base_path, 'Cheow_expression.csv')):
        if os.path.exists('./data/scGEM/Cheow_expression.csv'):
            base_path = './data/scGEM'
        else:
            print(f"Warning: Data not found in {base_path}")
    expr = os.path.join(base_path, 'Cheow_expression.csv')
    meth = os.path.join(base_path, 'Cheow_methylation.csv')
    lab_path = os.path.join(base_path, 'rna_label.txt')
    if not os.path.exists(lab_path):
        lab_path = os.path.join(base_path, 'met_label.txt')
    try:
        labels = np.loadtxt(lab_path, dtype=str)
    except Exception as e:
        print(f"Error loading labels: {e}")
        sys.exit(1)
    n = len(labels)
    X_df = pd.read_csv(expr, header=None)
    Y_df = pd.read_csv(meth, header=None)
    if len(X_df) == n + 1:
        X_df = pd.read_csv(expr, header=0)
        Y_df = pd.read_csv(meth, header=0)
    X, Y = X_df.values, Y_df.values
    if X.shape[0] != n:
        X = X.T
    if Y.shape[0] != n:
        Y = Y.T
    return X, Y, labels

def process_features(X, args):
    if args.rescale:
        X = StandardScaler().fit_transform(X)
    if args.pca > 0 and args.pca < min(X.shape):
        X = PCA(n_components=args.pca).fit_transform(X)
    return X

def get_graph(X, args, device):
    X_proc = process_features(X, args)
    if args.graph_mode == 'dense':
        sim = cosine_similarity(X_proc)
        sim = (sim + 1.0) / 2.0
        np.fill_diagonal(sim, 1.0)
        adj = torch.tensor(sim, dtype=torch.float32)
    else:
        adj_sparse = kneighbors_graph(X_proc, args.k, mode='connectivity', metric='cosine', include_self=True)
        adj_dense = adj_sparse.toarray()
        adj = torch.tensor(0.5 * (adj_dense + adj_dense.T), dtype=torch.float32)
    return adj.to(device)

def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    X_raw, Y_raw, labels = load_data(args.data_path)
    A = get_graph(X_raw, args, device)
    B = get_graph(Y_raw, args, device)
    start_time = time.time()
    P_opt, _ = DynamicVI_GW(
        A, B,
        rho0=args.rho,
        min_rho=args.min_rho,
        max_iter=args.max_iter,
        sinkhorn_iters=args.sinkhorn_iters,
        print_every=50,
        aggressive_early_stop=True
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    run_time = time.time() - start_time
    P_np = P_opt.detach().cpu().numpy()
    pred_d2g = labels[np.argmax(P_np, axis=0)]
    acc_d2g = accuracy_score(labels, pred_d2g) * 100
    pred_g2d = labels[np.argmax(P_np, axis=1)]
    acc_g2d = accuracy_score(labels, pred_g2d) * 100
    print("-" * 40)
    print(f"{'D.->G.':<10} {'G.->D.':<10} {'Time':<10}")
    print(f"{acc_d2g:<10.1f} {acc_g2d:<10.1f} {run_time:<10.4f}")
    print("-" * 40)

if __name__ == "__main__":
    main()
