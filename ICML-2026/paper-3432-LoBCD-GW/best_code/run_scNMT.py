import os
import time
import argparse
import numpy as np
import pandas as pd
import torch
import itertools
import contextlib
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.neighbors import kneighbors_graph

from LoBCD_GW2 import DynamicVI_GW



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/scNMT')
    parser.add_argument('--graph_mode', type=str, default='dense', choices=['dense', 'knn'])
    parser.add_argument('--k', type=int, default=15)
    parser.add_argument('--sinkhorn_iters', type=int, default=1)
    parser.add_argument('--max_iter', type=int, default=2000)
    parser.add_argument('--device', type=str, default='cuda')
    return parser.parse_args()


def load_data(base_path):
    files = {
        'rna': ('rna_30.txt', 'rna_stage.txt'),
        'met': ('met_30.txt', 'met_stage.txt'),
        'acc': ('acc_30.txt', 'acc_stage.txt')
    }
    data = {}
    for mod, (d_f, l_f) in files.items():
        d_p = os.path.join(base_path, d_f)
        l_p = os.path.join(base_path, l_f)
        feat = pd.read_csv(d_p, sep=r'\s+', header=None, engine='python').values
        label = pd.read_csv(l_p, sep=r'\s+', header=None, engine='python').values.flatten().astype(str)
        if feat.shape[0] != len(label):
            feat = feat.T
        data[mod] = (feat, label)
    return data


def get_graph(X, pca=0, device='cuda', graph_mode='dense', k=15):
    if pca > 0:
        X = StandardScaler().fit_transform(X)
        if pca < min(X.shape):
            X = PCA(n_components=pca, svd_solver="full").fit_transform(X)
    if graph_mode == 'dense':
        sim = cosine_similarity(X)
        sim = (sim + 1.0) / 2.0
        np.fill_diagonal(sim, 1.0)
        return torch.tensor(sim, dtype=torch.float32, device=device)
    knn = kneighbors_graph(X, n_neighbors=k, mode='distance', metric='cosine', include_self=True)
    dist = knn.toarray()
    sim = 1.0 - np.clip(dist, 0.0, 1.0)
    sim = np.maximum(sim, sim.T)
    np.fill_diagonal(sim, 1.0)
    return torch.tensor(sim, dtype=torch.float32, device=device)


def calc_acc(P, l_src, l_tgt, direction):
    P_np = P.detach().cpu().numpy()
    if direction == 'fwd':
        pred = l_tgt[np.argmax(P_np, axis=1)]
        true = l_src
    else:
        pred = l_src[np.argmax(P_np, axis=0)]
        true = l_tgt
    return accuracy_score(true, pred) * 100


def main():
    args = get_args()
    device = torch.device('cuda' if (args.device == 'cuda' and torch.cuda.is_available()) else 'cpu')
    data = load_data(args.data_path)

    pca_list_dg = [4, 6]
    rho_list_dg = [1e-4]

    pca_list_chg = [2]
    rho_list_chg = [1e-5]

    print("==== Search for D<->G ====")
    print(f"{'PCA':<5} {'Rho':<10} {'D->G':<8} {'G->D':<8} {'AvgPair':<8} {'Time(s)':<8}")
    print("-" * 70)

    best_d2g = (-1, None, None)
    best_g2d = (-1, None, None)
    best_pair_dg = (-1, None, None)

    for current_pca, current_rho in itertools.product(pca_list_dg, rho_list_dg):
        cfg_start = time.time()
        G = get_graph(data['rna'][0], pca=current_pca, device=device, graph_mode=args.graph_mode, k=args.k)
        D = get_graph(data['met'][0], pca=current_pca, device=device, graph_mode=args.graph_mode, k=args.k)
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            P_DG, _ = DynamicVI_GW(
                D, G,
                rho0=current_rho,
                min_rho=current_rho / 10.0,
                sinkhorn_iters=args.sinkhorn_iters,
                max_iter=args.max_iter,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        acc_d2g = calc_acc(P_DG, data['met'][1], data['rna'][1], 'fwd')
        acc_g2d = calc_acc(P_DG, data['met'][1], data['rna'][1], 'bwd')
        avg_pair = 0.5 * (acc_d2g + acc_g2d)
        cfg_time = time.time() - cfg_start
        if acc_d2g > best_d2g[0]:
            best_d2g = (acc_d2g, current_pca, current_rho)
        if acc_g2d > best_g2d[0]:
            best_g2d = (acc_g2d, current_pca, current_rho)
        if avg_pair > best_pair_dg[0]:
            best_pair_dg = (avg_pair, current_pca, current_rho)
        print(f"{current_pca:<5} {current_rho:<10.2e} {acc_d2g:<8.1f} {acc_g2d:<8.1f} {avg_pair:<8.1f} {cfg_time:<8.2f}")

    print("\n==== Search for Ch<->G ====")
    print(f"{'PCA':<5} {'Rho':<10} {'Ch->G':<8} {'G->Ch':<8} {'AvgPair':<8} {'Time(s)':<8}")
    print("-" * 70)

    best_ch2g = (-1, None, None)
    best_g2ch = (-1, None, None)
    best_pair_chg = (-1, None, None)

    for current_pca, current_rho in itertools.product(pca_list_chg, rho_list_chg):
        cfg_start = time.time()
        G = get_graph(data['rna'][0], pca=current_pca, device=device, graph_mode=args.graph_mode, k=args.k)
        Ch = get_graph(data['acc'][0], pca=current_pca, device=device, graph_mode=args.graph_mode, k=args.k)
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            P_ChG, _ = DynamicVI_GW(
                Ch, G,
                rho0=current_rho,
                min_rho=current_rho / 10.0,
                sinkhorn_iters=args.sinkhorn_iters,
                max_iter=args.max_iter,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        acc_ch2g = calc_acc(P_ChG, data['acc'][1], data['rna'][1], 'fwd')
        acc_g2ch = calc_acc(P_ChG, data['acc'][1], data['rna'][1], 'bwd')
        avg_pair = 0.5 * (acc_ch2g + acc_g2ch)
        cfg_time = time.time() - cfg_start
        if acc_ch2g > best_ch2g[0]:
            best_ch2g = (acc_ch2g, current_pca, current_rho)
        if acc_g2ch > best_g2ch[0]:
            best_g2ch = (acc_g2ch, current_pca, current_rho)
        if avg_pair > best_pair_chg[0]:
            best_pair_chg = (avg_pair, current_pca, current_rho)
        print(f"{current_pca:<5} {current_rho:<10.2e} {acc_ch2g:<8.1f} {acc_g2ch:<8.1f} {avg_pair:<8.1f} {cfg_time:<8.2f}")

    print("\n====== Best (D<->G) ======")
    print(f"D->G best: {best_d2g[0]:.2f}%   PCA={best_d2g[1]}, Rho={best_d2g[2]:.3e}")
    print(f"G->D best: {best_g2d[0]:.2f}%   PCA={best_g2d[1]}, Rho={best_g2d[2]:.3e}")
    print(f"D<->G best avg: {best_pair_dg[0]:.2f}%   PCA={best_pair_dg[1]}, Rho={best_pair_dg[2]:.3e}")

    print("\n====== Best (Ch<->G) ======")
    print(f"Ch->G best: {best_ch2g[0]:.2f}%   PCA={best_ch2g[1]}, Rho={best_ch2g[2]:.3e}")
    print(f"G->Ch best: {best_g2ch[0]:.2f}%   PCA={best_g2ch[1]}, Rho={best_g2ch[2]:.3e}")
    print(f"Ch<->G best avg: {best_pair_chg[0]:.2f}%   PCA={best_pair_chg[1]}, Rho={best_pair_chg[2]:.3e}")


if __name__ == "__main__":
    main()
