#!/usr/bin/env python3
"""Reproduction script for SuperMeshNet GCN on Dataset 1 (data_angle).

Reproduces the GCN complementary learning experiment from:
    Semi-Supervised Neural Super-Resolution for Mesh-Based Simulations (ICML 2026)

Settings: GCN, Nh=20, N=200, hidden_dim=30, lr_layers=3, hr_layers=3,
          optimizer=Adam, lr=1e-3, amp=True, ib_n=True (node-level centering)
          data_dir = "data/data_angle/"
          result_dir = "results/GCN_angle/"

Expected RMSE: ~0.0431 (paper reports 0.0431 ± 0.0009)
"""
import sys
sys.path.insert(0, '.')

import torch
import random
import torch.backends.cudnn as cudnn
import numpy as np

# Reproducibility
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(0)

from utils.dataset import GraphDataset_paired, GraphDataset_unpaired
from train.train_GCN import train_GCN_comp, train_GCN_sup
from utils.analysis import RMSE


def main():
    device = "cuda:0"
    data_dir = "data/data_angle/"
    result_dir = "results/GCN_angle/"

    # ------- Data split (deterministic RandomState) -------
    rng = np.random.RandomState(0)
    idx_list_test1 = rng.choice(list(range(1000)), 100, replace=False)
    idx_list_train = [i for i in range(1000) if i not in idx_list_test1]
    idx_list_train1_200 = rng.choice(idx_list_train, 200, replace=False)
    idx_list_train1_20 = rng.choice(idx_list_train1_200, 20, replace=False)
    idx_list_train2_180 = [i for i in idx_list_train1_200 if i not in idx_list_train1_20]

    print(f"Data split: test={len(idx_list_test1)}, "
          f"paired_train={len(idx_list_train1_200)}, "
          f"labeled={len(idx_list_train1_20)}, "
          f"unlabeled={len(idx_list_train2_180)}")

    # ------- Load data -------
    test1 = GraphDataset_paired(idx_list_test1, data_dir, device)

    # Complementary learning (semi-supervised, Nh=20, N=200)
    print("\n=== SuperMeshNet GCN complementary learning (Nh=20, N=200) ===")
    train1_paired = GraphDataset_paired(idx_list_train1_20, data_dir, device)
    train2_unpaired = GraphDataset_unpaired(idx_list_train2_180, data_dir, device)
    train_GCN_comp(device, train1_paired, train2_unpaired, test1, result_dir,
                   ib_n=True, num_exp=1)

    # Fully supervised baseline (Nh=200, N=200)
    print("\n=== Fully supervised baseline (Nh=200, N=200) ===")
    train_full = GraphDataset_paired(idx_list_train1_200, data_dir, device)
    train_GCN_sup(device, train_full, test1, result_dir, ib_n=False, num_exp=1)

    # ------- Evaluate -------
    rmse_comp, std_comp = RMSE(
        result_dir=result_dir, model="GCN", learning="comp",
        N_paired=20, N_total=200, ib="T", exp_list=[0])
    rmse_sup, std_sup = RMSE(
        result_dir=result_dir, model="GCN", learning="sup",
        N_paired=200, N_total=200, ib="F", exp_list=[0])

    print(f"\n{'='*60}")
    print(f"SuperMeshNet GCN comp  (Nh=20,  N=200): RMSE = {rmse_comp:.6f}")
    print(f"Fully supervised       (Nh=200, N=200): RMSE = {rmse_sup:.6f}")
    print(f"Paper GCN comp  (Nh=20, N=200): 0.0431 ± 0.0009")
    print(f"Paper GCN sup   (Nh=200,N=200): 0.0575 ± 0.0035")
    print(f"{'='*60}")

    return rmse_comp, rmse_sup


if __name__ == "__main__":
    main()
