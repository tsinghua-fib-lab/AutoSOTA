import argparse
import scanpy as sc
import os
import numpy as np
import anndata as ad
from scipy import stats
import torch
import scib
import pandas as pd
import json


def get_highly_variable_genes(adata, data_dimensions):
    log_adata = sc.pp.log1p(adata, copy=True)
    highly_var_genes = sc.pp.highly_variable_genes(
        log_adata, n_top_genes=1000, inplace=False)
    adata_filtered = adata[:, highly_var_genes['highly_variable']]
    filtered_data = adata_filtered.X.toarray()
    n_cells, n_genes = filtered_data.shape
    padded_data = np.zeros((n_cells, data_dimensions),dtype=np.float32)
    padded_data[:, :n_genes] = filtered_data
    return padded_data


def normalize_adata(adata):
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)


def get_scc_pcc(real_data, gen_data):
    scc = stats.spearmanr(real_data.mean(
        axis=0), gen_data.mean(axis=0)).correlation
    pcc = np.corrcoef(real_data.mean(
        axis=0), gen_data.mean(axis=0))[0][1]
    return scc, pcc


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)

    total0 = total.unsqueeze(0).expand(
        int(total.size(0)), int(total.size(0)), int(total.size(1)))

    total1 = total.unsqueeze(1).expand(
        int(total.size(0)), int(total.size(0)), int(total.size(1)))

    L2_distance = ((total0-total1)**2).sum(2)

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)

    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]

    kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                  for bandwidth_temp in bandwidth_list]

    return sum(kernel_val)


def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)

    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY - YX)
    return loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze Sparsity')
    parser.add_argument('--dataset_path', type=str,
                        default=None, help='Path to real data')
    parser.add_argument('--generated_path', type=str,
                        default=None, help='Path to generated data')
    parser.add_argument('--data_dimensions', type=int,
                        default=1000, help='Number of genes to filter')
    args = parser.parse_args()

    real_adata = sc.read_h5ad(args.dataset_path)
    real_adata = get_highly_variable_genes(real_adata, args.data_dimensions)
    dataset_sparsity = (np.size(real_adata) -
                        np.count_nonzero(real_adata))/np.size(real_adata)
    real_adata = ad.AnnData(real_adata)
    scores = {"Dataset Sparsity" : dataset_sparsity}
    print(f"Dataset Sparsity: {dataset_sparsity}")

    number_gen_cells = 10000
    files = [x for x in os.scandir(path=args.generated_path) if x.is_file(
    ) and x.name.split(".")[-1] == "h5ad"]
    all_generated_data = []
    for file in files:
        adata = sc.read_h5ad(os.path.join(args.generated_path, file))
        all_generated_data.append(adata.X)
    all_generated_data = np.concatenate(all_generated_data)
    all_generated_data = all_generated_data[:number_gen_cells]
    np.clip(all_generated_data, out=all_generated_data, a_min=0, a_max=1)
    sparsity = (np.size(all_generated_data) -
                np.count_nonzero(all_generated_data))/np.size(all_generated_data)
    scores["Sparsity"]=sparsity
    print(f"Sparsity: {sparsity}")
    generated_adata = ad.AnnData(all_generated_data)


    normalize_adata(real_adata)
    normalize_adata(generated_adata)

    real_data = real_adata.X
    generated_data = generated_adata.X

    scc, pcc = get_scc_pcc(real_data, generated_data)
    scores["SCC"] = scc
    scores["PCC"] = pcc
    print(f"SCC, PCC: {scc} {pcc}")

    adata = np.concatenate((real_data, generated_data), axis=0)
    adata = ad.AnnData(adata, dtype=np.float32)
    adata.obs_names = [f"true_Cell" for i in range(
        real_data.shape[0])]+[f"gen_Cell" for i in range(generated_data.shape[0])]
    sc.tl.pca(adata, svd_solver='arpack')
    # can not be set too large, the kernel might fail
    real = adata[adata.obs_names == 'true_Cell'].obsm['X_pca'][::2][:5000]
    gen = adata[adata.obs_names == 'gen_Cell'].obsm['X_pca'][::2][:5000]
    X = torch.Tensor(real)
    Y = torch.Tensor(gen)

    mmd = mmd_rbf(X, Y).item()
    scores["MMD"] = mmd
    print("MMD: "+str(mmd))

    adata.obs['batch'] = pd.Categorical([f"true_Cell" for i in range(
        real_data.shape[0])]+[f"gen_Cell" for i in range(generated_data.shape[0])])
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=20)
    lisi = scib.me.ilisi_graph(adata, batch_key="batch", type_="knn")
    scores["LISI"] = lisi
    print("LISI: "+str(lisi))

    save_path = os.path.join(args.generated_path, "metrics.json")
    with open(save_path, "w") as outfile:
        json.dump(scores, outfile)
