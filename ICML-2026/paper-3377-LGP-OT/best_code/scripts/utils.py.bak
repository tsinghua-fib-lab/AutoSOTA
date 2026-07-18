'''
Taken from (Zhang et al. 2024): https://github.com/rsinghlab/scNODE

Description:
    Utility functions for benchmarking.

Author:
    Jiaqi Zhang <jiaqi_zhang2@brown.edu>
'''
import numpy as np
import scanpy
import pandas as pd
import natsort
import torch
import torch.distributions as dist
from optim.evaluation import _ot

# --------------------------------
# Load scRNA-seq datasets

def loadZebrafishData(data_dir, split_type):
    cnt_data = pd.read_csv("{}/{}-count_data-hvg.csv".format(data_dir, split_type), header=0, index_col=0)
    meta_data = pd.read_csv("{}/meta_data.csv".format(data_dir), header=0, index_col=0)
    meta_data = meta_data.loc[cnt_data.index,:]
    cell_stage = meta_data["stage.nice"]
    unique_cell_stages = natsort.natsorted(np.unique(cell_stage))
    cell_tp = np.zeros((len(cell_stage), ))
    cell_tp[cell_tp == 0] = np.nan
    for idx, s in enumerate(unique_cell_stages):
        cell_tp[np.where(cell_stage == s)[0]] = idx
    cell_tp += 1
    meta_data["tp"] = cell_tp
    # -----
    cell_set_meta = pd.read_csv("{}/cell_groups_meta.csv".format(data_dir), header=0, index_col=0)
    meta_data = pd.concat([meta_data, cell_set_meta.loc[meta_data.index, :]], axis=1)
    ann_data = scanpy.AnnData(X=cnt_data, obs=meta_data)
    return ann_data


def loadDrosophilaData(data_dir, split_type):
    cnt_data = pd.read_csv("{}/{}-count_data-hvg.csv".format(data_dir, split_type), header=0, index_col=0)
    meta_data = pd.read_csv("{}/subsample_meta_data_with_celltype.csv".format(data_dir), header=0, index_col=0)
    meta_data = meta_data.loc[cnt_data.index,:]
    cell_stage = meta_data["time"]
    unique_cell_stages = natsort.natsorted(np.unique(cell_stage))
    cell_tp = np.zeros((len(cell_stage), ))
    cell_tp[cell_tp == 0] = np.nan
    for idx, s in enumerate(unique_cell_stages):
        cell_tp[np.where(cell_stage == s)[0]] = idx
    cell_tp += 1
    meta_data["tp"] = cell_tp
    ann_data = scanpy.AnnData(X=cnt_data, obs=meta_data)
    return ann_data


def loadWOTData(data_dir, split_type):
    cnt_data = pd.read_csv("{}/{}-norm_data-hvg.csv".format(data_dir, split_type), header=0, index_col=0)
    meta_data = pd.read_csv("{}/{}-meta_data.csv".format(data_dir, split_type), header=0, index_col=0)
    cell_idx = np.where(~np.isnan(meta_data["day"].values))[0] # remove cells with nan labels
    cnt_data = cnt_data.iloc[cell_idx, :]
    meta_data = meta_data.loc[cnt_data.index,:]
    cell_stage = meta_data["day"]
    unique_cell_stages = natsort.natsorted(np.unique(cell_stage))
    cell_tp = np.zeros((len(cell_stage), ))
    cell_tp[cell_tp == 0] = np.nan
    for idx, s in enumerate(unique_cell_stages):
        cell_tp[np.where(cell_stage == s)[0]] = idx
    cell_tp += 1
    meta_data["tp"] = cell_tp
    ann_data = scanpy.AnnData(X=cnt_data, obs=meta_data)
    return ann_data

# --------------------------------
# Dataset directories
zebrafish_data_dir = "../data/zebrafish_embryonic/new_processed"
wot_data_dir = "../data/Schiebinger2019/reduced_processed"
drosophila_data_dir = "../data/drosophila_embryonic/processed"


def loadSCData(data_name, split_type, data_dir="None"):
    '''
    Main function to load scRNA-seq dataset and pre-process it.
    '''
    print("[ Data={} | Split={} ] Loading data...".format(data_name, split_type))
    if data_name == "zebrafish":
        if data_dir == "None":
            data_dir = zebrafish_data_dir
        ann_data = loadZebrafishData(data_dir, split_type)
        ann_data.X = ann_data.X.astype(float)
        processed_data = preprocess(ann_data.copy())
        cell_types =  processed_data.obs["ZF6S-Cluster"].apply(lambda x: "NAN" if pd.isna(x) else x).values
    elif data_name == "drosophila":
        if data_dir == "None":
            data_dir = drosophila_data_dir
        ann_data = loadDrosophilaData(data_dir, split_type)
        print("Pre-processing...")
        ann_data.X = ann_data.X.astype(float)
        processed_data = preprocess(ann_data.copy())
        cell_types = processed_data.obs.manual_annot.values
    elif data_name == "wot":
        if data_dir == "None":
            data_dir = wot_data_dir
        ann_data = loadWOTData(data_dir, split_type)
        processed_data = ann_data.copy()
        cell_types = None
    else:
        raise ValueError("Unknown data name.")
    cell_tps = ann_data.obs["tp"]
    n_tps = len(np.unique(cell_tps))
    n_genes = ann_data.shape[1]
    return processed_data, cell_tps, cell_types, n_genes, n_tps


def tpSplitInd(data_name, split_type):
    '''
    Get the training/testing timepoint split for each dataset.
    '''
    if data_name == "zebrafish":
        if split_type == "two_forecasting": # medium
            train_tps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            test_tps = [10, 11]
        elif split_type == "three_interpolation": # easy
            train_tps = [0, 1, 2, 3, 5, 7, 9, 10, 11]
            test_tps = [4, 6, 8]
        elif split_type == "remove_recovery": # hard
            train_tps = [0, 1, 3, 5, 7, 9]
            test_tps = [2, 4, 6, 8, 10, 11]
        elif split_type == "all":
            train_tps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
            test_tps = []
        else:
            raise ValueError("Unknown split type {}!".format(split_type))
    elif data_name == "drosophila":
        if split_type == "three_forecasting": # medium
            train_tps = [0, 1, 2, 3, 4, 5, 6, 7]
            test_tps = [8, 9, 10]
        elif split_type == "three_interpolation": # easy
            train_tps = [0, 1, 2, 3, 5, 7, 9, 10]
            test_tps = [4, 6, 8]
        elif split_type == "remove_recovery": # hard
            train_tps = [0, 1, 3, 5, 7]
            test_tps = [2, 4, 6, 8, 9, 10]
        else:
            raise ValueError("Unknown split type {}!".format(split_type))
    elif data_name == "wot":
        unique_days = np.arange(19)
        if split_type == "three_forecasting": # medium
            train_tps = unique_days[:16].tolist()
            test_tps = unique_days[16:].tolist()
        elif split_type == "three_interpolation": # easy
            train_tps = unique_days.tolist()
            test_tps = [train_tps[5], train_tps[10], train_tps[15]]
            train_tps.remove(unique_days[5])
            train_tps.remove(unique_days[10])
            train_tps.remove(unique_days[15])
        elif split_type == "remove_recovery": # hard
            train_tps = unique_days.tolist()
            test_idx = [5, 7, 9, 11, 15, 16, 17, 18]
            test_tps = [train_tps[t] for t in test_idx]
            for t in test_idx:
                train_tps.remove(unique_days[t])
        else:
            raise ValueError("Unknown split type {}!".format(split_type))
    else:
        raise ValueError("Unknown data name.")
    return train_tps, test_tps


def splitBySpec(traj_data, train_tps, test_tps):
    '''
    Split timepoints into training and testing sets.
    '''
    train_data = [traj_data[t] for t in train_tps]
    test_data = [traj_data[t] for t in test_tps]
    return train_data, test_data
# --------------------------------

def modelParams(data_name, split_type):
    latent_dim = 32
    if data_name == "zebrafish":
        if split_type == "three_interpolation": # easy
            dec_latent_list = [50]
            M = 5
        elif split_type == "two_forecasting": # medium
            dec_latent_list = [50]
            M = 5
        elif split_type == "remove_recovery": # hard
            dec_latent_list = [50]
            M = 5
        elif split_type == "all":
            dec_latent_list = [50]
            M = 5
        else:
            raise ValueError("Unknown task name {}!".format(split_type))
    elif data_name == "drosophila":
        if split_type == "three_interpolation": # easy
            dec_latent_list = [50, 50]
            M = 6
        elif split_type == "three_forecasting": # medium
            dec_latent_list = [50]
            M = 6
        elif split_type == "remove_recovery": # hard
            dec_latent_list = [50]
            M = 6
        else:
            raise ValueError("Unknown task name {}!".format(split_type))
    elif data_name == "wot":
        if split_type == "three_interpolation": # easy
            dec_latent_list = [50, 50]
            M = 10
        elif split_type == "three_forecasting": # medium
            dec_latent_list = [50]
            M = 10
        elif split_type == "remove_recovery": # hard
            dec_latent_list = [50, 50]
            M = 6
        else:
            raise ValueError("Unknown task name {}!".format(split_type))
    else:
        raise ValueError("Unknown data name {}!".format(data_name))
    return latent_dim, dec_latent_list, M

# --------------------------------

def traj2Ann(traj_data):
    # traj_data: #trajs, #tps, # features
    traj_data_list = [traj_data[:, t, :] for t in range(traj_data.shape[1])]
    time_step = np.concatenate([np.repeat(t, traj_data.shape[0]) for t in range(traj_data.shape[1])])
    ann_data = scanpy.AnnData(X=np.concatenate(traj_data_list, axis=0))
    ann_data.obs["time_point"] = time_step
    return ann_data


def ann2traj(ann_data):
    time_idx = [np.where(ann_data.obs.time_point == t)[0] for t in natsort.natsorted(ann_data.obs.time_point.unique())]
    traj_data_list = [ann_data.X[idx, :] for idx in time_idx]
    traj_data = np.asarray(traj_data_list)
    traj_data = np.moveaxis(traj_data, [0, 1, 2], [1, 0, 2])
    return traj_data

# ---------------------------------

def preprocess(ann_data):
    # adopt recipe_zheng17 w/o HVG selection
    # omit scaling part to avoid information leakage
    scanpy.pp.normalize_per_cell(  # normalize with total UMI count per cell
        ann_data, key_n_counts='n_counts_all', counts_per_cell_after=1e4
    )
    scanpy.pp.log1p(ann_data)  # log transform: adata.X = log(adata.X + 1)
    return ann_data


def postprocess(data):
    # data: cell x gene matrix
    if isinstance(data, np.ndarray):
        norm_data = (data / np.sum(data, axis=1)[:, np.newaxis]) * 1e4
        log_data = np.log(norm_data + 1)
    else:
        norm_data = (data / torch.sum(data, dim=1).unsqueeze(dim=1)) * 1e4
        log_data = torch.log(norm_data + 1)
    return log_data

# ---------------------------------

def sampleOT(true_data, pred_data, sample_n, sample_T):
    ot_list = []
    for _ in range(sample_T):
        true_rand_idx = np.random.choice(np.arange(true_data.shape[0]), sample_n, replace=False)
        pred_rand_idx = np.random.choice(np.arange(pred_data.shape[0]), sample_n, replace=False)
        ot_list.append(_ot(true_data[true_rand_idx,:], pred_data[pred_rand_idx,:]))
    return np.mean(ot_list)


def sampleGaussian(mean, std):
    '''
    Sampling with the re-parametric trick.
    '''
    d = dist.normal.Normal(torch.Tensor([0.]), torch.Tensor([1.]))
    r = d.sample(mean.size()).squeeze(-1)
    x = r * std.float() + mean.float()
    return x