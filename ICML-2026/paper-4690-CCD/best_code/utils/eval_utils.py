import torch
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
import os
import torch.nn as nn
import fire

def grab_chunk_feats(feat_files,start_ind,chunk_size,to_half=False,feat_dim=512):
    end_ind = min(start_ind+chunk_size,len(feat_files))
    chunk_files = feat_files[start_ind:end_ind]
    
    all_feats_torch = torch.zeros(0,feat_dim,dtype=torch.half).cuda()
    for chunk in chunk_files:
        cur_feat = np.load(chunk)
        if to_half:
            cur_feat = torch.from_numpy(cur_feat).to(torch.half).cuda()
        else:
            cur_feat = torch.from_numpy(cur_feat).to(torch.float32).cuda()
        all_feats_torch = torch.cat([all_feats_torch,cur_feat],dim=0)
    return all_feats_torch

class Identity(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self,x):
        return x
    
    
def abs_ind_to_feat_file(abs_ind, cum_sz, feat_files):
    inds = np.argwhere(abs_ind - cum_sz >= 0)
    last_ind = inds[-1].item()
    ind_offset = cum_sz[last_ind]
    local_ind = abs_ind - ind_offset
    return feat_files[last_ind],last_ind,local_ind

def get_cum_sz(feat_files):
    cum_sz = [0]
    for feat in feat_files:
        cum_sz += [cum_sz[-1] + np.load(feat,mmap_mode='r').shape[0]]
    cum_sz = np.array(cum_sz).astype('int')
    return cum_sz

def get_emb(ff,local_ind):
    return np.load(ff,mmap_mode='r')[local_ind,:]

def get_raw_feature_mses(q_feats,feat_files,nns):
    # using the query features, computes stats of the raw
    # feature MSE, retrieved using indices from nns
    assert q_feats.shape[0] == nns.shape[0]
    mse_mat = np.zeros(nns.shape)
    cum_sz = get_cum_sz(feat_files)
    
    for k in range(nns.shape[0]):
        q_feat = q_feats[k,:]
        
        for j in range(nns.shape[1]):
            nn_ind = nns[k,j]
            ff,ffi,li = abs_ind_to_feat_file(nn_ind, cum_sz, feat_files)
            nn_feat = get_emb(ff,li).astype('float32')
            mse = ((q_feat - nn_feat)**2).sum()
            mse_mat[k,j] = mse
           
    return mse_mat
 

def compute_nn_rec_vs_gt_at5(nn_inds,gt_nn_file):

    gt_nn_inds = np.load(gt_nn_file) 
    n_correct = 0.0
    for i in range(min(gt_nn_inds.shape[0],nn_inds.shape[0])):
        gt_inds_i = list(nn_inds[i,:])
        nn_inds_i = list(gt_nn_inds[i,:])

        n_correct += len(set([k for k in nn_inds_i if k in gt_inds_i]))

    acc = n_correct / (float(nn_inds.shape[0]*nn_inds.shape[1]))
    return acc


@torch.no_grad()
def compute_net_feats_chunked(net,feats,cs):
    
    feats_out = []
    for k in range(0,feats.shape[0],cs):
        end_ind = min(feats.shape[0],k+cs)
        feats_chunk = torch.from_numpy(feats[k:end_ind,:]).float().cuda()
        feats_out += [net(feats_chunk).cpu().numpy()]
        
    feats_out = np.concatenate(feats_out,axis=0)
    return feats_out
        
        
    
    