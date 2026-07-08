import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import pickle
import os, sys
sys.path.append(os.path.abspath(os.getcwd()))

from utils.graph_utils import node_flags, graphs_to_tensor, mask_x
from utils.node_features import EigenFeatures

# -------- Create initial node features --------
def init_features(config_data, adjs):
    flags = node_flags(adjs)
    feature = []
    feat_dim = []
    for feat_type in config_data.feat.type:
        if feat_type=='deg':
            deg = adjs.sum(dim=-1).to(torch.long)
            feat = F.one_hot(deg, num_classes=config_data.max_feat_num).to(torch.float32)
        elif 'eig' in feat_type:
            idx = int(feat_type.split('eig')[-1])
            eigvec = EigenFeatures(idx)(adjs, flags)
            feat = eigvec[...,-1:] * config_data.feat.scale
        else:
            raise NotImplementedError(f'Feature: {feat_type} not implemented.')
        feature.append(feat)
        feat_dim.append(feat.shape[-1])
    feature = torch.cat(feature, dim=-1)

    return mask_x(feature, flags), feat_dim 


def feat_diff(x, adjs, flags, feat_dict):
    feat_diff = []
    sdim = 0
    indices = []
    for feat_type in feat_dict.type:
        if 'eig' in feat_type:
            indices.append(int(feat_type.split('eig')[-1]))
    if len(indices)>0:
        try:
            eigvec = EigenFeatures(max(indices))(adjs, flags)
        except:
            return [-1]*len(feat_dict.type)

    for feat_type, feat_dim in zip(feat_dict.type, feat_dict.dim):
        x_ = x[:,:,sdim:sdim+feat_dim]
        if 'eig' in feat_type:
            idx = int(feat_type.split('eig')[-1])
            x_feat = x_ / feat_dict.scale
            x_pm = (x_feat.squeeze(-1)[:,0] / x_feat.squeeze(-1)[:,0].abs())
            eig_pm = (eigvec[...,idx-1][:,0] / eigvec[...,idx-1][:,0].abs())
            eig_pm[torch.isnan(eig_pm)]=0
            pm = x_pm * eig_pm
            fdiff = (x_feat.squeeze(-1) - eigvec[...,idx-1] * pm[:,None]).abs().square().sum(-1) / flags.sum(-1)
            feat_diff.append(round(fdiff.mean().item(),4))
        else:
            if feat_type=='deg':
                feat = adjs.sum(dim=-1).to(torch.long)
            else:
                raise NotImplementedError(f'Feature: {feat_type} not implemented.')
            x_feat = torch.argmax(x_, dim=-1)
            fdiff = (x_feat - feat).abs().sum(-1) / flags.sum(-1)
            feat_diff.append(round(fdiff.mean().item(),2))
        sdim += feat_dim
    return feat_diff


def graphs_to_dataloader(config, graph_list, return_feat_dim=False):
    adjs_tensor = graphs_to_tensor(graph_list, config.data.max_node_num) 
    x_tensor, feat_dim = init_features(config.data, adjs_tensor) 

    dataset = TensorDataset(x_tensor, adjs_tensor)
    dataloader = DataLoader(dataset, batch_size=config.data.batch_size, shuffle=True)
    if return_feat_dim:
        return dataloader, feat_dim
    return dataloader


def dataloader(config, get_graph_list=False):
    with open(f'data/{config.data.data}.pkl', 'rb') as f:
        train_graphs, val_graphs, test_graphs = pickle.load(f)
    print(f'Dataset sizes: train {len(train_graphs)}, val {len(val_graphs)}, test {len(test_graphs)}')
    if get_graph_list:
        return train_graphs, val_graphs, test_graphs
    train_loader, feat_dim = graphs_to_dataloader(config, train_graphs, True)
    val_loader = graphs_to_dataloader(config, val_graphs)
    test_loader = graphs_to_dataloader(config, test_graphs)
    return train_loader, val_loader, test_loader, feat_dim
