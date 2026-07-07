from copy import deepcopy
import torch
from torch_geometric.transforms import SVDFeatureReduction
from torch_geometric.datasets import Planetoid, WebKB, Amazon, WikipediaNetwork
from torch_geometric.data import Data
from torch_geometric.utils import degree, add_self_loops
from fastargs.decorators import param
import math
from torch_geometric.datasets import WebKB, Planetoid, Amazon, Coauthor, WikipediaNetwork, Reddit, \
    Flickr, PPI, Yelp, Twitch, Actor, KarateClub, FacebookPagePage, LastFMAsia, HeterophilousGraphDataset,OGB_MAG
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from sklearn.decomposition import PCA
import torch.nn.functional as F

def pca_compression(data, out_dim):
    pca = PCA(n_components=out_dim)
    #data.x = F.normalize(torch.FloatTensor(pca.fit_transform(data.x)), dim=1, p=2)
    data.x = torch.FloatTensor(pca.fit_transform(data.x))

    return data

def x_padding(data, out_dim):
    
    assert data.x.size(-1) <= out_dim
    
    incremental_dimension = out_dim - data.x.size(-1)
    zero_features = torch.zeros((data.x.size(0), incremental_dimension), dtype=data.x.dtype, device=data.x.device)
    data.x = torch.cat([data.x, zero_features], dim=-1)

    return data


def x_svd(data, out_dim):
    
    assert data.x.size(-1) >= out_dim
    #return pca_compression(data, out_dim)
    reduction = SVDFeatureReduction(out_dim)
    return reduction(data)

WebKB_datasets = ['texas', 'cornell', 'wisconsin']
Planetoid_datasets = ['Cora', 'Citeseer', 'Pubmed']
Amazon_datasets = ['Photo', 'Computers']
Coauthor_datasets = ['CS', 'Physics']
WikipediaNetwork_datasets = ['chameleon', 'squirrel']
Reddit_datasets = ['Reddit']
OGB_datasets = ['ogbn-arxiv', 'ogbn-products', 'ogbn-proteins', 'ogbn-papers100M', 'ogbn-mag']

Flickr_datasets = ['Flickr']
PPI_datasets = ['PPI']
Yelp_datasets = ['Yelp']
Twitch_datasets = ['DE', 'EN', 'ES', 'FR', 'PT', 'RU']
Actor_datasets = ['Actor']
KarateClub_datasets = ['KarateClub']
FacebookPagePage_datasets = ['FacebookPagePage']
LastFMAsia_datasets = ['LastFMAsia']
heterophilous_datasets = ['roman_empire']

large_dataset = ['Reddit']


cache_dir = "/datasets"
def iterate_datasets(data_names, cache_dir= "/datasets"):
    
    if isinstance(data_names, str):
        data_names = [data_names]
    
    for data_name in data_names:
        if data_name in ['Cora', 'Citeseer', 'Pubmed']:
            data = Planetoid(root=cache_dir, name=data_name.capitalize())._data
        elif data_name in ['wisconsin', 'texas', 'cornell']:
            data = WebKB(root=cache_dir, name=data_name.capitalize())._data
        elif data_name in ['Computers', 'Photo']:
            data = Amazon(root=cache_dir, name=data_name.capitalize())._data
        elif data_name in ['chameleon', 'squirrel']:
            #preProcDs = WikipediaNetwork(root=cache_dir, name=data_name.capitalize(), geom_gcn_preprocess=False)
            data = WikipediaNetwork(root=cache_dir, name=data_name.capitalize(), geom_gcn_preprocess=True)._data
            #data.edge_index = preProcDs[0].edge_index
        elif data_name in ['Reddit']:
            data = Reddit(root=f'{cache_dir}/Reddit')._data
        elif data_name in FacebookPagePage_datasets:
            data = FacebookPagePage(root=f'{cache_dir}/Facebook')._data
        elif data_name in OGB_datasets:
            data = PygNodePropPredDataset(name=data_name, root=f'{cache_dir}/arxiv/')._data
        else:
            raise ValueError(f'Unknown dataset: {data_name}')
        
        assert isinstance(data, (Data, dict)), f'Unknown data type: {type(data)}'

        yield data if isinstance(data, Data) else Data(**data)

def iterate_dataset_feature_tokens(data_names, cache_dir= "/datasets"):
    
    if isinstance(data_names, str):
        data_names = [data_names]
    
    for data_name in data_names:
        if data_name in ['Cora', 'Citeseer', 'Pubmed']:
            data = Planetoid(root=cache_dir, name=data_name.capitalize())._data
        elif data_name in ['wisconsin', 'texas', 'cornell']:
            data = WebKB(root=cache_dir, name=data_name.capitalize())._data
        elif data_name in ['Computers', 'Photo']:
            data = Amazon(root=cache_dir, name=data_name.capitalize())._data
        elif data_name in ['chameleon', 'Squirrel']:
            preProcDs = WikipediaNetwork(root=cache_dir, name=data_name.capitalize(), geom_gcn_preprocess=False)
            data = WikipediaNetwork(root=cache_dir, name=data_name.capitalize(), geom_gcn_preprocess=True)._data
            data.edge_index = preProcDs[0].edge_index
        elif data_name in FacebookPagePage_datasets:
            data = FacebookPagePage(root=f'{cache_dir}/Facebook')
        else:
            raise ValueError(f'Unknown dataset: {data_name}')
        
        assert isinstance(data, (Data, dict)), f'Unknown data type: {type(data)}'

        yield data if isinstance(data, Data) else Data(**data)

# including projection operation, SVD
def preprocess(data, node_feature_dim=50):

    if hasattr(data, 'train_mask'):
        del data.train_mask
    if hasattr(data, 'val_mask'):
        del data.val_mask
    if hasattr(data, 'test_mask'):
        del data.test_mask

    if node_feature_dim <= 0:
        edge_index_with_loops = add_self_loops(data.edge_index, num_nodes=data.num_nodes)[0]
        data.x = degree(edge_index_with_loops[1]).reshape((-1,1))
    
    else:
        # import pdb
        # pdb.set_trace()  
        # print(data)
        # print(data.x)
        # print(node_feature_dim)
        
        if data.x.size(-1) > node_feature_dim:
            #data = x_svd(data, node_feature_dim)

            data = pca_compression(data, node_feature_dim)
        elif data.x.size(-1) < node_feature_dim:
            data = x_padding(data, node_feature_dim)
        else:
            pass
    
    return data

# For prompting
def loss_contrastive_learning(x1, x2):
    # T = 0.1
    T = 0.5
    batch_size, _ = x1.size()
    x1_abs = x1.norm(dim=1)
    x2_abs = x2.norm(dim=1)
    
    sim_matrix = torch.einsum('ik,jk->ij', x1+1e-7, x2+1e-7) / torch.einsum('i,j->ij', x1_abs+1e-7, x2_abs+1e-7)
    
    if(True in sim_matrix.isnan()):
        print('Emerging nan value')
    
    sim_matrix = torch.exp(sim_matrix / T)
    
    if(True in sim_matrix.isnan()):
        print('Emerging nan value')    
    
    pos_sim = sim_matrix[range(batch_size), range(batch_size)]

    if(True in pos_sim.isnan()):
        print('Emerging nan value')

    loss = pos_sim / ((sim_matrix.sum(dim=1) - pos_sim) + 1e-4)
    loss = - torch.log(loss).mean()
    if math.isnan(loss.item()):
        print("The value is NaN.")

    return loss

# used in pre_train.py
@param('general.reconstruct')
def gen_ran_output(data, simgrace, reconstruct):
    vice_model = deepcopy(simgrace)

    for (vice_name, vice_model_param), (name, param) in zip(vice_model.named_parameters(), simgrace.named_parameters()):
        if vice_name.split('.')[0] == 'projection_head':
            vice_model_param.data = param.data
        else:
            vice_model_param.data = param.data + 0.1 * torch.normal(0, torch.ones_like(
                param.data) * param.data.std())
    if(reconstruct==0.0):
    
        zj = vice_model.forward_cl(data)

        return zj
    
    else:
    
        zj, hj = vice_model.forward_cl(data)

        return zj, hj