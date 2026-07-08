import torch
import torch.nn as nn
from torch.nn.modules.linear import Linear
from torch.nn import functional as F
from utils.graph_utils import mask_x, mask_adjs, pow_tensor
from models.layers import XEyTransformerLayer

### code adpated from https://github.com/cvignac/DiGress/blob/main/dgd/models/transformer_model.py
class GraphTransformer(nn.Module):
    """
    n_layers : int -- number of layers
    dims : dict -- contains dimensions for each feature type
    """
    def __init__(self, n_layers: int, input_dims: dict, hidden_mlp_dims: dict, hidden_dims: dict,
                 output_dims: dict, feat_dict: list):
        super().__init__()
        self.n_layers = n_layers
        self.out_dim_X = output_dims['X']
        self.out_dim_E = output_dims['E']
        self.out_dim_y = output_dims['y']

        self.adj_in_dim = input_dims['E']
        self.feat_dict = feat_dict

        act_fn_in = nn.ReLU()
        act_fn_out = nn.ReLU()

        self.mlp_in_X = nn.Sequential(nn.Linear(input_dims['X'], hidden_mlp_dims['X']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['X'], hidden_dims['dx']), act_fn_in)

        self.mlp_in_E = nn.Sequential(nn.Linear(input_dims['E'], hidden_mlp_dims['E']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['E'], hidden_dims['de']), act_fn_in)

        self.mlp_in_y = nn.Sequential(nn.Linear(input_dims['y'], hidden_mlp_dims['y']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['y'], hidden_dims['dy']), act_fn_in)

        self.tf_layers = nn.ModuleList([XEyTransformerLayer(dx=hidden_dims['dx'],
                                                            de=hidden_dims['de'],
                                                            dy=hidden_dims['dy'],
                                                            n_head=hidden_dims['n_head'],
                                                            dim_ffX=hidden_dims['dim_ffX'],
                                                            dim_ffE=hidden_dims['dim_ffE'])
                                        for i in range(n_layers)])

        self.mlp_out_X = nn.Sequential(nn.Linear(hidden_dims['dx'], hidden_mlp_dims['X']), act_fn_out,
                                       nn.Linear(hidden_mlp_dims['X'], output_dims['X']))

        self.mlp_out_E = nn.Sequential(nn.Linear(hidden_dims['de'], hidden_mlp_dims['E']), act_fn_out,
                                       nn.Linear(hidden_mlp_dims['E'], output_dims['E']))

        if self.out_dim_y>0:
            self.mlp_out_y = nn.Sequential(nn.Linear(hidden_dims['dy'], hidden_mlp_dims['y']), act_fn_out,
                                            nn.Linear(hidden_mlp_dims['y'], output_dims['y']))

    def forward(self, X, E, y, node_mask):
        bs, n = X.shape[0], X.shape[1]

        # higher-order adj
        E = pow_tensor(E, self.adj_in_dim).permute(0,2,3,1)

        diag_mask = torch.eye(n)
        diag_mask = ~diag_mask.type_as(E).bool()
        diag_mask = diag_mask.unsqueeze(0).unsqueeze(-1).expand(bs, -1, -1, -1)

        X_to_out = X[..., :self.out_dim_X]
        E_to_out = E[..., :self.out_dim_E]

        new_E = self.mlp_in_E(E)
        new_E = (new_E + new_E.transpose(1, 2)) / 2

        X = mask_x(self.mlp_in_X(X), node_mask)
        E = mask_adjs(new_E.permute(0,3,1,2), node_mask).permute(0,2,3,1) # bs x n x n x c
        y = self.mlp_in_y(y)

        for layer in self.tf_layers:
            X, E, y = layer(X, E, y, node_mask)

        X = self.mlp_out_X(X)
        E = self.mlp_out_E(E)

        X = (X + X_to_out)
        E = (E + E_to_out) * diag_mask

        if self.out_dim_y>0:
            y_to_out = y[..., :self.out_dim_y]
            y = self.mlp_out_y(y)
            y = y + y_to_out

        E = 1/2 * (E + torch.transpose(E, 1, 2))

        sdim = 0
        X_list = []
        for feat_type, feat_dim in zip(self.feat_dict.type, self.feat_dict.dim):
            if 'eig' in feat_type:
                X_ = X[:,:,sdim:sdim+feat_dim]
                if self.feat_dict.norm:
                    norm = X_.square().sum(-2).sqrt() / self.feat_dict.scale
                    X_ = X_ / norm[:, None, :]
            else:
                X_ = F.softmax(X[:,:,sdim:sdim+feat_dim], dim=-1)
            X_list.append(X_)
            sdim += feat_dim
        X = torch.cat(X_list, dim=-1)
        E = torch.sigmoid(E)

        X = mask_x(X, node_mask) 
        E = mask_adjs(E.permute(0,3,1,2), node_mask).permute(0,2,3,1).squeeze(-1) * diag_mask.squeeze(-1)

        return X, E 


class GraphTransformer_Mol(nn.Module):
    """
    n_layers : int -- number of layers
    dims : dict -- contains dimensions for each feature type
    """
    def __init__(self, n_layers: int, input_dims: dict, hidden_mlp_dims: dict, hidden_dims: dict,
                 output_dims: dict, scale: float = 3.0):
        super().__init__()
        self.scale = scale
        self.input_dims = input_dims

        self.n_layers = n_layers
        self.out_dim_X = output_dims['X']
        self.out_dim_E = output_dims['E']
        self.out_dim_y = output_dims['y']

        act_fn_in = nn.ReLU()
        act_fn_out = nn.ReLU()

        self.mlp_in_X = nn.Sequential(nn.Linear(input_dims['X'], hidden_mlp_dims['X']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['X'], hidden_dims['dx']), act_fn_in)

        self.mlp_in_E = nn.Sequential(nn.Linear(input_dims['E'], hidden_mlp_dims['E']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['E'], hidden_dims['de']), act_fn_in)

        self.mlp_in_y = nn.Sequential(nn.Linear(input_dims['y'], hidden_mlp_dims['y']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['y'], hidden_dims['dy']), act_fn_in)

        self.tf_layers = nn.ModuleList([XEyTransformerLayer(dx=hidden_dims['dx'],
                                                            de=hidden_dims['de'],
                                                            dy=hidden_dims['dy'],
                                                            n_head=hidden_dims['n_head'],
                                                            dim_ffX=hidden_dims['dim_ffX'],
                                                            dim_ffE=hidden_dims['dim_ffE'])
                                        for i in range(n_layers)])

        self.mlp_out_X = nn.Sequential(nn.Linear(hidden_dims['dx'], hidden_mlp_dims['X']), act_fn_out,
                                       nn.Linear(hidden_mlp_dims['X'], output_dims['X']))

        self.mlp_out_E = nn.Sequential(nn.Linear(hidden_dims['de'], hidden_mlp_dims['E']), act_fn_out,
                                       nn.Linear(hidden_mlp_dims['E'], output_dims['E']))

        if self.out_dim_y>0:
            self.mlp_out_y = nn.Sequential(nn.Linear(hidden_dims['dy'], hidden_mlp_dims['y']), act_fn_out,
                                            nn.Linear(hidden_mlp_dims['y'], output_dims['y']))

    def forward(self, X, E, y, node_mask):
        bs, n = X.shape[0], X.shape[1]

        # higher-order adj
        E = pow_tensor(E, self.input_dims['E']).permute(0,2,3,1)

        diag_mask = torch.eye(n)
        diag_mask = ~diag_mask.type_as(E).bool()
        diag_mask = diag_mask.unsqueeze(0).unsqueeze(-1).expand(bs, -1, -1, -1)

        X_to_out = X[..., :self.out_dim_X]
        E_to_out = E[..., :self.out_dim_E]

        new_E = self.mlp_in_E(E)
        new_E = (new_E + new_E.transpose(1, 2)) / 2

        X = mask_x(self.mlp_in_X(X), node_mask)
        E = mask_adjs(new_E.permute(0,3,1,2), node_mask).permute(0,2,3,1) # bs x n x n x c
        y = self.mlp_in_y(y)

        for layer in self.tf_layers:
            X, E, y = layer(X, E, y, node_mask)

        X = self.mlp_out_X(X)
        E = self.mlp_out_E(E)

        X = (X + X_to_out)
        E = (E + E_to_out) * diag_mask

        if self.out_dim_y>0:
            y_to_out = y[..., :self.out_dim_y]
            y = self.mlp_out_y(y)
            y = y + y_to_out

        E = 1/2 * (E + torch.transpose(E, 1, 2))

        X = torch.nn.functional.softmax(X, dim=-1)
        E = torch.sigmoid(E)
        E = E[:,:,:,0] * 1./self.scale + E[:,:,:,1] * 2./self.scale
        
        X = mask_x(X, node_mask) 
        E = mask_adjs(E, node_mask) * diag_mask.squeeze(-1)

        return X, E