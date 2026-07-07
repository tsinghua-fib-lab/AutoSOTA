import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import graphgym.register as register
from graphgym.config import cfg
from graphgym.init import init_weights
from graphgym.models.act import act_dict
from graphgym.models.feature_augment import Preprocess
from graphgym.models.feature_encoder import (edge_encoder_dict,
                                             node_encoder_dict)
from graphgym.models.head import head_dict
from graphgym.models.layer import (BatchNorm1dEdge, BatchNorm1dNode,
                                   GeneralLayer, GeneralMultiLayer)


# Layer
def GNNLayer(dim_in, dim_out, has_act=True):
    """
    Wrapper for a GNN layer

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension
        has_act (bool): Whether has activation function after the layer

    """
    return GeneralLayer(cfg.gnn.layer_type, dim_in, dim_out, has_act)


def GNNPreMP(dim_in, dim_out):
    """
    Wrapper for NN layer before GNN message passing

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension
        num_layers (int): Number of layers

    """
    return GeneralMultiLayer('linear',
                             cfg.gnn.layers_pre_mp,
                             dim_in,
                             dim_out,
                             dim_inner=dim_out,
                             final_act=True)


# Block: multiple layers
class GNNSkipBlock(nn.Module):
    '''Skip block for GNN'''
    def __init__(self, dim_in, dim_out, num_layers):
        super(GNNSkipBlock, self).__init__()
        if num_layers == 1:
            self.f = [GNNLayer(dim_in, dim_out, has_act=False)]
        else:
            self.f = []
            for i in range(num_layers - 1):
                d_in = dim_in if i == 0 else dim_out
                self.f.append(GNNLayer(d_in, dim_out))
            d_in = dim_in if num_layers == 1 else dim_out
            self.f.append(GNNLayer(d_in, dim_out, has_act=False))
        self.f = nn.Sequential(*self.f)
        self.act = act_dict[cfg.gnn.act]
        if cfg.gnn.stage_type == 'skipsum':
            assert dim_in == dim_out, 'Sum skip must have same dim_in, dim_out'

    def forward(self, batch):
        node_feature = batch.node_feature
        if cfg.gnn.stage_type == 'skipsum':
            batch.node_feature = \
                node_feature + self.f(batch).node_feature
        elif cfg.gnn.stage_type == 'skipconcat':
            batch.node_feature = \
                torch.cat((node_feature, self.f(batch).node_feature), 1)
        else:
            batch.node_feature = self.f(batch).node_feature
        # else:
        #     raise ValueError(
        #         'cfg.gnn.stage_type must in [skipsum, skipconcat]')
        batch.node_feature = self.act(batch.node_feature)
        return batch


# Stage: NN except start and head
class GNNStackStage(nn.Module):
    '''Simple Stage that stack GNN layers'''
    def __init__(self, dim_in, dim_out, num_layers):
        super(GNNStackStage, self).__init__()
        for i in range(num_layers):
            d_in = dim_in if i == 0 else dim_out
            layer = GNNLayer(d_in, dim_out)
            self.add_module('layer{}'.format(i), layer)
        self.dim_out = dim_out

    def forward(self, batch):
        for layer in self.children():
            batch = layer(batch)
        if cfg.gnn.l2norm:
            batch.node_feature = F.normalize(batch.node_feature, p=2, dim=-1)
        return batch


class GNNSkipStage(nn.Module):
    ''' Stage with skip connections'''
    def __init__(self, dim_in, dim_out, num_layers):
        super(GNNSkipStage, self).__init__()
        assert num_layers % cfg.gnn.skip_every == 0, \
            'cfg.gnn.skip_every must be multiples of cfg.gnn.layer_mp' \
            '(excluding head layer)'
        self.mode = cfg.gnn.stage_type
        self.blocks = nn.ModuleList()
        for i in range(num_layers // cfg.gnn.skip_every):
            if self.mode in ['skipsum', 'mean', 'gpr', 'lstm', 'node_adaptive'] or 'ppr' in self.mode:
                d_in = dim_in if i == 0 else dim_out
            elif self.mode == 'skipconcat':
                d_in = dim_in if i == 0 else dim_in + i * dim_out
            else:
                raise NotImplementedError('stage_type not supported')
            block = GNNSkipBlock(d_in, dim_out, cfg.gnn.skip_every)
            #self.add_module('block{}'.format(i), block)
            self.blocks.append(block)
        if cfg.gnn.stage_type == 'skipconcat':
            self.dim_out = d_in + dim_out
        else:
            self.dim_out = dim_out
        
        # Inter-layer weights
        self.alpha, self.ppr_weights = 0.5, None
        self.gammas = None
        self.lstm, self.att = None, None
        self.s = None

        if 'ppr' in self.mode:
            self.K = num_layers
            self.alpha = float(self.mode.split('_')[-1])
            self._initialize_ppr()
        elif self.mode == 'gpr':
            self.K = num_layers
            self._initialize_gammas()
        elif self.mode == 'lstm':
            self.lstm = nn.LSTM(self.dim_out, (2 * self.dim_out) // 2, bidirectional=True, batch_first=True)
            self.att = nn.Linear(2 * ((2 * self.dim_out) // 2), 1)
        elif self.mode == 'node_adaptive':
            self._initialize_s()
        self.reset_parameters()

    def _initialize_ppr(self):
        TEMP = self.alpha*(1-self.alpha)**np.arange(self.K+1)
        TEMP[-1] = (1-self.alpha)**self.K
        self.ppr_weights = nn.Parameter(torch.tensor(TEMP, dtype=torch.float), requires_grad=False)
    
    def _initialize_gammas(self):
        # Our previous code use random_init
        # But the paper states random_init may lead to slight performance drop
        # Hence we follow the official implementation of GPRGNN to use PPR_init
        # But alpha is still a hyper param, just try 0.5 now
        ###################### random_init ##########################
        bound = np.sqrt(3/(self.K+1))
        TEMP = np.random.uniform(0, bound, self.K+1)
        TEMP = TEMP/np.sum(np.abs(TEMP))
        ###################### PPR_init ##########################
        # TEMP = self.alpha*(1-self.alpha)**np.arange(self.K+1)
        # TEMP[-1] = (1-self.alpha)**self.K
        self.gammas = nn.Parameter(torch.tensor(TEMP, dtype=torch.float))
    
    def _initialize_s(self):
        self.s = nn.Parameter(torch.Tensor(1, self.dim_out))
        nn.init.xavier_uniform_(self.s)
    
    def reset_parameters(self):
        # reset params for learnable candidates: gpr, lstm, node_adaptive
        if self.gammas is not None:
            self._initialize_gammas()
        if self.lstm is not None:
            self.lstm.reset_parameters()
        if self.att is not None:
            self.att.reset_parameters()
        if self.s is not None:
            self._initialize_s()

    def forward(self, batch):
        xs = [batch.node_feature]
        #for layer in self.children():
        for layer in self.blocks:
            batch = layer(batch)
            xs.append(batch.node_feature)
        
        if self.mode == 'mean':
            batch.node_feature = torch.stack(xs, dim=-1).mean(dim=-1)
        elif 'ppr' in self.mode:
            batch.node_feature = torch.einsum('i,ijk->jk', self.ppr_weights, torch.stack(xs))
        elif self.mode == 'gpr':
            batch.node_feature = torch.einsum('i,ijk->jk', torch.tanh(self.gammas), torch.stack(xs))
        elif self.mode == 'lstm':
            assert self.lstm is not None and self.att is not None
            x = torch.stack(xs, dim=1)              # [num_nodes, num_layers, num_channels]
            alpha, _ = self.lstm(x)
            alpha = self.att(alpha).squeeze(-1)     # [num_nodes, num_layers]
            alpha = torch.softmax(alpha, dim=-1)
            batch.node_feature = (x * alpha.unsqueeze(-1)).sum(dim=1)
        elif self.mode == 'node_adaptive':
            x = torch.stack(xs)                     # [num_layers, num_nodes, num_channels]
            temp = torch.einsum('KNd,Id->KN', x, self.s)
            scores = torch.tanh(temp)
            batch.node_feature = torch.einsum('KNd,KN->Nd', x, scores)
        
        if cfg.gnn.l2norm:
            batch.node_feature = F.normalize(batch.node_feature, p=2, dim=-1)
        return batch


stage_dict = {
    'stack': GNNStackStage,
    'skipsum': GNNSkipStage,
    'skipconcat': GNNSkipStage,
    'mean': GNNSkipStage,
    'ppr_0.1': GNNSkipStage,
    'ppr_0.5': GNNSkipStage,
    'gpr': GNNSkipStage,
    'lstm': GNNSkipStage,
    'node_adaptive': GNNSkipStage
}

stage_dict = {**register.stage_dict, **stage_dict}


# Model: start + stage + head
class GNN(nn.Module):
    '''General GNN model'''
    def __init__(self, dim_in, dim_out, **kwargs):
        """
            Parameters:
            node_encoding_classes - For integer features, gives the number
            of possible integer features to map.
        """
        super(GNN, self).__init__()
        GNNStage = stage_dict[cfg.gnn.stage_type]
        GNNHead = head_dict[cfg.dataset.task]

        if cfg.dataset.node_encoder:
            # Encode integer node features via nn.Embeddings
            NodeEncoder = node_encoder_dict[cfg.dataset.node_encoder_name]
            self.node_encoder = NodeEncoder(cfg.dataset.encoder_dim)
            if cfg.dataset.node_encoder_bn:
                self.node_encoder_bn = BatchNorm1dNode(cfg.dataset.encoder_dim)
            # Update dim_in to reflect the new dimension fo the node features
            dim_in = cfg.dataset.encoder_dim
        if cfg.dataset.edge_encoder:
            # Encode integer edge features via nn.Embeddings
            EdgeEncoder = edge_encoder_dict[cfg.dataset.edge_encoder_name]
            self.edge_encoder = EdgeEncoder(cfg.dataset.encoder_dim)
            if cfg.dataset.edge_encoder_bn:
                self.edge_encoder_bn = BatchNorm1dEdge(cfg.dataset.edge_dim)

        self.preprocess = Preprocess(dim_in)
        d_in = self.preprocess.dim_out
        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(d_in, cfg.gnn.dim_inner)
            d_in = cfg.gnn.dim_inner
        if cfg.gnn.layers_mp > 0:
            self.mp = GNNStage(dim_in=d_in,
                               dim_out=cfg.gnn.dim_inner,
                               num_layers=cfg.gnn.layers_mp)
            d_in = self.mp.dim_out
        self.post_mp = GNNHead(dim_in=d_in, dim_out=dim_out)

        self.apply(init_weights)

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch
