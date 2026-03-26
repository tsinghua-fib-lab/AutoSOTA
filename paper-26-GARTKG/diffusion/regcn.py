import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import dgl.function as fn

class BaseRGCN(nn.Module):
    def __init__(self, num_nodes, h_dim, out_dim, num_rels, num_bases=-1, num_basis=-1,
                 num_hidden_layers=1, dropout=0, self_loop=False, skip_connect=False, encoder_name="", opn="sub", rel_emb=None, use_cuda=False, analysis=False):
        super(BaseRGCN, self).__init__()
        self.num_nodes = num_nodes
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.num_basis = num_basis
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.skip_connect = skip_connect
        self.self_loop = self_loop
        self.encoder_name = encoder_name
        self.use_cuda = use_cuda
        self.run_analysis = analysis
        self.skip_connect = skip_connect
        print("use layer :{}".format(encoder_name))
        self.rel_emb = rel_emb
        self.opn = opn
        # create rgcn layers
        self.build_model()
        # create initial features
        self.features = self.create_features()

    def build_model(self):
        self.layers = nn.ModuleList()
        # i2h
        i2h = self.build_input_layer()
        if i2h is not None:
            self.layers.append(i2h)
        # h2h
        for idx in range(self.num_hidden_layers):

            h2h = self.build_hidden_layer(idx)
            self.layers.append(h2h)
        # h2o
        h2o = self.build_output_layer()
        if h2o is not None:
            self.layers.append(h2o)

    # initialize feature for each node
    def create_features(self):
        return None

    def build_input_layer(self):
        return None

    def build_hidden_layer(self, idx):
        raise NotImplementedError

    def build_output_layer(self):
        return None

    def forward(self, g):
        if self.features is not None:
            g.ndata['id'] = self.features
        print("h before GCN message passing")
        print(g.ndata['h'])
        print("h behind GCN message passing")
        for layer in self.layers:
            layer(g)
        print(g.ndata['h'])
        return g.ndata.pop('h')

class RGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, bias=None, activation=None,
                 self_loop=False, skip_connect=False, dropout=0.0, layer_norm=False):
        super(RGCNLayer, self).__init__()
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.skip_connect = skip_connect
        self.layer_norm = layer_norm

        if self.bias:
            self.bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.xavier_uniform_(self.bias,
                                    gain=nn.init.calculate_gain('relu'))

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))
            # self.loop_weight = nn.Parameter(torch.eye(out_feat), requires_grad=False)

        if self.skip_connect:
            self.skip_connect_weight = nn.Parameter(torch.Tensor(out_feat, out_feat))   # 和self-loop不一样，是跨层的计算
            nn.init.xavier_uniform_(self.skip_connect_weight,
                                    gain=nn.init.calculate_gain('relu'))

            self.skip_connect_bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.zeros_(self.skip_connect_bias)  # 初始化设置为0

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        if self.layer_norm:
            self.normalization_layer = nn.LayerNorm(out_feat, elementwise_affine=False)

    # define how propagation is done in subclass
    def propagate(self, g):
        raise NotImplementedError

    def forward(self, g, prev_h=[]):
        if self.self_loop:
            #print(self.loop_weight)
            loop_message = torch.mm(g.ndata['h'], self.loop_weight)
            if self.dropout is not None:
                loop_message = self.dropout(loop_message)
        # self.skip_connect_weight.register_hook(lambda g: print("grad of skip connect weight: {}".format(g)))
        if len(prev_h) != 0 and self.skip_connect:
            skip_weight = F.sigmoid(torch.mm(prev_h, self.skip_connect_weight) + self.skip_connect_bias)     # 使用sigmoid，让值在0~1
            # print("skip_ weight")
            # print(skip_weight)
            # print("skip connect weight")
            # print(self.skip_connect_weight)
            # print(torch.mm(prev_h, self.skip_connect_weight))

        self.propagate(g)  # 这里是在计算从周围节点传来的信息

        # apply bias and activation
        node_repr = g.ndata['h']
        if self.bias:
            node_repr = node_repr + self.bias
        # print(len(prev_h))
        if len(prev_h) != 0 and self.skip_connect:   # 两次计算loop_message的方式不一样，前者激活后再加权
            previous_node_repr = (1 - skip_weight) * prev_h
            if self.activation:
                node_repr = self.activation(node_repr)
            if self.self_loop:
                if self.activation:
                    loop_message = skip_weight * self.activation(loop_message)
                else:
                    loop_message = skip_weight * loop_message
                node_repr = node_repr + loop_message
            node_repr = node_repr + previous_node_repr
        else:
            if self.self_loop:
                node_repr = node_repr + loop_message
            if self.layer_norm:
                node_repr = self.normalization_layer(node_repr)
            if self.activation:
                node_repr = self.activation(node_repr)
            # print("node_repr")
            # print(node_repr)
        g.ndata['h'] = node_repr
        return node_repr
    
class RGCNBlockLayer(RGCNLayer):
    def __init__(self, in_feat, out_feat, num_rels, num_bases, bias=None,
                 activation=None, self_loop=False, dropout=0.0, skip_connect=False, layer_norm=False):
        super(RGCNBlockLayer, self).__init__(in_feat, out_feat, bias,
                                             activation, self_loop=self_loop, skip_connect=skip_connect,
                                             dropout=dropout)
        self.num_rels = num_rels
        self.num_bases = num_bases

        assert self.num_bases > 0

        self.out_feat = out_feat
        self.submat_in = in_feat // self.num_bases
        self.submat_out = out_feat // self.num_bases

        # assuming in_feat and out_feat are both divisible by num_bases
        self.weight = nn.Parameter(torch.Tensor(
            self.num_rels, self.num_bases * self.submat_in * self.submat_out))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

    def msg_func(self, edges):
        weight = self.weight.index_select(0, edges.data['type']).view(
                    -1, self.submat_in, self.submat_out)    # [edge_num, submat_in, submat_out]
        node = edges.src['h'].view(-1, 1, self.submat_in)   # [edge_num * num_bases, 1, submat_in]->
        msg = torch.bmm(node, weight).view(-1, self.out_feat)   # [edge_num, out_feat]
        return {'msg': msg}

    def propagate(self, g):
        g.update_all(self.msg_func, fn.sum(msg='msg', out='h'), self.apply_func)
        # g.updata_all ({'msg': msg} , fn.sum(msg='msg', out='h'), {'h': nodes.data['h'] * nodes.data[''norm]})

    def apply_func(self, nodes):
        return {'h': nodes.data['h'] * nodes.data['norm']}
    
class RGCNCell(BaseRGCN):
    def build_hidden_layer(self, idx):
        act = F.rrelu
        if idx:
            self.num_basis = 0
        print("activate function: {}".format(act))
        if self.skip_connect:
            sc = False if idx == 0 else True
        else:
            sc = False
        if self.encoder_name == "uvrgcn":
            return UnionRGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                             activation=act, dropout=self.dropout, self_loop=self.self_loop, skip_connect=sc, rel_emb=self.rel_emb)
        else:
            raise NotImplementedError


    def forward(self, g, init_ent_emb, init_rel_emb):
        if self.encoder_name == "uvrgcn":
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            x, r = init_ent_emb, init_rel_emb
            for i, layer in enumerate(self.layers):
                layer(g, [], r[i])
            return g.ndata.pop('h')
        else:
            if self.features is not None:
                print("----------------Feature is not None, Attention ------------")
                g.ndata['id'] = self.features
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            if self.skip_connect:
                prev_h = []
                for layer in self.layers:
                    prev_h = layer(g, prev_h)
            else:
                for layer in self.layers:
                    layer(g, [])
            return g.ndata.pop('h')
        
class UnionRGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, num_rels, num_bases=-1,  bias=None,
                 activation=None, self_loop=False, dropout=0.0, skip_connect=False, rel_emb=None):
        super(UnionRGCNLayer, self).__init__()

        self.in_feat = in_feat
        self.out_feat = out_feat
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.num_rels = num_rels
        self.rel_emb = None
        self.skip_connect = skip_connect
        self.ob = None
        self.sub = None

        # WL
        self.weight_neighbor = nn.Parameter(torch.Tensor(self.in_feat, self.out_feat))
        nn.init.xavier_uniform_(self.weight_neighbor, gain=nn.init.calculate_gain('relu'))

        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))
            self.evolve_loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.evolve_loop_weight, gain=nn.init.calculate_gain('relu'))

        if self.skip_connect:
            self.skip_connect_weight = nn.Parameter(torch.Tensor(out_feat, out_feat))   # 和self-loop不一样，是跨层的计算
            nn.init.xavier_uniform_(self.skip_connect_weight,gain=nn.init.calculate_gain('relu'))
            self.skip_connect_bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.zeros_(self.skip_connect_bias)  # 初始化设置为0

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def propagate(self, g):
        g.update_all(lambda x: self.msg_func(x), fn.sum(msg='msg', out='h'), self.apply_func)

    def forward(self, g, prev_h, emb_rel):
        self.rel_emb = emb_rel
        # self.sub = sub
        # self.ob = ob
        if self.self_loop:
            #loop_message = torch.mm(g.ndata['h'], self.loop_weight)
            # masked_index = torch.masked_select(torch.arange(0, g.number_of_nodes(), dtype=torch.long), (g.in_degrees(range(g.number_of_nodes())) > 0))
            masked_index = torch.masked_select(
                torch.arange(0, g.number_of_nodes(), dtype=torch.long).cuda(),
                (g.in_degrees(range(g.number_of_nodes())) > 0))
            loop_message = torch.mm(g.ndata['h'], self.evolve_loop_weight)
            loop_message[masked_index, :] = torch.mm(g.ndata['h'], self.loop_weight)[masked_index, :]
        if len(prev_h) != 0 and self.skip_connect:
            skip_weight = F.sigmoid(torch.mm(prev_h, self.skip_connect_weight) + self.skip_connect_bias)     # 使用sigmoid，让值在0~1

        # calculate the neighbor message with weight_neighbor
        self.propagate(g)
        node_repr = g.ndata['h']

        # print(len(prev_h))
        if len(prev_h) != 0 and self.skip_connect:  # 两次计算loop_message的方式不一样，前者激活后再加权
            if self.self_loop:
                node_repr = node_repr + loop_message
            node_repr = skip_weight * node_repr + (1 - skip_weight) * prev_h
        else:
            if self.self_loop:
                node_repr = node_repr + loop_message

        if self.activation:
            node_repr = self.activation(node_repr)
        if self.dropout is not None:
            node_repr = self.dropout(node_repr)
        g.ndata['h'] = node_repr
        return node_repr

    def msg_func(self, edges):
        # if reverse:
        #     relation = self.rel_emb.index_select(0, edges.data['type_o']).view(-1, self.out_feat)
        # else:
        #     relation = self.rel_emb.index_select(0, edges.data['type_s']).view(-1, self.out_feat)
        relation = self.rel_emb.index_select(0, edges.data['type']).view(-1, self.out_feat)
        edge_type = edges.data['type']
        edge_num = edge_type.shape[0]
        node = edges.src['h'].view(-1, self.out_feat)
        # node = torch.cat([torch.matmul(node[:edge_num // 2, :], self.sub),
        #                  torch.matmul(node[edge_num // 2:, :], self.ob)])
        # node = torch.matmul(node, self.sub)

        # after add inverse edges, we only use message pass when h as tail entity
        # 这里计算的是每个节点发出的消息，节点发出消息时其作为头实体
        # msg = torch.cat((node, relation), dim=1)
        msg = node + relation
        # calculate the neighbor message with weight_neighbor
        msg = torch.mm(msg, self.weight_neighbor)
        return {'msg': msg}

    def apply_func(self, nodes):
        return {'h': nodes.data['h'] * nodes.data['norm']}