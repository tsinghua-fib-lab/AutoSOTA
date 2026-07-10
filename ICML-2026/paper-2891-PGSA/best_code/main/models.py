import torch
from torch import nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np
from torch_geometric.nn import MessagePassing, GCNConv, SAGEConv, GATConv, GatedGraphConv, GINConv
from torch.autograd import Function
import copy
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_sparse import SparseTensor, fill_diag, sum as sparse_sum, matmul as sparse_matmul
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from math import pi
import torch_geometric.nn as pyg_nn
# from torch_geometric.utils import spmm
from torch import Tensor
from torch.nn import Parameter
from torch_sparse import SparseTensor, fill_diag, mul
from torch_sparse import sum as sparsesum
from typing import Optional
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import Adj, OptPairTensor, OptTensor
from torch_geometric.utils.num_nodes import maybe_num_nodes
# from torch_geometric.utils import is_torch_sparse_tensor
from torch_sparse import matmul as torch_sparse_matmul
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from directed_gnn import get_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


def spmm(src: Adj, other: Tensor, reduce: str = "sum") -> Tensor:
    """Matrix product of sparse matrix with dense matrix.

    Args:
        src (Tensor or torch_sparse.SparseTensor]): The input sparse matrix,
            either a :class:`torch_sparse.SparseTensor` or a
            :class:`torch.sparse.Tensor`.
        other (Tensor): The input dense matrix.
        reduce (str, optional): The reduce operation to use
            (:obj:`"sum"`, :obj:`"mean"`, :obj:`"min"`, :obj:`"max"`).
            (default: :obj:`"sum"`)

    :rtype: :class:`Tensor`
    """
    assert reduce in ['sum', 'add', 'mean', 'min', 'max']

    if isinstance(src, SparseTensor):
        return torch_sparse_matmul(src, other, reduce)

    if reduce in ['sum', 'add']:
        return torch.sparse.mm(src, other)

    # TODO: Support `mean` reduction for PyTorch SparseTensor
    raise ValueError(f"`{reduce}` reduction is not supported for "
                     f"`torch.sparse.Tensor`.")


class GCN_reweight(MessagePassing):
    r"""The graph convolutional operator from the `"Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.
    The adjacency matrix can include other values than :obj:`1` representing
    edge weights via the optional :obj:`edge_weight` tensor.

    Its node-wise formulation is given by:

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta}^{\top} \sum_{j \in
        \mathcal{N}(v) \cup \{ i \}} \frac{e_{j,i}}{\sqrt{\hat{d}_j
        \hat{d}_i}} \mathbf{x}_j

    with :math:`\hat{d}_i = 1 + \sum_{j \in \mathcal{N}(i)} e_{j,i}`, where
    :math:`e_{j,i}` denotes the edge weight from source node :obj:`j` to target
    node :obj:`i` (default: :obj:`1.0`)

    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        out_channels (int): Size of each output sample.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        normalize (bool, optional): Whether to add self-loops and compute
            symmetric normalization coefficients on the fly.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})`,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge weights :math:`(|\mathcal{E}|)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})`
    """

    _cached_edge_index: Optional[OptPairTensor]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, in_channels: int, out_channels: int,
                 improved: bool = False, cached: bool = False,
                 add_self_loops: bool = False, normalize: bool = True,
                 bias: bool = True, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.lin = Linear(in_channels, out_channels, bias=False,
                          weight_initializer='glorot')

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""
        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        x = self.lin(x)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)

        if self.bias is not None:
            out = out + self.bias

        return out

    # def message(self, x_j: Tensor, edge_weight: OptTensor, lmda, edge_rw) -> Tensor:
    #     x_j = (edge_weight.view(-1, 1) * x_j)
    #     # x_j = (1-lmda) * x_j + (lmda) * (edge_rw.view(-1, 1) * x_j)
    #     x_j =  x_j + (lmda) * (edge_rw.view(-1, 1) * x_j)
    #     return x_j
    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)


class GS_reweight(pyg_nn.MessagePassing):
    def __init__(self, in_channels, out_channels, reducer='mean',
                 normalize_embedding=False):
        super(GS_reweight, self).__init__(aggr='mean')
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.agg_lin = torch.nn.Linear(out_channels + in_channels, out_channels)

        self.normalize_emb = normalize_embedding

    def forward(self, x, edge_index, edge_weight, lmda):
        num_nodes = x.size(0)
        return self.propagate(edge_index, size=(num_nodes, num_nodes), x=x, edge_weight=edge_weight, lmda=lmda)

    def message(self, x_j, edge_index, edge_weight, lmda):
        # x_j = self.lin(x_j)
        # x_j = (1-lmda) * x_j + (lmda) * (edge_weight.view(-1, 1) * x_j)
        x_j = self.lin(x_j)
        x_j = x_j + (lmda) * (edge_weight.view(-1, 1) * x_j)
        # print(lmda)
        return x_j

    def update(self, aggr_out, x):
        aggr_out = torch.cat((aggr_out, x), dim=-1)
        aggr_out = self.agg_lin(aggr_out)
        aggr_out = F.relu(aggr_out)

        if self.normalize_emb:
            aggr_out = F.normalize(aggr_out, p=2, dim=-1)

        return aggr_out


class GNN(torch.nn.Module):
    def __init__(self, input_dim, output_dim, args):
        super(GNN, self).__init__()
        if args.backbone == 'GCN':
            self.prop_input = GCN_reweight(input_dim, args.hidden_dim)
            self.prop_hidden = GCN_reweight(args.hidden_dim, args.hidden_dim)
            self.prop_out = GCN_reweight(args.hidden_dim, args.conv_dim)
        elif args.backbone == 'GS':
            self.prop_input = GS_reweight(input_dim, args.hidden_dim)
            self.prop_hidden = GS_reweight(args.hidden_dim, args.hidden_dim)
            self.prop_out = GS_reweight(args.hidden_dim, args.conv_dim)

        self.dropout = args.dropout
        self.K = args.K
        self.bn = args.bn
        self.lmda = args.rw_lmda
        self.hidden_dim = args.hidden_dim

        # conv layers
        self.conv = nn.ModuleList()
        self.conv.append(self.prop_input)
        for i in range(args.K - 2):
            self.conv.append(self.prop_hidden)
        self.conv.append(self.prop_out)

        # bn layers
        self.bns = nn.ModuleList()
        for i in range(args.K - 1):
            self.bns.append(nn.BatchNorm1d(self.hidden_dim))

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.mod_edge_index, data.mod_edge_weight
        for i, layer in enumerate(self.conv):
            x = layer(x, edge_index, edge_weight=edge_weight)
            if self.bn and (i != len(self.conv) - 1):
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout)

        return x


class Adv_DANN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, f_layer, dc_layer, args):
        super(Adv_DANN, self).__init__()
        self.f_layer = f_layer
        if f_layer > 0:
            self.f = nn.ModuleList()
            for l in range(f_layer):
                self.f.append(nn.Linear(input_dim, input_dim))

        self.dc = nn.ModuleList()
        if dc_layer == 1:
            self.dc.append(nn.Linear(input_dim, out_dim))
        else:
            self.dc.append(nn.Linear(input_dim, hidden_dim))
            for l in range(dc_layer - 2):
                self.dc.append(nn.Linear(hidden_dim, hidden_dim))
            self.dc.append(nn.Linear(hidden_dim, out_dim))
        self.dropout = args.dropout
        self.resnet = args.resnet

    def forward(self, x, alpha):
        if self.f_layer > 0:
            x = self.f[0](x)
            if self.resnet:
                for i in range(1, len(self.f)):
                    if i != len(self.f) - 1:
                        x = self.f[i](x.clone()) + x.clone()
                    else:
                        x = self.f[i](x)
                    x = F.leaky_relu(x.clone())
                    x = F.dropout(x.clone(), p=self.dropout)
            else:
                for i in range(1, len(self.f)):
                    x = self.f[i](x)
                    x = F.relu(x)
                    x = F.dropout(x, p=self.dropout)

        y = GradReverse.apply(x, alpha)

        for i in range(len(self.dc) - 1):
            y = self.dc[i](y)
            y = F.relu(y)
            # y = F.dropout(y, p=self.dropout)

        y = self.dc[len(self.dc) - 1](y)
        y = torch.sigmoid(y)
        return x, y


class GNN_adv(torch.nn.Module):
    def __init__(self, input_dim, output_dim, args):
        super(GNN_adv, self).__init__()
        # GNN block
        self.GNN_block = GNN(input_dim, output_dim, args)
        # adv block
        self.hidden_dim = args.hidden_dim
        self.adv_block = Adv_DANN(args.conv_dim, args.cls_dim, 1, args.num_layers, args.dc_layers, args)
        self.K = args.K
        self.total_epoch = args.epochs
        self.mlp_classify = nn.ModuleList()
        # classification layer
        if args.class_layers == 1:
            self.mlp_classify.append(nn.Linear(args.conv_dim, output_dim))
        else:
            for i in range(args.class_layers - 1):
                self.mlp_classify.append(nn.Linear(args.conv_dim, args.cls_dim))
            self.mlp_classify.append(nn.Linear(args.cls_dim, output_dim))

        self.dropout = args.dropout
        self.alphatimes = args.alphatimes
        self.alphamin = args.alphamin

    def forward(self, data, alpha):
        GNN_embed = self.GNN_block.forward(data)
        final_embed, pred_domain = self.adv_block.forward(GNN_embed, alpha)

        # source label prediction
        x = final_embed
        for i in range(len(self.mlp_classify)):
            x = self.mlp_classify[i](x)
            if i != (len(self.mlp_classify) - 1):
                x = F.relu(x)
            # x = F.dropout(x, p=self.dropout)
        return [GNN_embed, final_embed], [x, pred_domain]


class directed_GNN_adv(torch.nn.Module):
    def __init__(self, input_dim, output_dim, args):
        super(directed_GNN_adv, self).__init__()
        # GNN block
        self.GNN_block = get_model(input_dim, output_dim, args)
        # adv block
        self.hidden_dim = args.hidden_dim
        self.adv_block = Adv_DANN(args.conv_dim, args.cls_dim, 1, args.num_layers, args.dc_layers, args)
        self.K = args.K
        self.total_epoch = args.epochs
        self.mlp_classify = nn.ModuleList()
        # classification layer
        if args.class_layers == 1:
            self.mlp_classify.append(nn.Linear(args.conv_dim, output_dim))
        else:
            for i in range(args.class_layers - 1):
                self.mlp_classify.append(nn.Linear(args.conv_dim, args.cls_dim))
            self.mlp_classify.append(nn.Linear(args.cls_dim, output_dim))

        self.dropout = args.dropout
        self.alphatimes = args.alphatimes
        self.alphamin = args.alphamin

    def forward(self, data, alpha):
        x, edge_index, edge_weight = data.x, data.mod_edge_index, data.mod_edge_weight
        GNN_embed = self.GNN_block.forward(x, edge_index, edge_weight)
        final_embed, pred_domain = self.adv_block.forward(GNN_embed, alpha)

        # source label prediction
        x = final_embed
        for i in range(len(self.mlp_classify)):
            x = self.mlp_classify[i](x)
            if i != (len(self.mlp_classify) - 1):
                x = F.relu(x)
            # x = F.dropout(x, p=self.dropout)
        return [GNN_embed, final_embed], [x, pred_domain]


class one_lyr_MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return F.relu(self.linear(x))


class SimpleGCN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = GCNConv(in_dim, out_dim)

    def forward(self, x, edge_index):
        return F.relu(self.conv(x, edge_index))


class GNNWithMLPClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, args):
        super().__init__()

        self.gnn = GCNConv(input_dim, args.conv_dim)

        self.mlp_classify = nn.ModuleList()
        dims = [args.conv_dim] + [args.cls_dim] * (args.class_layers - 1) + [output_dim]
        for i in range(len(dims) - 1):
            self.mlp_classify.append(nn.Linear(dims[i], dims[i + 1]))

        self.dropout = args.dropout

    def forward(self, data):
        gnn_embed = self.gnn(data.x, data.edge_index)
        final_embed = F.relu(gnn_embed)

        x = final_embed
        for i in range(len(self.mlp_classify)):
            x = self.mlp_classify[i](x)
            if i != len(self.mlp_classify) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return final_embed, x


class MLPWithMLPClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, args):
        super().__init__()
        self.encoder = nn.Linear(input_dim, args.mlp_conv_dim)


        self.mlp_classify = nn.ModuleList()
        dims = [args.mlp_conv_dim] + [args.mlp_cls_dim] * (args.mlp_class_layers - 1) + [output_dim]
        for i in range(len(dims) - 1):
            self.mlp_classify.append(nn.Linear(dims[i], dims[i + 1]))

        self.dropout = args.dropout

    def forward(self, data):
        embed = F.relu(self.encoder(data.x))
        x = embed
        for i in range(len(self.mlp_classify)):
            x = self.mlp_classify[i](x)
            if i != len(self.mlp_classify) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return embed, x