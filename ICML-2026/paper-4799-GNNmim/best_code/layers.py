from typing import Optional, Tuple, Union
from torch_geometric.typing import OptPairTensor, Adj, Size, OptTensor, PairTensor, NoneType
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter, Linear
import torch.nn as nn
# from torch_scatter import scatter_add
# from torch_sparse import SparseTensor, matmul, fill_diag, sum, mul, set_diag
# from torch_sparse import matmul
# from torch_sparse import fill_diag
# from torch_sparse import sum
# from torch_sparse import mul
# from torch_sparse import set_diag
from torch_geometric.typing import SparseTensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops, remove_self_loops, add_self_loops, softmax, to_dense_batch, to_dense_adj
from torch_geometric.utils.num_nodes import maybe_num_nodes

from torch_geometric.nn.inits import reset, glorot, zeros
from torch.nn import ModuleList, Linear, BatchNorm1d

"""
Pytorch Geometric
Ref: https://github.com/pyg-team/pytorch_geometric/blob/97d55577f1d0bf33c1bfbe0ef864923ad5cb844d/torch_geometric/nn/conv/gcn_conv.py
"""
@torch.jit._overload
def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):
    # type: (Tensor, OptTensor, Optional[int], bool, bool, Optional[int]) -> PairTensor  # noqa
    pass

@torch.jit._overload
def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):
    # type: (SparseTensor, OptTensor, Optional[int], bool, bool, Optional[int]) -> SparseTensor  # noqa
    pass

def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):

    fill_value = 2. if improved else 1.

    if isinstance(edge_index, SparseTensor):
        adj_t = edge_index
        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            # adj_t = fill_diag(adj_t, fill_value)
            adj_t = adj_t.set_diag(fill_value)
        # deg = sum(adj_t, dim=1)
        deg = adj_t.sum(dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        # adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
        # adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
        adj_t = adj_t.mul(deg_inv_sqrt.view(-1, 1))
        adj_t = adj_t.mul(deg_inv_sqrt.view(-1, 1))

        return adj_t

    else:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        if add_self_loops:
            edge_index, tmp_edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value, num_nodes)
            assert tmp_edge_weight is not None
            edge_weight = tmp_edge_weight

        row, col = edge_index[0], edge_index[1]
        # deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
        device = edge_weight.device
        deg = torch.zeros(num_nodes, dtype=edge_weight.dtype, device=device)
        col = col.to(device)
        edge_weight = edge_weight.to(device)
        deg.index_add_(0, col, edge_weight)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        return edge_index, deg_inv_sqrt[col] * edge_weight * deg_inv_sqrt[col]
        

class GCNConv(MessagePassing):
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
        \mathbf{x}^{\prime}_i = \mathbf{\Theta} \sum_{j \in \mathcal{N}(v) \cup
        \{ i \}} \frac{e_{j,i}}{\sqrt{\hat{d}_j \hat{d}_i}} \mathbf{x}_j
    with :math:`\hat{d}_i = 1 + \sum_{j \in \mathcal{N}(i)} e_{j,i}`, where
    :math:`e_{j,i}` denotes the edge weight from source node :obj:`j` to target
    node :obj:`i` (default: :obj:`1.0`)
    Args:
        in_channels (int): Size of each input sample.
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
    """

    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, in_channels: int, out_channels: int,
                 improved: bool = False, cached: bool = True,
                 normalize: bool = True, bias: bool = True, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super(GCNConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.lin = torch.nn.Linear(in_channels, out_channels, bias=False)
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()


    def reset_parameters(self):
        #glorot(self.weight)
        glorot(self.lin.weight)
        zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None, is_add_self_loops: bool = True) -> Tensor:
        original_size = edge_index.shape[1]

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, is_add_self_loops)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, is_add_self_loops)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache
        
        x = self.lin(x)
        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)

        if self.bias is not None:
            out += self.bias

        return out, edge_index

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return torch.matmul(adj_t, x,)
        # return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class GCN(torch.nn.Module):
    def __init__(self, n_layer, nfeat, nhid, nclass, \
                    normalize=True, is_add_self_loops=True):
        super(GCN, self).__init__()

        self.nclass = nclass
        self.n_layer = n_layer
        self.is_add_self_loops = is_add_self_loops

        self.conv1 = [GCNConv(nfeat, nhid, cached=False, normalize=normalize)]
        self.conv1 += [GCNConv(nhid, nhid, cached=False, normalize=normalize) for _ in range(n_layer-2)]
        self.conv1 += [GCNConv(nhid, nclass, cached=False, normalize=normalize)]
        
        self.conv1 = torch.nn.ModuleList(self.conv1)
        self.reg_params = list(self.conv1.parameters())


    def forward(self, x, edge_index, edge_weight=None, embed=False):
        for i in range(self.n_layer-1):
            x, edge_index = self.conv1[i](x, edge_index, edge_weight, is_add_self_loops=self.is_add_self_loops)
            x = F.relu(x)
            x = F.dropout(x, training=self.training, p=0.5)
        
        if embed: # when n_layer == 1
            x, edge_index = self.conv1[0](x, edge_index, edge_weight, is_add_self_loops=self.is_add_self_loops)
            return F.relu(x)
        else:
            x, _ = self.conv1[-1](x, edge_index, edge_weight, is_add_self_loops=self.is_add_self_loops)
            return x

class SAGEConv(MessagePassing):
    r"""The GraphSAGE operator from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper
    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i + \mathbf{W}_2 \cdot
        \mathrm{mean}_{j \in \mathcal{N(i)}} \mathbf{x}_j
    Args:
        in_channels (int or tuple): Size of each input sample. A tuple
            corresponds to the sizes of source and target dimensionalities.
        out_channels (int): Size of each output sample.
        normalize (bool, optional): If set to :obj:`True`, output features
            will be :math:`\ell_2`-normalized, *i.e.*,
            :math:`\frac{\mathbf{x}^{\prime}_i}
            {\| \mathbf{x}^{\prime}_i \|_2}`.
            (default: :obj:`False`)
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add transformed root node features to the output.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, normalize: bool = False,
                 root_weight: bool = True,
                 bias: bool = True, **kwargs):  # yapf: disable
        kwargs.setdefault('aggr', 'mean')
        super(SAGEConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_l = Linear(in_channels[0], out_channels, bias=bias)
        if self.root_weight:
            self.temp_weight = Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        if self.root_weight:
            self.temp_weight.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, edge_weight,
                size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size)
        out = self.lin_l(out)

        x_r = x[1]
        if self.root_weight and x_r is not None:
            out += self.temp_weight(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return torch.matmul(adj_t, x[0])
        # return matmul(adj_t, x[0], reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class SAGE(nn.Module):
    def __init__(self, n_layer, nfeat, nhid, nclass):
        super(SAGE, self).__init__()
        self.n_layer = n_layer
        self.nclass = nclass
        self.conv1 = nn.ModuleList()
        for i in range(n_layer):
            in_channels = nfeat if i == 0 else nhid
            self.conv1.append(SAGEConv(in_channels, nhid))

        self.classifier = torch.nn.Linear(nhid, nclass)

        self.reg_params = list(self.conv1.parameters())
        self.non_reg_params = list(self.classifier.parameters())


    def forward(self, x, edge_index, edge_weight=None):
        ori_n_edge = edge_index.size(1)
        for i, conv in enumerate(self.conv1):
            x = conv(x, edge_index, edge_weight)
            x = x.relu()

        x = F.dropout(x, 0.5, training=self.training)
        x = self.classifier(x)

        return x


class GATConv(MessagePassing):
    r"""The graph attentional operator from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper
    .. math::
        \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{i} +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},
    where the attention coefficients :math:`\alpha_{i,j}` are computed as
    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j]
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k]
        \right)\right)}.
    Args:
        in_channels (int or tuple): Size of each input sample. A tuple
            corresponds to the sizes of source and target dimensionalities.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    _alpha: OptTensor

    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, heads: int = 1, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0.0,
                 bias: bool = True, edge_dim: Optional[int] = None, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(GATConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        if isinstance(in_channels, int):
            self.temp_weight = torch.nn.Linear(in_channels, heads * out_channels, bias=False)
            self.lin_l = self.temp_weight#Linear(in_channels, heads * out_channels, bias=False)
            self.lin_r = self.lin_l
        else:
            self.lin_l = Linear(in_channels[0], heads * out_channels, False)
            self.lin_r = Linear(in_channels[1], heads * out_channels, False)

        self.att_l = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_r = Parameter(torch.Tensor(1, heads, out_channels))

        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False)
            self.att_edge = Parameter(torch.Tensor(1, heads, out_channels))
        else:
            self.lin_edge = None
            self.register_parameter('att_edge', None)


        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None

        self.reset_parameters()


    def reset_parameters(self):
        glorot(self.lin_l.weight)
        glorot(self.lin_r.weight)
        if self.lin_edge is not None:
            glorot(self.lin_edge.weight)
            self.lin_edge.reset_parameters()
        glorot(self.att_l)
        glorot(self.att_r)
        zeros(self.bias)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, edge_attr: OptTensor = None,
                size: Size = None, return_attention_weights=None, is_add_self_loops: bool = True):

        r"""
        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        H, C = self.heads, self.out_channels
        original_size = edge_index.shape[1]
        x_l: OptTensor = None
        x_r: OptTensor = None
        alpha_l: OptTensor = None
        alpha_r: OptTensor = None

        if isinstance(x, Tensor):
            assert x.dim() == 2, 'Static graphs not supported in `GATConv`.'
            #x_lyy = x_r = self.lin_l(x).view(-1, H, C)
            x = self.lin_l(x) #.view(-1, H, C)
            x_l = x_r = x.view(-1,H,C)

            alpha_l = (x_l * self.att_l).sum(dim=-1)
            alpha_r = (x_r * self.att_r).sum(dim=-1)
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_l = self.lin_l(x_l).view(-1, H, C)
            alpha_l = (x_l * self.att_l).sum(dim=-1)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)
                alpha_r = (x_r * self.att_r).sum(dim=-1)

        assert x_l is not None
        assert alpha_l is not None

        if is_add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                if size is not None:
                    num_nodes = min(size[0], size[1])
                edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(edge_index, edge_attr, num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                # edge_index = set_diag(edge_index)
                edge_index = edge_index.set_diag()
        
        # edge_updater_type: (alpha: OptPairTensor, edge_attr: OptTensor)
        alpha = self.edge_updater(edge_index, alpha=(alpha_l, alpha_r), edge_attr=edge_attr)
        
        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)
        out = self.propagate(edge_index, x=(x_l, x_r),
                             alpha=(alpha_l, alpha_r), size=size)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out, edge_index


    def edge_update(self, alpha_j: Tensor, alpha_i: OptTensor,
                    edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                    size_i: Optional[int]) -> Tensor:
        # Given edge-level attention coefficients for source and target nodes,
        # we simply need to sum them up to "emulate" concatenation:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i

        if edge_attr is not None and self.lin_edge is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            alpha_edge = (edge_attr * self.att_edge).sum(dim=-1)
            alpha = alpha + alpha_edge

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha


    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


class GAT(torch.nn.Module):
    def __init__(self, n_layer, nfeat, nhid, nclass, n_head, \
                    is_add_self_loops=True, transpose=False, edge_dim=None):
        super(GAT, self).__init__()

        self.nclass = nclass
        self.n_layer = n_layer
        self.conv1 = [GATConv(nfeat, nhid//n_head, heads=n_head, edge_dim=edge_dim, dropout=0.6)]
        self.conv1 += [GATConv(nhid, nhid//n_head, heads=n_head, edge_dim=edge_dim, dropout=0.6)
            for i in range(1,n_layer)]
        self.conv1 = torch.nn.ModuleList(self.conv1)

        if not transpose:
            self.classifier = torch.nn.Linear(nhid, nclass)
            self.non_reg_params = list(self.classifier.parameters())

        self.reg_params = list(self.conv1.parameters())

        self.is_add_self_loops = is_add_self_loops


    def forward(self, x, edge_index, edge_weight=None, edge_attr=None):

        for i in range(self.n_layer):
            x, edge_index= self.conv1[i](x, edge_index, edge_attr=edge_attr, is_add_self_loops=self.is_add_self_loops)
            x = F.elu(x)
            x = F.dropout(x, p=0.5, training=self.training)
        
        if not self.transpose:
            x = self.classifier(x)
        else:
            x = torch.mm(x, self.conv1[0].lin.weight)

        return x


class SGConv(MessagePassing):
    r"""The simple graph convolutional operator from the `"Simplifying Graph
    Convolutional Networks" <https://arxiv.org/abs/1902.07153>`_ paper

    .. math::
        \mathbf{X}^{\prime} = {\left(\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \right)}^K \mathbf{X} \mathbf{\Theta},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.
    The adjacency matrix can include other values than :obj:`1` representing
    edge weights via the optional :obj:`edge_weight` tensor.

    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        out_channels (int): Size of each output sample.
        K (int, optional): Number of hops :math:`K`. (default: :obj:`1`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`{\left(\mathbf{\hat{D}}^{-1/2}
            \mathbf{\hat{A}} \mathbf{\hat{D}}^{-1/2} \right)}^K \mathbf{X}` on
            first execution, and will use the cached version for further
            executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})`,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge weights :math:`(|\mathcal{E}|)` *(optional)*
        - **output:**
          node features :math:`(|\mathcal{V}|, F_{out})`
    """

    _cached_x: Optional[Tensor]

    def __init__(self, in_channels: int, out_channels: int, K: int = 1,
                 cached: bool = False, add_self_loops: bool = True,
                 bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.cached = cached
        self.add_self_loops = add_self_loops

        self._cached_x = None

        self.lin = Linear(in_channels, out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self._cached_x = None


    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""
        cache = self._cached_x
        if cache is None:
            if isinstance(edge_index, Tensor):
                edge_index, edge_weight = gcn_norm(  # yapf: disable
                    edge_index, edge_weight, x.size(self.node_dim), False,
                    self.add_self_loops)
            elif isinstance(edge_index, SparseTensor):
                edge_index = gcn_norm(  # yapf: disable
                    edge_index, edge_weight, x.size(self.node_dim), False,
                    self.add_self_loops)

            for k in range(self.K):
                # propagate_type: (x: Tensor, edge_weight: OptTensor)
                x = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                                   size=None)
                if self.cached:
                    self._cached_x = x
        else:
            x = cache.detach()

        return self.lin(x)


    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return torch.matmul(adj_t, x)
        # return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, K={self.K})')

class SGC(torch.nn.Module):
    def __init__(self, num_features, num_classes, K=2, is_add_self_loops=True, cached=False):
        super(SGC, self).__init__()
        self.conv1 = SGConv(num_features, num_classes, K=K, add_self_loops=is_add_self_loops, cached=cached)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight)
        return x

class MLP(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes, num_layers, dropout, batch_norm=True, classifier=False):
        super(MLP, self).__init__()
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.classifier = classifier
        if classifier:
            self.lins = ModuleList([Linear(num_features, num_classes)])
        else:
            self.lins = ModuleList([Linear(num_features, hidden_dim)])
            if batch_norm:
                self.bns = ModuleList([BatchNorm1d(hidden_dim)])
            for _ in range(num_layers - 2):
                self.lins.append(Linear(hidden_dim, hidden_dim))
                if batch_norm:
                    self.bns.append(BatchNorm1d(hidden_dim))
            self.lins.append(Linear(hidden_dim, num_classes))

    def forward(self, x, edge_index=None, edge_weight=None, embed=False):
        if self.classifier:
            x = self.lins[-1](x)
            return x
        else:
            if self.batch_norm:
                for lin, bn in zip(self.lins[:-1], self.bns):
                    x = bn(lin(x).relu_())
                    x = F.dropout(x, p=self.dropout)
            else:
                for lin in self.lins[:-1]:
                    x = lin(x).relu_()
                    x = F.dropout(x, p=self.dropout)

            if embed: # befor dropout
                return x

            x = self.lins[-1](x)

            return x