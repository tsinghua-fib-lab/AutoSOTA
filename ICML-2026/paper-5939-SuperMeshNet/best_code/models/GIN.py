import torch.nn.functional as F
import torch.nn as nn
from utils.knn_interpolate import knn_interpolate
import torch
from models.Common import MLP


from typing import Callable, Optional, Union

import torch
from torch import Tensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import reset
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    Size,
    SparseTensor,
)
from torch_geometric.utils import spmm
class GINConv(MessagePassing):
    r"""The graph isomorphism operator from the `"How Powerful are
    Graph Neural Networks?" <https://arxiv.org/abs/1810.00826>`_ paper.

    .. math::
        \mathbf{x}^{\prime}_i = h_{\mathbf{\Theta}} \left( (1 + \epsilon) \cdot
        \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \right)

    or

    .. math::
        \mathbf{X}^{\prime} = h_{\mathbf{\Theta}} \left( \left( \mathbf{A} +
        (1 + \epsilon) \cdot \mathbf{I} \right) \cdot \mathbf{X} \right),

    here :math:`h_{\mathbf{\Theta}}` denotes a neural network, *.i.e.* an MLP.

    Args:
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps node features :obj:`x` of shape :obj:`[-1, in_channels]` to
            shape :obj:`[-1, out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`.
        eps (float, optional): (Initial) :math:`\epsilon`-value.
            (default: :obj:`0.`)
        train_eps (bool, optional): If set to :obj:`True`, :math:`\epsilon`
            will be a trainable parameter. (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V}_t|, F_{out})` if bipartite
    """
    def __init__(self, nn: Callable, ib_e:bool=True, eps: float = 0., train_eps: bool = True,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.nn = nn
        self.ib_e=ib_e
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.empty(1))
        else:
            self.register_buffer('eps', torch.empty(1))
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)


    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        size: Size = None,
    ) -> Tensor:

        if isinstance(x, Tensor):
            x = (x, x)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size)
        
        if self.ib_e:
            out=out-torch.mean(out,0)

        x_r = x[1]
        if x_r is not None:
            out = out + (1 + self.eps) * x_r

        return self.nn(out)


    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: Adj, x: OptPairTensor) -> Tensor:
        if isinstance(adj_t, SparseTensor):
            adj_t = adj_t.set_value(None, layout=None)
        return spmm(adj_t, x[0], reduce=self.aggr)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn})'



class GIN_shared(torch.nn.Module):
    def __init__(self,  depth, y_input_size, pos_input_size, hidden_size, ib_n, ib_e):
        super().__init__()
        
        self.encoder=MLP(y_input_size+pos_input_size, hidden_size, hidden_size)
        conv_list1 = []
        for _ in range(depth):
            conv_list1.append(GINConv(MLP(hidden_size, hidden_size, hidden_size), ib_e))
        self.conv_list1=nn.ModuleList(conv_list1)
        conv_list2 = []
        for _ in range(depth):
            conv_list2.append(GINConv(MLP(hidden_size, hidden_size, hidden_size), ib_e))
        self.conv_list2=nn.ModuleList(conv_list2)
        
        self.ib_n=ib_n
        

    def forward(self, l_pos1, l_y1, l_e1, h_pos1, h_e1):
        x=torch.concatenate([l_y1,l_pos1],-1)
        x=self.encoder(x)
     
        for conv in self.conv_list1:       
            x_=conv(x,l_e1)
            x_=F.relu(x_)
            x=x+x_
            if self.ib_n:
                x=x-torch.mean(x,0)
            
        x=knn_interpolate(x,l_pos1,h_pos1)
        for conv in self.conv_list2:
            x_=conv(x,h_e1)
            x_=F.relu(x_)
            x=x+x_
            if self.ib_n:
                x=x-torch.mean(x,0)
            
        return x


        


